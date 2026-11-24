/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

/*
 * vmaf_picture_convert_gpu_plan(): the conversion vmaf_picture_convert()
 * runs, as a list of steps another engine can run bit for bit (the format:
 * libvmaf/picture_gpu_plan.h).
 *
 * The steps are not a re-derivation of zimg's maths: zimg's own graph
 * builder builds the graph for the same formats and parameters as the CPU
 * conversion (vmaf_picture_zimg_args), and an observer zimg calls before
 * creating each filter records it: every depth change, resize (with the
 * filter coefficients zimg computes for it) and colour conversion, in graph
 * order. Each colour operation zimg creates is then probed (a matrix by unit
 * vectors, a lookup table at its sample points) and checked against the
 * formula the plan states on awkward inputs; anything that differs, or a
 * step the plan has no kind for, gives -ENOTSUP, and the caller keeps the
 * CPU conversion.
 *
 * This reads zimg's internal headers, so it is built only where meson is
 * given zimg's source tree (-Dzimg_source_dir), the zimg the library links.
 */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/alloc.h"
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/static_map.h"
#include "common/zassert.h"
#include "colorspace/colorspace.h"
#include "colorspace/graph.h"
#include "colorspace/operation.h"
#include "depth/depth.h"
#include "depth/quantize.h"
#include "graph/filtergraph.h"
#include "graph/graphbuilder.h"
#include "resize/filter.h"
#include "resize/resize.h"
#include "zimg.h"

extern "C" {
#include "libvmaf/picture_gpu_plan.h"
#include "log.h"
#include "picture_zimg.h"
}

/*
 * From zimg 3.0.6's src/zimg/api/zimg.cpp (https://github.com/sekrit-twc/zimg,
 * WTFPL), unchanged: how zimg_filter_graph_build() turns the C API's formats
 * and parameters into its graph builder's state, which this file needs to
 * build the very same graph. Keep in step with the zimg linked; the unit
 * test (test_picture_gpu_plan.c), which runs plans against the CPU
 * conversion, catches a drift.
 */
#define ZIMG_RESIZE_UNRESIZE static_cast<zimg_resample_filter_e>(-1)

namespace {

constexpr unsigned API_VERSION_2_0 = ZIMG_MAKE_API_VERSION(2, 0);
constexpr unsigned API_VERSION_2_1 = ZIMG_MAKE_API_VERSION(2, 1);
constexpr unsigned API_VERSION_2_2 = ZIMG_MAKE_API_VERSION(2, 2);
constexpr unsigned API_VERSION_2_4 = ZIMG_MAKE_API_VERSION(2, 4);

#define API_VERSION_ASSERT(x) zassert_d((x) >= API_VERSION_2_0, "API version invalid")

template <class Map, class Key>
typename Map::mapped_type search_enum_map(const Map &map, const Key &key, const char *msg)
{
	auto it = map.find(key);
	if (it == map.end())
		zimg::error::throw_<zimg::error::EnumOutOfRange>(msg);
	return it->second;
}

template <class Map, class Key>
typename Map::mapped_type search_itu_enum_map(const Map &map, const Key &key, const char *msg)
{
	if (static_cast<int>(key) < 0 || static_cast<int>(key) > 255)
		zimg::error::throw_<zimg::error::EnumOutOfRange>(msg);

	auto it = map.find(key);
	if (it == map.end())
		zimg::error::throw_<zimg::error::NoColorspaceConversion>(msg);
	return it->second;
}

#include "picture_gpu_plan_zimg_api.inc"

} // namespace

namespace {

/* A float's bits, and back: the plan stores floats as words. */
uint32_t float_bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float bits_float(uint32_t u)
{
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/*
 * A float as a half float, rounded to nearest even, as x86's cvtps_ph (and
 * so zimg's to-gamma lookup) does: a NaN keeps its sign and top payload bits
 * and is made quiet. Integer operations only, so that a GPU can do the same
 * whatever it does with denormal floats.
 */
uint16_t half_from_float(float f)
{
    const uint32_t x = float_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t a = x & 0x7fffffffu;
    if (a > 0x7f800000u)  /* NaN */
        return (uint16_t)(sign | 0x7e00u | ((a >> 13) & 0x3ffu));
    if (a >= 0x477ff000u)  /* 65520 and up (and infinity) round to infinity */
        return (uint16_t)(sign | 0x7c00u);
    if (a >= 0x38800000u) {  /* a normal half: rebias the exponent, round */
        uint32_t r = a - 0x38000000u;
        r += 0xfffu + ((r >> 13) & 1u);
        return (uint16_t)(sign | (r >> 13));
    }
    if (a <= 0x33000000u)  /* 2^-25 and below round to zero */
        return (uint16_t)sign;
    /* a subnormal half: the value in units of 2^-24, rounded */
    const uint32_t shift = 126u - (a >> 23);
    const uint32_t m = (a & 0x7fffffu) | 0x800000u;
    uint32_t q = m >> shift;
    const uint32_t rest = m & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1u);
    if (rest > half || (rest == half && (q & 1u)))
        ++q;
    return (uint16_t)(sign | q);
}

float float_from_half(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    if (e == 0x1fu)
        return bits_float(sign | 0x7f800000u | (m << 13));
    if (e)
        return bits_float(sign | ((e + 112u) << 23) | (m << 13));
    if (!m)
        return bits_float(sign);
    e = 113u;  /* a subnormal: normalize it */
    while (!(m & 0x400u)) {
        m <<= 1;
        --e;
    }
    return bits_float(sign | (e << 23) | ((m & 0x3ffu) << 13));
}

/* The to-linear lookup's index for x (picture_gpu_plan.h). */
uint32_t to_linear_index(float x)
{
    const float v = std::fma(x, 32768.0f, 16384.0f);
    if (std::isnan(v) || std::fabs(v) >= 2147483648.0f)
        return 0;
    const float r = std::nearbyint(v);  /* the default rounding: to even */
    return r < 0.0f ? 0u : r > 65536.0f ? 65536u : (uint32_t)r;
}

/*
 * What zimg's graph builder does, in graph order, as the observer saw it.
 * Each event keeps what the plan needs; the resize's filter is zimg's own
 * object, which outlives the observer (it belongs to the parameters).
 */
struct Event {
    enum Kind { DEPTH, RESIZE, COLOUR } kind;
    int plane;
    zimg::PixelFormat pixel_in, pixel_out;               /* DEPTH */
    zimg::depth::DitherType dither;
    unsigned src_w, src_h, dst_w, dst_h;                 /* RESIZE */
    zimg::PixelType type;
    unsigned depth;
    const zimg::resize::Filter *filter;
    double shift_w, shift_h, subwidth, subheight;
    zimg::colorspace::ColorspaceDefinition csp_in, csp_out;  /* COLOUR */
    zimg::colorspace::OperationParams op_params;
    zimg::CPUClass cpu;
    unsigned width, height;
};

class Recorder final : public zimg::graph::FilterObserver {
public:
    std::vector<Event> events;
    bool unsupported = false;  /* an event the plan has no step for */

    void yuv_to_grey() override { unsupported = true; }
    void grey_to_yuv() override { unsupported = true; }
    void grey_to_rgb() override { unsupported = true; }
    void premultiply() override { unsupported = true; }
    void unpremultiply() override { unsupported = true; }
    void add_opaque() override { unsupported = true; }
    void discard_alpha() override { unsupported = true; }
    void unresize(const zimg::unresize::UnresizeConversion &, int) override { unsupported = true; }

    void colorspace(const zimg::colorspace::ColorspaceConversion &conv) override
    {
        Event e{};
        e.kind = Event::COLOUR;
        e.plane = -1;
        e.csp_in = conv.csp_in;
        e.csp_out = conv.csp_out;
        /* as ColorspaceConversion::create() makes them */
        e.op_params.set_peak_luminance(conv.peak_luminance)
                   .set_approximate_gamma(conv.approximate_gamma)
                   .set_scene_referred(conv.scene_referred);
        e.cpu = conv.cpu;
        e.width = conv.width;
        e.height = conv.height;
        events.push_back(e);
    }

    void depth(const zimg::depth::DepthConversion &conv, int plane) override
    {
        Event e{};
        e.kind = Event::DEPTH;
        e.plane = plane;
        e.pixel_in = conv.pixel_in;
        e.pixel_out = conv.pixel_out;
        e.dither = conv.dither_type;
        e.width = conv.width;
        e.height = conv.height;
        events.push_back(e);
    }

    void resize(const zimg::resize::ResizeConversion &conv, int plane) override
    {
        Event e{};
        e.kind = Event::RESIZE;
        e.plane = plane;
        e.src_w = conv.src_width;
        e.src_h = conv.src_height;
        e.dst_w = conv.dst_width;
        e.dst_h = conv.dst_height;
        e.type = conv.type;
        e.depth = conv.depth;
        e.filter = conv.filter;
        e.shift_w = conv.shift_w;
        e.shift_h = conv.shift_h;
        e.subwidth = conv.subwidth;
        e.subheight = conv.subheight;
        events.push_back(e);
    }
};

/* Thrown where zimg's graph has something the plan does not express. */
struct Unsupported {
    std::string why;
};

[[noreturn]] void unsupported(const std::string &why) { throw Unsupported{ why }; }

struct Read {
    uint32_t mode = VMAF_GPU_PLAN_READ_FLOAT;
    float scale = 0.0f, offset = 0.0f;
};

struct Write {
    uint32_t mode = VMAF_GPU_PLAN_WRITE_FLOAT;
    float scale = 0.0f, offset = 0.0f;
    uint32_t max = 0;
};

/* A step on values: each value is written once (by a step, or is a source
 * plane) and is given a buffer only once the steps are final. */
struct Step {
    uint32_t kind = 0;
    unsigned planes = 1;
    int in[3] = { -1, -1, -1 }, out[3] = { -1, -1, -1 };
    unsigned in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    Read read[3];
    Write write[3];
    std::vector<uint32_t> table;  /* offsets in it are relative to its start */
    uint32_t filter_width = 0;
    uint32_t limit = 0;
};

struct Value {
    unsigned w, h;
    bool is_float;
};

/* One colour operation, probed. */
struct ColourOp {
    uint32_t kind;
    std::vector<float> data;
};

/* Aligned float arrays for zimg's operations (their AVX loads). */
struct Channels {
    zimg::AlignedVector<float> c[3];
    explicit Channels(size_t n)
    {
        for (auto &v : c)
            v.resize(n);
    }
    void run(const zimg::colorspace::Operation &op, Channels &out, unsigned n) const
    {
        const float *src[3] = { c[0].data(), c[1].data(), c[2].data() };
        float *dst[3] = { out.c[0].data(), out.c[1].data(), out.c[2].data() };
        op.process(src, dst, 0, n);
    }
};

/* Inputs for checking an operation: random values in its working range and
 * the awkward ones (NaN, infinities, signed zeros, huge, just outside the
 * lookup's range, half float boundaries). */
std::vector<float> check_inputs()
{
    std::vector<float> v = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 1.5f, 2.0f,
        std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        2147483648.0f, -2147483648.0f, 1e20f, -1e20f, 65504.0f, 65520.0f, 65519.0f,
        std::nextafter(-0.5f, -1.0f), std::nextafter(1.5f, 2.0f), std::nextafter(1.5f, 1.0f),
        6.103515625e-05f, 5.9604645e-08f, 2.9802322e-08f, 2.9802326e-08f, 1e-30f, -1e-30f,
        std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
    };
    std::mt19937 rng(20261122u);
    std::uniform_real_distribution<float> wide(-0.6f, 1.6f);
    for (int i = 0; i < 4096; ++i)
        v.push_back(wide(rng));
    std::uniform_int_distribution<uint32_t> any;
    for (int i = 0; i < 512; ++i)
        v.push_back(bits_float(any(rng)));
    for (uint32_t h = 0; h < 65536; h += 61)  /* half codes, and their neighbours */
        for (int d = -1; d <= 1; ++d)
            v.push_back(std::nextafter(float_from_half((uint16_t)h), d < 0 ? -INFINITY : d > 0 ? INFINITY : float_from_half((uint16_t)h)));
    while (v.size() % 16)
        v.push_back(0.25f);
    return v;
}

bool same_bits(float a, float b) { return float_bits(a) == float_bits(b); }

/* The matrix the plan states, product by product, no fma. */
float matrix_row(const float *m, float a, float b, float c)
{
    const float ab = m[0] * a + m[1] * b;  /* (picture_gpu_plan.cpp is built without contraction) */
    return ab + m[2] * c;
}

/* An operation as a matrix: its unit-vector answers, checked on every input. */
bool probe_matrix(const zimg::colorspace::Operation &op, const std::vector<float> &inputs, ColourOp *result)
{
    const unsigned n = (unsigned)inputs.size();
    Channels in(n), out(n);
    for (unsigned i = 0; i < n; ++i)
        for (int ch = 0; ch < 3; ++ch)
            in.c[ch][i] = 0.0f;
    for (int j = 0; j < 3; ++j)
        in.c[j][j] = 1.0f;
    in.run(op, out, 16);
    float m[9];
    for (int r = 0; r < 3; ++r)
        for (int j = 0; j < 3; ++j)
            m[r * 3 + j] = out.c[r][j];
    /* each input in each channel position, the others rotated through */
    for (unsigned i = 0; i < n; ++i) {
        in.c[0][i] = inputs[i];
        in.c[1][i] = inputs[(i + 1) % n];
        in.c[2][i] = inputs[(i + 2) % n];
    }
    in.run(op, out, n);
    for (unsigned i = 0; i < n; ++i) {
        for (int r = 0; r < 3; ++r) {
            const float want = matrix_row(m + r * 3, in.c[0][i], in.c[1][i], in.c[2][i]);
            if (!same_bits(want, out.c[r][i]) && !(std::isnan(want) && std::isnan(out.c[r][i])))
                return false;
        }
    }
    result->kind = VMAF_GPU_PLAN_OP_MATRIX;
    result->data.assign(m, m + 9);
    return true;
}

/* An operation as a lookup on every channel alike: `entries` sample points
 * (`point` gives the input whose index is i), then every input checked. */
template <class Point, class Index>
bool probe_lookup(const zimg::colorspace::Operation &op, const std::vector<float> &inputs, uint32_t kind,
                  unsigned entries, Point point, Index index, ColourOp *result)
{
    const unsigned padded = (entries + 15) & ~15u;
    Channels in(padded), out(padded);
    for (unsigned i = 0; i < padded; ++i)
        for (int ch = 0; ch < 3; ++ch)
            in.c[ch][i] = point(i < entries ? i : 0);
    in.run(op, out, padded);
    std::vector<float> lut(entries);
    for (unsigned i = 0; i < entries; ++i) {
        lut[i] = out.c[0][i];
        if (!same_bits(out.c[1][i], lut[i]) || !same_bits(out.c[2][i], lut[i]))
            return false;
    }
    const unsigned n = (unsigned)inputs.size();
    Channels check(n), checked(n);
    for (unsigned i = 0; i < n; ++i) {
        check.c[0][i] = inputs[i];
        check.c[1][i] = inputs[(i + 1) % n];
        check.c[2][i] = inputs[(i + 2) % n];
    }
    check.run(op, checked, n);
    for (unsigned i = 0; i < n; ++i) {
        for (int ch = 0; ch < 3; ++ch) {
            const float want = lut[index(check.c[ch][i])];
            if (!same_bits(want, checked.c[ch][i]))
                return false;
        }
    }
    result->kind = kind;
    result->data = std::move(lut);
    return true;
}

ColourOp probe_operation(const zimg::colorspace::Operation &op, const std::vector<float> &inputs)
{
    ColourOp result{};
    if (probe_matrix(op, inputs, &result))
        return result;
    /* to linear: x_i = (i - 16384) / 32768 is exact and gives back index i */
    if (probe_lookup(op, inputs, VMAF_GPU_PLAN_OP_TO_LINEAR, 65537,
                     [](unsigned i) { return ((float)i - 16384.0f) / 32768.0f; }, to_linear_index, &result))
        return result;
    if (probe_lookup(op, inputs, VMAF_GPU_PLAN_OP_TO_GAMMA, 65536,
                     [](unsigned i) { return float_from_half((uint16_t)i); },
                     [](float x) { return (uint32_t)half_from_float(x); }, &result))
        return result;
    unsupported("a colour operation is not a matrix or lookup the plan states");
}

/* zimg's resize::resize_h_first() (resize.cpp, in an unnamed namespace there),
 * copied: which pass of a 2-D resize runs first. */
bool resize_h_first(double xscale, double yscale)
{
    const double h_first_cost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
    const double v_first_cost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;
    return h_first_cost < v_first_cost;
}

class Builder {
public:
    std::vector<Step> steps;
    std::vector<Value> values;  /* 0 to 2: the source planes */
    int cur[3];                 /* each plane's latest value */

    Builder(unsigned w, unsigned h, unsigned ss_w, unsigned ss_h)
    {
        values.push_back({ w, h, false });
        values.push_back({ w >> ss_w, h >> ss_h, false });
        values.push_back({ w >> ss_w, h >> ss_h, false });
        for (int p = 0; p < 3; ++p)
            cur[p] = p;
    }

    void add(const Event &e)
    {
        switch (e.kind) {
        case Event::DEPTH: for_planes(e.plane, [&](int p) { depth(e, p); }); break;
        case Event::RESIZE: for_planes(e.plane, [&](int p) { resize(e, p); }); break;
        case Event::COLOUR: colour(e); break;
        }
    }

    void fuse();

private:
    /* zimg reports the two chroma planes' step once, as plane 1 */
    template <class F>
    static void for_planes(int plane, F f)
    {
        if (plane == 0) {
            f(0);
        } else if (plane == 1) {
            f(1);
            f(2);
        } else {
            unsupported("a step on plane " + std::to_string(plane));
        }
    }

    int new_value(unsigned w, unsigned h, bool is_float)
    {
        values.push_back({ w, h, is_float });
        return (int)values.size() - 1;
    }

    void depth(const Event &e, int p)
    {
        const Value v = values[(size_t)cur[p]];
        if (e.pixel_in == e.pixel_out)
            return;  /* zimg copies */
        if (e.width != v.w || e.height != v.h)
            unsupported("a depth step whose size is not its plane's");
        Step s;
        s.kind = VMAF_GPU_PLAN_CONVERT;
        s.in[0] = cur[p];
        s.in_w = s.out_w = v.w;
        s.in_h = s.out_h = v.h;
        const bool from_word = e.pixel_in.type == zimg::PixelType::WORD && !v.is_float;
        const bool to_word = e.pixel_out.type == zimg::PixelType::WORD;
        if (from_word && e.pixel_out.type == zimg::PixelType::FLOAT) {
            /* zimg's depth convert to float (depth_convert.cpp) */
            const int32_t range = zimg::depth::integer_range(e.pixel_in);
            const int32_t offset = zimg::depth::integer_offset(e.pixel_in);
            s.read[0].mode = VMAF_GPU_PLAN_READ_W2F;
            s.read[0].scale = static_cast<float>(1.0 / range);
            s.read[0].offset = static_cast<float>(-offset * (1.0 / range));
            s.write[0].mode = VMAF_GPU_PLAN_WRITE_FLOAT;
            s.out[0] = new_value(v.w, v.h, true);
        } else if (e.pixel_in.type == zimg::PixelType::FLOAT && v.is_float && to_word
                   && e.dither == zimg::depth::DitherType::NONE) {
            /* zimg's dither with DitherType::NONE (a zero dither, dither.cpp) */
            std::tie(s.write[0].scale, s.write[0].offset) =
                zimg::depth::get_scale_offset(e.pixel_in, e.pixel_out);
            s.read[0].mode = VMAF_GPU_PLAN_READ_FLOAT;
            s.write[0].mode = VMAF_GPU_PLAN_WRITE_F2W;
            s.write[0].max = (uint32_t)zimg::depth::numeric_max((int)e.pixel_out.depth);
            s.out[0] = new_value(v.w, v.h, false);
        } else {
            unsupported("a depth step other than word to float or undithered float to word");
        }
        cur[p] = s.out[0];
        steps.push_back(std::move(s));
    }

    void resize_pass(const Event &e, int p, bool horizontal)
    {
        const Value v = values[(size_t)cur[p]];
        const unsigned src_dim = horizontal ? e.src_w : e.src_h;
        const unsigned dst_dim = horizontal ? e.dst_w : e.dst_h;
        const zimg::resize::FilterContext f = zimg::resize::compute_filter(
            *e.filter, src_dim, dst_dim, horizontal ? e.shift_w : e.shift_h,
            horizontal ? e.subwidth : e.subheight);
        if ((horizontal ? v.w : v.h) != src_dim || f.input_width != src_dim || f.filter_rows != dst_dim)
            unsupported("a resize whose filter does not match its plane");
        const bool is_float = e.type == zimg::PixelType::FLOAT;
        if (!is_float && e.type != zimg::PixelType::WORD)
            unsupported("a resize on bytes or half floats");
        if (is_float != v.is_float)
            unsupported("a resize reading another format than its plane's");
        /* zimg's AVX2 float kernels change their order past 8 taps */
        if (is_float && f.filter_width > 8)
            unsupported("a float filter of over 8 taps");
        Step s;
        s.kind = is_float ? (horizontal ? VMAF_GPU_PLAN_RESIZE_FLOAT_H : VMAF_GPU_PLAN_RESIZE_FLOAT_V)
                          : (horizontal ? VMAF_GPU_PLAN_RESIZE_INT16_H : VMAF_GPU_PLAN_RESIZE_INT16_V);
        s.in[0] = cur[p];
        s.in_w = v.w;
        s.in_h = v.h;
        s.out_w = horizontal ? dst_dim : v.w;
        s.out_h = horizontal ? v.h : dst_dim;
        s.read[0].mode = is_float ? VMAF_GPU_PLAN_READ_FLOAT : VMAF_GPU_PLAN_READ_WORD;
        s.write[0].mode = is_float ? VMAF_GPU_PLAN_WRITE_FLOAT : VMAF_GPU_PLAN_WRITE_WORD;
        s.filter_width = f.filter_width;
        s.limit = is_float ? 0 : (uint32_t)((1ul << e.depth) - 1);
        s.table.reserve((size_t)dst_dim * (1 + f.filter_width));
        for (unsigned i = 0; i < dst_dim; ++i)
            s.table.push_back(f.left[i]);
        for (unsigned i = 0; i < dst_dim; ++i) {
            for (unsigned k = 0; k < f.filter_width; ++k) {
                s.table.push_back(is_float ? float_bits(f.data[(size_t)i * f.stride + k])
                                           : (uint32_t)(int32_t)f.data_i16[(size_t)i * f.stride_i16 + k]);
            }
        }
        s.out[0] = new_value(s.out_w, s.out_h, is_float);
        cur[p] = s.out[0];
        steps.push_back(std::move(s));
    }

    /* as zimg's ResizeConversion::create() (resize.cpp) splits a resize */
    void resize(const Event &e, int p)
    {
        const bool skip_h = e.src_w == e.dst_w && e.shift_w == 0 && e.subwidth == e.src_w;
        const bool skip_v = e.src_h == e.dst_h && e.shift_h == 0 && e.subheight == e.src_h;
        if (skip_h && skip_v)
            return;  /* zimg copies */
        if (skip_h) {
            resize_pass(e, p, false);
        } else if (skip_v) {
            resize_pass(e, p, true);
        } else if (resize_h_first(e.dst_w / e.subwidth, e.dst_h / e.subheight)) {
            resize_pass(e, p, true);
            resize_pass(e, p, false);
        } else {
            resize_pass(e, p, false);
            resize_pass(e, p, true);
        }
    }

    void colour(const Event &e)
    {
        if (e.csp_in == e.csp_out)
            return;  /* zimg copies */
        Step s;
        s.kind = VMAF_GPU_PLAN_COLOUR;
        s.planes = 3;
        for (int p = 0; p < 3; ++p) {
            const Value v = values[(size_t)cur[p]];
            if (v.w != e.width || v.h != e.height || !v.is_float)
                unsupported("a colour step on planes not at its size in floats");
            s.in[p] = cur[p];
        }
        s.in_w = s.out_w = e.width;
        s.in_h = s.out_h = e.height;
        const auto path = zimg::colorspace::get_operation_path(e.csp_in, e.csp_out);
        const std::vector<float> inputs = check_inputs();
        std::vector<ColourOp> ops;
        for (const auto &factory : path) {
            const std::unique_ptr<zimg::colorspace::Operation> op = factory(e.op_params, e.cpu);
            ops.push_back(probe_operation(*op, inputs));
        }
        /* the table: (kind, offset) per operation, then their data */
        s.filter_width = (uint32_t)ops.size();
        uint32_t offset = (uint32_t)(2 * ops.size());
        for (const ColourOp &op : ops) {
            s.table.push_back(op.kind);
            s.table.push_back(offset);
            offset += (uint32_t)op.data.size();
        }
        for (const ColourOp &op : ops)
            for (float x : op.data)
                s.table.push_back(float_bits(x));
        for (int p = 0; p < 3; ++p) {
            s.out[p] = new_value(e.width, e.height, true);
            cur[p] = s.out[p];
        }
        steps.push_back(std::move(s));
    }
};

/* Which step reads value v, and how often. */
void readers(const std::vector<Step> &steps, int v, int *count, size_t *first)
{
    *count = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
        for (unsigned p = 0; p < steps[i].planes; ++p) {
            if (steps[i].in[p] == v) {
                if (!(*count)++)
                    *first = i;
            }
        }
    }
}

/*
 * A depth step folded into its neighbour: word to float into the one step
 * reading its float (the same fma on the same word), float to word into the
 * step writing its float (the same rounding of the same float). Exact: each
 * sample sees the same operations.
 */
void Builder::fuse()
{
    for (size_t i = 0; i < steps.size();) {
        Step &c = steps[i];
        bool folded = false;
        if (c.kind == VMAF_GPU_PLAN_CONVERT && c.read[0].mode == VMAF_GPU_PLAN_READ_W2F) {
            int count;
            size_t r = 0;
            readers(steps, c.out[0], &count, &r);
            if (count == 1 && (steps[r].kind == VMAF_GPU_PLAN_COLOUR || steps[r].kind == VMAF_GPU_PLAN_RESIZE_FLOAT_H
                               || steps[r].kind == VMAF_GPU_PLAN_RESIZE_FLOAT_V)) {
                for (unsigned p = 0; p < steps[r].planes; ++p) {
                    if (steps[r].in[p] == c.out[0]) {
                        steps[r].in[p] = c.in[0];
                        steps[r].read[p] = c.read[0];
                    }
                }
                folded = true;
            }
        } else if (c.kind == VMAF_GPU_PLAN_CONVERT && c.write[0].mode == VMAF_GPU_PLAN_WRITE_F2W) {
            int count;
            size_t r = 0;
            readers(steps, c.in[0], &count, &r);
            for (size_t w = 0; count == 1 && w < i && !folded; ++w) {
                Step &producer = steps[w];
                for (unsigned p = 0; p < producer.planes; ++p) {
                    if (producer.out[p] == c.in[0] && producer.write[p].mode == VMAF_GPU_PLAN_WRITE_FLOAT
                        && producer.kind != VMAF_GPU_PLAN_CONVERT) {
                        producer.out[p] = c.out[0];
                        producer.write[p] = c.write[0];
                        folded = true;
                    }
                }
            }
        }
        if (folded)
            steps.erase(steps.begin() + (std::ptrdiff_t)i);
        else
            ++i;
    }
}

/* The plan's words: buffers given to the values, then the records. */
std::vector<uint32_t> serialize(const Builder &b, const zimg_image_format &src, const zimg_image_format &dst)
{
    const std::vector<Step> &steps = b.steps;
    /* the last read of each value, to reuse its scratch buffer after it */
    std::vector<size_t> last_read(b.values.size(), 0);
    for (size_t i = 0; i < steps.size(); ++i)
        for (unsigned p = 0; p < steps[i].planes; ++p)
            last_read[(size_t)steps[i].in[p]] = i;

    struct Buffer {
        uint32_t kind, plane, elements;
    };
    std::vector<Buffer> buffers;
    std::vector<int> buffer_of(b.values.size(), -1);
    for (uint32_t p = 0; p < 3; ++p) {
        buffers.push_back({ VMAF_GPU_PLAN_SOURCE, p, b.values[p].w * b.values[p].h });
        buffer_of[p] = (int)p;
    }
    const unsigned dst_cw = dst.width >> dst.subsample_w, dst_ch = dst.height >> dst.subsample_h;
    for (uint32_t p = 0; p < 3; ++p) {
        const int v = b.cur[p];
        const Value &value = b.values[(size_t)v];
        const unsigned want_w = p ? dst_cw : dst.width, want_h = p ? dst_ch : dst.height;
        if (v < 3 || value.is_float || value.w != want_w || value.h != want_h)
            unsupported("a target plane the steps do not write in its format");
        buffers.push_back({ VMAF_GPU_PLAN_TARGET, p, want_w * want_h });
        buffer_of[(size_t)v] = (int)buffers.size() - 1;
    }
    std::vector<int> free_scratch;
    /* the scratch buffers of step i's inputs that no later step reads */
    auto free_dead_inputs = [&](size_t i) {
        const Step &s = steps[i];
        for (unsigned p = 0; p < s.planes; ++p) {
            const int buf = buffer_of[(size_t)s.in[p]];
            if (last_read[(size_t)s.in[p]] == i && buffers[(size_t)buf].kind == VMAF_GPU_PLAN_SCRATCH
                && std::find(free_scratch.begin(), free_scratch.end(), buf) == free_scratch.end())
                free_scratch.push_back(buf);
        }
    };
    /* a free scratch buffer for `elements`: the smallest one big enough,
     * else the biggest one (grown), else a new one */
    auto take_scratch = [&](uint32_t elements) {
        auto best = free_scratch.end();
        for (auto it = free_scratch.begin(); it != free_scratch.end(); ++it) {
            const uint32_t have = buffers[(size_t)*it].elements;
            if (best == free_scratch.end())
                best = it;
            const uint32_t best_have = buffers[(size_t)*best].elements;
            const bool fits = have >= elements, best_fits = best_have >= elements;
            if ((fits && (!best_fits || have < best_have)) || (!fits && !best_fits && have > best_have))
                best = it;
        }
        int chosen;
        if (best != free_scratch.end()) {
            chosen = *best;
            free_scratch.erase(best);
        } else {
            buffers.push_back({ VMAF_GPU_PLAN_SCRATCH, 0, 0 });
            chosen = (int)buffers.size() - 1;
        }
        buffers[(size_t)chosen].elements = std::max(buffers[(size_t)chosen].elements, elements);
        return chosen;
    };
    for (size_t i = 0; i < steps.size(); ++i) {
        const Step &s = steps[i];
        /* a sample-by-sample step reads each sample before writing it, so its
         * outputs may take its dead inputs' buffers; a resize may not */
        const bool per_sample = s.kind == VMAF_GPU_PLAN_COLOUR || s.kind == VMAF_GPU_PLAN_CONVERT;
        if (per_sample)
            free_dead_inputs(i);
        for (unsigned p = 0; p < s.planes; ++p) {
            const int v = s.out[p];
            if (buffer_of[(size_t)v] < 0)  /* not a target */
                buffer_of[(size_t)v] = take_scratch(b.values[(size_t)v].w * b.values[(size_t)v].h);
        }
        if (!per_sample)
            free_dead_inputs(i);
    }

    std::vector<uint32_t> out(VMAF_GPU_PLAN_HEADER_WORDS, 0);
    out[VMAF_GPU_PLAN_H_MAGIC] = VMAF_GPU_PLAN_MAGIC;
    out[VMAF_GPU_PLAN_H_VERSION] = VMAF_GPU_PLAN_VERSION;
    out[VMAF_GPU_PLAN_H_BUFFER_COUNT] = (uint32_t)buffers.size();
    out[VMAF_GPU_PLAN_H_BUFFERS] = (uint32_t)out.size();
    out[VMAF_GPU_PLAN_H_STEP_COUNT] = (uint32_t)steps.size();
    out[VMAF_GPU_PLAN_H_SRC_W] = src.width;
    out[VMAF_GPU_PLAN_H_SRC_H] = src.height;
    out[VMAF_GPU_PLAN_H_SRC_BPC] = src.depth;
    out[VMAF_GPU_PLAN_H_DST_W] = dst.width;
    out[VMAF_GPU_PLAN_H_DST_H] = dst.height;
    out[VMAF_GPU_PLAN_H_DST_BPC] = dst.depth;
    for (const Buffer &buf : buffers) {
        const uint32_t record[VMAF_GPU_PLAN_BUFFER_WORDS] = { buf.kind, buf.plane, buf.elements, 0 };
        out.insert(out.end(), record, record + VMAF_GPU_PLAN_BUFFER_WORDS);
    }
    out[VMAF_GPU_PLAN_H_STEPS] = (uint32_t)out.size();
    size_t table = out.size() + steps.size() * VMAF_GPU_PLAN_STEP_WORDS;
    for (const Step &s : steps) {
        uint32_t r[VMAF_GPU_PLAN_STEP_WORDS] = {};
        r[VMAF_GPU_PLAN_S_KIND] = s.kind;
        r[VMAF_GPU_PLAN_S_PLANES] = s.planes;
        for (unsigned p = 0; p < s.planes; ++p) {
            r[VMAF_GPU_PLAN_S_IN + p] = (uint32_t)buffer_of[(size_t)s.in[p]];
            r[VMAF_GPU_PLAN_S_OUT + p] = (uint32_t)buffer_of[(size_t)s.out[p]];
            uint32_t *read = r + VMAF_GPU_PLAN_S_READ + 4 * p;
            read[0] = s.read[p].mode;
            read[1] = float_bits(s.read[p].scale);
            read[2] = float_bits(s.read[p].offset);
            uint32_t *write = r + VMAF_GPU_PLAN_S_WRITE + 4 * p;
            write[0] = s.write[p].mode;
            write[1] = float_bits(s.write[p].scale);
            write[2] = float_bits(s.write[p].offset);
            write[3] = s.write[p].max;
        }
        r[VMAF_GPU_PLAN_S_IN_W] = s.in_w;
        r[VMAF_GPU_PLAN_S_IN_H] = s.in_h;
        r[VMAF_GPU_PLAN_S_OUT_W] = s.out_w;
        r[VMAF_GPU_PLAN_S_OUT_H] = s.out_h;
        r[VMAF_GPU_PLAN_S_TABLE] = s.table.empty() ? 0 : (uint32_t)table;
        r[VMAF_GPU_PLAN_S_FILTER_WIDTH] = s.filter_width;
        r[VMAF_GPU_PLAN_S_LIMIT] = s.limit;
        out.insert(out.end(), r, r + VMAF_GPU_PLAN_STEP_WORDS);
        table += s.table.size();
    }
    for (const Step &s : steps) {
        const uint32_t base = (uint32_t)out.size();
        if (s.kind == VMAF_GPU_PLAN_COLOUR) {  /* the operations' offsets made absolute */
            for (size_t k = 0; k < s.table.size(); ++k)
                out.push_back(k < 2 * s.filter_width && (k & 1) ? s.table[k] + base : s.table[k]);
        } else {
            out.insert(out.end(), s.table.begin(), s.table.end());
        }
    }
    out[VMAF_GPU_PLAN_H_WORDS] = (uint32_t)out.size();
    return out;
}

std::vector<uint32_t> build_plan(const VmafPicture *src, const VmafPictureConvertTarget *target)
{
    if (src->pix_fmt != VMAF_PIX_FMT_YUV420P || (src->bpc != 10 && src->bpc != 12))
        unsupported("the source is not 10- or 12-bit 4:2:0");
    zimg_image_format src_fmt, dst_fmt;
    zimg_graph_builder_params params;
    if (vmaf_picture_zimg_args(src, target, "vmaf_picture_convert_gpu_plan", &src_fmt, &dst_fmt, &params))
        throw std::invalid_argument("formats");
    if (dst_fmt.pixel_type != ZIMG_PIXEL_WORD || dst_fmt.color_family != ZIMG_COLOR_YUV)
        unsupported("the target is not words in YUV");

    /* as zimg_filter_graph_build() (api/zimg.cpp) builds the graph */
    zimg::graph::GraphBuilder::state src_state, dst_state;
    std::tie(src_state, dst_state) = import_graph_state(src_fmt, dst_fmt);
    std::unique_ptr<zimg::resize::Filter> filters[2];
    const zimg::graph::GraphBuilder::params graph_params = import_graph_params(params, filters);
    Recorder recorder;
    zimg::graph::GraphBuilder builder;
    builder.set_source(src_state).connect(dst_state, &graph_params, &recorder);
    if (recorder.unsupported)
        unsupported("zimg's graph has an alpha, grey or unresize step");

    Builder plan(src_fmt.width, src_fmt.height, src_fmt.subsample_w, src_fmt.subsample_h);
    for (const Event &e : recorder.events)
        plan.add(e);
    plan.fuse();
    return serialize(plan, src_fmt, dst_fmt);
}

} // namespace

extern "C" int vmaf_picture_convert_gpu_plan(const VmafPicture *src, const VmafPictureConvertTarget *target,
                                             uint32_t **plan, size_t *words)
{
    if (!src || !target || !plan || !words)
        return -EINVAL;
    *plan = nullptr;
    *words = 0;
    try {
        const std::vector<uint32_t> built = build_plan(src, target);
        uint32_t *out = static_cast<uint32_t *>(std::malloc(built.size() * sizeof(uint32_t)));
        if (!out)
            return -ENOMEM;
        std::memcpy(out, built.data(), built.size() * sizeof(uint32_t));
        *plan = out;
        *words = built.size();
        return 0;
    } catch (const Unsupported &e) {
        vmaf_log(VMAF_LOG_LEVEL_INFO, "vmaf_picture_convert_gpu_plan: not expressed as a plan: %s\n",
                 e.why.c_str());
        return -ENOTSUP;
    } catch (const std::invalid_argument &) {
        return -EINVAL;  /* vmaf_picture_zimg_args() logged why */
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    } catch (const zimg::error::OutOfMemory &) {
        return -ENOMEM;
    } catch (const zimg::error::Exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "vmaf_picture_convert_gpu_plan: zimg: %s\n", e.what());
        return -EINVAL;
    }
}

extern "C" void vmaf_picture_convert_gpu_plan_free(uint32_t *plan)
{
    std::free(plan);
}
