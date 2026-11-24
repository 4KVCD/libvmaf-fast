// Derived from libvmaf (https://github.com/Netflix/vmaf), Copyright 2016-2026
// Netflix, Inc., licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root of
// this repository.
//
// The picture conversion of the HDR VMAF models (vmaf_picture_convert: zimg's
// resizes, depth changes and colour operations) run from the plan
// vmaf_picture_convert_gpu_plan() builds (libvmaf/picture_gpu_plan.h states
// each step's arithmetic). One dispatch per step:
//   STEP 1  RESIZE_FLOAT_H / _V (push constant `vertical`)
//   STEP 2  RESIZE_INT16_H / _V
//   STEP 3  COLOUR
//   STEP 4  CONVERT
// The plan's samples come out bit for bit as zimg's on the CPU only if every
// float operation rounds as there: the shader is compiled with NO_CONTRACTION
// (every multiply and add decorated NoContraction, so none is fused), and
// fma() stands exactly where zimg's AVX2 code fuses (GLSL's Fma, which every
// GPU maker runs as one rounding; the host's guard compares a whole
// picture with the CPU's and falls back to it if a driver does otherwise).
// Rounding to an integer and to a half float is done without the driver's
// round() (GLSL leaves its ties to the driver).
//
// Bindings: 0 the plan's words (vv_context's copy of it, followed by where
// each of its buffers is: `places`), 1 the source planes (16-bit samples, two
// a word), 2 the scratch buffers (32-bit elements), 3 and 4 the target's luma
// and chroma planes (16-bit, two a word). Each thread makes two neighbouring
// samples of a row, so that a 16-bit target word is written whole by one
// thread.

[[vk::binding(0)]] StructuredBuffer<uint> plan;
[[vk::binding(1)]] StructuredBuffer<uint> source;
[[vk::binding(2)]] RWStructuredBuffer<uint> scratch;
[[vk::binding(3)]] RWStructuredBuffer<uint> targetLuma;
[[vk::binding(4)]] RWStructuredBuffer<uint> targetChroma;

[[vk::push_constant]] cbuffer Constants {
    uint step;      // the step record's offset in `plan`
    uint places;    // the offset of the buffers' places: three words each (where, base, stride)
    uint vertical;  // a resize's direction
    uint zero;      // 0.0f, from outside so that fma(c, x, +0.0) is not folded into c * x
};

// Where a plan buffer is (the places' first word).
#define PLACE_SOURCE 0
#define PLACE_SCRATCH 1
#define PLACE_LUMA 2
#define PLACE_CHROMA 3

// The step record (picture_gpu_plan.h's VMAF_GPU_PLAN_S_*).
#define S_PLANES 1
#define S_IN 2
#define S_OUT 5
#define S_IN_W 8
#define S_IN_H 9
#define S_OUT_W 10
#define S_OUT_H 11
#define S_READ 12
#define S_WRITE 24
#define S_TABLE 36
#define S_FILTER_WIDTH 37
#define S_LIMIT 38

#define READ_FLOAT 0
#define READ_WORD 1
#define READ_W2F 2
#define WRITE_F2W 2

#define OP_MATRIX 1
#define OP_TO_LINEAR 2

struct Place {
    uint where, base, stride;  // stride: rows of a target plane apart (samples); else packed
};

Place place_of(uint buffer)
{
    Place p;
    p.where = plan[places + 3 * buffer];
    p.base = plan[places + 3 * buffer + 1];
    p.stride = plan[places + 3 * buffer + 2];
    return p;
}

// Element `index` of a source or scratch buffer (rows packed).
uint load(Place p, uint index)
{
    if (p.where == PLACE_SOURCE) {
        const uint at = p.base + index;
        return (source[at >> 1] >> ((at & 1u) * 16u)) & 0xFFFFu;
    }
    return scratch[p.base + index];
}

// Samples x and x + 1 of row y (the second only if `both`) of a plan buffer
// `width` wide. A target's base and stride are even (the host checks), so
// that x, which is even, starts a word.
void store2(Place p, uint width, uint x, uint y, uint v0, uint v1, bool both)
{
    if (p.where == PLACE_SCRATCH) {
        const uint at = p.base + y * width + x;
        scratch[at] = v0;
        if (both)
            scratch[at + 1] = v1;
        return;
    }
    const uint word = (p.base + y * p.stride + x) >> 1;
    const uint packed = (v0 & 0xFFFFu) | (both ? (v1 << 16) : 0u);
    if (p.where == PLACE_LUMA)
        targetLuma[word] = packed;
    else
        targetChroma[word] = packed;
}

// A 32-bit value is 0 if it is NaN or its magnitude reaches 2^31 (where the
// CPU's cvtps_epi32 gives INT_MIN), tested on its bits: Vulkan does not
// promise isnan() without float controls.
bool out_of_int_range(float v) { return (asuint(v) & 0x7FFFFFFFu) >= 0x4F000000u; }

// v (not NaN) rounded to the nearest integer, ties to even, and clamped to
// 0..top (an integer below 2^23): clamping first gives the same. The
// fraction c - trunc(c) is exact. (Slang folds the usual (c + 2^23) - 2^23
// into c.)
uint round_clamped(float v, float top)
{
    const float c = min(max(v, 0.0), top);
    const uint t = uint(c);
    const float fraction = c - float(t);
    return t + ((fraction > 0.5 || (fraction == 0.5 && (t & 1u) != 0u)) ? 1u : 0u);
}

// The read of element `raw` as a float, or as an integer for READ_WORD
// (mode: the step's read words of the plane).
float read_value(uint mode, uint raw)
{
    if (plan[mode] == READ_W2F)
        return fma(asfloat(plan[mode + 1]), float(raw), asfloat(plan[mode + 2]));
    if (plan[mode] == READ_WORD)
        return float(raw);
    return asfloat(raw);
}

uint write_value(uint mode, float x)
{
    if (plan[mode] != WRITE_F2W)
        return asuint(x);
    const float v = fma(asfloat(plan[mode + 1]), x, asfloat(plan[mode + 2]));
    if (out_of_int_range(v))
        return 0;
    return min(round_clamped(v, 65535.0), plan[mode + 3]);
}

// A float's bits as a half float's, rounded to nearest even (cvtps_ph).
uint half_from_float(float f)
{
    const uint x = asuint(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint a = x & 0x7FFFFFFFu;
    if (a > 0x7F800000u)
        return sign | 0x7E00u | ((a >> 13) & 0x3FFu);  // NaN, quiet
    if (a >= 0x477FF000u)
        return sign | 0x7C00u;  // rounds past the largest half: infinity
    if (a >= 0x38800000u) {     // a normal half
        uint r = a - 0x38000000u;
        r += 0xFFFu + ((r >> 13) & 1u);
        return sign | (r >> 13);
    }
    if (a <= 0x33000000u)  // half of the smallest half denormal or less: zero
        return sign;
    const uint shift = 126u - (a >> 23);  // a half denormal
    const uint m = (a & 0x7FFFFFu) | 0x800000u;
    uint q = m >> shift;
    const uint rest = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
    if (rest > half || (rest == half && (q & 1u) != 0u))
        ++q;
    return sign | q;
}

#if STEP == 1 || STEP == 2
// Output sample (x, y) of a resize.
uint resize_sample(uint x, uint y)
{
    const uint inW = plan[step + S_IN_W];
    const uint fw = plan[step + S_FILTER_WIDTH];
    const uint i = vertical != 0 ? y : x;
    const uint table = plan[step + S_TABLE];
    const uint left = plan[table + i];
    const uint coef = table + (vertical != 0 ? plan[step + S_OUT_H] : plan[step + S_OUT_W]) + i * fw;
    const uint stepIn = vertical != 0 ? inW : 1u;
    uint at = vertical != 0 ? left * inW + x : y * inW + left;
    const Place in = place_of(plan[step + S_IN]);
#if STEP == 1
    // Tap k into accumulator k % 2 (zimg's AVX2 kernels); each accumulator's
    // first tap starts it: fma(c, x, +0.0) across, c * x down (they differ
    // only in a zero's sign).
    const uint mode = step + S_READ;
    float acc0 = 0.0, acc1 = 0.0;
    for (uint k = 0; k < fw; ++k, at += stepIn) {
        const float c = asfloat(plan[coef + k]);
        const float v = read_value(mode, load(in, at));
        const float first = vertical != 0 ? c * v : fma(c, v, asfloat(zero));
        if (k == 0)
            acc0 = first;
        else if (k == 1)
            acc1 = first;
        else if ((k & 1u) == 0u)
            acc0 = fma(c, v, acc0);
        else
            acc1 = fma(c, v, acc1);
    }
    return write_value(step + S_WRITE, fw == 1 ? acc0 : acc0 + acc1);
#else
    // Exact in 32-bit integers, any order.
    int sum = 0;
    for (uint k = 0; k < fw; ++k, at += stepIn)
        sum += int(plan[coef + k]) * (int(load(in, at)) - 32768);
    const int v = clamp((sum + 8192) >> 14, -32768, 32767);
    return uint(min(v, int(plan[step + S_LIMIT]) - 32768) + 32768);
#endif
}
#endif

#if STEP == 3
// One colour operation on the three channels.
void colour_op(uint kind, uint at, inout float v[3])
{
    float next[3];
    if (kind == OP_MATRIX) {
        // Every product and sum rounded, in this order (zimg's AVX code).
        for (uint r = 0; r < 3; ++r) {
            const float ab = asfloat(plan[at + 3 * r]) * v[0] + asfloat(plan[at + 3 * r + 1]) * v[1];
            next[r] = ab + asfloat(plan[at + 3 * r + 2]) * v[2];
        }
    } else if (kind == OP_TO_LINEAR) {
        for (uint ch = 0; ch < 3; ++ch) {
            const float s = fma(v[ch], 32768.0, 16384.0);
            const uint i = out_of_int_range(s) ? 0u : round_clamped(s, 65536.0);
            next[ch] = asfloat(plan[at + i]);
        }
    } else {
        for (uint ch = 0; ch < 3; ++ch)
            next[ch] = asfloat(plan[at + half_from_float(v[ch])]);
    }
    v = next;
}

uint3 colour_sample(uint index)
{
    float v[3];
    for (uint p = 0; p < 3; ++p)
        v[p] = read_value(step + S_READ + 4 * p, load(place_of(plan[step + S_IN + p]), index));
    const uint list = plan[step + S_TABLE];
    for (uint k = 0; k < plan[step + S_FILTER_WIDTH]; ++k)
        colour_op(plan[list + 2 * k], plan[list + 2 * k + 1], v);
    uint3 out;
    for (uint p = 0; p < 3; ++p)
        out[p] = write_value(step + S_WRITE + 4 * p, v[p]);
    return out;
}
#endif

[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint outW = plan[step + S_OUT_W], outH = plan[step + S_OUT_H];
    const uint x = id.x * 2, y = id.y;
    if (x >= outW || y >= outH)
        return;
    const bool both = x + 1 < outW;
#if STEP == 1 || STEP == 2
    const uint v0 = resize_sample(x, y), v1 = both ? resize_sample(x + 1, y) : 0u;
    store2(place_of(plan[step + S_OUT]), outW, x, y, v0, v1, both);
#elif STEP == 3
    // Each sample's three planes are read before any is written: the step
    // may write a buffer it reads.
    const uint index = y * outW + x;
    const uint3 a = colour_sample(index), b = both ? colour_sample(index + 1) : uint3(0, 0, 0);
    for (uint p = 0; p < 3; ++p)
        store2(place_of(plan[step + S_OUT + p]), outW, x, y, a[p], b[p], both);
#else
    const uint index = y * outW + x;
    const Place in = place_of(plan[step + S_IN]);
    const uint v0 = write_value(step + S_WRITE, read_value(step + S_READ, load(in, index)));
    const uint v1 = both ? write_value(step + S_WRITE, read_value(step + S_READ, load(in, index + 1))) : 0u;
    store2(place_of(plan[step + S_OUT]), outW, x, y, v0, v1, both);
#endif
}
