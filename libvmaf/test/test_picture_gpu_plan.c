/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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
 * vmaf_picture_convert_gpu_plan(): a plain C run of each plan, following the
 * arithmetic picture_gpu_plan.h states, must give vmaf_picture_convert()'s
 * samples bit for bit, for the HDR models' conversion target at the sizes
 * whose zimg graphs differ (same size, letterbox and odd-size upscales, a
 * lanczos downscale, other primaries, a 12-bit source).
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "libvmaf/picture.h"
#include "libvmaf/picture_gpu_plan.h"

static float bits_float(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint32_t float_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

/* A float as a half float's bits, rounded to nearest even (cvtps_ph). */
static uint32_t half_from_float(float f)
{
    const uint32_t x = float_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t a = x & 0x7fffffffu;
    if (a > 0x7f800000u)
        return sign | 0x7e00u | ((a >> 13) & 0x3ffu);
    if (a >= 0x477ff000u)
        return sign | 0x7c00u;
    if (a >= 0x38800000u) {
        uint32_t r = a - 0x38000000u;
        r += 0xfffu + ((r >> 13) & 1u);
        return sign | (r >> 13);
    }
    if (a <= 0x33000000u)
        return sign;
    const uint32_t shift = 126u - (a >> 23);
    const uint32_t m = (a & 0x7fffffu) | 0x800000u;
    uint32_t q = m >> shift;
    const uint32_t rest = m & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1u);
    if (rest > half || (rest == half && (q & 1u)))
        ++q;
    return sign | q;
}

/* A plan's buffers: the 16-bit planes and the 32-bit scratch elements. */
typedef struct {
    const uint32_t *plan;
    uint16_t *words[64];
    uint32_t *scratch[64];
} Run;

static uint32_t load(const Run *r, uint32_t buf, size_t i)
{
    return r->scratch[buf] ? r->scratch[buf][i] : r->words[buf][i];
}

static void store(Run *r, uint32_t buf, size_t i, uint32_t v)
{
    if (r->scratch[buf])
        r->scratch[buf][i] = v;
    else
        r->words[buf][i] = (uint16_t) v;
}

/* A step's read of an element as a float, or as an integer for WORD. */
static float read_float(const uint32_t *mode, uint32_t raw)
{
    switch (mode[0]) {
    case VMAF_GPU_PLAN_READ_W2F:
        return fmaf(bits_float(mode[1]), (float) raw, bits_float(mode[2]));
    case VMAF_GPU_PLAN_READ_WORD:
        return (float) raw;
    default:
        return bits_float(raw);
    }
}

static uint32_t write_value(const uint32_t *mode, float x)
{
    if (mode[0] == VMAF_GPU_PLAN_WRITE_F2W) {
        const float v = fmaf(bits_float(mode[1]), x, bits_float(mode[2]));
        if (isnan(v) || fabsf(v) >= 2147483648.0f)
            return 0;
        float q = nearbyintf(v);
        q = q < 0.0f ? 0.0f : q > 65535.0f ? 65535.0f : q;
        return (uint32_t) q < mode[3] ? (uint32_t) q : mode[3];
    }
    return float_bits(x);
}

static void run_resize(Run *r, const uint32_t *s)
{
    const uint32_t kind = s[VMAF_GPU_PLAN_S_KIND];
    const int horizontal = kind == VMAF_GPU_PLAN_RESIZE_FLOAT_H ||
                           kind == VMAF_GPU_PLAN_RESIZE_INT16_H;
    const int is_float = kind == VMAF_GPU_PLAN_RESIZE_FLOAT_H ||
                         kind == VMAF_GPU_PLAN_RESIZE_FLOAT_V;
    const uint32_t in_w = s[VMAF_GPU_PLAN_S_IN_W];
    const uint32_t out_w = s[VMAF_GPU_PLAN_S_OUT_W], out_h = s[VMAF_GPU_PLAN_S_OUT_H];
    const uint32_t fw = s[VMAF_GPU_PLAN_S_FILTER_WIDTH];
    const uint32_t dim = horizontal ? out_w : out_h;
    const uint32_t *left = r->plan + s[VMAF_GPU_PLAN_S_TABLE];
    const uint32_t *coef = left + dim;
    const uint32_t *rd = s + VMAF_GPU_PLAN_S_READ, *wr = s + VMAF_GPU_PLAN_S_WRITE;
    const uint32_t in = s[VMAF_GPU_PLAN_S_IN], out = s[VMAF_GPU_PLAN_S_OUT];
    const int32_t limit = (int32_t) s[VMAF_GPU_PLAN_S_LIMIT] - 32768;

    for (uint32_t y = 0; y < out_h; y++) {
        for (uint32_t x = 0; x < out_w; x++) {
            const uint32_t i = horizontal ? x : y;
            const uint32_t *c = coef + (size_t) i * fw;
            size_t at = horizontal ? (size_t) y * in_w + left[i] : (size_t) left[i] * in_w + x;
            const size_t step = horizontal ? 1 : in_w;
            uint32_t value;
            if (is_float) {
                float acc[2] = { 0.0f, 0.0f };
                for (uint32_t k = 0; k < fw; k++, at += step) {
                    const float cf = bits_float(c[k]);
                    const float v = read_float(rd, load(r, in, at));
                    if (k < 2)
                        acc[k] = horizontal ? fmaf(cf, v, 0.0f) : cf * v;
                    else
                        acc[k % 2] = fmaf(cf, v, acc[k % 2]);
                }
                value = write_value(wr, fw == 1 ? acc[0] : acc[0] + acc[1]);
            } else {
                int32_t sum = 0;
                for (uint32_t k = 0; k < fw; k++, at += step)
                    sum += (int32_t) c[k] * ((int32_t) load(r, in, at) - 32768);
                int32_t v = (sum + 8192) >> 14;
                v = v < -32768 ? -32768 : v > 32767 ? 32767 : v;
                value = (uint32_t) ((v < limit ? v : limit) + 32768);
            }
            store(r, out, (size_t) y * out_w + x, value);
        }
    }
}

static float colour_op(const uint32_t *plan, uint32_t kind, uint32_t at,
                       const float in[3], int ch)
{
    const uint32_t *t = plan + at;
    switch (kind) {
    case VMAF_GPU_PLAN_OP_MATRIX: {
        const uint32_t *m = t + 3 * ch;
        const float ab = bits_float(m[0]) * in[0] + bits_float(m[1]) * in[1];
        return ab + bits_float(m[2]) * in[2];
    }
    case VMAF_GPU_PLAN_OP_TO_LINEAR: {
        const float v = fmaf(in[ch], 32768.0f, 16384.0f);
        uint32_t i = 0;
        if (!isnan(v) && fabsf(v) < 2147483648.0f) {
            const float q = nearbyintf(v);
            i = q < 0.0f ? 0 : q > 65536.0f ? 65536 : (uint32_t) q;
        }
        return bits_float(t[i]);
    }
    default:
        return bits_float(t[half_from_float(in[ch])]);
    }
}

static void run_colour(Run *r, const uint32_t *s)
{
    const size_t n = (size_t) s[VMAF_GPU_PLAN_S_OUT_W] * s[VMAF_GPU_PLAN_S_OUT_H];
    const uint32_t ops = s[VMAF_GPU_PLAN_S_FILTER_WIDTH];
    const uint32_t *list = r->plan + s[VMAF_GPU_PLAN_S_TABLE];
    for (size_t i = 0; i < n; i++) {
        float v[3], next[3];
        for (int p = 0; p < 3; p++)
            v[p] = read_float(s + VMAF_GPU_PLAN_S_READ + 4 * p,
                              load(r, s[VMAF_GPU_PLAN_S_IN + p], i));
        for (uint32_t k = 0; k < ops; k++) {
            for (int ch = 0; ch < 3; ch++)
                next[ch] = colour_op(r->plan, list[2 * k], list[2 * k + 1], v, ch);
            memcpy(v, next, sizeof(v));
        }
        for (int p = 0; p < 3; p++)
            store(r, s[VMAF_GPU_PLAN_S_OUT + p], i,
                  write_value(s + VMAF_GPU_PLAN_S_WRITE + 4 * p, v[p]));
    }
}

static void run_convert(Run *r, const uint32_t *s)
{
    const size_t n = (size_t) s[VMAF_GPU_PLAN_S_OUT_W] * s[VMAF_GPU_PLAN_S_OUT_H];
    for (size_t i = 0; i < n; i++) {
        const float v = read_float(s + VMAF_GPU_PLAN_S_READ, load(r, s[VMAF_GPU_PLAN_S_IN], i));
        store(r, s[VMAF_GPU_PLAN_S_OUT], i, write_value(s + VMAF_GPU_PLAN_S_WRITE, v));
    }
}

/* Runs `plan` on `src`; 0 when every target sample equals `dst`'s. */
static int run_plan_matches(const uint32_t *plan, size_t words,
                            const VmafPicture *src, const VmafPicture *dst)
{
    if (words < VMAF_GPU_PLAN_HEADER_WORDS || plan[VMAF_GPU_PLAN_H_MAGIC] != VMAF_GPU_PLAN_MAGIC ||
        plan[VMAF_GPU_PLAN_H_VERSION] != VMAF_GPU_PLAN_VERSION || plan[VMAF_GPU_PLAN_H_WORDS] != words)
        return 1;
    const uint32_t buffers = plan[VMAF_GPU_PLAN_H_BUFFER_COUNT];
    if (buffers > 64)
        return 2;
    Run r;
    memset(&r, 0, sizeof(r));
    r.plan = plan;
    int mismatch = 0;
    for (uint32_t b = 0; b < buffers; b++) {
        const uint32_t *rec = plan + plan[VMAF_GPU_PLAN_H_BUFFERS] + b * VMAF_GPU_PLAN_BUFFER_WORDS;
        const uint32_t kind = rec[VMAF_GPU_PLAN_B_KIND], p = rec[VMAF_GPU_PLAN_B_PLANE];
        const size_t n = rec[VMAF_GPU_PLAN_B_ELEMENTS];
        if (kind == VMAF_GPU_PLAN_SCRATCH) {
            r.scratch[b] = calloc(n, sizeof(uint32_t));
            continue;
        }
        r.words[b] = calloc(n, sizeof(uint16_t));
        if (kind == VMAF_GPU_PLAN_SOURCE) {
            if (n != (size_t) src->w[p] * src->h[p])
                mismatch = 3;
            for (unsigned y = 0; y < src->h[p] && !mismatch; y++)
                memcpy(r.words[b] + (size_t) y * src->w[p],
                       (const uint8_t *) src->data[p] + (size_t) y * src->stride[p],
                       src->w[p] * sizeof(uint16_t));
        }
    }
    const uint32_t *step = plan + plan[VMAF_GPU_PLAN_H_STEPS];
    for (uint32_t i = 0; i < plan[VMAF_GPU_PLAN_H_STEP_COUNT] && !mismatch; i++, step += VMAF_GPU_PLAN_STEP_WORDS) {
        switch (step[VMAF_GPU_PLAN_S_KIND]) {
        case VMAF_GPU_PLAN_COLOUR: run_colour(&r, step); break;
        case VMAF_GPU_PLAN_CONVERT: run_convert(&r, step); break;
        default: run_resize(&r, step); break;
        }
    }
    for (uint32_t b = 0; b < buffers; b++) {
        const uint32_t *rec = plan + plan[VMAF_GPU_PLAN_H_BUFFERS] + b * VMAF_GPU_PLAN_BUFFER_WORDS;
        const uint32_t p = rec[VMAF_GPU_PLAN_B_PLANE];
        if (rec[VMAF_GPU_PLAN_B_KIND] == VMAF_GPU_PLAN_TARGET && !mismatch) {
            if (rec[VMAF_GPU_PLAN_B_ELEMENTS] != (size_t) dst->w[p] * dst->h[p])
                mismatch = 4;
            for (unsigned y = 0; y < dst->h[p] && !mismatch; y++) {
                const uint16_t *want = (const uint16_t *) ((const uint8_t *) dst->data[p] + (size_t) y * dst->stride[p]);
                if (memcmp(want, r.words[b] + (size_t) y * dst->w[p], dst->w[p] * sizeof(uint16_t)))
                    mismatch = 5;
            }
        }
        free(r.words[b]);
        free(r.scratch[b]);
    }
    return mismatch;
}

/* Every luma code of the depth in turn (the out-of-range ones too), chroma
 * cycling through its extremes, mid and pseudo-random values. */
static void fill(VmafPicture *pic)
{
    const unsigned max = (1u << pic->bpc) - 1;
    const unsigned extremes[5] = { 0, max, (max + 1) / 2, 16u << (pic->bpc - 8), 240u << (pic->bpc - 8) };
    uint32_t lcg = 12345;
    unsigned code = 0;
    for (unsigned p = 0; p < 3; p++) {
        for (unsigned y = 0; y < pic->h[p]; y++) {
            uint16_t *row = (uint16_t *) ((uint8_t *) pic->data[p] + (size_t) y * pic->stride[p]);
            for (unsigned x = 0; x < pic->w[p]; x++) {
                lcg = lcg * 1664525u + 1013904223u;
                if (!p)
                    row[x] = (uint16_t) (code++ & max);
                else
                    row[x] = (uint16_t) ((lcg >> 8) % 3 ? (lcg >> 12) & max : extremes[(lcg >> 4) % 5]);
            }
        }
    }
}

static const VmafColor hdr_target_color = {
    .range = VMAF_COLOR_RANGE_LIMITED,
    .primaries = VMAF_COLOR_PRIMARIES_BT2020,
    .trc = VMAF_COLOR_TRC_SMPTE2084,
    .matrix = VMAF_COLOR_MATRIX_ICTCP,
};

/* The plan for src_w x src_h (bpc, primaries) to the HDR target at dst_w x
 * dst_h against the CPU conversion. */
static char *check_plan(unsigned bpc, enum VmafColorPrimaries primaries,
                        unsigned src_w, unsigned src_h, unsigned dst_w,
                        unsigned dst_h, enum VmafResampleFilter filter)
{
    VmafPicture src, dst;
    int err = vmaf_picture_alloc(&src, VMAF_PIX_FMT_YUV420P, bpc, src_w, src_h);
    mu_assert("problem during vmaf_picture_alloc", !err);
    src.color = (VmafColor) {
        .range = VMAF_COLOR_RANGE_LIMITED,
        .primaries = primaries,
        .trc = VMAF_COLOR_TRC_SMPTE2084,
        .matrix = VMAF_COLOR_MATRIX_BT2020_NCL,
    };
    fill(&src);
    const VmafPictureConvertTarget target = {
        .pix_fmt = VMAF_PIX_FMT_YUV420P, .bpc = 16, .w = dst_w, .h = dst_h,
        .color = hdr_target_color, .resample_filter = filter,
    };
    VmafPictureConvertContext *ctx;
    err = vmaf_picture_convert_context_init(&ctx, &src, &target);
    mu_assert("problem during vmaf_picture_convert_context_init", !err);
    err = vmaf_picture_convert(ctx, &dst, &src);
    mu_assert("problem during vmaf_picture_convert", !err);

    uint32_t *plan;
    size_t words;
    err = vmaf_picture_convert_gpu_plan(&src, &target, &plan, &words);
    mu_assert("vmaf_picture_convert_gpu_plan failed", !err && plan);
    const int mismatch = run_plan_matches(plan, words, &src, &dst);
    if (mismatch)
        fprintf(stderr, "(mismatch %d) ", mismatch);
    mu_assert("the plan's samples differ from vmaf_picture_convert's", !mismatch);

    vmaf_picture_convert_gpu_plan_free(plan);
    vmaf_picture_convert_context_close(ctx);
    vmaf_picture_unref(&dst);
    vmaf_picture_unref(&src);
    return NULL;
}

static char *test_plan_same_size()
{
    return check_plan(10, VMAF_COLOR_PRIMARIES_BT2020, 64, 36, 64, 36, VMAF_RESAMPLE_DEFAULT);
}

static char *test_plan_letterbox_upscale()
{
    return check_plan(10, VMAF_COLOR_PRIMARIES_BT2020, 64, 34, 68, 36, VMAF_RESAMPLE_BICUBIC);
}

static char *test_plan_odd_size_upscale()
{
    return check_plan(10, VMAF_COLOR_PRIMARIES_BT2020, 30, 18, 62, 36, VMAF_RESAMPLE_BICUBIC);
}

static char *test_plan_lanczos_downscale()
{
    return check_plan(10, VMAF_COLOR_PRIMARIES_BT2020, 128, 72, 64, 36, VMAF_RESAMPLE_LANCZOS);
}

static char *test_plan_p3_12bit_wide()
{
    return check_plan(12, VMAF_COLOR_PRIMARIES_SMPTE432, 130, 34, 138, 36, VMAF_RESAMPLE_BICUBIC);
}

static char *test_plan_refuses_8bit()
{
    VmafPicture src;
    int err = vmaf_picture_alloc(&src, VMAF_PIX_FMT_YUV420P, 8, 64, 36);
    mu_assert("problem during vmaf_picture_alloc", !err);
    src.color = hdr_target_color;
    src.color.matrix = VMAF_COLOR_MATRIX_BT2020_NCL;
    const VmafPictureConvertTarget target = {
        .pix_fmt = VMAF_PIX_FMT_YUV420P, .bpc = 16, .color = hdr_target_color,
    };
    uint32_t *plan = (uint32_t *) &err;
    size_t words = 1;
    err = vmaf_picture_convert_gpu_plan(&src, &target, &plan, &words);
    mu_assert("an 8-bit source should have no plan", err == -ENOTSUP && !plan && !words);
    vmaf_picture_unref(&src);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_plan_same_size);
    mu_run_test(test_plan_letterbox_upscale);
    mu_run_test(test_plan_odd_size_upscale);
    mu_run_test(test_plan_lanczos_downscale);
    mu_run_test(test_plan_p3_12bit_wide);
    mu_run_test(test_plan_refuses_8bit);
    return NULL;
}
