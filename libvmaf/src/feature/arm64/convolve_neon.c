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
 * libvmaf-fast's NEON copy of x86/convolve_avx2.c: iqa/convolve.c's 1D
 * separable filter. Each output is the scalar code's
 *
 *     double sum = 0.0;
 *     for (each tap) sum += img[i] * k[j];
 *     dst = (float)sum;            (the kernel is normalized: scale 1)
 *
 * where img[i] * k[j] is a float product (one rounding), converted to
 * double and added. Here 16 outputs at a time, then 4: each 4 float
 * products in one vector, widened into two vectors of 2 doubles and added
 * there, the taps in the scalar order. A float product followed by a double
 * add cannot be fused into one rounding (the types differ), so this is the
 * scalar result whatever the compiler's -ffp-contract. The 1 to 3 columns
 * after the last whole vector are done by the scalar loop itself.
 */

#include <arm_neon.h>
#include <stddef.h>
#include <stdlib.h>

#include "convolve_neon.h"

/* sum of the taps of one output in double, the scalar code's loop */
static inline float tap_sum_scalar(const float *p, ptrdiff_t step, int taps, const float *k)
{
    double sum = 0.0;
    for (int t = 0; t < taps; t++)
        sum += (double)(p[t * step] * k[t]);
    return (float)sum;
}

/* 4 outputs: the taps of p[0..3], p[step..step+3], ... */
static inline float32x4_t tap_sum4(const float *p, ptrdiff_t step, int taps, const float *k)
{
    float64x2_t lo = vdupq_n_f64(0.0);
    float64x2_t hi = vdupq_n_f64(0.0);
    for (int t = 0; t < taps; t++) {
        const float32x4_t prod = vmulq_n_f32(vld1q_f32(p + t * step), k[t]);
        lo = vaddq_f64(lo, vcvt_f64_f32(vget_low_f32(prod)));
        hi = vaddq_f64(hi, vcvt_high_f64_f32(prod));
    }
    return vcvt_high_f32_f64(vcvt_f32_f64(lo), hi);
}

/*
 * 16 outputs, dst[0..15], each made as tap_sum4 makes it, in eight
 * accumulators (the low and high halves of prod0..prod3), so the eight adds
 * of one tap do not depend on each other (as x86/convolve_avx2.c's
 * taps16_avx2). Each output's operations and their order are unchanged.
 */
static inline void tap_sum16(const float *p, ptrdiff_t step, int taps, const float *k,
                             float *dst)
{
    float64x2_t acc0 = vdupq_n_f64(0.0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    float64x2_t acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
    for (int t = 0; t < taps; t++) {
        const float *in = p + t * step;
        const float32x4_t in0 = vld1q_f32(in);
        const float32x4_t in1 = vld1q_f32(in + 4);
        const float32x4_t in2 = vld1q_f32(in + 8);
        const float32x4_t in3 = vld1q_f32(in + 12);
        const float32x4_t prod0 = vmulq_n_f32(in0, k[t]);
        const float32x4_t prod1 = vmulq_n_f32(in1, k[t]);
        const float32x4_t prod2 = vmulq_n_f32(in2, k[t]);
        const float32x4_t prod3 = vmulq_n_f32(in3, k[t]);
        acc0 = vaddq_f64(acc0, vcvt_f64_f32(vget_low_f32(prod0)));
        acc1 = vaddq_f64(acc1, vcvt_high_f64_f32(prod0));
        acc2 = vaddq_f64(acc2, vcvt_f64_f32(vget_low_f32(prod1)));
        acc3 = vaddq_f64(acc3, vcvt_high_f64_f32(prod1));
        acc4 = vaddq_f64(acc4, vcvt_f64_f32(vget_low_f32(prod2)));
        acc5 = vaddq_f64(acc5, vcvt_high_f64_f32(prod2));
        acc6 = vaddq_f64(acc6, vcvt_f64_f32(vget_low_f32(prod3)));
        acc7 = vaddq_f64(acc7, vcvt_high_f64_f32(prod3));
    }
    vst1q_f32(dst, vcvt_high_f32_f64(vcvt_f32_f64(acc0), acc1));
    vst1q_f32(dst + 4, vcvt_high_f32_f64(vcvt_f32_f64(acc2), acc3));
    vst1q_f32(dst + 8, vcvt_high_f32_f64(vcvt_f32_f64(acc4), acc5));
    vst1q_f32(dst + 12, vcvt_high_f32_f64(vcvt_f32_f64(acc6), acc7));
}

/* dst[x] for 0 <= x < n: the taps of src[x], src[x + step], ... */
static void filter_row(const float *src, ptrdiff_t step, int taps, const float *k, int n,
                       float *dst)
{
    int x = 0;
    for (; x + 16 <= n; x += 16)
        tap_sum16(src + x, step, taps, k, dst + x);
    for (; x + 4 <= n; x += 4)
        vst1q_f32(dst + x, tap_sum4(src + x, step, taps, k));
    for (; x < n; x++)
        dst[x] = tap_sum_scalar(src + x, step, taps, k);
}

void iqa_convolve_neon(float *img, int w, int h, const float *kernel_h, const float *kernel_v,
                       int kw, int kh, float *workspace, float *result, int *rw, int *rh)
{
    const int uc = kw / 2;
    const int vc = kh / 2;
    const int kh_even = (kh & 1) ? 0 : 1;
    const int dst_w = w - kw + 1;
    const int dst_h = h - kh + 1;
    float *dst = result ? result : img;  /* NULL: in place, as the scalar code */

    float *cache = workspace;
    if (!cache) {
        cache = calloc((size_t)w * (size_t)h, sizeof(float));
        if (!cache) {
            if (rw) *rw = 0;
            if (rh) *rh = 0;
            return;
        }
    }

    /* horizontal: cache row ky, columns uc .. uc + dst_w - 1. The scalar
     * code also writes row dst_h + vc for an even-tap kernel, which the
     * vertical pass never reads (and which is past the cache), so it is
     * skipped as in x86/convolve_avx2.c. */
    for (int ky = 0; ky < dst_h + 2 * vc - kh_even; ky++) {
        const ptrdiff_t row = (ptrdiff_t)ky * w;
        filter_row(img + row, 1, kw, kernel_h, dst_w, cache + row + uc);
    }
    /* vertical: output row y from cache rows y .. y + kh - 1, column x + uc */
    for (int y = 0; y < dst_h; y++)
        filter_row(cache + (ptrdiff_t)y * w + uc, w, kh, kernel_v, dst_w,
                   dst + (ptrdiff_t)y * dst_w);

    if (cache != workspace)
        free(cache);
    if (rw) *rw = dst_w;
    if (rh) *rh = dst_h;
}
