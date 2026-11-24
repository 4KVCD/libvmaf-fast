/**
 *
 *  Copyright (c) 2011, Tom Distler (http://tdistler.com)
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent AND BSD-3-Clause
 */

/*
 * libvmaf-fast: iqa/decimate.c's _iqa_decimate at factor 2 (MS-SSIM's
 * downscaling to its next scale), 16 or 4 output pixels at a time, as
 * x86/decimate_avx2.c. _iqa_filter_pixel adds each tap's float product to a
 * double: here each lane does the same, a float multiply (rounded to float),
 * a conversion to double and a double add, in the same tap order, so every
 * output is the scalar code's. Nothing is fused: the product is rounded to
 * float before it meets the double sum, in the C code too.
 * The input rows are first split into even and odd samples, so each tap's
 * inputs for neighbouring outputs are consecutive floats.
 */

#include <arm_neon.h>
#include <stddef.h>

#include "decimate_neon.h"

void iqa_deinterleave_neon(const float *row, int w, float *even, float *odd)
{
    int i = 0;
    for (; i + 8 <= w; i += 8) {
        const float32x4_t a = vld1q_f32(row + i);
        const float32x4_t b = vld1q_f32(row + i + 4);
        vst1q_f32(even + i / 2, vuzp1q_f32(a, b));
        vst1q_f32(odd + i / 2, vuzp2q_f32(a, b));
    }
    for (; i + 2 <= w; i += 2) {
        even[i / 2] = row[i];
        odd[i / 2] = row[i + 1];
    }
    if (i < w)
        even[i / 2] = row[i];
}

/* 4 outputs' sums rounded to float, as _iqa_filter_pixel's
 * (float)(sum * 1.0f). */
static inline void store4(float *dst, float64x2_t lo, float64x2_t hi)
{
    vst1q_f32(dst, vcvt_high_f32_f64(vcvt_f32_f64(lo), hi));
}

/* The largest kernel taken (MS-SSIM's is 9 x 9). */
#define MAX_TAPS (15 * 15)

int iqa_decimate2_row_neon(const float *const *even, const float *const *odd,
                           const float *kernel, int kw, int kh, int x0, int x1, float *dst)
{
    if (kw * kh > MAX_TAPS)
        return x0;
    /* each tap's input for output x is src[tap][x + shift[tap]]: input
     * column 2x + u is even sample x + u / 2 (u even) or odd sample
     * x + (u - 1) / 2 */
    const float *src[MAX_TAPS];
    int shift[MAX_TAPS];
    const int uc = kw / 2;
    int taps = 0;
    for (int v = 0; v < kh; ++v) {
        for (int u = -uc; u <= uc; ++u, ++taps) {
            src[taps] = u & 1 ? odd[v] : even[v];
            shift[taps] = (u - (u & 1)) / 2;
        }
    }

    int x = x0;
    for (; x + 16 <= x1; x += 16) {
        /* eight double sums, each one output pair's chain as in C */
        float64x2_t acc0 = vdupq_n_f64(0.0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        float64x2_t acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
        for (int t = 0; t < taps; ++t) {
            const float tap = kernel[t];
            const float *in = src[t] + (x + shift[t]);
            const float32x4_t in0 = vld1q_f32(in);
            const float32x4_t in1 = vld1q_f32(in + 4);
            const float32x4_t in2 = vld1q_f32(in + 8);
            const float32x4_t in3 = vld1q_f32(in + 12);
            const float32x4_t p0 = vmulq_n_f32(in0, tap);
            const float32x4_t p1 = vmulq_n_f32(in1, tap);
            const float32x4_t p2 = vmulq_n_f32(in2, tap);
            const float32x4_t p3 = vmulq_n_f32(in3, tap);
            acc0 = vaddq_f64(acc0, vcvt_f64_f32(vget_low_f32(p0)));
            acc1 = vaddq_f64(acc1, vcvt_high_f64_f32(p0));
            acc2 = vaddq_f64(acc2, vcvt_f64_f32(vget_low_f32(p1)));
            acc3 = vaddq_f64(acc3, vcvt_high_f64_f32(p1));
            acc4 = vaddq_f64(acc4, vcvt_f64_f32(vget_low_f32(p2)));
            acc5 = vaddq_f64(acc5, vcvt_high_f64_f32(p2));
            acc6 = vaddq_f64(acc6, vcvt_f64_f32(vget_low_f32(p3)));
            acc7 = vaddq_f64(acc7, vcvt_high_f64_f32(p3));
        }
        store4(dst + x, acc0, acc1);
        store4(dst + x + 4, acc2, acc3);
        store4(dst + x + 8, acc4, acc5);
        store4(dst + x + 12, acc6, acc7);
    }
    for (; x + 4 <= x1; x += 4) {
        float64x2_t acc0 = vdupq_n_f64(0.0), acc1 = acc0;
        for (int t = 0; t < taps; ++t) {
            const float32x4_t in = vld1q_f32(src[t] + (x + shift[t]));
            const float32x4_t p = vmulq_n_f32(in, kernel[t]);
            acc0 = vaddq_f64(acc0, vcvt_f64_f32(vget_low_f32(p)));
            acc1 = vaddq_f64(acc1, vcvt_high_f64_f32(p));
        }
        store4(dst + x, acc0, acc1);
    }
    return x;
}
