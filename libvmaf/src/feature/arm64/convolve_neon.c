/**
 *
 *  Copyright (c) 2011, Tom Distler (http://tdistler.com)
 *  Copyright 2016-2026 Netflix, Inc.
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent AND BSD-3-Clause
 */

/*
 * libvmaf-fast's NEON copy of iqa's 1D separable convolution, which SSIM
 * spends most of its time in. Each output of the C code is
 *
 *     double sum = 0.0;
 *     for (each tap) sum += img[i] * kernel[j];
 *     out = (float)(sum * scale);
 *
 * where img[i] * kernel[j] is a float product, rounded to float, then
 * widened to double and added. This file does the same operations for
 * eight outputs at once: a float multiply, an exact widening to double, a
 * double add per tap in tap order, and one rounding to float at the end
 * (scale is 1 for a normalized kernel). No step may be fused into a
 * multiply-add, so meson builds this file with -ffp-contract=off.
 */

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#include "convolve_neon.h"

/* One output: tap t of it is src[t * tap_stride]. */
static float convolve1(const float *src, ptrdiff_t tap_stride, const float *kernel, int taps)
{
    double sum = 0.0;
    for (int t = 0; t < taps; t++)
        sum += src[t * tap_stride] * kernel[t];
    return (float) sum;
}

/* Eight neighbouring outputs: tap t of output i is src[i + t * tap_stride]. */
static void convolve8(const float *src, ptrdiff_t tap_stride, const float *kernel, int taps,
                      float *out)
{
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);
    for (int t = 0; t < taps; t++, src += tap_stride) {
        const float32x4_t lo = vld1q_f32(src);
        const float32x4_t hi = vld1q_f32(src + 4);
        const float32x4_t prod_lo = vmulq_n_f32(lo, kernel[t]);
        const float32x4_t prod_hi = vmulq_n_f32(hi, kernel[t]);
        acc0 = vaddq_f64(acc0, vcvt_f64_f32(vget_low_f32(prod_lo)));
        acc1 = vaddq_f64(acc1, vcvt_high_f64_f32(prod_lo));
        acc2 = vaddq_f64(acc2, vcvt_f64_f32(vget_low_f32(prod_hi)));
        acc3 = vaddq_f64(acc3, vcvt_high_f64_f32(prod_hi));
    }
    vst1q_f32(out, vcvt_high_f32_f64(vcvt_f32_f64(acc0), acc1));
    vst1q_f32(out + 4, vcvt_high_f32_f64(vcvt_f32_f64(acc2), acc3));
}

/*
 * n neighbouring outputs, output x starting at src[x]. When n is not a
 * multiple of eight, the last group of eight overlaps the one before it
 * and writes the same values again; rows of fewer than eight outputs go
 * one output at a time.
 */
static void convolve_row(const float *src, ptrdiff_t tap_stride, const float *kernel, int taps,
                         int n, float *out)
{
    if (n < 8) {
        for (int x = 0; x < n; x++)
            out[x] = convolve1(src + x, tap_stride, kernel, taps);
        return;
    }
    for (int x = 0; x + 8 <= n; x += 8)
        convolve8(src + x, tap_stride, kernel, taps, out + x);
    if (n % 8)
        convolve8(src + n - 8, tap_stride, kernel, taps, out + n - 8);
}

void iqa_convolve_neon(float *img, int w, int h, const float *kernel_h, const float *kernel_v,
                       int kw, int kh, int normalized, float *workspace, float *result, int *rw,
                       int *rh)
{
    assert(normalized == 1);
    assert(w >= kw && h >= kh);
    (void) normalized;

    const int uc = kw / 2;
    const int vc = kh / 2;
    const int kh_even = !(kh & 1);
    const int dst_w = w - kw + 1;
    const int dst_h = h - kh + 1;
    float *const dst = result ? result : img;

    float *cache = workspace;
    if (!cache) {
        cache = calloc((size_t) w * h, sizeof(float));
        if (!cache) {
            if (rw)
                *rw = 0;
            if (rh)
                *rh = 0;
            return;
        }
    }

    /* Horizontal pass into the cache, at the columns and rows the C code
     * writes (an even kernel's last row is never read, so not written). */
    const int cache_rows = dst_h + 2 * vc - kh_even;
    for (int y = 0; y < cache_rows; y++) {
        const float *const row = img + (ptrdiff_t) y * w;
        convolve_row(row, 1, kernel_h, kw, dst_w, cache + (ptrdiff_t) y * w + uc);
    }

    /* Vertical pass: output (x, y) sums the cache column x + uc from row y down. */
    for (int y = 0; y < dst_h; y++) {
        const float *const top = cache + (ptrdiff_t) y * w + uc;
        convolve_row(top, w, kernel_v, kh, dst_w, dst + (ptrdiff_t) y * dst_w);
    }

    if (cache != workspace)
        free(cache);
    if (rw)
        *rw = dst_w;
    if (rh)
        *rh = dst_h;
}
