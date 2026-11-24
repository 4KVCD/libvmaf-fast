/**
 *
 *  Copyright (c) 2011, Tom Distler (http://tdistler.com)
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent AND BSD-3-Clause
 */

/*
 * libvmaf-fast: iqa/decimate.c's _iqa_decimate at factor 2 (MS-SSIM's
 * downscaling to its next scale), 8 or 16 output pixels at a time.
 * _iqa_filter_pixel adds each tap's float product to a double: here each
 * lane does the same, a float multiply (rounded to float), a conversion to
 * double and a double add, in the same tap order, so every output is the
 * scalar code's.
 * Speed: an output reads the input at twice its own x, so the input rows
 * are first split into even and odd samples; then each tap's inputs for
 * neighbouring outputs are consecutive floats (no shuffles per tap), and
 * the float-to-double conversions, two per 8 products, set the pace. Four
 * double accumulators per 16 outputs let neighbouring pixels' additions
 * overlap (each pixel's own sum is one chain, as in the scalar code).
 */

#include <immintrin.h>
#include <stddef.h>

#include "decimate_avx2.h"

void iqa_deinterleave_avx2(const float *row, int w, float *even, float *odd)
{
    int i = 0;
    for (; i + 16 <= w; i += 16) {
        const __m256 a = _mm256_loadu_ps(row + i);
        const __m256 b = _mm256_loadu_ps(row + i + 8);
        /* in each 128-bit half: a0 a2 b0 b2 | a4 a6 b4 b6, and the odd ones */
        const __m256 e = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
        const __m256 o = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
        /* the 64-bit pairs reordered: a0 a2 a4 a6 b0 b2 b4 b6 */
        _mm256_storeu_ps(even + i / 2, _mm256_castpd_ps(_mm256_permute4x64_pd(
                                           _mm256_castps_pd(e), _MM_SHUFFLE(3, 1, 2, 0))));
        _mm256_storeu_ps(odd + i / 2, _mm256_castpd_ps(_mm256_permute4x64_pd(
                                          _mm256_castps_pd(o), _MM_SHUFFLE(3, 1, 2, 0))));
    }
    for (; i < w; ++i) {
        if (i & 1)
            odd[i / 2] = row[i];
        else
            even[i / 2] = row[i];
    }
}

/* Adds the float products of 4 inputs and a tap to their double sums. */
static inline __m256d add_products(__m256d acc, const float *in, __m128 tap)
{
    return _mm256_add_pd(acc, _mm256_cvtps_pd(_mm_mul_ps(_mm_loadu_ps(in), tap)));
}

/* 4 + 4 sums rounded to float, as _iqa_filter_pixel's (float)(sum * 1.0f). */
static inline void store8(float *dst, __m256d lo, __m256d hi)
{
    _mm256_storeu_ps(dst, _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo)));
}

/* The largest kernel taken (MS-SSIM's is 9 x 9). */
#define MAX_TAPS (15 * 15)

int iqa_decimate2_row_avx2(const float *const *even, const float *const *odd,
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
        __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
        for (int t = 0; t < taps; ++t) {
            const __m128 tap = _mm_set1_ps(kernel[t]);
            const float *in = src[t] + (x + shift[t]);
            s0 = add_products(s0, in, tap);
            s1 = add_products(s1, in + 4, tap);
            s2 = add_products(s2, in + 8, tap);
            s3 = add_products(s3, in + 12, tap);
        }
        store8(dst + x, s0, s1);
        store8(dst + x + 8, s2, s3);
    }
    for (; x + 8 <= x1; x += 8) {
        __m256d s0 = _mm256_setzero_pd(), s1 = s0;
        for (int t = 0; t < taps; ++t) {
            const __m128 tap = _mm_set1_ps(kernel[t]);
            const float *in = src[t] + (x + shift[t]);
            s0 = add_products(s0, in, tap);
            s1 = add_products(s1, in + 4, tap);
        }
        store8(dst + x, s0, s1);
    }
    return x;
}
