/*
 * Copyright 2001-2012 Xiph.Org and contributors.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

/*
 * libvmaf-fast: third_party/xiph/psnr_hvs.c's blocks, 8 neighbouring blocks
 * of a block row at a time, one block per lane, with the same scores to the
 * last bit: every lane does the scalar code's operations on its block in
 * the same order (float sums in pixel and coefficient order, the integer
 * DCT with the same wrapping multiplies and shifts, sqrt in double), and
 * the 64 error terms of each block are added to the plane's one float sum
 * one by one, in the scalar code's order.
 * Speed: that one float sum is a single chain through the whole plane; the
 * rest of a block's work (its own sums are chains of 64 additions too)
 * overlaps across the 8 lanes. Terms of 0 are not added: x + 0.0f is x
 * for every x the sum can hold (it starts at +0 and every term is a square,
 * so it is never -0), so the chain is only as long as the visible errors.
 */

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "psnr_hvs_avx2.h"

#define BLOCKS 8

/* Transposes 8 vectors of 8 32-bit values: r[a] lane b becomes r[b] lane a. */
static inline void transpose8(__m256i r[8])
{
    const __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]);  /* a0 b0 a1 b1 | a4 b4 a5 b5 */
    const __m256i t1 = _mm256_unpackhi_epi32(r[0], r[1]);  /* a2 b2 a3 b3 | a6 b6 a7 b7 */
    const __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    const __m256i t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    const __m256i t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    const __m256i t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    const __m256i u0 = _mm256_unpacklo_epi64(t0, t2);      /* a0 b0 c0 d0 | a4 b4 c4 d4 */
    const __m256i u1 = _mm256_unpackhi_epi64(t0, t2);      /* a1 b1 c1 d1 | a5 b5 c5 d5 */
    const __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
    const __m256i u3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i u4 = _mm256_unpacklo_epi64(t4, t6);      /* e0 f0 g0 h0 | e4 f4 g4 h4 */
    const __m256i u5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i u6 = _mm256_unpacklo_epi64(t5, t7);
    const __m256i u7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256(u0, u4, 0x20);
    r[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
    r[2] = _mm256_permute2x128_si256(u2, u6, 0x20);
    r[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
    r[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    r[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    r[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    r[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

/* OD_DCT_RSHIFT(a, 1): a / 2 rounded towards 0. */
static inline __m256i half_of(__m256i a)
{
    return _mm256_srai_epi32(_mm256_add_epi32(_mm256_srli_epi32(a, 31), a), 1);
}

/* (t * c + r) >> s, with the C code's wrapping 32-bit multiply */
#define MUL_SHIFT(t, c, r, s) \
    _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32((t), _mm256_set1_epi32(c)), \
                                       _mm256_set1_epi32(r)), (s))

/* od_bin_fdct8 of 8 lanes at once: y[k] from the inputs x[0..7]. */
static void fdct8(__m256i y[8], const __m256i x[8])
{
    __m256i t0 = x[0], t4 = x[1], t2 = x[2], t6 = x[3];
    __m256i t7 = x[4], t3 = x[5], t5 = x[6], t1 = x[7];
    t1 = _mm256_sub_epi32(t0, t1);
    const __m256i t1h = half_of(t1);
    t0 = _mm256_sub_epi32(t0, t1h);
    t4 = _mm256_add_epi32(t4, t5);
    const __m256i t4h = half_of(t4);
    t5 = _mm256_sub_epi32(t5, t4h);
    t3 = _mm256_sub_epi32(t2, t3);
    t2 = _mm256_sub_epi32(t2, half_of(t3));
    t6 = _mm256_add_epi32(t6, t7);
    const __m256i t6h = half_of(t6);
    t7 = _mm256_sub_epi32(t6h, t7);
    t0 = _mm256_add_epi32(t0, t6h);
    t6 = _mm256_sub_epi32(t0, t6);
    t2 = _mm256_sub_epi32(t4h, t2);
    t4 = _mm256_sub_epi32(t2, t4);
    t0 = _mm256_sub_epi32(t0, MUL_SHIFT(t4, 13573, 16384, 15));
    t4 = _mm256_add_epi32(t4, MUL_SHIFT(t0, 11585, 8192, 14));
    t0 = _mm256_sub_epi32(t0, MUL_SHIFT(t4, 13573, 16384, 15));
    t6 = _mm256_sub_epi32(t6, MUL_SHIFT(t2, 21895, 16384, 15));
    t2 = _mm256_add_epi32(t2, MUL_SHIFT(t6, 15137, 8192, 14));
    t6 = _mm256_sub_epi32(t6, MUL_SHIFT(t2, 21895, 16384, 15));
    t3 = _mm256_add_epi32(t3, MUL_SHIFT(t5, 19195, 16384, 15));
    t5 = _mm256_add_epi32(t5, MUL_SHIFT(t3, 11585, 8192, 14));
    t3 = _mm256_sub_epi32(t3, MUL_SHIFT(t5, 7489, 4096, 13));
    t7 = _mm256_sub_epi32(half_of(t5), t7);
    t5 = _mm256_sub_epi32(t5, t7);
    t3 = _mm256_sub_epi32(t1h, t3);
    t1 = _mm256_sub_epi32(t1, t3);
    t7 = _mm256_add_epi32(t7, MUL_SHIFT(t1, 3227, 16384, 15));
    t1 = _mm256_sub_epi32(t1, MUL_SHIFT(t7, 6393, 16384, 15));
    t7 = _mm256_add_epi32(t7, MUL_SHIFT(t1, 3227, 16384, 15));
    t5 = _mm256_add_epi32(t5, MUL_SHIFT(t3, 2485, 4096, 13));
    t3 = _mm256_sub_epi32(t3, MUL_SHIFT(t5, 18205, 16384, 15));
    t5 = _mm256_add_epi32(t5, MUL_SHIFT(t3, 2485, 4096, 13));
    y[0] = t0; y[1] = t1; y[2] = t2; y[3] = t3;
    y[4] = t4; y[5] = t5; y[6] = t6; y[7] = t7;
}

/* od_bin_fdct8x8 in place: c[i * 8 + j] holds sample (i, j) of every lane's
 * block and then its coefficient (i vertical, j horizontal). */
static void fdct8x8(__m256i c[64])
{
    __m256i z[64], in[8];
    for (int col = 0; col < 8; ++col) {  /* the columns first */
        for (int k = 0; k < 8; ++k)
            in[k] = c[k * 8 + col];
        fdct8(z + 8 * col, in);
    }
    for (int i = 0; i < 8; ++i) {        /* then the rows */
        for (int m = 0; m < 8; ++m)
            in[m] = z[8 * m + i];
        fdct8(c + 8 * i, in);
    }
}

/* Row i of 8 blocks as 32-bit samples, lane j, from p (8 samples). */
static inline __m256i load_row(const unsigned char *p, int depth)
{
    return depth > 8 ? _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p))
                     : _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
}

/* One picture's 8 blocks at (x + b * step, y): their DCT coefficients into
 * c and their masking thresholds (the scalar code's s_mask after its sqrt)
 * as the result. */
static __m256 picture_blocks(const unsigned char *pic, int stride, int depth, int x, int y,
                             int step, const float *mask, __m256i c[64])
{
    const int bytes = depth > 8 ? 2 : 1;
    for (int i = 0; i < 8; ++i) {
        __m256i r[BLOCKS];
        const unsigned char *row = pic + (ptrdiff_t)(y + i) * stride;
        for (int b = 0; b < BLOCKS; ++b)
            r[b] = load_row(row + (ptrdiff_t)(x + b * step) * bytes, depth);
        transpose8(r);
        for (int j = 0; j < 8; ++j)
            c[i * 8 + j] = r[j];
    }

    /* the whole block's mean and its 4 quarters' (sub: rows i / 4, columns
     * j / 4), then their variances, each a float sum in pixel order */
    const __m256 zero = _mm256_setzero_ps();
    __m256 gmean = zero, means[4] = { zero, zero, zero, zero };
    for (int k = 0; k < 64; ++k) {
        const int sub = (k >> 5) + ((k >> 1) & 2);
        const __m256 v = _mm256_cvtepi32_ps(c[k]);
        gmean = _mm256_add_ps(gmean, v);
        means[sub] = _mm256_add_ps(means[sub], v);
    }
    gmean = _mm256_div_ps(gmean, _mm256_set1_ps(64.f));
    for (int s = 0; s < 4; ++s)
        means[s] = _mm256_div_ps(means[s], _mm256_set1_ps(16.f));
    __m256 gvar = zero, vars[4] = { zero, zero, zero, zero };
    for (int k = 0; k < 64; ++k) {
        const int sub = (k >> 5) + ((k >> 1) & 2);
        const __m256 v = _mm256_cvtepi32_ps(c[k]);
        const __m256 dg = _mm256_sub_ps(v, gmean);
        const __m256 dm = _mm256_sub_ps(v, means[sub]);
        gvar = _mm256_add_ps(gvar, _mm256_mul_ps(dg, dg));
        vars[sub] = _mm256_add_ps(vars[sub], _mm256_mul_ps(dm, dm));
    }
    gvar = _mm256_mul_ps(gvar, _mm256_set1_ps(1 / 63.f * 64));
    for (int s = 0; s < 4; ++s)
        vars[s] = _mm256_mul_ps(vars[s], _mm256_set1_ps(1 / 15.f * 16));
    /* gvar = (vars[0] + ... + vars[3]) / gvar where gvar > 0 */
    const __m256 quarters = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(vars[0], vars[1]), vars[2]), vars[3]);
    gvar = _mm256_blendv_ps(gvar, _mm256_div_ps(quarters, gvar), _mm256_cmp_ps(gvar, zero, _CMP_GT_OQ));

    fdct8x8(c);

    /* the AC coefficients' energy, weighted by the masking table */
    __m256 energy = zero;
    for (int k = 1; k < 64; ++k) {
        const __m256 square = _mm256_cvtepi32_ps(_mm256_mullo_epi32(c[k], c[k]));
        energy = _mm256_add_ps(energy, _mm256_mul_ps(square, _mm256_set1_ps(mask[k])));
    }
    /* sqrt(energy * gvar) / 32.f, in double as the scalar code's sqrt */
    const __m256 product = _mm256_mul_ps(energy, gvar);
    const __m256d thirty_two = _mm256_set1_pd(32.0);
    const __m256d lo = _mm256_div_pd(_mm256_sqrt_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(product))),
                                     thirty_two);
    const __m256d hi = _mm256_div_pd(_mm256_sqrt_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(product, 1))),
                                     thirty_two);
    return _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo));
}

/* The index of the lowest set bit of m (not 0). */
static inline int lowest_bit(uint64_t m)
{
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanForward64(&index, m);
    return (int)index;
#else
    return __builtin_ctzll(m);
#endif
}

int psnr_hvs_blocks_avx2(const unsigned char *src, int src_stride, const unsigned char *dst,
                         int dst_stride, int depth, int y, int w, int step, const float *mask,
                         const float *csf, float *ret, int *pixels)
{
    float sum = *ret;
    int x = 0;
    for (; x + (BLOCKS - 1) * step < w - 7; x += BLOCKS * step) {
        __m256i cs[64], cd[64];
        const __m256 s_mask = picture_blocks(src, src_stride, depth, x, y, step, mask, cs);
        const __m256 d_mask = picture_blocks(dst, dst_stride, depth, x, y, step, mask, cd);
        /* the larger threshold: d_mask where d_mask > s_mask, as in C */
        const __m256 threshold = _mm256_max_ps(d_mask, s_mask);

        /* each coefficient's error term, 8 at a time per block once
         * transposed: terms[b][k] */
        const __m256 zero = _mm256_setzero_ps();
        float terms[BLOCKS][64];
        uint64_t nonzero[BLOCKS] = { 0 };
        for (int g = 0; g < 8; ++g) {
            __m256i t[8];
            for (int n = 0; n < 8; ++n) {
                const int k = 8 * g + n;
                __m256 err = _mm256_cvtepi32_ps(_mm256_abs_epi32(_mm256_sub_epi32(cs[k], cd[k])));
                if (k) {
                    const __m256 below = _mm256_div_ps(threshold, _mm256_set1_ps(mask[k]));
                    err = _mm256_blendv_ps(_mm256_sub_ps(err, below), zero,
                                           _mm256_cmp_ps(err, below, _CMP_LT_OQ));
                }
                const __m256 weighted = _mm256_mul_ps(err, _mm256_set1_ps(csf[k]));
                t[n] = _mm256_castps_si256(_mm256_mul_ps(weighted, weighted));
            }
            transpose8(t);
            for (int b = 0; b < BLOCKS; ++b) {
                const __m256 v = _mm256_castsi256_ps(t[b]);
                _mm256_storeu_ps(&terms[b][8 * g], v);
                /* NaN counts as not 0: it is added */
                const unsigned bits = (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(v, zero, _CMP_NEQ_UQ));
                nonzero[b] |= (uint64_t)bits << (8 * g);
            }
        }
        /* the plane's sum, block by block, coefficient by coefficient */
        for (int b = 0; b < BLOCKS; ++b) {
            for (uint64_t m = nonzero[b]; m; m &= m - 1)
                sum += terms[b][lowest_bit(m)];
        }
        *pixels += 64 * BLOCKS;
    }
    *ret = sum;
    return x;
}
