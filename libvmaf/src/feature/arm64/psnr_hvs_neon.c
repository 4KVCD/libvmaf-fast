/*
 * Copyright 2001-2012 Xiph.Org and contributors.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

/*
 * libvmaf-fast: third_party/xiph/psnr_hvs.c's blocks, 4 neighbouring blocks
 * of a block row at a time, one block per lane, with the same scores to the
 * last bit, as x86/psnr_hvs_avx2.c: every lane does the scalar code's
 * operations on its block in the same order (float sums in pixel and
 * coefficient order, the integer DCT with the same wrapping multiplies and
 * shifts, sqrt in double), and each block's 64 error terms are added to the
 * plane's one float sum one by one, in the scalar code's order.
 * On arm64 the C code is built with floating-point contraction, so clang
 * fuses a * b + c inside one expression there: the variance sums, the
 * masking energy and the plane's error sum. Those three use fused
 * multiply-adds here too (vfmaq, fmaf), every other step stays unfused.
 * Terms of 0 are not added: fmaf(0, 0, sum) is sum for every sum the plane
 * can hold (it starts at +0 and only grows by squares, so it is never -0),
 * so the chain is only as long as the visible errors.
 */

#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "psnr_hvs_neon.h"

#define BLOCKS 4

/* Transposes 4 vectors of 4 32-bit values: r[a] lane b becomes r[b] lane a. */
static inline void transpose4(int32x4_t r[4])
{
    const int32x4_t t0 = vtrn1q_s32(r[0], r[1]);  /* a0 b0 a2 b2 */
    const int32x4_t t1 = vtrn2q_s32(r[0], r[1]);  /* a1 b1 a3 b3 */
    const int32x4_t t2 = vtrn1q_s32(r[2], r[3]);  /* c0 d0 c2 d2 */
    const int32x4_t t3 = vtrn2q_s32(r[2], r[3]);  /* c1 d1 c3 d3 */
    r[0] = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
    r[1] = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
    r[2] = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
    r[3] = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
}

/* OD_DCT_RSHIFT(a, 1): a / 2 rounded towards 0, as (a + (a >>> 31)) >> 1
 * with the sum wrapping and the last shift arithmetic. */
static inline int32x4_t half_of(int32x4_t a)
{
    const uint32x4_t u = vreinterpretq_u32_s32(a);
    return vshrq_n_s32(vreinterpretq_s32_u32(vsraq_n_u32(u, u, 31)), 1);
}

/* (t * c + r) >> s with the C code's wrapping 32-bit multiply and add; a
 * rounding shift would not wrap where t * c + r passes 2^31. */
#define MUL_SHIFT(t, c, r, s) vshrq_n_s32(vmlaq_n_s32(vdupq_n_s32(r), (t), (c)), (s))

/* od_bin_fdct8 of 4 lanes at once: y[k] from the inputs x[0..7]. */
static void fdct8(int32x4_t y[8], const int32x4_t x[8])
{
    int32x4_t t0 = x[0], t4 = x[1], t2 = x[2], t6 = x[3];
    int32x4_t t7 = x[4], t3 = x[5], t5 = x[6], t1 = x[7];
    t1 = vsubq_s32(t0, t1);
    const int32x4_t t1h = half_of(t1);
    t0 = vsubq_s32(t0, t1h);
    t4 = vaddq_s32(t4, t5);
    const int32x4_t t4h = half_of(t4);
    t5 = vsubq_s32(t5, t4h);
    t3 = vsubq_s32(t2, t3);
    t2 = vsubq_s32(t2, half_of(t3));
    t6 = vaddq_s32(t6, t7);
    const int32x4_t t6h = half_of(t6);
    t7 = vsubq_s32(t6h, t7);
    t0 = vaddq_s32(t0, t6h);
    t6 = vsubq_s32(t0, t6);
    t2 = vsubq_s32(t4h, t2);
    t4 = vsubq_s32(t2, t4);
    t0 = vsubq_s32(t0, MUL_SHIFT(t4, 13573, 16384, 15));
    t4 = vaddq_s32(t4, MUL_SHIFT(t0, 11585, 8192, 14));
    t0 = vsubq_s32(t0, MUL_SHIFT(t4, 13573, 16384, 15));
    t6 = vsubq_s32(t6, MUL_SHIFT(t2, 21895, 16384, 15));
    t2 = vaddq_s32(t2, MUL_SHIFT(t6, 15137, 8192, 14));
    t6 = vsubq_s32(t6, MUL_SHIFT(t2, 21895, 16384, 15));
    t3 = vaddq_s32(t3, MUL_SHIFT(t5, 19195, 16384, 15));
    t5 = vaddq_s32(t5, MUL_SHIFT(t3, 11585, 8192, 14));
    t3 = vsubq_s32(t3, MUL_SHIFT(t5, 7489, 4096, 13));
    t7 = vsubq_s32(half_of(t5), t7);
    t5 = vsubq_s32(t5, t7);
    t3 = vsubq_s32(t1h, t3);
    t1 = vsubq_s32(t1, t3);
    t7 = vaddq_s32(t7, MUL_SHIFT(t1, 3227, 16384, 15));
    t1 = vsubq_s32(t1, MUL_SHIFT(t7, 6393, 16384, 15));
    t7 = vaddq_s32(t7, MUL_SHIFT(t1, 3227, 16384, 15));
    t5 = vaddq_s32(t5, MUL_SHIFT(t3, 2485, 4096, 13));
    t3 = vsubq_s32(t3, MUL_SHIFT(t5, 18205, 16384, 15));
    t5 = vaddq_s32(t5, MUL_SHIFT(t3, 2485, 4096, 13));
    y[0] = t0; y[1] = t1; y[2] = t2; y[3] = t3;
    y[4] = t4; y[5] = t5; y[6] = t6; y[7] = t7;
}

/* od_bin_fdct8x8 in place: c[i * 8 + j] holds sample (i, j) of every lane's
 * block and then its coefficient (i vertical, j horizontal). */
static void fdct8x8(int32x4_t c[64])
{
    int32x4_t z[64], in[8];
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

/* The 8 samples at p widened to 16 bits: 16-bit little-endian samples above
 * 8 bits, as the C code reads them byte by byte, else bytes. */
static inline uint16x8_t load_row(const unsigned char *p, int depth)
{
    if (depth > 8)
        return vreinterpretq_u16_u8(vld1q_u8(p));
    return vmovl_u8(vld1_u8(p));
}

/* One picture's 4 blocks at (x + b * step, y): their DCT coefficients into
 * c and their masking thresholds (the scalar code's s_mask after its sqrt)
 * as the result. */
static float32x4_t picture_blocks(const unsigned char *pic, int stride, int depth, int x, int y,
                                  int step, const float *mask, int32x4_t c[64])
{
    const int bytes = depth > 8 ? 2 : 1;
    for (int i = 0; i < 8; ++i) {
        const unsigned char *row = pic + (ptrdiff_t)(y + i) * stride;
        int32x4_t lo[BLOCKS], hi[BLOCKS];  /* columns 0-3 and 4-7 of each block */
        for (int b = 0; b < BLOCKS; ++b) {
            const uint16x8_t r = load_row(row + (ptrdiff_t)(x + b * step) * bytes, depth);
            lo[b] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(r)));
            hi[b] = vreinterpretq_s32_u32(vmovl_high_u16(r));
        }
        transpose4(lo);
        transpose4(hi);
        for (int j = 0; j < 4; ++j) {
            c[i * 8 + j] = lo[j];
            c[i * 8 + 4 + j] = hi[j];
        }
    }

    /* the whole block's mean and its 4 quarters' (sub: rows i / 4, columns
     * j / 4), then their variances, each a float sum in pixel order */
    const float32x4_t zero = vdupq_n_f32(0.f);
    float32x4_t gmean = zero, means[4] = { zero, zero, zero, zero };
    for (int k = 0; k < 64; ++k) {
        const int sub = (k >> 5) + ((k >> 1) & 2);
        const float32x4_t v = vcvtq_f32_s32(c[k]);
        gmean = vaddq_f32(gmean, v);
        means[sub] = vaddq_f32(means[sub], v);
    }
    /* / 64 and / 16 as exact multiplies by powers of two */
    gmean = vmulq_n_f32(gmean, 1 / 64.f);
    for (int s = 0; s < 4; ++s)
        means[s] = vmulq_n_f32(means[s], 1 / 16.f);
    float32x4_t gvar = zero, vars[4] = { zero, zero, zero, zero };
    for (int k = 0; k < 64; ++k) {
        const int sub = (k >> 5) + ((k >> 1) & 2);
        const float32x4_t v = vcvtq_f32_s32(c[k]);
        const float32x4_t dg = vsubq_f32(v, gmean);
        const float32x4_t dm = vsubq_f32(v, means[sub]);
        gvar = vfmaq_f32(gvar, dg, dg);
        vars[sub] = vfmaq_f32(vars[sub], dm, dm);
    }
    gvar = vmulq_n_f32(gvar, 1 / 63.f * 64);
    for (int s = 0; s < 4; ++s)
        vars[s] = vmulq_n_f32(vars[s], 1 / 15.f * 16);
    /* gvar = (vars[0] + ... + vars[3]) / gvar where gvar > 0 */
    const float32x4_t quarters = vaddq_f32(vaddq_f32(vaddq_f32(vars[0], vars[1]), vars[2]), vars[3]);
    gvar = vbslq_f32(vcgtq_f32(gvar, zero), vdivq_f32(quarters, gvar), gvar);

    fdct8x8(c);

    /* the AC coefficients' energy, weighted by the masking table */
    float32x4_t energy = zero;
    for (int k = 1; k < 64; ++k) {
        const float32x4_t square = vcvtq_f32_s32(vmulq_s32(c[k], c[k]));
        energy = vfmaq_n_f32(energy, square, mask[k]);
    }
    /* sqrt(energy * gvar) / 32.f, in double as the scalar code's sqrt (the
     * / 32 as an exact multiply) */
    const float32x4_t product = vmulq_f32(energy, gvar);
    const float64x2_t lo = vmulq_n_f64(vsqrtq_f64(vcvt_f64_f32(vget_low_f32(product))), 1 / 32.0);
    const float64x2_t hi = vmulq_n_f64(vsqrtq_f64(vcvt_high_f64_f32(product)), 1 / 32.0);
    return vcvt_high_f32_f64(vcvt_f32_f64(lo), hi);
}

/* Stores each block's weighted error of one coefficient in term: a = (err
 * less threshold / mask, 0 below it) * csf. Returns bit in the lanes whose a
 * is not 0 (NaN is not 0: the C code adds it). */
static inline uint32x4_t masked_term(int32x4_t s, int32x4_t d, float32x4_t threshold, float mask,
                                     float csf, float *term, uint32_t bit)
{
    const float32x4_t err = vcvtq_f32_s32(vabdq_s32(s, d));
    const float32x4_t below = vdivq_f32(threshold, vdupq_n_f32(mask));
    const float32x4_t masked = vbslq_f32(vcltq_f32(err, below), vdupq_n_f32(0.f), vsubq_f32(err, below));
    const float32x4_t weighted = vmulq_n_f32(masked, csf);
    vst1q_f32(term, weighted);
    return vbicq_u32(vdupq_n_u32(bit), vceqzq_f32(weighted));
}

int psnr_hvs_blocks_neon(const unsigned char *src, int src_stride, const unsigned char *dst,
                         int dst_stride, int depth, int y, int w, int step, const float *mask,
                         const float *csf, float *ret, int *pixels)
{
    float sum = *ret;
    int x = 0;
    for (; x + (BLOCKS - 1) * step < w - 7; x += BLOCKS * step) {
        int32x4_t cs[64], cd[64];
        const float32x4_t s_mask = picture_blocks(src, src_stride, depth, x, y, step, mask, cs);
        const float32x4_t d_mask = picture_blocks(dst, dst_stride, depth, x, y, step, mask, cd);
        /* d_mask where d_mask > s_mask, as the C code's if (also for NaN) */
        const float32x4_t threshold = vbslq_f32(vcgtq_f32(d_mask, s_mask), d_mask, s_mask);

        /* each coefficient's weighted error a (the C code adds a * a), and
         * per block bit k % 32 of nonzero[k / 32] for each a that is not 0.
         * The DC coefficient (k = 0) is not masked. */
        float terms[64][BLOCKS];
        const float32x4_t dc = vmulq_n_f32(vcvtq_f32_s32(vabdq_s32(cs[0], cd[0])), csf[0]);
        vst1q_f32(terms[0], dc);
        uint32x4_t nonzero[2] = { vbicq_u32(vdupq_n_u32(1u), vceqzq_f32(dc)), vdupq_n_u32(0) };
        for (int k = 1; k < 64; ++k) {
            const uint32x4_t bit = masked_term(cs[k], cd[k], threshold, mask[k], csf[k], terms[k], 1u << (k & 31));
            nonzero[k >> 5] = vorrq_u32(nonzero[k >> 5], bit);
        }
        uint32_t bits0[BLOCKS], bits1[BLOCKS];
        vst1q_u32(bits0, nonzero[0]);
        vst1q_u32(bits1, nonzero[1]);
        /* the plane's sum, block by block, coefficient by coefficient */
        for (int b = 0; b < BLOCKS; ++b) {
            for (uint64_t m = bits0[b] | (uint64_t)bits1[b] << 32; m; m &= m - 1) {
                const float a = terms[__builtin_ctzll(m)][b];
                sum = fmaf(a, a, sum);
            }
        }
        *pixels += 64 * BLOCKS;
    }
    *ret = sum;
    return x;
}
