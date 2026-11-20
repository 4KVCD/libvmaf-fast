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
 * The kernels of arm64/xpsnr_any_neon.c for one sample type, included there
 * with T (the sample type), FN (a name for its functions, and the loads
 * FN(load8) and FN(load16) of that type) and SSE_ROW (the squared errors
 * of a row, as a 32-bit sum) defined. The same kernels, with the same
 * arguments, as x86/xpsnr_any_avx2_kernels.h.
 *
 * Each kernel sums one row of a block. Each step of a row makes a full
 * vector of results and drops the lanes past the block (lanes_below_16 and
 * lanes_below_32). Its loads read the picture directly where the row goes
 * on far enough (`readable`: the samples from the block's first column to
 * the row's end); only at the picture's right edge does a step read a
 * zero-filled copy of the samples the C code reads there, so nothing is
 * read past the row.
 */

/* Copies n samples of a row and zero-fills the rest of dst's size. */
static inline void FN(copy_padded)(T *dst, int size, const T *src, int n)
{
    memcpy(dst, src, sizeof(T) * n);
    memset(dst + n, 0, sizeof(T) * (size - n));
}

/* --- b_val 1: |12 c - 2 (4 neighbours) - (4 diagonals)| for every sample */

/* |f| of the 8 samples from mid[0], the rows above and below at up and
 * down. f = 12 c - (2 sides + corners), and both parts are at most
 * 12 * 4095 < 2^16, so |f| is their 16-bit absolute difference. */
static inline uint16x8_t FN(highpass8)(const T *up, const T *mid, const T *down)
{
    const uint16x8_t centre = FN(load8)(mid);
    const uint16x8_t left = FN(load8)(mid - 1);
    const uint16x8_t right = FN(load8)(mid + 1);
    const uint16x8_t above = FN(load8)(up);
    const uint16x8_t below = FN(load8)(down);
    const uint16x8_t above_left = FN(load8)(up - 1);
    const uint16x8_t above_right = FN(load8)(up + 1);
    const uint16x8_t below_left = FN(load8)(down - 1);
    const uint16x8_t below_right = FN(load8)(down + 1);
    const uint16x8_t sides = vaddq_u16(vaddq_u16(left, right), vaddq_u16(above, below));
    const uint16x8_t corners = vaddq_u16(vaddq_u16(above_left, above_right),
                                         vaddq_u16(below_left, below_right));
    const uint16x8_t positive = vmulq_n_u16(centre, 12);
    const uint16x8_t negative = vmlaq_n_u16(corners, sides, 2);
    return vabdq_u16(positive, negative);
}

/* columns x_act .. w_act - 1 of the row at mid */
static inline uint64_t FN(highpass_row)(int x_act, int w_act, const T *mid, ptrdiff_t o, int readable)
{
    uint32x4_t row = vdupq_n_u32(0);
    for (int x = x_act; x < w_act; x += 8) {
        const int n = w_act - x;  /* results of the block left */
        uint16x8_t f;
        if (x + 9 <= readable) {  /* columns x - 1 .. x + 8 */
            f = FN(highpass8)(mid - o + x, mid + x, mid + o + x);
        } else {
            /* the C code's columns x - 1 .. w_act (n < 8 here) */
            T rows[3][10];
            for (int r = 0; r < 3; r++)
                FN(copy_padded)(rows[r], 10, mid + (r - 1) * o + x - 1, n + 2);
            f = FN(highpass8)(rows[0] + 1, rows[1] + 1, rows[2] + 1);
        }
        if (n < 8)
            f = vandq_u16(f, lanes_below_16(n));
        row = vpadalq_u16(row, f);
    }
    return vaddlvq_u32(row);
}

/* --- b_val 2: FN(highds) of xpsnr_template.c, at every second sample of
 * every second row. With the column sums of row pairs around y,
 * A = rows y + y+1, B = rows y-1 + y+2, C = rows y-2 + y+3, its f at x is
 *     12 (A[x] + A[x+1]) - 3 (B[x] + B[x+1]) - 3 (A[x-1] + A[x+2])
 *   - 2 (B[x-1] + B[x+2]) - (C[x-1] + C[x] + C[x+1] + C[x+2])
 *   - (A[x-2] + B[x-2] + A[x+3] + B[x+3]),
 * term by term the C code's 36 samples. */

/* rows p + q at the 16 columns from p[0] and q[0], in 16-bit lanes (at
 * most 2 * 4095): columns 0 .. 7 in val[0], 8 .. 15 in val[1] */
static inline uint16x8x2_t FN(add_rows)(const T *p, const T *q)
{
    const uint16x8x2_t a = FN(load16)(p);
    const uint16x8x2_t b = FN(load16)(q);
    return (uint16x8x2_t) { { vaddq_u16(a.val[0], b.val[0]), vaddq_u16(a.val[1], b.val[1]) } };
}

/* A, B or C (rows p + q) of 16 columns, split into the even columns and
 * the odd ones: vuzp1q takes the even lanes of its two arguments in
 * order, vuzp2q the odd ones */
static inline EvenOdd FN(pair_sum)(const T *p, const T *q)
{
    const uint16x8x2_t sum = FN(add_rows)(p, q);
    return (EvenOdd) { vuzp1q_u16(sum.val[0], sum.val[1]), vuzp2q_u16(sum.val[0], sum.val[1]) };
}

/* A, B and C of 16 columns from column `at` of rows y-2 .. y+3 (r[0..5]) */
static inline void FN(pair_sums)(const T *const r[6], int at, EvenOdd *a, EvenOdd *b, EvenOdd *c)
{
    *a = FN(pair_sum)(r[2] + at, r[3] + at);
    *b = FN(pair_sum)(r[1] + at, r[4] + at);
    *c = FN(pair_sum)(r[0] + at, r[5] + at);
}

/* |f| at the 8 columns x, x + 2, ... x + 14: lane k of val[0] is column
 * x + 2k, of val[1] column x + 8 + 2k. Taking 16 columns from x - 2, x
 * and x + 2, the even ones of each give x - 2 + 2k, x + 2k, x + 2 + 2k
 * and the odd ones the columns after. */
static inline uint32x4x2_t FN(highds8)(const T *const r[6], int x)
{
    EvenOdd a_l, b_l, c_l, a_m, b_m, c_m, a_r, b_r, c_r;
    FN(pair_sums)(r, x - 2, &a_l, &b_l, &c_l);
    FN(pair_sums)(r, x, &a_m, &b_m, &c_m);
    FN(pair_sums)(r, x + 2, &a_r, &b_r, &c_r);
    /* each at most 4 * 4095, and the ring of 16 samples 16 * 4095 < 2^16 */
    const uint16x8_t centre = vaddq_u16(a_m.even, a_m.odd);      /* A[x] + A[x+1] */
    const uint16x8_t above_below = vaddq_u16(b_m.even, b_m.odd); /* B[x] + B[x+1] */
    const uint16x8_t sides = vaddq_u16(a_l.odd, a_r.even);       /* A[x-1] + A[x+2] */
    const uint16x8_t corners = vaddq_u16(b_l.odd, b_r.even);     /* B[x-1] + B[x+2] */
    const uint16x8_t ring = vaddq_u16(                           /* the 16 outer samples */
        vaddq_u16(vaddq_u16(c_l.odd, c_m.even), vaddq_u16(c_m.odd, c_r.even)),
        vaddq_u16(vaddq_u16(a_l.even, b_l.even), vaddq_u16(a_r.odd, b_r.odd)));
    const uint16x8_t threes = vaddq_u16(above_below, sides);
    /* f = 12 centre - (3 threes + 2 corners + ring), both parts up to
     * 48 * 4095, so in 32-bit lanes */
    const uint32x4_t positive_lo = vmull_n_u16(vget_low_u16(centre), 12);
    const uint32x4_t positive_hi = vmull_high_n_u16(centre, 12);
    uint32x4_t negative_lo = vmull_n_u16(vget_low_u16(threes), 3);
    uint32x4_t negative_hi = vmull_high_n_u16(threes, 3);
    negative_lo = vmlal_n_u16(negative_lo, vget_low_u16(corners), 2);
    negative_hi = vmlal_high_n_u16(negative_hi, corners, 2);
    negative_lo = vaddw_u16(negative_lo, vget_low_u16(ring));
    negative_hi = vaddw_high_u16(negative_hi, ring);
    return (uint32x4x2_t) { { vabdq_u32(positive_lo, negative_lo), vabdq_u32(positive_hi, negative_hi) } };
}

/* columns x_act, x_act + 2, ... below w_act of the row (pair) at row_y */
static inline uint64_t FN(highds_row)(int x_act, int w_act, const T *row_y, ptrdiff_t o, int readable)
{
    const T *r[6];
    for (int i = 0; i < 6; i++)
        r[i] = row_y + (i - 2) * o;
    uint32x4_t row = vdupq_n_u32(0);
    for (int x = x_act; x < w_act; x += 16) {
        const int n = (w_act - x + 1) / 2;  /* results of the block left */
        uint32x4x2_t f;
        if (x + 18 <= readable) {  /* columns x - 2 .. x + 17 */
            f = FN(highds8)(r, x);
        } else {
            /* the C code's columns x - 2 .. x + 2n + 1 (n < 8 here) */
            T rows[6][20];
            const T *t[6];
            for (int i = 0; i < 6; i++) {
                FN(copy_padded)(rows[i], 20, r[i] + x - 2, 2 * n + 4);
                t[i] = rows[i];
            }
            f = FN(highds8)(t, 2);
        }
        if (n < 8) {
            f.val[0] = vandq_u32(f.val[0], lanes_below_32(n));
            f.val[1] = vandq_u32(f.val[1], lanes_below_32(n - 4));
        }
        row = vaddq_u32(row, f.val[0]);
        row = vaddq_u32(row, f.val[1]);
    }
    return vaddlvq_u32(row);
}

/* --- temporal: |m0 - 2 m1 + m2| (m2 = m1 for the first order, m0 - m1) */

/* for every sample (b_val 1), 16 at a time: m0 + m2 and 2 m1 are at most
 * 2 * 4095, so |t| is their 16-bit absolute difference; columns 0 .. 7 in
 * val[0], 8 .. 15 in val[1] */
static inline uint16x8x2_t FN(temporal16)(const T *p0, const T *p1, const T *p2)
{
    const uint16x8x2_t outer = FN(add_rows)(p0, p2);
    const uint16x8x2_t middle = FN(load16)(p1);
    return (uint16x8x2_t) { { vabdq_u16(outer.val[0], vshlq_n_u16(middle.val[0], 1)),
                              vabdq_u16(outer.val[1], vshlq_n_u16(middle.val[1], 1)) } };
}

/* the `width` samples of one row of m0, m1 and m2 */
static inline uint64_t FN(temporal_row)(int width, const T *p0, const T *p1, const T *p2, int readable)
{
    uint32x4_t row0 = vdupq_n_u32(0);
    uint32x4_t row1 = vdupq_n_u32(0);
    for (int x = 0; x < width; x += 16) {
        const int n = width - x;  /* samples of the block left */
        uint16x8x2_t t;
        if (x + 16 <= readable) {
            t = FN(temporal16)(p0 + x, p1 + x, p2 + x);
        } else {
            /* the C code's n < 16 samples */
            T rows[3][16];
            FN(copy_padded)(rows[0], 16, p0 + x, n);
            FN(copy_padded)(rows[1], 16, p1 + x, n);
            FN(copy_padded)(rows[2], 16, p2 + x, n);
            t = FN(temporal16)(rows[0], rows[1], rows[2]);
        }
        if (n < 16) {
            t.val[0] = vandq_u16(t.val[0], lanes_below_16(n));
            t.val[1] = vandq_u16(t.val[1], lanes_below_16(n - 8));
        }
        row0 = vpadalq_u16(row0, t.val[0]);
        row1 = vpadalq_u16(row1, t.val[1]);
    }
    return vaddlvq_u32(vaddq_u32(row0, row1));
}

/* the 2x2 sums of rows p and q at the 16 columns from p[0]: lane k is
 * columns 2k and 2k + 1 (vpaddq adds neighbouring lanes, in order), at
 * most 4 * 4095 */
static inline uint16x8_t FN(sums2x2)(const T *p, const T *q)
{
    const uint16x8x2_t sum = FN(add_rows)(p, q);
    return vpaddq_u16(sum.val[0], sum.val[1]);
}

/* over the 2x2 sums at every second sample of every second row (b_val 2,
 * FN(diff2nd) of xpsnr_template.c), 8 at a time: s0 + s2 and 2 s1 are at
 * most 8 * 4095, so |t| is their 16-bit absolute difference */
static inline uint16x8_t FN(temporal_ds8)(const T *const p[6])
{
    const uint16x8_t s0 = FN(sums2x2)(p[0], p[1]);
    const uint16x8_t s1 = FN(sums2x2)(p[2], p[3]);
    const uint16x8_t s2 = FN(sums2x2)(p[4], p[5]);
    return vabdq_u16(vaddq_u16(s0, s2), vaddq_u16(s1, s1));
}

/* the row pair at m0, m1 and m2 (rows y and y + 1), `width` samples wide */
static inline uint64_t FN(temporal_ds_row)(int width, const T *m0, ptrdiff_t o0, const T *m1, ptrdiff_t o1,
                                           const T *m2, ptrdiff_t o2, int readable)
{
    const T *r[6] = { m0, m0 + o0, m1, m1 + o1, m2, m2 + o2 };
    uint32x4_t row = vdupq_n_u32(0);
    for (int x = 0; x < width; x += 16) {
        const int n = (width - x + 1) / 2;  /* results of the block left */
        const T *p[6];
        T rows[6][16];
        if (x + 16 <= readable) {  /* columns x .. x + 15 */
            for (int i = 0; i < 6; i++)
                p[i] = r[i] + x;
        } else {
            /* the C code's columns x .. x + 2n - 1 (n < 8 here) */
            for (int i = 0; i < 6; i++) {
                FN(copy_padded)(rows[i], 16, r[i] + x, 2 * n);
                p[i] = rows[i];
            }
        }
        uint16x8_t t = FN(temporal_ds8)(p);
        if (n < 8)
            t = vandq_u16(t, lanes_below_16(n));
        row = vpadalq_u16(row, t);
    }
    return vaddlvq_u32(row);
}

/* --- one strip of blocks, row by row (see ../xpsnr_strip.h) */

#define XPSNR_STRIP FN(xpsnr_strip_neon)
#include "../xpsnr_strip_template.h"
#undef XPSNR_STRIP
