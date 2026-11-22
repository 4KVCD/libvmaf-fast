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
 * with T (the sample type), FN (a name for its functions), LOAD4_S32 (4
 * samples into 32-bit lanes), LOAD8_U16 (8 samples into 16-bit lanes) and
 * SSE_ROW (VMAFx's squared errors of a row, as a 32-bit sum) defined. The
 * same kernels as x86/xpsnr_any_avx2_kernels.h (see there for the
 * formulas), each step half as wide.
 *
 * Each kernel sums one row of a block. Each step of a row makes a full
 * vector of results and drops the lanes past the block (lanes_below). Its
 * loads read the picture directly where the row goes on far enough
 * (`readable`: the samples from the block's first column to the row's
 * end); only at the picture's right edge does a step read a zero-filled
 * copy of the samples the C code reads there, so nothing is read past the
 * row.
 */

/* Copies n samples of a row and zero-fills the rest of dst's size. */
static inline void FN(copy_padded)(T *dst, int size, const T *src, int n)
{
    memcpy(dst, src, sizeof(T) * n);
    memset(dst + n, 0, sizeof(T) * (size - n));
}

/* --- b_val 1: |12 c - 2 (4 neighbours) - (4 diagonals)| for every sample */

/* |f| of the 4 samples from mid[0], the rows above and below at up and down */
static inline uint32x4_t FN(highpass4)(const T *up, const T *mid, const T *down)
{
    const int32x4_t centre = LOAD4_S32(mid);
    const int32x4_t sides = vaddq_s32(vaddq_s32(LOAD4_S32(mid - 1), LOAD4_S32(mid + 1)),
                                      vaddq_s32(LOAD4_S32(up), LOAD4_S32(down)));
    const int32x4_t corners = vaddq_s32(vaddq_s32(LOAD4_S32(up - 1), LOAD4_S32(up + 1)),
                                        vaddq_s32(LOAD4_S32(down - 1), LOAD4_S32(down + 1)));
    const int32x4_t f = vsubq_s32(vsubq_s32(vmulq_n_s32(centre, 12), vshlq_n_s32(sides, 1)), corners);
    return vreinterpretq_u32_s32(vabsq_s32(f));
}

/* columns x_act .. w_act - 1 of the row at mid */
static inline uint64_t FN(highpass_row)(int x_act, int w_act, const T *mid, ptrdiff_t o, int readable)
{
    uint32x4_t row = vdupq_n_u32(0);
    for (int x = x_act; x < w_act; x += 4) {
        const int n = w_act - x;  /* results of the block left */
        uint32x4_t f;
        if (x + 5 <= readable) {  /* columns x - 1 .. x + 4 */
            f = FN(highpass4)(mid - o + x, mid + x, mid + o + x);
        } else {
            /* the C code's columns x - 1 .. w_act (n < 4 here) */
            T rows[3][6];
            for (int r = 0; r < 3; r++)
                FN(copy_padded)(rows[r], 6, mid + (r - 1) * o + x - 1, n + 2);
            f = FN(highpass4)(rows[0] + 1, rows[1] + 1, rows[2] + 1);
        }
        if (n < 4)
            f = vandq_u32(f, lanes_below(n));
        row = vaddq_u32(row, f);
    }
    return vaddlvq_u32(row);
}

/* --- b_val 2: FN(highds) of xpsnr_template.c, at every second sample of
 * every second row, from the column sums of row pairs A, B and C (see
 * x86/xpsnr_any_avx2_kernels.h) */

/* A, B and C of 8 columns from column `at` of rows y-2 .. y+3 (r[0..5]),
 * in 16-bit lanes (at most 2 * 4095) */
static inline void FN(pair_sums)(const T *const r[6], int at, uint16x8_t *a, uint16x8_t *b,
                                 uint16x8_t *c)
{
    *a = vaddq_u16(LOAD8_U16(r[2] + at), LOAD8_U16(r[3] + at));
    *b = vaddq_u16(LOAD8_U16(r[1] + at), LOAD8_U16(r[4] + at));
    *c = vaddq_u16(LOAD8_U16(r[0] + at), LOAD8_U16(r[5] + at));
}

/* |f| at the 4 columns x, x + 2, x + 4, x + 6: lane k is column x + 2k.
 * Loading 8 columns from x - 2, x and x + 2, the even ones of each give
 * x - 2 + 2k, x + 2k, x + 2 + 2k and the odd ones the columns after. */
static inline uint32x4_t FN(highds4)(const T *const r[6], int x)
{
    uint16x8_t a_l, b_l, c_l, a_m, b_m, c_m, a_r, b_r, c_r;
    FN(pair_sums)(r, x - 2, &a_l, &b_l, &c_l);
    FN(pair_sums)(r, x, &a_m, &b_m, &c_m);
    FN(pair_sums)(r, x + 2, &a_r, &b_r, &c_r);
    /* even(v): columns at + 2k; odd(v): at + 2k + 1 (the low and high
     * halves of each 32-bit lane, arm64 being little-endian) */
#define EVEN(v) vandq_u32(vreinterpretq_u32_u16(v), vdupq_n_u32(0xFFFF))
#define ODD(v) vshrq_n_u32(vreinterpretq_u32_u16(v), 16)
    const uint32x4_t centre = vaddq_u32(EVEN(a_m), ODD(a_m));     /* A[x] + A[x+1] */
    const uint32x4_t above_below = vaddq_u32(EVEN(b_m), ODD(b_m)); /* B[x] + B[x+1] */
    const uint32x4_t sides = vaddq_u32(ODD(a_l), EVEN(a_r));       /* A[x-1] + A[x+2] */
    const uint32x4_t corners = vaddq_u32(ODD(b_l), EVEN(b_r));     /* B[x-1] + B[x+2] */
    const uint32x4_t ring = vaddq_u32(                             /* the 16 outer samples */
        vaddq_u32(vaddq_u32(ODD(c_l), EVEN(c_m)), vaddq_u32(ODD(c_m), EVEN(c_r))),
        vaddq_u32(vaddq_u32(EVEN(a_l), EVEN(b_l)), vaddq_u32(ODD(a_r), ODD(b_r))));
#undef EVEN
#undef ODD
    int32x4_t f = vmulq_n_s32(vreinterpretq_s32_u32(centre), 12);
    f = vsubq_s32(f, vmulq_n_s32(vreinterpretq_s32_u32(above_below), 3));
    f = vsubq_s32(f, vmulq_n_s32(vreinterpretq_s32_u32(sides), 3));
    f = vsubq_s32(f, vshlq_n_s32(vreinterpretq_s32_u32(corners), 1));
    f = vsubq_s32(f, vreinterpretq_s32_u32(ring));
    return vreinterpretq_u32_s32(vabsq_s32(f));
}

/* columns x_act, x_act + 2, ... below w_act of the row (pair) at row_y */
static inline uint64_t FN(highds_row)(int x_act, int w_act, const T *row_y, ptrdiff_t o, int readable)
{
    const T *r[6];
    for (int i = 0; i < 6; i++)
        r[i] = row_y + (i - 2) * o;
    uint32x4_t row = vdupq_n_u32(0);
    for (int x = x_act; x < w_act; x += 8) {
        const int n = (w_act - x + 1) / 2;  /* results of the block left */
        uint32x4_t f;
        if (x + 10 <= readable) {  /* columns x - 2 .. x + 9 */
            f = FN(highds4)(r, x);
        } else {
            /* the C code's columns x - 2 .. x + 2n + 1 (n < 4 here) */
            T rows[6][12];
            const T *t[6];
            for (int i = 0; i < 6; i++) {
                FN(copy_padded)(rows[i], 12, r[i] + x - 2, 2 * n + 4);
                t[i] = rows[i];
            }
            f = FN(highds4)(t, 2);
        }
        if (n < 4)
            f = vandq_u32(f, lanes_below(n));
        row = vaddq_u32(row, f);
    }
    return vaddlvq_u32(row);
}

/* --- temporal: |m0 - 2 m1 + m2| (m2 = m1 for the first order, m0 - m1) */

/* for every sample (b_val 1), 8 at a time in 16-bit lanes (|t| <= 2 * 4095);
 * keep: the 16-bit lanes to count; pairs of them added into the 32-bit
 * lanes of row */
static inline uint32x4_t FN(temporal8)(uint32x4_t row, const T *p0, const T *p1, const T *p2,
                                       uint16x8_t keep)
{
    const int16x8_t m0 = vreinterpretq_s16_u16(LOAD8_U16(p0));
    const int16x8_t m1 = vreinterpretq_s16_u16(LOAD8_U16(p1));
    const int16x8_t m2 = vreinterpretq_s16_u16(LOAD8_U16(p2));
    const int16x8_t t = vsubq_s16(vaddq_s16(m0, m2), vaddq_s16(m1, m1));
    return vpadalq_u16(row, vandq_u16(vreinterpretq_u16_s16(vabsq_s16(t)), keep));
}

/* the `width` samples of one row of m0, m1 and m2 */
static inline uint64_t FN(temporal_row)(int width, const T *p0, const T *p1, const T *p2, int readable)
{
    const uint16x8_t all = vdupq_n_u16(0xFFFF);
    uint32x4_t row = vdupq_n_u32(0);
    for (int x = 0; x < width; x += 8) {
        const int n = width - x;  /* samples of the block left */
        if (x + 8 <= readable) {
            row = FN(temporal8)(row, p0 + x, p1 + x, p2 + x, n < 8 ? lanes_below_16(n) : all);
        } else {
            /* the C code's n < 8 samples; zeros give t = 0 */
            T rows[3][8];
            FN(copy_padded)(rows[0], 8, p0 + x, n);
            FN(copy_padded)(rows[1], 8, p1 + x, n);
            FN(copy_padded)(rows[2], 8, p2 + x, n);
            row = FN(temporal8)(row, rows[0], rows[1], rows[2], all);
        }
    }
    return vaddlvq_u32(row);
}

/* over the 2x2 sums at every second sample of every second row (b_val 2,
 * FN(diff2nd) of xpsnr_template.c): 4 at a time, each 2x2 sum the 16-bit
 * sum of two rows (<= 2 * 4095), its neighbouring columns added by vpaddl */
static inline uint32x4_t FN(temporal_ds4)(const T *const p[6])
{
    const int32x4_t s0 = vreinterpretq_s32_u32(vpaddlq_u16(vaddq_u16(LOAD8_U16(p[0]), LOAD8_U16(p[1]))));
    const int32x4_t s1 = vreinterpretq_s32_u32(vpaddlq_u16(vaddq_u16(LOAD8_U16(p[2]), LOAD8_U16(p[3]))));
    const int32x4_t s2 = vreinterpretq_s32_u32(vpaddlq_u16(vaddq_u16(LOAD8_U16(p[4]), LOAD8_U16(p[5]))));
    const int32x4_t t = vsubq_s32(vaddq_s32(s0, s2), vaddq_s32(s1, s1));
    return vreinterpretq_u32_s32(vabsq_s32(t));
}

/* the row pair at m0, m1 and m2 (rows y and y + 1), `width` samples wide */
static inline uint64_t FN(temporal_ds_row)(int width, const T *m0, ptrdiff_t o0, const T *m1, ptrdiff_t o1,
                                           const T *m2, ptrdiff_t o2, int readable)
{
    const T *r[6] = { m0, m0 + o0, m1, m1 + o1, m2, m2 + o2 };
    uint32x4_t row = vdupq_n_u32(0);
    for (int x = 0; x < width; x += 8) {
        const int n = (width - x + 1) / 2;  /* results of the block left */
        const T *p[6];
        T rows[6][8];
        if (x + 8 <= readable) {  /* columns x .. x + 7 */
            for (int i = 0; i < 6; i++)
                p[i] = r[i] + x;
        } else {
            /* the C code's columns x .. x + 2n - 1 (n < 4 here) */
            for (int i = 0; i < 6; i++) {
                FN(copy_padded)(rows[i], 8, r[i] + x, 2 * n);
                p[i] = rows[i];
            }
        }
        uint32x4_t t = FN(temporal_ds4)(p);
        if (n < 4)
            t = vandq_u32(t, lanes_below(n));
        row = vaddq_u32(row, t);
    }
    return vaddlvq_u32(row);
}

/* --- one strip of blocks, row by row (see x86/xpsnr_any_avx2.h) */

void FN(xpsnr_strip_neon)(const XpsnrStrip *st, uint64_t *sse, uint64_t *sa, uint64_t *ta)
{
    const int w = st->w, b = st->b, b_val = st->b_val;
    const int blocks = (w + b - 1) / b;
    const ptrdiff_t so = st->so, sr = st->sr, s1 = st->s1;
    /* the first order is the second with m2 = m1 */
    const T *const m2 = st->second_order ? st->m2 : st->m1;
    const ptrdiff_t s2 = st->second_order ? st->s2 : st->s1;

    memset(sse, 0, sizeof(*sse) * blocks);
    memset(sa, 0, sizeof(*sa) * blocks);
    memset(ta, 0, sizeof(*ta) * blocks);
    /* b_val 2 works on row pairs: its sums of activity take every second
     * row (y_act is 0 or 2), its squared errors both */
    for (int y = 0; y < st->height; y += b_val) {
        const T *const org = (const T *) st->org + y * so;
        const T *const rec = (const T *) st->rec + y * sr;
        const T *const p1 = (const T *) st->m1 + y * s1;
        const T *const p2 = m2 + y * s2;
        const bool spatial = y >= st->y_act && y < st->h_act;
        for (int bx = 0, x0 = 0; bx < blocks; bx++, x0 += b) {
            const int bw = b < w - x0 ? b : w - x0;
            const int readable = w - x0;  /* the samples to the row's end */
            for (int r = 0; r < b_val; r++)
                sse[bx] += SSE_ROW(org + r * so + x0, rec + r * sr + x0, bw);
            if (spatial) {
                /* the columns FN(block) of xpsnr_template.c takes */
                const int x_act = x0 > 0 ? 0 : b_val;
                const int w_act = x0 + bw < w ? bw : bw - b_val;
                if (b_val == 1)
                    sa[bx] += FN(highpass_row)(x_act, w_act, org + x0, so, readable);
                else if (w_act > 12)  /* narrower: FFmpeg drops the sum */
                    sa[bx] += FN(highds_row)(x_act, w_act, org + x0, so, readable);
            }
            ta[bx] += b_val == 1
                ? FN(temporal_row)(bw, org + x0, p1 + x0, p2 + x0, readable)
                : FN(temporal_ds_row)(bw, org + x0, so, p1 + x0, s1, p2 + x0, s2, readable);
        }
    }
}
