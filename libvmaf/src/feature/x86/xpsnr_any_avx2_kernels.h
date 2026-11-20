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
 * The kernels of x86/xpsnr_any_avx2.c for one sample type, included there
 * with T (the sample type), FN (a name for its functions), LOAD8_EPI32 (8
 * samples into 32-bit lanes), LOAD16_EPI16 (16 samples into 16-bit lanes)
 * and SSE_ROW (VMAFx's squared errors of a row, as a 32-bit sum) defined.
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

/* |f| of the 8 samples from mid[0], the rows above and below at up and down */
static inline __m256i FN(highpass8)(const T *up, const T *mid, const T *down)
{
    const __m256i centre = LOAD8_EPI32(mid);
    const __m256i sides = _mm256_add_epi32(_mm256_add_epi32(LOAD8_EPI32(mid - 1), LOAD8_EPI32(mid + 1)),
                                           _mm256_add_epi32(LOAD8_EPI32(up), LOAD8_EPI32(down)));
    const __m256i corners = _mm256_add_epi32(_mm256_add_epi32(LOAD8_EPI32(up - 1), LOAD8_EPI32(up + 1)),
                                             _mm256_add_epi32(LOAD8_EPI32(down - 1), LOAD8_EPI32(down + 1)));
    const __m256i f = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_mullo_epi32(centre, _mm256_set1_epi32(12)),
                                                        _mm256_slli_epi32(sides, 1)),
                                       corners);
    return _mm256_abs_epi32(f);
}

/* columns x_act .. w_act - 1 of the row at mid */
static inline uint64_t FN(highpass_row)(int x_act, int w_act, const T *mid, ptrdiff_t o, int readable)
{
    __m256i row = _mm256_setzero_si256();
    for (int x = x_act; x < w_act; x += 8) {
        const int n = w_act - x;  /* results of the block left */
        __m256i f;
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
            f = _mm256_and_si256(f, lanes_below(n));
        row = _mm256_add_epi32(row, f);
    }
    return sum_lanes(row);
}

/* --- b_val 2: FN(highds) of xpsnr_template.c, at every second sample of
 * every second row. With the column sums of row pairs around y,
 * A = rows y + y+1, B = rows y-1 + y+2, C = rows y-2 + y+3, its f at x is
 *     12 (A[x] + A[x+1]) - 3 (B[x] + B[x+1]) - 3 (A[x-1] + A[x+2])
 *   - 2 (B[x-1] + B[x+2]) - (C[x-1] + C[x] + C[x+1] + C[x+2])
 *   - (A[x-2] + B[x-2] + A[x+3] + B[x+3]),
 * term by term the C code's 36 samples. */

/* A, B and C of 16 columns from column `at` of rows y-2 .. y+3 (r[0..5]),
 * in 16-bit lanes (at most 2 * 4095) */
static inline void FN(pair_sums)(const T *const r[6], int at, __m256i *a, __m256i *b, __m256i *c)
{
    *a = _mm256_add_epi16(LOAD16_EPI16(r[2] + at), LOAD16_EPI16(r[3] + at));
    *b = _mm256_add_epi16(LOAD16_EPI16(r[1] + at), LOAD16_EPI16(r[4] + at));
    *c = _mm256_add_epi16(LOAD16_EPI16(r[0] + at), LOAD16_EPI16(r[5] + at));
}

/* |f| at the 8 columns x, x + 2, ... x + 14: lane k is column x + 2k.
 * Loading 16 columns from x - 2, x and x + 2, the even ones of each give
 * x - 2 + 2k, x + 2k, x + 2 + 2k and the odd ones the columns after. */
static inline __m256i FN(highds8)(const T *const r[6], int x)
{
    __m256i a_l, b_l, c_l, a_m, b_m, c_m, a_r, b_r, c_r;
    FN(pair_sums)(r, x - 2, &a_l, &b_l, &c_l);
    FN(pair_sums)(r, x, &a_m, &b_m, &c_m);
    FN(pair_sums)(r, x + 2, &a_r, &b_r, &c_r);
    /* even(v): columns at + 2k; odd(v): at + 2k + 1 */
#define EVEN(v) _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF))
#define ODD(v) _mm256_srli_epi32(v, 16)
    const __m256i centre = _mm256_add_epi32(EVEN(a_m), ODD(a_m));     /* A[x] + A[x+1] */
    const __m256i above_below = _mm256_add_epi32(EVEN(b_m), ODD(b_m)); /* B[x] + B[x+1] */
    const __m256i sides = _mm256_add_epi32(ODD(a_l), EVEN(a_r));       /* A[x-1] + A[x+2] */
    const __m256i corners = _mm256_add_epi32(ODD(b_l), EVEN(b_r));     /* B[x-1] + B[x+2] */
    const __m256i ring = _mm256_add_epi32(                             /* the 16 outer samples */
        _mm256_add_epi32(_mm256_add_epi32(ODD(c_l), EVEN(c_m)), _mm256_add_epi32(ODD(c_m), EVEN(c_r))),
        _mm256_add_epi32(_mm256_add_epi32(EVEN(a_l), EVEN(b_l)), _mm256_add_epi32(ODD(a_r), ODD(b_r))));
#undef EVEN
#undef ODD
    const __m256i three = _mm256_set1_epi32(3);
    __m256i f = _mm256_mullo_epi32(centre, _mm256_set1_epi32(12));
    f = _mm256_sub_epi32(f, _mm256_mullo_epi32(above_below, three));
    f = _mm256_sub_epi32(f, _mm256_mullo_epi32(sides, three));
    f = _mm256_sub_epi32(f, _mm256_slli_epi32(corners, 1));
    f = _mm256_sub_epi32(f, ring);
    return _mm256_abs_epi32(f);
}

/* columns x_act, x_act + 2, ... below w_act of the row (pair) at row */
static inline uint64_t FN(highds_row)(int x_act, int w_act, const T *row_y, ptrdiff_t o, int readable)
{
    const T *r[6];
    for (int i = 0; i < 6; i++)
        r[i] = row_y + (i - 2) * o;
    __m256i row = _mm256_setzero_si256();
    for (int x = x_act; x < w_act; x += 16) {
        const int n = (w_act - x + 1) / 2;  /* results of the block left */
        __m256i f;
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
        if (n < 8)
            f = _mm256_and_si256(f, lanes_below(n));
        row = _mm256_add_epi32(row, f);
    }
    return sum_lanes(row);
}

/* --- temporal: |m0 - 2 m1 + m2| (m2 = m1 for the first order, m0 - m1) */

/* for every sample (b_val 1), 16 at a time in 16-bit lanes (|t| <= 2 * 4095);
 * keep: the 16-bit lanes to count; then pairs added into 32-bit lanes */
static inline __m256i FN(temporal16)(const T *p0, const T *p1, const T *p2, __m256i keep)
{
    const __m256i m0 = LOAD16_EPI16(p0), m1 = LOAD16_EPI16(p1), m2 = LOAD16_EPI16(p2);
    const __m256i t = _mm256_sub_epi16(_mm256_add_epi16(m0, m2), _mm256_add_epi16(m1, m1));
    return _mm256_madd_epi16(_mm256_and_si256(_mm256_abs_epi16(t), keep), _mm256_set1_epi16(1));
}

/* the `width` samples of one row of m0, m1 and m2 */
static inline uint64_t FN(temporal_row)(int width, const T *p0, const T *p1, const T *p2, int readable)
{
    const __m256i all = _mm256_set1_epi16(-1);
    __m256i row = _mm256_setzero_si256();
    for (int x = 0; x < width; x += 16) {
        const int n = width - x;  /* samples of the block left */
        if (x + 16 <= readable) {
            /* the 16-bit lanes below n */
            const __m256i keep = n < 16 ? _mm256_cmpgt_epi16(_mm256_set1_epi16((short) n),
                                              _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
                                        : all;
            row = _mm256_add_epi32(row, FN(temporal16)(p0 + x, p1 + x, p2 + x, keep));
        } else {
            /* the C code's n < 16 samples; zeros give t = 0 */
            T rows[3][16];
            FN(copy_padded)(rows[0], 16, p0 + x, n);
            FN(copy_padded)(rows[1], 16, p1 + x, n);
            FN(copy_padded)(rows[2], 16, p2 + x, n);
            row = _mm256_add_epi32(row, FN(temporal16)(rows[0], rows[1], rows[2], all));
        }
    }
    return sum_lanes(row);
}

/* over the 2x2 sums at every second sample of every second row (b_val 2,
 * FN(diff2nd) of xpsnr_template.c): 8 at a time, each 2x2 sum the 16-bit
 * sum of two rows (<= 2 * 4095), its neighbouring columns added by madd */
static inline __m256i FN(temporal_ds8)(const T *const p[6])
{
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i s0 = _mm256_madd_epi16(_mm256_add_epi16(LOAD16_EPI16(p[0]), LOAD16_EPI16(p[1])), ones);
    const __m256i s1 = _mm256_madd_epi16(_mm256_add_epi16(LOAD16_EPI16(p[2]), LOAD16_EPI16(p[3])), ones);
    const __m256i s2 = _mm256_madd_epi16(_mm256_add_epi16(LOAD16_EPI16(p[4]), LOAD16_EPI16(p[5])), ones);
    const __m256i t = _mm256_sub_epi32(_mm256_add_epi32(s0, s2), _mm256_add_epi32(s1, s1));
    return _mm256_abs_epi32(t);
}

/* the row pair at m0, m1 and m2 (rows y and y + 1), `width` samples wide */
static inline uint64_t FN(temporal_ds_row)(int width, const T *m0, ptrdiff_t o0, const T *m1, ptrdiff_t o1,
                                           const T *m2, ptrdiff_t o2, int readable)
{
    const T *r[6] = { m0, m0 + o0, m1, m1 + o1, m2, m2 + o2 };
    __m256i row = _mm256_setzero_si256();
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
        __m256i t = FN(temporal_ds8)(p);
        if (n < 8)
            t = _mm256_and_si256(t, lanes_below(n));
        row = _mm256_add_epi32(row, t);
    }
    return sum_lanes(row);
}

/* --- one strip of blocks, row by row (see ../xpsnr_strip.h) */

#define XPSNR_STRIP FN(xpsnr_strip_avx2)
#include "../xpsnr_strip_template.h"
#undef XPSNR_STRIP
