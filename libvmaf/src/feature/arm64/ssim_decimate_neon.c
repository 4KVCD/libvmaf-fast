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
 * libvmaf-fast's NEON copy of x86/ssim_decimate_avx2.c (see there and
 * ssim.c): ssim.c's decimate_picture, each output the sum over a scale x
 * scale box, made as column sums (each column's `scale` rows) and then box
 * sums (`scale` column sums). Every sum is exact, so the order the samples
 * are added in does not change a result:
 * - integer sums of the samples, where 1/(scale*scale) and the sample
 *   conversion are powers of two and every box sum is below 2^24, so that
 *   (float)sum * unit is the C code's (float)((double)sum * unit);
 * - table[sample] in double for the other scales, each an exact float, read
 *   one lane at a time (NEON has no gather).
 */

#include <arm_neon.h>
#include <stdint.h>

#include "ssim_decimate_neon.h"

void ssim_column_sums_neon_8(const uint8_t *const *rows, int scale, int w, uint32_t *col)
{
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        /* at most 16 rows of 255 in a 16-bit lane: scale * 255 < 2^16 for
         * scale <= 257, and the integer path's scales are at most 16 */
        uint16x8_t sum = vdupq_n_u16(0);
        for (int v = 0; v < scale; v++)
            sum = vaddw_u8(sum, vld1_u8(rows[v] + x));
        vst1q_u32(col + x, vmovl_u16(vget_low_u16(sum)));
        vst1q_u32(col + x + 4, vmovl_high_u16(sum));
    }
    for (; x < w; x++) {
        uint32_t sum = 0;
        for (int v = 0; v < scale; v++)
            sum += rows[v][x];
        col[x] = sum;
    }
}

void ssim_column_sums_neon_16(const uint16_t *const *rows, int scale, int w, uint32_t *col)
{
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        uint32x4_t lo = vdupq_n_u32(0);
        uint32x4_t hi = vdupq_n_u32(0);
        for (int v = 0; v < scale; v++) {
            const uint16x8_t s = vld1q_u16(rows[v] + x);
            lo = vaddw_u16(lo, vget_low_u16(s));
            hi = vaddw_high_u16(hi, s);
        }
        vst1q_u32(col + x, lo);
        vst1q_u32(col + x + 4, hi);
    }
    for (; x < w; x++) {
        uint32_t sum = 0;
        for (int v = 0; v < scale; v++)
            sum += rows[v][x];
        col[x] = sum;
    }
}

/* The sums of 4 groups of `scale` consecutive column sums, the first group
 * at col[0]: lane i = col[i * scale] + ... + col[i * scale + scale - 1].
 * scale is 2, 4 or a multiple of 4. */
static inline uint32x4_t box_sums_4(const uint32_t *col, int scale)
{
    if (scale == 2) {
        /* vld2 splits even and odd columns: lane i gets col[2i] and col[2i + 1] */
        const uint32x4x2_t v = vld2q_u32(col);
        return vaddq_u32(v.val[0], v.val[1]);
    }
    if (scale == 4) {
        const uint32x4x4_t v = vld4q_u32(col);
        return vaddq_u32(vaddq_u32(v.val[0], v.val[1]), vaddq_u32(v.val[2], v.val[3]));
    }
    /* a multiple of 4: each group's vectors added into one, its lanes then
     * added up */
    uint32_t sums[4];
    for (int g = 0; g < 4; g++) {
        const uint32_t *p = col + g * scale;
        uint32x4_t v = vld1q_u32(p);
        for (int m = 4; m < scale; m += 4)
            v = vaddq_u32(v, vld1q_u32(p + m));
        sums[g] = vaddvq_u32(v);
    }
    return vld1q_u32(sums);
}

void ssim_box_sums_neon(const uint32_t *col, int scale, int x0, int x1, float unit, float *out)
{
    const int uc = scale / 2;  /* each box starts at x * scale - uc */
    int x = x0;
    if (scale == 2 || scale % 4 == 0) {
        for (; x + 4 <= x1; x += 4) {
            const uint32x4_t sum = box_sums_4(col + x * scale - uc, scale);
            vst1q_f32(out + x, vmulq_n_f32(vcvtq_f32_u32(sum), unit));
        }
    }
    for (; x < x1; x++) {
        const uint32_t *p = col + x * scale - uc;
        uint32_t sum = 0;
        for (int u = 0; u < scale; u++)
            sum += p[u];
        out[x] = (float)sum * unit;
    }
}

/* two lanes of table[] */
static inline float64x2_t table_pair(const double *table, unsigned a, unsigned b)
{
    return vld1q_lane_f64(table + b, vld1q_dup_f64(table + a), 1);
}

void ssim_column_table_sums_neon_8(const uint8_t *const *rows, int scale, int w,
                                   const double *table, double *col)
{
    int x = 0;
    for (; x + 2 <= w; x += 2) {
        float64x2_t sum = vdupq_n_f64(0.0);
        for (int v = 0; v < scale; v++)
            sum = vaddq_f64(sum, table_pair(table, rows[v][x], rows[v][x + 1]));
        vst1q_f64(col + x, sum);
    }
    for (; x < w; x++) {
        double sum = 0.0;
        for (int v = 0; v < scale; v++)
            sum += table[rows[v][x]];
        col[x] = sum;
    }
}

void ssim_column_table_sums_neon_16(const uint16_t *const *rows, int scale, int w,
                                    const double *table, double *col)
{
    int x = 0;
    for (; x + 2 <= w; x += 2) {
        float64x2_t sum = vdupq_n_f64(0.0);
        for (int v = 0; v < scale; v++)
            sum = vaddq_f64(sum, table_pair(table, rows[v][x], rows[v][x + 1]));
        vst1q_f64(col + x, sum);
    }
    for (; x < w; x++) {
        double sum = 0.0;
        for (int v = 0; v < scale; v++)
            sum += table[rows[v][x]];
        col[x] = sum;
    }
}

void ssim_box_table_sums_neon(const double *col, int scale, int x0, int x1, float *out)
{
    const int uc = scale / 2;
    int x = x0;
    for (; x + 4 <= x1; x += 4) {
        /* the boxes of outputs x .. x + 3 start `scale` apart */
        const double *p = col + x * scale - uc;
        float64x2_t lo = vdupq_n_f64(0.0);
        float64x2_t hi = vdupq_n_f64(0.0);
        for (int u = 0; u < scale; u++) {
            const double *q = p + u;
            lo = vaddq_f64(lo, vld1q_lane_f64(q + scale, vld1q_dup_f64(q), 1));
            hi = vaddq_f64(hi, vld1q_lane_f64(q + 3 * scale, vld1q_dup_f64(q + 2 * scale), 1));
        }
        vst1q_f32(out + x, vcvt_high_f32_f64(vcvt_f32_f64(lo), hi));
    }
    for (; x < x1; x++) {
        const double *p = col + x * scale - uc;
        double sum = 0.0;
        for (int u = 0; u < scale; u++)
            sum += p[u];
        out[x] = (float)sum;
    }
}
