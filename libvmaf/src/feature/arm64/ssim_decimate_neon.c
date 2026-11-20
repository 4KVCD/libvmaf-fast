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
 * libvmaf-fast's NEON code for ssim.c's decimate_picture, the Arm
 * counterpart of x86/ssim_decimate_avx2.c: each output sample is the sum
 * over a scale x scale box of the picture, made in two steps, first the
 * column sums (for every column, the sum of the box's `scale` rows), then
 * the box sums (for every output, the sum of `scale` column sums). Every
 * sum is exact (see ssim.c), so the order the samples are added in does
 * not change a result.
 *
 * Two kinds of sums, as in ssim.c:
 * - Integer: the samples themselves, when 1/(scale*scale) and the sample
 *   conversion are powers of two. The caller makes sure scale * scale *
 *   max sample < 2^24, so a box sum converts to float exactly and times a
 *   power of two is (float)(sum * unit) in double.
 * - Table: table[sample] in double, each an exact float, for the other
 *   scales. NEON has no gather, so two lookups fill one vector.
 */

#include <arm_neon.h>
#include <stdint.h>

#include "ssim_decimate_neon.h"

/* Sums of 8-bit samples fit 16-bit lanes for up to 257 rows, and the
 * caller's limit (scale * scale * 255 < 2^24) means scale <= 256. */
void ssim_column_sums_neon_8(const uint8_t *const *rows, int scale, int w, uint32_t *col)
{
    int x = 0;
    for (; x + 16 <= w; x += 16) {
        uint16x8_t sum0 = vdupq_n_u16(0);
        uint16x8_t sum1 = vdupq_n_u16(0);
        for (int v = 0; v < scale; v++) {
            const uint8x16_t s = vld1q_u8(rows[v] + x);
            sum0 = vaddw_u8(sum0, vget_low_u8(s));
            sum1 = vaddw_high_u8(sum1, s);
        }
        vst1q_u32(col + x, vmovl_u16(vget_low_u16(sum0)));
        vst1q_u32(col + x + 4, vmovl_high_u16(sum0));
        vst1q_u32(col + x + 8, vmovl_u16(vget_low_u16(sum1)));
        vst1q_u32(col + x + 12, vmovl_high_u16(sum1));
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
        uint32x4_t sum0 = vdupq_n_u32(0);
        uint32x4_t sum1 = vdupq_n_u32(0);
        for (int v = 0; v < scale; v++) {
            const uint16x8_t s = vld1q_u16(rows[v] + x);
            sum0 = vaddw_u16(sum0, vget_low_u16(s));
            sum1 = vaddw_high_u16(sum1, s);
        }
        vst1q_u32(col + x, sum0);
        vst1q_u32(col + x + 4, sum1);
    }
    for (; x < w; x++) {
        uint32_t sum = 0;
        for (int v = 0; v < scale; v++)
            sum += rows[v][x];
        col[x] = sum;
    }
}

/* The sum of a box's columns, 4 lanes apart: lane i = col[i] + col[i + 4]
 * + ... (scale / 4 vectors), scale a multiple of 4. */
static inline uint32x4_t box_columns(const uint32_t *col, int scale)
{
    uint32x4_t sum = vld1q_u32(col);
    for (int m = 4; m < scale; m += 4) {
        const uint32x4_t next = vld1q_u32(col + m);
        sum = vaddq_u32(sum, next);
    }
    return sum;
}

/* The sums of 4 neighbouring boxes of `scale` column sums, the first box at
 * col[0]. vpaddq adds neighbouring lanes of its two arguments, in order. */
static inline uint32x4_t box_sums4(const uint32_t *col, int scale)
{
    if (scale == 2) {
        const uint32x4_t a = vld1q_u32(col);
        const uint32x4_t b = vld1q_u32(col + 4);
        return vpaddq_u32(a, b);
    }
    /* each box's columns in one vector, then each vector's 4 lanes */
    const uint32x4_t box0 = box_columns(col, scale);
    const uint32x4_t box1 = box_columns(col + scale, scale);
    const uint32x4_t box2 = box_columns(col + 2 * scale, scale);
    const uint32x4_t box3 = box_columns(col + 3 * scale, scale);
    return vpaddq_u32(vpaddq_u32(box0, box1), vpaddq_u32(box2, box3));
}

void ssim_box_sums_neon(const uint32_t *col, int scale, int x0, int x1, float unit, float *out)
{
    const int uc = scale / 2;  /* each box starts at x * scale - uc */
    int x = x0;
    for (; x + 4 <= x1; x += 4) {
        const uint32x4_t sum = box_sums4(col + x * scale - uc, scale);
        vst1q_f32(out + x, vmulq_n_f32(vcvtq_f32_u32(sum), unit));
    }
    for (; x < x1; x++) {
        const uint32_t *p = col + x * scale - uc;
        uint32_t sum = 0;
        for (int u = 0; u < scale; u++)
            sum += p[u];
        out[x] = (float)sum * unit;
    }
}

/* table[a] and table[b] in one vector */
static inline float64x2_t gather2(const double *table, unsigned a, unsigned b)
{
    const float64x1_t lo = vld1_f64(table + a);
    const float64x1_t hi = vld1_f64(table + b);
    return vcombine_f64(lo, hi);
}

void ssim_column_table_sums_neon_8(const uint8_t *const *rows, int scale, int w,
                                   const double *table, double *col)
{
    int x = 0;
    for (; x + 4 <= w; x += 4) {
        float64x2_t sum0 = vdupq_n_f64(0.0);
        float64x2_t sum1 = vdupq_n_f64(0.0);
        for (int v = 0; v < scale; v++) {
            const uint8_t *s = rows[v] + x;
            sum0 = vaddq_f64(sum0, gather2(table, s[0], s[1]));
            sum1 = vaddq_f64(sum1, gather2(table, s[2], s[3]));
        }
        vst1q_f64(col + x, sum0);
        vst1q_f64(col + x + 2, sum1);
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
    for (; x + 4 <= w; x += 4) {
        float64x2_t sum0 = vdupq_n_f64(0.0);
        float64x2_t sum1 = vdupq_n_f64(0.0);
        for (int v = 0; v < scale; v++) {
            const uint16_t *s = rows[v] + x;
            sum0 = vaddq_f64(sum0, gather2(table, s[0], s[1]));
            sum1 = vaddq_f64(sum1, gather2(table, s[2], s[3]));
        }
        vst1q_f64(col + x, sum0);
        vst1q_f64(col + x + 2, sum1);
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
        /* the boxes of outputs x .. x + 3 start scale columns apart */
        const double *p = col + x * scale - uc;
        float64x2_t sum0 = vdupq_n_f64(0.0);
        float64x2_t sum1 = vdupq_n_f64(0.0);
        for (int u = 0; u < scale; u++) {
            sum0 = vaddq_f64(sum0, gather2(p + u, 0, scale));
            sum1 = vaddq_f64(sum1, gather2(p + u, 2 * scale, 3 * scale));
        }
        vst1q_f32(out + x, vcvt_high_f32_f64(vcvt_f32_f64(sum0), sum1));
    }
    for (; x < x1; x++) {
        const double *p = col + x * scale - uc;
        double sum = 0.0;
        for (int u = 0; u < scale; u++)
            sum += p[u];
        out[x] = (float)sum;
    }
}
