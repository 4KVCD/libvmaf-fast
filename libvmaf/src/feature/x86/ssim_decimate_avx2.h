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

#ifndef LIBVMAF_FEATURE_X86_SSIM_DECIMATE_AVX2_H
#define LIBVMAF_FEATURE_X86_SSIM_DECIMATE_AVX2_H

#include <stdint.h>

/* col[x] = rows[0][x] + ... + rows[scale - 1][x] for x < w */
void ssim_column_sums_avx2_8(const uint8_t *const *rows, int scale, int w, uint32_t *col);
void ssim_column_sums_avx2_16(const uint16_t *const *rows, int scale, int w, uint32_t *col);

/* out[x] = (float)(col[x * scale - scale / 2] + ... (scale of them)) * unit
 * for x0 <= x < x1, every box inside col. scale is 2, 4 or a multiple of 8;
 * every box sum < 2^24 and unit a power of two, so this is exact. */
void ssim_box_sums_avx2(const uint32_t *col, int scale, int x0, int x1, float unit, float *out);

/* col[x] = table[rows[0][x]] + ... + table[rows[scale - 1][x]] in double,
 * ssim.c's table of each sample's float product: table[s] is
 * (float)s * product, one float multiply, so it is made here, not read */
void ssim_column_table_sums_avx2_8(const uint8_t *const *rows, int scale, int w,
                                   float product, double *col);
void ssim_column_table_sums_avx2_16(const uint16_t *const *rows, int scale, int w,
                                    float product, double *col);

/* out[x] = (float)(col[x * scale - scale / 2] + ... (scale of them)) for
 * x0 <= x < x1, every box inside col */
void ssim_box_table_sums_avx2(const double *col, int scale, int x0, int x1, float *out);

#endif /* LIBVMAF_FEATURE_X86_SSIM_DECIMATE_AVX2_H */
