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

#ifndef ARM64_SSIM_DECIMATE_NEON_H_
#define ARM64_SSIM_DECIMATE_NEON_H_

#include <stdint.h>

/* The NEON counterparts of x86/ssim_decimate_avx2.h, with the same
 * arguments and results. */

/* col[x] = rows[0][x] + ... + rows[scale - 1][x] for x < w */
void ssim_column_sums_neon_8(const uint8_t *const *rows, int scale, int w, uint32_t *col);
void ssim_column_sums_neon_16(const uint16_t *const *rows, int scale, int w, uint32_t *col);

/* out[x] = (float)(col[x * scale - scale / 2] + ... (scale of them)) * unit
 * for x0 <= x < x1, every box inside col. scale is a power of two; every
 * box sum < 2^24 and unit a power of two, so this is exact. */
void ssim_box_sums_neon(const uint32_t *col, int scale, int x0, int x1, float unit, float *out);

/* col[x] = table[rows[0][x]] + ... + table[rows[scale - 1][x]] in double */
void ssim_column_table_sums_neon_8(const uint8_t *const *rows, int scale, int w,
                                   const double *table, double *col);
void ssim_column_table_sums_neon_16(const uint16_t *const *rows, int scale, int w,
                                    const double *table, double *col);

/* out[x] = (float)(col[x * scale - scale / 2] + ... (scale of them)) for
 * x0 <= x < x1, every box inside col */
void ssim_box_table_sums_neon(const double *col, int scale, int x0, int x1, float *out);

#endif /* ARM64_SSIM_DECIMATE_NEON_H_ */
