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
 * libvmaf-fast's own AVX2 code for XPSNR's sums of each block: squared
 * errors (VMAFx's rows, x86/psnr_avx2.c) and the four activity sums (3x3
 * highpass, downsampled highpass, temporal, downsampled temporal), for any
 * block width.
 *
 * The blocks of a strip are read row by row, every block's row in turn,
 * rather than block by block: a block's row is a few cache lines, so going
 * down a block jumps a whole picture row each time, which the hardware
 * prefetcher does not follow. Row by row, memory is read in order.
 *
 * Each sum is the same integer as the C loops of xpsnr_template.c: every
 * term is computed exactly in 32-bit lanes (for samples of up to 12 bits,
 * all xpsnr.c takes: |f| < 96 * 4095 for the spatial sums), and a sum of
 * integers does not depend on the order it is added in. A block row's
 * lanes hold at most b / 8 terms, far below 2^32, and are added into the
 * 64-bit total after every row; the squared errors of a block row are a
 * 32-bit sum, as FFmpeg's.
 *
 * Reads stay inside the picture: a row's last vector loads straight from
 * the picture when the row goes on far enough (`readable` samples from the
 * block's left edge) and drops the lanes past the block; only at the
 * picture's right edge are the samples copied into a zero-filled buffer.
 */

#include <immintrin.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "psnr_avx2.h"
#include "xpsnr_any_avx2.h"

/* lanes 0 .. n - 1 all ones, the rest zero */
static inline __m256i lanes_below(int n)
{
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}

/* the 8 unsigned 32-bit lanes added up */
static inline uint64_t sum_lanes(__m256i v)
{
    const __m256i wide = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(v)),
                                          _mm256_cvtepu32_epi64(_mm256_extracti128_si256(v, 1)));
    const __m128i half = _mm_add_epi64(_mm256_castsi256_si128(wide), _mm256_extracti128_si256(wide, 1));
    return (uint64_t)_mm_cvtsi128_si64(half) + (uint64_t)_mm_extract_epi64(half, 1);
}

#define T uint8_t
#define FN(name) name##_8
#define LOAD8_EPI32(p) _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(p)))
#define LOAD16_EPI16(p) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(p)))
#define SSE_ROW(org, rec, w) psnr_sse_line_8_avx2(org, rec, w)
#include "xpsnr_any_avx2_kernels.h"
#undef T
#undef FN
#undef LOAD8_EPI32
#undef LOAD16_EPI16
#undef SSE_ROW

#define T uint16_t
#define FN(name) name##_16
#define LOAD8_EPI32(p) _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p)))
#define LOAD16_EPI16(p) _mm256_loadu_si256((const __m256i *)(p))
/* FFmpeg sums a row's squared errors in 32 bits */
#define SSE_ROW(org, rec, w) ((uint32_t) psnr_sse_line_16_avx2(org, rec, w))
#include "xpsnr_any_avx2_kernels.h"
#undef T
#undef FN
#undef LOAD8_EPI32
#undef LOAD16_EPI16
#undef SSE_ROW
