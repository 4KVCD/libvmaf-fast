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
 * libvmaf-fast's own AVX2 code for ssim.c's decimate_picture: each output
 * sample is the sum over a scale x scale box of the picture, which is done
 * here in two steps. First the column sums: for every column, the sum of
 * the box's `scale` rows. Then the box sums: for every output, the sum of
 * `scale` column sums. Both are exact (see ssim.c), so the order the
 * samples are added in does not change a result.
 *
 * Two kinds of sums, as in ssim.c:
 * - Integer: the samples themselves, when 1/(scale*scale) and the sample
 *   conversion are powers of two. The caller makes sure scale * scale *
 *   max sample < 2^24, so a box sum converts to float exactly and times a
 *   power of two is (float)(sum * unit) in double.
 * - Table: table[sample] in double, each an exact float, for the other
 *   scales. Each entry is (float)sample * product, a single rounding of
 *   the exact product (see ssim_workspace_alloc), so the same float is
 *   made here with one float multiply, 8 samples at a time, instead of
 *   gathering the table 4 at a time (the gathers were most of a
 *   letterboxed film's SSIM time: 4K cropped to 3840x1608 is scale 6).
 *   Then widened to double and added, as before; no FMA, which would
 *   round the product and the sum as one.
 */

#include <immintrin.h>
#include <stdint.h>

#include "ssim_decimate_avx2.h"

void ssim_column_sums_avx2_8(const uint8_t *const *rows, int scale, int w, uint32_t *col)
{
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        __m256i sum = _mm256_setzero_si256();
        for (int v = 0; v < scale; v++) {
            const __m128i s = _mm_loadl_epi64((const __m128i *)(rows[v] + x));
            sum = _mm256_add_epi32(sum, _mm256_cvtepu8_epi32(s));
        }
        _mm256_storeu_si256((__m256i *)(col + x), sum);
    }
    for (; x < w; x++) {
        uint32_t sum = 0;
        for (int v = 0; v < scale; v++)
            sum += rows[v][x];
        col[x] = sum;
    }
}

void ssim_column_sums_avx2_16(const uint16_t *const *rows, int scale, int w, uint32_t *col)
{
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        __m256i sum = _mm256_setzero_si256();
        for (int v = 0; v < scale; v++) {
            const __m128i s = _mm_loadu_si128((const __m128i *)(rows[v] + x));
            sum = _mm256_add_epi32(sum, _mm256_cvtepu16_epi32(s));
        }
        _mm256_storeu_si256((__m256i *)(col + x), sum);
    }
    for (; x < w; x++) {
        uint32_t sum = 0;
        for (int v = 0; v < scale; v++)
            sum += rows[v][x];
        col[x] = sum;
    }
}

/* The 8 sums of 8 consecutive groups of 8 values, v[0] to v[7] (one group
 * each), in lanes 0 to 7. hadd adds neighbouring pairs within each 128-bit
 * half, so after three of them each half holds a quarter of every group;
 * the halves are then swapped into place and added. */
static inline __m256i sum_groups_of_8(const __m256i v[8])
{
    /* [v0 01, v0 23, v1 01, v1 23 | v0 45, v0 67, v1 45, v1 67], ... */
    const __m256i h01 = _mm256_hadd_epi32(v[0], v[1]);
    const __m256i h23 = _mm256_hadd_epi32(v[2], v[3]);
    const __m256i h45 = _mm256_hadd_epi32(v[4], v[5]);
    const __m256i h67 = _mm256_hadd_epi32(v[6], v[7]);
    /* [v0 0-3, v1 0-3, v2 0-3, v3 0-3 | v0 4-7, v1 4-7, v2 4-7, v3 4-7] */
    const __m256i q0 = _mm256_hadd_epi32(h01, h23);
    const __m256i q1 = _mm256_hadd_epi32(h45, h67);
    /* first halves of all 8 groups, plus their second halves */
    const __m256i first = _mm256_permute2x128_si256(q0, q1, 0x20);
    const __m256i second = _mm256_permute2x128_si256(q0, q1, 0x31);
    return _mm256_add_epi32(first, second);
}

/* The sums of 8 groups of `scale` consecutive column sums, the first group
 * at col[0]: lane i = col[i * scale] + ... + col[i * scale + scale - 1].
 * scale is 2, 4 or a multiple of 8. */
static inline __m256i box_sums_8(const uint32_t *col, int scale)
{
    if (scale == 2) {
        /* [g0, g1, g4, g5 | g2, g3, g6, g7] in 64-bit pairs: put them in order */
        const __m256i a = _mm256_loadu_si256((const __m256i *)col);
        const __m256i b = _mm256_loadu_si256((const __m256i *)(col + 8));
        return _mm256_permute4x64_epi64(_mm256_hadd_epi32(a, b), 0xD8);
    }
    if (scale == 4) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)col);
        const __m256i b = _mm256_loadu_si256((const __m256i *)(col + 8));
        const __m256i c = _mm256_loadu_si256((const __m256i *)(col + 16));
        const __m256i d = _mm256_loadu_si256((const __m256i *)(col + 24));
        /* [g0, g2, g4, g6 | g1, g3, g5, g7] */
        const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(a, b), _mm256_hadd_epi32(c, d));
        return _mm256_permutevar8x32_epi32(h, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    }
    /* a multiple of 8: each group's vectors added into one, then summed */
    __m256i v[8];
    for (int g = 0; g < 8; g++) {
        const uint32_t *p = col + g * scale;
        v[g] = _mm256_loadu_si256((const __m256i *)p);
        for (int m = 8; m < scale; m += 8)
            v[g] = _mm256_add_epi32(v[g], _mm256_loadu_si256((const __m256i *)(p + m)));
    }
    return sum_groups_of_8(v);
}

void ssim_box_sums_avx2(const uint32_t *col, int scale, int x0, int x1, float unit, float *out)
{
    const int uc = scale / 2;  /* each box starts at x * scale - uc */
    const __m256 vunit = _mm256_set1_ps(unit);
    int x = x0;
    for (; x + 8 <= x1; x += 8) {
        const __m256i sum = box_sums_8(col + x * scale - uc, scale);
        _mm256_storeu_ps(out + x, _mm256_mul_ps(_mm256_cvtepi32_ps(sum), vunit));
    }
    for (; x < x1; x++) {
        const uint32_t *p = col + x * scale - uc;
        uint32_t sum = 0;
        for (int u = 0; u < scale; u++)
            sum += p[u];
        out[x] = (float)sum * unit;
    }
}

/* Adds table[s] of 8 samples, the float products (float)s * product, to the
 * column sums: lanes 0-3 to lo, 4-7 to hi, each widened to double. */
static inline void add_products_8(__m256i samples, __m256 product, __m256d *lo, __m256d *hi)
{
    const __m256 p = _mm256_mul_ps(_mm256_cvtepi32_ps(samples), product);
    *lo = _mm256_add_pd(*lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p)));
    *hi = _mm256_add_pd(*hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p, 1)));
}

void ssim_column_table_sums_avx2_8(const uint8_t *const *rows, int scale, int w,
                                   float product, double *col)
{
    const __m256 vproduct = _mm256_set1_ps(product);
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        __m256d lo = _mm256_setzero_pd(), hi = _mm256_setzero_pd();
        for (int v = 0; v < scale; v++) {
            const __m128i s = _mm_loadl_epi64((const __m128i *)(rows[v] + x));
            add_products_8(_mm256_cvtepu8_epi32(s), vproduct, &lo, &hi);
        }
        _mm256_storeu_pd(col + x, lo);
        _mm256_storeu_pd(col + x + 4, hi);
    }
    for (; x < w; x++) {
        double sum = 0.0;
        for (int v = 0; v < scale; v++) {
            const float p = (float)rows[v][x] * product;  /* its own statement: never fused */
            sum += p;
        }
        col[x] = sum;
    }
}

void ssim_column_table_sums_avx2_16(const uint16_t *const *rows, int scale, int w,
                                    float product, double *col)
{
    const __m256 vproduct = _mm256_set1_ps(product);
    int x = 0;
    for (; x + 8 <= w; x += 8) {
        __m256d lo = _mm256_setzero_pd(), hi = _mm256_setzero_pd();
        for (int v = 0; v < scale; v++) {
            const __m128i s = _mm_loadu_si128((const __m128i *)(rows[v] + x));
            add_products_8(_mm256_cvtepu16_epi32(s), vproduct, &lo, &hi);
        }
        _mm256_storeu_pd(col + x, lo);
        _mm256_storeu_pd(col + x + 4, hi);
    }
    for (; x < w; x++) {
        double sum = 0.0;
        for (int v = 0; v < scale; v++) {
            const float p = (float)rows[v][x] * product;
            sum += p;
        }
        col[x] = sum;
    }
}

void ssim_box_table_sums_avx2(const double *col, int scale, int x0, int x1, float *out)
{
    const int uc = scale / 2;
    /* the first column of the boxes of outputs x .. x + 3, less x * scale */
    const __m128i starts = _mm_setr_epi32(-uc, scale - uc, 2 * scale - uc, 3 * scale - uc);
    int x = x0;
    for (; x + 4 <= x1; x += 4) {
        const double *base = col + x * scale;
        __m256d sum = _mm256_setzero_pd();
        for (int u = 0; u < scale; u++)
            sum = _mm256_add_pd(sum, _mm256_i32gather_pd(base + u, starts, 8));
        _mm_storeu_ps(out + x, _mm256_cvtpd_ps(sum));
    }
    for (; x < x1; x++) {
        const double *p = col + x * scale - uc;
        double sum = 0.0;
        for (int u = 0; u < scale; u++)
            sum += p[u];
        out[x] = (float)sum;
    }
}
