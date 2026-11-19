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

#ifndef LIBVMAF_FEATURE_X86_XPSNR_ANY_AVX2_H
#define LIBVMAF_FEATURE_X86_XPSNR_ANY_AVX2_H

#include <stddef.h>
#include <stdint.h>

/* One strip of XPSNR's blocks: the picture's rows from a multiple of the
 * block size, `height` of them (the block size, or fewer at the bottom). */
typedef struct XpsnrStrip {
    /* the strip's first rows of the original, the reconstruction and the
     * two pictures before (m1, m2: a row of zeros with stride 0 before the
     * first) */
    const void *org, *rec, *m1, *m2;
    ptrdiff_t so, sr, s1, s2;  /* strides in samples */
    int w, b, b_val;           /* picture width, block size, FFmpeg's b_val */
    int height;
    int y_act, h_act;          /* the strip's rows of spatial activity, as FN(block) has them */
    int second_order;          /* temporal: m0 - 2 m1 + m2 (else m0 - m1) */
} XpsnrStrip;

/*
 * The integer sums of xpsnr_template.c's FN(block) for every block of the
 * strip, block bx's in sse[bx], sa[bx] and ta[bx] (ta before XPSNR_GAMMA):
 * the same integers, with the strip read row by row so that memory is read
 * in order. Blocks too small for activity (FN(block) returns their squared
 * errors only) get sums of activity the caller does not use. Samples of up
 * to 12 bits.
 */
void xpsnr_strip_avx2_8(const XpsnrStrip *st, uint64_t *sse, uint64_t *sa, uint64_t *ta);
void xpsnr_strip_avx2_16(const XpsnrStrip *st, uint64_t *sse, uint64_t *sa, uint64_t *ta);

#endif /* LIBVMAF_FEATURE_X86_XPSNR_ANY_AVX2_H */
