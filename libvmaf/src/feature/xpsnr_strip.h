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

#ifndef LIBVMAF_FEATURE_XPSNR_STRIP_H
#define LIBVMAF_FEATURE_XPSNR_STRIP_H

#include <stddef.h>
#include <stdint.h>

/* One strip of XPSNR's blocks for the SIMD kernels (x86/xpsnr_any_avx2.c,
 * arm64/xpsnr_any_neon.c): the picture's rows from a multiple of the block
 * size, `height` of them (the block size, or fewer at the bottom). */
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

#endif /* LIBVMAF_FEATURE_XPSNR_STRIP_H */
