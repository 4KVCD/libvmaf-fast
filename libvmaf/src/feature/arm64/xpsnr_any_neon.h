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

#ifndef ARM64_XPSNR_ANY_NEON_H_
#define ARM64_XPSNR_ANY_NEON_H_

#include <stdint.h>

#include "../xpsnr_strip.h"

/* The NEON counterparts of x86/xpsnr_any_avx2.h, with the same arguments
 * and results: the integer sums of xpsnr_template.c's FN(block) for every
 * block of the strip, block bx's in sse[bx], sa[bx] and ta[bx] (ta before
 * XPSNR_GAMMA). Samples of up to 12 bits. */
void xpsnr_strip_neon_8(const XpsnrStrip *st, uint64_t *sse, uint64_t *sa, uint64_t *ta);
void xpsnr_strip_neon_16(const XpsnrStrip *st, uint64_t *sse, uint64_t *sa, uint64_t *ta);

#endif /* ARM64_XPSNR_ANY_NEON_H_ */
