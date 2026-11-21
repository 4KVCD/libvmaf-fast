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

#ifndef LIBVMAF_FEATURE_ARM64_XPSNR_ANY_NEON_H
#define LIBVMAF_FEATURE_ARM64_XPSNR_ANY_NEON_H

#include <stdint.h>

#include "../xpsnr_strip.h"

/* x86/xpsnr_any_avx2.h's functions (see there), for NEON. */
void xpsnr_strip_neon_8(const XpsnrStrip *st, uint64_t *sse, uint64_t *sa, uint64_t *ta);
void xpsnr_strip_neon_16(const XpsnrStrip *st, uint64_t *sse, uint64_t *sa, uint64_t *ta);

#endif /* LIBVMAF_FEATURE_ARM64_XPSNR_ANY_NEON_H */
