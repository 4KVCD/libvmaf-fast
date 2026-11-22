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

#ifndef ARM64_PSNR_NEON_H_
#define ARM64_PSNR_NEON_H_

#include <stdint.h>

/* A row's sum of squared errors, as x86/psnr_avx2.c's: the 8-bit one in 32
 * bits (wrapping as the C code's does), the 16-bit one in 64. */
uint32_t psnr_sse_line_8_neon(const uint8_t *ref, const uint8_t *dis, unsigned w);

uint64_t psnr_sse_line_16_neon(const uint16_t *ref, const uint16_t *dis, unsigned w);

#endif /* ARM64_PSNR_NEON_H_ */
