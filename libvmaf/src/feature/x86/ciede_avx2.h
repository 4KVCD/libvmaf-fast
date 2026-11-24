/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *  SPDX-License-Identifier: BSD-2-Clause-Patent
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

#ifndef X86_AVX2_CIEDE_H_
#define X86_AVX2_CIEDE_H_

/* dE00 of n pixels (the whole fours of them; the caller does the rest):
 * lab[0..2] the first colours' L, a and b, lab[3..5] the second's. k25_7 is
 * ciede.c's powf(25., 7). */
void ciede2000_avx2(const float *const lab[6], unsigned n, float k25_7, float *de00);

#endif /* X86_AVX2_CIEDE_H_ */
