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

#ifndef ARM64_CONVOLVE_NEON_H_
#define ARM64_CONVOLVE_NEON_H_

/* iqa/convolve.c's 1D separable filter for a normalized kernel, as
 * x86/convolve_avx2.c's iqa_convolve_avx2 (the same arguments): result
 * NULL convolves img in place; workspace (w * h floats) NULL allocates
 * one. */
void iqa_convolve_neon(float *img, int w, int h, const float *kernel_h, const float *kernel_v,
                       int kw, int kh, float *workspace, float *result, int *rw, int *rh);

#endif /* ARM64_CONVOLVE_NEON_H_ */
