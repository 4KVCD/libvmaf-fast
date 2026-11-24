/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#ifndef FEATURE_MS_SSIM_H_
#define FEATURE_MS_SSIM_H_

#include <libvmaf/picture.h>

/* MS-SSIM of two w x h float pictures, rows ref_stride bytes apart (both the
 * same); 0 on success. Allocates its buffers each call. */
int compute_ms_ssim(const float *ref, const float *cmp, int w, int h,
                    int ref_stride, int cmp_stride, double *score,
                    double* l_scores, double* c_scores, double* s_scores);

/* libvmaf-fast: compute_ms_ssim's buffers for pictures up to w x h, kept by
 * the caller from frame to frame (compute_ms_ssim_ws); the same scores. */
typedef struct MsSsimWorkspace MsSsimWorkspace;
MsSsimWorkspace *ms_ssim_workspace_alloc(int w, int h);
void ms_ssim_workspace_free(MsSsimWorkspace *ws);
int compute_ms_ssim_ws(const float *ref, const float *cmp, int w, int h,
                       int ref_stride, int cmp_stride, MsSsimWorkspace *ws,
                       double *score, double* l_scores, double* c_scores,
                       double* s_scores);

/* libvmaf-fast: compute_ms_ssim_ws of plane `plane` of two pictures as
 * picture_copy converts them (offset 0), read row by row: the same scores
 * without the planes as float pictures. */
int compute_ms_ssim_picture(const VmafPicture *ref, const VmafPicture *dis, int plane,
                            MsSsimWorkspace *ws, double *score, double *l_scores,
                            double *c_scores, double *s_scores);

/* libvmaf-fast: whether a w x h picture is large enough for MS-SSIM's 5
 * scales (the 11 x 11 window inside the smallest): 176 x 176 or more. */
int ms_ssim_size_fits(int w, int h);

#endif /* FEATURE_MS_SSIM_H_ */
