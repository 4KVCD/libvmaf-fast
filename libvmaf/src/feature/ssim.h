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

#include "libvmaf/picture.h"

int compute_ssim(const float *ref, const float *cmp, int w, int h,
                 int ref_stride, int cmp_stride, double *score,
                 double *l_score, double *c_score, double *s_score,
                 int scale_override);

/* The scale compute_ssim() decimates a w x h picture by. */
int ssim_scale(int w, int h, int scale_override);

/*
 * The buffers and tables compute_ssim_decimated() needs for w x h pictures
 * of bpc bits at a scale, kept from frame to frame by its caller (one
 * thread at a time). NULL for a scale of 1, a bit depth it does not take,
 * or no memory. floats_from: NULL, or a workspace whose decimated float
 * buffers (the bulk of the memory) this one uses too, for planes scored
 * one after the other; it must hold at least as many decimated samples
 * (else NULL) and stay allocated while this one is used (freeing this one
 * leaves the borrowed buffers alone).
 */
typedef struct SsimWorkspace SsimWorkspace;
SsimWorkspace *ssim_workspace_alloc(int w, int h, int scale, unsigned bpc,
                                    SsimWorkspace *floats_from);
void ssim_workspace_free(SsimWorkspace *ws);

/* The samples of a w x h picture decimated by scale. */
size_t ssim_decimated_samples(int w, int h, int scale);

/*
 * compute_ssim() of one plane of two pictures (0: Y, 1: Cb, 2: Cr) as
 * float_ssim converts it (picture_copy), for a scale above 1: the same
 * score, to the last bit, without the full-size float images. ws: a
 * workspace for this plane's size, the bit depth and scale, or NULL (or
 * another's) to allocate one for the call. -EINVAL for a scale of 1, a bit
 * depth picture_copy does not take, or a plane the pictures do not have.
 */
int compute_ssim_decimated(VmafPicture *ref, VmafPicture *cmp, int plane, int scale,
                           SsimWorkspace *ws, double *score, double *l_score,
                           double *c_score, double *s_score);
