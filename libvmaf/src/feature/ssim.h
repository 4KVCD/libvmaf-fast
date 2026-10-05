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
 * compute_ssim() of two pictures' luma planes as float_ssim converts them
 * (picture_copy), for a scale above 1: the same score, to the last bit,
 * without the full-size float images. -EINVAL for a scale of 1 or a bit
 * depth picture_copy does not take.
 */
int compute_ssim_decimated(VmafPicture *ref, VmafPicture *cmp, int scale,
                           double *score, double *l_score, double *c_score,
                           double *s_score);
