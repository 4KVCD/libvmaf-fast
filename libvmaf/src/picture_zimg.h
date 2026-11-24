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

#ifndef __VMAF_SRC_PICTURE_ZIMG_H__
#define __VMAF_SRC_PICTURE_ZIMG_H__

#include <zimg.h>

#include "libvmaf/picture.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * zimg's formats and graph parameters for converting `src` (its format,
 * size and colour) to `target`: the one setup both the CPU conversion
 * (vmaf_picture_convert_context_init) and the GPU plan builder
 * (vmaf_picture_convert_gpu_plan) use, so the two cannot drift apart.
 * `caller` prefixes the error messages logged.
 */
int vmaf_picture_zimg_args(const VmafPicture *src,
                           const VmafPictureConvertTarget *target,
                           const char *caller, zimg_image_format *src_fmt,
                           zimg_image_format *dst_fmt,
                           zimg_graph_builder_params *params);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_SRC_PICTURE_ZIMG_H__ */
