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

#ifndef __VMAF_MEM_H__
#define __VMAF_MEM_H__

#include <stddef.h>

#define MAX_ALIGN 32

#define ALIGN_FLOOR(x) ((x) - (x) % MAX_ALIGN)
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))

void *aligned_malloc(size_t size, size_t alignment);

void aligned_free(void *ptr);

/* libvmaf-fast: zeroed, aligned memory from calloc, which hands large blocks
 * over as the system's zeroed pages, touched only when first written (UCRT,
 * glibc, macOS). Zeroing a CPU pass's 50 pictures of 4K 10-bit video with
 * memset took 234 ms of its start. Freed with aligned_calloc_free only. */
void *aligned_calloc(size_t size, size_t alignment);

void aligned_calloc_free(void *ptr);

#endif /* __VMAF_MEM_H__ */
