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

/*
 * libvmaf-fast's NEON copy of x86/xpsnr_any_avx2.c (see there for the
 * method): XPSNR's sums of each block, its squared errors and its four
 * activity sums, the strip read row by row. Every term is an exact integer
 * in 32-bit lanes, so the sums are the C loops' integers. The vectors are
 * half as wide as AVX2's: 4 results a step for the spatial and downsampled
 * temporal sums, 8 samples a step for the temporal one.
 */

#include <arm_neon.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "psnr_neon.h"
#include "xpsnr_any_neon.h"

/* lanes 0 .. n - 1 all ones, the rest zero */
static inline uint32x4_t lanes_below(int n)
{
    static const uint32_t index[4] = { 0, 1, 2, 3 };
    return vcltq_u32(vld1q_u32(index), vdupq_n_u32((uint32_t) n));
}

/* the same for 8 16-bit lanes */
static inline uint16x8_t lanes_below_16(int n)
{
    static const uint16_t index[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    return vcltq_u16(vld1q_u16(index), vdupq_n_u16((uint16_t) n));
}

/* 4 samples into 32-bit lanes; 8 samples into 16-bit lanes. The 8-bit
 * LOAD4 copies exactly 4 bytes, so that it reads nothing past them. */
static inline int32x4_t load4_8(const uint8_t *p)
{
    uint32_t four;
    memcpy(&four, p, sizeof(four));
    const uint8x8_t bytes = vreinterpret_u8_u32(vdup_n_u32(four));
    return vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(bytes))));
}

static inline int32x4_t load4_16(const uint16_t *p)
{
    return vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p)));
}

#define T uint8_t
#define FN(name) name##_8
#define LOAD4_S32(p) load4_8(p)
#define LOAD8_U16(p) vmovl_u8(vld1_u8(p))
#define SSE_ROW(org, rec, w) psnr_sse_line_8_neon(org, rec, w)
#include "xpsnr_any_neon_kernels.h"
#undef T
#undef FN
#undef LOAD4_S32
#undef LOAD8_U16
#undef SSE_ROW

#define T uint16_t
#define FN(name) name##_16
#define LOAD4_S32(p) load4_16(p)
#define LOAD8_U16(p) vld1q_u16(p)
/* FFmpeg sums a row's squared errors in 32 bits */
#define SSE_ROW(org, rec, w) ((uint32_t) psnr_sse_line_16_neon(org, rec, w))
#include "xpsnr_any_neon_kernels.h"
#undef T
#undef FN
#undef LOAD4_S32
#undef LOAD8_U16
#undef SSE_ROW
