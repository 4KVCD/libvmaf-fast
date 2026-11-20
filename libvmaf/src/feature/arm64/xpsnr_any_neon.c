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
 * libvmaf-fast's NEON code for XPSNR's sums of each block, the Arm
 * counterpart of x86/xpsnr_any_avx2.c: squared errors (the PSNR rows of
 * arm64/psnr_neon.c) and the four activity sums (3x3 highpass, downsampled
 * highpass, temporal, downsampled temporal), for any block width, a strip
 * of blocks at a time, row by row, so that memory is read in order.
 *
 * Each sum is the same integer as the C loops of xpsnr_template.c. Samples
 * have at most 12 bits, so most terms are exact in 16-bit lanes: each is
 * the difference |a - b| of two sums of samples that fit 16 bits (vabd).
 * Only the downsampled highpass needs 32-bit lanes. Lanes are added into
 * 32-bit totals by pairs (vpadal), far below 2^32 for a block row, and the
 * row's total into the 64-bit block sum; the order integers are added in
 * does not change their sum.
 *
 * As on x86, a row's last vector loads straight from the picture when the
 * row goes on far enough and drops the lanes past the block; only at the
 * picture's right edge are the samples copied into a zero-filled buffer.
 */

#include <arm_neon.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "psnr_neon.h"
#include "xpsnr_any_neon.h"

/* the 16-bit lanes below n all ones, the rest zero (none for n <= 0) */
static inline uint16x8_t lanes_below_16(int n)
{
    static const int16_t index[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const int16x8_t lane = vld1q_s16(index);
    return vcltq_s16(lane, vdupq_n_s16((int16_t) n));
}

/* the 32-bit lanes below n all ones, the rest zero (none for n <= 0) */
static inline uint32x4_t lanes_below_32(int n)
{
    static const int32_t index[4] = { 0, 1, 2, 3 };
    const int32x4_t lane = vld1q_s32(index);
    return vcltq_s32(lane, vdupq_n_s32(n));
}

/* 16 columns of 16-bit sums, split into the even columns and the odd ones */
typedef struct EvenOdd {
    uint16x8_t even, odd;
} EvenOdd;

/* 8 samples from p in 16-bit lanes */
static inline uint16x8_t load8_8(const uint8_t *p)
{
    return vmovl_u8(vld1_u8(p));
}

static inline uint16x8_t load8_16(const uint16_t *p)
{
    return vld1q_u16(p);
}

/* 16 samples from p in 16-bit lanes: columns 0 .. 7 in val[0], 8 .. 15 in
 * val[1]. Clang turns a widening load followed by an add into one
 * widening add (uaddl). */
static inline uint16x8x2_t load16_8(const uint8_t *p)
{
    const uint8x16_t v = vld1q_u8(p);
    return (uint16x8x2_t) { { vmovl_u8(vget_low_u8(v)), vmovl_high_u8(v) } };
}

static inline uint16x8x2_t load16_16(const uint16_t *p)
{
    const uint16x8_t lo = vld1q_u16(p);
    const uint16x8_t hi = vld1q_u16(p + 8);
    return (uint16x8x2_t) { { lo, hi } };
}

#define T uint8_t
#define FN(name) name##_8
#define SSE_ROW(org, rec, w) psnr_sse_line_8_neon(org, rec, w)
#include "xpsnr_any_neon_kernels.h"
#undef T
#undef FN
#undef SSE_ROW

#define T uint16_t
#define FN(name) name##_16
/* FFmpeg sums a row's squared errors in 32 bits */
#define SSE_ROW(org, rec, w) ((uint32_t) psnr_sse_line_16_neon(org, rec, w))
#include "xpsnr_any_neon_kernels.h"
#undef T
#undef FN
#undef SSE_ROW
