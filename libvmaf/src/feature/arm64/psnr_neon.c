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
 * libvmaf-fast's NEON rows of squared errors for integer_psnr.c and
 * xpsnr.c, the Arm counterpart of x86/psnr_avx2.c. A squared error is the
 * square of the absolute difference, so the rows square |ref - dis|
 * (vabd), which needs no sign: at 8 bits each square fits 16 bits, at 16
 * bits 32. The 8-bit sum is 32 bits like the C code's: its lanes wrap as
 * that sum does, and a sum modulo 2^32 does not depend on the order.
 */

#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>

#include "psnr_neon.h"

/* The squares of |ref - dis| of 16 samples, in 16-bit lanes, added in pairs
 * into the 32-bit sums lo (samples 0 .. 7) and hi (8 .. 15). */
static inline void add_squares_8(uint32x4_t *lo, uint32x4_t *hi, const uint8_t *ref, const uint8_t *dis)
{
    const uint8x16_t r = vld1q_u8(ref);
    const uint8x16_t d = vld1q_u8(dis);
    const uint8x16_t e = vabdq_u8(r, d);
    *lo = vpadalq_u16(*lo, vmull_u8(vget_low_u8(e), vget_low_u8(e)));
    *hi = vpadalq_u16(*hi, vmull_high_u8(e, e));
}

/* 32 samples a step, into four sums */
uint32_t psnr_sse_line_8_neon(const uint8_t *ref, const uint8_t *dis, unsigned w)
{
    uint32x4_t acc0 = vdupq_n_u32(0);
    uint32x4_t acc1 = vdupq_n_u32(0);
    uint32x4_t acc2 = vdupq_n_u32(0);
    uint32x4_t acc3 = vdupq_n_u32(0);
    unsigned x = 0;
    for (; x + 32 <= w; x += 32) {
        add_squares_8(&acc0, &acc1, ref + x, dis + x);
        add_squares_8(&acc2, &acc3, ref + x + 16, dis + x + 16);
    }
    if (x + 16 <= w) {
        add_squares_8(&acc0, &acc1, ref + x, dis + x);
        x += 16;
    }
    uint32_t sse = vaddvq_u32(vaddq_u32(vaddq_u32(acc0, acc1), vaddq_u32(acc2, acc3)));
    for (; x < w; x++) {
        const int16_t e = ref[x] - dis[x];
        sse += e * e;
    }
    return sse;
}

/* The squares of |ref - dis| of 8 samples, in 32-bit lanes, added in pairs
 * into the 64-bit sums lo (samples 0 .. 3) and hi (4 .. 7). */
static inline void add_squares_16(uint64x2_t *lo, uint64x2_t *hi, const uint16_t *ref, const uint16_t *dis)
{
    const uint16x8_t r = vld1q_u16(ref);
    const uint16x8_t d = vld1q_u16(dis);
    const uint16x8_t e = vabdq_u16(r, d);
    *lo = vpadalq_u32(*lo, vmull_u16(vget_low_u16(e), vget_low_u16(e)));
    *hi = vpadalq_u32(*hi, vmull_high_u16(e, e));
}

/* 16 samples a step, into four sums */
uint64_t psnr_sse_line_16_neon(const uint16_t *ref, const uint16_t *dis, unsigned w)
{
    uint64x2_t acc0 = vdupq_n_u64(0);
    uint64x2_t acc1 = vdupq_n_u64(0);
    uint64x2_t acc2 = vdupq_n_u64(0);
    uint64x2_t acc3 = vdupq_n_u64(0);
    unsigned x = 0;
    for (; x + 16 <= w; x += 16) {
        add_squares_16(&acc0, &acc1, ref + x, dis + x);
        add_squares_16(&acc2, &acc3, ref + x + 8, dis + x + 8);
    }
    if (x + 8 <= w) {
        add_squares_16(&acc0, &acc1, ref + x, dis + x);
        x += 8;
    }
    uint64_t sse = vaddvq_u64(vaddq_u64(vaddq_u64(acc0, acc1), vaddq_u64(acc2, acc3)));
    for (; x < w; x++) {
        const uint32_t e = abs(ref[x] - dis[x]);
        sse += (uint64_t) e * e;
    }
    return sse;
}
