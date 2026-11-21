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
 * libvmaf-fast's NEON copy of x86/psnr_avx2.c: a row's sum of squared
 * errors. Every term is an exact integer and integer sums do not depend on
 * the order they are added in, so the sums are the C code's (integer_psnr.c,
 * xpsnr_template.c).
 */

#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>

#include "psnr_neon.h"

uint32_t psnr_sse_line_8_neon(const uint8_t *ref, const uint8_t *dis, unsigned w)
{
    /* Each lane's sum wraps modulo 2^32 like the C code's uint32_t sum, so
     * the lanes' total is the C code's too. */
    uint32x4_t sum = vdupq_n_u32(0);
    unsigned j = 0;
    for (; j + 16 <= w; j += 16) {
        const uint8x16_t r = vld1q_u8(ref + j);
        const uint8x16_t d = vld1q_u8(dis + j);
        /* r - d widened to 16 bits (-255 .. 255), then squared and widened
         * to 32 bits (at most 65025), pairs added into the four lanes */
        const int16x8_t lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(r), vget_low_u8(d)));
        const int16x8_t hi = vreinterpretq_s16_u16(vsubl_high_u8(r, d));
        int32x4_t sq = vmull_s16(vget_low_s16(lo), vget_low_s16(lo));
        sq = vmlal_high_s16(sq, lo, lo);
        sq = vmlal_s16(sq, vget_low_s16(hi), vget_low_s16(hi));
        sq = vmlal_high_s16(sq, hi, hi);
        sum = vaddq_u32(sum, vreinterpretq_u32_s32(sq));
    }
    uint32_t result = vaddvq_u32(sum);
    for (; j < w; j++) {
        const int16_t e = ref[j] - dis[j];
        result += (uint32_t)(e * e);
    }
    return result;
}

uint64_t psnr_sse_line_16_neon(const uint16_t *ref, const uint16_t *dis, unsigned w)
{
    uint64x2_t sum = vdupq_n_u64(0);
    unsigned j = 0;
    for (; j + 8 <= w; j += 8) {
        const uint16x8_t r = vld1q_u16(ref + j);
        const uint16x8_t d = vld1q_u16(dis + j);
        /* |r - d| fits 16 bits; its square (up to 65535^2) fits 32, and
         * pairs of squares are widened to 64 bits as they are added */
        const uint16x8_t e = vabdq_u16(r, d);
        const uint32x4_t sq_lo = vmull_u16(vget_low_u16(e), vget_low_u16(e));
        const uint32x4_t sq_hi = vmull_high_u16(e, e);
        sum = vpadalq_u32(sum, sq_lo);
        sum = vpadalq_u32(sum, sq_hi);
    }
    uint64_t result = vaddvq_u64(sum);
    for (; j < w; j++) {
        const uint32_t e = (uint32_t)abs((int32_t)ref[j] - (int32_t)dis[j]);
        result += (uint64_t)e * e;
    }
    return result;
}
