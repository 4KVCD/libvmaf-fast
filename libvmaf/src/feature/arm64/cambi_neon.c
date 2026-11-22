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
 * libvmaf-fast's NEON code for CAMBI's kernels, each the scalar code of
 * cambi.c lane by lane (x86/cambi_avx2.c is upstream's AVX2 of the same).
 * All but the c-values are integer work. A c-value is the largest of a few
 * float products, each one IEEE multiply of the same two floats as the
 * scalar code's, so the same float; NEON has no gather, so the histogram
 * and reciprocal_lut entries are read one lane at a time.
 */

#include <arm_neon.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "libvmaf/picture.h"
#include "cambi_neon.h"

/* dst[i, j] = src[2i, 2j] in place: reads stay ahead of writes, within a
 * row and from row to row. vld2 splits 16 samples into the even and odd
 * ones. */
void decimate_neon(VmafPicture *image, unsigned width, unsigned height)
{
    uint16_t *data = image->data[0];
    const ptrdiff_t stride = image->stride[0] >> 1;
    for (unsigned i = 0; i < height; i++) {
        const uint16_t *src = &data[2 * i * stride];
        uint16_t *dst = &data[i * stride];
        unsigned j = 0;
        for (; j + 8 <= width; j += 8)
            vst1q_u16(&dst[j], vld2q_u16(&src[2 * j]).val[0]);
        for (; j < width; j++)
            dst[j] = src[2 * j];
    }
}

/* cambi.c's mode3: a if it equals b or c, else b if b equals c, else the
 * smallest */
static inline uint16x8_t mode3_neon(uint16x8_t a, uint16x8_t b, uint16x8_t c)
{
    const uint16x8_t a_dup = vorrq_u16(vceqq_u16(a, b), vceqq_u16(a, c));
    const uint16x8_t bc_eq = vceqq_u16(b, c);
    const uint16x8_t min_abc = vminq_u16(vminq_u16(a, b), c);
    return vbslq_u16(a_dup, a, vbslq_u16(bc_eq, b, min_abc));
}

static inline uint16_t mode3_scalar(uint16_t a, uint16_t b, uint16_t c)
{
    if (a == b || a == c) return a;
    if (b == c) return b;
    const uint16_t ab = a < b ? a : b;
    return ab < c ? ab : c;
}

void filter_mode_neon(const VmafPicture *image, int width, int height, uint16_t *buffer)
{
    uint16_t *data = image->data[0];
    const ptrdiff_t stride = image->stride[0] >> 1;
    int curr_line = 0;
    for (int i = 0; i < height; i++) {
        /* horizontal: buffer[curr_line] = mode3 of each sample and its two
         * neighbours, columns 1 .. width - 2; the first and last copied */
        const uint16_t *src = &data[i * stride];
        uint16_t *line = &buffer[curr_line * width];
        line[0] = src[0];
        int j = 1;
        for (; j + 8 < width; j += 8)  /* columns j .. j + 7 <= width - 2 */
            vst1q_u16(&line[j], mode3_neon(vld1q_u16(&src[j - 1]), vld1q_u16(&src[j]),
                                           vld1q_u16(&src[j + 1])));
        for (; j < width - 1; j++)
            line[j] = mode3_scalar(src[j - 1], src[j], src[j + 1]);
        line[width - 1] = src[width - 1];

        /* vertical: row i - 1 = mode3 of the three buffered lines */
        if (i > 1) {
            uint16_t *dst = &data[(i - 1) * stride];
            int k = 0;
            for (; k + 8 <= width; k += 8)
                vst1q_u16(&dst[k], mode3_neon(vld1q_u16(&buffer[k]), vld1q_u16(&buffer[width + k]),
                                              vld1q_u16(&buffer[2 * width + k])));
            for (; k < width; k++)
                dst[k] = mode3_scalar(buffer[k], buffer[width + k], buffer[2 * width + k]);
        }
        curr_line = (curr_line + 1 == 3 ? 0 : curr_line + 1);
    }
}

void cambi_increment_range_neon(uint16_t *arr, int left, int right)
{
    const uint16x8_t one = vdupq_n_u16(1);
    int col = left;
    for (; col + 8 <= right; col += 8)
        vst1q_u16(&arr[col], vaddq_u16(vld1q_u16(&arr[col]), one));
    for (; col < right; col++)
        arr[col]++;
}

void cambi_decrement_range_neon(uint16_t *arr, int left, int right)
{
    const uint16x8_t one = vdupq_n_u16(1);
    int col = left;
    for (; col + 8 <= right; col += 8)
        vst1q_u16(&arr[col], vsubq_u16(vld1q_u16(&arr[col]), one));
    for (; col < right; col++)
        arr[col]--;
}

/* 1 where a sample equals its right neighbour and (except on the last row)
 * the one below; the last column counts as equal to its right */
void get_derivative_data_for_row_neon(const uint16_t *image_data, uint16_t *derivative_buffer,
                                      int width, int height, int row, int stride)
{
    const uint16_t *cur = &image_data[row * stride];
    const bool last_row = row == height - 1;
    const uint16_t *below = last_row ? cur : &image_data[(row + 1) * stride];
    const uint16x8_t one = vdupq_n_u16(1);
    int col = 0;
    for (; col + 8 < width; col += 8) {  /* reads col + 8 <= width - 1 */
        const uint16x8_t v = vld1q_u16(&cur[col]);
        const uint16x8_t equal = vandq_u16(vceqq_u16(v, vld1q_u16(&cur[col + 1])),
                                           vceqq_u16(v, vld1q_u16(&below[col])));
        vst1q_u16(&derivative_buffer[col], vandq_u16(equal, one));
    }
    for (; col < width; col++) {
        const bool horizontal = col == width - 1 || cur[col] == cur[col + 1];
        const bool vertical = last_row || cur[col] == below[col];
        derivative_buffer[col] = horizontal && vertical;
    }
}

/* the running sums of 4 lanes: lane i = x[0] + ... + x[i] (mod 2^32, as
 * the scalar uint32_t sums) */
static inline uint32x4_t inclusive_prefix_u32(uint32x4_t x)
{
    const uint32x4_t zero = vdupq_n_u32(0);
    x = vaddq_u32(x, vextq_u32(zero, x, 3));  /* + the lane before */
    return vaddq_u32(x, vextq_u32(zero, x, 2)); /* + the sums two lanes before */
}

void compute_dp_row_neon(uint32_t *dp_curr, const uint32_t *dp_prev,
                         const uint16_t *deriv, int width, int pad_size, bool deriv_valid)
{
    const int dp_offset = pad_size + 1;
    const int actual_width = deriv_valid ? width : 0;
    int j = 0;
    uint32x4_t carry = vdupq_n_u32(0);  /* the row's sum so far, in every lane */
    for (; j + 4 <= actual_width; j += 4) {
        const uint32x4_t scan = vaddq_u32(inclusive_prefix_u32(vmovl_u16(vld1_u16(&deriv[j]))), carry);
        vst1q_u32(&dp_curr[dp_offset + j], vaddq_u32(vld1q_u32(&dp_prev[dp_offset + j]), scan));
        carry = vdupq_laneq_u32(scan, 3);
    }
    uint32_t prefix = vgetq_lane_u32(carry, 0);
    for (; j < actual_width; j++) {
        prefix += deriv[j];
        dp_curr[dp_offset + j] = dp_prev[dp_offset + j] + prefix;
    }
    const uint32x4_t vprefix = vdupq_n_u32(prefix);
    const int n = width + pad_size;
    for (; j + 4 <= n; j += 4)
        vst1q_u32(&dp_curr[dp_offset + j], vaddq_u32(vld1q_u32(&dp_prev[dp_offset + j]), vprefix));
    for (; j < n; j++)
        dp_curr[dp_offset + j] = dp_prev[dp_offset + j] + prefix;
}

/* mask_row[j] = (the box sum from the four corners > mask_index), the sum
 * and the comparison unsigned as in cambi.c */
void compute_mask_row_neon(uint16_t *mask_row, const uint32_t *dp_bottom, const uint32_t *dp_top,
                           int width, int pad_size, uint32_t mask_index)
{
    const int delta = 2 * pad_size + 1;
    const uint32x4_t midx = vdupq_n_u32(mask_index);
    const uint32x4_t one = vdupq_n_u32(1);
    int j = 0;
    for (; j + 4 <= width; j += 4) {
        const uint32x4_t sum = vsubq_u32(vaddq_u32(vld1q_u32(&dp_bottom[j + delta]), vld1q_u32(&dp_top[j])),
                                         vaddq_u32(vld1q_u32(&dp_bottom[j]), vld1q_u32(&dp_top[j + delta])));
        vst1_u16(&mask_row[j], vmovn_u32(vandq_u32(vcgtq_u32(sum, midx), one)));
    }
    for (; j < width; j++) {
        const uint32_t sum = dp_bottom[j + delta] + dp_top[j] - dp_bottom[j] - dp_top[j + delta];
        mask_row[j] = (uint16_t)(sum > mask_index);
    }
}

/* base[idx] of the 4 lanes, one lane at a time */
static inline uint32x4_t gather4_u16(const uint16_t *base, uint32x4_t idx)
{
    uint16x4_t v = vdup_n_u16(0);
    v = vld1_lane_u16(base + vgetq_lane_u32(idx, 0), v, 0);
    v = vld1_lane_u16(base + vgetq_lane_u32(idx, 1), v, 1);
    v = vld1_lane_u16(base + vgetq_lane_u32(idx, 2), v, 2);
    v = vld1_lane_u16(base + vgetq_lane_u32(idx, 3), v, 3);
    return vmovl_u16(v);
}

static inline float32x4_t gather4_f32(const float *base, uint32x4_t idx)
{
    float32x4_t v = vdupq_n_f32(0.0f);
    v = vld1q_lane_f32(base + vgetq_lane_u32(idx, 0), v, 0);
    v = vld1q_lane_f32(base + vgetq_lane_u32(idx, 1), v, 1);
    v = vld1q_lane_f32(base + vgetq_lane_u32(idx, 2), v, 2);
    v = vld1q_lane_f32(base + vgetq_lane_u32(idx, 3), v, 3);
    return v;
}

/* cambi.c's c_value_pixel for one pixel (the columns after the last whole
 * vector) */
static float c_value_scalar(const uint16_t *histograms, int col, int width, uint16_t raw,
                            uint16_t num_diffs, const uint16_t *tvi_thresholds, uint16_t vlt_luma,
                            const int *diff_weights, const int *all_diffs,
                            const float *reciprocal_lut, uint16_t v_band_base, uint16_t v_band_size)
{
    const uint16_t value = (uint16_t)(raw + num_diffs);
    const int compact = (int)raw - (int)v_band_base;
    if ((unsigned)compact >= v_band_size)
        return 0.0f;
    const uint16_t p_0 = histograms[compact * width + col];
    float c_v = 0.0f;
    for (int d = 0; d < num_diffs; d++) {
        if (value <= tvi_thresholds[d] && value + all_diffs[num_diffs + d + 1] > vlt_luma) {
            const int idx1 = compact + all_diffs[num_diffs + d + 1];
            const int idx2 = compact + all_diffs[num_diffs - d - 1];
            const uint16_t p_1 = histograms[idx1 * width + col];
            const uint16_t p_2 = idx2 >= 0 ? histograms[idx2 * width + col] : 0;
            const uint16_t p_max = p_1 > p_2 ? p_1 : p_2;
            const float val = (float)(diff_weights[d] * p_0 * p_max) * reciprocal_lut[p_max + p_0];
            if (val > c_v)
                c_v = val;
        }
    }
    return c_v;
}

void calculate_c_values_row_neon(
    float *c_values, const uint16_t *histograms, const uint16_t *image, const uint16_t *mask,
    int row, int width, ptrdiff_t stride,
    const uint16_t num_diffs, const uint16_t *tvi_thresholds, uint16_t vlt_luma,
    const int *diff_weights, const int *all_diffs,
    const float *reciprocal_lut)
{
    /* the band of values whose histogram cells are kept (see cambi.c) */
    const int v_lo_signed = (int)vlt_luma - 3 * (int)num_diffs + 1;
    const uint16_t v_band_base = v_lo_signed > 0 ? (uint16_t)v_lo_signed : 0;
    const uint16_t v_band_size = tvi_thresholds[num_diffs - 1] + 1 - v_band_base;

    const uint16_t *image_row = &image[row * stride];
    const uint16_t *mask_row = &mask[row * stride];
    float *c_row = &c_values[row * width];

    static const uint32_t lane_index[4] = { 0, 1, 2, 3 };
    const uint32x4_t width_v = vdupq_n_u32((uint32_t)width);
    const int32x4_t zero = vdupq_n_s32(0);

    int col = 0;
    for (; col + 4 <= width; col += 4) {
        const uint32x4_t active = vtstq_u32(vmovl_u16(vld1_u16(&mask_row[col])),
                                            vreinterpretq_u32_s32(vdupq_n_s32(-1)));
        /* the pixel's compact band index; lanes outside the band (or
         * masked) give 0, as the scalar code */
        const int32x4_t raw = vreinterpretq_s32_u32(vmovl_u16(vld1_u16(&image_row[col])));
        const int32x4_t compact = vsubq_s32(raw, vdupq_n_s32(v_band_base));
        const uint32x4_t in_band = vandq_u32(
            active, vcltq_u32(vreinterpretq_u32_s32(compact), vdupq_n_u32(v_band_size)));
        if (vmaxvq_u32(in_band) == 0) {
            vst1q_f32(&c_row[col], vdupq_n_f32(0.0f));
            continue;
        }
        const int32x4_t value = vaddq_s32(raw, vdupq_n_s32(num_diffs));
        /* the column of each lane, and the band index clamped into the band
         * so that the lanes left out still read inside the histograms */
        const uint32x4_t col_v = vaddq_u32(vdupq_n_u32((uint32_t)col), vld1q_u32(lane_index));
        const int32x4_t band_max = vdupq_n_s32((int)v_band_size - 1);
        const int32x4_t safe = vminq_s32(vmaxq_s32(compact, zero), band_max);
        const uint32x4_t p_0 = gather4_u16(histograms,
                                           vmlaq_u32(col_v, vreinterpretq_u32_s32(safe), width_v));

        float32x4_t c_v = vdupq_n_f32(0.0f);
        for (int d = 0; d < num_diffs; d++) {
            const int delta_plus = all_diffs[num_diffs + d + 1];
            const int delta_minus = all_diffs[num_diffs - d - 1];
            /* value <= tvi_thresholds[d] && value + delta_plus > vlt_luma */
            const uint32x4_t fires = vandq_u32(
                in_band, vandq_u32(vcleq_s32(value, vdupq_n_s32(tvi_thresholds[d])),
                                   vcgtq_s32(vaddq_s32(value, vdupq_n_s32(delta_plus)),
                                             vdupq_n_s32(vlt_luma))));
            if (vmaxvq_u32(fires) == 0)
                continue;
            /* p_1 and p_2 (0 where its index is below the band) */
            const int32x4_t idx1 = vminq_s32(vaddq_s32(safe, vdupq_n_s32(delta_plus)), band_max);
            const int32x4_t idx2_raw = vaddq_s32(safe, vdupq_n_s32(delta_minus));
            const int32x4_t idx2 = vmaxq_s32(idx2_raw, zero);
            const uint32x4_t p_1 = gather4_u16(histograms,
                                               vmlaq_u32(col_v, vreinterpretq_u32_s32(idx1), width_v));
            const uint32x4_t p_2 = vandq_u32(
                gather4_u16(histograms, vmlaq_u32(col_v, vreinterpretq_u32_s32(idx2), width_v)),
                vcgeq_s32(idx2_raw, zero));
            const uint32x4_t p_max = vmaxq_u32(p_1, p_2);
            /* (float)(weight * p_0 * p_max) * reciprocal_lut[p_max + p_0];
             * the products fit an int, as the scalar code's */
            const int32x4_t num = vmulq_n_s32(vreinterpretq_s32_u32(vmulq_u32(p_0, p_max)), diff_weights[d]);
            const float32x4_t val = vmulq_f32(vcvtq_f32_s32(num),
                                              gather4_f32(reciprocal_lut, vaddq_u32(p_max, p_0)));
            /* c_v = val where the step fires and val > c_v */
            c_v = vbslq_f32(vandq_u32(fires, vcgtq_f32(val, c_v)), val, c_v);
        }
        vst1q_f32(&c_row[col], c_v);
    }
    for (; col < width; col++)
        c_row[col] = mask_row[col]
            ? c_value_scalar(histograms, col, width, image_row[col], num_diffs, tvi_thresholds,
                             vlt_luma, diff_weights, all_diffs, reciprocal_lut, v_band_base,
                             v_band_size)
            : 0.0f;
}
