// Derived from libvmaf (https://github.com/Netflix/vmaf): cambi.c and
// luminance_tools.c, Copyright 2021-2026 Netflix, Inc., licensed under the
// BSD+Patent License (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// See v1_host.h. Everything below the line marked "libvmaf" is libvmaf's
// code as written (only static, and the names of what is exported), compiled
// as C with the same compiler and runtime as libvmaf.dll, so that its doubles
// and CRT pow round as libvmaf's do.

#include "v1_host.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// ------------------------------------------------------------ libvmaf

/* luminance_tools.h */
typedef double (*VmafEOTF)(double V);

enum VmafPixelRange {
    VMAF_PIXEL_RANGE_UNKNOWN,
    VMAF_PIXEL_RANGE_LIMITED,
    VMAF_PIXEL_RANGE_FULL,
};

typedef struct VmafLumaRange {
    int foot;
    int head;
} VmafLumaRange;

/* luminance_tools.c */
#define LT_MAX(x, y) (((x) > (y)) ? (x) : (y))

#define BT1886_GAMMA (2.4)
#define BT1886_LW (300.0)
#define BT1886_LB (0.01)

static inline int lt_clip(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

static inline int range_foot_head(int bitdepth, enum VmafPixelRange pix_range, int *foot, int *head) {
    switch (pix_range) {
    case VMAF_PIXEL_RANGE_LIMITED:
        *foot = 16 * (1 << (bitdepth - 8));
        *head = 235 * (1 << (bitdepth - 8));
        break;
    case VMAF_PIXEL_RANGE_FULL:
        *foot = 0;
        *head = (1 << bitdepth) - 1;
        break;
    default:
        return -EINVAL;
    }
    return 0;
}

static inline double normalize_range(int sample, VmafLumaRange range) {
    int clipped_sample = lt_clip(sample, range.foot, range.head);
    return (double)(clipped_sample - range.foot) / (range.head - range.foot);
}

static int vmaf_luminance_init_luma_range(VmafLumaRange *luma_range, int bitdepth, enum VmafPixelRange pix_range) {
    int err = range_foot_head(bitdepth, pix_range, &(luma_range->foot), &(luma_range->head));
    return err;
}

static double vmaf_luminance_bt1886_eotf(double V) {
    double a = pow(pow(BT1886_LW, 1.0 / BT1886_GAMMA) - pow(BT1886_LB, 1.0 / BT1886_GAMMA), BT1886_GAMMA);
    double b = pow(BT1886_LB, 1.0 / BT1886_GAMMA) / (pow(BT1886_LW, 1.0 / BT1886_GAMMA) - pow(BT1886_LB, 1.0 / BT1886_GAMMA));
    return a * pow(LT_MAX(V + b, 0), BT1886_GAMMA);
}

static double vmaf_luminance_pq_eotf(double V) {
    const double m_1 = 0.1593017578125;
    const double m_2 = 78.84375;
    const double c_1 = 0.8359375;
    const double c_2 = 18.8515625;
    const double c_3 = 18.6875;  // c_3 = c_1 + c_2 - 1
    double num = pow(V, 1.0 / m_2) - c_1;
    double num_clipped = LT_MAX(num, 0);
    double den = c_2 - c_3 * pow(V, 1.0 / m_2);
    return 10000 * pow(num_clipped / den, 1.0 / m_1);
}

static double vmaf_luminance_get_luminance(int sample, VmafLumaRange luma_range, VmafEOTF eotf) {
    double normalized = normalize_range(sample, luma_range);
    return eotf(normalized);
}

/* cambi.c */
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p (1920 * 1080)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p (2560 * 1440)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p (3840 * 2160)
#define CAMBI_MIN_WIDTH_HEIGHT (216)
#define CAMBI_RECIPROCAL_LUT_SIZE 4226
#define NUM_SCALES 5
static const int g_scale_weights[NUM_SCALES] = {16, 8, 4, 2, 1};

/* Suprathreshold contrast response */
static const int g_contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                           8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define SWAP_FLOATS(x, y) \
    {                     \
        float temp = x;   \
        x = y;            \
        y = temp;         \
    }

#define MASK_FILTER_SIZE 7

enum CambiTVIBisectFlag {
    CAMBI_TVI_BISECT_TOO_SMALL,
    CAMBI_TVI_BISECT_CORRECT,
    CAMBI_TVI_BISECT_TOO_BIG
};

static inline int clip(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

static bool tvi_condition(int sample, int diff, double tvi_threshold,
                          VmafLumaRange luma_range, VmafEOTF eotf) {
    double mean_luminance = vmaf_luminance_get_luminance(sample, luma_range, eotf);
    double diff_luminance = vmaf_luminance_get_luminance(sample + diff, luma_range, eotf);
    double delta_luminance = diff_luminance - mean_luminance;
    return (delta_luminance > tvi_threshold * mean_luminance);
}

static enum CambiTVIBisectFlag tvi_hard_threshold_condition(int sample, int diff, double tvi_threshold,
                                                            VmafLumaRange luma_range, VmafEOTF eotf) {
    bool condition;
    condition = tvi_condition(sample, diff, tvi_threshold, luma_range, eotf);
    if (!condition) return CAMBI_TVI_BISECT_TOO_BIG;

    condition = tvi_condition(sample + 1, diff, tvi_threshold, luma_range, eotf);
    if (condition) return CAMBI_TVI_BISECT_TOO_SMALL;

    return CAMBI_TVI_BISECT_CORRECT;
}

static int get_tvi_for_diff(int diff, double tvi_threshold, int bitdepth, VmafLumaRange luma_range, VmafEOTF eotf) {
    enum CambiTVIBisectFlag tvi_bisect;
    const int max_val = (1 << bitdepth) - 1;

    int foot = luma_range.foot;
    int head = luma_range.head;
    head = head - diff - 1;

    tvi_bisect = tvi_hard_threshold_condition(foot, diff, tvi_threshold, luma_range, eotf);
    if (tvi_bisect == CAMBI_TVI_BISECT_TOO_BIG) return 0;
    if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT) return foot;

    tvi_bisect = tvi_hard_threshold_condition(head, diff, tvi_threshold, luma_range, eotf);
    if (tvi_bisect == CAMBI_TVI_BISECT_TOO_SMALL) return max_val;
    if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT) return head;

    // bisect
    while (1) {
        int mid = foot + (head - foot) / 2;
        tvi_bisect = tvi_hard_threshold_condition(mid, diff, tvi_threshold, luma_range, eotf);
        if (tvi_bisect == CAMBI_TVI_BISECT_TOO_BIG)
            head = mid;
        else if (tvi_bisect == CAMBI_TVI_BISECT_TOO_SMALL)
            foot = mid;
        else if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT)
            return mid;
        else // Should never get here (todo: add assert)
            (void)0;
    }
}

static int get_vlt_luma(double visibility_luminance_threshold, VmafLumaRange luma_range, VmafEOTF eotf) {
    // find the smallest luma value above the visibility_luminance_threshold

    uint16_t sample = luma_range.foot;

    while (vmaf_luminance_get_luminance(sample, luma_range, eotf) < visibility_luminance_threshold) {
        sample++;
    }
    if (sample == luma_range.foot) {
        return 0;
    } else {
        return sample;
    }
}

static inline void adjust_window_size(uint16_t *window_size,
                                      unsigned input_width,
                                      unsigned input_height,
                                      bool cambi_high_res_speedup)
{
    // Adjustment weight: (input_width + input_height) / (CAMBI_4K_WIDTH + CAMBI_4K_HEIGHT)
    (*window_size) = (((*window_size) * (input_width+input_height)) / 375) >> 4;
    if (cambi_high_res_speedup) {
        (*window_size) = ((*window_size) + 1) >> 1;
    }
    // round up to odd
    *window_size |= 1;
}

static inline uint16_t ceil_log2(uint32_t num) {
    if (num==0)
        return 0;

    uint32_t tmp = num - 1;
    uint16_t shift = 0;
    while (tmp>0) {
        tmp >>= 1;
        shift += 1;
    }
    return shift;
}

static inline uint16_t get_mask_index(unsigned input_width, unsigned input_height,
                                      uint16_t filter_size) {
    uint32_t shifted_wh = (input_width >> 6) * (input_height >> 6);
    return (filter_size * filter_size + 3 * (ceil_log2(shifted_wh) - 11) - 1)>>1;
}

static double average_topk_elements(const float *arr, int topk_elements) {
    double sum = 0;
    for (int i = 0; i < topk_elements; i++)
        sum += arr[i];

    return (double)sum / topk_elements;
}

static void quick_select(float *arr, int n, int k) {
    if (n == k) return;
    int left = 0;
    int right = n - 1;
    while (left < right) {
        float pivot = arr[k];
        int i = left;
        int j = right;
        do {
            while (arr[i] > pivot) {
                i++;
            }
            while (arr[j] < pivot) {
                j--;
            }
            if (i <= j) {
                SWAP_FLOATS(arr[i], arr[j]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k) {
            left = i;
        }
        if (k < i) {
            right = j;
        }
    }
}

static double spatial_pooling(float *c_values, double topk, unsigned width, unsigned height) {
    int num_elements = height * width;
    int topk_num_elements = clip(topk * num_elements, 1, num_elements);
    quick_select(c_values, num_elements, topk_num_elements);
    return average_topk_elements(c_values, topk_num_elements);
}

static inline uint16_t get_pixels_in_window(uint16_t window_length) {
    uint16_t odd_length = 2 * (window_length >> 1) + 1;
    return odd_length * odd_length;
}

// Inner product weighting scores for each scale
static inline double weight_scores_per_scale(double *scores_per_scale, uint16_t normalization) {
    double score = 0.0;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++)
        score += (scores_per_scale[scale] * g_scale_weights[scale]);

    return score / normalization;
}

// ------------------------------------------------------------ the engine's

int v1_cambi_constants(const V1CambiOptions *o, unsigned w, unsigned h, V1CambiConstants *out)
{
    memset(out, 0, sizeof *out);
    // init(): enc and src sizes are the picture's (no enc_* or src_* options).
    if (w < CAMBI_MIN_WIDTH_HEIGHT && h < CAMBI_MIN_WIDTH_HEIGHT)
        return -EINVAL;
    int speedup = o->high_res_speedup;
    int enc_pix = w * h;
    switch (speedup) {
        case 1080:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p) speedup = 0;
            break;
        case 1440:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p) speedup = 0;
            break;
        case 2160:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p) speedup = 0;
            break;
        default:
            speedup = 0;
    }
    out->speedup = speedup;
    if (o->max_log_contrast < 0 || o->max_log_contrast > 5)
        return -EINVAL;
    const int num_diffs = 1 << o->max_log_contrast;
    out->num_diffs = num_diffs;
    for (int d = 0; d < num_diffs; d++)
        out->diff_weights[d] = g_contrast_weights[d];

    VmafLumaRange luma_range;
    int err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err) return err;
    VmafEOTF eotf = o->eotf == 1 ? vmaf_luminance_pq_eotf : vmaf_luminance_bt1886_eotf;
    for (int d = 0; d < num_diffs; d++) {
        out->tvi_for_diff[d] = (uint16_t)get_tvi_for_diff(d + 1, o->tvi_threshold, 10, luma_range, eotf);
        out->tvi_for_diff[d] = (uint16_t)(out->tvi_for_diff[d] + num_diffs);
    }
    out->vlt_luma = (uint16_t)get_vlt_luma(o->vis_lum_threshold, luma_range, eotf);

    uint16_t window_size = (uint16_t)o->window_size;
    adjust_window_size(&window_size, w, h, (bool)speedup);
    if (window_size * window_size >= CAMBI_RECIPROCAL_LUT_SIZE)
        return -EINVAL;
    out->window_size = window_size;

    int v_lo_signed = (int)out->vlt_luma - 3 * (int)num_diffs + 1;
    out->v_band_base = v_lo_signed > 0 ? (uint16_t)v_lo_signed : 0;
    out->v_band_size = (uint16_t)(out->tvi_for_diff[num_diffs - 1] + 1 - out->v_band_base);

    out->mask_index = get_mask_index(w, h, MASK_FILTER_SIZE);
    out->pixels_in_window = get_pixels_in_window(window_size);

    int scaled_width = (int)w, scaled_height = (int)h;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        if (scale > 0 || speedup) {
            scaled_width = (scaled_width + 1) >> 1;
            scaled_height = (scaled_height + 1) >> 1;
        }
        out->scale_w[scale] = scaled_width;
        out->scale_h[scale] = scaled_height;
        int num_elements = scaled_height * scaled_width;
        out->topk_elements[scale] = clip(o->topk * num_elements, 1, num_elements);
    }
    return 0;
}

double v1_cambi_pool(float *c_values, double topk, unsigned width, unsigned height)
{
    return spatial_pooling(c_values, topk, width, height);
}

double v1_cambi_score(const double scores_per_scale[V1_CAMBI_SCALES], int pixels_in_window, double max_val)
{
    double scores[NUM_SCALES];
    memcpy(scores, scores_per_scale, sizeof scores);
    double dist_score = weight_scores_per_scale(scores, (uint16_t)pixels_in_window);
    return MIN(dist_score, max_val);
}
