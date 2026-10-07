// Derived from libvmaf (https://github.com/Netflix/vmaf): cambi.c and
// luminance_tools.c, Copyright 2021-2026 Netflix, Inc., licensed under the
// BSD+Patent License (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// The CPU side of VMAF v1's CAMBI in the Vulkan engine (v1_host.c): libvmaf's
// own functions, copied as written and compiled as C, for what CAMBI
// calculates once per video (its thresholds, from CRT pow) and what the GPU
// cannot do bit for bit (its top-k average when the sum is not exact in a
// double, quick_select's order then deciding the rounding).

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define V1_CAMBI_SCALES 5
#define V1_CAMBI_MAX_DIFFS 32

typedef struct V1CambiOptions {
    int high_res_speedup;           // cambi_high_res_speedup: 0, 1080, 1440 or 2160
    double vis_lum_threshold;       // cambi_vis_lum_threshold
    double max_val;                 // cambi_max_val
    double topk;                    // the topk in effect (topk, else cambi_topk)
    int window_size;                // window_size
    double tvi_threshold;           // tvi_threshold
    int max_log_contrast;           // max_log_contrast
    int eotf;                       // the eotf in effect: 0 bt1886, 1 pq
} V1CambiOptions;

typedef struct V1CambiConstants {
    int speedup;                    // cambi_high_res_speedup after its resolution check
    int num_diffs;
    int tvi_for_diff[V1_CAMBI_MAX_DIFFS];   // + num_diffs, as cambi.c keeps them
    int diff_weights[V1_CAMBI_MAX_DIFFS];
    int vlt_luma;
    int window_size;                // adjusted
    int v_band_base, v_band_size;
    int mask_index;
    int pixels_in_window;
    int scale_w[V1_CAMBI_SCALES], scale_h[V1_CAMBI_SCALES];
    int topk_elements[V1_CAMBI_SCALES];
} V1CambiConstants;

// cambi.c's init (and its per-frame sizes) for a w x h picture; 0, or a
// negative error as init returns it.
int v1_cambi_constants(const V1CambiOptions *options, unsigned w, unsigned h, V1CambiConstants *out);

// cambi.c's spatial_pooling of a scale's c-values (reordered in place).
double v1_cambi_pool(float *c_values, double topk, unsigned width, unsigned height);

// cambi.c's weight_scores_per_scale and extract's MIN with cambi_max_val.
double v1_cambi_score(const double scores_per_scale[V1_CAMBI_SCALES], int pixels_in_window, double max_val);

#ifdef __cplusplus
}
#endif
