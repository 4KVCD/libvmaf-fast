// Derived from libvmaf (https://github.com/Netflix/vmaf): speed.c and
// vif_tools.c, Copyright 2016-2025 Netflix, Inc., licensed under the
// BSD+Patent License (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// The CPU side of VMAF v1's SpEED chroma in the Vulkan engine (v1_speed.c).
// The GPU filters each chroma plane and takes it down by 16 (speed.c's
// filter_and_downscale); what is left, est_params and get_speed_score on a
// picture of some thousand values, is libvmaf's own code, copied as written
// and compiled as libvmaf compiles it: its covariance in double with the
// kernel libvmaf picks for this CPU (AVX-512, AVX2 or plain, which round
// differently), its eigenvalues and its CRT log2.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct V1SpeedOptions {
    double kernelscale;      // speed_kernelscale
    double prescale;         // speed_prescale
    int bilinear;            // speed_prescale_method is "bilinear" (the only one calculated here)
    double sigma_nn;         // speed_sigma_nn
    double nn_floor;         // speed_nn_floor
    int weight_var_mode;     // speed_weight_var_mode
    double max_val;          // speed_chroma_max_val
} V1SpeedOptions;

#define V1_SPEED_MAX_TAPS 64

typedef struct V1SpeedFilters {
    int scaled;                            // the plane is rescaled (bilinear) first
    int scaled_w, scaled_h;                // its size then
    int operating_w, operating_h;          // the plane's size after filter_and_downscale
    int antialias_taps, blur_taps;
    float antialias[V1_SPEED_MAX_TAPS];    // speed_get_antialias_filter
    float blur[V1_SPEED_MAX_TAPS];         // vif_get_filter at scale 4
} V1SpeedFilters;

typedef struct V1Speed V1Speed;

// speed_init's checks and sizes for a w x h chroma plane, and the filters;
// 0, -3 for what speed.c refuses (too small, an invalid kernelscale), -4 for
// what is not calculated here (a prescale method other than bilinear).
int v1_speed_filters(const V1SpeedOptions *options, unsigned w, unsigned h, V1SpeedFilters *out);

// For a rescaled plane: the bilinear scaling's tables, as vif_tools.c works
// them out -- for each column of the scaled plane (scaled_w) the two source
// columns and the weight, and for each row (scaled_h) the same.
void v1_speed_bilinear_tables(unsigned w, unsigned h, const V1SpeedFilters *filters, int *x1, int *x2, float *dx,
                              int *y1, int *y2, float *dy);

// The buffers est_params works in, for one thread.
V1Speed *v1_speed_new(const V1SpeedOptions *options, unsigned w, unsigned h);
void v1_speed_free(V1Speed *speed);

// extract_chroma from the filtered planes (operating_w x operating_h, rows
// operating_w apart): ref U, dis U, ref V, dis V. out: the u, v and uv
// scores with speed_chroma_max_val, as libvmaf appends them.
void v1_speed_chroma(V1Speed *speed, float *planes[4], double out[3]);

#ifdef __cplusplus
}
#endif
