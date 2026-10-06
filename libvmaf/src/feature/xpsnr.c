/*
 * Copyright (c) 2024 Christian R. Helmrich
 * Copyright (c) 2024 Christian Lehmann
 * Copyright (c) 2024 Christian Stoffers
 *
 * XPSNR, the extended perceptually weighted peak signal-to-noise ratio, of
 * the luma plane: feature "xpsnr_y" of extractor "xpsnr".
 *
 * From FFmpeg 9.0.1's libavfilter/vf_xpsnr.c (Helmrich, Lehmann and
 * Stoffers, Fraunhofer HHI), changed for libvmaf-fast in October 2026: a
 * libvmaf feature extractor instead of an FFmpeg filter; the luma plane
 * only; each frame scored from its own pictures and the two before it
 * (VMAF_FEATURE_EXTRACTOR_PREV_PICTURES), which FFmpeg's filter copies into
 * buffers of its own as it goes, so that frames can be scored on different
 * threads; 8-bit samples read as they are, not copied into 16-bit buffers;
 * min-smoothing of the weights after they are all known, not as they come
 * (the same steps, in the same order, from the same values). Every sum and
 * every rounding is FFmpeg's: the same score, to the last bit, for a picture
 * of an even size, or of any size up to 2048x1152 pixels. FFmpeg's filter
 * reads and writes beyond the picture for an odd width or height above
 * that, which init refuses, as it does more than 12 bits. Option "psnr":
 * psnr_y as well, the psnr extractor's, from the squared errors summed here
 * anyway.
 *
 * This file is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version. Unlike the rest of libvmaf (BSD+Patent).
 *
 * This file is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this file (libvmaf/COPYING.LGPLv2.1); if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301 USA
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* Before feature_extractor.h: VmafFeatureExtractor has a field more with
 * HAVE_CUDA, which libvmaf.c has; without it here fex->prev_ref and the
 * rest are read 8 bytes off (integer_motion.c has it through cpu.h). */
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "opt.h"

/* FFmpeg's sums and products, each rounded on its own: no fused
 * multiply-add, which the compiler may otherwise make of them. */
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#else
#pragma STDC FP_CONTRACT OFF
#endif

#define XPSNR_GAMMA 2
#define XPSNR_MAX(a, b) ((a) > (b) ? (a) : (b))
#define XPSNR_MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct XpsnrState {
    int frame_rate;
    bool weights_from_dist;
    bool psnr;
    uint32_t w, h;
    unsigned bpc;
    /* block size, blocks a row and a column, FFmpeg's b_val */
    uint32_t b, w_blk, h_blk;
    int b_val;
    double avg_act;
    uint64_t num64;
    double *sse_luma;
    double *weights;
    /* a row of zero samples: the pictures before the first */
    void *zeros;
} XpsnrState;

typedef struct XpsnrPlanes {
    const void *org, *m1, *m2, *rec;
    /* strides in samples; 0 for the row of zeros */
    ptrdiff_t so, s1, s2, sr;
} XpsnrPlanes;

static const VmafOption options[] = {
    {
        .name = "frame_rate",
        .help = "whole frames a second, as FFmpeg's xpsnr filter takes it "
                "from its second input (the distorted video, the original "
                "being first): below 32 the temporal activity is the "
                "difference from the frame before, else the difference of "
                "the two differences before",
        .offset = offsetof(XpsnrState, frame_rate),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = INT_MAX,
    },
    {
        .name = "weights_from_dist",
        .help = "perceptual weights from the distorted pictures' activity, "
                "not the reference's: FFmpeg's xpsnr filter takes them from "
                "its first input, which a command built as for its libvmaf "
                "filter makes the distorted video",
        .offset = offsetof(XpsnrState, weights_from_dist),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "psnr",
        .help = "psnr_y too, as the psnr extractor gives it with its default "
                "options, from the squared errors XPSNR sums anyway: a pass "
                "over both pictures saved, where psnr is not registered",
        .offset = offsetof(XpsnrState, psnr),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    { 0 }
};

#define T uint8_t
#define FN(name) name##_8
#include "xpsnr_template.c"
#undef T
#undef FN
#define T uint16_t
#define FN(name) name##_16
#include "xpsnr_template.c"
#undef T
#undef FN

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    XpsnrState *s = fex->priv;
    (void) pix_fmt;

    if (bpc < 8 || bpc > 12 || !w || !h)
        return -EINVAL;
    s->w = w;
    s->h = h;
    s->bpc = bpc;
    s->b_val = ((int) w * (int) h > 2048 * 1152 ? 2 : 1); /* threshold is a bit more than HD resolution */
    if (s->b_val > 1 && ((w & 1) || (h & 1)))
        return -EINVAL;

    const double r = (double) (s->w * s->h) / (3840.0 * 2160.0); /* UHD ratio */
    s->b = XPSNR_MAX(1, 4 * (int32_t) (32.0 * sqrt(r) + 0.5)); /* block size, integer multiple of 4 */
    s->w_blk = (s->w + s->b - 1) / s->b;
    s->h_blk = (s->h + s->b - 1) / s->b;
    s->avg_act = sqrt(16.0 * (double) (1 << (2 * bpc - 9)) / sqrt(XPSNR_MAX(0.00001, r))); /* the sqrt(a_pic) */
    const uint64_t max_error_64 = (uint64_t) ((1 << bpc) - 1) * (uint64_t) ((1 << bpc) - 1);
    s->num64 = (uint64_t) w * (uint64_t) h * max_error_64;

    s->sse_luma = malloc(sizeof(double) * s->w_blk * s->h_blk);
    s->weights = malloc(sizeof(double) * s->w_blk * s->h_blk);
    s->zeros = calloc(w + 8, sizeof(uint16_t));
    if (!s->sse_luma || !s->weights || !s->zeros) {
        free(s->sse_luma);
        free(s->weights);
        free(s->zeros);
        s->sse_luma = s->weights = NULL;
        s->zeros = NULL;
        return -ENOMEM;
    }
    return 0;
}

/* FFmpeg's in-line "min-smoothing" of the weights of a picture of at most
 * 640x480, as in the paper: each step reads the weights up to its own block
 * -- its own as computed, those before as earlier steps left them -- and
 * changes the one before it, so that taking the steps after all weights are
 * known changes nothing. */
static void smooth_weights(const XpsnrState *s, double *weights)
{
    const uint32_t b = s->b, w_blk = s->w_blk;
    uint32_t idx_blk = 0;

    for (uint32_t y = 0; y < s->h; y += b) {
        for (uint32_t x = 0; x < s->w; x += b, idx_blk++) {
            double ms_act_prev;

            if (x == 0) /* first column */
                ms_act_prev = (idx_blk > 1 ? weights[idx_blk - 2] : 0);
            else  /* after first column */
                ms_act_prev = (x > b ? XPSNR_MAX(weights[idx_blk - 2], weights[idx_blk]) : weights[idx_blk]);

            if (idx_blk > w_blk) /* after the first row and first column */
                ms_act_prev = XPSNR_MAX(ms_act_prev, weights[idx_blk - 1 - w_blk]); /* min (L, T) */
            if ((idx_blk > 0) && (weights[idx_blk - 1] > ms_act_prev))
                weights[idx_blk - 1] = ms_act_prev;

            if ((x + b >= s->w) && (y + b >= s->h) && (idx_blk > w_blk)) { /* last block in picture */
                ms_act_prev = XPSNR_MAX(weights[idx_blk - 1], weights[idx_blk - w_blk]);
                if (weights[idx_blk] > ms_act_prev)
                    weights[idx_blk] = ms_act_prev;
            }
        }
    }
}

/* FFmpeg's get_wsse for the luma plane; *sse, the plane's sum of squared
 * errors. */
static uint64_t wsse_luma(XpsnrState *s, const XpsnrPlanes *p, uint64_t *sse)
{
    const uint32_t b = s->b, w = s->w, h = s->h;

    if (b < 4) { /* picture is too small for XPSNR, calculate nonweighted PSNR */
        *sse = s->bpc == 8 ? sse_block_8(p->org, p->so, p->rec, p->sr, w, h)
                           : sse_block_16(p->org, p->so, p->rec, p->sr, w, h);
        return *sse;
    }

    double *const sse_luma = s->sse_luma;
    double *const weights = s->weights;
    uint32_t idx_blk = 0;
    *sse = 0;

    for (uint32_t y = 0; y < h; y += b) { /* calculate block SSE and perceptual weights */
        const uint32_t block_height = (y + b > h ? h - y : b);

        for (uint32_t x = 0; x < w; x += b, idx_blk++) {
            const uint32_t block_width = (x + b > w ? w - x : b);
            double ms_act = 1.0;

            const uint64_t block_sse = s->bpc == 8
                ? block_8(s, p, x, y, block_width, block_height, &ms_act)
                : block_16(s, p, x, y, block_width, block_height, &ms_act);
            sse_luma[idx_blk] = (double) block_sse;
            *sse += block_sse;
            weights[idx_blk] = 1.0 / sqrt(ms_act);
        }
    }
    if (w * h <= 640 * 480)
        smooth_weights(s, weights);

    double wsse = 0.0;
    for (uint32_t i = 0; i < idx_blk; i++) /* calculate sum for luma (Y) XPSNR */
        wsse += sse_luma[i] * weights[i];
    return (wsse <= 0.0 ? 0 : (uint64_t) (wsse * s->avg_act + 0.5));
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    XpsnrState *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;

    /* FFmpeg's original (its first input) and reconstruction */
    const VmafPicture *org = s->weights_from_dist ? dist_pic : ref_pic;
    const VmafPicture *rec = s->weights_from_dist ? ref_pic : dist_pic;
    const VmafPicture *m1 = s->weights_from_dist ? &fex->prev_dist : &fex->prev_ref;
    const VmafPicture *m2 = s->weights_from_dist ? &fex->prev_prev_dist : &fex->prev_prev_ref;
    const ptrdiff_t bytes = s->bpc > 8 ? 2 : 1;

    if (org->bpc != s->bpc || org->w[0] != s->w || org->h[0] != s->h)
        return -EINVAL;

    const XpsnrPlanes p = {
        .org = org->data[0],
        .rec = rec->data[0],
        .m1 = m1->ref ? m1->data[0] : s->zeros,
        .m2 = m2->ref ? m2->data[0] : s->zeros,
        .so = org->stride[0] / bytes,
        .sr = rec->stride[0] / bytes,
        .s1 = m1->ref ? m1->stride[0] / bytes : 0,
        .s2 = m2->ref ? m2->stride[0] / bytes : 0,
    };

    uint64_t sse;
    const uint64_t wsse64 = wsse_luma(s, &p, &sse);
    /* FFmpeg's get_avg_xpsnr of a single frame */
    const double sqrt_wsse = sqrt((double) wsse64);
    double xpsnr = INFINITY;
    if (sqrt_wsse >= 1.0) {
        const double avg_dist = sqrt_wsse / 1.0;
        xpsnr = 10.0 * log10((double) s->num64 / ((double) avg_dist * (double) avg_dist));
    }

    int err = vmaf_feature_collector_append(feature_collector, "xpsnr_y", xpsnr, index);
    if (s->psnr && !err) {
        /* integer_psnr.c's psnr_y, its default options: every product and
         * quotient as it makes them */
        const uint32_t peak = (1u << s->bpc) - 1;
        const double mse = ((double) sse) / (s->w * s->h);
        const double psnr_max = (6 * s->bpc) + 12;
        const double psnr = XPSNR_MIN(10. * log10(peak * peak / XPSNR_MAX(mse, 1e-16)), psnr_max);
        err = vmaf_feature_collector_append(feature_collector, "psnr_y", psnr, index);
    }
    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    XpsnrState *s = fex->priv;
    free(s->sse_luma);
    free(s->weights);
    free(s->zeros);
    return 0;
}

static const char *provided_features[] = {
    "xpsnr_y",
    NULL
};

VmafFeatureExtractor vmaf_fex_xpsnr = {
    .name = "xpsnr",
    .options = options,
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(XpsnrState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_PREV_PICTURES,
};
