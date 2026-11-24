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

#include <errno.h>
#include <math.h>
#include <stddef.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#include "mem.h"
#include "ssim.h"
#include "picture_copy.h"

/* The picture planes float_ssim scores: Y, and with enable_chroma Cb and Cr
 * (libvmaf-fast), each at its own size, with its own decimation scale. */
#define SSIM_PLANES 3

typedef struct SsimState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
    bool enable_chroma;
    double max_db[SSIM_PLANES];
    int scale;
    /* libvmaf-fast: each plane's compute_ssim_decimated buffers, kept from
     * frame to frame; NULL where that plane is not decimated */
    SsimWorkspace *ws[SSIM_PLANES];
} SsimState;

static const VmafOption options[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(SsimState, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write SSIM values as dB",
        .offset = offsetof(SsimState, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(SsimState, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling, 2-10=explicit)",
        .offset = offsetof(SsimState, scale),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    {
        /* libvmaf-fast */
        .name = "enable_chroma",
        .help = "also score the chroma planes (float_ssim_cb, float_ssim_cr), "
                "each at its own size",
        .offset = offsetof(SsimState, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    { 0 }
};

static const char *const ssim_name[SSIM_PLANES] = {
    "float_ssim", "float_ssim_cb", "float_ssim_cr"
};

/* The two float pictures compute_ssim takes, once, at the luma's size: a
 * chroma plane, never larger, fits in them at the same row spacing. */
static int full_size_buffers(SsimState *s, unsigned w, unsigned h)
{
    if (s->ref)
        return 0;
    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) return -ENOMEM;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) {
        aligned_free(s->ref);
        s->ref = NULL;
        return -ENOMEM;
    }
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    SsimState *s = fex->priv;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        s->enable_chroma = false;
    const unsigned planes = s->enable_chroma ? SSIM_PLANES : 1;
    /* each plane's size, as vmaf_picture_alloc makes it */
    const int ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;

    const unsigned peak = (1 << bpc) - 1;
    unsigned pw[SSIM_PLANES], ph[SSIM_PLANES];
    int scale[SSIM_PLANES];
    int largest = -1;  /* the decimated plane with the most decimated samples */
    for (unsigned p = 0; p < planes; p++) {
        pw[p] = p ? w >> ss_hor : w;
        ph[p] = p ? h >> ss_ver : h;
        if (s->clip_db) {
            const double mse = 0.5 / (pw[p] * ph[p]);
            s->max_db[p] = ceil(10. * log10(peak * peak / mse));
        } else {
            s->max_db[p] = INFINITY;
        }
        scale[p] = ssim_scale(pw[p], ph[p], s->scale);
        if (scale[p] > 1 && (largest < 0 ||
            ssim_decimated_samples(pw[p], ph[p], scale[p]) >
                ssim_decimated_samples(pw[largest], ph[largest], scale[largest])))
            largest = p;
    }

    /* libvmaf-fast: decimated pictures need no full-size float copies, and
     * the planes, scored one after the other, share the largest one's
     * decimated floats */
    const bool decimates = bpc == 8 || bpc == 10 || bpc == 12 || bpc == 16;
    if (largest >= 0 && decimates) {
        s->ws[largest] = ssim_workspace_alloc(pw[largest], ph[largest], scale[largest],
                                              bpc, NULL);
        if (!s->ws[largest]) return -ENOMEM;
    }
    for (unsigned p = 0; p < planes; p++) {
        if (scale[p] > 1 && decimates) {
            if ((int)p == largest) continue;
            s->ws[p] = ssim_workspace_alloc(pw[p], ph[p], scale[p], bpc, s->ws[largest]);
            if (!s->ws[p]) return -ENOMEM;
        } else {
            const int err = full_size_buffers(s, w, h);
            if (err) return err;
        }
    }
    return 0;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}

/* One plane's SSIM and its luminance, contrast and structure terms. */
static int plane_ssim(SsimState *s, VmafPicture *ref_pic, VmafPicture *dist_pic,
                      int p, double *score, double *l_score, double *c_score,
                      double *s_score)
{
    /* libvmaf-fast: decimated from the samples, the same score (ssim.c) */
    const int scale = ssim_scale(ref_pic->w[p], ref_pic->h[p], s->scale);
    int err = scale > 1 ? compute_ssim_decimated(ref_pic, dist_pic, p, scale, s->ws[p],
                                                 score, l_score, c_score, s_score)
                        : -EINVAL;
    if (err != -EINVAL)
        return err;
    err = full_size_buffers(s, ref_pic->w[0], ref_pic->h[0]);
    if (err) return err;
    picture_copy(s->ref, s->float_stride, ref_pic, 0, ref_pic->bpc, p);
    picture_copy(s->dist, s->float_stride, dist_pic, 0, dist_pic->bpc, p);
    return compute_ssim(s->ref, s->dist, ref_pic->w[p], ref_pic->h[p],
                        s->float_stride, s->float_stride,
                        score, l_score, c_score, s_score, s->scale);
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    SsimState *s = fex->priv;

    (void) ref_pic_90;
    (void) dist_pic_90;

    const int planes = s->enable_chroma ? SSIM_PLANES : 1;
    for (int p = 0; p < planes; p++) {
        double score, l_score, c_score, s_score;
        int err = plane_ssim(s, ref_pic, dist_pic, p, &score, &l_score, &c_score, &s_score);
        if (err) return err;

        if (s->enable_db)
            score = convert_to_db(score, s->max_db[p]);

        err = vmaf_feature_collector_append(feature_collector, ssim_name[p],
                                            score, index);
        /* the terms of the luma, as before chroma was scored */
        if (s->enable_lcs && p == 0) {
            err |= vmaf_feature_collector_append(feature_collector, "float_ssim_l",
                                                l_score, index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ssim_c",
                                                c_score, index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ssim_s",
                                                s_score, index);
        }
        if (err) return err;
    }

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    SsimState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    for (unsigned p = 0; p < SSIM_PLANES; p++)
        ssim_workspace_free(s->ws[p]);
    return 0;
}

static const char *provided_features[] = {
    "float_ssim", "float_ssim_cb", "float_ssim_cr",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_ssim = {
    .name = "float_ssim",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(SsimState),
    .provided_features = provided_features,
};
