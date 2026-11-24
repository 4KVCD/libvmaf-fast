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

#include "log.h"
#include "mem.h"
#include "ms_ssim.h"
#include "picture_copy.h"

/* The picture planes float_ms_ssim scores: Y, and with enable_chroma Cb and
 * Cr (libvmaf-fast), each at its own size. */
#define MS_SSIM_PLANES 3

typedef struct MsSsimState {
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
    bool enable_chroma;
    double max_db[MS_SSIM_PLANES];
    /* libvmaf-fast: compute_ms_ssim's buffers, kept from frame to frame */
    MsSsimWorkspace *ws;
} MsSsimState;

static const VmafOption options[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(MsSsimState, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write MS-SSIM values as dB",
        .offset = offsetof(MsSsimState, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(MsSsimState, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        /* libvmaf-fast */
        .name = "enable_chroma",
        .help = "also score the chroma planes (float_ms_ssim_cb, float_ms_ssim_cr), "
                "each at its own size",
        .offset = offsetof(MsSsimState, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    { 0 }
};

static const char *const ms_ssim_name[MS_SSIM_PLANES] = {
    "float_ms_ssim", "float_ms_ssim_cb", "float_ms_ssim_cr"
};

static void free_buffers(MsSsimState *s)
{
    ms_ssim_workspace_free(s->ws);
    s->ws = NULL;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    MsSsimState *s = fex->priv;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        s->enable_chroma = false;
    const unsigned planes = s->enable_chroma ? MS_SSIM_PLANES : 1;
    /* each plane's size, as vmaf_picture_alloc makes it */
    const int ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;

    const unsigned peak = (1 << bpc) - 1;
    for (unsigned p = 0; p < planes; p++) {
        const unsigned pw = p ? w >> ss_hor : w;
        const unsigned ph = p ? h >> ss_ver : h;
        /* libvmaf-fast: a chroma plane too small for the 5 scales fails here
         * (the luma fails at its first frame, as it always has) */
        if (p && !ms_ssim_size_fits(pw, ph)) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "%s: the %ux%u chroma planes are smaller than MS-SSIM's 176x176\n",
                     fex->name, pw, ph);
            return -EINVAL;
        }
        if (s->clip_db) {
            const double mse = 0.5 / (pw * ph);
            s->max_db[p] = ceil(10. * log10(peak * peak / mse));
        } else {
            s->max_db[p] = INFINITY;
        }
    }

    /* the planes are read from the pictures (compute_ms_ssim_picture); a
     * chroma plane fits in the luma's buffers */
    s->ws = ms_ssim_workspace_alloc(w, h);
    if (!s->ws)
        return -ENOMEM;
    return 0;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}

/* The 5 scales' luminance, contrast and structure terms of the luma. */
static int append_lcs(VmafFeatureCollector *feature_collector, unsigned index,
                      const double l_scores[5], const double c_scores[5],
                      const double s_scores[5])
{
    static const char *const l_name[5] = {
        "float_ms_ssim_l_scale0", "float_ms_ssim_l_scale1", "float_ms_ssim_l_scale2",
        "float_ms_ssim_l_scale3", "float_ms_ssim_l_scale4",
    };
    static const char *const c_name[5] = {
        "float_ms_ssim_c_scale0", "float_ms_ssim_c_scale1", "float_ms_ssim_c_scale2",
        "float_ms_ssim_c_scale3", "float_ms_ssim_c_scale4",
    };
    static const char *const s_name[5] = {
        "float_ms_ssim_s_scale0", "float_ms_ssim_s_scale1", "float_ms_ssim_s_scale2",
        "float_ms_ssim_s_scale3", "float_ms_ssim_s_scale4",
    };
    int err = 0;
    for (int i = 0; i < 5; i++)
        err |= vmaf_feature_collector_append(feature_collector, l_name[i], l_scores[i], index);
    for (int i = 0; i < 5; i++)
        err |= vmaf_feature_collector_append(feature_collector, c_name[i], c_scores[i], index);
    for (int i = 0; i < 5; i++)
        err |= vmaf_feature_collector_append(feature_collector, s_name[i], s_scores[i], index);
    return err;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MsSsimState *s = fex->priv;

    (void) ref_pic_90;
    (void) dist_pic_90;

    const int planes = s->enable_chroma ? MS_SSIM_PLANES : 1;
    for (int p = 0; p < planes; p++) {
        double score, l_scores[5], c_scores[5], s_scores[5];
        int err = compute_ms_ssim_picture(ref_pic, dist_pic, p, s->ws,
                                          &score, l_scores, c_scores, s_scores);
        if (err) return err;

        if (s->enable_db)
            score = convert_to_db(score, s->max_db[p]);

        err = vmaf_feature_collector_append(feature_collector, ms_ssim_name[p],
                                            score, index);
        /* the terms of the luma, as before chroma was scored */
        if (s->enable_lcs && p == 0)
            err |= append_lcs(feature_collector, index, l_scores, c_scores, s_scores);
        if (err) return err;
    }

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    free_buffers(fex->priv);
    return 0;
}

static const char *provided_features[] = {
    "float_ms_ssim", "float_ms_ssim_cb", "float_ms_ssim_cr",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_ms_ssim = {
    .name = "float_ms_ssim",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(MsSsimState),
    .provided_features = provided_features,
};
