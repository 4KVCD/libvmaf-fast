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
 * libvmaf-fast: psnr_hvs scores 8 blocks at a time with AVX2. Its scores
 * must be the scalar code's to the last bit.
 */

#include <stdint.h>
#include <string.h>

#include "test.h"
#include "cpu.h"
#include "dict.h"
#include "feature/feature_collector.h"
#include "feature/feature_extractor.h"
#include "libvmaf/picture.h"

/* Noise over gradients, flat squares (blocks without variance) and hard
 * edges in every plane; the distorted picture is the reference with noise
 * of seed-dependent strength, none in places. */
static int make_picture(VmafPicture *pic, unsigned bpc, unsigned w, unsigned h, uint32_t seed,
                        int noise)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, bpc, w, h);
    if (err) return err;
    const unsigned peak = (1u << bpc) - 1;
    uint32_t state = seed * 2654435761u + 1;
    for (int p = 0; p < 3; p++) {
        for (unsigned y = 0; y < pic->h[p]; y++) {
            uint8_t *row = (uint8_t *)pic->data[p] + y * pic->stride[p];
            for (unsigned x = 0; x < pic->w[p]; x++) {
                uint32_t value = ((x * 37 + y * 11 + p * 101) % 512) * (peak + 1) / 512;
                if ((x / 24 + y / 24) % 3 == 0)
                    value = peak / 3;                   /* flat */
                else if (((x / 13) + (y / 9)) % 7 == 0)
                    value = (x & 1) ? peak : 0;         /* edges */
                state = state * 1664525u + 1013904223u;
                if (noise && (x / 40) % 4)
                    value += (state >> 24) % (unsigned)noise;
                if (value > peak)
                    value = peak;
                if (bpc > 8)
                    ((uint16_t *)row)[x] = (uint16_t)value;
                else
                    row[x] = (uint8_t)value;
            }
        }
    }
    return 0;
}

static const char *const names[] = { "psnr_hvs_y", "psnr_hvs_cb", "psnr_hvs_cr", "psnr_hvs" };

/* psnr_hvs's 4 scores of one pair; 0 on success. */
static int run_psnr_hvs(VmafPicture *ref, VmafPicture *dis, double scores[4])
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("psnr_hvs");
    VmafFeatureExtractorContext *ctx;
    if (!fex || vmaf_feature_extractor_context_create(&ctx, fex, NULL))
        return -1;
    int err = vmaf_feature_extractor_context_init(ctx, ref->pix_fmt, ref->bpc, ref->w[0], ref->h[0]);
    VmafFeatureCollector *fc = NULL;
    if (!err)
        err = vmaf_feature_collector_init(&fc);
    if (!err)
        err = vmaf_feature_extractor_context_extract(ctx, ref, NULL, dis, NULL, 0, fc);
    for (int i = 0; i < 4 && !err; i++)
        err = vmaf_feature_collector_get_score(fc, names[i], &scores[i], 0);
    if (fc)
        vmaf_feature_collector_destroy(fc);
    vmaf_feature_extractor_context_close(ctx);
    vmaf_feature_extractor_context_destroy(ctx);
    return err;
}

static char *test_avx2_psnr_hvs_is_the_scalar_psnr_hvs()
{
    /* groups of 8 blocks with blocks left over in every plane; a chroma
     * plane too narrow for one group; every bit depth psnr_hvs takes; mild
     * noise (most terms masked to 0), strong noise and none */
    static const unsigned sizes[][2] = { { 203, 117 }, { 120, 64 }, { 64, 16 } };
    static const unsigned depths[] = { 8, 10, 12 };
    static const int noises[] = { 3, 200, 0 };
    vmaf_init_cpu();
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        for (size_t d = 0; d < sizeof depths / sizeof depths[0]; d++) {
            const unsigned w = sizes[i][0], h = sizes[i][1], bpc = depths[d];
            const int noise = noises[(i + d) % 3] << (bpc - 8);
            VmafPicture ref, dis;
            mu_assert("alloc", !make_picture(&ref, bpc, w, h, 1, 0) &&
                               !make_picture(&dis, bpc, w, h, 1, noise));
            double got[4], want[4];
            int err = run_psnr_hvs(&ref, &dis, got);
            vmaf_set_cpu_flags_mask(0);
            err |= run_psnr_hvs(&ref, &dis, want);
            vmaf_set_cpu_flags_mask(~0u);
            vmaf_picture_unref(&ref);
            vmaf_picture_unref(&dis);
            mu_assert("psnr_hvs failed", !err);
            mu_assert("the AVX2 psnr_hvs is not the scalar code's", !memcmp(got, want, sizeof got));
        }
    }
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_avx2_psnr_hvs_is_the_scalar_psnr_hvs);
    return NULL;
}
