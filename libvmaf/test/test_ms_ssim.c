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
 * libvmaf-fast: MS-SSIM calculates each scale's SSIM in stripes, downscales
 * row by row with AVX2, reads scale 0 from the picture and scores the chroma
 * planes on request (enable_chroma). The stripes must give one pass's sums,
 * the row downscaling _iqa_filter_pixel's values, the AVX2 code the C code's
 * and the picture the float picture's, to the last bit, and a chroma plane
 * the score it gets as a picture's luma.
 */

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "cpu.h"
#include "dict.h"
#include "mem.h"
#include "feature/feature_collector.h"
#include "feature/feature_extractor.h"
#include "feature/iqa/decimate.h"
#include "feature/ms_ssim.h"
#include "feature/picture_copy.h"
#include "feature/ssim.h"
#include "libvmaf/picture.h"

/* Noise over smooth gradients, with hard edges, in every plane. */
static int make_picture(VmafPicture *pic, enum VmafPixelFormat fmt, unsigned bpc, unsigned w,
                        unsigned h, uint32_t seed)
{
    int err = vmaf_picture_alloc(pic, fmt, bpc, w, h);
    if (err) return err;
    uint32_t state = seed * 2654435761u + 1;
    for (int p = 0; p < 3 && pic->w[p]; p++) {
        for (unsigned y = 0; y < pic->h[p]; y++) {
            uint8_t *row = (uint8_t *)pic->data[p] + y * pic->stride[p];
            for (unsigned x = 0; x < pic->w[p]; x++) {
                state = state * 1664525u + 1013904223u;
                uint32_t value = ((x * 37 + y * 11 + p * 101) % 512) * 2 + (state >> 26);
                if (((x / 13) + (y / 9)) % 7 == 0)
                    value = (x & 1) ? 1023 : 0;
                value = value > 1023 ? 1023 : value;  /* 10 bits, scaled to bpc */
                if (bpc == 8)
                    row[x] = (uint8_t)(value >> 2);
                else
                    ((uint16_t *)row)[x] = (uint16_t)(value << (bpc - 10));
            }
        }
    }
    return 0;
}

/* Plane 0 of a 10-bit pair as floats, rows w floats apart. */
static void luma_floats(VmafPicture *ref, VmafPicture *dis, float *ref_f, float *dis_f)
{
    const int stride = ref->w[0] * sizeof(float);
    picture_copy(ref_f, stride, ref, 0, 10, 0);
    picture_copy(dis_f, stride, dis, 0, 10, 0);
}

static char *test_stripes_are_one_ssim_pass()
{
    /* more output rows than one stripe, the last stripe partial; odd sizes */
    static const unsigned sizes[][2] = { { 400, 300 }, { 401, 299 } };
    vmaf_init_cpu();
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        const unsigned w = sizes[i][0], h = sizes[i][1];
        VmafPicture ref, dis;
        mu_assert("alloc", !make_picture(&ref, VMAF_PIX_FMT_YUV420P, 10, w, h, 1) &&
                           !make_picture(&dis, VMAF_PIX_FMT_YUV420P, 10, w, h, 2));
        float *ref_f = malloc(w * h * sizeof(float)), *dis_f = malloc(w * h * sizeof(float));
        mu_assert("alloc", ref_f && dis_f);
        luma_floats(&ref, &dis, ref_f, dis_f);
        const int stride = w * sizeof(float);
        /* with the C code and, where the CPU has it, the AVX2 code */
        static const unsigned masks[] = { 0, ~0u };
        for (int m = 0; m < 2; m++) {
            vmaf_set_cpu_flags_mask(masks[m]);
            double ms, l[5], c[5], s[5], score, l0, c0, s0;
            mu_assert("compute_ms_ssim failed",
                      !compute_ms_ssim(ref_f, dis_f, w, h, stride, stride, &ms, l, c, s));
            /* scale 1: compute_ssim's Gaussian SSIM of the whole picture at once */
            mu_assert("compute_ssim failed",
                      !compute_ssim(ref_f, dis_f, w, h, stride, stride, &score, &l0, &c0, &s0, 1));
            mu_assert("scale 0's terms are not one pass's", l[0] == l0 && c[0] == c0 && s[0] == s0);
        }
        vmaf_set_cpu_flags_mask(~0u);
        free(ref_f);
        free(dis_f);
        vmaf_picture_unref(&ref);
        vmaf_picture_unref(&dis);
    }
    return NULL;
}

static char *test_avx2_ms_ssim_is_the_c_ms_ssim()
{
    /* the smallest size, odd sizes at every scale, and AVX2 row ends */
    static const unsigned sizes[][2] = { { 176, 176 }, { 177, 181 }, { 203, 199 }, { 357, 241 } };
    vmaf_init_cpu();
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        const unsigned w = sizes[i][0], h = sizes[i][1];
        VmafPicture ref, dis;
        mu_assert("alloc", !make_picture(&ref, VMAF_PIX_FMT_YUV420P, 10, w, h, 3) &&
                           !make_picture(&dis, VMAF_PIX_FMT_YUV420P, 10, w, h, 4));
        float *ref_f = malloc(w * h * sizeof(float)), *dis_f = malloc(w * h * sizeof(float));
        mu_assert("alloc", ref_f && dis_f);
        luma_floats(&ref, &dis, ref_f, dis_f);
        const int stride = w * sizeof(float);
        double got[16], want[16];  /* the score, then l, c and s of the 5 scales */
        int err = compute_ms_ssim(ref_f, dis_f, w, h, stride, stride, &got[0], &got[1], &got[6], &got[11]);
        vmaf_set_cpu_flags_mask(0);
        err |= compute_ms_ssim(ref_f, dis_f, w, h, stride, stride, &want[0], &want[1], &want[6], &want[11]);
        vmaf_set_cpu_flags_mask(~0u);
        free(ref_f);
        free(dis_f);
        vmaf_picture_unref(&ref);
        vmaf_picture_unref(&dis);
        mu_assert("compute_ms_ssim failed", !err);
        mu_assert("the AVX2 MS-SSIM is not the C code's", !memcmp(got, want, sizeof got));
    }
    return NULL;
}

/* A float picture's rows for _iqa_decimate2_rows, copied into scratch (as
 * a converted picture's are). */
struct test_rows {
    const float *img;
    int w;
};

static const float *test_row(void *ctx, int r, float *scratch)
{
    const struct test_rows *t = ctx;
    memcpy(scratch, t->img + (size_t)r * t->w, t->w * sizeof(float));
    return scratch;
}

static char *test_row_decimation_is_the_scalar_decimation()
{
    /* the smallest pictures each kernel takes, odd and even sides, AVX2
     * groups of 16 and 8 with and without a remainder */
    static const int sizes[][2] = { { 9, 9 }, { 10, 11 }, { 15, 15 }, { 40, 23 }, { 61, 38 }, { 177, 9 } };
    static const int sides[] = { 3, 9, 15 };
    vmaf_init_cpu();
    uint32_t state = 7;
    float kernel[15 * 15];
    for (int i = 0; i < 15 * 15; i++) {
        state = state * 1664525u + 1013904223u;
        kernel[i] = (float)(state >> 8) / (float)(1 << 24) - 0.3f;
    }
    for (size_t s = 0; s < sizeof sides / sizeof sides[0]; s++) {
        struct _kernel k = { .kernel = kernel, .w = sides[s], .h = sides[s], .normalized = 1,
                             .bnd_opt = KBND_SYMMETRIC };
        for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
            const int w = sizes[i][0], h = sizes[i][1];
            if (w < k.w || h < k.h)
                continue;
            const int sw = w / 2 + (w & 1), sh = h / 2 + (h & 1);
            float *img = malloc(w * h * sizeof(float));
            float *want = malloc(sw * sh * sizeof(float)), *got = malloc(sw * sh * sizeof(float));
            mu_assert("alloc", img && want && got);
            for (int j = 0; j < w * h; j++) {
                state = state * 1664525u + 1013904223u;
                img[j] = (float)(state >> 16) / 64.0f;
            }
            for (int y = 0; y < sh; y++)
                for (int x = 0; x < sw; x++)
                    want[y * sw + x] = _iqa_filter_pixel(img, w, h, 2 * x, 2 * y, &k, 1.0f);
            struct test_rows rows = { img, w };
            static const unsigned masks[] = { 0, ~0u };
            int same = 1, err = 0;
            for (int m = 0; m < 2; m++) {
                vmaf_set_cpu_flags_mask(masks[m]);
                int rw = 0, rh = 0;
                err |= _iqa_decimate2_rows(test_row, &rows, w, h, &k, got, &rw, &rh);
                err |= rw != sw || rh != sh;
                same &= !memcmp(got, want, sw * sh * sizeof(float));
            }
            vmaf_set_cpu_flags_mask(~0u);
            free(img);
            free(want);
            free(got);
            mu_assert("_iqa_decimate2_rows failed", !err);
            mu_assert("the row decimation is not _iqa_filter_pixel's", same);
        }
    }
    return NULL;
}

static char *test_picture_path_is_the_float_path()
{
    /* the smallest size, odd sizes at every scale, a width that is not a
     * multiple of the AVX2 groups; every bit depth picture_copy converts */
    static const unsigned sizes[][2] = { { 176, 176 }, { 177, 181 }, { 357, 241 } };
    static const unsigned depths[] = { 8, 10, 12, 16 };
    vmaf_init_cpu();
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        for (size_t d = 0; d < sizeof depths / sizeof depths[0]; d++) {
            const unsigned w = sizes[i][0], h = sizes[i][1], bpc = depths[d];
            VmafPicture ref, dis;
            mu_assert("alloc", !make_picture(&ref, VMAF_PIX_FMT_YUV420P, bpc, w, h, 7) &&
                               !make_picture(&dis, VMAF_PIX_FMT_YUV420P, bpc, w, h, 8));
            /* rows 3 floats longer than the picture: the float path reads
             * them row by row too */
            const unsigned fw = w + 3;
            float *ref_f = malloc(fw * h * sizeof(float)), *dis_f = malloc(fw * h * sizeof(float));
            MsSsimWorkspace *ws = ms_ssim_workspace_alloc(w, h);
            mu_assert("alloc", ref_f && dis_f && ws);
            picture_copy(ref_f, w * sizeof(float), &ref, 0, bpc, 0);
            picture_copy(dis_f, w * sizeof(float), &dis, 0, bpc, 0);
            double want[16], got[16];  /* the score, then l, c and s of the 5 scales */
            int err = compute_ms_ssim(ref_f, dis_f, w, h, w * sizeof(float), w * sizeof(float),
                                      &want[0], &want[1], &want[6], &want[11]);
            int same = 1;
            static const unsigned masks[] = { 0, ~0u };
            for (int m = 0; m < 2; m++) {
                vmaf_set_cpu_flags_mask(masks[m]);
                err |= compute_ms_ssim_picture(&ref, &dis, 0, ws, &got[0], &got[1], &got[6], &got[11]);
                same &= !memcmp(got, want, sizeof got);
            }
            vmaf_set_cpu_flags_mask(~0u);
            picture_copy(ref_f, fw * sizeof(float), &ref, 0, bpc, 0);
            picture_copy(dis_f, fw * sizeof(float), &dis, 0, bpc, 0);
            err |= compute_ms_ssim_ws(ref_f, dis_f, w, h, fw * sizeof(float), fw * sizeof(float), ws,
                                      &got[0], &got[1], &got[6], &got[11]);
            same &= !memcmp(got, want, sizeof got);
            ms_ssim_workspace_free(ws);
            free(ref_f);
            free(dis_f);
            vmaf_picture_unref(&ref);
            vmaf_picture_unref(&dis);
            mu_assert("MS-SSIM failed", !err);
            mu_assert("MS-SSIM of the picture or of strided floats is not of the floats", same);
        }
    }
    return NULL;
}

/* float_ms_ssim's scores of one pair: feature names[i] into scores[i].
 * Returns the extractor's init error, or -1 for any other failure. */
static int run_float_ms_ssim(VmafPicture *ref, VmafPicture *dis, int enable_chroma,
                             const char *const *names, double *scores, int count)
{
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("float_ms_ssim");
    VmafDictionary *opts = NULL;
    if (!fex || vmaf_dictionary_set(&opts, "enable_chroma", enable_chroma ? "true" : "false", 0))
        return -1;
    VmafFeatureExtractorContext *ctx;
    if (vmaf_feature_extractor_context_create(&ctx, fex, opts))
        return -1;
    int err = vmaf_feature_extractor_context_init(ctx, ref->pix_fmt, ref->bpc, ref->w[0], ref->h[0]);
    if (!err) {
        VmafFeatureCollector *fc;
        err = vmaf_feature_collector_init(&fc) ? -1 : 0;
        if (!err) {
            if (vmaf_feature_extractor_context_extract(ctx, ref, NULL, dis, NULL, 0, fc))
                err = -1;
            for (int i = 0; i < count && !err; i++)
                err = vmaf_feature_collector_get_score(fc, names[i], &scores[i], 0) ? -1 : 0;
            vmaf_feature_collector_destroy(fc);
        }
        vmaf_feature_extractor_context_close(ctx);
    }
    vmaf_feature_extractor_context_destroy(ctx);
    return err;
}

/* A 10-bit 4:0:0 picture whose luma is the given plane of pic. */
static int plane_as_luma(VmafPicture *pic, int plane, VmafPicture *out)
{
    int err = vmaf_picture_alloc(out, VMAF_PIX_FMT_YUV400P, 10, pic->w[plane], pic->h[plane]);
    if (err) return err;
    for (unsigned y = 0; y < pic->h[plane]; y++)
        memcpy((uint8_t *)out->data[0] + y * out->stride[0],
               (uint8_t *)pic->data[plane] + y * pic->stride[plane], pic->w[plane] * 2);
    return 0;
}

static char *test_chroma_scores_are_their_planes_as_luma()
{
    vmaf_init_cpu();
    VmafPicture ref, dis;
    mu_assert("alloc", !make_picture(&ref, VMAF_PIX_FMT_YUV420P, 10, 354, 352, 5) &&
                       !make_picture(&dis, VMAF_PIX_FMT_YUV420P, 10, 354, 352, 6));
    static const char *const all[] = { "float_ms_ssim", "float_ms_ssim_cb", "float_ms_ssim_cr" };
    double with_chroma[3], luma_alone;
    mu_assert("float_ms_ssim with chroma failed",
              !run_float_ms_ssim(&ref, &dis, 1, all, with_chroma, 3));
    mu_assert("float_ms_ssim failed", !run_float_ms_ssim(&ref, &dis, 0, all, &luma_alone, 1));
    mu_assert("the luma's score changes with enable_chroma", with_chroma[0] == luma_alone);

    for (int p = 1; p < 3; p++) {
        VmafPicture ref_p, dis_p;
        mu_assert("alloc", !plane_as_luma(&ref, p, &ref_p) && !plane_as_luma(&dis, p, &dis_p));
        double as_luma;
        const int err = run_float_ms_ssim(&ref_p, &dis_p, 1, all, &as_luma, 1);
        vmaf_picture_unref(&ref_p);
        vmaf_picture_unref(&dis_p);
        mu_assert("float_ms_ssim of a 4:0:0 picture failed", !err);
        mu_assert("a chroma plane's score is not its score as a luma", with_chroma[p] == as_luma);
    }
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dis);

    /* 175 x 176 chroma planes are too small for the 5 scales: init fails */
    mu_assert("alloc", !make_picture(&ref, VMAF_PIX_FMT_YUV420P, 10, 350, 352, 5) &&
                       !make_picture(&dis, VMAF_PIX_FMT_YUV420P, 10, 350, 352, 6));
    double unused;
    const int err = run_float_ms_ssim(&ref, &dis, 1, all, &unused, 1);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dis);
    mu_assert("too small chroma planes are not refused at init", err == -EINVAL);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_stripes_are_one_ssim_pass);
    mu_run_test(test_avx2_ms_ssim_is_the_c_ms_ssim);
    mu_run_test(test_row_decimation_is_the_scalar_decimation);
    mu_run_test(test_picture_path_is_the_float_path);
    mu_run_test(test_chroma_scores_are_their_planes_as_luma);
    return NULL;
}
