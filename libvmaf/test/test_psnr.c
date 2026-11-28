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

#include "test.h"
#include "feature/integer_psnr.c"

#define EPS 0.00001

static int almost_equal(double a, double b) {
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

static int get_picture_16b(VmafPicture *pic, int pic_index)
{
    int count = 0;
    uint16_t sample_pic[2][16] = {
        {0, 0, 0, 0, 0, 0},
        {65535, 65535, 65535, 65535, 65535, 65535},
    };

    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, 16, 2, 2);
    if (err) return err;
    for (int c = 0; c < 3; c++) {
        uint16_t *data = (uint16_t *) pic->data[c];
        int stride = pic->stride[c] >> 1;
        for (unsigned i = 0; i < pic->h[c]; i++) {
            for (unsigned j = 0; j < pic->w[c]; j++) {
                data[i * stride + j] = sample_pic[pic_index][count++];
            }
        }
    }
    return 0;
}

static char *test_16b_large_diff()
{
    VmafPicture pic1, pic2;
    int err = 0;
    err |= get_picture_16b(&pic1, 0);
    err |= get_picture_16b(&pic2, 1);
    mu_assert("test_16b_large_diff alloc error", !err);
    VmafFeatureCollector *fc;
    err |= vmaf_feature_collector_init(&fc);
    mu_assert("test_16b_large_diff vmaf_feature_collector_init error", !err);
    PsnrState psnr_state = {
        .enable_chroma = 1,
        .enable_luma = 1,
        .enable_mse = 1,
        .enable_apsnr = 0,
        .peak = 65535,
    };

    err |= psnr_hbd(&pic1, &pic2, 0, fc, &psnr_state);
    mu_assert("failed psnr_hbd", err == 0);

    double psnr_y, psnr_cb, psnr_cr;
    err |= vmaf_feature_collector_get_score(fc, "psnr_y", &psnr_y, 0);
    err |= vmaf_feature_collector_get_score(fc, "psnr_cb", &psnr_cb, 0);
    err |= vmaf_feature_collector_get_score(fc, "psnr_cr", &psnr_cr, 0);
    double mse_y, mse_cb, mse_cr;
    err |= vmaf_feature_collector_get_score(fc, "mse_y", &mse_y, 0);
    err |= vmaf_feature_collector_get_score(fc, "mse_cb", &mse_cb, 0);
    err |= vmaf_feature_collector_get_score(fc, "mse_cr", &mse_cr, 0);
    mu_assert("test_16b_large_diff vmaf_feature_collector_get_score error", !err);

    mu_assert("wrong psnr_y", almost_equal(psnr_y, 0.0));
    mu_assert("wrong psnr_cb", almost_equal(psnr_cb, 0.0));
    mu_assert("wrong psnr_cr", almost_equal(psnr_cr, 0.0));
    mu_assert("wrong mse_y", almost_equal(mse_y, 4294836225.0));
    mu_assert("wrong mse_cb", almost_equal(mse_cb, 4294836225.0));
    mu_assert("wrong mse_cr", almost_equal(mse_cr, 4294836225.0));
    
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&pic1);
    vmaf_picture_unref(&pic2);

    return NULL;
}


/* libvmaf-fast: the rows init picks (VMAFx's AVX2 ones where the CPU has
 * AVX2, and for 16-bit samples of at most 12 bits libvmaf-fast's) sum every
 * width to the scalar rows' value, for random errors and the largest, at
 * each bit depth: short rows for the tails, and rows past the 12-bit rows'
 * 1024 samples between widenings, 4K's 3840 among them. */
static char *test_rows_init_picks_are_the_scalar_rows()
{
    enum { N = 4100 };
    static uint8_t r8[N], d8[N];
    static uint16_t r16[N], d16[N];
    static const unsigned depths[] = { 8, 10, 12, 16 };
    static const unsigned long_rows[] = { 1023, 1024, 1025, 2047, 2048, 3840, N };
    uint32_t state = 1;
    vmaf_init_cpu();
    for (size_t d = 0; d < sizeof depths / sizeof depths[0]; d++) {
        const unsigned bpc = depths[d];
        const uint16_t peak = (uint16_t)((1u << bpc) - 1);
        PsnrState s = { .enable_luma = true };
        VmafFeatureExtractor fex = { .priv = &s };
        mu_assert("init failed", !init(&fex, VMAF_PIX_FMT_YUV400P, bpc, 64, 64));
        for (int pass = 0; pass < 2; pass++) {
            for (unsigned j = 0; j < N; j++) {
                state = state * 1664525u + 1013904223u;
                r8[j] = pass ? (j & 1) * 255 : (uint8_t) (state >> 24);
                d8[j] = pass ? 255 - r8[j] : (uint8_t) (state >> 16);
                r16[j] = pass ? (j & 1) * peak : (uint16_t) (state >> 16) & peak;
                d16[j] = pass ? peak - r16[j] : (uint16_t) state & peak;
            }
            for (unsigned i = 0; i < 100 + sizeof long_rows / sizeof long_rows[0]; i++) {
                const unsigned w = i < 100 ? i + 1 : long_rows[i - 100];
                if (bpc == 8)
                    mu_assert("an 8-bit row is not the scalar row's",
                              s.sse_line_8(r8, d8, w) == sse_line_8_c(r8, d8, w));
                else
                    mu_assert("a 16-bit row is not the scalar row's",
                              s.sse_line_16(r16, d16, w) == sse_line_16_c(r16, d16, w));
            }
        }
    }
    return NULL;
}

/* libvmaf-fast: without the luma (enable_luma off, where xpsnr gives psnr_y)
 * only psnr_cb and psnr_cr are written, with the values they have beside it;
 * and a psnr with no plane to score is refused. */
static char *test_chroma_without_luma()
{
    VmafPicture pic1, pic2;
    int err = get_picture_16b(&pic1, 0) | get_picture_16b(&pic2, 1);
    mu_assert("alloc error", !err);
    VmafFeatureCollector *fc;
    mu_assert("collector error", !vmaf_feature_collector_init(&fc));
    PsnrState s = { .enable_chroma = 1, .enable_luma = 0, .peak = 65535 };
    mu_assert("failed psnr_hbd", psnr_hbd(&pic1, &pic2, 0, fc, &s) == 0);
    double score;
    mu_assert("psnr_y written", vmaf_feature_collector_get_score(fc, "psnr_y", &score, 0) != 0);
    mu_assert("no psnr_cb", !vmaf_feature_collector_get_score(fc, "psnr_cb", &score, 0));
    mu_assert("wrong psnr_cb", almost_equal(score, 0.0));
    mu_assert("no psnr_cr", !vmaf_feature_collector_get_score(fc, "psnr_cr", &score, 0));
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&pic1);
    vmaf_picture_unref(&pic2);

    PsnrState none = { .enable_chroma = 1, .enable_luma = 0 };
    VmafFeatureExtractor fex = { .priv = &none };
    mu_assert("no plane not refused", init(&fex, VMAF_PIX_FMT_YUV400P, 8, 64, 64) == -EINVAL);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_16b_large_diff);
    mu_run_test(test_chroma_without_luma);
    mu_run_test(test_rows_init_picks_are_the_scalar_rows);

    return NULL;
}