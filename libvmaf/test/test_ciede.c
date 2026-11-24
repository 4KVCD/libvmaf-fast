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
#include "feature/ciede.c"

static int close_enough(float a, float b)
{
    const float epsilon = 1e-9f;
    return fabs(a - b) < epsilon;
}

static const KSubArgs default_ksub = { .l = 0.65, .c = 1.0, .h = 4.0 };

static char *test_ciede()
{
    const LABColor color_1 = { .l = 0.052488625, .a = -0.587470829, .b = -8.98771572 };
    const LABColor color_2 = { .l = 0.465437293, .a = 0.386364758, .b = -12.7648535 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 2.54780269",
              close_enough(de00, 2.54780269));

    return NULL;
}

static char *test_ciede2()
{
    const LABColor color_1 = { .l = 87.156334, .a = -12.049645, .b = -1.205325 };
    const LABColor color_2 = { .l = 83.455727, .a = -9.040445, .b = -8.894289 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 4.22714281",
              close_enough(de00, 4.22714281));

    return NULL;
}

static char *test_ciede3()
{
    const LABColor color_1 = { .l = 79.718491, .a = 9.109915, .b = 13.727915 };
    const LABColor color_2 = { .l = 78.717224, .a = 7.526546, .b = 5.597448 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 4.26012468",
              close_enough(de00, 4.26012468));

    return NULL;
}

static char *test_ciede4()
{
    const LABColor color_1 = { .l = 99.205299, .a = -3.339410, .b = 1.205873 };
    const LABColor color_2 = { .l = 97.991730, .a = -2.497345, .b = 2.473533 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 1.26915979",
              close_enough(de00, 1.26915979));

    return NULL;
}

static unsigned next_random(unsigned *seed)
{
    *seed = *seed * 1103515245u + 12345u;
    return *seed >> 8;
}

/* libvmaf-fast: the SIMD dE00 (x86/ciede_avx2.c or arm64/ciede_neon.c, where
 * the CPU has AVX2 or NEON) is the scalar one but for rare last-bit
 * differences of its own atan2, sin, cos, exp and powers: colours near each
 * other (a lossy video's error), far apart, greys and identical pairs,
 * 10-bit. */
static char *test_simd_matches_scalar()
{
    CiedeState s = { 0 };
    VmafFeatureExtractor fex = { .priv = &s };
    enum { N = 20000 };
    vmaf_init_cpu();
    mu_assert("init failed", !init(&fex, VMAF_PIX_FMT_YUV444P, 10, N, 1));
    if (!s.ciede2000_simd) {  // no SIMD on this CPU
        close(&fex);
        return NULL;
    }
    unsigned seed = 1;
    for (unsigned i = 0; i < N; i++) {
        const unsigned kind = i % 8;  // 0-3 near, 4-5 far, 6 grey, 7 identical
        unsigned yuv[2][3];
        for (unsigned p = 0; p < 3; p++) {
            int x = next_random(&seed) % 1024;
            int y = kind == 4 || kind == 5 ? (int)(next_random(&seed) % 1024) : x + (int)(next_random(&seed) % 33) - 16;
            if (kind == 6 && p) x = y = 512;
            if (kind == 7) y = x;
            yuv[0][p] = x;
            yuv[1][p] = y < 0 ? 0 : y > 1023 ? 1023 : y;
        }
        for (unsigned k = 0; k < 2; k++) {
            const LABColor lab = get_lab_color(yuv[k][0], yuv[k][1], yuv[k][2], 10);
            s.row_lab[3 * k][i] = lab.l;
            s.row_lab[3 * k + 1][i] = lab.a;
            s.row_lab[3 * k + 2][i] = lab.b;
        }
    }
    s.ciede2000_simd((const float *const *)s.row_lab, N, powf(25., 7), s.row_de00);
    unsigned differ = 0;
    for (unsigned i = 0; i < N; i++) {
        const LABColor color_1 = { .l = s.row_lab[0][i], .a = s.row_lab[1][i], .b = s.row_lab[2][i] };
        const LABColor color_2 = { .l = s.row_lab[3][i], .a = s.row_lab[4][i], .b = s.row_lab[5][i] };
        const float de00 = ciede2000(color_1, color_2, default_ksub);
        if (de00 == s.row_de00[i])
            continue;
        differ++;
        mu_assert("a SIMD dE00 is more than its last bit off the scalar one",
                  fabsf(de00 - s.row_de00[i]) <= fabsf(nextafterf(de00, 2 * de00) - de00));
    }
    mu_assert("more than a few SIMD dE00 values differ from the scalar ones", differ <= 4);
    close(&fex);
    return NULL;
}

/* libvmaf-fast: a 4:2:0 picture of odd size (9x5: a SIMD four, its scalar
 * remainder, chroma 4x2) scores the mean of the scalar dE00 of every pixel,
 * the last row and column reading the last chroma row and column. */
static char *test_extract_odd_size()
{
    enum { W = 9, H = 5 };
    VmafPicture pic[2];
    unsigned seed = 7;
    for (unsigned k = 0; k < 2; k++) {
        mu_assert("alloc failed", !vmaf_picture_alloc(&pic[k], VMAF_PIX_FMT_YUV420P, 10, W, H));
        for (unsigned p = 0; p < 3; p++) {
            for (unsigned i = 0; i < pic[k].h[p]; i++) {
                uint16_t *row = (uint16_t *)pic[k].data[p] + i * (pic[k].stride[p] / 2);
                for (unsigned j = 0; j < pic[k].w[p]; j++)
                    row[j] = 64 + next_random(&seed) % 877;
            }
        }
    }
    double sum = 0.;
    for (unsigned i = 0; i < H; i++) {
        for (unsigned j = 0; j < W; j++) {
            const unsigned ci = i / 2 < pic[0].h[1] ? i / 2 : pic[0].h[1] - 1;
            const unsigned cj = j / 2 < pic[0].w[1] ? j / 2 : pic[0].w[1] - 1;
            const LABColor lab_1 = get_lab_color(sample(&pic[0], 0, i, j), sample(&pic[0], 1, ci, cj),
                                                 sample(&pic[0], 2, ci, cj), 10);
            const LABColor lab_2 = get_lab_color(sample(&pic[1], 0, i, j), sample(&pic[1], 1, ci, cj),
                                                 sample(&pic[1], 2, ci, cj), 10);
            sum += ciede2000(lab_1, lab_2, default_ksub);
        }
    }
    const double expected = 45. - 20. * log10(sum / (W * H));

    CiedeState s = { 0 };
    VmafFeatureExtractor fex = { .priv = &s };
    VmafFeatureCollector *fc;
    mu_assert("collector failed", !vmaf_feature_collector_init(&fc));
    mu_assert("init failed", !init(&fex, VMAF_PIX_FMT_YUV420P, 10, W, H));
    mu_assert("extract failed", !extract(&fex, &pic[0], NULL, &pic[1], NULL, 0, fc));
    double score;
    mu_assert("no score", !vmaf_feature_collector_get_score(fc, "ciede2000", &score, 0));
    mu_assert("wrong score", fabs(score - expected) < 1e-9);
    close(&fex);
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&pic[0]);
    vmaf_picture_unref(&pic[1]);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_ciede);
    mu_run_test(test_ciede2);
    mu_run_test(test_ciede3);
    mu_run_test(test_ciede4);
    mu_run_test(test_simd_matches_scalar);
    mu_run_test(test_extract_odd_size);
    return NULL;
}
