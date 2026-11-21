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
 * libvmaf-fast: float_ssim decimates from the picture's samples
 * (compute_ssim_decimated) instead of from float copies of it. The score and
 * its luminance, contrast and structure terms must be those of
 * picture_copy() + compute_ssim(), to the last bit, at every bit depth
 * picture_copy() converts, for scales that are and are not powers of two,
 * and for odd sizes (which compute_ssim() decimates to one column or row
 * more, mirrored); with the AVX2 decimation and the C one, and with a
 * workspace kept from call to call.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "feature/ssim.h"
#include "cpu.h"
#include "feature/picture_copy.h"
#include "mem.h"

/* A picture of noise over smooth gradients, with hard edges and extremes. */
static int make_picture(VmafPicture *pic, unsigned bpc, unsigned w, unsigned h, uint32_t seed)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, bpc, w, h);
    if (err) return err;
    const uint32_t peak = (1u << bpc) - 1;
    uint32_t state = seed * 2654435761u + 1;
    for (unsigned y = 0; y < h; y++) {
        for (unsigned x = 0; x < w; x++) {
            state = state * 1664525u + 1013904223u;
            uint32_t value = ((x * 37 + y * 11) % 512) * peak / 511;
            value = (value + (state >> 24) * peak / 1023) % (peak + 1);
            if (((x / 23) + (y / 17)) % 7 == 0)
                value = (x & 1) ? peak : 0;
            if (bpc > 8)
                ((uint16_t *) pic->data[0])[y * (pic->stride[0] / 2) + x] = (uint16_t) value;
            else
                ((uint8_t *) pic->data[0])[y * pic->stride[0] + x] = (uint8_t) value;
        }
    }
    return 0;
}

static int compare(unsigned bpc, unsigned w, unsigned h)
{
    VmafPicture ref, dis;
    if (make_picture(&ref, bpc, w, h, w + h) || make_picture(&dis, bpc, w, h, w * 3 + bpc))
        return -1;
    const int scale = ssim_scale(w, h, 0);
    const size_t stride = ALIGN_CEIL(w * sizeof(float));
    float *ref_f = aligned_malloc(stride * h, 32), *dis_f = aligned_malloc(stride * h, 32);
    picture_copy(ref_f, stride, &ref, 0, bpc, 0);
    picture_copy(dis_f, stride, &dis, 0, bpc, 0);
    double want[4];
    int err = compute_ssim(ref_f, dis_f, w, h, stride, stride, &want[0], &want[1], &want[2], &want[3], 0);
    int differs = 0;
    /* with the AVX2 code (where the CPU has it): with a workspace for the
     * call, and with one kept, used again dirty; then the C code, once */
    static const unsigned masks[] = { ~0u, 0 };
    for (int m = 0; m < 2 && !err; m++) {
        vmaf_set_cpu_flags_mask(masks[m]);
        SsimWorkspace *ws = ssim_workspace_alloc(w, h, scale, bpc);
        SsimWorkspace *const use[] = { NULL, ws, ws };
        err |= !ws;
        for (int u = m ? 2 : 0; u < 3 && !err; u++) {
            double got[4];
            err |= compute_ssim_decimated(&ref, &dis, scale, use[u], &got[0], &got[1], &got[2], &got[3]);
            differs |= memcmp(want, got, sizeof want) != 0;
        }
        ssim_workspace_free(ws);
    }
    vmaf_set_cpu_flags_mask(~0u);
    aligned_free(ref_f);
    aligned_free(dis_f);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dis);
    if (err) return -1;
    return differs;
}

static char *test_decimated_ssim_is_compute_ssims()
{
    /* every scale at 10 bits; the other depths, which change only how
     * samples are read and converted, on the integer path (scale 2) and
     * the table path (scale 3) */
    static const unsigned cases[][3] = {
        { 640, 480, 10 },   /* scale 2 */
        { 643, 481, 10 },   /* scale 2, odd */
        { 1280, 720, 10 },  /* 3 */
        { 1365, 767, 10 },  /* 3, odd */
        { 1920, 1080, 10 }, /* 4 */
        { 1921, 1081, 10 }, /* 4, odd */
        { 2560, 1280, 10 }, /* 5 */
        { 2560, 1440, 10 }, /* 6 */
        { 1920, 1800, 10 }, /* 7 */
        { 3840, 2160, 10 }, /* 8 */
        { 643, 481, 8 }, { 643, 481, 12 }, { 643, 481, 16 },
        { 1365, 767, 8 }, { 1365, 767, 12 }, { 1365, 767, 16 },
    };
    vmaf_init_cpu();
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const int result = compare(cases[i][2], cases[i][0], cases[i][1]);
        mu_assert("compute_ssim_decimated failed", result >= 0);
        mu_assert("compute_ssim_decimated is not compute_ssim's score", result == 0);
    }
    return NULL;
}

/* libvmaf-fast: compute_ssim with the AVX2 code where the CPU has AVX2
 * (VMAFx's x86/convolve_avx2.c and x86/ssim_avx2.c) gives the scalar
 * code's score to the last bit; odd sizes, so that the vector loops leave
 * columns and samples over; and pictures narrower or shorter than the 11x11
 * window, where neither has an output (the AVX2 filter asserted there), and
 * just as wide or tall (one output). */
static char *test_avx2_ssim_is_the_scalar_ssim()
{
    static const unsigned sizes[][2] = {
        { 203, 117 }, { 64, 48 }, { 9, 9 }, { 10, 64 }, { 64, 10 }, { 11, 11 }, { 11, 64 },
    };
    vmaf_init_cpu();
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        const unsigned w = sizes[i][0], h = sizes[i][1];
        VmafPicture ref, dis;
        mu_assert("alloc", !make_picture(&ref, 10, w, h, 5) && !make_picture(&dis, 10, w, h, 6));
        const size_t stride = ALIGN_CEIL(w * sizeof(float));
        float *ref_f = aligned_malloc(stride * h, 32), *dis_f = aligned_malloc(stride * h, 32);
        picture_copy(ref_f, stride, &ref, 0, 10, 0);
        picture_copy(dis_f, stride, &dis, 0, 10, 0);
        double got[4], want[4];
        int err = compute_ssim(ref_f, dis_f, w, h, stride, stride, &got[0], &got[1], &got[2], &got[3], 0);
        vmaf_set_cpu_flags_mask(0);
        err |= compute_ssim(ref_f, dis_f, w, h, stride, stride, &want[0], &want[1], &want[2], &want[3], 0);
        vmaf_set_cpu_flags_mask(~0u);
        aligned_free(ref_f);
        aligned_free(dis_f);
        vmaf_picture_unref(&ref);
        vmaf_picture_unref(&dis);
        mu_assert("compute_ssim failed", !err);
        mu_assert("the AVX2 SSIM is not the scalar SSIM", !memcmp(got, want, sizeof got));
    }
    return NULL;
}

static char *test_scale_one_and_odd_depths_are_refused()
{
    VmafPicture ref, dis;
    double score, l, c, s;
    mu_assert("alloc", !make_picture(&ref, 9, 320, 240, 1) && !make_picture(&dis, 9, 320, 240, 2));
    /* picture_copy() reads 9-bit samples as 8-bit: left to compute_ssim() */
    mu_assert("9-bit not refused", compute_ssim_decimated(&ref, &dis, 2, NULL, &score, &l, &c, &s) != 0);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dis);
    mu_assert("alloc", !make_picture(&ref, 8, 320, 240, 1) && !make_picture(&dis, 8, 320, 240, 2));
    mu_assert("scale 1 not refused", compute_ssim_decimated(&ref, &dis, 1, NULL, &score, &l, &c, &s) != 0);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dis);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_decimated_ssim_is_compute_ssims);
    mu_run_test(test_avx2_ssim_is_the_scalar_ssim);
    mu_run_test(test_scale_one_and_odd_depths_are_refused);
    return NULL;
}
