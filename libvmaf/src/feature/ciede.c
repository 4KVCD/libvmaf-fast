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

// This is in large part a port of the ciede2000 implementation from av-metrics
// (https://github.com/rust-av/av-metrics) which has the following license:

/*
The MIT License (MIT)
Copyright (c) 2019 Joshua Holmer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "mem.h"
#include "opt.h"
#if ARCH_X86
#include "x86/ciede_avx2.h"
#elif ARCH_AARCH64
#include "arm64/ciede_neon.h"
#endif

// The Lab colours of recently seen (y, u, v) triplets, direct-mapped: on
// real video 62 to 91 % of the conversions repeat (4K 10-bit and 1080p 8-bit
// x265/x264 encodes), and get_lab_color is a third of a pixel's time. An
// entry holds its full key, so a hit gives exactly get_lab_color's result.
// 2^16 entries (1.5 MB) per extractor context, one per libvmaf thread.
enum { LAB_CACHE_BITS = 16 };

typedef struct LabCacheEntry {
    uint64_t key;  // y | u << 16 | v << 32 | 1 << 48 (the 1 marks it used)
    float l, a, b;
} LabCacheEntry;

typedef struct CiedeState {
    // 4:2:2 pictures, upscaled to 4:4:4 (scale_chroma_planes); 4:2:0 and
    // 4:4:4 are read in place.
    VmafPicture ref;
    VmafPicture dist;
    void (*scale_chroma_planes)(VmafPicture *in, VmafPicture *out);
    LabCacheEntry *lab_cache;
    // A row's Lab colours (the reference's L, a, b, then the distorted
    // video's) and its dE00 values, w floats each.
    float *row_lab[6];
    float *row_de00;
    // dE00 of a row's whole fours of pixels where the CPU has SIMD
    // (x86/ciede_avx2.c, arm64/ciede_neon.c), else NULL; the scalar
    // ciede2000 does the rest.
    void (*ciede2000_simd)(const float *const lab[6], unsigned n, float k25_7, float *de00);
} CiedeState;

static void scale_chroma_planes_hbd(VmafPicture *in, VmafPicture *out)
{
    const int ss_hor = in->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = in->pix_fmt == VMAF_PIX_FMT_YUV420P;

    for (unsigned p = 0; p < 3; p++) {
        uint16_t *in_buf = in->data[p];
        uint16_t *out_buf = out->data[p];
        for (unsigned i = 0; i < out->h[p]; i++) {
            for (unsigned j = 0; j < out->w[p]; j++) {
                out_buf[j] = in_buf[(j / ((p && ss_ver) ? 2 : 1))];
            }
            in_buf += (((p && ss_hor) ? i % 2 : 1) * in->stride[p]) / 2;
            out_buf += out->stride[p] / 2;
        }
    }
}

static void scale_chroma_planes(VmafPicture *in, VmafPicture *out)
{
    const int ss_hor = in->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = in->pix_fmt == VMAF_PIX_FMT_YUV420P;

    for (unsigned p = 0; p < 3; p++) {
        uint8_t *in_buf = in->data[p];
        uint8_t *out_buf = out->data[p];
        for (unsigned i = 0; i < out->h[p]; i++) {
            for (unsigned j = 0; j < out->w[p]; j++) {
                out_buf[j] = in_buf[(j / ((p && ss_ver) ? 2 : 1))];
            }
            in_buf += ((p && ss_hor) ? i % 2 : 1) * in->stride[p];
            out_buf += out->stride[p];
        }
    }
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    CiedeState *s = fex->priv;
    int err = 0;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;

    if (pix_fmt != VMAF_PIX_FMT_YUV444P) {
        switch (bpc) {
        case 8:
            s->scale_chroma_planes = scale_chroma_planes;
            break;
        case 10:
        case 12:
        case 16:
            s->scale_chroma_planes = scale_chroma_planes_hbd;
            break;
        default:
            return -EINVAL;
        }
    }

    s->lab_cache = calloc((size_t)1 << LAB_CACHE_BITS, sizeof(*s->lab_cache));
    // one block for the seven rows
    s->row_lab[0] = malloc(sizeof(float) * 7 * w);
    if (!s->lab_cache || !s->row_lab[0]) {
        // close() does not run after a failed init
        free(s->lab_cache);
        free(s->row_lab[0]);
        s->lab_cache = NULL;
        s->row_lab[0] = NULL;
        return -ENOMEM;
    }
    for (unsigned k = 1; k < 6; k++)
        s->row_lab[k] = s->row_lab[k - 1] + w;
    s->row_de00 = s->row_lab[5] + w;
    s->ciede2000_simd = NULL;
#if ARCH_X86
    if (vmaf_get_cpu_flags() & VMAF_X86_CPU_FLAG_AVX2)
        s->ciede2000_simd = ciede2000_avx2;
#elif ARCH_AARCH64
    if (vmaf_get_cpu_flags() & VMAF_ARM_CPU_FLAG_NEON)
        s->ciede2000_simd = ciede2000_neon;
#endif

    if (pix_fmt != VMAF_PIX_FMT_YUV422P)
        return 0;

    err |= vmaf_picture_alloc(&s->ref, VMAF_PIX_FMT_YUV444P, bpc, w, h);
    err |= vmaf_picture_alloc(&s->dist, VMAF_PIX_FMT_YUV444P, bpc, w, h);
    if (err) {
        if (s->ref.data[0])
            vmaf_picture_unref(&s->ref);
        if (s->dist.data[0])
            vmaf_picture_unref(&s->dist);
        free(s->lab_cache);
        free(s->row_lab[0]);
        s->lab_cache = NULL;
        s->row_lab[0] = NULL;
    }
    return err;
}

static float get_h_prime(const float x, const float y)
{
    if ((x == 0.0) && (y == 0.0))
        return 0.0;
    float hue_angle = atan2(x, y);
    if (hue_angle < 0.0)
        hue_angle += 2. * M_PI;
    return hue_angle;
}

static float get_delta_h_prime(const float c1, const float c2,
                               const float h_prime_1, const float h_prime_2)
{
    if ((c1 == 0.0) || (c2 == 0.0))
        return 0.0;
    if (fabsf(h_prime_1 - h_prime_2) <= M_PI)
        return h_prime_2 - h_prime_1;
    if (h_prime_2 <= h_prime_1)
        return h_prime_2 - h_prime_1 + 2. * M_PI;
    else
        return h_prime_2 - h_prime_1 - 2. * M_PI;

}

static float get_upcase_h_bar_prime(const float h_prime_1,
                                    const float h_prime_2)
{
    return fabs((h_prime_1 - h_prime_2)) > M_PI ?
        (h_prime_1 + h_prime_2 + 2.0 * M_PI) / 2.0 :
        (h_prime_1 + h_prime_2) / 2.0;
}

static float get_upcase_t(const float upcase_h_bar_prime)
{
    return 1.0 -
           0.17 * cos(upcase_h_bar_prime - M_PI / 6.0) +
           0.24 * cos(2.0 * upcase_h_bar_prime) +
           0.32 * cos(3.0 * upcase_h_bar_prime + M_PI / 30.0) -
           0.20 * cos(4.0 * upcase_h_bar_prime - 7.0 * M_PI / 20.0);
}

static float radians_to_degrees(const float radians)
{
    return radians * (180.0 / M_PI);
}

static float degrees_to_radians(const float degrees)
{
    return degrees * (M_PI / 180.0);
}

static float get_r_sub_t(const float c_bar_prime,
                         const float upcase_h_bar_prime)
{
    const float degrees =
        (radians_to_degrees(upcase_h_bar_prime) - 275.0) * (1.0 / 25.0);

    return -2.0 *
          sqrt(powf(c_bar_prime, 7) / (powf(c_bar_prime, 7) + powf(25., 7))) *
          sin(degrees_to_radians(60.0 * exp(-(powf(degrees, 2)))));
}

typedef struct LABColor {
    const float l;
    const float a;
    const float b;
} LABColor;

typedef struct KSubArgs {
    const float l;
    const float c;
    const float h;
} KSubArgs;

// The squares below were pow(x, 2) of floats: a float's square fits a
// double's 53-bit mantissa exactly, and pow (faithful in every C library)
// returns an exactly representable result exactly, so each product is the
// same double as the pow call, at a fraction of its time (ciede2000 1.9x
// faster, identical on 6 million colour pairs). Contraction is off in them:
// clang would otherwise fuse x * x + y * y into an FMA where FMA is native
// (Arm), and the sum would no longer be the two pow results added.
static double square_sum_2(double x, double y)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    return x * x + y * y;
}

static double square_sum_3(double x, double y, double z)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    return x * x + y * y + z * z;
}

static float ciede2000(LABColor color_1, LABColor color_2, KSubArgs ksub)
{
    const float delta_l_prime = color_2.l - color_1.l;
    const float l_bar = (color_1.l + color_2.l) / 2;
    const float c1 = sqrt(square_sum_2(color_1.a, color_1.b));
    const float c2 = sqrt(square_sum_2(color_2.a, color_2.b));
    const float c_bar = (c1 + c2) / 2;
    const double c_bar_7 = pow(c_bar, 7);
    const double g = 1 - sqrt(c_bar_7 / (c_bar_7 + pow(25, 7)));
    const float a_prime_1 = color_1.a + (color_1.a / 2) * g;
    const float a_prime_2 = color_2.a + (color_2.a / 2) * g;
    const float c_prime_1 = sqrt(square_sum_2(a_prime_1, color_1.b));
    const float c_prime_2 = sqrt(square_sum_2(a_prime_2, color_2.b));
    const float c_bar_prime = (c_prime_1 + c_prime_2) / 2;
    const float delta_c_prime = c_prime_2 - c_prime_1;
    const float l_offset = l_bar - 50;
    const double l_offset_2 = (double)l_offset * l_offset;
    const float s_sub_l = 1. + ((0.015 * l_offset_2) / sqrt(20 + l_offset_2));
    const float s_sub_c = 1. + 0.045 * c_bar_prime;
    const float h_prime_1 = get_h_prime(color_1.b, a_prime_1);
    const float h_prime_2 = get_h_prime(color_2.b, a_prime_2);
    const float delta_h_prime = get_delta_h_prime(c1, c2, h_prime_1, h_prime_2);
    const float delta_upcase_h_prime =
            2.0 * sqrt(c_prime_1 * c_prime_2) * sin(delta_h_prime / 2.0);
    const float upcase_h_bar_prime =
        get_upcase_h_bar_prime(h_prime_1, h_prime_2);
    const float upcase_t = get_upcase_t(upcase_h_bar_prime);
    const float s_sub_upcase_h = 1.0 + 0.015 * c_bar_prime * upcase_t;
    const float r_sub_t = get_r_sub_t(c_bar_prime, upcase_h_bar_prime);
    const float lightness = delta_l_prime / (ksub.l * s_sub_l);
    const float chroma  = delta_c_prime / (ksub.c * s_sub_c);
    const float hue = delta_upcase_h_prime / (ksub.h * s_sub_upcase_h);

    return sqrt(square_sum_3(lightness, chroma, hue) + r_sub_t * chroma * hue);
}

static double pow_2_4(double x)
{
    return pow(x, 2.4);
}

static double rgb_to_xyz_map(double c)
{
    if (c > 10. / 255.) {
        const double A = 0.055;
        const double D = 1.0 / 1.055;
        return pow_2_4((c + A) * D);
    } else {
        const double D = 1.0 / 12.92;
        return (c * D);
    }
}

static double cbrt_approx(double c)
{
    return pow(c, 1.0 / 3.0);
}

static float xyz_to_lab_map(double c)
{
    const double KAPPA = 24389.0 / 27.0;
    const double EPSILON = 216.0 / 24389.0;

    if (c > EPSILON) {
        return cbrt_approx(c);
    } else {
        return (KAPPA * c + 16.0) * (1.0 / 116.0);
    }
}

static LABColor get_lab_color(double y, double u, double v, unsigned bpc)
{
    const double scale = 1 << (bpc - 8);

    y = (y - 16.  * scale) * (1. / (219. * scale));
    u = (u - 128. * scale) * (1. / (224. * scale));
    v = (v - 128. * scale) * (1. / (224. * scale));

    // Assumes BT.709
    double r = y + 1.28033 * v;
    double g = y - 0.21482 * u - 0.38059 * v;
    double b = y + 2.12798 * u;

    r = rgb_to_xyz_map(r);
    g = rgb_to_xyz_map(g);
    b = rgb_to_xyz_map(b);

    double x = r * 0.4124564390896921 + g * 0.357576077643909 +
              b * 0.18043748326639894;
          y = r * 0.21267285140562248 + g * 0.715152155287818 +
              b * 0.07217499330655958;
    double z = r * 0.019333895582329317 + g * 0.119192025881303 +
              b * 0.9503040785363677;

    x = xyz_to_lab_map(x * (1.0 / 0.95047));
    y = xyz_to_lab_map(y);
    z = xyz_to_lab_map(z * (1.0 / 1.08883));

    LABColor lab_color = {
        .l = (116.0 * y) - 16.0,
        .a = 500.0 * (x - y),
        .b = 200.0 * (y - z),
    };

    return lab_color;
}

// get_lab_color of the sample values (integers, as the float the extractor
// read them into always held them exactly), through the cache.
static LABColor lab_color_cached(LabCacheEntry *cache, unsigned y, unsigned u, unsigned v, unsigned bpc)
{
    const uint64_t key = y | (uint64_t)u << 16 | (uint64_t)v << 32 | (uint64_t)1 << 48;
    LabCacheEntry *entry = &cache[(key * 0x9E3779B97F4A7C15ull) >> (64 - LAB_CACHE_BITS)];
    if (entry->key != key) {
        const LABColor lab = get_lab_color(y, u, v, bpc);
        entry->key = key;
        entry->l = lab.l;
        entry->a = lab.a;
        entry->b = lab.b;
    }
    const LABColor lab = { .l = entry->l, .a = entry->a, .b = entry->b };
    return lab;
}

static unsigned min_unsigned(unsigned a, unsigned b) { return a < b ? a : b; }

static unsigned sample(const VmafPicture *pic, unsigned plane, unsigned i, unsigned j)
{
    if (pic->bpc == 8)
        return ((const uint8_t *)pic->data[plane])[i * pic->stride[plane] + j];
    return ((const uint16_t *)pic->data[plane])[i * (pic->stride[plane] / 2) + j];
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    CiedeState *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;

    VmafPicture *ref = ref_pic;
    VmafPicture *dist = dist_pic;
    // 4:2:0's chroma is read where it is, a sample for each 2x2 block of luma
    // (as the 4:4:4 copy the extractor made before held it); 4:2:2's is
    // still upscaled first, which reads its rows as 4:2:0's.
    unsigned shift = ref_pic->pix_fmt == VMAF_PIX_FMT_YUV420P;
    if (ref_pic->pix_fmt == VMAF_PIX_FMT_YUV422P) {
        ref = &s->ref;
        dist = &s->dist;
        s->scale_chroma_planes(ref_pic, ref);
        s->scale_chroma_planes(dist_pic, dist);
    }
    if (ref->bpc != 8 && ref->bpc != 10 && ref->bpc != 12 && ref->bpc != 16)
        return -EINVAL;

    const unsigned w = ref->w[0];
    const KSubArgs default_ksub = { .l = 0.65, .c = 1.0, .h = 4.0 };
    const unsigned simd_end = s->ciede2000_simd ? w & ~3u : 0;
    double de00_sum = 0.;
    for (unsigned i = 0; i < ref->h[0]; i++) {
        // An odd size's last luma row and column read the last chroma row and
        // column (the chroma planes are half the size rounded down); the
        // upscale read past the planes there.
        const unsigned ci = min_unsigned(i >> shift, ref->h[1] - 1);
        for (unsigned j = 0; j < w; j++) {
            const unsigned cj = min_unsigned(j >> shift, ref->w[1] - 1);
            const LABColor color_1 = lab_color_cached(s->lab_cache, sample(ref, 0, i, j), sample(ref, 1, ci, cj),
                                                      sample(ref, 2, ci, cj), ref->bpc);
            const LABColor color_2 = lab_color_cached(s->lab_cache, sample(dist, 0, i, j), sample(dist, 1, ci, cj),
                                                      sample(dist, 2, ci, cj), dist->bpc);
            s->row_lab[0][j] = color_1.l;
            s->row_lab[1][j] = color_1.a;
            s->row_lab[2][j] = color_1.b;
            s->row_lab[3][j] = color_2.l;
            s->row_lab[4][j] = color_2.a;
            s->row_lab[5][j] = color_2.b;
        }
        if (simd_end)
            s->ciede2000_simd((const float *const *)s->row_lab, simd_end, powf(25., 7), s->row_de00);
        for (unsigned j = simd_end; j < w; j++) {
            const LABColor color_1 = { .l = s->row_lab[0][j], .a = s->row_lab[1][j], .b = s->row_lab[2][j] };
            const LABColor color_2 = { .l = s->row_lab[3][j], .a = s->row_lab[4][j], .b = s->row_lab[5][j] };
            s->row_de00[j] = ciede2000(color_1, color_2, default_ksub);
        }
        // added in raster order, as one pixel at a time did
        for (unsigned j = 0; j < w; j++)
            de00_sum += s->row_de00[j];
    }

    const double score = 45. - 20. *
                         log10(de00_sum / (ref_pic->w[0] * ref_pic->h[0]));
    return vmaf_feature_collector_append(feature_collector, "ciede2000", score,
                                         index);
}

static int close(VmafFeatureExtractor *fex)
{
    CiedeState *s = fex->priv;
    if (s->ref.data[0] && s->dist.data[0]) {
        vmaf_picture_unref(&s->ref);
        vmaf_picture_unref(&s->dist);
    }
    free(s->lab_cache);
    s->lab_cache = NULL;
    free(s->row_lab[0]);
    s->row_lab[0] = NULL;
    return 0;
}

static const char *provided_features[] = {
    "ciede2000",
    NULL
};

VmafFeatureExtractor vmaf_fex_ciede = {
    .name = "ciede",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(CiedeState),
    .provided_features = provided_features,
};
