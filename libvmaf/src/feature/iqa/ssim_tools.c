/*
 * Copyright (c) 2011, Tom Distler (http://tdistler.com)
 * All rights reserved.
 *
 * The BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * - Neither the name of the tdistler.com nor the names of its contributors may
 *   be used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * (06/10/2016) Updated by zli-nflx (zli@netflix.com) to output mean luminence,
 * contrast and structure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h> /* zli-nflx */

#include "iqa.h"
#include "convolve.h"
#include "ssim_tools.h"
#include "cpu.h"
/* libvmaf-fast: the SIMD copies of the three loops of floats of
 * _iqa_ssim_ws, AVX2 (VMAFx's, x86/ssim_avx2.c) or NEON (arm64/ssim_neon.c) */
#if ARCH_X86
#include "../x86/ssim_avx2.h"
#define SSIM_SIMD_FLAG VMAF_X86_CPU_FLAG_AVX2
#define ssim_precompute_simd ssim_precompute_avx2
#define ssim_variance_simd ssim_variance_avx2
#define ssim_accumulate_simd ssim_accumulate_avx2
#elif ARCH_AARCH64
#include "../arm64/ssim_neon.h"
#define SSIM_SIMD_FLAG VMAF_ARM_CPU_FLAG_NEON
#define ssim_precompute_simd ssim_precompute_neon
#define ssim_variance_simd ssim_variance_neon
#define ssim_accumulate_simd ssim_accumulate_neon
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* _calc_luminance */
IQA_INLINE static double _calc_luminance(float mu1, float mu2, float C1, float alpha)
{
    double result;
    float sign;
    /* For MS-SSIM* */
    if (C1 == 0 && mu1*mu1 == 0 && mu2*mu2 == 0)
        return 1.0;
    result = (2.0 * mu1 * mu2 + C1) / (mu1*mu1 + mu2*mu2 + C1);
    if (alpha == 1.0f)
        return result;
    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow(fabs(result),(double)alpha);
}

/* _calc_contrast */
IQA_INLINE static double _calc_contrast(double sigma_comb_12, float sigma1_sqd, float sigma2_sqd, float C2, float beta)
{
    double result;
    float sign;
    /* For MS-SSIM* */
    if (C2 == 0 && sigma1_sqd + sigma2_sqd == 0)
        return 1.0;
    result = (2.0 * sigma_comb_12 + C2) / (sigma1_sqd + sigma2_sqd + C2);
    if (beta == 1.0f)
        return result;
    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow(fabs(result),(double)beta);
}

/* _calc_structure */
IQA_INLINE static double _calc_structure(float sigma_12, double sigma_comb_12, float sigma1, float sigma2, float C3, float gamma)
{
    double result;
    float sign;
    /* For MS-SSIM* */
    if (C3 == 0 && sigma_comb_12 == 0) {
        if (sigma1 == 0 && sigma2 == 0)
            return 1.0;
        else if (sigma1 == 0 || sigma2 == 0)
            return 0.0;
    }
    result = (sigma_12 + C3) / (sigma_comb_12 + C3);
    if (gamma == 1.0f)
        return result;
    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow(fabs(result),(double)gamma);
}

/* _iqa_ssim */
float _iqa_ssim(float *ref, float *cmp, int w, int h, const struct _kernel *k,
		const struct _map_reduce *mr, const struct iqa_ssim_args *args
		, float *l_mean, float *c_mean, float *s_mean /* zli-nflx */
		)
{
    return _iqa_ssim_ws(ref, cmp, w, h, k, mr, args, l_mean, c_mean, s_mean, NULL);
}

float _iqa_ssim_ws(float *ref, float *cmp, int w, int h, const struct _kernel *k,
                   const struct _map_reduce *mr, const struct iqa_ssim_args *args,
                   float *l_mean, float *c_mean, float *s_mean, float *workspace)
{
    float alpha=1.0f, beta=1.0f, gamma=1.0f;
    int L=255;
    float K1=0.01f, K2=0.03f;
    float C1,C2,C3;
    int x,y,offset;
    float *ref_mu,*cmp_mu,*ref_sigma_sqd,*cmp_sigma_sqd,*sigma_both;
    double ssim_sum;
    // double numerator, denominator; /* zli-nflx */
    double luminance_comp, contrast_comp, structure_comp, sigma_root;
    struct _ssim_int sint;
    double l_sum, c_sum, s_sum, l, c, s; /* zli-nflx */
    float sigma_ref_sigma_cmp; /* zli-nflx */

    assert(!args); /* zli-nflx: for now only works for default case */

    /* Initialize algorithm parameters */
    if (args) {
        if (!mr)
            return INFINITY;
        alpha = args->alpha;
        beta  = args->beta;
        gamma = args->gamma;
        L     = args->L;
        K1    = args->K1;
        K2    = args->K2;
    }
    C1 = (K1*L)*(K1*L);
    C2 = (K2*L)*(K2*L);
    C3 = C2 / 2.0f;

    /* libvmaf-fast: the buffers (and the convolutions' cache) from the
     * caller's workspace, which it keeps from frame to frame */
    float *conv_cache = NULL;
    if (workspace) {
        const size_t n = (size_t)w * h;
        ref_mu = workspace;
        cmp_mu = workspace + n;
        ref_sigma_sqd = workspace + 2 * n;
        cmp_sigma_sqd = workspace + 3 * n;
        sigma_both = workspace + 4 * n;
        conv_cache = workspace + 5 * n;
    } else {
        ref_mu = (float*)malloc(w*h*sizeof(float));
        cmp_mu = (float*)malloc(w*h*sizeof(float));
        ref_sigma_sqd = (float*)malloc(w*h*sizeof(float));
        cmp_sigma_sqd = (float*)malloc(w*h*sizeof(float));
        sigma_both = (float*)malloc(w*h*sizeof(float));
    }
    if (!ref_mu || !cmp_mu || !ref_sigma_sqd || !cmp_sigma_sqd || !sigma_both) {
        if (ref_mu) free(ref_mu);
        if (cmp_mu) free(cmp_mu);
        if (ref_sigma_sqd) free(ref_sigma_sqd);
        if (cmp_sigma_sqd) free(cmp_sigma_sqd);
        if (sigma_both) free(sigma_both);
        return INFINITY;
    }

    /* Calculate mean */
    _iqa_convolve_ws(ref, w, h, k, ref_mu, 0, 0, conv_cache);
    _iqa_convolve_ws(cmp, w, h, k, cmp_mu, 0, 0, conv_cache);

    /* libvmaf-fast: the SIMD copies of the three loops of floats below,
     * each pixel's arithmetic the scalar code's. */
#ifdef SSIM_SIMD_FLAG
    const int simd = !!(vmaf_get_cpu_flags() & SSIM_SIMD_FLAG);
    if (simd)
        ssim_precompute_simd(ref, cmp, ref_sigma_sqd, cmp_sigma_sqd, sigma_both, w*h);
    else
#endif
    for (y=0; y<h; ++y) {
        offset = y*w;
        for (x=0; x<w; ++x, ++offset) {
            ref_sigma_sqd[offset] = ref[offset] * ref[offset];
            cmp_sigma_sqd[offset] = cmp[offset] * cmp[offset];
            sigma_both[offset] = ref[offset] * cmp[offset];
        }
    }

    /* Calculate sigma */
    _iqa_convolve_ws(ref_sigma_sqd, w, h, k, 0,  0,  0, conv_cache);
    _iqa_convolve_ws(cmp_sigma_sqd, w, h, k, 0,  0,  0, conv_cache);
    _iqa_convolve_ws(sigma_both,    w, h, k, 0, &w, &h, conv_cache); /* Update the width and height */

    /* The convolution results are smaller by the kernel width and height */
#ifdef SSIM_SIMD_FLAG
    /* libvmaf-fast: the pixels the loops below visit: none where the
     * picture is smaller than the window both ways, w and h both negative
     * (w*h is not) */
    const int pixels = w > 0 && h > 0 ? w * h : 0;
    if (simd)
        ssim_variance_simd(ref_sigma_sqd, cmp_sigma_sqd, sigma_both, ref_mu, cmp_mu, pixels);
    else
#endif
    for (y=0; y<h; ++y) {
        offset = y*w;
        for (x=0; x<w; ++x, ++offset) {
            ref_sigma_sqd[offset] -= ref_mu[offset] * ref_mu[offset];
            cmp_sigma_sqd[offset] -= cmp_mu[offset] * cmp_mu[offset];

            ref_sigma_sqd[offset] = MAX(0.0, ref_sigma_sqd[offset]); /* zli-nflx */
            cmp_sigma_sqd[offset] = MAX(0.0, cmp_sigma_sqd[offset]); /* zli-nflx */

            sigma_both[offset] -= ref_mu[offset] * cmp_mu[offset];
        }
    }

    ssim_sum = 0.0;
    l_sum = 0.0; /* zli-nflx */
    c_sum = 0.0; /* zli-nflx */
    s_sum = 0.0; /* zli-nflx */
#ifdef SSIM_SIMD_FLAG
    /* Over all the pixels in one call, so its four sums are added up in the
     * order of the loop below. */
    if (simd && !args)
        ssim_accumulate_simd(ref_mu, cmp_mu, ref_sigma_sqd, cmp_sigma_sqd, sigma_both, pixels,
                             C1, C2, C3, &ssim_sum, &l_sum, &c_sum, &s_sum);
    else
#endif
    for (y=0; y<h; ++y) {
        offset = y*w;
        for (x=0; x<w; ++x, ++offset) {

            if (!args) {

            	/* The default case */
                // numerator   = (2.0 * ref_mu[offset] * cmp_mu[offset] + C1) * (2.0 * sigma_both[offset] + C2);
                // denominator = (ref_mu[offset]*ref_mu[offset] + cmp_mu[offset]*cmp_mu[offset] + C1) *
                //     (ref_sigma_sqd[offset] + cmp_sigma_sqd[offset] + C2);
                // ssim_sum += numerator / denominator;

                /* zli-nflx: */
                sigma_ref_sigma_cmp = sqrt(ref_sigma_sqd[offset] * cmp_sigma_sqd[offset]);
                l = (2.0 * ref_mu[offset] * cmp_mu[offset] + C1) / (ref_mu[offset]*ref_mu[offset] + cmp_mu[offset]*cmp_mu[offset] + C1);
                c = (2.0 * sigma_ref_sigma_cmp + C2) /  (ref_sigma_sqd[offset] + cmp_sigma_sqd[offset] + C2);

                /* zli-nflx: fix corner case where ref and cmp are identical, and the local filtered region is
                 * completely flat (zero std). In this case, sigma_both can become negative due to inprecise
                 * float representation but sigma_ref_sigma_cmp is zero, resulting s < 1.0. The desiired
                 * behavior should be that ssim score is 1.0. */
                const float clamped_sigma_both = (sigma_both[offset] < 0.0f &&
                        sigma_ref_sigma_cmp <= 0.0f) ? 0.0f : sigma_both[offset];
                s = (clamped_sigma_both + C3) / (sigma_ref_sigma_cmp + C3);

                ssim_sum += l * c * s;
                l_sum += l;
                c_sum += c;
                s_sum += s;
            }
            else {
                /* User tweaked alpha, beta, or gamma */

                /* passing a negative number to sqrt() cause a domain error */
                if (ref_sigma_sqd[offset] < 0.0f)
                    ref_sigma_sqd[offset] = 0.0f;
                if (cmp_sigma_sqd[offset] < 0.0f)
                    cmp_sigma_sqd[offset] = 0.0f;
                sigma_root = sqrt(ref_sigma_sqd[offset] * cmp_sigma_sqd[offset]);

                luminance_comp = _calc_luminance(ref_mu[offset], cmp_mu[offset], C1, alpha);
                contrast_comp  = _calc_contrast(sigma_root, ref_sigma_sqd[offset], cmp_sigma_sqd[offset], C2, beta);
                structure_comp = _calc_structure(sigma_both[offset], sigma_root, ref_sigma_sqd[offset], cmp_sigma_sqd[offset], C3, gamma);

                sint.l = luminance_comp;
                sint.c = contrast_comp;
                sint.s = structure_comp;

                if (mr->map(&sint, mr->context))
                    return INFINITY;
            }
        }
    }

    if (!workspace) {
        free(ref_mu);
        free(cmp_mu);
        free(ref_sigma_sqd);
        free(cmp_sigma_sqd);
        free(sigma_both);
    }

    if (!args) {
    	*l_mean = (float)(l_sum / (double)(w*h)); /* zli-nflx */
    	*c_mean = (float)(c_sum / (double)(w*h)); /* zli-nflx */
    	*s_mean = (float)(s_sum / (double)(w*h)); /* zli-nflx */
        return (float)(ssim_sum / (double)(w*h));
    }
    return mr->reduce(w, h, mr->context);
}

