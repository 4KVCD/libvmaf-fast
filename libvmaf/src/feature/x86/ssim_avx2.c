/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright (c) 2011, Tom Distler (http://tdistler.com)
 *  Copyright 2026 Lusoris
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent AND BSD-3-Clause
 */

/*
 * libvmaf-fast: copied from VMAFx (github.com/VMAFx/vmafx, commit 1ed4affa,
 * core/src/feature/x86/ssim_avx2.c; the ADR named below is in that
 * repository), with three changes:
 * - ssim_accumulate_avx2's double part runs in vectors of 4 too
 *   (ssim_accumulate_half), not one pixel at a time: each lane does the
 *   same double operations, so the same sums, in less than half the time.
 * - VMAFx's iqa/ssim_accumulate_lane.h (EUPL-1.2) is not used: its two
 *   helpers are below, made from this fork's own scalar loop in
 *   iqa/ssim_tools.c _iqa_ssim (BSD-3-Clause), which they mirror.
 * - ssim_variance_avx2 clamps with _mm256_max_ps(zero, x), not (x, zero):
 *   maxps returns its second operand unless the first is greater, so this
 *   is _iqa_ssim's MAX(0.0, x) for -0.0 and NaN too.
 * Called from iqa/ssim_tools.c's _iqa_ssim.
 */

#include <immintrin.h>
#include <math.h>

#include "ssim_avx2.h"

/* One pixel's terms of _iqa_ssim's default case, from the float values the
 * vector code computed exactly as the scalar code does (srsc =
 * sigma_ref_sigma_cmp, l_den and c_den the denominators of l and c, sv_f =
 * s): l and c in double, as the scalar code's 2.0 makes them, and the four
 * sums in the scalar code's order. */
static inline void ssim_accumulate_lane(float rm, float cm, float srsc, float l_den, float c_den,
                                        float sv_f, float C1, float C2, double *local_ssim,
                                        double *local_l, double *local_c, double *local_s)
{
    const double l = (2.0 * rm * cm + C1) / l_den;
    const double c = (2.0 * srsc + C2) / c_den;
    const double s = sv_f;
    *local_ssim += l * c * s;
    *local_l += l;
    *local_c += c;
    *local_s += s;
}

/* _iqa_ssim's default case for one pixel, all in scalar code (the columns
 * after the last whole vector). sqrtf of a float is the float of sqrt of
 * it in double, as the scalar code takes it. */
static inline void ssim_accumulate_scalar_step(float rm, float cm, float rs, float cs, float sb,
                                               float C1, float C2, float C3, double *local_ssim,
                                               double *local_l, double *local_c, double *local_s)
{
    const float srsc = sqrtf(rs * cs);
    const float l_den = rm * rm + cm * cm + C1;
    const float c_den = rs + cs + C2;
    const float csb = (sb < 0.0f && srsc <= 0.0f) ? 0.0f : sb;
    const float sv_f = (csb + C3) / (srsc + C3);
    ssim_accumulate_lane(rm, cm, srsc, l_den, c_den, sv_f, C1, C2, local_ssim, local_l, local_c,
                         local_s);
}

/*
 * Bit-exact AVX2 SSIM SIMD helpers — mirror the scalar reference in
 * core/src/feature/iqa/ssim_tools.c byte-for-byte under
 * FLT_EVAL_METHOD == 0. `precompute` and `variance` are pure elementwise
 * float operations (no reductions) — float IEEE-754 ops are identical to
 * scalar by construction. `accumulate` is the tricky one: scalar computes
 * l and c in *double* (because of the `2.0 * ...` literal) and does the
 * final `l*c*s` product in double too, whereas doing it all in vector
 * float diverges by ~0.13 float ULPs on MS-SSIM. We keep SIMD for the
 * float-valued intermediates (srsc, denominators, sv) and do the
 * double-precision numerator + division + product in double, preserving
 * both the type promotions and the reduction order (libvmaf-fast: in
 * double vectors, ssim_accumulate_half).
 */

void ssim_precompute_avx2(const float *ref, const float *cmp, float *ref_sq, float *cmp_sq,
                          float *ref_cmp, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 r = _mm256_loadu_ps(ref + i);
        __m256 c = _mm256_loadu_ps(cmp + i);
        _mm256_storeu_ps(ref_sq + i, _mm256_mul_ps(r, r));
        _mm256_storeu_ps(cmp_sq + i, _mm256_mul_ps(c, c));
        _mm256_storeu_ps(ref_cmp + i, _mm256_mul_ps(r, c));
    }
    for (; i < n; i++) {
        ref_sq[i] = ref[i] * ref[i];
        cmp_sq[i] = cmp[i] * cmp[i];
        ref_cmp[i] = ref[i] * cmp[i];
    }
}

void ssim_variance_avx2(float *ref_sigma_sqd, float *cmp_sigma_sqd, float *sigma_both,
                        const float *ref_mu, const float *cmp_mu, int n)
{
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 rm = _mm256_loadu_ps(ref_mu + i);
        __m256 cm = _mm256_loadu_ps(cmp_mu + i);

        __m256 rs = _mm256_loadu_ps(ref_sigma_sqd + i);
        rs = _mm256_sub_ps(rs, _mm256_mul_ps(rm, rm));
        rs = _mm256_max_ps(zero, rs);
        _mm256_storeu_ps(ref_sigma_sqd + i, rs);

        __m256 cs = _mm256_loadu_ps(cmp_sigma_sqd + i);
        cs = _mm256_sub_ps(cs, _mm256_mul_ps(cm, cm));
        cs = _mm256_max_ps(zero, cs);
        _mm256_storeu_ps(cmp_sigma_sqd + i, cs);

        __m256 sb = _mm256_loadu_ps(sigma_both + i);
        sb = _mm256_sub_ps(sb, _mm256_mul_ps(rm, cm));
        _mm256_storeu_ps(sigma_both + i, sb);
    }
    for (; i < n; i++) {
        ref_sigma_sqd[i] -= ref_mu[i] * ref_mu[i];
        if (ref_sigma_sqd[i] < 0.0f)
            ref_sigma_sqd[i] = 0.0f;
        cmp_sigma_sqd[i] -= cmp_mu[i] * cmp_mu[i];
        if (cmp_sigma_sqd[i] < 0.0f)
            cmp_sigma_sqd[i] = 0.0f;
        sigma_both[i] -= ref_mu[i] * cmp_mu[i];
    }
}

/* sum + v[0] + v[1] + v[2] + v[3], added one at a time in that order */
static inline double add_lanes_in_order(double sum, __m256d v)
{
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    sum += _mm_cvtsd_f64(lo);
    sum += _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
    sum += _mm_cvtsd_f64(hi);
    sum += _mm_cvtsd_f64(_mm_unpackhi_pd(hi, hi));
    return sum;
}

/* ssim_accumulate_lane for 4 pixels: l, c and l * c * s in double, as
 * (2.0 * rm) * cm + C1 over l_den, (2.0 * srsc + C2) over c_den, and
 * (l * c) * s (every float converted to double exactly), then the four sums
 * in pixel order. */
static inline void ssim_accumulate_half(__m128 rm, __m128 cm, __m128 srsc, __m128 l_den,
                                        __m128 c_den, __m128 sv_f, float C1, float C2,
                                        double *local_ssim, double *local_l, double *local_c,
                                        double *local_s)
{
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d l_num = _mm256_add_pd(
        _mm256_mul_pd(_mm256_mul_pd(two, _mm256_cvtps_pd(rm)), _mm256_cvtps_pd(cm)),
        _mm256_set1_pd(C1));
    const __m256d l = _mm256_div_pd(l_num, _mm256_cvtps_pd(l_den));
    const __m256d c_num = _mm256_add_pd(_mm256_mul_pd(two, _mm256_cvtps_pd(srsc)),
                                        _mm256_set1_pd(C2));
    const __m256d c = _mm256_div_pd(c_num, _mm256_cvtps_pd(c_den));
    const __m256d s = _mm256_cvtps_pd(sv_f);
    const __m256d lcs = _mm256_mul_pd(_mm256_mul_pd(l, c), s);
    *local_ssim = add_lanes_in_order(*local_ssim, lcs);
    *local_l = add_lanes_in_order(*local_l, l);
    *local_c = add_lanes_in_order(*local_c, c);
    *local_s = add_lanes_in_order(*local_s, s);
}

/* 8-wide SIMD block: the float intermediates in 8 lanes, then the double
 * part in two halves of 4 (ssim_accumulate_half). */
static inline void ssim_accumulate_block_avx2(const float *ref_mu, const float *cmp_mu,
                                              const float *ref_sigma_sqd,
                                              const float *cmp_sigma_sqd, const float *sigma_both,
                                              int i, __m256 vC1, __m256 vC2, __m256 vC3,
                                              __m256 vzero, float C1, float C2, double *local_ssim,
                                              double *local_l, double *local_c, double *local_s)
{
    const __m256 rm = _mm256_loadu_ps(ref_mu + i);
    const __m256 cm = _mm256_loadu_ps(cmp_mu + i);
    const __m256 rs = _mm256_loadu_ps(ref_sigma_sqd + i);
    const __m256 cs = _mm256_loadu_ps(cmp_sigma_sqd + i);
    const __m256 sb = _mm256_loadu_ps(sigma_both + i);
    const __m256 srsc = _mm256_sqrt_ps(_mm256_mul_ps(rs, cs));
    const __m256 l_den =
        _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rm, rm), _mm256_mul_ps(cm, cm)), vC1);
    const __m256 c_den = _mm256_add_ps(_mm256_add_ps(rs, cs), vC2);
    const __m256 sb_neg = _mm256_cmp_ps(sb, vzero, _CMP_LT_OQ);
    const __m256 srsc_le0 = _mm256_cmp_ps(srsc, vzero, _CMP_LE_OQ);
    const __m256 clamped_sb = _mm256_blendv_ps(sb, vzero, _mm256_and_ps(sb_neg, srsc_le0));
    const __m256 sv_f = _mm256_div_ps(_mm256_add_ps(clamped_sb, vC3), _mm256_add_ps(srsc, vC3));

    /* libvmaf-fast: the double part in vectors too, 4 pixels at a time
     * (was ssim_accumulate_lane per pixel): each lane does
     * ssim_accumulate_lane's operations in its order, every one a single
     * IEEE double operation, so the same values; the four sums then take
     * them pixel by pixel, in the scalar loop's order. */
    ssim_accumulate_half(_mm256_castps256_ps128(rm), _mm256_castps256_ps128(cm),
                         _mm256_castps256_ps128(srsc), _mm256_castps256_ps128(l_den),
                         _mm256_castps256_ps128(c_den), _mm256_castps256_ps128(sv_f), C1, C2,
                         local_ssim, local_l, local_c, local_s);
    ssim_accumulate_half(_mm256_extractf128_ps(rm, 1), _mm256_extractf128_ps(cm, 1),
                         _mm256_extractf128_ps(srsc, 1), _mm256_extractf128_ps(l_den, 1),
                         _mm256_extractf128_ps(c_den, 1), _mm256_extractf128_ps(sv_f, 1), C1, C2,
                         local_ssim, local_l, local_c, local_s);
}

void ssim_accumulate_avx2(const float *ref_mu, const float *cmp_mu, const float *ref_sigma_sqd,
                          const float *cmp_sigma_sqd, const float *sigma_both, int n, float C1,
                          float C2, float C3, double *ssim_sum, double *l_sum, double *c_sum,
                          double *s_sum)
{
    const __m256 vC1 = _mm256_set1_ps(C1);
    const __m256 vC2 = _mm256_set1_ps(C2);
    const __m256 vC3 = _mm256_set1_ps(C3);
    const __m256 vzero = _mm256_setzero_ps();
    double local_ssim = 0.0;
    double local_l = 0.0;
    double local_c = 0.0;
    double local_s = 0.0;

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        ssim_accumulate_block_avx2(ref_mu, cmp_mu, ref_sigma_sqd, cmp_sigma_sqd, sigma_both, i, vC1,
                                   vC2, vC3, vzero, C1, C2, &local_ssim, &local_l, &local_c,
                                   &local_s);
    }
    for (; i < n; i++) {
        ssim_accumulate_scalar_step(ref_mu[i], cmp_mu[i], ref_sigma_sqd[i], cmp_sigma_sqd[i],
                                    sigma_both[i], C1, C2, C3, &local_ssim, &local_l, &local_c,
                                    &local_s);
    }
    *ssim_sum += local_ssim;
    *l_sum += local_l;
    *c_sum += local_c;
    *s_sum += local_s;
}
