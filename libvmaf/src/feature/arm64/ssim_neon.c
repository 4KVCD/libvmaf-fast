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
 * libvmaf-fast's NEON code for the three loops of floats of
 * iqa/ssim_tools.c's _iqa_ssim_ws, each pixel's arithmetic the scalar
 * code's as Clang compiles it for arm64.
 *
 * Clang's default, -ffp-contract=on (Apple's Clang and Homebrew's FFmpeg
 * build libvmaf with it, as scripts/build_libvmaf_fast_macos.sh in
 * VideoMetricsLab does), fuses a multiply and an add written in one
 * expression into one instruction with one rounding (fmadd, fmsub). So
 * where the scalar code has such an expression, this code uses the fused
 * instruction (vfmaq, vfmsq, fma), and a separate multiply and add
 * elsewhere. The sites, from Clang's IR of ssim_tools.c (llvm.fmuladd):
 *
 *   variance:  x -= mu_a * mu_b                    x = fms(mu_a, mu_b, x)
 *   l:         2.0 * rm * cm + C1                  fma(2.0 * rm, cm, C1)
 *              rm * rm + cm * cm + C1              fma(rm, rm, cm * cm) + C1
 *   c:         2.0 * srsc + C2                     fma(2.0, srsc, C2)
 *   sums:      ssim_sum += l * c * s               ssim_sum = fma(l * c, s, ssim_sum)
 *
 * (fma(a, b, c) = a * b + c rounded once.) GCC's default
 * -ffp-contract=fast may fuse other expressions; VideoMetricsLab's arm64
 * builds are Clang's.
 *
 * Everything else is one IEEE operation in both, so the same values: the
 * float square root (sqrt in double of a float product, rounded back to
 * float, is the float square root), the conversions, the divisions. The
 * four sums take the pixels one at a time in the scalar loop's order.
 */

#include <arm_neon.h>
#include <math.h>

#include "ssim_neon.h"

void ssim_precompute_neon(const float *ref, const float *cmp, float *ref_sq, float *cmp_sq,
                          float *ref_cmp, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t r = vld1q_f32(ref + i);
        const float32x4_t c = vld1q_f32(cmp + i);
        vst1q_f32(ref_sq + i, vmulq_f32(r, r));
        vst1q_f32(cmp_sq + i, vmulq_f32(c, c));
        vst1q_f32(ref_cmp + i, vmulq_f32(r, c));
    }
    for (; i < n; i++) {
        ref_sq[i] = ref[i] * ref[i];
        cmp_sq[i] = cmp[i] * cmp[i];
        ref_cmp[i] = ref[i] * cmp[i];
    }
}

/* the scalar code's MAX(0.0, x): x unless 0 > x, so -0.0 and NaN stay */
static inline float32x4_t max_zero_first(float32x4_t x)
{
    const float32x4_t zero = vdupq_n_f32(0.0f);
    return vbslq_f32(vcgtq_f32(zero, x), zero, x);
}

static inline float max_zero_first_scalar(float x)
{
    return 0.0f > x ? 0.0f : x;
}

void ssim_variance_neon(float *ref_sigma_sqd, float *cmp_sigma_sqd, float *sigma_both,
                        const float *ref_mu, const float *cmp_mu, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t rm = vld1q_f32(ref_mu + i);
        const float32x4_t cm = vld1q_f32(cmp_mu + i);
        /* vfmsq_f32(x, a, b) = x - a * b, rounded once */
        const float32x4_t rs = vfmsq_f32(vld1q_f32(ref_sigma_sqd + i), rm, rm);
        const float32x4_t cs = vfmsq_f32(vld1q_f32(cmp_sigma_sqd + i), cm, cm);
        vst1q_f32(ref_sigma_sqd + i, max_zero_first(rs));
        vst1q_f32(cmp_sigma_sqd + i, max_zero_first(cs));
        vst1q_f32(sigma_both + i, vfmsq_f32(vld1q_f32(sigma_both + i), rm, cm));
    }
    for (; i < n; i++) {
        ref_sigma_sqd[i] = max_zero_first_scalar(fmaf(-ref_mu[i], ref_mu[i], ref_sigma_sqd[i]));
        cmp_sigma_sqd[i] = max_zero_first_scalar(fmaf(-cmp_mu[i], cmp_mu[i], cmp_sigma_sqd[i]));
        sigma_both[i] = fmaf(-ref_mu[i], cmp_mu[i], sigma_both[i]);
    }
}

/* The four sums of the scalar loop: each pixel's l * c * s fused into
 * ssim_sum, then its l, c and s added. */
typedef struct SsimSums {
    double ssim, l, c, s;
} SsimSums;

static inline void add_pixel(SsimSums *sums, double l, double c, double s)
{
    sums->ssim = fma(l * c, s, sums->ssim);
    sums->l += l;
    sums->c += c;
    sums->s += s;
}

/* One pixel's l and c in double and its s (a float, widened where it is
 * added), all in scalar code (the pixels after the last whole vector). */
static inline void scalar_terms(float rm, float cm, float rs, float cs, float sb, float C1,
                                float C2, float C3, double *l, double *c, float *s)
{
    const float srsc = sqrtf(rs * cs);
    *l = fma(2.0 * rm, (double)cm, (double)C1) / (double)(fmaf(rm, rm, cm * cm) + C1);
    *c = fma(2.0, (double)srsc, (double)C2) / (double)(rs + cs + C2);
    const float csb = (sb < 0.0f && srsc <= 0.0f) ? 0.0f : sb;
    *s = (csb + C3) / (srsc + C3);
}

static inline void accumulate_scalar(SsimSums *sums, float rm, float cm, float rs, float cs,
                                     float sb, float C1, float C2, float C3)
{
    double l, c;
    float s;
    scalar_terms(rm, cm, rs, cs, sb, C1, C2, C3, &l, &c, &s);
    add_pixel(sums, l, c, s);
}

/* l and c of 2 pixels in double, from their float terms (lanes 0 and 1 of
 * the float vectors' low or high half, already widened). */
static inline void lc_pair(float64x2_t rm, float64x2_t cm, float64x2_t srsc, float64x2_t l_den,
                           float64x2_t c_den, double C1, double C2, float64x2_t *l,
                           float64x2_t *c)
{
    const float64x2_t two = vdupq_n_f64(2.0);
    /* vfmaq_f64(a, b, c) = a + b * c, rounded once */
    *l = vdivq_f64(vfmaq_f64(vdupq_n_f64(C1), vmulq_f64(two, rm), cm), l_den);
    *c = vdivq_f64(vfmaq_f64(vdupq_n_f64(C2), two, srsc), c_den);
}

static inline void accumulate_pair(SsimSums *sums, float64x2_t rm, float64x2_t cm,
                                   float64x2_t srsc, float64x2_t l_den, float64x2_t c_den,
                                   float64x2_t s, double C1, double C2)
{
    float64x2_t l, c;
    lc_pair(rm, cm, srsc, l_den, c_den, C1, C2, &l, &c);
    add_pixel(sums, vgetq_lane_f64(l, 0), vgetq_lane_f64(c, 0), vgetq_lane_f64(s, 0));
    add_pixel(sums, vgetq_lane_f64(l, 1), vgetq_lane_f64(c, 1), vgetq_lane_f64(s, 1));
}

/* The float terms of 4 pixels at i. */
typedef struct FloatTerms {
    float32x4_t rm, cm, srsc, l_den, c_den, s;
} FloatTerms;

static inline FloatTerms float_terms(const float *ref_mu, const float *cmp_mu,
                                     const float *ref_sigma_sqd, const float *cmp_sigma_sqd,
                                     const float *sigma_both, int i, float32x4_t vC1,
                                     float32x4_t vC2, float32x4_t vC3)
{
    const float32x4_t zero = vdupq_n_f32(0.0f);
    FloatTerms t;
    t.rm = vld1q_f32(ref_mu + i);
    t.cm = vld1q_f32(cmp_mu + i);
    const float32x4_t rs = vld1q_f32(ref_sigma_sqd + i);
    const float32x4_t cs = vld1q_f32(cmp_sigma_sqd + i);
    const float32x4_t sb = vld1q_f32(sigma_both + i);
    t.srsc = vsqrtq_f32(vmulq_f32(rs, cs));
    t.l_den = vaddq_f32(vfmaq_f32(vmulq_f32(t.cm, t.cm), t.rm, t.rm), vC1);
    t.c_den = vaddq_f32(vaddq_f32(rs, cs), vC2);
    /* sigma_both < 0 where the region is flat counts as 0 */
    const uint32x4_t flat = vandq_u32(vcltq_f32(sb, zero), vcleq_f32(t.srsc, zero));
    const float32x4_t csb = vbslq_f32(flat, zero, sb);
    t.s = vdivq_f32(vaddq_f32(csb, vC3), vaddq_f32(t.srsc, vC3));
    return t;
}

void ssim_accumulate_neon(const float *ref_mu, const float *cmp_mu, const float *ref_sigma_sqd,
                          const float *cmp_sigma_sqd, const float *sigma_both, int n, float C1,
                          float C2, float C3, double *ssim_sum, double *l_sum, double *c_sum,
                          double *s_sum)
{
    const float32x4_t vC1 = vdupq_n_f32(C1);
    const float32x4_t vC2 = vdupq_n_f32(C2);
    const float32x4_t vC3 = vdupq_n_f32(C3);
    SsimSums sums = { *ssim_sum, *l_sum, *c_sum, *s_sum };

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const FloatTerms t = float_terms(ref_mu, cmp_mu, ref_sigma_sqd, cmp_sigma_sqd,
                                         sigma_both, i, vC1, vC2, vC3);
        /* the double part, two pixels at a time */
        accumulate_pair(&sums, vcvt_f64_f32(vget_low_f32(t.rm)),
                        vcvt_f64_f32(vget_low_f32(t.cm)), vcvt_f64_f32(vget_low_f32(t.srsc)),
                        vcvt_f64_f32(vget_low_f32(t.l_den)), vcvt_f64_f32(vget_low_f32(t.c_den)),
                        vcvt_f64_f32(vget_low_f32(t.s)), C1, C2);
        accumulate_pair(&sums, vcvt_high_f64_f32(t.rm), vcvt_high_f64_f32(t.cm),
                        vcvt_high_f64_f32(t.srsc), vcvt_high_f64_f32(t.l_den),
                        vcvt_high_f64_f32(t.c_den), vcvt_high_f64_f32(t.s), C1, C2);
    }
    for (; i < n; i++)
        accumulate_scalar(&sums, ref_mu[i], cmp_mu[i], ref_sigma_sqd[i], cmp_sigma_sqd[i],
                          sigma_both[i], C1, C2, C3);

    *ssim_sum = sums.ssim;
    *l_sum = sums.l;
    *c_sum = sums.c;
    *s_sum = sums.s;
}

void ssim_terms_neon(const float *ref_mu, const float *cmp_mu, const float *ref_sigma_sqd,
                     const float *cmp_sigma_sqd, const float *sigma_both, int n, float C1,
                     float C2, float C3, double *l, double *c, float *s)
{
    const float32x4_t vC1 = vdupq_n_f32(C1);
    const float32x4_t vC2 = vdupq_n_f32(C2);
    const float32x4_t vC3 = vdupq_n_f32(C3);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const FloatTerms t = float_terms(ref_mu, cmp_mu, ref_sigma_sqd, cmp_sigma_sqd,
                                         sigma_both, i, vC1, vC2, vC3);
        float64x2_t lo_l, lo_c, hi_l, hi_c;
        lc_pair(vcvt_f64_f32(vget_low_f32(t.rm)), vcvt_f64_f32(vget_low_f32(t.cm)),
                vcvt_f64_f32(vget_low_f32(t.srsc)), vcvt_f64_f32(vget_low_f32(t.l_den)),
                vcvt_f64_f32(vget_low_f32(t.c_den)), C1, C2, &lo_l, &lo_c);
        lc_pair(vcvt_high_f64_f32(t.rm), vcvt_high_f64_f32(t.cm), vcvt_high_f64_f32(t.srsc),
                vcvt_high_f64_f32(t.l_den), vcvt_high_f64_f32(t.c_den), C1, C2, &hi_l, &hi_c);
        vst1q_f64(l + i, lo_l);
        vst1q_f64(l + i + 2, hi_l);
        vst1q_f64(c + i, lo_c);
        vst1q_f64(c + i + 2, hi_c);
        vst1q_f32(s + i, t.s);
    }
    for (; i < n; i++)
        scalar_terms(ref_mu[i], cmp_mu[i], ref_sigma_sqd[i], cmp_sigma_sqd[i], sigma_both[i], C1,
                     C2, C3, l + i, c + i, s + i);
}

void ssim_add_terms_neon(const double *l, const double *c, const float *s, int n,
                         double *ssim_sum, double *l_sum, double *c_sum, double *s_sum)
{
    SsimSums sums = { *ssim_sum, *l_sum, *c_sum, *s_sum };
    for (int i = 0; i < n; i++)
        add_pixel(&sums, l[i], c[i], s[i]);
    *ssim_sum = sums.ssim;
    *l_sum = sums.l;
    *c_sum = sums.c;
    *s_sum = sums.s;
}
