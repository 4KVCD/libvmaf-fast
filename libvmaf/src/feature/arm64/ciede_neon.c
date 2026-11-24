/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *  SPDX-License-Identifier: BSD-2-Clause-Patent
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

/* ciede.c's ciede2000(), two pixels per vector (float32x2 for its floats,
 * float64x2 for its doubles), four per loop step.
 *
 * Every add, multiply, divide and square root is ciede2000()'s own, in its
 * types (float or double) and order, so each lane rounds where the scalar
 * code rounds. On arm64 the compiler fuses a * b + c within one expression
 * of ciede.c (FP contraction is on): a_prime, s_sub_c, s_sub_upcase_h and
 * get_upcase_t's sum and arguments are fused multiply-adds there, so they
 * are vfmaq here, and every other product and sum stays a separate multiply
 * and add (intrinsics are never contracted). Halving and doubling are
 * written as * 0.5 and x + x: exact, so the same values as / 2 and 2 *.
 *
 * Only the C library's atan2, sin, cos, exp and pow differ: they are replaced
 * by the double-precision functions below (within about an ulp of a double),
 * and pow(x, 7) by products. Their results are rounded to float right after,
 * as in ciede2000(), which hides the difference in all but rare cases: a
 * pixel's dE00 then differs in the last bit of its float, and a frame's score
 * in about the 12th digit.
 *
 * The polynomials and argument reductions are fdlibm's (k_sin.c, k_cos.c,
 * e_exp.c, s_atan.c, e_atan2.c: Copyright (C) 1993 by Sun Microsystems,
 * Inc. Developed at SunSoft, a Sun Microsystems, Inc. business. Permission
 * to use, copy, modify, and distribute this software is freely granted,
 * provided that this notice is preserved.), evaluated with fused
 * multiply-adds, which are at least as accurate as fdlibm's separate steps. */

#include <arm_neon.h>
#include <math.h>

#include "ciede_neon.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* sin(r) and cos(r) for |r| <= pi/4 (fdlibm's __kernel_sin and __kernel_cos,
 * the latter without its correction for |r| > 0.3, which is worth less than
 * the float rounding after it). */
static inline float64x2_t kernel_sin(float64x2_t r)
{
    const float64x2_t z = vmulq_f64(r, r);
    const float64x2_t v = vmulq_f64(z, r);
    float64x2_t p = vfmaq_n_f64(vdupq_n_f64(-2.50507602534068634195e-08), z, 1.58969099521155010221e-10);
    p = vfmaq_f64(vdupq_n_f64(2.75573137070700676789e-06), z, p);
    p = vfmaq_f64(vdupq_n_f64(-1.98412698298579493134e-04), z, p);
    p = vfmaq_f64(vdupq_n_f64(8.33333333332248946124e-03), z, p);
    return vfmaq_f64(r, v, vfmaq_f64(vdupq_n_f64(-1.66666666666666324348e-01), z, p));
}

static inline float64x2_t kernel_cos(float64x2_t r)
{
    const float64x2_t z = vmulq_f64(r, r);
    float64x2_t p = vfmaq_n_f64(vdupq_n_f64(2.08757232129817482790e-09), z, -1.13596475577881948265e-11);
    p = vfmaq_f64(vdupq_n_f64(-2.75573143513906633035e-07), z, p);
    p = vfmaq_f64(vdupq_n_f64(2.48015872894767294178e-05), z, p);
    p = vfmaq_f64(vdupq_n_f64(-1.38888888888741095749e-03), z, p);
    p = vfmaq_f64(vdupq_n_f64(4.16666666666666019037e-02), z, p);
    /* 1 - (z / 2 - z^2 p) */
    return vsubq_f64(vdupq_n_f64(1.0), vfmsq_f64(vmulq_n_f64(z, 0.5), z, vmulq_f64(z, p)));
}

/* x - k pi/2 with k the nearest whole number, and k (whose last two bits are
 * the quadrant), for |x| < 2^20 (Cody and Waite's reduction with fdlibm's two
 * parts of pi/2: k * pio2_1 is exact). */
static inline float64x2_t reduce_pio2(float64x2_t x, int64x2_t *k_out)
{
    const float64x2_t k = vrndnq_f64(vmulq_n_f64(x, 6.36619772367581382433e-01));
    *k_out = vcvtq_s64_f64(k);
    const float64x2_t r = vfmsq_n_f64(x, k, 1.57079632673412561417e+00);
    return vfmsq_n_f64(r, k, 6.07710050650619224932e-11);
}

/* x with its sign flipped where the lane's mask is set */
static inline float64x2_t negate_where(float64x2_t x, uint64x2_t mask)
{
    const uint64x2_t sign = vandq_u64(mask, vdupq_n_u64(0x8000000000000000ull));
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(x), sign));
}

/* sin(x): in quadrant k the kernel is sin (k even) or cos (k odd) of the
 * remainder, negative for k = 2 and 3 (bit 1 of k). */
static inline float64x2_t vsin(float64x2_t x)
{
    int64x2_t k;
    const float64x2_t r = reduce_pio2(x, &k);
    const uint64x2_t odd = vtstq_s64(k, vdupq_n_s64(1));
    const float64x2_t s = vbslq_f64(odd, kernel_cos(r), kernel_sin(r));
    return negate_where(s, vtstq_s64(k, vdupq_n_s64(2)));
}

/* cos(x) = sin(x + pi/2): quadrant k + 1, negative where bit 1 of k + 1 is set */
static inline float64x2_t vcos(float64x2_t x)
{
    int64x2_t k;
    const float64x2_t r = reduce_pio2(x, &k);
    const uint64x2_t odd = vtstq_s64(k, vdupq_n_s64(1));
    const float64x2_t c = vbslq_f64(odd, kernel_sin(r), kernel_cos(r));
    return negate_where(c, vtstq_s64(vaddq_s64(k, vdupq_n_s64(1)), vdupq_n_s64(2)));
}

/* exp(x) for -700 < x < 700 (fdlibm's __ieee754_exp). */
static inline float64x2_t vexp(float64x2_t x)
{
    const float64x2_t k = vrndnq_f64(vmulq_n_f64(x, 1.44269504088896338700e+00));
    const float64x2_t hi = vfmsq_n_f64(x, k, 6.93147180369123816490e-01);
    const float64x2_t lo = vmulq_n_f64(k, 1.90821492927058770002e-10);
    const float64x2_t r = vsubq_f64(hi, lo);
    const float64x2_t t = vmulq_f64(r, r);
    float64x2_t p = vfmaq_n_f64(vdupq_n_f64(-1.65339022054652515390e-06), t, 4.13813679705723846039e-08);
    p = vfmaq_f64(vdupq_n_f64(6.61375632143793436117e-05), t, p);
    p = vfmaq_f64(vdupq_n_f64(-2.77777777770155933842e-03), t, p);
    p = vfmaq_f64(vdupq_n_f64(1.66666666666666019037e-01), t, p);
    const float64x2_t c = vfmsq_f64(r, t, p);
    /* y = 1 - ((lo - r c / (2 - c)) - hi) */
    const float64x2_t rc = vdivq_f64(vmulq_f64(r, c), vsubq_f64(vdupq_n_f64(2.0), c));
    const float64x2_t y = vsubq_f64(vdupq_n_f64(1.0), vsubq_f64(vsubq_f64(lo, rc), hi));
    /* y * 2^k, by adding k to y's exponent */
    const int64x2_t exponent = vshlq_n_s64(vcvtq_s64_f64(k), 52);
    return vreinterpretq_f64_s64(vaddq_s64(vreinterpretq_s64_f64(y), exponent));
}

/* atan2(y, x), not both 0 (fdlibm's __ieee754_atan2 and atan). */
static inline float64x2_t vatan2(float64x2_t y, float64x2_t x)
{
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t t = vdivq_f64(vabsq_f64(y), vabsq_f64(x)); /* +inf where x is 0 */
    /* fdlibm's intervals: t below 7/16, 11/16, 19/16, 39/16, and above */
    const uint64x2_t i0 = vcgeq_f64(t, vdupq_n_f64(0.4375));
    const uint64x2_t i1 = vcgeq_f64(t, vdupq_n_f64(0.6875));
    const uint64x2_t i2 = vcgeq_f64(t, vdupq_n_f64(1.1875));
    const uint64x2_t i3 = vcgeq_f64(t, vdupq_n_f64(2.4375));
    float64x2_t num = t, den = one;
    num = vbslq_f64(i0, vsubq_f64(vaddq_f64(t, t), one), num);
    den = vbslq_f64(i0, vaddq_f64(vdupq_n_f64(2.0), t), den);
    num = vbslq_f64(i1, vsubq_f64(t, one), num);
    den = vbslq_f64(i1, vaddq_f64(t, one), den);
    num = vbslq_f64(i2, vsubq_f64(t, vdupq_n_f64(1.5)), num);
    den = vbslq_f64(i2, vfmaq_n_f64(one, t, 1.5), den);
    num = vbslq_f64(i3, vdupq_n_f64(-1.0), num);
    den = vbslq_f64(i3, t, den);
    const float64x2_t u = vdivq_f64(num, den);
    const float64x2_t z = vmulq_f64(u, u);
    const float64x2_t w = vmulq_f64(z, z);
    float64x2_t s1 = vfmaq_n_f64(vdupq_n_f64(4.97687799461593236017e-02), w, 1.62858201153657823623e-02);
    s1 = vfmaq_f64(vdupq_n_f64(6.66107313738753120669e-02), w, s1);
    s1 = vfmaq_f64(vdupq_n_f64(9.09088713343650656196e-02), w, s1);
    s1 = vfmaq_f64(vdupq_n_f64(1.42857142725034663711e-01), w, s1);
    s1 = vmulq_f64(z, vfmaq_f64(vdupq_n_f64(3.33333333333329318027e-01), w, s1));
    float64x2_t s2 = vfmaq_n_f64(vdupq_n_f64(-5.83357013379057348645e-02), w, -3.65315727442169155270e-02);
    s2 = vfmaq_f64(vdupq_n_f64(-7.69187620504482999495e-02), w, s2);
    s2 = vfmaq_f64(vdupq_n_f64(-1.11111104054623557880e-01), w, s2);
    s2 = vmulq_f64(w, vfmaq_f64(vdupq_n_f64(-1.99999999998764832476e-01), w, s2));
    const float64x2_t us = vmulq_f64(u, vaddq_f64(s1, s2));
    /* atan of the interval's start, in two parts */
    float64x2_t hi = vbslq_f64(i0, vdupq_n_f64(4.63647609000806093515e-01), vdupq_n_f64(0.0));
    float64x2_t lo = vbslq_f64(i0, vdupq_n_f64(2.26987774529616870924e-17), vdupq_n_f64(0.0));
    hi = vbslq_f64(i1, vdupq_n_f64(7.85398163397448278999e-01), hi);
    lo = vbslq_f64(i1, vdupq_n_f64(3.06161699786838301793e-17), lo);
    hi = vbslq_f64(i2, vdupq_n_f64(9.82793723247329054082e-01), hi);
    lo = vbslq_f64(i2, vdupq_n_f64(1.39033110312309984516e-17), lo);
    hi = vbslq_f64(i3, vdupq_n_f64(1.57079632679489655800e+00), hi);
    lo = vbslq_f64(i3, vdupq_n_f64(6.12323399573676603587e-17), lo);
    /* atan(t) */
    const float64x2_t a = vbslq_f64(i0, vsubq_f64(hi, vsubq_f64(vsubq_f64(us, lo), u)), vsubq_f64(u, us));
    /* the quadrant: x < 0 gives pi - a, y < 0 negates (x = -0 counts as
     * x = 0, pi/2, as in fdlibm) */
    const float64x2_t pi_minus_a =
        vsubq_f64(vdupq_n_f64(3.1415926535897931160E+00), vsubq_f64(a, vdupq_n_f64(1.2246467991473531772E-16)));
    const float64x2_t r = vbslq_f64(vcltzq_f64(x), pi_minus_a, a);
    /* r >= 0: take y's sign bit */
    return vbslq_f64(vdupq_n_u64(0x8000000000000000ull), y, r);
}

/* get_h_prime(): the hue angle in [0, 2 pi), 0 for a grey. */
static inline float32x2_t h_prime(float32x2_t x, float32x2_t y)
{
    const float32x2_t hue = vcvt_f32_f64(vatan2(vcvt_f64_f32(x), vcvt_f64_f32(y)));
    const float32x2_t wrapped = vcvt_f32_f64(vaddq_f64(vcvt_f64_f32(hue), vdupq_n_f64(2. * M_PI)));
    const float32x2_t angle = vbsl_f32(vcltz_f32(hue), wrapped, hue);
    const uint32x2_t grey = vand_u32(vceqz_f32(x), vceqz_f32(y));
    return vreinterpret_f32_u32(vbic_u32(vreinterpret_u32_f32(angle), grey));
}

/* x^7 by products: within two ulps of pow(x, 7) */
static inline float64x2_t pow7(float64x2_t x)
{
    const float64x2_t x2 = vmulq_f64(x, x);
    const float64x2_t x4 = vmulq_f64(x2, x2);
    return vmulq_f64(vmulq_f64(x4, x2), x);
}

/* ciede.c's square_sum_2(): not fused */
static inline float64x2_t square_sum_2(float32x2_t x, float32x2_t y)
{
    const float64x2_t xd = vcvt_f64_f32(x);
    const float64x2_t yd = vcvt_f64_f32(y);
    return vaddq_f64(vmulq_f64(xd, xd), vmulq_f64(yd, yd));
}

/* get_delta_h_prime() */
static inline float32x2_t delta_h_prime(float32x2_t c1, float32x2_t c2, float32x2_t h_prime_1,
                                        float32x2_t h_prime_2)
{
    const float32x2_t h_diff = vsub_f32(h_prime_2, h_prime_1);
    const float64x2_t h_diff_d = vcvt_f64_f32(h_diff);
    /* fabsf(...) <= M_PI compares in double */
    const uint32x2_t near = vmovn_u64(vcleq_f64(vcvt_f64_f32(vabs_f32(h_diff)), vdupq_n_f64(M_PI)));
    const float32x2_t plus = vcvt_f32_f64(vaddq_f64(h_diff_d, vdupq_n_f64(2. * M_PI)));
    const float32x2_t minus = vcvt_f32_f64(vsubq_f64(h_diff_d, vdupq_n_f64(2. * M_PI)));
    float32x2_t delta = vbsl_f32(vcle_f32(h_prime_2, h_prime_1), plus, minus);
    delta = vbsl_f32(near, h_diff, delta);
    const uint32x2_t grey = vorr_u32(vceqz_f32(c1), vceqz_f32(c2));
    return vreinterpret_f32_u32(vbic_u32(vreinterpret_u32_f32(delta), grey));
}

/* get_upcase_h_bar_prime() */
static inline float32x2_t upcase_h_bar_prime(float32x2_t h_prime_1, float32x2_t h_prime_2)
{
    const float64x2_t h_sum = vcvt_f64_f32(vadd_f32(h_prime_1, h_prime_2));
    const float64x2_t h_gap = vcvt_f64_f32(vabs_f32(vsub_f32(h_prime_1, h_prime_2)));
    const uint64x2_t far = vcgtq_f64(h_gap, vdupq_n_f64(M_PI));
    const float64x2_t sum = vbslq_f64(far, vaddq_f64(h_sum, vdupq_n_f64(2.0 * M_PI)), h_sum);
    return vcvt_f32_f64(vmulq_n_f64(sum, 0.5));
}

/* get_upcase_t(): fused as ciede.c is, arguments 3 h + pi / 30 and
 * 4 h - 7 pi / 20 included */
static inline float32x2_t upcase_t(float64x2_t h)
{
    const float64x2_t cos1 = vcos(vsubq_f64(h, vdupq_n_f64(M_PI / 6.0)));
    const float64x2_t cos2 = vcos(vaddq_f64(h, h));
    const float64x2_t cos3 = vcos(vfmaq_n_f64(vdupq_n_f64(M_PI / 30.0), h, 3.0));
    const float64x2_t cos4 = vcos(vfmaq_n_f64(vdupq_n_f64(-(7.0 * M_PI / 20.0)), h, 4.0));
    float64x2_t t = vfmaq_n_f64(vdupq_n_f64(1.0), cos1, -0.17);
    t = vfmaq_n_f64(t, cos2, 0.24);
    t = vfmaq_n_f64(t, cos3, 0.32);
    t = vfmaq_n_f64(t, cos4, -0.20);
    return vcvt_f32_f64(t);
}

/* get_r_sub_t(): powf(c, 7) as the float of the double product, and
 * powf(d, 2) as the float of the exact double square (both what a correctly
 * rounded powf gives, but for the product's rare ties) */
static inline float32x2_t r_sub_t(float32x2_t c_bar_prime, float64x2_t h, float32x2_t k25_7)
{
    const float32x2_t degrees_of_h = vcvt_f32_f64(vmulq_n_f64(h, 180.0 / M_PI));
    const float64x2_t offset = vsubq_f64(vcvt_f64_f32(degrees_of_h), vdupq_n_f64(275.0));
    const float32x2_t degrees = vcvt_f32_f64(vmulq_n_f64(offset, 1.0 / 25.0));
    const float32x2_t c7 = vcvt_f32_f64(pow7(vcvt_f64_f32(c_bar_prime)));
    const float32x2_t ratio = vdiv_f32(c7, vadd_f32(c7, k25_7));
    const float64x2_t degrees_d = vcvt_f64_f32(degrees);
    const float32x2_t degrees_2 = vcvt_f32_f64(vmulq_f64(degrees_d, degrees_d));
    const float32x2_t angle = vcvt_f32_f64(vmulq_n_f64(vexp(vnegq_f64(vcvt_f64_f32(degrees_2))), 60.0));
    const float32x2_t radians = vcvt_f32_f64(vmulq_n_f64(vcvt_f64_f32(angle), M_PI / 180.0));
    const float64x2_t root = vmulq_n_f64(vsqrtq_f64(vcvt_f64_f32(ratio)), -2.0);
    return vcvt_f32_f64(vmulq_f64(root, vsin(vcvt_f64_f32(radians))));
}

/* ciede2000() of pixels i and i + 1 */
static inline float32x2_t ciede2000_2(const float *const lab[6], unsigned i, float32x2_t k25_7)
{
    const float32x2_t l1 = vld1_f32(lab[0] + i);
    const float32x2_t a1 = vld1_f32(lab[1] + i);
    const float32x2_t b1 = vld1_f32(lab[2] + i);
    const float32x2_t l2 = vld1_f32(lab[3] + i);
    const float32x2_t a2 = vld1_f32(lab[4] + i);
    const float32x2_t b2 = vld1_f32(lab[5] + i);

    const float32x2_t delta_l_prime = vsub_f32(l2, l1);
    const float32x2_t l_bar = vmul_n_f32(vadd_f32(l1, l2), 0.5f);
    const float32x2_t c1 = vcvt_f32_f64(vsqrtq_f64(square_sum_2(a1, b1)));
    const float32x2_t c2 = vcvt_f32_f64(vsqrtq_f64(square_sum_2(a2, b2)));
    const float32x2_t c_bar = vmul_n_f32(vadd_f32(c1, c2), 0.5f);
    const float64x2_t c_bar_7 = pow7(vcvt_f64_f32(c_bar));
    const float64x2_t g = vsubq_f64(vdupq_n_f64(1.0),
                                    vsqrtq_f64(vdivq_f64(c_bar_7, vaddq_f64(c_bar_7, vdupq_n_f64(6103515625.0)))));
    /* a + (a / 2) g, fused */
    const float64x2_t half_a1 = vcvt_f64_f32(vmul_n_f32(a1, 0.5f));
    const float64x2_t half_a2 = vcvt_f64_f32(vmul_n_f32(a2, 0.5f));
    const float32x2_t a_prime_1 = vcvt_f32_f64(vfmaq_f64(vcvt_f64_f32(a1), half_a1, g));
    const float32x2_t a_prime_2 = vcvt_f32_f64(vfmaq_f64(vcvt_f64_f32(a2), half_a2, g));
    const float32x2_t c_prime_1 = vcvt_f32_f64(vsqrtq_f64(square_sum_2(a_prime_1, b1)));
    const float32x2_t c_prime_2 = vcvt_f32_f64(vsqrtq_f64(square_sum_2(a_prime_2, b2)));
    const float32x2_t c_bar_prime = vmul_n_f32(vadd_f32(c_prime_1, c_prime_2), 0.5f);
    const float32x2_t delta_c_prime = vsub_f32(c_prime_2, c_prime_1);
    const float64x2_t c_bar_prime_d = vcvt_f64_f32(c_bar_prime);
    const float64x2_t l_offset = vcvt_f64_f32(vsub_f32(l_bar, vdup_n_f32(50.0f)));
    const float64x2_t l_offset_2 = vmulq_f64(l_offset, l_offset);
    const float64x2_t s_sub_l_term =
        vdivq_f64(vmulq_n_f64(l_offset_2, 0.015), vsqrtq_f64(vaddq_f64(vdupq_n_f64(20.0), l_offset_2)));
    const float32x2_t s_sub_l = vcvt_f32_f64(vaddq_f64(vdupq_n_f64(1.0), s_sub_l_term));
    const float32x2_t s_sub_c = vcvt_f32_f64(vfmaq_n_f64(vdupq_n_f64(1.0), c_bar_prime_d, 0.045));
    const float32x2_t h_prime_1 = h_prime(b1, a_prime_1);
    const float32x2_t h_prime_2 = h_prime(b2, a_prime_2);

    const float32x2_t delta_h = delta_h_prime(c1, c2, h_prime_1, h_prime_2);
    const float64x2_t root = vsqrtq_f64(vcvt_f64_f32(vmul_f32(c_prime_1, c_prime_2)));
    const float64x2_t sine = vsin(vmulq_n_f64(vcvt_f64_f32(delta_h), 0.5));
    const float32x2_t delta_upcase_h_prime = vcvt_f32_f64(vmulq_f64(vaddq_f64(root, root), sine));

    const float64x2_t h = vcvt_f64_f32(upcase_h_bar_prime(h_prime_1, h_prime_2));
    const float32x2_t t = upcase_t(h);
    const float64x2_t hue_scale = vmulq_n_f64(c_bar_prime_d, 0.015);
    const float32x2_t s_sub_upcase_h = vcvt_f32_f64(vfmaq_f64(vdupq_n_f64(1.0), hue_scale, vcvt_f64_f32(t)));
    const float32x2_t rt = r_sub_t(c_bar_prime, h, k25_7);

    /* ksub: 0.65, 1 (left out: x * 1 is x), 4 */
    const float32x2_t lightness = vdiv_f32(delta_l_prime, vmul_n_f32(s_sub_l, 0.65f));
    const float32x2_t chroma = vdiv_f32(delta_c_prime, s_sub_c);
    const float32x2_t hue = vdiv_f32(delta_upcase_h_prime, vmul_n_f32(s_sub_upcase_h, 4.0f));
    const float64x2_t ld = vcvt_f64_f32(lightness);
    const float64x2_t cd = vcvt_f64_f32(chroma);
    const float64x2_t hd = vcvt_f64_f32(hue);
    /* ciede.c's square_sum_3(): not fused */
    const float64x2_t squares = vaddq_f64(vaddq_f64(vmulq_f64(ld, ld), vmulq_f64(cd, cd)), vmulq_f64(hd, hd));
    const float64x2_t rotation = vcvt_f64_f32(vmul_f32(vmul_f32(rt, chroma), hue));
    return vcvt_f32_f64(vsqrtq_f64(vaddq_f64(squares, rotation)));
}

void ciede2000_neon(const float *const lab[6], unsigned n, float k25_7, float *de00)
{
    const float32x2_t k = vdup_n_f32(k25_7);
    for (unsigned i = 0; i + 4 <= n; i += 4) {
        const float32x2_t de0 = ciede2000_2(lab, i, k);
        const float32x2_t de1 = ciede2000_2(lab, i + 2, k);
        vst1q_f32(de00 + i, vcombine_f32(de0, de1));
    }
}
