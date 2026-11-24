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

/* ciede.c's ciede2000(), four pixels at a time.
 *
 * Every add, multiply, divide and square root is ciede2000()'s own, in its
 * types (float or double) and order, so each lane rounds where the scalar
 * code rounds. Only the C library's atan2, sin, cos, exp and pow differ:
 * they are replaced by the double-precision functions below (within about
 * an ulp of a double), and pow(x, 7) by products. Their results are rounded
 * to float right after, as in ciede2000(), which hides the difference in
 * all but rare cases: a pixel's dE00 then differs in the last bit of its
 * float, and a frame's score in about the 12th digit.
 *
 * The polynomials and argument reductions are fdlibm's (k_sin.c, k_cos.c,
 * e_exp.c, s_atan.c, e_atan2.c: Copyright (C) 1993 by Sun Microsystems,
 * Inc. Developed at SunSoft, a Sun Microsystems, Inc. business. Permission
 * to use, copy, modify, and distribute this software is freely granted,
 * provided that this notice is preserved.)
 *
 * Compiled without FMA contraction (meson.build's avx2_cflags; MSVC does not
 * contract), so a * b + c rounds twice, as in the scalar code. */

#include <immintrin.h>
#include <math.h>

#include "ciede_avx2.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef __m256d vd; /* four doubles */
typedef __m128 vf;  /* four floats */

static inline vd cd(double x) { return _mm256_set1_pd(x); }
static inline vf cf(float x) { return _mm_set1_ps(x); }
static inline vd to_d(vf x) { return _mm256_cvtps_pd(x); }
static inline vf to_f(vd x) { return _mm256_cvtpd_ps(x); }
static inline vd add(vd a, vd b) { return _mm256_add_pd(a, b); }
static inline vd sub(vd a, vd b) { return _mm256_sub_pd(a, b); }
static inline vd mul(vd a, vd b) { return _mm256_mul_pd(a, b); }
static inline vd dvd(vd a, vd b) { return _mm256_div_pd(a, b); }
/* mask ? b : a */
static inline vd sel(vd a, vd b, vd mask) { return _mm256_blendv_pd(a, b, mask); }
static inline vf self(vf a, vf b, vf mask) { return _mm_blendv_ps(a, b, mask); }
/* A double compare's mask as a float one: blendv reads only the sign bit,
 * which the conversion keeps (all ones is a negative NaN, 0 stays 0). */
static inline vf mask_to_f(vd mask) { return _mm256_cvtpd_ps(mask); }
static inline vd absd(vd x) { return _mm256_andnot_pd(cd(-0.0), x); }
static inline vf absf(vf x) { return _mm_andnot_ps(cf(-0.0f), x); }

/* Doubles that hold whole numbers (|k| < 2^51) as 64-bit integers. */
static inline __m256i whole_to_i64(vd k)
{
    const vd magic = cd(6755399441055744.0); /* 2^52 + 2^51 */
    return _mm256_sub_epi64(_mm256_castpd_si256(add(k, magic)), _mm256_castpd_si256(magic));
}

/* sin(r) and cos(r) for |r| <= pi/4 (fdlibm's __kernel_sin and __kernel_cos,
 * the latter without its correction for |r| > 0.3, which is worth less than
 * the float rounding after it). */
static inline vd kernel_sin(vd r)
{
    const vd z = mul(r, r), v = mul(z, r);
    vd p = add(cd(-2.50507602534068634195e-08), mul(z, cd(1.58969099521155010221e-10)));
    p = add(cd(2.75573137070700676789e-06), mul(z, p));
    p = add(cd(-1.98412698298579493134e-04), mul(z, p));
    p = add(cd(8.33333333332248946124e-03), mul(z, p));
    return add(r, mul(v, add(cd(-1.66666666666666324348e-01), mul(z, p))));
}

static inline vd kernel_cos(vd r)
{
    const vd z = mul(r, r);
    vd p = add(cd(2.08757232129817482790e-09), mul(z, cd(-1.13596475577881948265e-11)));
    p = add(cd(-2.75573143513906633035e-07), mul(z, p));
    p = add(cd(2.48015872894767294178e-05), mul(z, p));
    p = add(cd(-1.38888888888741095749e-03), mul(z, p));
    p = add(cd(4.16666666666666019037e-02), mul(z, p));
    return sub(cd(1.0), sub(mul(cd(0.5), z), mul(z, mul(z, p))));
}

/* x - k pi/2 with k the nearest whole number, and k's last two bits (the
 * quadrant), for |x| < 2^20 (Cody and Waite's reduction with fdlibm's two
 * parts of pi/2: k * pio2_1 is exact). */
static inline vd reduce_pio2(vd x, __m256i *quadrant)
{
    const vd k = _mm256_round_pd(mul(x, cd(6.36619772367581382433e-01)),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    *quadrant = whole_to_i64(k);
    return sub(sub(x, mul(k, cd(1.57079632673412561417e+00))), mul(k, cd(6.07710050650619224932e-11)));
}

static inline vd quadrant_is(__m256i quadrant, long long bit)
{
    const __m256i b = _mm256_set1_epi64x(bit);
    return _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(quadrant, b), b));
}

static inline vd vsin(vd x)
{
    __m256i q;
    const vd r = reduce_pio2(x, &q);
    const vd s = sel(kernel_sin(r), kernel_cos(r), quadrant_is(q, 1));
    return _mm256_xor_pd(s, _mm256_and_pd(quadrant_is(q, 2), cd(-0.0)));
}

static inline vd vcos(vd x)
{
    __m256i q;
    const vd r = reduce_pio2(x, &q);
    const vd c = sel(kernel_cos(r), kernel_sin(r), quadrant_is(q, 1));
    /* quadrants 1 and 2 negative: bit 0 xor bit 1 */
    const vd negative = _mm256_xor_pd(quadrant_is(q, 1), quadrant_is(q, 2));
    return _mm256_xor_pd(c, _mm256_and_pd(negative, cd(-0.0)));
}

/* exp(x) for -700 < x < 700 (fdlibm's __ieee754_exp). */
static inline vd vexp(vd x)
{
    const vd k = _mm256_round_pd(mul(x, cd(1.44269504088896338700e+00)),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const vd hi = sub(x, mul(k, cd(6.93147180369123816490e-01)));
    const vd lo = mul(k, cd(1.90821492927058770002e-10));
    const vd r = sub(hi, lo);
    const vd t = mul(r, r);
    vd p = add(cd(-1.65339022054652515390e-06), mul(t, cd(4.13813679705723846039e-08)));
    p = add(cd(6.61375632143793436117e-05), mul(t, p));
    p = add(cd(-2.77777777770155933842e-03), mul(t, p));
    p = add(cd(1.66666666666666019037e-01), mul(t, p));
    const vd c = sub(r, mul(t, p));
    const vd y = sub(cd(1.0), sub(sub(lo, dvd(mul(r, c), sub(cd(2.0), c))), hi));
    /* y * 2^k, by adding k to y's exponent */
    return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(y), _mm256_slli_epi64(whole_to_i64(k), 52)));
}

/* atan2(y, x), not both 0 (fdlibm's __ieee754_atan2 and atan). */
static inline vd vatan2(vd y, vd x)
{
    const vd t = dvd(absd(y), absd(x)); /* +inf where x is 0 */
    /* fdlibm's intervals: t below 7/16, 11/16, 19/16, 39/16, and above */
    const vd i0 = _mm256_cmp_pd(t, cd(0.4375), _CMP_GE_OQ);
    const vd i1 = _mm256_cmp_pd(t, cd(0.6875), _CMP_GE_OQ);
    const vd i2 = _mm256_cmp_pd(t, cd(1.1875), _CMP_GE_OQ);
    const vd i3 = _mm256_cmp_pd(t, cd(2.4375), _CMP_GE_OQ);
    vd num = t, den = cd(1.0);
    num = sel(num, sub(mul(cd(2.0), t), cd(1.0)), i0);
    den = sel(den, add(cd(2.0), t), i0);
    num = sel(num, sub(t, cd(1.0)), i1);
    den = sel(den, add(t, cd(1.0)), i1);
    num = sel(num, sub(t, cd(1.5)), i2);
    den = sel(den, add(cd(1.0), mul(cd(1.5), t)), i2);
    num = sel(num, cd(-1.0), i3);
    den = sel(den, t, i3);
    const vd u = dvd(num, den);
    const vd z = mul(u, u), w = mul(z, z);
    vd s1 = add(cd(4.97687799461593236017e-02), mul(w, cd(1.62858201153657823623e-02)));
    s1 = add(cd(6.66107313738753120669e-02), mul(w, s1));
    s1 = add(cd(9.09088713343650656196e-02), mul(w, s1));
    s1 = add(cd(1.42857142725034663711e-01), mul(w, s1));
    s1 = mul(z, add(cd(3.33333333333329318027e-01), mul(w, s1)));
    vd s2 = add(cd(-5.83357013379057348645e-02), mul(w, cd(-3.65315727442169155270e-02)));
    s2 = add(cd(-7.69187620504482999495e-02), mul(w, s2));
    s2 = add(cd(-1.11111104054623557880e-01), mul(w, s2));
    s2 = mul(w, add(cd(-1.99999999998764832476e-01), mul(w, s2)));
    const vd us = mul(u, add(s1, s2));
    vd hi = sel(cd(0.0), cd(4.63647609000806093515e-01), i0), lo = sel(cd(0.0), cd(2.26987774529616870924e-17), i0);
    hi = sel(hi, cd(7.85398163397448278999e-01), i1);
    lo = sel(lo, cd(3.06161699786838301793e-17), i1);
    hi = sel(hi, cd(9.82793723247329054082e-01), i2);
    lo = sel(lo, cd(1.39033110312309984516e-17), i2);
    hi = sel(hi, cd(1.57079632679489655800e+00), i3);
    lo = sel(lo, cd(6.12323399573676603587e-17), i3);
    const vd a = sel(sub(u, us), sub(hi, sub(sub(us, lo), u)), i0); /* atan(t) */
    /* the quadrant: x < 0 gives pi - a, y < 0 negates (x = -0 counts as
     * x = 0, pi/2, as in fdlibm) */
    const vd x_negative = _mm256_cmp_pd(x, cd(0.0), _CMP_LT_OQ);
    const vd r = sel(a, sub(cd(3.1415926535897931160E+00), sub(a, cd(1.2246467991473531772E-16))), x_negative);
    return _mm256_or_pd(r, _mm256_and_pd(y, cd(-0.0)));
}

/* get_h_prime(): the hue angle in [0, 2 pi), 0 for a grey. */
static inline vf h_prime(vf x, vf y)
{
    vf hue = to_f(vatan2(to_d(x), to_d(y)));
    hue = self(hue, to_f(add(to_d(hue), cd(2. * M_PI))), _mm_cmplt_ps(hue, cf(0.0f)));
    const vf grey = _mm_and_ps(_mm_cmpeq_ps(x, cf(0.0f)), _mm_cmpeq_ps(y, cf(0.0f)));
    return _mm_andnot_ps(grey, hue);
}

/* x^7 by products: within two ulps of pow(x, 7) */
static inline vd pow7(vd x)
{
    const vd x2 = mul(x, x), x4 = mul(x2, x2);
    return mul(mul(x4, x2), x);
}

static inline vd square_sum_2(vd x, vd y) { return add(mul(x, x), mul(y, y)); }

static inline vf ciede2000_4(vf l1, vf a1, vf b1, vf l2, vf a2, vf b2, vf k25_7)
{
    const vf two = cf(2.0f);
    const vf delta_l_prime = _mm_sub_ps(l2, l1);
    const vf l_bar = _mm_div_ps(_mm_add_ps(l1, l2), two);
    const vf c1 = to_f(_mm256_sqrt_pd(square_sum_2(to_d(a1), to_d(b1))));
    const vf c2 = to_f(_mm256_sqrt_pd(square_sum_2(to_d(a2), to_d(b2))));
    const vf c_bar = _mm_div_ps(_mm_add_ps(c1, c2), two);
    const vd c_bar_7 = pow7(to_d(c_bar));
    const vd g = sub(cd(1.0), _mm256_sqrt_pd(dvd(c_bar_7, add(c_bar_7, cd(6103515625.0)))));
    const vf a_prime_1 = to_f(add(to_d(a1), mul(to_d(_mm_div_ps(a1, two)), g)));
    const vf a_prime_2 = to_f(add(to_d(a2), mul(to_d(_mm_div_ps(a2, two)), g)));
    const vf c_prime_1 = to_f(_mm256_sqrt_pd(square_sum_2(to_d(a_prime_1), to_d(b1))));
    const vf c_prime_2 = to_f(_mm256_sqrt_pd(square_sum_2(to_d(a_prime_2), to_d(b2))));
    const vf c_bar_prime = _mm_div_ps(_mm_add_ps(c_prime_1, c_prime_2), two);
    const vf delta_c_prime = _mm_sub_ps(c_prime_2, c_prime_1);
    const vd l_offset = to_d(_mm_sub_ps(l_bar, cf(50.0f)));
    const vd l_offset_2 = mul(l_offset, l_offset);
    const vf s_sub_l = to_f(add(cd(1.), dvd(mul(cd(0.015), l_offset_2), _mm256_sqrt_pd(add(cd(20.0), l_offset_2)))));
    const vf s_sub_c = to_f(add(cd(1.), mul(cd(0.045), to_d(c_bar_prime))));
    const vf h_prime_1 = h_prime(b1, a_prime_1);
    const vf h_prime_2 = h_prime(b2, a_prime_2);

    /* get_delta_h_prime() */
    const vf h_diff = _mm_sub_ps(h_prime_2, h_prime_1);
    const vd h_diff_d = to_d(h_diff);
    const vd near = _mm256_cmp_pd(to_d(absf(h_diff)), cd(M_PI), _CMP_LE_OQ);
    const vf h2_le_h1 = _mm_cmple_ps(h_prime_2, h_prime_1);
    vf delta_h_prime = self(to_f(sub(h_diff_d, cd(2. * M_PI))), to_f(add(h_diff_d, cd(2. * M_PI))), h2_le_h1);
    delta_h_prime = self(delta_h_prime, h_diff, mask_to_f(near));
    const vf grey = _mm_or_ps(_mm_cmpeq_ps(c1, cf(0.0f)), _mm_cmpeq_ps(c2, cf(0.0f)));
    delta_h_prime = _mm_andnot_ps(grey, delta_h_prime);

    const vf delta_upcase_h_prime =
        to_f(mul(mul(cd(2.0), _mm256_sqrt_pd(to_d(_mm_mul_ps(c_prime_1, c_prime_2)))),
                 vsin(dvd(to_d(delta_h_prime), cd(2.0)))));

    /* get_upcase_h_bar_prime() */
    const vd h_sum = to_d(_mm_add_ps(h_prime_1, h_prime_2));
    const vd far = _mm256_cmp_pd(to_d(absf(_mm_sub_ps(h_prime_1, h_prime_2))), cd(M_PI), _CMP_GT_OQ);
    const vf upcase_h_bar_prime = to_f(sel(dvd(h_sum, cd(2.0)), dvd(add(h_sum, cd(2.0 * M_PI)), cd(2.0)), far));
    const vd h = to_d(upcase_h_bar_prime);

    /* get_upcase_t() */
    vd t = sub(cd(1.0), mul(cd(0.17), vcos(sub(h, cd(M_PI / 6.0)))));
    t = add(t, mul(cd(0.24), vcos(mul(cd(2.0), h))));
    t = add(t, mul(cd(0.32), vcos(add(mul(cd(3.0), h), cd(M_PI / 30.0)))));
    t = sub(t, mul(cd(0.20), vcos(sub(mul(cd(4.0), h), cd(7.0 * M_PI / 20.0)))));
    const vf upcase_t = to_f(t);
    const vf s_sub_upcase_h = to_f(add(cd(1.0), mul(mul(cd(0.015), to_d(c_bar_prime)), to_d(upcase_t))));

    /* get_r_sub_t(): powf(c, 7) as the float of the double product, and
     * powf(d, 2) as the float of the exact double square (both what a
     * correctly rounded powf gives, but for the product's rare ties) */
    const vf degrees_of_h = to_f(mul(h, cd(180.0 / M_PI)));
    const vf degrees = to_f(mul(sub(to_d(degrees_of_h), cd(275.0)), cd(1.0 / 25.0)));
    const vf c7 = to_f(pow7(to_d(c_bar_prime)));
    const vf ratio = _mm_div_ps(c7, _mm_add_ps(c7, k25_7));
    const vd degrees_d = to_d(degrees);
    const vf degrees_2 = to_f(mul(degrees_d, degrees_d));
    const vf angle = to_f(mul(cd(60.0), vexp(to_d(_mm_xor_ps(degrees_2, cf(-0.0f))))));
    const vf radians = to_f(mul(to_d(angle), cd(M_PI / 180.0)));
    const vf r_sub_t = to_f(mul(mul(cd(-2.0), _mm256_sqrt_pd(to_d(ratio))), vsin(to_d(radians))));

    /* ksub: 0.65, 1, 4 */
    const vf lightness = _mm_div_ps(delta_l_prime, _mm_mul_ps(cf(0.65f), s_sub_l));
    const vf chroma = _mm_div_ps(delta_c_prime, _mm_mul_ps(cf(1.0f), s_sub_c));
    const vf hue = _mm_div_ps(delta_upcase_h_prime, _mm_mul_ps(cf(4.0f), s_sub_upcase_h));
    const vd ld = to_d(lightness), cdd = to_d(chroma), hd = to_d(hue);
    const vd squares = add(add(mul(ld, ld), mul(cdd, cdd)), mul(hd, hd));
    return to_f(_mm256_sqrt_pd(add(squares, to_d(_mm_mul_ps(_mm_mul_ps(r_sub_t, chroma), hue)))));
}

void ciede2000_avx2(const float *const lab[6], unsigned n, float k25_7, float *de00)
{
    const vf k = cf(k25_7);
    for (unsigned i = 0; i + 4 <= n; i += 4) {
        const vf out = ciede2000_4(_mm_loadu_ps(lab[0] + i), _mm_loadu_ps(lab[1] + i), _mm_loadu_ps(lab[2] + i),
                                   _mm_loadu_ps(lab[3] + i), _mm_loadu_ps(lab[4] + i), _mm_loadu_ps(lab[5] + i), k);
        _mm_storeu_ps(de00 + i, out);
    }
}
