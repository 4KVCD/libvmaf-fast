// The x86 SIMD intrinsics VMAF v1's SpEED code uses (v1_speed.c,
// v1_speed_cov.c and libvmaf's x86/speed_avx2.c), in plain C, for a build on
// a CPU without them (Apple's): included in place of the compiler's
// <immintrin.h> by fast/scripts/build_vmaf_vulkan_macos.sh (-I portable).
//
// Each is the same IEEE operation, lane by lane, as the instruction it stands
// for, so the sums come out exactly as on x86, in the same order: the float
// adds, multiplies, subtracts and divides are single roundings (built without
// contraction, -ffp-contract=off), _mm256_fmadd_pd is fma() (one rounding, as
// the instruction), and the conversions are exact. Only the AVX2 kernels are
// built so (v1_speed.c takes them where cpuid is not), so a Mac's VMAF v1 is
// that of an x86 PC with AVX2 and without AVX-512.
#pragma once

#include <math.h>
#include <string.h>

typedef struct { float f[4]; } __m128;
typedef struct { double d[4]; } __m256d;

#define VV_INLINE static inline

VV_INLINE __m128 _mm_setzero_ps(void) { __m128 r = {{0.0f, 0.0f, 0.0f, 0.0f}}; return r; }
VV_INLINE __m128 _mm_set1_ps(float a) { __m128 r = {{a, a, a, a}}; return r; }
VV_INLINE __m128 _mm_set_ss(float a) { __m128 r = {{a, 0.0f, 0.0f, 0.0f}}; return r; }
VV_INLINE __m128 _mm_load_ss(const float *p) { __m128 r = {{p[0], 0.0f, 0.0f, 0.0f}}; return r; }
VV_INLINE __m128 _mm_loadu_ps(const float *p) { __m128 r; memcpy(r.f, p, sizeof r.f); return r; }
VV_INLINE void _mm_storeu_ps(float *p, __m128 a) { memcpy(p, a.f, sizeof a.f); }
VV_INLINE float _mm_cvtss_f32(__m128 a) { return a.f[0]; }

#define VV_LANES4(name, op)                                                      \
    VV_INLINE __m128 name(__m128 a, __m128 b)                                    \
    {                                                                            \
        __m128 r;                                                                \
        for (int i = 0; i < 4; ++i)                                              \
            r.f[i] = a.f[i] op b.f[i];                                           \
        return r;                                                                \
    }
VV_LANES4(_mm_add_ps, +)
VV_LANES4(_mm_sub_ps, -)
VV_LANES4(_mm_mul_ps, *)
VV_LANES4(_mm_div_ps, /)
#undef VV_LANES4

// The scalar forms: the first lane done, the others a's.
VV_INLINE __m128 _mm_add_ss(__m128 a, __m128 b) { a.f[0] = a.f[0] + b.f[0]; return a; }
VV_INLINE __m128 _mm_mul_ss(__m128 a, __m128 b) { a.f[0] = a.f[0] * b.f[0]; return a; }

VV_INLINE __m256d _mm256_setzero_pd(void) { __m256d r = {{0.0, 0.0, 0.0, 0.0}}; return r; }
VV_INLINE __m256d _mm256_set1_pd(double a) { __m256d r = {{a, a, a, a}}; return r; }
VV_INLINE __m256d _mm256_loadu_pd(const double *p) { __m256d r; memcpy(r.d, p, sizeof r.d); return r; }
VV_INLINE void _mm256_storeu_pd(double *p, __m256d a) { memcpy(p, a.d, sizeof a.d); }
VV_INLINE __m256d _mm256_cvtps_pd(__m128 a)
{
    __m256d r = {{(double)a.f[0], (double)a.f[1], (double)a.f[2], (double)a.f[3]}};
    return r;
}
VV_INLINE __m256d _mm256_add_pd(__m256d a, __m256d b)
{
    __m256d r;
    for (int i = 0; i < 4; ++i)
        r.d[i] = a.d[i] + b.d[i];
    return r;
}
VV_INLINE __m256d _mm256_sub_pd(__m256d a, __m256d b)
{
    __m256d r;
    for (int i = 0; i < 4; ++i)
        r.d[i] = a.d[i] - b.d[i];
    return r;
}
// a*b + c rounded once, as VFMADD does.
VV_INLINE __m256d _mm256_fmadd_pd(__m256d a, __m256d b, __m256d c)
{
    __m256d r;
    for (int i = 0; i < 4; ++i)
        r.d[i] = fma(a.d[i], b.d[i], c.d[i]);
    return r;
}

#undef VV_INLINE
