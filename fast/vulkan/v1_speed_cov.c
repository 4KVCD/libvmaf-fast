// Derived from libvmaf (https://github.com/Netflix/vmaf): x86/speed_avx2.c and
// x86/speed_avx512.c, Copyright 2016-2025 Netflix, Inc., licensed under the
// BSD+Patent License (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// VMAF v1's SpEED covariances (v1_speed.c's compute_covariance_matrix), four
// at a time: libvmaf's covariance kernel's arithmetic, in its order, for one
// x and four y's -- x's centred values loaded and subtracted once for the
// four. The planes are read as doubles (v1_speed.c converts them once; the
// kernel's _mm256_cvtps_pd/_mm512_cvtps_pd converts them, exactly, the same).
// Compiled twice, as the kernel is: /arch:AVX2 (v1_cov4_avx2, speed_avx2.c's
// order) and /arch:AVX512 (v1_cov4_avx512, speed_avx512.c's).

#include <stddef.h>
#include <immintrin.h>

// out[k]: the kernel's sum for x (centred on mean_x) and y[k] (on mean_y[k]).
#if defined(__AVX512F__)
void v1_cov4_avx512(const double *x, const double *const y[4], const double mean_y[4], size_t stride,
                    size_t height, size_t width, double mean_x, double out[4])
{
    __m512d a00 = _mm512_setzero_pd(), a01 = _mm512_setzero_pd();
    __m512d a10 = _mm512_setzero_pd(), a11 = _mm512_setzero_pd();
    __m512d a20 = _mm512_setzero_pd(), a21 = _mm512_setzero_pd();
    __m512d a30 = _mm512_setzero_pd(), a31 = _mm512_setzero_pd();
    const __m512d mx = _mm512_set1_pd(mean_x);
    const __m512d my0 = _mm512_set1_pd(mean_y[0]), my1 = _mm512_set1_pd(mean_y[1]);
    const __m512d my2 = _mm512_set1_pd(mean_y[2]), my3 = _mm512_set1_pd(mean_y[3]);
    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;

    for (size_t i = 0; i < height; i++) {
        const double *rx = x + i * stride;
        const double *r0 = y[0] + i * stride, *r1 = y[1] + i * stride;
        const double *r2 = y[2] + i * stride, *r3 = y[3] + i * stride;
        size_t j = 0;

        for (; j + 15 < width; j += 16) {
            const __m512d cx0 = _mm512_sub_pd(_mm512_loadu_pd(rx + j), mx);
            const __m512d cx1 = _mm512_sub_pd(_mm512_loadu_pd(rx + j + 8), mx);
            a00 = _mm512_fmadd_pd(cx0, _mm512_sub_pd(_mm512_loadu_pd(r0 + j), my0), a00);
            a01 = _mm512_fmadd_pd(cx1, _mm512_sub_pd(_mm512_loadu_pd(r0 + j + 8), my0), a01);
            a10 = _mm512_fmadd_pd(cx0, _mm512_sub_pd(_mm512_loadu_pd(r1 + j), my1), a10);
            a11 = _mm512_fmadd_pd(cx1, _mm512_sub_pd(_mm512_loadu_pd(r1 + j + 8), my1), a11);
            a20 = _mm512_fmadd_pd(cx0, _mm512_sub_pd(_mm512_loadu_pd(r2 + j), my2), a20);
            a21 = _mm512_fmadd_pd(cx1, _mm512_sub_pd(_mm512_loadu_pd(r2 + j + 8), my2), a21);
            a30 = _mm512_fmadd_pd(cx0, _mm512_sub_pd(_mm512_loadu_pd(r3 + j), my3), a30);
            a31 = _mm512_fmadd_pd(cx1, _mm512_sub_pd(_mm512_loadu_pd(r3 + j + 8), my3), a31);
        }

        for (; j + 7 < width; j += 8) {
            const __m512d cx = _mm512_sub_pd(_mm512_loadu_pd(rx + j), mx);
            a00 = _mm512_fmadd_pd(cx, _mm512_sub_pd(_mm512_loadu_pd(r0 + j), my0), a00);
            a10 = _mm512_fmadd_pd(cx, _mm512_sub_pd(_mm512_loadu_pd(r1 + j), my1), a10);
            a20 = _mm512_fmadd_pd(cx, _mm512_sub_pd(_mm512_loadu_pd(r2 + j), my2), a20);
            a30 = _mm512_fmadd_pd(cx, _mm512_sub_pd(_mm512_loadu_pd(r3 + j), my3), a30);
        }

        for (; j < width; j++) {
            const double val_x = rx[j];
            t0 += (val_x - mean_x) * (r0[j] - mean_y[0]);
            t1 += (val_x - mean_x) * (r1[j] - mean_y[1]);
            t2 += (val_x - mean_x) * (r2[j] - mean_y[2]);
            t3 += (val_x - mean_x) * (r3[j] - mean_y[3]);
        }
    }

    out[0] = _mm512_reduce_add_pd(_mm512_add_pd(a00, a01)) + t0;
    out[1] = _mm512_reduce_add_pd(_mm512_add_pd(a10, a11)) + t1;
    out[2] = _mm512_reduce_add_pd(_mm512_add_pd(a20, a21)) + t2;
    out[3] = _mm512_reduce_add_pd(_mm512_add_pd(a30, a31)) + t3;
}
#else
static double sum4(__m256d acc0, __m256d acc1, double tail)
{
    const __m256d acc = _mm256_add_pd(acc0, acc1);
    double tmp[4];
    _mm256_storeu_pd(tmp, acc);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tail;
}

void v1_cov4_avx2(const double *x, const double *const y[4], const double mean_y[4], size_t stride,
                  size_t height, size_t width, double mean_x, double out[4])
{
    __m256d a00 = _mm256_setzero_pd(), a01 = _mm256_setzero_pd();
    __m256d a10 = _mm256_setzero_pd(), a11 = _mm256_setzero_pd();
    __m256d a20 = _mm256_setzero_pd(), a21 = _mm256_setzero_pd();
    __m256d a30 = _mm256_setzero_pd(), a31 = _mm256_setzero_pd();
    const __m256d mx = _mm256_set1_pd(mean_x);
    const __m256d my0 = _mm256_set1_pd(mean_y[0]), my1 = _mm256_set1_pd(mean_y[1]);
    const __m256d my2 = _mm256_set1_pd(mean_y[2]), my3 = _mm256_set1_pd(mean_y[3]);
    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;

    for (size_t i = 0; i < height; i++) {
        const double *rx = x + i * stride;
        const double *r0 = y[0] + i * stride, *r1 = y[1] + i * stride;
        const double *r2 = y[2] + i * stride, *r3 = y[3] + i * stride;
        size_t j = 0;

        for (; j + 7 < width; j += 8) {
            const __m256d cx0 = _mm256_sub_pd(_mm256_loadu_pd(rx + j), mx);
            const __m256d cx1 = _mm256_sub_pd(_mm256_loadu_pd(rx + j + 4), mx);
            a00 = _mm256_fmadd_pd(cx0, _mm256_sub_pd(_mm256_loadu_pd(r0 + j), my0), a00);
            a01 = _mm256_fmadd_pd(cx1, _mm256_sub_pd(_mm256_loadu_pd(r0 + j + 4), my0), a01);
            a10 = _mm256_fmadd_pd(cx0, _mm256_sub_pd(_mm256_loadu_pd(r1 + j), my1), a10);
            a11 = _mm256_fmadd_pd(cx1, _mm256_sub_pd(_mm256_loadu_pd(r1 + j + 4), my1), a11);
            a20 = _mm256_fmadd_pd(cx0, _mm256_sub_pd(_mm256_loadu_pd(r2 + j), my2), a20);
            a21 = _mm256_fmadd_pd(cx1, _mm256_sub_pd(_mm256_loadu_pd(r2 + j + 4), my2), a21);
            a30 = _mm256_fmadd_pd(cx0, _mm256_sub_pd(_mm256_loadu_pd(r3 + j), my3), a30);
            a31 = _mm256_fmadd_pd(cx1, _mm256_sub_pd(_mm256_loadu_pd(r3 + j + 4), my3), a31);
        }

        for (; j + 3 < width; j += 4) {
            const __m256d cx = _mm256_sub_pd(_mm256_loadu_pd(rx + j), mx);
            a00 = _mm256_fmadd_pd(cx, _mm256_sub_pd(_mm256_loadu_pd(r0 + j), my0), a00);
            a10 = _mm256_fmadd_pd(cx, _mm256_sub_pd(_mm256_loadu_pd(r1 + j), my1), a10);
            a20 = _mm256_fmadd_pd(cx, _mm256_sub_pd(_mm256_loadu_pd(r2 + j), my2), a20);
            a30 = _mm256_fmadd_pd(cx, _mm256_sub_pd(_mm256_loadu_pd(r3 + j), my3), a30);
        }

        for (; j < width; j++) {
            const double val_x = rx[j];
            t0 += (val_x - mean_x) * (r0[j] - mean_y[0]);
            t1 += (val_x - mean_x) * (r1[j] - mean_y[1]);
            t2 += (val_x - mean_x) * (r2[j] - mean_y[2]);
            t3 += (val_x - mean_x) * (r3[j] - mean_y[3]);
        }
    }

    out[0] = sum4(a00, a01, t0);
    out[1] = sum4(a10, a11, t1);
    out[2] = sum4(a20, a21, t2);
    out[3] = sum4(a30, a31, t3);
}
#endif
