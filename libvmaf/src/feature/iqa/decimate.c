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
 */

#include "decimate.h"
#include <stdlib.h>
#include "cpu.h"
#if ARCH_X86
#include "../x86/decimate_avx2.h"
#endif

/* The largest kernel side the row decimation takes (MS-SSIM's is 9). */
#define MAX_KH 15

/* libvmaf-fast: the factor-2 decimation of _iqa_decimate2_rows. The kh
 * input rows an output row needs are kept in a ring, each split into its
 * even and odd samples: the taps of neighbouring outputs then read
 * consecutive floats. Every output is _iqa_filter_pixel's: the float
 * product of each tap, in its order (rows outer, columns inner), added to a
 * double, rounded to float; taps outside the picture read it mirrored
 * (KBND_SYMMETRIC), which for rows means reading another real row of the
 * ring. */

/* Splits a row of w floats into its even and odd samples. */
static void deinterleave(const float *row, int w, float *even, float *odd, int avx2)
{
#if ARCH_X86
    if (avx2) {
        iqa_deinterleave_avx2(row, w, even, odd);
        return;
    }
#endif
    (void)avx2;
    for (int i = 0; i < w; ++i) {
        if (i & 1)
            odd[i / 2] = row[i];
        else
            even[i / 2] = row[i];
    }
}

/* One output at x whose taps may fall outside the picture's w columns:
 * those read the mirrored column. */
static float edge_pixel(const float *const *even, const float *const *odd,
                        const struct _kernel *k, int w, int x)
{
    const int uc = k->w / 2;
    double sum = 0.0;
    int t = 0;
    for (int v = 0; v < k->h; ++v) {
        for (int u = 0; u < k->w; ++u, ++t) {
            int c = 2 * x - uc + u;
            if (c < 0)
                c = -1 - c;
            else if (c >= w)
                c = 2 * w - 1 - c;
            const float *row = (c & 1) ? odd[v] : even[v];
            sum += row[c / 2] * k->kernel[t];
        }
    }
    return (float)(sum * 1.0f);
}

/* Outputs [x0, x1) whose taps are all inside the picture, without AVX2:
 * input column 2x - uc + u is even sample x + (u - uc) / 2 when u - uc is
 * even, odd sample x + (u - uc - 1) / 2 when it is odd. */
static void inner_pixels(const float *const *even, const float *const *odd,
                         const struct _kernel *k, int x0, int x1, float *dst)
{
    const float *src[MAX_KH * MAX_KH];
    int shift[MAX_KH * MAX_KH];
    const int uc = k->w / 2;
    int taps = 0;
    for (int v = 0; v < k->h; ++v) {
        for (int u = -uc; u <= uc; ++u, ++taps) {
            src[taps] = u & 1 ? odd[v] : even[v];
            shift[taps] = (u - (u & 1)) / 2;
        }
    }
    for (int x = x0; x < x1; ++x) {
        double sum = 0.0;
        for (int t = 0; t < taps; ++t)
            sum += src[t][x + shift[t]] * k->kernel[t];
        dst[x] = (float)(sum * 1.0f);
    }
}

/* The mirrored row of a tap row r of an h-row picture. */
static int mirror_row(int r, int h)
{
    return r < 0 ? -1 - r : r >= h ? 2 * h - 1 - r : r;
}

int _iqa_decimate2_rows(iqa_row_fn row_fn, void *ctx, int w, int h,
                        const struct _kernel *k, float *result, int *rw, int *rh)
{
    /* the mirroring above reaches only rows and columns inside the picture
     * when it is at least as large as the kernel */
    if (!k || !(k->w & 1) || !(k->h & 1) || k->w > MAX_KH || k->h > MAX_KH ||
        k->bnd_opt != KBND_SYMMETRIC || w < k->w || h < k->h)
        return 1;
    const int sw = w / 2 + (w & 1);
    const int sh = h / 2 + (h & 1);
    const int uc = k->w / 2;
    const int vc = k->h / 2;
    const int kh = k->h;
    const int half = sw;  /* even samples in a row (odd ones: w / 2) */
    float *ring = malloc(((size_t)kh * 2 * half + w) * sizeof(float));
    if (!ring)
        return 1;
    float *scratch = ring + (size_t)kh * 2 * half;  /* for row_fn */
    int ring_row[MAX_KH];  /* the input row each ring slot holds */
    for (int i = 0; i < kh; ++i)
        ring_row[i] = -1;
#if ARCH_X86
    const int avx2 = !!(vmaf_get_cpu_flags() & VMAF_X86_CPU_FLAG_AVX2);
#else
    const int avx2 = 0;
#endif

    const int x_lo = (uc + 1) / 2;  /* the first x with 2x - uc >= 0 */
    /* the end of the x with 2x + uc <= w - 1 */
    int x_hi = (w - 1 - uc) / 2 + 1;
    if (x_hi > sw)
        x_hi = sw;
    for (int y = 0; y < sh; ++y) {
        /* an output row's rows (mirrored ones included) lie within kh
         * consecutive rows, so their slots row % kh are all different */
        const float *even[MAX_KH], *odd[MAX_KH];
        for (int v = 0; v < kh; ++v) {
            const int r = mirror_row(2 * y - vc + v, h);
            const int slot = r % kh;
            float *e = ring + (size_t)slot * 2 * half;
            if (ring_row[slot] != r) {
                deinterleave(row_fn(ctx, r, scratch), w, e, e + half, avx2);
                ring_row[slot] = r;
            }
            even[v] = e;
            odd[v] = e + half;
        }
        float *out = result + (size_t)y * sw;
        int x = 0;
        for (; x < x_lo; ++x)
            out[x] = edge_pixel(even, odd, k, w, x);
#if ARCH_X86
        if (avx2)
            x = iqa_decimate2_row_avx2(even, odd, k->kernel, k->w, kh, x, x_hi, out);
#endif
        inner_pixels(even, odd, k, x, x_hi, out);
        for (x = x_hi; x < sw; ++x)
            out[x] = edge_pixel(even, odd, k, w, x);
    }
    free(ring);
    if (rw) *rw = sw;
    if (rh) *rh = sh;
    return 0;
}

/* A float picture's rows for _iqa_decimate2_rows, read in place. */
struct float_rows {
    const float *img;
    int w;
};

static const float *float_row(void *ctx, int r, float *scratch)
{
    const struct float_rows *f = ctx;
    (void)scratch;
    return f->img + (size_t)r * f->w;
}

int _iqa_decimate(float *img, int w, int h, int factor, const struct _kernel *k, float *result, int *rw, int *rh)
{
    int x,y;
    int sw = w/factor + (w&1);
    int sh = h/factor + (h&1);
    int dst_offset;
    float *dst=img;

    if (result)
        dst = result;

#if ARCH_X86
    /* libvmaf-fast: MS-SSIM's factor-2 decimation into another buffer, with
     * AVX2 (x86/decimate_avx2.c). In place, a pixel's taps can read outputs
     * already written, which only the scalar loop reproduces. */
    struct float_rows rows = { img, w };
    if (factor == 2 && result && result != img &&
        (vmaf_get_cpu_flags() & VMAF_X86_CPU_FLAG_AVX2) &&
        !_iqa_decimate2_rows(float_row, &rows, w, h, k, dst, rw, rh))
        return 0;
#endif

    /* Downsample */
    for (y=0; y<sh; ++y) {
        dst_offset = y*sw;
        for (x=0; x<sw; ++x,++dst_offset) {
            dst[dst_offset] = _iqa_filter_pixel(img, w, h, x*factor, y*factor, k, 1.0f);
        }
    }

    if (rw) *rw = sw;
    if (rh) *rh = sh;
    return 0;
}
