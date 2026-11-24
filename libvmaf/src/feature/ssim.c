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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include <errno.h>

#include "mem.h"
#include "iqa/math_utils.h"
#include "iqa/decimate.h"
#include "iqa/ssim_tools.h"
#include "ssim.h"
#include "cpu.h"
/* libvmaf-fast's SIMD box sums: AVX2 (x86/ssim_decimate_avx2.c) or NEON
 * (arm64/ssim_decimate_neon.c) */
#if ARCH_X86
#include "x86/ssim_decimate_avx2.h"
#define SSIM_DECIMATE_SIMD_FLAG VMAF_X86_CPU_FLAG_AVX2
#define ssim_column_sums_simd_8 ssim_column_sums_avx2_8
#define ssim_column_sums_simd_16 ssim_column_sums_avx2_16
#define ssim_box_sums_simd ssim_box_sums_avx2
#define ssim_column_table_sums_simd_8 ssim_column_table_sums_avx2_8
#define ssim_column_table_sums_simd_16 ssim_column_table_sums_avx2_16
#define ssim_box_table_sums_simd ssim_box_table_sums_avx2
#elif ARCH_AARCH64
#include "arm64/ssim_decimate_neon.h"
#define SSIM_DECIMATE_SIMD_FLAG VMAF_ARM_CPU_FLAG_NEON
#define ssim_column_sums_simd_8 ssim_column_sums_neon_8
#define ssim_column_sums_simd_16 ssim_column_sums_neon_16
#define ssim_box_sums_simd ssim_box_sums_neon
#define ssim_column_table_sums_simd_8 ssim_column_table_sums_neon_8
#define ssim_column_table_sums_simd_16 ssim_column_table_sums_neon_16
#define ssim_box_table_sums_simd ssim_box_table_sums_neon
#endif

/* _ssim_map */
int _ssim_map(const struct _ssim_int *si, void *ctx)
{
    double *ssim_sum = (double*)ctx;
    *ssim_sum += si->l * si->c * si->s;
    return 0;
}

/* _ssim_reduce */
float _ssim_reduce(int w, int h, void *ctx)
{
    double *ssim_sum = (double*)ctx;
    return (float)(*ssim_sum / (double)(w*h));
}

int ssim_scale(int w, int h, int scale_override)
{
    if (scale_override > 0)
        return scale_override;
    return _max( 1, _round( (float)_min(w,h) / 256.0f ) );
}

/*
 * libvmaf-fast: compute_ssim()'s decimation, made from a picture's samples.
 *
 * compute_ssim() converts each picture to floats (picture_copy: the sample,
 * divided by 4, 16 or 256 above 8 bits), copies that into a second buffer,
 * and decimates it in place (_iqa_decimate): each output sample is
 * _iqa_filter_pixel() of the scale x scale box, every float value times the
 * float 1/(scale*scale), the products added up in a double, mirrored at the
 * edges by KBND_SYMMETRIC. At 4K that is two 33 MB float images written and
 * read again a frame, which made float_ssim memory-bound.
 *
 * Here each output is the same sum, from the samples. Every product is an
 * exact float (a value of up to 16 significant bits times one float), and
 * every partial sum of up to 100 of them is exact in a double (they span
 * fewer than 53 bits), so the order they are added in cannot change the
 * result. Where 1/(scale*scale) is a power of two (scales 2, 4 and 8: 540p,
 * 1080p and 4K), each product is the value times it exactly, and the sum is
 * the integer sum of the samples times one power of two. Otherwise each
 * sample's product is looked up in a table of them.
 *
 * With SIMD (x86/ssim_decimate_avx2.c, arm64/ssim_decimate_neon.c) each
 * output row is made in two steps, the sums of each column's `scale` rows
 * and then the sums of `scale` of those; the boxes at the left and right
 * edges, which mirror, from the same column sums here.
 */
static int ssim_mirror(int i, int n)
{
    if (i < 0) return -1 - i;                /* KBND_SYMMETRIC */
    if (i >= n) return (n - (i - n)) - 1;
    return i;
}

/* Everything compute_ssim_decimated needs for pictures of one size, bit
 * depth and scale, kept from frame to frame: at 1080p and 4K each float
 * buffer is just over 512 KB, which Windows' heap gives straight from the
 * system, so allocating them every frame took page faults worth a fifth of
 * the SSIM's time. */
struct SsimWorkspace {
    int w, h, scale, sw, sh;
    unsigned bpc;
    double unit;          /* 1/(scale*scale) over the conversion's divisor */
    double *table;        /* each sample's float product; NULL: unit works */
    bool simd_sums;       /* integer box sums fit ssim_box_sums_simd */
    int *columns;         /* each output's box columns, mirrored: sw * scale */
    double *acc;          /* the C code's row of box sums: sw */
    uint32_t *col;        /* integer column sums: w */
    double *col_table;    /* table column sums: w */
    const uint8_t **rows8;    /* the box rows of one output row: scale */
    const uint16_t **rows16;
    float *ref_f, *cmp_f; /* the decimated pictures: sw * sh each */
    float *iqa;           /* _iqa_ssim_ws's buffers: 6 * sw * sh */
    bool borrowed_floats; /* ref_f, cmp_f and iqa are another workspace's */
};

/* One side of a decimated picture, as compute_ssim() makes it. */
static int decimated_side(int n, int scale)
{
    return n / scale + (n & 1);
}

size_t ssim_decimated_samples(int w, int h, int scale)
{
    return (size_t)decimated_side(w, scale) * decimated_side(h, scale);
}

SsimWorkspace *ssim_workspace_alloc(int w, int h, int scale, unsigned bpc,
                                    SsimWorkspace *floats_from)
{
    if (scale <= 1 || w <= 0 || h <= 0)
        return NULL;
    if (bpc != 8 && bpc != 10 && bpc != 12 && bpc != 16)
        return NULL;  /* picture_copy reads others as 8-bit */
    if (floats_from &&
        ssim_decimated_samples(floats_from->w, floats_from->h, floats_from->scale) <
            ssim_decimated_samples(w, h, scale))
        return NULL;  /* its float buffers are too small for these pictures */
    SsimWorkspace *ws = calloc(1, sizeof(*ws));
    if (!ws) return NULL;
    ws->w = w;
    ws->h = h;
    ws->scale = scale;
    ws->bpc = bpc;
    ws->sw = decimated_side(w, scale);
    ws->sh = decimated_side(h, scale);
    const float divisor = bpc == 10 ? 4.0f : bpc == 12 ? 16.0f : bpc == 16 ? 256.0f : 1.0f;
    ws->unit = 1.0 / (double)(scale * scale) / (double)divisor;
    const int power_of_two = (scale & (scale - 1)) == 0;
    if (!power_of_two) {
        /* low_pass.kernel[] as compute_ssim() makes it */
        const float k = 1.0f / (scale * scale);
        const size_t values = (size_t)1 << bpc;
        ws->table = malloc(sizeof(double) * values);
        if (!ws->table) goto fail;
        for (size_t i = 0; i < values; i++) {
            /* picture_copy's value, times the kernel's float, in float */
            const float value = bpc == 8 ? (float)i + 0 : (float)i / divisor + 0;
            const float product = value * k;
            ws->table[i] = (double)product;
        }
    }
    const uint64_t max_box = (uint64_t)scale * scale * ((1u << bpc) - 1);
    ws->simd_sums = power_of_two && max_box < (1u << 24);
    const size_t small = (size_t)ws->sw * ws->sh;
    ws->columns = malloc(sizeof(int) * ws->sw * scale);
    ws->acc = malloc(sizeof(double) * ws->sw);
    ws->col = malloc(sizeof(uint32_t) * w);
    ws->col_table = malloc(sizeof(double) * w);
    ws->rows8 = malloc(sizeof(*ws->rows8) * scale);
    ws->rows16 = malloc(sizeof(*ws->rows16) * scale);
    if (floats_from) {
        /* compute_ssim_decimated writes all it reads of them in each call */
        ws->borrowed_floats = true;
        ws->ref_f = floats_from->ref_f;
        ws->cmp_f = floats_from->cmp_f;
        ws->iqa = floats_from->iqa;
    } else {
        ws->ref_f = malloc(sizeof(float) * small);
        ws->cmp_f = malloc(sizeof(float) * small);
        ws->iqa = malloc(sizeof(float) * small * 6);
    }
    if (!ws->columns || !ws->acc || !ws->col || !ws->col_table || !ws->rows8 ||
        !ws->rows16 || !ws->ref_f || !ws->cmp_f || !ws->iqa)
        goto fail;
    /* _iqa_filter_pixel's window: the scale samples from x * scale - uc */
    const int uc = scale / 2;
    for (int x = 0; x < ws->sw; x++)
        for (int u = 0; u < scale; u++)
            ws->columns[x * scale + u] = ssim_mirror(x * scale - uc + u, w);
    return ws;
fail:
    ssim_workspace_free(ws);
    return NULL;
}

void ssim_workspace_free(SsimWorkspace *ws)
{
    if (!ws) return;
    free(ws->table);
    free(ws->columns);
    free(ws->acc);
    free(ws->col);
    free(ws->col_table);
    free(ws->rows8);
    free(ws->rows16);
    if (!ws->borrowed_floats) {
        free(ws->ref_f);
        free(ws->cmp_f);
        free(ws->iqa);
    }
    free(ws);
}

#ifdef SSIM_DECIMATE_SIMD_FLAG
/* Output x of a row from the row's column sums, its box mirrored at the
 * picture's edge (ws->columns), as the C code's sum. */
static float mirrored_box_sum(const SsimWorkspace *ws, int x)
{
    const int *c = ws->columns + x * ws->scale;
    if (ws->table) {
        double sum = 0.0;
        for (int u = 0; u < ws->scale; u++)
            sum += ws->col_table[c[u]];
        return (float)sum;
    }
    uint32_t sum = 0;
    for (int u = 0; u < ws->scale; u++)
        sum += ws->col[c[u]];
    return (float)((double)sum * ws->unit);
}

/* One output row from its box rows (ws->rows8 or rows16) with SIMD: the
 * column sums, the box sums inside the picture, and the mirrored boxes at
 * the edges from the same column sums. */
static void decimate_row_simd(const SsimWorkspace *ws, int wide, float *out)
{
    const int scale = ws->scale, sw = ws->sw, uc = scale / 2;
    /* outputs whose box lies inside the picture: x * scale - uc >= 0 and
     * x * scale - uc + scale <= w */
    const int inside_lo = 1;
    int inside_hi = (ws->w + uc - scale) / scale + 1;
    if (inside_hi > sw) inside_hi = sw;
    if (inside_hi < inside_lo) inside_hi = inside_lo;
    if (ws->table) {
        if (wide)
            ssim_column_table_sums_simd_16(ws->rows16, scale, ws->w, ws->table, ws->col_table);
        else
            ssim_column_table_sums_simd_8(ws->rows8, scale, ws->w, ws->table, ws->col_table);
        ssim_box_table_sums_simd(ws->col_table, scale, inside_lo, inside_hi, out);
    } else {
        if (wide)
            ssim_column_sums_simd_16(ws->rows16, scale, ws->w, ws->col);
        else
            ssim_column_sums_simd_8(ws->rows8, scale, ws->w, ws->col);
        ssim_box_sums_simd(ws->col, scale, inside_lo, inside_hi, (float)ws->unit, out);
    }
    for (int x = 0; x < inside_lo; x++)
        out[x] = mirrored_box_sum(ws, x);
    for (int x = inside_hi; x < sw; x++)
        out[x] = mirrored_box_sum(ws, x);
}
#endif

static void decimate_picture(VmafPicture *pic, int plane, SsimWorkspace *ws, int simd, float *dst)
{
    const int h = ws->h, scale = ws->scale, sw = ws->sw, sh = ws->sh;
    const int wide = pic->bpc > 8;
    const ptrdiff_t stride = wide ? pic->stride[plane] / 2 : pic->stride[plane];
    const uint8_t *data8 = pic->data[plane];
    const uint16_t *data16 = pic->data[plane];
    const double *table = ws->table;
    const int uc = scale / 2;
    double *acc = ws->acc;
#ifndef SSIM_DECIMATE_SIMD_FLAG
    (void) simd;
#endif
    for (int y = 0; y < sh; y++) {
        float *out = dst + (size_t)y * sw;
#ifdef SSIM_DECIMATE_SIMD_FLAG
        if (simd && (table || ws->simd_sums)) {
            for (int v = 0; v < scale; v++) {
                const int row = ssim_mirror(y * scale - uc + v, h);
                if (wide)
                    ws->rows16[v] = data16 + row * stride;
                else
                    ws->rows8[v] = data8 + row * stride;
            }
            decimate_row_simd(ws, wide, out);
            continue;
        }
#endif
        for (int x = 0; x < sw; x++)
            acc[x] = 0.0;
        for (int v = 0; v < scale; v++) {
            const int row = ssim_mirror(y * scale - uc + v, h);
            const uint8_t *r8 = data8 + row * stride;
            const uint16_t *r16 = data16 + row * stride;
            for (int x = 0; x < sw; x++) {
                const int *c = ws->columns + x * scale;
                if (table) {
                    double sum = 0.0;
                    for (int u = 0; u < scale; u++)
                        sum += table[wide ? r16[c[u]] : r8[c[u]]];
                    acc[x] += sum;
                } else {
                    uint32_t sum = 0;
                    for (int u = 0; u < scale; u++)
                        sum += wide ? r16[c[u]] : r8[c[u]];
                    acc[x] += (double)sum;
                }
            }
        }
        for (int x = 0; x < sw; x++)
            out[x] = (float)(table ? acc[x] : acc[x] * ws->unit);
    }
}

static int ssim_of_floats(float *ref_f, float *cmp_f, int w, int h, float *iqa,
                          double *score, double *l_score, double *c_score,
                          double *s_score);

int compute_ssim_decimated(VmafPicture *ref, VmafPicture *cmp, int plane, int scale,
                           SsimWorkspace *ws, double *score, double *l_score,
                           double *c_score, double *s_score)
{
    if (plane < 0 || plane > 2 || !ref->data[plane] || !cmp->data[plane])
        return -EINVAL;
    const int w = ref->w[plane], h = ref->h[plane];
    const unsigned bpc = ref->bpc;
    if (scale <= 1 || cmp->w[plane] != (unsigned)w || cmp->h[plane] != (unsigned)h)
        return -EINVAL;
    if (bpc != 8 && bpc != 10 && bpc != 12 && bpc != 16)
        return -EINVAL;  /* picture_copy reads others as 8-bit */
    SsimWorkspace *own = NULL;
    if (!ws || ws->w != w || ws->h != h || ws->scale != scale || ws->bpc != bpc) {
        ws = own = ssim_workspace_alloc(w, h, scale, bpc, NULL);
        if (!ws) return -ENOMEM;
    }
    int simd = 0;
#ifdef SSIM_DECIMATE_SIMD_FLAG
    simd = !!(vmaf_get_cpu_flags() & SSIM_DECIMATE_SIMD_FLAG);
#endif
    decimate_picture(ref, plane, ws, simd, ws->ref_f);
    decimate_picture(cmp, plane, ws, simd, ws->cmp_f);
    const int err = ssim_of_floats(ws->ref_f, ws->cmp_f, ws->sw, ws->sh, ws->iqa,
                                   score, l_score, c_score, s_score);
    ssim_workspace_free(own);
    return err;
}

/* compute_ssim()'s SSIM of two (decimated) float images, as it is there,
 * with iqa (6 * w * h floats) for _iqa_ssim_ws's buffers. */
static int ssim_of_floats(float *ref_f, float *cmp_f, int w, int h, float *iqa,
                          double *score, double *l_score, double *c_score,
                          double *s_score)
{
    struct _kernel window;
    struct _map_reduce mr;
    float l, c, s;
    const struct iqa_ssim_args *args = 0;
    window.kernel = (float*)g_gaussian_window;
    window.kernel_h = (float*)g_gaussian_window_h;
    window.kernel_v = (float*)g_gaussian_window_v;
    window.w = window.h = GAUSSIAN_LEN;
    window.normalized = 1;
    window.bnd_opt = KBND_SYMMETRIC;
    const float result = _iqa_ssim_ws(ref_f, cmp_f, w, h, &window, &mr, args, &l, &c, &s, iqa);
    *score = (double)result;
    *l_score = (double)l;
    *c_score = (double)c;
    *s_score = (double)s;
    return 0;
}

int compute_ssim(const float *ref, const float *cmp, int w, int h,
        int ref_stride, int cmp_stride, double *score,
        double *l_score, double *c_score, double *s_score,
        int scale_override)
{

    int ret = 1;

    int scale;
    int x,y,src_offset,offset;
    float *ref_f,*cmp_f;
    struct _kernel low_pass;
    struct _kernel window;
    float result = INFINITY;
    float l, c, s;
    double ssim_sum=0.0;
    struct _map_reduce mr;

    /* check stride */
    int stride = ref_stride; /* stride in bytes */
    if (stride != cmp_stride)
    {
        printf("error: for ssim, ref_stride (%d) != dis_stride (%d) bytes.\n", ref_stride, cmp_stride);
        fflush(stdout);
        goto fail_or_end;
    }
    stride /= sizeof(float); /* stride_ in pixels */

    /* specify some default parameters */
    const struct iqa_ssim_args *args = 0; /* 0 for default */
    int gaussian = 1; /* 0 for 8x8 square window, 1 for 11x11 circular-symmetric Gaussian window (default) */

    /* initialize algorithm parameters */
    if (scale_override > 0) {
        scale = scale_override;
    } else {
        scale = _max( 1, _round( (float)_min(w,h) / 256.0f ) );
    }
    if (args) {
        if(args->f) {
            scale = args->f;
        }
        mr.map     = _ssim_map;
        mr.reduce  = _ssim_reduce;
        mr.context = (void*)&ssim_sum;
    }
    window.kernel = (float*)g_square_window;
    window.kernel_h = (float*)g_square_window_h;
    window.kernel_v = (float*)g_square_window_v;
    window.w = window.h = SQUARE_LEN;
    window.normalized = 1;
    window.bnd_opt = KBND_SYMMETRIC;
    if (gaussian) {
        window.kernel = (float*)g_gaussian_window;
        window.kernel_h = (float*)g_gaussian_window_h;
        window.kernel_v = (float*)g_gaussian_window_v;
        window.w = window.h = GAUSSIAN_LEN;
    }

    /* convert image values to floats, forcing stride = width. */
    ref_f = (float*)malloc(w*h*sizeof(float));
    cmp_f = (float*)malloc(w*h*sizeof(float));
    if (!ref_f || !cmp_f) {
        if (ref_f) free(ref_f);
        if (cmp_f) free(cmp_f);
        printf("error: unable to malloc ref_f or cmp_f.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    for (y=0; y<h; ++y) {
        src_offset = y * stride;
        offset = y * w;
        for (x=0; x<w; ++x, ++offset, ++src_offset) {
            ref_f[offset] = (float)ref[src_offset];
            cmp_f[offset] = (float)cmp[src_offset];
        }
    }

    /* scale the images down if required */
    if (scale > 1) {
        /* generate simple low-pass filter */
        low_pass.kernel = (float*)malloc(scale*scale*sizeof(float));
        low_pass.kernel_h = (float*)malloc(scale*sizeof(float)); /* zli-nflx */
        low_pass.kernel_v = (float*)malloc(scale*sizeof(float)); /* zli-nflx */
        if (!(low_pass.kernel && low_pass.kernel_h && low_pass.kernel_v)) { /* zli-nflx */
            free(ref_f);
            free(cmp_f);
            if (low_pass.kernel) free(low_pass.kernel); /* zli-nflx */
            if (low_pass.kernel_h) free(low_pass.kernel_h); /* zli-nflx */
            if (low_pass.kernel_v) free(low_pass.kernel_v); /* zli-nflx */
            printf("error: unable to malloc low-pass filter kernel.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        low_pass.w = low_pass.h = scale;
        low_pass.normalized = 0;
        low_pass.bnd_opt = KBND_SYMMETRIC;
        for (offset=0; offset<scale*scale; ++offset)
            low_pass.kernel[offset] = 1.0f/(scale*scale);
        for (offset=0; offset<scale; ++offset)  /* zli-nflx */
            low_pass.kernel_h[offset] = 1.0f/(scale); /* zli-nflx */
        for (offset=0; offset<scale; ++offset) /* zli-nflx */
            low_pass.kernel_v[offset] = 1.0f/(scale); /* zli-nflx */

        /* resample */
        if (_iqa_decimate(ref_f, w, h, scale, &low_pass, 0, 0, 0) ||
            _iqa_decimate(cmp_f, w, h, scale, &low_pass, 0, &w, &h)) { /* update w/h */
            free(ref_f);
            free(cmp_f);
            free(low_pass.kernel);
            free(low_pass.kernel_h); /* zli-nflx */
            free(low_pass.kernel_v); /* zli-nflx */
            printf("error: decimation fails on ref_f or cmp_f.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        free(low_pass.kernel);
        free(low_pass.kernel_h); /* zli-nflx */
        free(low_pass.kernel_v); /* zli-nflx */
    }

    result = _iqa_ssim(ref_f, cmp_f, w, h, &window, &mr, args, &l, &c, &s);

    free(ref_f);
    free(cmp_f);

    *score = (double)result;
    *l_score = (double)l;
    *c_score = (double)c;
    *s_score = (double)s;

    ret = 0;
fail_or_end:
    return ret;

}

