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

#include <errno.h>

#include "mem.h"
#include "iqa/math_utils.h"
#include "iqa/decimate.h"
#include "iqa/ssim_tools.h"
#include "ssim.h"

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
 */
static int ssim_mirror(int i, int n)
{
    if (i < 0) return -1 - i;                /* KBND_SYMMETRIC */
    if (i >= n) return (n - (i - n)) - 1;
    return i;
}

static int decimate_picture(VmafPicture *pic, int scale, float *dst,
                            int sw, int sh, const double *table)
{
    const int w = pic->w[0], h = pic->h[0];
    const int wide = pic->bpc > 8;
    const ptrdiff_t stride = wide ? pic->stride[0] / 2 : pic->stride[0];
    const uint8_t *data8 = pic->data[0];
    const uint16_t *data16 = pic->data[0];
    /* _iqa_filter_pixel's window: the scale samples from x * scale - uc */
    const int uc = scale / 2;
    double *acc = malloc(sizeof(double) * sw);
    int *columns = malloc(sizeof(int) * sw * scale);
    if (!acc || !columns) {
        free(acc);
        free(columns);
        return -ENOMEM;
    }
    for (int x = 0; x < sw; x++)
        for (int u = 0; u < scale; u++)
            columns[x * scale + u] = ssim_mirror(x * scale - uc + u, w);
    /* table == NULL: 1/(scale*scale) and the conversion are powers of two */
    const double unit = 1.0 / (double)(scale * scale) /
                        (pic->bpc == 10 ? 4.0 : pic->bpc == 12 ? 16.0 :
                         pic->bpc == 16 ? 256.0 : 1.0);
    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < sw; x++)
            acc[x] = 0.0;
        for (int v = 0; v < scale; v++) {
            const int row = ssim_mirror(y * scale - uc + v, h);
            const uint8_t *r8 = data8 + row * stride;
            const uint16_t *r16 = data16 + row * stride;
            for (int x = 0; x < sw; x++) {
                const int *c = columns + x * scale;
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
        float *out = dst + (size_t)y * sw;
        for (int x = 0; x < sw; x++)
            out[x] = (float)(table ? acc[x] : acc[x] * unit);
    }
    free(acc);
    free(columns);
    return 0;
}

static int ssim_of_floats(float *ref_f, float *cmp_f, int w, int h,
                          double *score, double *l_score, double *c_score,
                          double *s_score);

int compute_ssim_decimated(VmafPicture *ref, VmafPicture *cmp, int scale,
                           double *score, double *l_score, double *c_score,
                           double *s_score)
{
    const int w = ref->w[0], h = ref->h[0];
    const unsigned bpc = ref->bpc;
    if (scale <= 1 || cmp->w[0] != (unsigned)w || cmp->h[0] != (unsigned)h)
        return -EINVAL;
    if (bpc != 8 && bpc != 10 && bpc != 12 && bpc != 16)
        return -EINVAL;  /* picture_copy reads others as 8-bit */
    const int sw = w / scale + (w & 1), sh = h / scale + (h & 1);
    /* low_pass.kernel[] as compute_ssim() makes it */
    const float k = 1.0f / (scale * scale);
    const int power_of_two = (scale & (scale - 1)) == 0;
    double *table = NULL;
    if (!power_of_two) {
        const size_t values = (size_t)1 << bpc;
        const float scaler = bpc == 10 ? 4.0f : bpc == 12 ? 16.0f : bpc == 16 ? 256.0f : 1.0f;
        table = malloc(sizeof(double) * values);
        if (!table) return -ENOMEM;
        for (size_t i = 0; i < values; i++) {
            /* picture_copy's value, times the kernel's float, in float */
            const float value = bpc == 8 ? (float)i + 0 : (float)i / scaler + 0;
            const float product = value * k;
            table[i] = (double)product;
        }
    }
    float *ref_f = malloc(sizeof(float) * sw * sh);
    float *cmp_f = malloc(sizeof(float) * sw * sh);
    int err = (!ref_f || !cmp_f) ? -ENOMEM : 0;
    if (!err) err = decimate_picture(ref, scale, ref_f, sw, sh, table);
    if (!err) err = decimate_picture(cmp, scale, cmp_f, sw, sh, table);
    if (!err)
        err = ssim_of_floats(ref_f, cmp_f, sw, sh, score, l_score, c_score, s_score);
    free(ref_f);
    free(cmp_f);
    free(table);
    return err;
}

/* compute_ssim()'s SSIM of two (decimated) float images, as it is there. */
static int ssim_of_floats(float *ref_f, float *cmp_f, int w, int h,
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
    const float result = _iqa_ssim(ref_f, cmp_f, w, h, &window, &mr, args, &l, &c, &s);
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

