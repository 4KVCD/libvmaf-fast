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

#include "mem.h"
#include "iqa/math_utils.h"
#include "iqa/decimate.h"
#include "iqa/ssim_tools.h"
#include "ms_ssim.h"
#include "picture_copy.h"

/* Low-pass filter for down-sampling (9/7 biorthogonal wavelet filter) */
#define LPF_LEN 9
static const float g_lpf[LPF_LEN][LPF_LEN] = {
   { 0.000714f,-0.000450f,-0.002090f, 0.007132f, 0.016114f, 0.007132f,-0.002090f,-0.000450f, 0.000714f},
   {-0.000450f, 0.000283f, 0.001316f,-0.004490f,-0.010146f,-0.004490f, 0.001316f, 0.000283f,-0.000450f},
   {-0.002090f, 0.001316f, 0.006115f,-0.020867f,-0.047149f,-0.020867f, 0.006115f, 0.001316f,-0.002090f},
   { 0.007132f,-0.004490f,-0.020867f, 0.071207f, 0.160885f, 0.071207f,-0.020867f,-0.004490f, 0.007132f},
   { 0.016114f,-0.010146f,-0.047149f, 0.160885f, 0.363505f, 0.160885f,-0.047149f,-0.010146f, 0.016114f},
   { 0.007132f,-0.004490f,-0.020867f, 0.071207f, 0.160885f, 0.071207f,-0.020867f,-0.004490f, 0.007132f},
   {-0.002090f, 0.001316f, 0.006115f,-0.020867f,-0.047149f,-0.020867f, 0.006115f, 0.001316f,-0.002090f},
   {-0.000450f, 0.000283f, 0.001316f,-0.004490f,-0.010146f,-0.004490f, 0.001316f, 0.000283f,-0.000450f},
   { 0.000714f,-0.000450f,-0.002090f, 0.007132f, 0.016114f, 0.007132f,-0.002090f,-0.000450f, 0.000714f},
};

static const float g_lpf_h[LPF_LEN] = {
    0.026727f, -0.016828f, -0.078201f, 0.266846f, 0.602914f, 0.266846f, -0.078201f, -0.016828f, 0.026727f
};

static const float g_lpf_v[LPF_LEN] = {
    0.026727f, -0.016828f, -0.078201f, 0.266846f, 0.602914f, 0.266846f, -0.078201f, -0.016828f, 0.026727f
};

/* Alpha, beta, and gamma values for each scale */
static float g_alphas[] = { 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f };
static float g_betas[]  = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };
static float g_gammas[] = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };

/* libvmaf-fast: each scale's SSIM is calculated in stripes of this many
 * output rows, whose buffers stay in the CPU's caches (at 4K, 6.8 MB instead
 * of six full-size pictures per scale). The same sums in the same order as
 * one pass over the picture, so the same score. */
#define STRIPE_ROWS 64

/* The input rows of one stripe. */
#define STRIPE_IN_ROWS (STRIPE_ROWS + GAUSSIAN_LEN - 1)

/* libvmaf-fast: compute_ms_ssim's buffers, kept from frame to frame:
 * scaled[0] holds the reference's scales 1 to 4, scaled[1] the distorted's;
 * scale 0 is read where it is (the caller's floats or picture), its stripes
 * copied or converted into in[0] and in[1] where it is not w floats per
 * row; stripe holds one stripe's _iqa_ssim_add_sums buffers. */
struct MsSsimWorkspace {
    int w, h;  /* the largest picture */
    float *scaled[2][SCALES];
    float *in[2];
    float *stripe;
};

/* A side of the next scale, from this scale's. */
static int next_scale(int size)
{
    return size / 2 + (size & 1);
}

void ms_ssim_workspace_free(MsSsimWorkspace *ws)
{
    if (!ws)
        return;
    for (int i = 0; i < 2; ++i) {
        for (int idx = 1; idx < SCALES; ++idx)
            free(ws->scaled[i][idx]);
        free(ws->in[i]);
    }
    free(ws->stripe);
    free(ws);
}

MsSsimWorkspace *ms_ssim_workspace_alloc(int w, int h)
{
    MsSsimWorkspace *ws = calloc(1, sizeof(*ws));
    if (!ws)
        return NULL;
    ws->w = w;
    ws->h = h;
    int cur_w = w, cur_h = h;
    for (int idx = 1; idx < SCALES; ++idx) {
        cur_w = next_scale(cur_w);
        cur_h = next_scale(cur_h);
        for (int i = 0; i < 2; ++i) {
            ws->scaled[i][idx] = malloc((size_t)cur_w * cur_h * sizeof(float));
            if (!ws->scaled[i][idx])
                goto fail;
        }
    }
    for (int i = 0; i < 2; ++i) {
        ws->in[i] = malloc((size_t)STRIPE_IN_ROWS * w * sizeof(float));
        if (!ws->in[i])
            goto fail;
    }
    ws->stripe = malloc(6 * (size_t)STRIPE_IN_ROWS * w * sizeof(float));
    if (!ws->stripe)
        goto fail;
    return ws;
fail:
    ms_ssim_workspace_free(ws);
    return NULL;
}

int ms_ssim_size_fits(int w, int h)
{
    /* make sure we won't scale below 1x1: the window fits in every scale */
    for (int idx = 0; idx < SCALES; ++idx) {
        if (w < GAUSSIAN_LEN || h < GAUSSIAN_LEN)
            return 0;
        w /= 2;
        h /= 2;
    }
    return 1;
}

/* A scale's picture: w floats per row at `whole`, or, where whole is NULL,
 * rows read one at a time through row_fn. */
struct Scale {
    const float *whole;
    iqa_row_fn row_fn;
    void *ctx;
};

/* The input rows [y, y + rows) of a stripe, w floats each: in the scale's
 * picture, or copied into buf. */
static float *stripe_input(const struct Scale *pic, int y, int rows, int w, float *buf)
{
    if (pic->whole)
        return (float *)pic->whole + (size_t)y * w;
    for (int r = 0; r < rows; ++r) {
        float *dst = buf + (size_t)r * w;
        const float *row = pic->row_fn(pic->ctx, y + r, dst);
        if (row != dst)
            memcpy(dst, row, w * sizeof(float));
    }
    return buf;
}

/* One scale's l, c and s: _iqa_ssim's means over the w x h pictures,
 * calculated in stripes of STRIPE_ROWS output rows. */
static void ssim_in_stripes(const struct Scale pics[2], int w, int h, const struct _kernel *window,
                            MsSsimWorkspace *ws, float *l, float *c, float *s)
{
    const int out_w = w - window->w + 1;
    const int out_h = h - window->h + 1;
    double ssim_sum = 0.0, l_sum = 0.0, c_sum = 0.0, s_sum = 0.0;
    for (int y = 0; y < out_h; y += STRIPE_ROWS) {
        const int rows = out_h - y < STRIPE_ROWS ? out_h - y : STRIPE_ROWS;
        const int in_rows = rows + window->h - 1;
        _iqa_ssim_add_sums(stripe_input(&pics[0], y, in_rows, w, ws->in[0]),
                           stripe_input(&pics[1], y, in_rows, w, ws->in[1]), w, in_rows,
                           window, ws->stripe, &ssim_sum, &l_sum, &c_sum, &s_sum);
    }
    *l = (float)(l_sum / (double)(out_w * out_h));
    *c = (float)(c_sum / (double)(out_w * out_h));
    *s = (float)(s_sum / (double)(out_w * out_h));
}

/* MS-SSIM (Wang) with libvmaf's defaults, a Gaussian window and 5 scales,
 * of the w x h pictures whose scale 0 is pics[0] (the reference) and
 * pics[1] (the distorted). */
static int ms_ssim_of(const struct Scale pics[2], int w, int h, MsSsimWorkspace *ws,
                      double *score, double *l_scores, double *c_scores, double *s_scores)
{
    if (!ms_ssim_size_fits(w, h)) {
        printf("error: scale below 1x1!\n");
        return 1;
    }
    if (w > ws->w || h > ws->h)
        return 1;

    struct _kernel lpf;
    lpf.kernel = (float*)g_lpf;
    lpf.kernel_h = (float*)g_lpf_h; /* zli-nflx */
    lpf.kernel_v = (float*)g_lpf_v; /* zli-nflx */
    lpf.w = lpf.h = LPF_LEN;
    lpf.normalized = 1;
    lpf.bnd_opt = KBND_SYMMETRIC;

    struct _kernel window;
    window.kernel = (float*)g_gaussian_window;
    window.kernel_h = (float*)g_gaussian_window_h; /* zli-nflx */
    window.kernel_v = (float*)g_gaussian_window_v; /* zli-nflx */
    window.w = window.h = GAUSSIAN_LEN;
    window.normalized = 1;
    window.bnd_opt = KBND_SYMMETRIC;

    struct Scale cur[2] = { pics[0], pics[1] };
    int cur_w = w, cur_h = h;
    double msssim = 1.0;
    for (int idx = 0; idx < SCALES; ++idx) {
        float l, c, s;
        ssim_in_stripes(cur, cur_w, cur_h, &window, ws, &l, &c, &s);

        msssim *= pow(l, g_alphas[idx]) * pow(c, g_betas[idx]) * pow(s, g_gammas[idx]);
        l_scores[idx] = l;
        c_scores[idx] = c;
        s_scores[idx] = s;

        if (msssim == INFINITY) {
            printf("error: ms_ssim is INFINITY.\n");
            fflush(stdout);
            return 1;
        }
        if (idx + 1 == SCALES)
            break;

        /* the next scale: scale 0 decimated as it is read, the others from
         * the workspace */
        for (int i = 0; i < 2; ++i) {
            float *next = ws->scaled[i][idx + 1];
            const int err = idx == 0
                ? _iqa_decimate2_rows(cur[i].row_fn, cur[i].ctx, cur_w, cur_h, &lpf, next, 0, 0)
                : _iqa_decimate((float *)cur[i].whole, cur_w, cur_h, 2, &lpf, next, 0, 0);
            if (err) {
                printf("error: decimation fails on ref_imgs or cmp_imgs.\n");
                fflush(stdout);
                return 1;
            }
            cur[i] = (struct Scale){ .whole = next };
        }
        cur_w = next_scale(cur_w);
        cur_h = next_scale(cur_h);
    }

    *score = msssim;
    return 0;
}

/* A float picture's rows, read in place. */
struct FloatRows {
    const float *pic;
    int stride;  /* in floats */
};

static const float *float_row(void *ctx, int r, float *scratch)
{
    const struct FloatRows *f = ctx;
    (void)scratch;
    return f->pic + (size_t)r * f->stride;
}

int compute_ms_ssim_ws(const float *ref, const float *cmp, int w, int h,
        int ref_stride, int cmp_stride, MsSsimWorkspace *ws, double *score,
        double* l_scores, double* c_scores, double* s_scores)
{
    if (ref_stride != cmp_stride) {
        printf("error: for ms_ssim, ref_stride (%d) != dis_stride (%d) bytes.\n", ref_stride, cmp_stride);
        fflush(stdout);
        return 1;
    }
    const int stride = ref_stride / sizeof(float); /* stride in pixels */
    struct FloatRows rows[2] = { { ref, stride }, { cmp, stride } };
    struct Scale pics[2];
    for (int i = 0; i < 2; ++i)
        pics[i] = (struct Scale){ stride == w ? rows[i].pic : NULL, float_row, &rows[i] };
    return ms_ssim_of(pics, w, h, ws, score, l_scores, c_scores, s_scores);
}

int compute_ms_ssim(const float *ref, const float *cmp, int w, int h,
        int ref_stride, int cmp_stride, double *score,
        double* l_scores, double* c_scores, double* s_scores)
{
    MsSsimWorkspace *ws = ms_ssim_workspace_alloc(w, h);
    if (!ws) {
        printf("error: unable to allocate the ms_ssim buffers.\n");
        fflush(stdout);
        return 1;
    }
    const int err = compute_ms_ssim_ws(ref, cmp, w, h, ref_stride, cmp_stride, ws, score,
                                       l_scores, c_scores, s_scores);
    ms_ssim_workspace_free(ws);
    return err;
}

/* A picture plane's rows as picture_copy converts them (offset 0). */
struct PictureRows {
    const VmafPicture *pic;
    int plane;
};

static const float *picture_row(void *ctx, int r, float *scratch)
{
    const struct PictureRows *p = ctx;
    picture_copy_row(scratch, p->pic, 0, p->pic->bpc, p->plane, r);
    return scratch;
}

int compute_ms_ssim_picture(const VmafPicture *ref, const VmafPicture *dis, int plane,
                            MsSsimWorkspace *ws, double *score, double *l_scores,
                            double *c_scores, double *s_scores)
{
    struct PictureRows rows[2] = { { ref, plane }, { dis, plane } };
    const struct Scale pics[2] = { { NULL, picture_row, &rows[0] }, { NULL, picture_row, &rows[1] } };
    return ms_ssim_of(pics, ref->w[plane], ref->h[plane], ws, score, l_scores, c_scores, s_scores);
}
