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
#include "parallel.h"
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
 * output rows, whose buffers stay in the CPU's caches: at 4K, 4.2 MB per
 * thread (stripe_scratch_floats) instead of six full-size pictures per
 * scale. The same sums in the
 * same order as one pass over the picture, so the same score. 24 rows was
 * the fastest at 4K with 24 threads (64 rows: 20.6 ms a frame, 32: 13.6, 24:
 * 12.3, 16: 12.7; Core Ultra 9 285K); a stripe filters the 10 rows it shares
 * with the next one again, which costs 4 % at 4K with one thread. */
#define STRIPE_ROWS 24

/* The input rows of one stripe. */
#define STRIPE_IN_ROWS (STRIPE_ROWS + GAUSSIAN_LEN - 1)

/* libvmaf-fast: the step down to the next scale is made in bands of this
 * many output rows, on as many threads as there are (parallel.h): each
 * output is written once, so the order does not matter. A band reads again
 * the 7 input rows it shares with the one before (39 rows for 32 new ones);
 * smaller bands would spread the last step (135 output rows at 4K, 9 bands
 * per picture) over more threads but read more rows twice. */
#define DECIMATE_BAND_ROWS 16

/* A stripe's scratch, for a w-wide scale: _iqa_ssim_maps' six buffers of
 * STRIPE_IN_ROWS rows, then the two input stripes (where the scale is not w
 * floats per row). The maps' terms (_iqa_ssim_terms) are written from the
 * sixth buffer on, which _iqa_ssim_maps uses only while it runs, over the
 * input stripes it has read by then. */
static size_t stripe_map_floats(int w)
{
    return (size_t)STRIPE_IN_ROWS * w;
}

static size_t stripe_scratch_floats(int w)
{
    const size_t inputs = 2 * stripe_map_floats(w);
    const size_t terms = _iqa_ssim_terms_bytes(STRIPE_ROWS * w) / sizeof(float);
    const size_t after_maps = stripe_map_floats(w) + inputs;
    return 5 * stripe_map_floats(w) + (terms > after_maps ? terms : after_maps);
}

/* libvmaf-fast: compute_ms_ssim's buffers, kept from frame to frame:
 * scaled[0] holds the reference's scales 1 to 4, scaled[1] the distorted's;
 * scale 0 is read where it is (the caller's floats or picture). scratch is
 * the calling thread's stripe scratch (stripe_scratch_floats); helper
 * threads (parallel.h) use their own. */
struct MsSsimWorkspace {
    int w, h;  /* the largest picture */
    float *scaled[2][SCALES];
    float *scratch;
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
    }
    aligned_free(ws->scratch);
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
    ws->scratch = aligned_malloc(stripe_scratch_floats(w) * sizeof(float), 64);
    if (!ws->scratch)
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

/* One scale's stripes as a parallel job (parallel.h): the scale, and the
 * running sums the stripes add to in turn. */
struct Stripes {
    const struct Scale *pics;
    int w, out_h;
    const struct _kernel *window;
    double ssim_sum, l_sum, c_sum, s_sum;
};

/* A stripe's addition, run in its turn: its terms into the scale's sums. */
struct StripeSums {
    struct Stripes *st;
    const void *terms;
    int pixels;
};

static void add_stripe(void *arg)
{
    const struct StripeSums *a = arg;
    struct Stripes *st = a->st;
    _iqa_ssim_add_pixel_terms(a->terms, a->pixels, &st->ssim_sum, &st->l_sum, &st->c_sum,
                              &st->s_sum);
}

/* One stripe: its maps and their terms made in the thread's scratch while
 * other stripes' are, then the terms added after every stripe above it has
 * added its own, so the sums are added in the order of one pass down the
 * picture. */
static int ssim_stripe(void *ctx, unsigned chunk, void *scratch, VmafParallelJob *job)
{
    struct Stripes *st = ctx;
    const int y = (int)chunk * STRIPE_ROWS;
    const int rows = st->out_h - y < STRIPE_ROWS ? st->out_h - y : STRIPE_ROWS;
    const int in_rows = rows + st->window->h - 1;
    const size_t map_floats = stripe_map_floats(st->w);
    float *maps = scratch;
    float *terms = maps + 5 * map_floats;
    float *in = maps + 6 * map_floats;
    const int pixels = _iqa_ssim_maps(stripe_input(&st->pics[0], y, in_rows, st->w, in),
                                      stripe_input(&st->pics[1], y, in_rows, st->w,
                                                   in + map_floats),
                                      st->w, in_rows, st->window, maps);
    _iqa_ssim_terms(maps, st->w, in_rows, pixels, terms);
    struct StripeSums sums = { st, terms, pixels };
    vmaf_parallel_in_turn(job, chunk, add_stripe, &sums);
    return 0;
}

/* One scale's l, c and s: _iqa_ssim's means over the w x h pictures,
 * calculated in stripes of STRIPE_ROWS output rows, shared with the helper
 * threads. */
static int ssim_in_stripes(const struct Scale pics[2], int w, int h, const struct _kernel *window,
                           MsSsimWorkspace *ws, float *l, float *c, float *s)
{
    const int out_w = w - window->w + 1;
    const int out_h = h - window->h + 1;
    struct Stripes st = { .pics = pics, .w = w, .out_h = out_h, .window = window };
    const unsigned n_stripes = (unsigned)((out_h + STRIPE_ROWS - 1) / STRIPE_ROWS);
    const int err = vmaf_parallel_for(n_stripes, ssim_stripe, &st,
                                      stripe_scratch_floats(w) * sizeof(float), ws->scratch,
                                      true);
    if (err)
        return err;
    *l = (float)(st.l_sum / (double)(out_w * out_h));
    *c = (float)(st.c_sum / (double)(out_w * out_h));
    *s = (float)(st.s_sum / (double)(out_w * out_h));
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

/* One step down to the next scale as a parallel job (parallel.h): both
 * pictures of a w x h scale, `bands` bands of DECIMATE_BAND_ROWS output rows
 * each. */
struct Decimation {
    const struct Scale *from;
    float *to[2];
    int w, h, bands;
    const struct _kernel *lpf;
};

/* One band of one picture, its rows read through the scale's row function
 * (thread-safe: it writes only into the scratch it is given) or in place. */
static int decimate_band(void *ctx, unsigned chunk, void *scratch, VmafParallelJob *job)
{
    (void)job;
    const struct Decimation *d = ctx;
    const struct Scale *pic = &d->from[chunk / d->bands];
    const int y0 = (int)(chunk % d->bands) * DECIMATE_BAND_ROWS;
    const int out_h = next_scale(d->h);
    const int y1 = out_h - y0 < DECIMATE_BAND_ROWS ? out_h : y0 + DECIMATE_BAND_ROWS;
    struct FloatRows whole = { pic->whole, d->w };
    const iqa_row_fn row_fn = pic->row_fn ? pic->row_fn : float_row;
    void *rows = pic->row_fn ? pic->ctx : &whole;
    return _iqa_decimate2_band(row_fn, rows, d->w, d->h, d->lpf, d->to[chunk / d->bands], y0, y1,
                               scratch);
}

/* The next scale of both pictures into `to`. */
static int decimate(const struct Scale pics[2], int w, int h, const struct _kernel *lpf,
                    float *const to[2], MsSsimWorkspace *ws)
{
    struct Decimation d = { .from = pics, .to = { to[0], to[1] }, .w = w, .h = h, .lpf = lpf };
    d.bands = (next_scale(h) + DECIMATE_BAND_ROWS - 1) / DECIMATE_BAND_ROWS;
    /* the band buffer (about 10 rows) fits in a stripe's scratch (34 * 8) */
    return vmaf_parallel_for(2 * (unsigned)d.bands, decimate_band, &d,
                             _iqa_decimate2_buffer_floats(w, lpf) * sizeof(float), ws->scratch,
                             false);
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
        /* fails only where a helper thread cannot get its stripe scratch */
        if (ssim_in_stripes(cur, cur_w, cur_h, &window, ws, &l, &c, &s)) {
            printf("error: unable to allocate the ms_ssim stripe buffers.\n");
            fflush(stdout);
            return 1;
        }

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
        float *const next[2] = { ws->scaled[0][idx + 1], ws->scaled[1][idx + 1] };
        if (decimate(cur, cur_w, cur_h, &lpf, next, ws)) {
            printf("error: decimation fails on ref_imgs or cmp_imgs.\n");
            fflush(stdout);
            return 1;
        }
        for (int i = 0; i < 2; ++i)
            cur[i] = (struct Scale){ .whole = next[i] };
        cur_w = next_scale(cur_w);
        cur_h = next_scale(cur_h);
    }

    *score = msssim;
    return 0;
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
