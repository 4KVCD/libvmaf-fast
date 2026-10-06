/*
 * Copyright (c) 2024 Christian R. Helmrich
 * Copyright (c) 2024 Christian Lehmann
 * Copyright (c) 2024 Christian Stoffers
 *
 * libvmaf-fast's xpsnr extractor (feature/xpsnr.c) against FFmpeg 9.0.1's
 * filter (libavfilter/vf_xpsnr.c), copied here as it is for the luma plane:
 * frames in order, the originals before kept in its two buffers, changed in
 * place as it goes. The extractor scores each frame from its pictures and
 * the two before, on any thread and subsampled: every score must be the
 * filter's to the last bit -- block weights with and without downsampling,
 * the min-smoothing of small pictures, blocks too narrow for the activity,
 * pictures too small for blocks, odd sizes up to 2048x1152, first- and
 * second-order temporal activity, 8, 10 and 12 bits, identical pictures.
 *
 * This file is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This file is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this file (libvmaf/COPYING.LGPLv2.1); if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301 USA
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/feature.h"
#include "libvmaf/picture.h"

#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#else
#pragma STDC FP_CONTRACT OFF
#endif

/* ---- FFmpeg's filter, for the luma plane ---- */

#define XPSNR_GAMMA 2
#define FFMAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct Ffmpeg {
    int depth, frame_rate;
    int plane_width, plane_height;
    int16_t *buf_org_m1, *buf_org_m2;
    double *sse_luma, *weights;
} Ffmpeg;

static uint64_t sse_line_16bit(const uint8_t *_main_line, const uint8_t *_ref_line, int outw)
{
    int j;
    unsigned m2 = 0;
    const uint16_t *main_line = (const uint16_t *) _main_line;
    const uint16_t *ref_line = (const uint16_t *) _ref_line;

    for (j = 0; j < outw; j++) {
        unsigned error = main_line[j] - ref_line[j];

        m2 += error * error;
    }

    return m2;
}

static uint64_t highds(const int x_act, const int y_act, const int w_act, const int h_act, const int16_t *o_m0, const int o)
{
    uint64_t sa_act = 0;

    for (int y = y_act; y < h_act; y += 2) {
        for (int x = x_act; x < w_act; x += 2) {
            const int f = 12 * ((int)o_m0[ y   *o + x  ] + (int)o_m0[ y   *o + x+1] + (int)o_m0[(y+1)*o + x  ] + (int)o_m0[(y+1)*o + x+1])
                         - 3 * ((int)o_m0[(y-1)*o + x  ] + (int)o_m0[(y-1)*o + x+1] + (int)o_m0[(y+2)*o + x  ] + (int)o_m0[(y+2)*o + x+1])
                         - 3 * ((int)o_m0[ y   *o + x-1] + (int)o_m0[ y   *o + x+2] + (int)o_m0[(y+1)*o + x-1] + (int)o_m0[(y+1)*o + x+2])
                         - 2 * ((int)o_m0[(y-1)*o + x-1] + (int)o_m0[(y-1)*o + x+2] + (int)o_m0[(y+2)*o + x-1] + (int)o_m0[(y+2)*o + x+2])
                             - ((int)o_m0[(y-2)*o + x-1] + (int)o_m0[(y-2)*o + x  ] + (int)o_m0[(y-2)*o + x+1] + (int)o_m0[(y-2)*o + x+2]
                              + (int)o_m0[(y+3)*o + x-1] + (int)o_m0[(y+3)*o + x  ] + (int)o_m0[(y+3)*o + x+1] + (int)o_m0[(y+3)*o + x+2]
                              + (int)o_m0[(y-1)*o + x-2] + (int)o_m0[ y   *o + x-2] + (int)o_m0[(y+1)*o + x-2] + (int)o_m0[(y+2)*o + x-2]
                              + (int)o_m0[(y-1)*o + x+3] + (int)o_m0[ y   *o + x+3] + (int)o_m0[(y+1)*o + x+3] + (int)o_m0[(y+2)*o + x+3]);
            sa_act += (uint64_t) abs(f);
        }
    }
    return sa_act;
}

static uint64_t diff1st(const uint32_t w_act, const uint32_t h_act, const int16_t *o_m0, int16_t *o_m1, const int o)
{
    uint64_t ta_act = 0;

    for (uint32_t y = 0; y < h_act; y += 2) {
        for (uint32_t x = 0; x < w_act; x += 2) {
            const int t = (int)o_m0[y*o + x] + (int)o_m0[y*o + x+1] + (int)o_m0[(y+1)*o + x] + (int)o_m0[(y+1)*o + x+1]
                       - ((int)o_m1[y*o + x] + (int)o_m1[y*o + x+1] + (int)o_m1[(y+1)*o + x] + (int)o_m1[(y+1)*o + x+1]);
            ta_act += (uint64_t) abs(t);
            o_m1[y*o + x  ] = o_m0[y*o + x  ];  o_m1[(y+1)*o + x  ] = o_m0[(y+1)*o + x  ];
            o_m1[y*o + x+1] = o_m0[y*o + x+1];  o_m1[(y+1)*o + x+1] = o_m0[(y+1)*o + x+1];
        }
    }
    return (ta_act * XPSNR_GAMMA);
}

static uint64_t diff2nd(const uint32_t w_act, const uint32_t h_act, const int16_t *o_m0, int16_t *o_m1, int16_t *o_m2, const int o)
{
    uint64_t ta_act = 0;

    for (uint32_t y = 0; y < h_act; y += 2) {
        for (uint32_t x = 0; x < w_act; x += 2) {
            const int t = (int)o_m0[y*o + x] + (int)o_m0[y*o + x+1] + (int)o_m0[(y+1)*o + x] + (int)o_m0[(y+1)*o + x+1]
                   - 2 * ((int)o_m1[y*o + x] + (int)o_m1[y*o + x+1] + (int)o_m1[(y+1)*o + x] + (int)o_m1[(y+1)*o + x+1])
                        + (int)o_m2[y*o + x] + (int)o_m2[y*o + x+1] + (int)o_m2[(y+1)*o + x] + (int)o_m2[(y+1)*o + x+1];
            ta_act += (uint64_t) abs(t);
            o_m2[y*o + x  ] = o_m1[y*o + x  ];  o_m2[(y+1)*o + x  ] = o_m1[(y+1)*o + x  ];
            o_m2[y*o + x+1] = o_m1[y*o + x+1];  o_m2[(y+1)*o + x+1] = o_m1[(y+1)*o + x+1];
            o_m1[y*o + x  ] = o_m0[y*o + x  ];  o_m1[(y+1)*o + x  ] = o_m0[(y+1)*o + x  ];
            o_m1[y*o + x+1] = o_m0[y*o + x+1];  o_m1[(y+1)*o + x+1] = o_m0[(y+1)*o + x+1];
        }
    }
    return (ta_act * XPSNR_GAMMA);
}

static inline uint64_t calc_squared_error(const int16_t *blk_org,     const uint32_t stride_org,
                                          const int16_t *blk_rec,     const uint32_t stride_rec,
                                          const uint32_t block_width, const uint32_t block_height)
{
    uint64_t sse = 0;  /* sum of squared errors */

    for (uint32_t y = 0; y < block_height; y++) {
        sse += sse_line_16bit((const uint8_t *) blk_org, (const uint8_t *) blk_rec, (int) block_width);
        blk_org += stride_org;
        blk_rec += stride_rec;
    }

    /* return nonweighted sum of squared errors */
    return sse;
}

static inline double calc_squared_error_and_weight (Ffmpeg const *s,
                                                    const int16_t *pic_org,     const uint32_t stride_org,
                                                    int16_t       *pic_org_m1,  int16_t       *pic_org_m2,
                                                    const int16_t *pic_rec,     const uint32_t stride_rec,
                                                    const uint32_t offset_x,    const uint32_t offset_y,
                                                    const uint32_t block_width, const uint32_t block_height,
                                                    const uint32_t bit_depth,   const uint32_t int_frame_rate, double *ms_act)
{
    const int         o = (int) stride_org;
    const int         r = (int) stride_rec;
    const int16_t *o_m0 = pic_org    + offset_y * o + offset_x;
    int16_t       *o_m1 = pic_org_m1 + offset_y * o + offset_x;
    int16_t       *o_m2 = pic_org_m2 + offset_y * o + offset_x;
    const int16_t *r_m0 = pic_rec    + offset_y * r + offset_x;
    const int     b_val = (s->plane_width * s->plane_height > 2048 * 1152 ? 2 : 1); /* threshold is a bit more than HD resolution */
    const int     x_act = (offset_x > 0 ? 0 : b_val);
    const int     y_act = (offset_y > 0 ? 0 : b_val);
    const int     w_act = (offset_x + block_width  < (uint32_t) s->plane_width  ? (int) block_width  : (int) block_width  - b_val);
    const int     h_act = (offset_y + block_height < (uint32_t) s->plane_height ? (int) block_height : (int) block_height - b_val);

    const double sse = (double) calc_squared_error (o_m0, stride_org,
                                                    r_m0, stride_rec,
                                                    block_width, block_height);
    uint64_t sa_act = 0;  /* spatial abs. activity */
    uint64_t ta_act = 0; /* temporal abs. activity */

    if (w_act <= x_act || h_act <= y_act) /* small */
        return sse;

    if (b_val > 1) { /* highpass with downsampling */
        if (w_act > 12)
            sa_act = highds(x_act, y_act, w_act, h_act, o_m0, o);
        else
            highds(x_act, y_act, w_act, h_act, o_m0, o);
    } else { /* <=HD highpass without downsampling */
        for (int y = y_act; y < h_act; y++) {
            for (int x = x_act; x < w_act; x++) {
                const int f = 12 * (int)o_m0[y*o + x] - 2 * ((int)o_m0[y*o + x-1] + (int)o_m0[y*o + x+1] + (int)o_m0[(y-1)*o + x] + (int)o_m0[(y+1)*o + x])
                                 - ((int)o_m0[(y-1)*o + x-1] + (int)o_m0[(y-1)*o + x+1] + (int)o_m0[(y+1)*o + x-1] + (int)o_m0[(y+1)*o + x+1]);
                sa_act += (uint64_t) abs(f);
            }
        }
    }

    /* calculate weight (average squared activity) */
    *ms_act = (double) sa_act / ((double) (w_act - x_act) * (double) (h_act - y_act));

    if (b_val > 1) { /* highpass with downsampling */
        if (int_frame_rate < 32) /* 1st-order diff */
            ta_act = diff1st(block_width, block_height, o_m0, o_m1, o);
        else /* 2nd-order diff (diff of two diffs) */
            ta_act = diff2nd(block_width, block_height, o_m0, o_m1, o_m2, o);
    } else { /* <=HD highpass without downsampling */
        if (int_frame_rate < 32) { /* 1st-order diff */
            for (uint32_t y = 0; y < block_height; y++) {
                for (uint32_t x = 0; x < block_width; x++) {
                    const int t = (int)o_m0[y * o + x] - (int)o_m1[y * o + x];

                    ta_act += XPSNR_GAMMA * (uint64_t) abs(t);
                    o_m1[y * o + x] = o_m0[y * o + x];
                }
            }
        } else { /* 2nd-order diff (diff of 2 diffs) */
            for (uint32_t y = 0; y < block_height; y++) {
                for (uint32_t x = 0; x < block_width; x++) {
                    const int t = (int)o_m0[y * o + x] - 2 * (int)o_m1[y * o + x] + (int)o_m2[y * o + x];

                    ta_act += XPSNR_GAMMA * (uint64_t) abs(t);
                    o_m2[y * o + x] = o_m1[y * o + x];
                    o_m1[y * o + x] = o_m0[y * o + x];
                }
            }
        }
    }

    /* weight += mean squared temporal activity */
    *ms_act += (double) ta_act / ((double) block_width * (double) block_height);

    /* lower limit, accounts for high-pass gain */
    if (*ms_act < (double) (1 << (bit_depth - 6)))
        *ms_act = (double) (1 << (bit_depth - 6));

    *ms_act *= *ms_act; /* since SSE is squared */

    /* return nonweighted sum of squared errors */
    return sse;
}

/* get_wsse for the luma plane, and get_avg_xpsnr of the frame */
static double ffmpeg_frame(Ffmpeg *s, const int16_t *p_org, const int16_t *p_rec)
{
    const uint32_t       w = s->plane_width; /* luma image width in pixels */
    const uint32_t       h = s->plane_height;/* luma image height in pixels */
    const double         r = (double)(w * h) / (3840.0 * 2160.0); /* UHD ratio */
    const uint32_t       b = FFMAX(1, 4 * (int32_t) (32.0 * sqrt(r) +
                                                     0.5)); /* block size, integer multiple of 4 for SIMD */
    const uint32_t   w_blk = (w + b - 1) / b; /* luma width in units of blocks */
    const double   avg_act = sqrt(16.0 * (double) (1 << (2 * s->depth - 9)) / sqrt(FFMAX(0.00001,
                                                                                   r))); /* the sqrt(a_pic) */
    uint32_t x, y, idx_blk = 0; /* the "16.0" above is due to fixed-point code */
    double *const sse_luma = s->sse_luma;
    double *const  weights = s->weights;
    uint64_t wsse64;

    if (b >= 4) {
        const uint32_t s_org = w;
        const uint32_t s_rec = w;
        double     wsse_luma = 0.0;

        for (y = 0; y < h; y += b) { /* calculate block SSE and perceptual weights */
            const uint32_t block_height = (y + b > h ? h - y : b);

            for (x = 0; x < w; x += b, idx_blk++) {
                const uint32_t block_width = (x + b > w ? w - x : b);
                double ms_act = 1.0, ms_act_prev = 0.0;

                sse_luma[idx_blk] = calc_squared_error_and_weight(s, p_org, s_org,
                                                                  s->buf_org_m1 /* pixel  */,
                                                                  s->buf_org_m2 /* memory */,
                                                                  p_rec, s_rec,
                                                                  x, y,
                                                                  block_width, block_height,
                                                                  s->depth, s->frame_rate, &ms_act);
                weights[idx_blk] = 1.0 / sqrt(ms_act);

                if (w * h <= 640 * 480) { /* in-line "min-smoothing" as in paper */
                    if (x == 0) /* first column */
                        ms_act_prev = (idx_blk > 1 ? weights[idx_blk - 2] : 0);
                    else  /* after first column */
                        ms_act_prev = (x > b ? FFMAX(weights[idx_blk - 2], weights[idx_blk]) : weights[idx_blk]);

                    if (idx_blk > w_blk) /* after the first row and first column */
                        ms_act_prev = FFMAX(ms_act_prev, weights[idx_blk - 1 - w_blk]); /* min (L, T) */
                    if ((idx_blk > 0) && (weights[idx_blk - 1] > ms_act_prev))
                        weights[idx_blk - 1] = ms_act_prev;

                    if ((x + b >= w) && (y + b >= h) && (idx_blk > w_blk)) { /* last block in picture */
                        ms_act_prev = FFMAX(weights[idx_blk - 1], weights[idx_blk - w_blk]);
                        if (weights[idx_blk] > ms_act_prev)
                            weights[idx_blk] = ms_act_prev;
                    }
                }
            } /* for x */
        } /* for y */

        for (y = idx_blk = 0; y < h; y += b) { /* calculate sum for luma (Y) XPSNR */
            for (x = 0; x < w; x += b, idx_blk++) {
                wsse_luma += sse_luma[idx_blk] * weights[idx_blk];
            }
        }
        wsse64 = (wsse_luma <= 0.0 ? 0 : (uint64_t) (wsse_luma * avg_act + 0.5));
    } else {
        wsse64 = calc_squared_error (p_org, w, p_rec, w, w, h);
    }

    const double sqrt_wsse_val = sqrt((double) wsse64);
    const uint64_t num_frames_64 = 1;
    const uint64_t max_error_64 = (uint64_t) ((1 << s->depth) - 1) * (uint64_t) ((1 << s->depth) - 1);
    if (sqrt_wsse_val >= (double) num_frames_64) { /* square-mean-root average */
        const double avg_dist = sqrt_wsse_val / (double) num_frames_64;
        const uint64_t  num64 = (uint64_t) w * (uint64_t) h * max_error_64;

        return 10.0 * log10((double) num64 / ((double) avg_dist * (double) avg_dist));
    }
    return INFINITY / (double) num_frames_64;
}

/* ---- the sequences ---- */

/* Frame `t` of a scene: gradients and texture drifting a few samples a
 * frame, a flat patch, hard edges; `noise` scatters samples (the encode). */
static void fill(int16_t *out, unsigned w, unsigned h, unsigned bpc, unsigned t, uint32_t noise, uint32_t seed)
{
    const int peak = (1 << bpc) - 1;
    uint32_t state = seed * 2654435761u + t * 40503u + 1;
    for (unsigned y = 0; y < h; y++) {
        for (unsigned x = 0; x < w; x++) {
            state = state * 1664525u + 1013904223u;
            const unsigned xs = x + 3 * t, ys = y + t;
            int v = (int) (((xs * 7 + ys * 3) % 256) * (unsigned) peak / 255);
            v = (v * 3 + (int) ((((xs / 5) ^ (ys / 3)) & 31) * (unsigned) peak / 31)) / 4;
            if (x > w / 3 && x < w / 2 && y > h / 4 && y < h / 2)
                v = peak / 2;
            if ((xs / 29 + ys / 19) % 9 == 0)
                v = (x & 1) ? peak : 0;
            if (noise)
                v += (int) ((state >> 16) % (2 * noise + 1)) - (int) noise;
            out[y * w + x] = (int16_t) (v < 0 ? 0 : v > peak ? peak : v);
        }
    }
}

static int to_picture(VmafPicture *pic, const int16_t *luma, unsigned w, unsigned h, unsigned bpc,
                      enum VmafPixelFormat pix_fmt)
{
    int err = vmaf_picture_alloc(pic, pix_fmt, bpc, w, h);
    if (err) return err;
    for (unsigned y = 0; y < h; y++) {
        for (unsigned x = 0; x < w; x++) {
            if (bpc > 8)
                ((uint16_t *) pic->data[0])[y * (pic->stride[0] / 2) + x] = (uint16_t) luma[y * w + x];
            else
                ((uint8_t *) pic->data[0])[y * pic->stride[0] + x] = (uint8_t) luma[y * w + x];
        }
    }
    return 0;
}

typedef struct Case {
    unsigned w, h, bpc;
    int frame_rate, threads, n_subsample, weights_from_dist, frames;
    uint32_t noise;
    enum VmafPixelFormat pix_fmt;
} Case;

/* 0, or the first frame whose score is not FFmpeg's */
static int run_case(const Case *c)
{
    const size_t n = (size_t) c->w * c->h;
    int16_t *ref = malloc(n * sizeof(int16_t)), *dis = malloc(n * sizeof(int16_t));
    Ffmpeg s = { .depth = (int) c->bpc, .frame_rate = c->frame_rate,
                 .plane_width = (int) c->w, .plane_height = (int) c->h };
    s.buf_org_m1 = calloc(n, sizeof(int16_t));
    s.buf_org_m2 = calloc(n, sizeof(int16_t));
    s.sse_luma = malloc(n * sizeof(double));
    s.weights = malloc(n * sizeof(double));
    double *want = malloc(c->frames * sizeof(double));

    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE, .n_threads = c->threads,
                              .n_subsample = c->n_subsample };
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, cfg);
    VmafFeatureDictionary *opts = NULL;
    char rate[16];
    snprintf(rate, sizeof(rate), "%d", c->frame_rate);
    err |= vmaf_feature_dictionary_set(&opts, "frame_rate", rate);
    err |= vmaf_feature_dictionary_set(&opts, "weights_from_dist", c->weights_from_dist ? "true" : "false");
    err |= vmaf_use_feature(vmaf, "xpsnr", opts);
    int bad = err ? -1 : 0;

    for (int t = 0; t < c->frames && !bad; t++) {
        fill(ref, c->w, c->h, c->bpc, (unsigned) t, 0, 7);
        if (c->noise)
            fill(dis, c->w, c->h, c->bpc, (unsigned) t, c->noise, 7);
        else
            memcpy(dis, ref, n * sizeof(int16_t));
        /* the filter's original is its first input */
        want[t] = c->weights_from_dist ? ffmpeg_frame(&s, dis, ref) : ffmpeg_frame(&s, ref, dis);
        VmafPicture rp, dp;
        if (to_picture(&rp, ref, c->w, c->h, c->bpc, c->pix_fmt) ||
            to_picture(&dp, dis, c->w, c->h, c->bpc, c->pix_fmt) ||
            vmaf_read_pictures(vmaf, &rp, &dp, (unsigned) t))
            bad = -1;
    }
    if (!bad && vmaf_read_pictures(vmaf, NULL, NULL, 0))
        bad = -1;
    for (int t = 0; t < c->frames && !bad; t += c->n_subsample > 1 ? c->n_subsample : 1) {
        double got;
        if (vmaf_feature_score_at_index(vmaf, "xpsnr_y", &got, (unsigned) t) ||
            memcmp(&got, &want[t], sizeof(double)) != 0) {
            fprintf(stderr, "%ux%u %u-bit, %d fps, frame %d: %.17g, FFmpeg's %.17g\n",
                    c->w, c->h, c->bpc, c->frame_rate, t, got, want[t]);
            bad = t + 1;
        }
    }
    vmaf_close(vmaf);
    free(ref); free(dis); free(s.buf_org_m1); free(s.buf_org_m2); free(s.sse_luma); free(s.weights); free(want);
    return bad;
}

static char *test_every_frame_is_ffmpegs()
{
    const Case cases[] = {
        /* too small for blocks: unweighted */
        { 40, 30, 8, 24, 0, 1, 1, 4, 9, VMAF_PIX_FMT_YUV400P },
        /* min-smoothing, threads, subsampled, both weightings, both orders */
        { 320, 240, 8, 24, 4, 1, 1, 6, 9, VMAF_PIX_FMT_YUV400P },
        { 320, 240, 10, 60, 4, 3, 0, 7, 30, VMAF_PIX_FMT_YUV420P },
        { 638, 478, 12, 25, 2, 1, 1, 5, 60, VMAF_PIX_FMT_YUV400P },
        /* up to 2048x1152: odd sizes, no downsampling */
        { 1279, 719, 10, 30, 3, 1, 1, 6, 12, VMAF_PIX_FMT_YUV400P },
        { 1280, 720, 12, 50, 0, 1, 0, 5, 40, VMAF_PIX_FMT_YUV400P },
        { 1920, 1080, 8, 59, 4, 2, 1, 5, 6, VMAF_PIX_FMT_YUV420P },
        /* downsampled: a last block of 40 columns, one of 10 (too narrow
         * for the spatial activity), 4K */
        { 2560, 1440, 10, 24, 4, 1, 1, 5, 15, VMAF_PIX_FMT_YUV400P },
        { 3850, 2160, 8, 60, 4, 1, 1, 4, 8, VMAF_PIX_FMT_YUV400P },
        { 3840, 2160, 10, 23, 4, 2, 1, 5, 20, VMAF_PIX_FMT_YUV400P },
        { 3840, 2160, 10, 60, 0, 1, 0, 4, 20, VMAF_PIX_FMT_YUV400P },
        /* identical pictures: infinite */
        { 1280, 720, 10, 24, 2, 1, 1, 3, 0, VMAF_PIX_FMT_YUV400P },
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const int bad = run_case(&cases[i]);
        mu_assert("an XPSNR is not FFmpeg's (see above)", bad == 0);
    }
    return NULL;
}

static int refused(unsigned w, unsigned h, unsigned bpc)
{
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE };
    VmafContext *vmaf = NULL;
    if (vmaf_init(&vmaf, cfg) || vmaf_use_feature(vmaf, "xpsnr", NULL))
        return 0;
    VmafPicture rp, dp;
    vmaf_picture_alloc(&rp, VMAF_PIX_FMT_YUV400P, bpc, w, h);
    vmaf_picture_alloc(&dp, VMAF_PIX_FMT_YUV400P, bpc, w, h);
    const int err = vmaf_read_pictures(vmaf, &rp, &dp, 0);
    vmaf_close(vmaf);
    return err != 0;
}

static char *test_what_ffmpeg_reads_outside_the_picture_for_is_refused()
{
    mu_assert("an odd width above 2048x1152 not refused", refused(2561, 1440, 10));
    mu_assert("an odd height above 2048x1152 not refused", refused(2560, 1441, 10));
    mu_assert("16 bits not refused", refused(640, 480, 16));
    mu_assert("an odd size up to 2048x1152 refused", !refused(1279, 719, 10));
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_every_frame_is_ffmpegs);
    mu_run_test(test_what_ffmpeg_reads_outside_the_picture_for_is_refused);
    return NULL;
}
