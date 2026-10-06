/*
 * Copyright (c) 2024 Christian R. Helmrich
 * Copyright (c) 2024 Christian Lehmann
 * Copyright (c) 2024 Christian Stoffers
 *
 * XPSNR's arithmetic for one sample type, included by xpsnr.c with T (the
 * sample type) and FN (a name for its functions) defined. From FFmpeg
 * 9.0.1's libavfilter/vf_xpsnr.c, changed for libvmaf-fast in October 2026
 * as xpsnr.c says.
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

/* FFmpeg's sse_line_16bit, a row at a time: its sum of a row is 32 bits. */
static uint64_t FN(sse_block)(const T *org, ptrdiff_t so, const T *rec, ptrdiff_t sr,
                              uint32_t block_width, uint32_t block_height)
{
    uint64_t sse = 0;

    for (uint32_t y = 0; y < block_height; y++) {
        unsigned m2 = 0;
        for (uint32_t x = 0; x < block_width; x++) {
            const unsigned error = (unsigned) ((int) org[x] - (int) rec[x]);
            m2 += error * error;
        }
        sse += m2;
        org += so;
        rec += sr;
    }
    return sse;
}

static uint64_t FN(highds)(const int x_act, const int y_act, const int w_act, const int h_act,
                           const T *o_m0, const ptrdiff_t o)
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

/* FFmpeg's diff1st and diff2nd, with the pictures before for its history
 * buffers (each its own stride; 0: a row of zeros, before the first). */
static uint64_t FN(diff1st)(const uint32_t w_act, const uint32_t h_act, const T *o_m0, const ptrdiff_t o,
                            const T *o_m1, const ptrdiff_t o1)
{
    uint64_t ta_act = 0;

    for (uint32_t y = 0; y < h_act; y += 2) {
        for (uint32_t x = 0; x < w_act; x += 2) {
            const int t = (int)o_m0[y*o + x] + (int)o_m0[y*o + x+1] + (int)o_m0[(y+1)*o + x] + (int)o_m0[(y+1)*o + x+1]
                       - ((int)o_m1[y*o1 + x] + (int)o_m1[y*o1 + x+1] + (int)o_m1[(y+1)*o1 + x] + (int)o_m1[(y+1)*o1 + x+1]);
            ta_act += (uint64_t) abs(t);
        }
    }
    return (ta_act * XPSNR_GAMMA);
}

static uint64_t FN(diff2nd)(const uint32_t w_act, const uint32_t h_act, const T *o_m0, const ptrdiff_t o,
                            const T *o_m1, const ptrdiff_t o1, const T *o_m2, const ptrdiff_t o2)
{
    uint64_t ta_act = 0;

    for (uint32_t y = 0; y < h_act; y += 2) {
        for (uint32_t x = 0; x < w_act; x += 2) {
            const int t = (int)o_m0[y*o + x] + (int)o_m0[y*o + x+1] + (int)o_m0[(y+1)*o + x] + (int)o_m0[(y+1)*o + x+1]
                   - 2 * ((int)o_m1[y*o1 + x] + (int)o_m1[y*o1 + x+1] + (int)o_m1[(y+1)*o1 + x] + (int)o_m1[(y+1)*o1 + x+1])
                        + (int)o_m2[y*o2 + x] + (int)o_m2[y*o2 + x+1] + (int)o_m2[(y+1)*o2 + x] + (int)o_m2[(y+1)*o2 + x+1];
            ta_act += (uint64_t) abs(t);
        }
    }
    return (ta_act * XPSNR_GAMMA);
}

/* FFmpeg's calc_squared_error_and_weight. */
static double FN(block)(const XpsnrState *s, const XpsnrPlanes *p, const uint32_t offset_x,
                        const uint32_t offset_y, const uint32_t block_width, const uint32_t block_height,
                        double *ms_act)
{
    const ptrdiff_t o = p->so, o1 = p->s1, o2 = p->s2;
    const T *o_m0 = (const T *) p->org + offset_y * o  + offset_x;
    const T *o_m1 = (const T *) p->m1  + offset_y * o1 + offset_x;
    const T *o_m2 = (const T *) p->m2  + offset_y * o2 + offset_x;
    const T *r_m0 = (const T *) p->rec + offset_y * p->sr + offset_x;
    const int b_val = s->b_val;
    const int x_act = (offset_x > 0 ? 0 : b_val);
    const int y_act = (offset_y > 0 ? 0 : b_val);
    const int w_act = (offset_x + block_width  < s->w ? (int) block_width  : (int) block_width  - b_val);
    const int h_act = (offset_y + block_height < s->h ? (int) block_height : (int) block_height - b_val);

    const double sse = (double) FN(sse_block)(o_m0, o, r_m0, p->sr, block_width, block_height);
    uint64_t sa_act = 0;  /* spatial abs. activity */
    uint64_t ta_act = 0; /* temporal abs. activity */

    if (w_act <= x_act || h_act <= y_act) /* small */
        return sse;

    if (b_val > 1) { /* highpass with downsampling */
        /* FFmpeg calls highds here for narrower blocks too, and drops what
         * it returns: their spatial activity is 0. */
        if (w_act > 12)
            sa_act = FN(highds)(x_act, y_act, w_act, h_act, o_m0, o);
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
        if (s->frame_rate < 32) /* 1st-order diff */
            ta_act = FN(diff1st)(block_width, block_height, o_m0, o, o_m1, o1);
        else /* 2nd-order diff (diff of two diffs) */
            ta_act = FN(diff2nd)(block_width, block_height, o_m0, o, o_m1, o1, o_m2, o2);
    } else { /* <=HD highpass without downsampling */
        if (s->frame_rate < 32) { /* 1st-order diff */
            for (uint32_t y = 0; y < block_height; y++) {
                for (uint32_t x = 0; x < block_width; x++) {
                    const int t = (int)o_m0[y * o + x] - (int)o_m1[y * o1 + x];

                    ta_act += XPSNR_GAMMA * (uint64_t) abs(t);
                }
            }
        } else { /* 2nd-order diff (diff of 2 diffs) */
            for (uint32_t y = 0; y < block_height; y++) {
                for (uint32_t x = 0; x < block_width; x++) {
                    const int t = (int)o_m0[y * o + x] - 2 * (int)o_m1[y * o1 + x] + (int)o_m2[y * o2 + x];

                    ta_act += XPSNR_GAMMA * (uint64_t) abs(t);
                }
            }
        }
    }

    /* weight += mean squared temporal activity */
    *ms_act += (double) ta_act / ((double) block_width * (double) block_height);

    /* lower limit, accounts for high-pass gain */
    if (*ms_act < (double) (1 << (s->bpc - 6)))
        *ms_act = (double) (1 << (s->bpc - 6));

    *ms_act *= *ms_act; /* since SSE is squared */

    /* return nonweighted sum of squared errors */
    return sse;
}
