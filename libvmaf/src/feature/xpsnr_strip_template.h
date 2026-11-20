/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

/*
 * XPSNR's sums for one strip of blocks, row by row, the same for every
 * SIMD instruction set: included at the end of x86/xpsnr_any_avx2_kernels.h
 * and arm64/xpsnr_any_neon_kernels.h, with T (the sample type), FN (a name
 * for its functions), SSE_ROW (the squared errors of a row, as a 32-bit
 * sum), XPSNR_STRIP (the function's name) and their kernels
 * FN(highpass_row), FN(highds_row), FN(temporal_row) and
 * FN(temporal_ds_row) defined. See xpsnr_strip.h.
 */

void XPSNR_STRIP(const XpsnrStrip *st, uint64_t *sse, uint64_t *sa, uint64_t *ta)
{
    const int w = st->w, b = st->b, b_val = st->b_val;
    const int blocks = (w + b - 1) / b;
    const ptrdiff_t so = st->so, sr = st->sr, s1 = st->s1;
    /* the first order is the second with m2 = m1 */
    const T *const m2 = st->second_order ? st->m2 : st->m1;
    const ptrdiff_t s2 = st->second_order ? st->s2 : st->s1;

    memset(sse, 0, sizeof(*sse) * blocks);
    memset(sa, 0, sizeof(*sa) * blocks);
    memset(ta, 0, sizeof(*ta) * blocks);
    /* b_val 2 works on row pairs: its sums of activity take every second
     * row (y_act is 0 or 2), its squared errors both */
    for (int y = 0; y < st->height; y += b_val) {
        const T *const org = (const T *) st->org + y * so;
        const T *const rec = (const T *) st->rec + y * sr;
        const T *const p1 = (const T *) st->m1 + y * s1;
        const T *const p2 = m2 + y * s2;
        const bool spatial = y >= st->y_act && y < st->h_act;
        for (int bx = 0, x0 = 0; bx < blocks; bx++, x0 += b) {
            const int bw = b < w - x0 ? b : w - x0;
            const int readable = w - x0;  /* the samples to the row's end */
            for (int r = 0; r < b_val; r++)
                sse[bx] += SSE_ROW(org + r * so + x0, rec + r * sr + x0, bw);
            if (spatial) {
                /* the columns FN(block) of xpsnr_template.c takes */
                const int x_act = x0 > 0 ? 0 : b_val;
                const int w_act = x0 + bw < w ? bw : bw - b_val;
                if (b_val == 1)
                    sa[bx] += FN(highpass_row)(x_act, w_act, org + x0, so, readable);
                else if (w_act > 12)  /* narrower: FFmpeg drops the sum */
                    sa[bx] += FN(highds_row)(x_act, w_act, org + x0, so, readable);
            }
            ta[bx] += b_val == 1
                ? FN(temporal_row)(bw, org + x0, p1 + x0, p2 + x0, readable)
                : FN(temporal_ds_row)(bw, org + x0, so, p1 + x0, s1, p2 + x0, s2, readable);
        }
    }
}
