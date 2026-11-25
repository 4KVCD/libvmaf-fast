/**
 *
 *  Copyright (c) 2011, Tom Distler (http://tdistler.com)
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent AND BSD-3-Clause
 */

#ifndef ARM64_NEON_DECIMATE_H_
#define ARM64_NEON_DECIMATE_H_

/* Splits a row of w floats into its even samples (row[0], row[2], ...) and
 * its odd ones (row[1], row[3], ...). */
void iqa_deinterleave_neon(const float *row, int w, float *even, float *odd);

/*
 * Output pixels x in [x0, x1) of one output row, as x86/decimate_avx2.h's
 * iqa_decimate2_row_avx2: even[v] and odd[v] are the split input rows of
 * the kernel's row v (kh rows, centred on the output's input row). Every
 * tap of the kw x kh 2D kernel (odd sizes) must be inside the picture. Each
 * output is _iqa_filter_pixel's: the float product of every tap, in its
 * order (rows outer, columns inner), added to a double, then rounded to
 * float. Returns the first x it did not do (a multiple of 4 past x0); the
 * caller does the rest with _iqa_filter_pixel.
 */
int iqa_decimate2_row_neon(const float *const *even, const float *const *odd,
                           const float *kernel, int kw, int kh, int x0, int x1, float *dst);

#endif /* ARM64_NEON_DECIMATE_H_ */
