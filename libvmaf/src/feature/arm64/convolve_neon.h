/**
 *
 *  Copyright (c) 2011, Tom Distler (http://tdistler.com)
 *  Copyright 2016-2026 Netflix, Inc.
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent AND BSD-3-Clause
 */

#ifndef ARM64_CONVOLVE_NEON_H_
#define ARM64_CONVOLVE_NEON_H_

/*
 * NEON copy of iqa's 1D separable convolution (iqa/convolve.c,
 * IQA_CONVOLVE_1D), the Arm counterpart of x86/convolve_avx2.c, with the
 * same arguments: a normalized kernel (normalized must be 1) given as its
 * two 1D halves, no border handling. workspace is a w * h float cache, or
 * NULL to allocate one; result NULL convolves in place. Bit-identical to
 * the C code.
 */
void iqa_convolve_neon(float *img, int w, int h, const float *kernel_h, const float *kernel_v,
                       int kw, int kh, int normalized, float *workspace, float *result, int *rw,
                       int *rh);

#endif /* ARM64_CONVOLVE_NEON_H_ */
