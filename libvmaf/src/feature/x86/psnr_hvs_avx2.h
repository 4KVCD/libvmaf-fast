/*
 * Copyright 2001-2012 Xiph.Org and contributors.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef LIBVMAF_FEATURE_X86_PSNR_HVS_AVX2_H
#define LIBVMAF_FEATURE_X86_PSNR_HVS_AVX2_H

/*
 * libvmaf-fast: the blocks of one block row of calc_psnrhvs
 * (third_party/xiph/psnr_hvs.c), 8 at a time: blocks at x = 0, step, ...
 * of the row at y, while all 8 of a group fit (x < w - 7). Adds their error
 * terms to *ret in the scalar code's order and 64 per block to *pixels,
 * with the same values. mask and csf are the 8 x 8 tables, row by row.
 * Returns the x of the first block it did not do.
 */
int psnr_hvs_blocks_avx2(const unsigned char *src, int src_stride, const unsigned char *dst,
                         int dst_stride, int depth, int y, int w, int step, const float *mask,
                         const float *csf, float *ret, int *pixels);

#endif /* LIBVMAF_FEATURE_X86_PSNR_HVS_AVX2_H */
