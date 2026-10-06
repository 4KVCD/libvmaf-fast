// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// VIF, horizontal pass and statistics: filter1d_8_horizontal_kernel /
// filter1d_16_horizontal_kernel of filter1d.cu with vif_statistic_calculation
// of vif_statistics.cuh. One shader per scale (SCALE, FW, FW_RD).
//
// One pass gives two sets of sums: one per enhancement gain limit (VMAF's 100
// and VMAF NEG's 1). The limit only caps the gain g after everything else is
// calculated, so the filters and the rest of the statistics are shared.
//
// The CUDA kernel calculates g and what follows from it in double:
//     g     = sigma12 / (sigma1_sq + eps)
//     sv_sq = sigma2_sq - g * sigma12       one fused multiply-add in the
//                                           compiled kernel (DFMA)
//     g*g*sigma1_sq                         two multiplications
// Here they are integer arithmetic with the same rounding (round53 for each
// double operation), or, with NATIVE_F64, the GPU's own division and
// multiplications; the fused step is integer in both, as a driver may or may
// not fuse fma().
#include "common.slang"
#include "vif_filter.slang"

struct Params {
    int w, h;
    int rdStride;      // samples per row of the next scale's images
    uint slot;         // this scale's first sum
    uint epsLo, epsHi; // the kernel's eps as a double (NATIVE_F64)
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> tmp;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> refRd;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> disRd;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;
[[vk::binding(4, 0)]] StructuredBuffer<uint> logTable;  // log_generate(32768 + i)

#define OFF ((FW - FW_RD) / 2)
#define SUMS 9
// x, num_x, den_log, num_non_log, den_non_log, then x2 and num_log per limit
#define GROUP 64

// The group's threads add their sums up in two steps, in 32 bits: a GPU
// without 64-bit integers of its own is slow at each 64-bit one, and here
// every thread took part in them. A pixel's sums as WORDS values of which 64
// cannot overflow 32 bits:
//   0  -x (the shift k, 3..16) | num_x << 11 | den_non_log << 18   (the two
//      are 0 or 1 a pixel)
//   1  den_log (a logarithm: 16 bits)
//   2, 3  num_non_log's low and high 16 bits (sigma2_sq: 31 bits)
//   4, 6  x2 of each limit (a difference of two shifts)
//   5, 7  num_log of each limit (a difference of two logarithms)
#define WORDS 8
groupshared int gSums[GROUP * WORDS];
// The row's stretch the group's filters read, read once from tmp by the
// group: each pixel's words are read by FW threads.
#define TILE (GROUP + FW - 1)
// Word by word (structure of arrays): the lanes of a load read words next
// to each other.
groupshared uint gTile[VIF_TMP_WORDS * TILE];
groupshared int gPartial[(GROUP / 8) * WORDS];

#include "vif_stats.slang"

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint local : SV_GroupIndex)
{
    int x = int(id.x), y = int(id.y);
    int sums[SUMS];
    for (int i = 0; i < SUMS; ++i)
        sums[i] = 0;
    // The tile: the columns x - FW/2 .. x + FW/2 of the group's pixels,
    // mirrored at the edges as the filter mirrors them. Columns only the
    // group's pixels beyond the row read are clamped into it (and unused).
    for (int i = int(local); i < TILE; i += GROUP) {
        int column = abs(x - int(local) - FW / 2 + i);
        if (column >= pc.w)
            column = 2 * pc.w - column - 2;
        column = clamp(column, 0, pc.w - 1);
        uint from = (uint(y * pc.w) + uint(column)) * VIF_TMP_WORDS;
        for (int word = 0; word < VIF_TMP_WORDS; ++word)
            gTile[word * TILE + i] = tmp[from + word];
    }
    GroupMemoryBarrierWithGroupSync();
    if (x < pc.w && y < pc.h) {
        uint mu1 = 0, mu2 = 0, refRdSum = 0, disRdSum = 0;
        // The 48-bit sums of coefficient * value in two 32-bit halves (of the
        // value's high and low 16 bits), neither of which can overflow: the
        // coefficients add up to 65536. No 64-bit multiply-adds, which a GPU
        // without 64-bit integers of its own is slow at.
        uint refSqHigh = 0, refSqLow = 0, disSqHigh = 0, disSqLow = 0, refDisHigh = 0, refDisLow = 0;
        // The filters are symmetric: the two taps of a coefficient are added
        // up before it multiplies them, half the multiplications. The sums
        // are the same integers (none of these can overflow 32 bits).
        for (int fj = 0; fj <= FW / 2; ++fj) {
            uint a = local + uint(fj), b = local + uint(FW - 1 - fj);
            uint coefficient = kVifFilter[SCALE][fj];
            uint meansA = gTile[a], meansB = fj < FW / 2 ? gTile[b] : 0u;
            uint refSqA = gTile[TILE + a], refSqB = fj < FW / 2 ? gTile[TILE + b] : 0u;
            uint disSqA = gTile[2 * TILE + a], disSqB = fj < FW / 2 ? gTile[2 * TILE + b] : 0u;
            uint refDisA = gTile[3 * TILE + a], refDisB = fj < FW / 2 ? gTile[3 * TILE + b] : 0u;
            mu1 += coefficient * ((meansA & 0xFFFFu) + (meansB & 0xFFFFu));
            mu2 += coefficient * ((meansA >> 16) + (meansB >> 16));
            refSqHigh += coefficient * ((refSqA >> 16) + (refSqB >> 16));
            refSqLow += coefficient * ((refSqA & 0xFFFFu) + (refSqB & 0xFFFFu));
            disSqHigh += coefficient * ((disSqA >> 16) + (disSqB >> 16));
            disSqLow += coefficient * ((disSqA & 0xFFFFu) + (disSqB & 0xFFFFu));
            refDisHigh += coefficient * ((refDisA >> 16) + (refDisB >> 16));
            refDisLow += coefficient * ((refDisA & 0xFFFFu) + (refDisB & 0xFFFFu));
#if FW_RD > 0
            if (fj >= OFF && (x & 1) == 0) {
                uint next = kVifFilter[SCALE + 1][fj - OFF];
                uint samplesA = gTile[4 * TILE + a], samplesB = fj < FW / 2 ? gTile[4 * TILE + b] : 0u;
                refRdSum += next * ((samplesA & 0xFFFFu) + (samplesB & 0xFFFFu));
                disRdSum += next * ((samplesA >> 16) + (samplesB >> 16));
            }
#endif
        }
        // (uint32_t)((sum + 32768) >> 16)
        statistics(mu1, mu2, refSqHigh + ((refSqLow + 32768u) >> 16), disSqHigh + ((disSqLow + 32768u) >> 16),
                   refDisHigh + ((refDisLow + 32768u) >> 16), sums);
#if FW_RD > 0
        if ((x & 1) == 0 && (y & 1) == 0) {
            int at = (y / 2) * pc.rdStride + x / 2;
            refRd[at] = ((refRdSum + 32768u) >> 16) & 0xFFFFu;
            disRd[at] = ((disRdSum + 32768u) >> 16) & 0xFFFFu;
        }
#endif
    }
    gSums[local * WORDS + 0] = -sums[0] | (sums[1] << 11) | (sums[4] << 18);
    gSums[local * WORDS + 1] = sums[2];
    gSums[local * WORDS + 2] = sums[3] & 0xFFFF;
    gSums[local * WORDS + 3] = sums[3] >> 16;
    gSums[local * WORDS + 4] = sums[5];
    gSums[local * WORDS + 5] = sums[6];
    gSums[local * WORDS + 6] = sums[7];
    gSums[local * WORDS + 7] = sums[8];
    GroupMemoryBarrierWithGroupSync();
    if ((local & 7) == 0) {
        for (int i = 0; i < WORDS; ++i) {
            int sum = 0;
            for (uint t = 0; t < 8; ++t)
                sum += gSums[(local + t) * WORDS + i];
            gPartial[(local / 8) * WORDS + i] = sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        int words[WORDS];
        for (int i = 0; i < WORDS; ++i) {
            words[i] = 0;
            for (uint t = 0; t < GROUP / 8; ++t)
                words[i] += gPartial[t * WORDS + i];
        }
        int64_t totals[SUMS];
        totals[0] = -int64_t(words[0] & 0x7FF);
        totals[1] = int64_t((words[0] >> 11) & 0x7F);
        totals[2] = int64_t(words[1]);
        totals[3] = int64_t(words[2]) + (int64_t(words[3]) << 16);
        totals[4] = int64_t((words[0] >> 18) & 0x7F);
        totals[5] = int64_t(words[4]);
        totals[6] = int64_t(words[5]);
        totals[7] = int64_t(words[6]);
        totals[8] = int64_t(words[7]);
        for (int i = 0; i < SUMS; ++i) {
            if (totals[i] != 0)
                acc_add64(acc, pc.slot + uint(i), uint64_t(totals[i]));
        }
    }
}
