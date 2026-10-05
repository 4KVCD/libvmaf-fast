// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// VIF's vertical and horizontal passes in one (vif_vert.slang then
// vif_hori.slang), for a tile of TW x TH pixels a group: the vertical
// pass's results for the columns the tile's horizontal filters read are made
// in group memory, not written to and read back from memory -- on an
// integrated GPU, which shares the memory's bandwidth with everything else,
// most of the two passes' time. The integers are the two passes', computed
// by the same code; only where they are kept differs.
#include "common.slang"
#include "vif_filter.slang"

struct Params {
    int w, h;
    int stride;          // words per picture row (scale 0), or samples per row
    uint shiftVP, addVP;
    uint shiftSq, addSq;
    int rdStride;        // samples per row of the next scale's images
    uint slot;           // this scale's first sum
    uint gainA, gainB;   // the two gain limits; integers
    uint epsLo, epsHi;   // the kernel's eps as a double (unused: no NATIVE_F64)
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> refIn;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> refRd;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> disRd;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> acc;
[[vk::binding(5, 0)]] StructuredBuffer<uint> logTable;  // log_generate(32768 + i)

#define NATIVE_F64 0
#define OFF ((FW - FW_RD) / 2)
#define SMALL (SRC_PICTURE && !BPC16)
#define SUMS 9
#define TW 64                  // the tile: pixels along a row
#ifndef TH
#define TH 4                   // and rows
#endif
#define THREADS 128
#define TCOLS (TW + FW - 1)    // the columns the tile's horizontal filters read
// The group's sums: the nine of statistics(), the fourth (num_non_log, 31
// bits a pixel) as its low and high 16 bits. 512 pixels' worth of each
// fits in 32 bits.
#define VALUES 10
#define TILE_WORDS (VIF_TMP_WORDS * TH * TCOLS)
// The vertical pass's results, word by word, row by row.
groupshared uint gTile[TILE_WORDS];
// The group's sums, each subgroup's added in (the engine uses this shader
// only where the GPU has subgroup arithmetic). Integer sums: the same in
// any order.
groupshared int gTotals[VALUES];

#include "vif_stats.slang"


// vif_vert.slang's Sum48.
struct Sum48 { uint high, low, carries; };

void add48(inout Sum48 sum, uint a, uint b)
{
    sum.high += (a >> 16) * b;
    uint low = (a & 0xFFFFu) * b;
    sum.low += low;
    sum.carries += sum.low < low ? 1u : 0u;
}

uint64_t total48(Sum48 sum)
{
    return (uint64_t(sum.high) << 16) + (uint64_t(sum.carries) << 32) + uint64_t(sum.low);
}

// vif_vert.slang's main for one pixel, into the tile at `at`.
void vertical(int x, int y, uint at)
{
    uint mu1 = 0, mu2 = 0, refRdSum = 0, disRdSum = 0;
#if SMALL
    uint refSq = 0, disSq = 0, refDis = 0;
#else
    Sum48 refSq = { 0, 0, 0 }, disSq = { 0, 0, 0 }, refDis = { 0, 0, 0 };
#endif
    for (int fi = 0; fi < FW; ++fi) {
        int row = abs(y - FW / 2 + fi);
        if (row >= pc.h)
            row = 2 * pc.h - row - 2;
#if SRC_PICTURE
        uint r = pixel(refIn, x, row, pc.stride), d = pixel(disIn, x, row, pc.stride);
#else
        uint r = refIn[row * pc.stride + x], d = disIn[row * pc.stride + x];
#endif
        uint coefficient = kVifFilter[SCALE][fi];
        uint cr = coefficient * r, cd = coefficient * d;
        mu1 += cr;
        mu2 += cd;
#if SMALL
        refSq += cr * r;
        disSq += cd * d;
        refDis += cr * d;
#else
        add48(refSq, cr, r);
        add48(disSq, cd, d);
        add48(refDis, cr, d);
#endif
#if FW_RD > 0
        if (fi >= OFF && fi < FW - OFF) {
            uint next = kVifFilter[SCALE + 1][fi - OFF];
            refRdSum += next * r;
            disRdSum += next * d;
        }
#endif
    }
    const uint plane = TH * TCOLS;
    gTile[at] = (((mu1 + pc.addVP) >> pc.shiftVP) & 0xFFFFu) | ((((mu2 + pc.addVP) >> pc.shiftVP) & 0xFFFFu) << 16);
#if SMALL
    gTile[plane + at] = refSq;
    gTile[2 * plane + at] = disSq;
    gTile[3 * plane + at] = refDis;
#else
    gTile[plane + at] = uint((total48(refSq) + pc.addSq) >> pc.shiftSq);
    gTile[2 * plane + at] = uint((total48(disSq) + pc.addSq) >> pc.shiftSq);
    gTile[3 * plane + at] = uint((total48(refDis) + pc.addSq) >> pc.shiftSq);
#endif
#if FW_RD > 0
    gTile[4 * plane + at] = (((refRdSum + pc.addVP) >> pc.shiftVP) & 0xFFFFu)
                            | ((((disRdSum + pc.addVP) >> pc.shiftVP) & 0xFFFFu) << 16);
#endif
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const int x0 = int(group.x) * TW, y0 = int(group.y) * TH;
    if (local < VALUES)
        gTotals[local] = 0;  // before the barrier below; added to after it
    // The vertical pass for the tile's columns x0 - FW/2 .. x0 + TW - 1 + FW/2,
    // mirrored at the edges as the horizontal filter mirrors them (columns
    // only pixels beyond the row read are clamped into it, and unused).
    for (uint e = local; e < uint(TH * TCOLS); e += THREADS) {
        const int row = int(e / uint(TCOLS)), column = int(e % uint(TCOLS));
        const int y = y0 + row;
        if (y >= pc.h)
            continue;
        int x = abs(x0 - FW / 2 + column);
        if (x >= pc.w)
            x = 2 * pc.w - x - 2;
        vertical(clamp(x, 0, pc.w - 1), y, uint(row * TCOLS + column));
    }
    GroupMemoryBarrierWithGroupSync();

    int totals[VALUES];
    for (int i = 0; i < VALUES; ++i)
        totals[i] = 0;
    const uint plane = TH * TCOLS;
    for (uint o = local; o < uint(TW * TH); o += THREADS) {
        const int row = int(o / uint(TW)), px = int(o % uint(TW));
        const int x = x0 + px, y = y0 + row;
        if (x >= pc.w || y >= pc.h)
            continue;
        const uint base = uint(row * TCOLS + px);
        uint mu1 = 0, mu2 = 0, refRdSum = 0, disRdSum = 0;
        uint refSqHigh = 0, refSqLow = 0, disSqHigh = 0, disSqLow = 0, refDisHigh = 0, refDisLow = 0;
        // vif_hori.slang's filter: symmetric taps added before their
        // coefficient multiplies them, 48-bit sums in two 32-bit halves.
        for (int fj = 0; fj <= FW / 2; ++fj) {
            const uint a = base + uint(fj), b = base + uint(FW - 1 - fj);
            const uint coefficient = kVifFilter[SCALE][fj];
            const bool pair = fj < FW / 2;
            uint meansA = gTile[a], meansB = pair ? gTile[b] : 0u;
            uint refSqA = gTile[plane + a], refSqB = pair ? gTile[plane + b] : 0u;
            uint disSqA = gTile[2 * plane + a], disSqB = pair ? gTile[2 * plane + b] : 0u;
            uint refDisA = gTile[3 * plane + a], refDisB = pair ? gTile[3 * plane + b] : 0u;
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
                uint samplesA = gTile[4 * plane + a], samplesB = pair ? gTile[4 * plane + b] : 0u;
                refRdSum += next * ((samplesA & 0xFFFFu) + (samplesB & 0xFFFFu));
                disRdSum += next * ((samplesA >> 16) + (samplesB >> 16));
            }
#endif
        }
        int sums[SUMS];
        for (int i = 0; i < SUMS; ++i)
            sums[i] = 0;
        statistics(mu1, mu2, refSqHigh + ((refSqLow + 32768u) >> 16), disSqHigh + ((disSqLow + 32768u) >> 16),
                   refDisHigh + ((refDisLow + 32768u) >> 16), sums);
        totals[0] += sums[0];
        totals[1] += sums[1];
        totals[2] += sums[2];
        totals[3] += sums[3] & 0xFFFF;
        totals[4] += sums[3] >> 16;
        for (int i = 4; i < SUMS; ++i)
            totals[i + 1] += sums[i];
#if FW_RD > 0
        if ((x & 1) == 0 && (y & 1) == 0) {
            int at = (y / 2) * pc.rdStride + x / 2;
            refRd[at] = ((refRdSum + 32768u) >> 16) & 0xFFFFu;
            disRd[at] = ((disRdSum + 32768u) >> 16) & 0xFFFFu;
        }
#endif
    }

    for (int i = 0; i < VALUES; ++i) {
        const int sum = WaveActiveSum(totals[i]);
        if (WaveIsFirstLane())
            InterlockedAdd(gTotals[i], sum);
    }
    GroupMemoryBarrierWithGroupSync();
    if (local < SUMS) {  // a sum a thread
        const int64_t value = local < 3 ? int64_t(gTotals[local])
                              : local == 3 ? int64_t(gTotals[3]) + (int64_t(gTotals[4]) << 16)
                                           : int64_t(gTotals[local + 1]);
        if (value != 0)
            acc_add64(acc, pc.slot + local, uint64_t(value));
    }
}
