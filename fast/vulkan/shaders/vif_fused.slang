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
// The next scale's images, one word a position: the reference's sample in
// the low 16 bits, the distorted's in the high (scales 1-3 read refIn so;
// disIn and disRd are not used). Two words, one per picture, were twice the
// writes here and the loads there.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> refRd;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> disRd;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> acc;
[[vk::binding(5, 0)]] StructuredBuffer<uint> logTable;  // log_generate(32768 + i)

#define NATIVE_F64 0
#define OFF ((FW - FW_RD) / 2)
#define SMALL (SRC_PICTURE && !BPC16)
#define SUMS 9
// The tile: TW pixels along a row, TH rows, a thread a pixel. Timed on the
// Core Ultra 9 285K's iGPU (1080p VIF scale 0): of 20 shapes, two rows of 80
// or 96 were the fastest once the group's sums were subgroup sums; with the
// sums packed and the sums of squares narrow, one row of 160 (2.85 ms,
// against 2.91 for 80 x 2, 2.90 for 192 x 1, 2.94 for 128 x 1, 2.98 for
// 64 x 2, 3.01 for 112 x 2, 3.43 for 240 x 1). Rows share no vertical work;
// a wider tile computes fewer columns beyond its pixels (16 of 176 here).
#define TW 160
#ifndef TH
#define TH 1
#endif
#define THREADS 160
#if TW * TH > 255
#error "group_words packs the sums of at most 255 pixels"
#endif
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
// The ten values added up in seven words: the small ones packed, as
// group_words says.
#define WORDS 7
groupshared int gTotals[WORDS];

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

#ifndef NARROW
#define NARROW 0
#endif
#if NARROW
// Pictures of 9 to 12 bits (shiftSq 2..8): the sums of squares and
// products kept as high and low parts split at bit 8, without vif_vert's
// carries. Each product a * b is a coefficient times a sample (a below
// 2^29) times a sample: the low parts, (a & 255) * b, add up to under 2^29
// over the 17 taps, and the high parts, (a >> 8) * b, are kept modulo 2^32,
// which keeps their total, high * 2^8 + low, right modulo 2^40. Only bits
// shiftSq .. shiftSq + 31 of it are kept (the result is a 32-bit word), all
// below 2^40: the same word as the 48-bit sum's, whatever the samples. (At
// 10 bits nothing wraps at all: the high parts add up to under 2^28.)
#define add48(sum, a, b) { sum.high += ((a) >> 8) * (b); sum.low += ((a) & 0xFFu) * (b); }
#define total48(sum) ((uint64_t(sum.high) << 8) + uint64_t(sum.low))
#endif

// vif_vert.slang's main for one pixel, into the tile at `at`.
void vertical(int x, int y, uint at)
{
    uint mu1 = 0, mu2 = 0, refRdSum = 0, disRdSum = 0;
    // The next scale's samples come from even rows only.
    const bool nextRow = (y & 1) == 0;
#if SMALL
    uint refSq = 0, disSq = 0, refDis = 0;
#else
    Sum48 refSq = { 0, 0, 0 }, disSq = { 0, 0, 0 }, refDis = { 0, 0, 0 };
#endif
    // Rows inside the picture (all but a few groups' at the top and bottom):
    // a word index stepped a row at a time, no mirroring. The samples and
    // what is done with them are the same either way.
    if (y >= FW / 2 && y + FW / 2 < pc.h) {
#if SRC_PICTURE
#if BPC16
        int index = (y - FW / 2) * pc.stride + (x >> 1);
        const uint shift = uint(x & 1) * 16, mask = 0xFFFFu;
#else
        int index = (y - FW / 2) * pc.stride + (x >> 2);
        const uint shift = uint(x & 3) * 8, mask = 0xFFu;
#endif
#else
        int index = (y - FW / 2) * pc.stride + x;
#endif
        for (int fi = 0; fi < FW; ++fi) {
#if SRC_PICTURE
            const uint r = (refIn[index] >> shift) & mask, d = (disIn[index] >> shift) & mask;
#else
            const uint samples = refIn[index], r = samples & 0xFFFFu, d = samples >> 16;
#endif
            index += pc.stride;
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
            if (nextRow && fi >= OFF && fi < FW - OFF) {
                uint next = kVifFilter[SCALE + 1][fi - OFF];
                refRdSum += next * r;
                disRdSum += next * d;
            }
#endif
        }
    } else {
        for (int fi = 0; fi < FW; ++fi) {
            int row = abs(y - FW / 2 + fi);
            if (row >= pc.h)
                row = 2 * pc.h - row - 2;
#if SRC_PICTURE
            uint r = pixel(refIn, x, row, pc.stride), d = pixel(disIn, x, row, pc.stride);
#else
            const uint samples = refIn[row * pc.stride + x], r = samples & 0xFFFFu, d = samples >> 16;
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
            if (nextRow && fi >= OFF && fi < FW - OFF) {
                uint next = kVifFilter[SCALE + 1][fi - OFF];
                refRdSum += next * r;
                disRdSum += next * d;
            }
#endif
        }
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
    if (nextRow)
        gTile[4 * plane + at] = (((refRdSum + pc.addVP) >> pc.shiftVP) & 0xFFFFu)
                                | ((((disRdSum + pc.addVP) >> pc.shiftVP) & 0xFFFFu) << 16);
#endif
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const int x0 = int(group.x) * TW, y0 = int(group.y) * TH;
    if (local < WORDS)
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
        uint mu1 = 0, mu2 = 0;
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
    }
#if FW_RD > 0
    // The next scale's samples: even pixels of even rows, a thread each (all
    // of a subgroup's lanes busy, where in the loop above half would wait).
    for (uint o = local; o < uint(TW / 2 * TH); o += THREADS) {
        const int row = int(o / uint(TW / 2)), px = 2 * int(o % uint(TW / 2));
        const int x = x0 + px, y = y0 + row;
        if (x >= pc.w || y >= pc.h || (y & 1) != 0)
            continue;
        const uint base = uint(row * TCOLS + px);
        uint refRdSum = 0, disRdSum = 0;
        for (int fj = OFF; fj <= FW / 2; ++fj) {
            const uint a = base + uint(fj), b = base + uint(FW - 1 - fj);
            const bool pair = fj < FW / 2;
            const uint next = kVifFilter[SCALE + 1][fj - OFF];
            const uint samplesA = gTile[4 * plane + a], samplesB = pair ? gTile[4 * plane + b] : 0u;
            refRdSum += next * ((samplesA & 0xFFFFu) + (samplesB & 0xFFFFu));
            disRdSum += next * ((samplesA >> 16) + (samplesB >> 16));
        }
        const int at = (y / 2) * pc.rdStride + x / 2;
        refRd[at] = (((refRdSum + 32768u) >> 16) & 0xFFFFu) | ((((disRdSum + 32768u) >> 16) & 0xFFFFu) << 16);
    }
#endif

    // Each subgroup's sum of every word, added into the group's: the words'
    // sums are the values' sums, packed the same way (nothing overflows its
    // field: see group_words).
    const int words[WORDS] = { totals[0] + (totals[1] << 13) + (totals[5] << 21), totals[2], totals[3], totals[4],
                               totals[7], totals[9], totals[6] + (totals[8] << 15) };
    for (int i = 0; i < WORDS; ++i) {
        const int sum = WaveActiveSum(words[i]);
        if (WaveIsFirstLane())
            InterlockedAdd(gTotals[i], sum);
    }
    GroupMemoryBarrierWithGroupSync();
    if (local < SUMS) {  // a sum a thread
        // group_words: a group has at most 255 pixels, and a pixel adds
        // -16..-3 to x, 0 or 1 to num_x and den_non_log, -63..63 to each
        // limit's x2: x (at least -4080) in the first word's low 13 bits,
        // num_x and den_non_log (at most 255) 8 bits each above it; the two x2
        // (within +-16065) in 15 bits each of the last word.
        const int first = gTotals[0], last = gTotals[6];
        const int x = (first << 19) >> 19, x2 = (last << 17) >> 17;
        const int counts = (first - x) >> 13;
        const int values[SUMS] = { x, counts & 0xFF, gTotals[1], 0, counts >> 8, x2, gTotals[4], (last - x2) >> 15,
                                   gTotals[5] };
        const int64_t value = local == 3 ? int64_t(gTotals[2]) + (int64_t(gTotals[3]) << 16) : int64_t(values[local]);
        if (value != 0)
            acc_add64(acc, pc.slot + local, uint64_t(value));
    }
}
