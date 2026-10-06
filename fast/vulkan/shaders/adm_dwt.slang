// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// ADM, the wavelet transform of both pictures. SCALE0: adm_dwt2_8_vert_hori_kernel
// of adm_dwt2.cu (the pictures to 16-bit bands); otherwise
// dwt_s123_combined_vert_kernel and _hori_kernel (the previous scale's
// approximation band to 32-bit bands). Each output takes its four columns'
// vertical sums itself instead of from a temporary image: the same integers.
//
// The bands go to two images (common.slang's adm_hvd): a and h, v, d. The
// other scales read the previous one's a: at scale 0 16-bit, two a word (the
// CUDA code's 32-bit i4 copy, whose values fit: below), else one a word.
#include "common.slang"

struct Params {
    int w, h;          // of the input
    int inStride;      // words per picture row, or positions per band row
    int outStride;
    int shiftV, addV;
    int shiftH, addH;
    int aStride;       // scale 0's a: positions a row, even (two a word)
};
[[vk::push_constant]] Params pc;

#if !SCALE0
// Scales 1-3, built one by one (DWT_SCALE 1-3) with their shifts (the
// engine's kV, kH) known: the 64-bit sums shift by constants, and scale 1's
// vertical step, which shifts by none and keeps 32 bits, makes 32-bit ones.
static const int kShiftV[4] = { 0, 0, 16, 16 }, kAddV[4] = { 0, 0, 32768, 32768 };
static const int kShiftH[4] = { 16, 15, 16, 15 }, kAddH[4] = { 32768, 16384, 32768, 16384 };
#define SHIFT_V kShiftV[DWT_SCALE]
#define ADD_V kAddV[DWT_SCALE]
#define SHIFT_H kShiftH[DWT_SCALE]
#define ADD_H kAddH[DWT_SCALE]
#endif
[[vk::binding(0, 0)]] StructuredBuffer<uint> refIn;   // the picture, or the previous scale's a
[[vk::binding(1, 0)]] StructuredBuffer<uint> disIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<int> refOut;  // h, v, d
[[vk::binding(3, 0)]] RWStructuredBuffer<int> disOut;
[[vk::binding(4, 0)]] RWStructuredBuffer<int> refA;    // a
[[vk::binding(5, 0)]] RWStructuredBuffer<int> disA;

static const int kLo[4] = { 15826, 27411, 7345, -4240 };
static const int kHi[4] = { -4240, -7345, 27411, -15826 };
static const int kLoSum = 46342;

// calculate_indices
int4 indices(int n, int limit)
{
    int4 at = int4(2 * n - 1, 2 * n, 2 * n + 1, 2 * n + 2);
    if (n == 0)
        at = int4(1, 0, 1, 2);
    if (at.x >= limit) at.x = 2 * limit - at.x - 1;
    if (at.y >= limit) at.y = 2 * limit - at.y - 1;
    if (at.z >= limit) at.z = 2 * limit - at.z - 1;
    if (at.w >= limit) at.w = 2 * limit - at.w - 1;
    return at;
}

// Writes h, v, d and returns a.
int transform(StructuredBuffer<uint> source, RWStructuredBuffer<int> bands, int4 columns, int4 rows, int at)
{
#if SCALE0
    int lo[4], hi[4];
    for (int c = 0; c < 4; ++c) {
        int sumLo = 0, sumHi = 0;
        for (int i = 0; i < 4; ++i) {
            int sample = int(pixel(source, columns[c], rows[i], pc.inStride));
            sumLo += kLo[i] * sample;
            sumHi += kHi[i] * sample;
        }
        sumLo -= kLoSum * pc.addV;
        lo[c] = sext16((sumLo + pc.addV) >> pc.shiftV);
        hi[c] = sext16((sumHi + pc.addV) >> pc.shiftV);
    }
    int a = 0, v = 0, h = 0, d = 0;
    for (int c = 0; c < 4; ++c) {
        a += kLo[c] * lo[c];
        v += kHi[c] * lo[c];
        h += kLo[c] * hi[c];
        d += kHi[c] * hi[c];
    }
    bands[at * ADM_HVD_WORDS0] = (((h + pc.addH) >> pc.shiftH) & 0xFFFF) | (((v + pc.addH) >> pc.shiftH) << 16);
    bands[at * ADM_HVD_WORDS0 + 1] = sext16((d + pc.addH) >> pc.shiftH);
    // In [-27411, 27410]: lo in [-32768, 32767], so a, kLo's sum, is in
    // [-1796403056, 1796356714] (no overflow), and shiftH, addH are 16, 32768.
    return (a + pc.addH) >> pc.shiftH;
#else
    int lo[4], hi[4];
    for (int c = 0; c < 4; ++c) {
        int64_t sumLo = 0, sumHi = 0;
        for (int i = 0; i < 4; ++i) {
#if DWT_SCALE == 1
            // Scale 0's a, two a word (inStride positions a row, even).
            const int position = rows[i] * pc.inStride + columns[c];
            const int64_t sample = int64_t(int(source[position >> 1] << (16 - 16 * (position & 1))) >> 16);
#else
            const int64_t sample = int64_t(int(source[rows[i] * pc.inStride + columns[c]]));
#endif
            sumLo += int64_t(kLo[i]) * sample;
            sumHi += int64_t(kHi[i]) * sample;
        }
        lo[c] = int((sumLo + ADD_V) >> SHIFT_V);
        hi[c] = int((sumHi + ADD_V) >> SHIFT_V);
    }
    int64_t a = 0, v = 0, h = 0, d = 0;
    for (int c = 0; c < 4; ++c) {
        a += int64_t(kLo[c]) * lo[c];
        v += int64_t(kHi[c]) * lo[c];
        h += int64_t(kLo[c]) * hi[c];
        d += int64_t(kHi[c]) * hi[c];
    }
    bands[at * ADM_HVD_WORDS] = int((h + ADD_H) >> SHIFT_H);
    bands[at * ADM_HVD_WORDS + 1] = int((v + ADD_H) >> SHIFT_H);
    bands[at * ADM_HVD_WORDS + 2] = int((d + ADD_H) >> SHIFT_H);
    return int((a + ADD_H) >> SHIFT_H);
#endif
}

#if SCALE0
// The group's a, ref's then dis's, for the even columns' threads to pair.
groupshared int gA[2 * 16 * 8];
#endif

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID, uint local : SV_GroupIndex)
{
    int x = int(id.x), y = int(id.y);
    const bool inside = x < (pc.w + 1) / 2 && y < (pc.h + 1) / 2;
#if SCALE0
    int aRef = 0, aDis = 0;
#else
    if (!inside)
        return;
#endif
    if (inside) {
        int4 columns = indices(x, pc.w), rows = indices(y, pc.h);
        int at = y * pc.outStride + x;
#if SCALE0
        aRef = transform(refIn, refOut, columns, rows, at);
        aDis = transform(disIn, disOut, columns, rows, at);
#else
        refA[at] = transform(refIn, refOut, columns, rows, at);
        disA[at] = transform(disIn, disOut, columns, rows, at);
#endif
    }
#if SCALE0
    // a two a word, the even column's low half, written by its thread (the
    // group 16 columns wide from an even one: x + 1 is local + 1). Past the
    // last column, 0.
    gA[local] = aRef;
    gA[128 + local] = aDis;
    GroupMemoryBarrierWithGroupSync();
    if (inside && (x & 1) == 0) {
        const int word = (y * pc.aStride + x) >> 1;
        refA[word] = (aRef & 0xFFFF) | (gA[local + 1] << 16);
        disA[word] = (aDis & 0xFFFF) | (gA[128 + local + 1] << 16);
    }
#endif
}
