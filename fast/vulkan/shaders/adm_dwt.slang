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
// The parameters as specialization constants (constant k: Params' word k):
// the engine compiles the shader for its passes' (a pipeline for each set),
// and the compiler folds what they decide.
[vk::constant_id(0)] const int kSpec_w = 0;
#define P_w kSpec_w
[vk::constant_id(1)] const int kSpec_h = 0;
#define P_h kSpec_h
[vk::constant_id(2)] const int kSpec_inStride = 0;
#define P_inStride kSpec_inStride
[vk::constant_id(3)] const int kSpec_outStride = 0;
#define P_outStride kSpec_outStride
[vk::constant_id(4)] const int kSpec_shiftV = 0;
#define P_shiftV kSpec_shiftV
[vk::constant_id(5)] const int kSpec_addV = 0;
#define P_addV kSpec_addV
[vk::constant_id(6)] const int kSpec_shiftH = 0;
#define P_shiftH kSpec_shiftH
[vk::constant_id(7)] const int kSpec_addH = 0;
#define P_addH kSpec_addH
[vk::constant_id(8)] const int kSpec_aStride = 0;
#define P_aStride kSpec_aStride

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
#ifndef A16
#define A16 0
#endif
#if A16
// Scale 0's a stored a 16-bit value at a time (where the GPU can: the
// engine's samples16), two a word as below: no pairing in group memory.
[[vk::binding(4, 0)]] RWStructuredBuffer<int16_t> refA;
[[vk::binding(5, 0)]] RWStructuredBuffer<int16_t> disA;
#else
[[vk::binding(4, 0)]] RWStructuredBuffer<int> refA;    // a
[[vk::binding(5, 0)]] RWStructuredBuffer<int> disA;
#endif
// Scale 0's a paired into words by the group's even columns' threads.
#define PAIR_A (SCALE0 && !A16)

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

#if SCALE0
// Row y's samples at columns 2x - 1 .. 2x + 2, all inside the picture: from
// the words holding them (16-bit: three; 8-bit: two), where a load a sample
// was four.
int4 row_samples(StructuredBuffer<uint> source, int x, int y)
{
    const int first = y * P_inStride;
#if BPC16
    const uint w0 = source[first + x - 1], w1 = source[first + x], w2 = source[first + x + 1];
    return int4(int(w0 >> 16), int(w1 & 0xFFFFu), int(w1 >> 16), int(w2 & 0xFFFFu));
#else
    // Bytes 3 of word x/2 - 1 and 0..2 of word x/2 (x even), or 1..3 of word
    // (x - 1)/2 and 0 of the next (x odd).
    const int word = (2 * x - 1) >> 2;
    const uint a = source[first + word], b = source[first + word + 1];
    const uint shift = uint((2 * x - 1) & 3) * 8;  // 24 or 8
    const uint four = (a >> shift) | (b << (32u - shift));
    return int4(int(four & 0xFFu), int((four >> 8) & 0xFFu), int((four >> 16) & 0xFFu), int(four >> 24));
#endif
}
#endif

// Writes position (x, y)'s h, v, d and returns its a.
int transform(StructuredBuffer<uint> source, RWStructuredBuffer<int> bands, int4 columns, int4 rows, int x, int y,
              int at)
{
#if SCALE0
    // Positions away from the picture's edges (their 4 x 4 samples all
    // inside it) read them a row at a time.
    int samples[4][4];  // [row][column]
    if (x >= 1 && 2 * x + 2 < P_w && y >= 1 && 2 * y + 2 < P_h) {
        for (int i = 0; i < 4; ++i) {
            const int4 four = row_samples(source, x, rows[i]);
            for (int c = 0; c < 4; ++c)
                samples[i][c] = four[c];
        }
    } else {
        for (int i = 0; i < 4; ++i)
            for (int c = 0; c < 4; ++c)
                samples[i][c] = int(pixel(source, columns[c], rows[i], P_inStride));
    }
    int lo[4], hi[4];
    for (int c = 0; c < 4; ++c) {
        int sumLo = 0, sumHi = 0;
        for (int i = 0; i < 4; ++i) {
            int sample = samples[i][c];
            sumLo += kLo[i] * sample;
            sumHi += kHi[i] * sample;
        }
        sumLo -= kLoSum * P_addV;
        lo[c] = sext16((sumLo + P_addV) >> P_shiftV);
        hi[c] = sext16((sumHi + P_addV) >> P_shiftV);
    }
    int a = 0, v = 0, h = 0, d = 0;
    for (int c = 0; c < 4; ++c) {
        a += kLo[c] * lo[c];
        v += kHi[c] * lo[c];
        h += kLo[c] * hi[c];
        d += kHi[c] * hi[c];
    }
    bands[at * ADM_HVD_WORDS0] = (((h + P_addH) >> P_shiftH) & 0xFFFF) | (((v + P_addH) >> P_shiftH) << 16);
    bands[at * ADM_HVD_WORDS0 + 1] = sext16((d + P_addH) >> P_shiftH);
    // In [-27411, 27410]: lo in [-32768, 32767], so a, kLo's sum, is in
    // [-1796403056, 1796356714] (no overflow), and shiftH, addH are 16, 32768.
    return (a + P_addH) >> P_shiftH;
#else
    int lo[4], hi[4];
    for (int c = 0; c < 4; ++c) {
        int64_t sumLo = 0, sumHi = 0;
        for (int i = 0; i < 4; ++i) {
#if DWT_SCALE == 1
            // Scale 0's a, two a word (inStride positions a row, even).
            const int position = rows[i] * P_inStride + columns[c];
            const int64_t sample = int64_t(int(source[position >> 1] << (16 - 16 * (position & 1))) >> 16);
#else
            const int64_t sample = int64_t(int(source[rows[i] * P_inStride + columns[c]]));
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

#if PAIR_A
// The group's a, ref's then dis's, for the even columns' threads to pair.
groupshared int gA[2 * 16 * 8];
#endif

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID, uint local : SV_GroupIndex)
{
    int x = int(id.x), y = int(id.y);
    const bool inside = x < (P_w + 1) / 2 && y < (P_h + 1) / 2;
#if PAIR_A
    int aRef = 0, aDis = 0;
#else
    if (!inside)
        return;
#endif
    if (inside) {
        int4 columns = indices(x, P_w), rows = indices(y, P_h);
        int at = y * P_outStride + x;
#if PAIR_A
        aRef = transform(refIn, refOut, columns, rows, x, y, at);
        aDis = transform(disIn, disOut, columns, rows, x, y, at);
#elif A16
        // In [-27411, 27410] (transform): 16-bit, two a word.
        refA[y * P_aStride + x] = int16_t(transform(refIn, refOut, columns, rows, x, y, at));
        disA[y * P_aStride + x] = int16_t(transform(disIn, disOut, columns, rows, x, y, at));
#else
        refA[at] = transform(refIn, refOut, columns, rows, x, y, at);
        disA[at] = transform(disIn, disOut, columns, rows, x, y, at);
#endif
    }
#if PAIR_A
    // a two a word, the even column's low half, written by its thread (the
    // group 16 columns wide from an even one: x + 1 is local + 1). Past the
    // last column, 0.
    gA[local] = aRef;
    gA[128 + local] = aDis;
    GroupMemoryBarrierWithGroupSync();
    if (inside && (x & 1) == 0) {
        const int word = (y * P_aStride + x) >> 1;
        refA[word] = (aRef & 0xFFFF) | (gA[local + 1] << 16);
        disA[word] = (aDis & 0xFFFF) | (gA[128 + local + 1] << 16);
    }
#endif
}
