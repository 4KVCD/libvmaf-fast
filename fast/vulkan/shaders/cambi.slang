// Derived from libvmaf (https://github.com/Netflix/vmaf): cambi.c and
// cambi.h, Copyright 2021-2026 Netflix, Inc., licensed under the BSD+Patent
// License (https://opensource.org/licenses/BSDplusPatent); see LICENSE at
// the root of this repository.
//
// CAMBI (VMAF v1's banding feature) of the distorted picture, as libvmaf's CPU
// code calculates it, bit for bit. One file, one step per STAGE:
//   PRE       the picture to 10 bits (cambi_preprocessing; anti-dithering
//             below 10 bits), and the 9..15-bit validation
//   DERIV     zero derivative: equal to its right and lower neighbours
//   MASK      the spatial mask: more than mask_index zero derivatives in
//             the 7x7 square around (zeros outside the picture)
//   DECIMATE  every other row and column, of the picture and of the mask
//   MODE      filter_mode: the mode of 3, across then down; the first and
//             last rows as they were
//   CVALUES   calculate_c_values: each masked pixel's histogram of its
//             (window x window) square counted directly -- what the CPU's
//             sliding histograms hold for it -- and c_value_pixel, with the
//             CPU's reciprocal table (its entries as written, not 1.0f/i)
//   CLEAR     zeros
//   HIST      a level of the radix select of spatial_pooling's top k: the
//             c-values' bits are ordered as the floats are (all >= 0)
//   SELECT    the bin the k-th largest value is in, at that level
//   SUM       the values above the k-th largest, exactly: each nonzero
//             c-value is at least 0.5, a multiple of 2^-24 -- an integer in
//             units of 2^-24 -- so their sum is exact in 64 bits, as the
//             CPU's double sum is while it is under 2^53 units (v1_host)
#include "common.slang"

[[vk::push_constant]] uniform uint pc[24];

#if STAGE == 1  // PRE
[[vk::binding(0, 0)]] StructuredBuffer<uint> picture;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> image;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> acc;
// pc: w, h, strideWords, bpc, antiDither, errorSlot

uint to10(int x, int y)
{
    const uint bpc = pc[3];
    uint v = pixel(picture, x, y, int(pc[2]));
    if (bpc <= 8) return v << (10 - bpc);
    if (bpc == 9) return v << 1;
    if (bpc == 10) return v;
    const uint shift = bpc - 10;
    return (v + (1u << (shift - 1))) >> shift;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int w = int(pc[0]), h = int(pc[1]), x = int(id.x), y = int(id.y);
    if (x >= w || y >= h) return;
    const uint bpc = pc[3];
    if (bpc > 8 && bpc < 16 && pixel(picture, x, y, int(pc[2])) > (1u << bpc) - 1u)
        InterlockedOr(acc[2 * pc[5]], 1u);
    uint v = to10(x, y);
    if (pc[4] != 0) {  // anti_dithering_filter: of the converted picture, which it reads ahead of what it wrote
        if (x < w - 1 && y < h - 1)
            v = (v + to10(x + 1, y) + to10(x, y + 1) + to10(x + 1, y + 1)) >> 2;
        else if (y < h - 1)
            v = (v + to10(x, y + 1)) >> 1;
        else if (x < w - 1)
            v = (v + to10(x + 1, y)) >> 1;
    }
    image[y * w + x] = v;
}

#elif STAGE == 2  // DERIV
[[vk::binding(0, 0)]] StructuredBuffer<uint> image;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> zero;
// pc: w, h

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int w = int(pc[0]), h = int(pc[1]), x = int(id.x), y = int(id.y);
    if (x >= w || y >= h) return;
    const uint v = image[y * w + x];
    const bool across = x == w - 1 || v == image[y * w + x + 1];
    const bool down = y == h - 1 || v == image[(y + 1) * w + x];
    zero[y * w + x] = (across && down) ? 1u : 0u;
}

#elif STAGE == 3  // MASK
[[vk::binding(0, 0)]] StructuredBuffer<uint> zero;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> mask;
// pc: w, h, maskIndex

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int w = int(pc[0]), h = int(pc[1]), x = int(id.x), y = int(id.y);
    if (x >= w || y >= h) return;
    uint sum = 0;
    for (int r = max(y - 3, 0); r <= min(y + 3, h - 1); ++r)
        for (int c = max(x - 3, 0); c <= min(x + 3, w - 1); ++c)
            sum += zero[r * w + c];
    mask[y * w + x] = sum > pc[2] ? 1u : 0u;
}

#elif STAGE == 4  // DECIMATE
[[vk::binding(0, 0)]] StructuredBuffer<uint> imageIn;
[[vk::binding(1, 0)]] StructuredBuffer<uint> maskIn;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> imageOut;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> maskOut;
// pc: w, h (the decimated size), inW

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int w = int(pc[0]), h = int(pc[1]), x = int(id.x), y = int(id.y);
    if (x >= w || y >= h) return;
    const int from = (2 * y) * int(pc[2]) + 2 * x;
    imageOut[y * w + x] = imageIn[from];
    maskOut[y * w + x] = maskIn[from];
}

#elif STAGE == 5  // MODE
[[vk::binding(0, 0)]] StructuredBuffer<uint> imageIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> imageOut;
// pc: w, h

uint min3(uint a, uint b, uint c)
{
    if (a <= b && a <= c) return a;
    if (b <= c) return b;
    return c;
}

uint mode3(uint a, uint b, uint c)
{
    if (a == b || a == c) return a;
    if (b == c) return b;
    return min3(a, b, c);
}

uint across(int x, int y, int w)
{
    if (x == 0 || x == w - 1) return imageIn[y * w + x];
    return mode3(imageIn[y * w + x - 1], imageIn[y * w + x], imageIn[y * w + x + 1]);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int w = int(pc[0]), h = int(pc[1]), x = int(id.x), y = int(id.y);
    if (x >= w || y >= h) return;
    if (y == 0 || y == h - 1)
        imageOut[y * w + x] = imageIn[y * w + x];
    else
        imageOut[y * w + x] = mode3(across(x, y - 1, w), across(x, y, w), across(x, y + 1, w));
}

#elif STAGE == 6  // CVALUES
[[vk::binding(0, 0)]] StructuredBuffer<uint> image;
[[vk::binding(1, 0)]] StructuredBuffer<uint> mask;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> cValues;
[[vk::binding(3, 0)]] StructuredBuffer<float> reciprocal;
// pc: w, h, pad, vltLuma, bandBase, bandSize, tvi[0..3] (+ num_diffs), weights[0..3]
#define NUM_DIFFS 4

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int w = int(pc[0]), h = int(pc[1]), x = int(id.x), y = int(id.y);
    if (x >= w || y >= h) return;
    float c = 0.0;
    const int I = int(image[y * w + x]);
    const int bandBase = int(pc[4]), bandSize = int(pc[5]);
    const int compact = I - bandBase;
    if (mask[y * w + x] != 0 && compact >= 0 && compact < bandSize) {
        // The masked, in-band pixels of the clipped square, by value - I.
        uint counts[2 * NUM_DIFFS + 1];
        for (int i = 0; i < 2 * NUM_DIFFS + 1; ++i) counts[i] = 0;
        const int pad = int(pc[2]);
        for (int r = max(y - pad, 0); r <= min(y + pad, h - 1); ++r) {
            for (int col = max(x - pad, 0); col <= min(x + pad, w - 1); ++col) {
                const int v = int(image[r * w + col]);
                const int delta = v - I;
                if (delta >= -NUM_DIFFS && delta <= NUM_DIFFS && mask[r * w + col] != 0
                    && v - bandBase >= 0 && v - bandBase < bandSize)
                    counts[delta + NUM_DIFFS] += 1;
            }
        }
        const uint p0 = counts[NUM_DIFFS];
        const int value = I + NUM_DIFFS;
        for (int d = 0; d < NUM_DIFFS; ++d) {
            if (value <= int(pc[6 + d]) && value + d + 1 > int(pc[3])) {
                const uint p1 = counts[NUM_DIFFS + d + 1];
                const uint p2 = compact - d - 1 >= 0 ? counts[NUM_DIFFS - d - 1] : 0u;
                const uint pm = p1 > p2 ? p1 : p2;
                // (float)(diff_weights[d] * p_0 * p_m) * reciprocal_lut[p_m + p_0]: an exact int, one rounding.
                const float val = float(int(pc[10 + d]) * int(p0) * int(pm)) * reciprocal[pm + p0];
                if (val > c) c = val;
            }
        }
    }
    cValues[y * w + x] = asuint(c);
}

#elif STAGE == 7  // CLEAR
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> target;
// pc: count

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < pc[0]) target[id.x] = 0;
}

#elif STAGE == 8  // HIST
[[vk::binding(0, 0)]] StructuredBuffer<uint> cValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> hist;    // every scale's and level's bins
[[vk::binding(2, 0)]] StructuredBuffer<uint> state;     // every scale's prefix, remaining
// pc: n, level, groupsTotal, histOffset (this scale's and level's bins), stateOffset

groupshared uint gHist[4096];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const uint level = pc[1];
    const uint bins = level < 2 ? 4096u : 256u;
    for (uint b = local; b < bins; b += 256) gHist[b] = 0;
    GroupMemoryBarrierWithGroupSync();
    const uint prefix = state[pc[4]];
    const uint stride = pc[2] * 256;
    for (uint i = group.x * 256 + local; i < pc[0]; i += stride) {
        const uint bits = cValues[i];
        uint bin;
        if (level == 0) {
            bin = bits >> 20;
        } else if (level == 1) {
            if ((bits >> 20) != prefix) continue;
            bin = (bits >> 8) & 0xFFFu;
        } else {
            if ((bits >> 8) != prefix) continue;
            bin = bits & 0xFFu;
        }
        InterlockedAdd(gHist[bin], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint b = local; b < bins; b += 256)
        if (gHist[b] != 0) InterlockedAdd(hist[pc[3] + b], gHist[b]);
}

#elif STAGE == 9  // SELECT
[[vk::binding(0, 0)]] StructuredBuffer<uint> hist;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> state;   // prefix, remaining
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> acc;
// pc: level, k, slotThreshold, slotRemaining, histOffset, stateOffset

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    const uint level = pc[0];
    const uint bins = level < 2 ? 4096u : 256u;
    const uint at = pc[5];
    uint remaining = level == 0 ? pc[1] : state[at + 1];
    uint prefix = level == 0 ? 0u : state[at];
    uint chosen = 0;
    for (int b = int(bins) - 1; b >= 0; --b) {
        const uint count = hist[pc[4] + uint(b)];
        if (count >= remaining) { chosen = uint(b); break; }
        remaining -= count;
    }
    prefix = level == 0 ? chosen : (prefix << (level < 2 ? 12u : 8u)) | chosen;
    state[at] = prefix;
    state[at + 1] = remaining;
    if (level == 2) {
        acc[2 * pc[2]] = prefix;       // the k-th largest value's bits
        acc[2 * pc[3]] = remaining;    // how many of it are in the top k
    }
}

#elif STAGE == 10  // SUM
[[vk::binding(0, 0)]] StructuredBuffer<uint> cValues;
[[vk::binding(1, 0)]] StructuredBuffer<uint> state;     // the threshold's bits
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> acc;
// pc: n, groupsTotal, slotSum, slotInexact, stateOffset

groupshared uint64_t gSum[256];
groupshared uint gInexact;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    if (local == 0) gInexact = 0;
    GroupMemoryBarrierWithGroupSync();
    const uint threshold = state[pc[4]];
    const uint stride = pc[1] * 256;
    uint64_t sum = 0;
    for (uint i = group.x * 256 + local; i < pc[0]; i += stride) {
        const uint bits = cValues[i];
        if (bits <= threshold) continue;
        // bits > threshold >= 0: a positive float, mantissa * 2^(e - 150), in units of 2^-24.
        const uint e = bits >> 23;
        const uint64_t mantissa = uint64_t((bits & 0x7FFFFFu) | 0x800000u);
        if (e >= 126 && e < 142) {  // [0.5, 32768): at most 2^39 units, 2^60 for 2^21 of them
            sum += mantissa << (e - 126);
        } else {
            gInexact = 1;  // below 0.5, or too large: v1_host pools it on the CPU
        }
    }
    gSum[local] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 128; step > 0; step >>= 1) {
        if (local < step) gSum[local] += gSum[local + step];
        GroupMemoryBarrierWithGroupSync();
    }
    if (local == 0) {
        if (gSum[0] != 0) acc_add64(acc, pc[2], gSum[0]);
        if (gInexact != 0) InterlockedOr(acc[2 * pc[3]], 1u);
    }
}
#endif

#if STAGE == 11  // ARGS
// Which scales spatial_pooling pools on the CPU (v1_host): those whose top k
// is not exact in a double -- a c-value not a whole number of 2^-24, or a
// sum of 2^53 units or more -- (or every scale, `force`, for the tests).
// Writes their KEEP passes' dispatch sizes, 0 for the others, and marks
// them (bit 1 of the scale's inexact slot).
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> acc;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> args;
// pc: force, slotBase, groups[0..4]

bool units(uint bits, out uint64_t value)
{
    value = 0;
    if (bits == 0) return true;
    const uint e = bits >> 23;
    if (e < 126 || e >= 142) return false;
    value = uint64_t((bits & 0x7FFFFFu) | 0x800000u) << (e - 126);
    return true;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    for (uint scale = 0; scale < 5; ++scale) {
        const uint slot = pc[1] + scale * 4;
        const uint64_t sum = uint64_t(acc[2 * slot]) | (uint64_t(acc[2 * slot + 1]) << 32);
        const uint64_t remaining = uint64_t(acc[2 * (slot + 2)]);
        uint64_t threshold;
        bool need = pc[0] != 0 || (acc[2 * (slot + 3)] & 1u) != 0 || !units(acc[2 * (slot + 1)], threshold);
        if (!need) need = sum + remaining * threshold >= (1ull << 53);
        args[3 * scale] = need ? pc[2 + scale] : 0u;
        args[3 * scale + 1] = 1;
        args[3 * scale + 2] = 1;
        if (need) acc[2 * (slot + 3)] |= 2u;
    }
}

#elif STAGE == 12  // KEEP
// A scale's c-values, into the frame's buffer for the CPU's pooling.
[[vk::binding(0, 0)]] StructuredBuffer<uint> cValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> keep;
// pc: n, offset

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < pc[0]) keep[pc[1] + id.x] = cValues[id.x];
}
#endif
