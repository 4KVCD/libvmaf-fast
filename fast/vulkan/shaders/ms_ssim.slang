// Derived from libvmaf (https://github.com/Netflix/vmaf): ms_ssim.c,
// iqa/ssim_tools.c, iqa/decimate.c and iqa/convolve.c, Copyright 2016-2020
// Netflix, Inc. and Tom Distler, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent) and the BSD license of
// tdistler.com's IQA library; see LICENSE at the root of this repository.
//
// MS-SSIM (vv_v1_ms_ssim): one plane's five scales, each a pass of SSIM
// (STAGE 1) and, for the first four, a pass making the next scale (STAGE 2),
// both reading the same scale, so the two run side by side.
//
// float_ms_ssim's arithmetic on the CPU, per scale:
//  - SSIM over the window's valid places only (an 11-tap Gaussian, so
//    (w - 10) x (h - 10) of them): the Gaussian is applied across, then down,
//    each output the float products of the taps added in double and rounded
//    to float; to the pictures, their squares and their product (each a float
//    product), then the sigmas in float (ssim_maps). Here the taps are added
//    in float with each addition's rounding error kept and added back (two-
//    sum): nearly always the double sum's float.
//  - Each place's luminance l and contrast c in double, structure s in float
//    (_iqa_ssim_terms); here all three in float (Vulkan's division and square
//    root need not be correctly rounded either), so a scale's means come out
//    within about 1e-7 of the CPU's.
//  - The next scale (ceil(w / 2) x ceil(h / 2)): a 9x9 low-pass filter, not
//    separable, centred on every second sample, mirrored at the edges
//    (KBND_SYMMETRIC), the taps added as above.
// Each place's l, c and s is added as a whole number of 2^-32 (round to
// nearest; s may be negative: the 64-bit sums wrap as two's complement), so
// the frame's sums are the same in any order; collect() makes the scale's
// means and the score from them as ms_ssim.c does.
//
// Built with NO_CONTRACTION (every float multiply and add rounded on its
// own), which the two-sums need, except on MoltenVK
// (build_vmaf_vulkan_shaders.py's MOLTENVK_CONTRACTED).

#include "common.slang"

[[vk::push_constant]] uniform uint pc[32];
// pc: 0-1 this scale's w and h; 2 where it is read from: 0 the frame's planes
// (scale 0: bindings 0 and 1), 1 the scales buffer; 3 the planes' stride
// (words); 4-5 the reference's and the distorted's first word in bindings 0
// and 1; 6 the bit depth; 7-8 the reference's and the distorted's first
// float in the scales buffer (pc[2] 1).
// STAGE 1: 9 the l sum's slot (c's and s's follow); 10-12 C1, C2, C3 (float
// bits, ssim_default_constants'). STAGE 2: 9-10 the next scale's reference
// and distorted first floats.
[[vk::binding(0, 0)]] StructuredBuffer<uint> refPlane;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disPlane;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> scales;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;

#ifndef STAGE
#define STAGE 1
#endif

// A sample of the scale at (x, y), x < w and y < h: a plane's as
// picture_copy_row makes it a float (above 8 bits divided by 2^(bits - 8)),
// or the scales buffer's.
float sample(bool reference, int x, int y)
{
    const int w = int(pc[0]);
    if (pc[2] != 0)
        return scales[int(reference ? pc[7] : pc[8]) + y * w + x];
    const int stride = int(pc[3]);
    const uint first = reference ? pc[4] : pc[5];
#if BPC16
    const uint word = reference ? refPlane[first + uint(y * stride + (x >> 1))]
                                : disPlane[first + uint(y * stride + (x >> 1))];
    const float value = float((word >> ((x & 1) * 16)) & 0xFFFFu);
    return value / float(1u << (pc[6] - 8u));
#else
    const uint word = reference ? refPlane[first + uint(y * stride + (x >> 2))]
                                : disPlane[first + uint(y * stride + (x >> 2))];
    return float((word >> ((x & 3) * 8)) & 0xFFu);
#endif
}

// A sum of float products kept as the rounded sum and the error of its
// additions (Knuth's two-sum), added back when it is read.
struct Sum {
    float sum, error;
    [mutating] void add(float term)
    {
        const float total = sum + term;
        const float back = total - sum;
        error += (sum - (total - back)) + (term - back);
        sum = total;
    }
    float value() { return sum + error; }
};

#if STAGE == 1

// g_gaussian_window_h (= _v)
static const float kGaussian[11] = { 0.001028, 0.007599, 0.036001, 0.109361, 0.213006, 0.266012,
                                     0.213006, 0.109361, 0.036001, 0.007599, 0.001028 };

#define TILE 16
#define SPAN (TILE + 10)  // the input rows and columns a tile's outputs read

groupshared float gRef[SPAN * SPAN], gDis[SPAN * SPAN];
// Across: the tile's columns of the five maps for all SPAN rows.
groupshared float gAcross[5][SPAN * TILE];
// The tile's l, c and s sums: of the low parts (0-2), then of the high parts
// (3-5, two's complement), each exact in 32 bits (below).
groupshared uint gSums[6];

// A place's l, c or s as a whole number of 2^-32 (exact for |v| >= 2^-8),
// split into its high part (whole 2^16 units, floor) and low 16 bits, so a
// tile's 256 places add up exactly in 32 bits: low sums < 2^24, high sums
// within +-2^26 even for |s| up to 4 (float error can take s past 1). Every
// step is exact: t is a whole number below 2^34 in a float, so is h, and
// t - 65536 h is a whole number below 2^16. The tile's total, high * 65536
// + low, is the sum of the places' 64-bit units in any order.
void units(float v, out int high, out uint low)
{
    const float t = round(v * 4294967296.0);
    const float h = floor(t * (1.0 / 65536.0));
    high = int(h);
    low = uint(t - h * 65536.0);
}

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 group : SV_GroupID, uint3 inGroup : SV_GroupThreadID, uint local : SV_GroupIndex)
{
    const int w = int(pc[0]), h = int(pc[1]);
    const int x0 = int(group.x) * TILE, y0 = int(group.y) * TILE;
    // The tile's inputs (past the picture's edge: its last sample, read only
    // by outputs that are not counted).
    for (uint i = local; i < SPAN * SPAN; i += TILE * TILE) {
        const int x = min(x0 + int(i % SPAN), w - 1), y = min(y0 + int(i / SPAN), h - 1);
        gRef[i] = sample(true, x, y);
        gDis[i] = sample(false, x, y);
    }
    if (local < 6)
        gSums[local] = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint i = local; i < SPAN * TILE; i += TILE * TILE) {
        const uint row = i / TILE, column = i % TILE;
        Sum muRef = {}, muDis = {}, refSq = {}, disSq = {}, both = {};
        for (uint u = 0; u < 11; ++u) {
            const float r = gRef[row * SPAN + column + u], d = gDis[row * SPAN + column + u];
            const float g = kGaussian[u];
            muRef.add(r * g);
            muDis.add(d * g);
            refSq.add((r * r) * g);
            disSq.add((d * d) * g);
            both.add((r * d) * g);
        }
        gAcross[0][i] = muRef.value();
        gAcross[1][i] = muDis.value();
        gAcross[2][i] = refSq.value();
        gAcross[3][i] = disSq.value();
        gAcross[4][i] = both.value();
    }
    GroupMemoryBarrierWithGroupSync();
    float maps[5];
    for (uint m = 0; m < 5; ++m) {
        Sum down = {};
        for (uint v = 0; v < 11; ++v)
            down.add(gAcross[m][(inGroup.y + v) * TILE + inGroup.x] * kGaussian[v]);
        maps[m] = down.value();
    }
    int high[3] = { 0, 0, 0 };
    uint low[3] = { 0, 0, 0 };
    if (x0 + int(inGroup.x) < w - 10 && y0 + int(inGroup.y) < h - 10) {
        const float C1 = asfloat(pc[10]), C2 = asfloat(pc[11]), C3 = asfloat(pc[12]);
        const float muRef = maps[0], muDis = maps[1];
        // ssim_maps' sigmas, then _iqa_ssim_terms' l, c and s
        const float sigmaRef = max(0.0, maps[2] - muRef * muRef);
        const float sigmaDis = max(0.0, maps[3] - muDis * muDis);
        const float sigmaBoth = maps[4] - muRef * muDis;
        const float rootProduct = sqrt(sigmaRef * sigmaDis);
        const float l = (2.0 * muRef * muDis + C1) / (muRef * muRef + muDis * muDis + C1);
        const float c = (2.0 * rootProduct + C2) / (sigmaRef + sigmaDis + C2);
        const float clamped = (sigmaBoth < 0.0 && rootProduct <= 0.0) ? 0.0 : sigmaBoth;
        const float s = (clamped + C3) / (rootProduct + C3);
        units(l, high[0], low[0]);
        units(c, high[1], low[1]);
        units(s, high[2], low[2]);
    }
    // 32-bit shared-memory adds: no 64-bit arithmetic, which Intel's GPUs
    // emulate (4K 10-bit luma on a Core Ultra 9 285K: 33.0 -> 46.3 pairs a
    // second), and no subgroup operations (any GPU, MoltenVK included).
    for (uint t = 0; t < 3; ++t) {
        InterlockedAdd(gSums[t], low[t]);
        InterlockedAdd(gSums[3 + t], uint(high[t]));
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0)
        for (uint t = 0; t < 3; ++t) {
            const int64_t total = int64_t(int(gSums[3 + t])) * 65536 + int64_t(gSums[t]);
            if (total != 0)
                acc_add64(acc, pc[9] + t, uint64_t(total));
        }
}

#else  // STAGE 2: the next scale

// g_lpf: the 9/7 biorthogonal wavelet's low-pass filter, rows then columns
static const float kLowPass[81] = {
     0.000714, -0.000450, -0.002090,  0.007132,  0.016114,  0.007132, -0.002090, -0.000450,  0.000714,
    -0.000450,  0.000283,  0.001316, -0.004490, -0.010146, -0.004490,  0.001316,  0.000283, -0.000450,
    -0.002090,  0.001316,  0.006115, -0.020867, -0.047149, -0.020867,  0.006115,  0.001316, -0.002090,
     0.007132, -0.004490, -0.020867,  0.071207,  0.160885,  0.071207, -0.020867, -0.004490,  0.007132,
     0.016114, -0.010146, -0.047149,  0.160885,  0.363505,  0.160885, -0.047149, -0.010146,  0.016114,
     0.007132, -0.004490, -0.020867,  0.071207,  0.160885,  0.071207, -0.020867, -0.004490,  0.007132,
    -0.002090,  0.001316,  0.006115, -0.020867, -0.047149, -0.020867,  0.006115,  0.001316, -0.002090,
    -0.000450,  0.000283,  0.001316, -0.004490, -0.010146, -0.004490,  0.001316,  0.000283, -0.000450,
     0.000714, -0.000450, -0.002090,  0.007132,  0.016114,  0.007132, -0.002090, -0.000450,  0.000714 };

#define TILE 16
#define SPAN (2 * TILE + 7)  // a tile's outputs read inputs 2 x0 - 4 to 2 (x0 + 15) + 4

groupshared float gRef[SPAN * SPAN], gDis[SPAN * SPAN];

// A tap's column or row c of a side n: mirrored at the edges (KBND_SYMMETRIC),
// then kept inside, for the places past the edge only uncounted outputs read.
int mirror(int c, int n)
{
    c = c < 0 ? -1 - c : (c >= n ? 2 * n - 1 - c : c);
    return clamp(c, 0, n - 1);
}

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 group : SV_GroupID, uint3 inGroup : SV_GroupThreadID, uint local : SV_GroupIndex)
{
    const int w = int(pc[0]), h = int(pc[1]);
    const int nw = w / 2 + (w & 1), nh = h / 2 + (h & 1);
    const int x0 = int(group.x) * TILE, y0 = int(group.y) * TILE;
    for (uint i = local; i < SPAN * SPAN; i += TILE * TILE) {
        const int x = mirror(2 * x0 - 4 + int(i % SPAN), w), y = mirror(2 * y0 - 4 + int(i / SPAN), h);
        gRef[i] = sample(true, x, y);
        gDis[i] = sample(false, x, y);
    }
    GroupMemoryBarrierWithGroupSync();
    const int x = x0 + int(inGroup.x), y = y0 + int(inGroup.y);
    if (x >= nw || y >= nh)
        return;
    Sum ref = {}, dis = {};
    for (uint v = 0; v < 9; ++v) {
        for (uint u = 0; u < 9; ++u) {
            const uint at = (2 * inGroup.y + v) * SPAN + 2 * inGroup.x + u;
            ref.add(gRef[at] * kLowPass[v * 9 + u]);
            dis.add(gDis[at] * kLowPass[v * 9 + u]);
        }
    }
    scales[int(pc[9]) + y * nw + x] = ref.value();
    scales[int(pc[10]) + y * nw + x] = dis.value();
}

#endif
