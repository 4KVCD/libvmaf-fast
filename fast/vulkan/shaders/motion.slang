// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// Motion: calculate_motion_score_kernel_8bpc / _16bpc of motion_score.cu.
// Blurs the reference with the 5-tap filter (each column rounded before the
// horizontal pass, as the CUDA kernel does), keeps the blur for the next
// frame and sums its absolute difference from the previous frame's blur.
// Each group reads its tile's samples once into group memory and makes the
// columns' vertical sums there.
#include "common.slang"

struct Params {
    int w, h, strideWords;
    int shiftY, addY;   // the picture's bit depth, and half of 1 << depth
    uint slot;
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> blur;
[[vk::binding(2, 0)]] StructuredBuffer<uint> previous;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;

static const uint kFilter[5] = { 3571, 16004, 26386, 16004, 3571 };

#define TW 32             // the group's tile: TW x TH pixels, two a thread
#define TH 16
#define REACH 2           // the filter's reach either side
#define IN_W (TW + 2 * REACH)
#define IN_H (TH + 2 * REACH)

// The tile's samples and the REACH around them, mirrored at the picture's
// edges as the kernel mirrors them; then (reused) the group's sum.
groupshared uint gIn[IN_W * IN_H];
// The columns' vertical sums, rounded: TH rows of IN_W.
groupshared uint gVertical[TH * IN_W];

int mirror(int idx, int sup)
{
    int o = abs(idx);
    return (o < sup) ? o : (sup - (o - sup + 2));
}

[shader("compute")]
[numthreads(TW / 2, TH, 1)]
void main(uint3 group : SV_GroupID, uint3 lid : SV_GroupThreadID, uint local : SV_GroupIndex)
{
    // Each sample of the tile read once, not 25 times: the kernel's sums of
    // the same samples, in the same order.
    const int x0 = int(group.x) * TW, y0 = int(group.y) * TH;
    for (uint e = local; e < uint(IN_W * IN_H); e += uint(TW / 2 * TH)) {
        // Positions only pixels beyond the picture read are clamped into
        // it, and unused.
        const int y = clamp(mirror(y0 - REACH + int(e / uint(IN_W)), pc.h), 0, pc.h - 1);
        const int x = clamp(mirror(x0 - REACH + int(e % uint(IN_W)), pc.w), 0, pc.w - 1);
        gIn[e] = pixel(src, x, y, pc.strideWords);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint e = local; e < uint(TH * IN_W); e += uint(TW / 2 * TH)) {
        const uint row = e / uint(IN_W), column = e % uint(IN_W);
        uint vertical = 0;
        for (uint yf = 0; yf < 5; ++yf)
            vertical += kFilter[yf] * gIn[(row + yf) * uint(IN_W) + column];
        gVertical[e] = (vertical + uint(pc.addY)) >> uint(pc.shiftY);
    }
    GroupMemoryBarrierWithGroupSync();  // gIn is free for the sum from here

    // The blur is 16-bit (each sum below 2^32, so blurred < 2^16): two
    // pixels a word, rows of (w + 1) / 2 words.
    const int x = x0 + 2 * int(lid.x), y = y0 + int(lid.y);
    uint distance = 0;
    if (x < pc.w && y < pc.h) {
        uint blurred[2];
        for (uint i = 0; i < 2; ++i) {
            uint sum = 0;
            for (uint xf = 0; xf < 5; ++xf)
                sum += kFilter[xf] * gVertical[lid.y * uint(IN_W) + 2 * lid.x + i + xf];
            blurred[i] = (sum + 32768u) >> 16;
        }
        const bool second = x + 1 < pc.w;
        if (!second)
            blurred[1] = 0;
        const int at = y * ((pc.w + 1) / 2) + x / 2;
        const uint before = previous[at];
        blur[at] = blurred[0] | (blurred[1] << 16);
        distance = uint(abs(int(blurred[0]) - int(before & 0xFFFFu)));
        if (second)
            distance += uint(abs(int(blurred[1]) - int(before >> 16)));
    }
    gIn[local] = distance;
    GroupMemoryBarrierWithGroupSync();
    if (local < 16) {
        uint sum = 0;
        for (uint i = 0; i < 16; ++i)
            sum += gIn[local * 16 + i];
        gIn[local * 16] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        uint sum = 0;
        for (uint i = 0; i < 16; ++i)
            sum += gIn[i * 16];
        if (sum != 0)
            acc_add64(acc, pc.slot, uint64_t(sum));
    }
}
