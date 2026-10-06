// Derived from libvmaf (https://github.com/Netflix/vmaf): integer_motion.c,
// Copyright 2016-2026 Netflix, Inc., licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root
// of this repository.
//
// Motion for VMAF v1: motion_score_pipeline_8 / _16 of libvmaf's
// integer_motion.c (the CPU code: VMAF v1's motion has no CUDA version). The
// difference of two reference frames is blurred with the 5-tap filter, each
// column rounded before the horizontal pass, and its absolute values summed.
#include "common.slang"

struct Params {
    int w, h, strideWords;
    int shiftY, addY;   // the picture's bit depth, and half of 1 << depth
    uint slot;
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> current;
[[vk::binding(1, 0)]] StructuredBuffer<uint> previous;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> acc;

static const int kFilter[5] = { 3571, 16004, 26386, 16004, 3571 };

groupshared uint gSum[256];

int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 2;
    return idx;
}

#define TILE 16
#define SPAN (TILE + 4)
groupshared int gDiff[SPAN * SPAN];     // previous - current around the tile (mirrored at the edges)
groupshared int gColumn[TILE * SPAN];   // the vertical pass, rounded, per output row and column

// A workgroup takes 16 x 16 pixels: the differences it reads are loaded once
// into shared memory, the vertical pass done there for its rows, and the
// horizontal pass per pixel -- the same products, sums and rounding as
// motion_score_pipeline's, per pixel, at the same mirrored places.
[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 group : SV_GroupID, uint3 inGroup : SV_GroupThreadID, uint local : SV_GroupIndex)
{
    const int x0 = int(group.x) * TILE, y0 = int(group.y) * TILE;
    for (uint i = local; i < SPAN * SPAN; i += TILE * TILE) {
        const int column = mirror(x0 - 2 + int(i % SPAN), pc.w), row = mirror(y0 - 2 + int(i / SPAN), pc.h);
        gDiff[i] = int(pixel(previous, column, row, pc.strideWords)) - int(pixel(current, column, row, pc.strideWords));
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = local; i < TILE * SPAN; i += TILE * TILE) {
        const uint lx = i % SPAN, ly = i / SPAN;
        int64_t vertical = 0;
        for (uint k = 0; k < 5; ++k)
            vertical += int64_t(kFilter[k] * gDiff[(ly + k) * SPAN + lx]);
        gColumn[i] = int((vertical + pc.addY) >> pc.shiftY);
    }
    GroupMemoryBarrierWithGroupSync();
    const int x = x0 + int(inGroup.x), y = y0 + int(inGroup.y);
    uint distance = 0;
    if (x < pc.w && y < pc.h) {
        int64_t blurred = 0;
        for (uint xf = 0; xf < 5; ++xf)
            blurred += int64_t(kFilter[xf]) * gColumn[inGroup.y * SPAN + inGroup.x + xf];
        distance = uint(abs(int((blurred + 32768) >> 16)));
    }
    gSum[local] = distance;
    GroupMemoryBarrierWithGroupSync();
    if (local < 16) {
        uint sum = 0;
        for (uint i = 0; i < 16; ++i)
            sum += gSum[local * 16 + i];
        gSum[local * 16] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        uint sum = 0;
        for (uint i = 0; i < 16; ++i)
            sum += gSum[i * 16];
        if (sum != 0)
            acc_add64(acc, pc.slot, uint64_t(sum));
    }
}
