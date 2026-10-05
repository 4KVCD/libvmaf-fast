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
// The vertical pass's rounded values for the group's 16 rows and its 16
// columns with two either side: each worked out once, by one thread, where
// every pixel worked out the five it needs (each column five times over).
groupshared int gColumns[16][20];

int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 2;
    return idx;
}

// The column's 5-tap blur of the frames' difference at (column, y), rounded
// as the CPU code rounds it. In 32 bits up to 15-bit samples (65536 times a
// difference below 2^15); 16-bit ones need 64.
int vertical(int column, int y)
{
    int sum32 = 0;
    int64_t sum64 = 0;
    for (int k = 0; k < 5; ++k) {
        int row = clamp(mirror(y - 2 + k, pc.h), 0, pc.h - 1);
        int difference = int(pixel(previous, column, row, pc.strideWords))
                       - int(pixel(current, column, row, pc.strideWords));
        if (pc.shiftY < 16)
            sum32 += kFilter[k] * difference;
        else
            sum64 += int64_t(kFilter[k]) * difference;
    }
    if (pc.shiftY < 16)
        return (sum32 + pc.addY) >> pc.shiftY;
    return int((sum64 + pc.addY) >> pc.shiftY);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID, uint local : SV_GroupIndex)
{
    int x = int(id.x), y = int(id.y);
    int first = x - int(lid.x) - 2;  // the tile's first column in the picture
    for (int c = int(lid.x); c < 20; c += 16) {
        int column = clamp(mirror(first + c, pc.w), 0, pc.w - 1);
        gColumns[lid.y][c] = vertical(column, y);
    }
    GroupMemoryBarrierWithGroupSync();
    uint distance = 0;
    if (x < pc.w && y < pc.h) {
        int64_t blurred = 0;
        // Tap xf is the tile's column lid.x + xf: first + lid.x + xf is
        // x - 2 + xf, mirrored there as the CPU code mirrors it.
        for (int xf = 0; xf < 5; ++xf)
            blurred += int64_t(kFilter[xf]) * gColumns[lid.y][int(lid.x) + xf];
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
