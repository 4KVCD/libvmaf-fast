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
// The parameters as specialization constants (constant k: Params' word k):
// the engine compiles the shader for its passes' (a pipeline for each set),
// and the compiler folds what they decide.
[vk::constant_id(0)] const int kSpec_w = 0;
#define P_w kSpec_w
[vk::constant_id(1)] const int kSpec_h = 0;
#define P_h kSpec_h
[vk::constant_id(2)] const int kSpec_strideWords = 0;
#define P_strideWords kSpec_strideWords
[vk::constant_id(3)] const int kSpec_shiftY = 0;
#define P_shiftY kSpec_shiftY
[vk::constant_id(4)] const int kSpec_addY = 0;
#define P_addY kSpec_addY
[vk::constant_id(5)] const uint kSpec_slot = 0;
#define P_slot kSpec_slot
[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> blur;
[[vk::binding(2, 0)]] StructuredBuffer<uint> previous;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;

static const uint kFilter[5] = { 3571, 16004, 26386, 16004, 3571 };

// The group's tile: TW x TH pixels, two a thread. Timed at 4K on the Core
// Ultra 9 285K's iGPU: 64 x 8 1.33 ms, against 1.37 for 32 x 16, 1.34 for
// 96 x 8 and 64 x 12, 1.40 for 64 x 16 and 128 x 8, 1.44 for 32 x 32 and
// 64 x 4, 1.51 for 128 x 4, 1.56 for 16 x 32 and 128 x 2.
#define TW 64
#define TH 8
#define REACH 2           // the filter's reach either side
#define IN_W (TW + 2 * REACH)
#define IN_H (TH + 2 * REACH)

// The tile's samples and the REACH around them, mirrored at the picture's
// edges as the kernel mirrors them, two a word (16-bit: even columns in the
// low half, odd in the high; IN_W is even). Group memory's bandwidth bounded
// the pass with a sample or a sum a word.
groupshared uint gIn[IN_W / 2 * IN_H];
// The columns' vertical sums, rounded, two a word: TH rows of IN_W / 2. They
// are below 2^16 for samples of the picture's depth (5 taps, weights adding
// up to 65536, shifted by the depth: at most (65536 (2^depth - 1) +
// 2^(depth - 1)) >> depth, below 2^16). A larger sample (a 10-bit picture's
// word holds 16 bits) can make one 2^16 or more, which the kernel keeps
// (32-bit): a tile with such a sample sets gWide, and its horizontal pass
// makes its columns' sums again from gIn, in 32 bits.
groupshared uint gVertical[TH * IN_W / 2];
groupshared uint gWide;
// Each thread's distance, for the group's sum.
groupshared uint gDistances[TW / 2 * TH];

// Column `column`'s vertical sum (the kernel's blurred_y, rounded) at tile
// row `row`, from gIn: the wide tiles' way.
uint vertical_at(uint row, uint column)
{
    uint sum = 0;
    [loop]
    for (uint yf = 0; yf < 5; ++yf)
        sum += kFilter[yf] * ((gIn[(row + yf) * uint(IN_W / 2) + column / 2] >> ((column & 1u) * 16)) & 0xFFFFu);
    return (sum + uint(P_addY)) >> uint(P_shiftY);
}

int mirror(int idx, int sup)
{
    int o = abs(idx);
    return (o < sup) ? o : (sup - (o - sup + 2));
}

// The sample at (x0 - REACH + column, y), mirrored into the picture.
uint sample_at(int x0, int y, int column)
{
    const int x = clamp(mirror(x0 - REACH + column, P_w), 0, P_w - 1);
    return pixel(src, x, y, P_strideWords);
}

[shader("compute")]
[numthreads(TW / 2, TH, 1)]
void main(uint3 group : SV_GroupID, uint3 lid : SV_GroupThreadID, uint local : SV_GroupIndex)
{
    // Each sample of the tile read once, not 25 times: the kernel's sums of
    // the same samples, in the same order.
    const int x0 = int(group.x) * TW, y0 = int(group.y) * TH;
    if (local == 0)
        gWide = 0;  // before the first barrier below; set after it
#if BPC16
    // The bits above the depth, in both samples of a word.
    const uint above = ((0xFFFFu << uint(P_shiftY)) & 0xFFFFu) * 0x10001u;
    uint seen = 0;  // this thread's samples' bits above the depth
    // A tile inside the picture, its samples and those around it (columns
    // from x0 - REACH: even, a word's first sample): the picture's words as
    // they are, nothing mirrored.
    if (x0 >= REACH && x0 + TW + REACH <= P_w && y0 >= REACH && y0 + TH + REACH <= P_h) {
        for (uint e = local; e < uint(IN_W / 2 * IN_H); e += uint(TW / 2 * TH)) {
            const uint row = e / uint(IN_W / 2), pair = e % uint(IN_W / 2);
            const uint word = src[(y0 - REACH + int(row)) * P_strideWords + (x0 - REACH) / 2 + int(pair)];
            gIn[e] = word;
            seen |= word;
        }
    } else
#endif
    for (uint e = local; e < uint(IN_W / 2 * IN_H); e += uint(TW / 2 * TH)) {
        // Positions only pixels beyond the picture read are clamped into
        // it, and unused.
        const int row = int(e / uint(IN_W / 2)), pair = int(e % uint(IN_W / 2));
        const int y = clamp(mirror(y0 - REACH + row, P_h), 0, P_h - 1);
        const uint word = sample_at(x0, y, 2 * pair) | (sample_at(x0, y, 2 * pair + 1) << 16);
        gIn[e] = word;
#if BPC16
        seen |= word;
#endif
    }
    GroupMemoryBarrierWithGroupSync();
#if BPC16
    if ((seen & above) != 0)
        gWide = 1;
#endif
    // Two columns' vertical sums a word.
    for (uint e = local; e < uint(TH * IN_W / 2); e += uint(TW / 2 * TH)) {
        const uint row = e / uint(IN_W / 2), pair = e % uint(IN_W / 2);
        uint low = 0, high = 0;
        for (uint yf = 0; yf < 5; ++yf) {
            const uint word = gIn[(row + yf) * uint(IN_W / 2) + pair];
            low += kFilter[yf] * (word & 0xFFFFu);
            high += kFilter[yf] * (word >> 16);
        }
        gVertical[e] = ((low + uint(P_addY)) >> uint(P_shiftY)) | (((high + uint(P_addY)) >> uint(P_shiftY)) << 16);
    }
    GroupMemoryBarrierWithGroupSync();

    // The blur is 16-bit (each sum below 2^32, so blurred < 2^16): two
    // pixels a word, rows of (w + 1) / 2 words.
    const int x = x0 + 2 * int(lid.x), y = y0 + int(lid.y);
    uint distance = 0;
    if (x < P_w && y < P_h) {
        // Columns 2 lid.x .. 2 lid.x + 5 of the vertical sums: three words.
        const uint word = lid.y * uint(IN_W / 2) + lid.x;
        const uint w0 = gVertical[word], w1 = gVertical[word + 1], w2 = gVertical[word + 2];
        uint columns[6] = { w0 & 0xFFFFu, w0 >> 16, w1 & 0xFFFFu, w1 >> 16, w2 & 0xFFFFu, w2 >> 16 };
        [branch]
        if (gWide != 0) {
            [loop]
            for (uint c = 0; c < 6; ++c)
                columns[c] = vertical_at(lid.y, 2 * lid.x + c);
        }
        uint blurred[2];
        for (uint i = 0; i < 2; ++i) {
            uint sum = 0;
            for (uint xf = 0; xf < 5; ++xf)
                sum += kFilter[xf] * columns[i + xf];
            blurred[i] = (sum + 32768u) >> 16;
        }
        const bool second = x + 1 < P_w;
        if (!second)
            blurred[1] = 0;
        const int at = y * ((P_w + 1) / 2) + x / 2;
        const uint before = previous[at];
        blur[at] = blurred[0] | (blurred[1] << 16);
        distance = uint(abs(int(blurred[0]) - int(before & 0xFFFFu)));
        if (second)
            distance += uint(abs(int(blurred[1]) - int(before >> 16)));
    }
    gDistances[local] = distance;
    GroupMemoryBarrierWithGroupSync();
    if (local < uint(TW / 2 * TH / 16)) {
        uint sum = 0;
        for (uint i = 0; i < 16; ++i)
            sum += gDistances[local * 16 + i];
        gDistances[local * 16] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        uint sum = 0;
        for (uint i = 0; i < uint(TW / 2 * TH / 16); ++i)
            sum += gDistances[i * 16];
        if (sum != 0)
            acc_add64(acc, P_slot, uint64_t(sum));
    }
}
