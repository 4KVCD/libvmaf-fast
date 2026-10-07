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
#if FROM_IMAGE
    uint unused[20];
    int previousX, previousY, currentX, currentY;  // (common.slang's PICTURE_PREVIOUS, _REFERENCE)
#endif
};
[[vk::push_constant]] Params pc;
#if FROM_IMAGE
[[vk::binding(0, 0)]] Texture2D<uint> current;
[[vk::binding(1, 0)]] Texture2D<uint> previous;
#else
[[vk::binding(0, 0)]] StructuredBuffer<uint> current;
[[vk::binding(1, 0)]] StructuredBuffer<uint> previous;
#endif
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> acc;

static const int kFilter[5] = { 3571, 16004, 26386, 16004, 3571 };

int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 2;
    return idx;
}

#define TILE_W 64
#define TILE_H 16
#define SPAN_W (TILE_W + 4)
#define SPAN_H (TILE_H + 4)
#define GROUP 256
groupshared int gDiff[SPAN_H * SPAN_W];     // previous - current around the tile (mirrored at the edges)
groupshared int gColumn[TILE_H * SPAN_W];   // the vertical pass, rounded, per output row and column
groupshared uint gSum[GROUP / 32];

// One output of the horizontal pass: (sum of filter * column + 32768) >> 16.
// The sum's magnitude reaches 2^32: in 32-bit integers, it is split into the
// columns' high bits (>> 8) and low bytes, which sum exactly, and put
// together in the shift. (16-bit pictures: in 64 bits, as on the CPU.)
int horizontal(uint base)
{
    if (pc.shiftY > 15) {
        int64_t blurred = 0;
        [unroll] for (uint xf = 0; xf < 5; ++xf)
            blurred += int64_t(kFilter[xf]) * gColumn[base + xf];
        return int((blurred + 32768) >> 16);
    }
    int high = 0, low = 0;
    [unroll] for (uint xf = 0; xf < 5; ++xf) {
        const int c = gColumn[base + xf];
        high += kFilter[xf] * (c >> 8);
        low += kFilter[xf] * (c & 255);
    }
    return (high >> 8) + ((((high & 255) << 8) + low + 32768) >> 16);
}

// A workgroup takes 64 x 16 pixels: the differences it reads are loaded once
// into shared memory (a word, 2 or 4 samples, at a time away from the
// picture's edges), the vertical pass done there for its rows, and the
// horizontal pass per pixel -- the same products, sums and rounding as
// motion_score_pipeline's, per pixel, at the same mirrored places. The
// vertical sums fit 32 bits up to 15-bit pictures (|sum| <= 65536 * 32767).
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const int x0 = int(group.x) * TILE_W, y0 = int(group.y) * TILE_H;
#if FROM_IMAGE
    // The differences a sample at a time from the two textures (mirrored at
    // the picture's edges).
    for (uint i = local; i < SPAN_H * SPAN_W; i += GROUP) {
        const int column = mirror(x0 - 2 + int(i % SPAN_W), pc.w), row = mirror(y0 - 2 + int(i / SPAN_W), pc.h);
        gDiff[i] = column >= 0 && column < pc.w && row >= 0 && row < pc.h  // (else: past the tile's last outputs, unused)
            ? int(image_pixel(previous, column, row, pc.previousX, pc.previousY))
                - int(image_pixel(current, column, row, pc.currentX, pc.currentY)) : 0;
    }
#else
    const bool inside = x0 >= 4 && x0 + TILE_W + 4 <= pc.w;
    if (inside) {
#if BPC16
        // 2 samples a word, from x0 - 2 (even): SPAN_W / 2 words a row.
        for (uint i = local; i < SPAN_H * (SPAN_W / 2); i += GROUP) {
            const uint lw = i % (SPAN_W / 2), ly = i / (SPAN_W / 2);
            const int row = mirror(y0 - 2 + int(ly), pc.h);
            const uint at = uint(row * pc.strideWords + ((x0 - 2) >> 1)) + lw;
            const uint p = previous[at], c = current[at];
            gDiff[ly * SPAN_W + lw * 2] = int(p & 0xFFFFu) - int(c & 0xFFFFu);
            gDiff[ly * SPAN_W + lw * 2 + 1] = int(p >> 16) - int(c >> 16);
        }
#else
        // 4 samples a word, from x0 - 4: words covering x0 - 4 .. x0 + 67.
        #define WORDS_8 ((SPAN_W + 2 + 3) / 4)
        for (uint i = local; i < SPAN_H * WORDS_8; i += GROUP) {
            const uint lw = i % WORDS_8, ly = i / WORDS_8;
            const int row = mirror(y0 - 2 + int(ly), pc.h);
            const uint at = uint(row * pc.strideWords + ((x0 - 4) >> 2)) + lw;
            const uint p = previous[at], c = current[at];
            [unroll] for (uint b = 0; b < 4; ++b) {
                const int lx = int(lw * 4 + b) - 2;  // the column less x0 - 2
                if (lx >= 0 && lx < SPAN_W)
                    gDiff[ly * SPAN_W + uint(lx)] = int((p >> (b * 8)) & 0xFFu) - int((c >> (b * 8)) & 0xFFu);
            }
        }
#endif
    } else {
        for (uint i = local; i < SPAN_H * SPAN_W; i += GROUP) {
            const int column = mirror(x0 - 2 + int(i % SPAN_W), pc.w), row = mirror(y0 - 2 + int(i / SPAN_W), pc.h);
            gDiff[i] = column >= 0 && column < pc.w && row >= 0 && row < pc.h  // (else: past the tile's last outputs, unused)
                ? int(pixel(previous, column, row, pc.strideWords)) - int(pixel(current, column, row, pc.strideWords)) : 0;
        }
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    for (uint i = local; i < TILE_H * SPAN_W; i += GROUP) {
        const uint lx = i % SPAN_W, ly = i / SPAN_W;
        if (pc.shiftY > 15) {
            int64_t vertical = 0;
            [unroll] for (uint k = 0; k < 5; ++k)
                vertical += int64_t(kFilter[k] * gDiff[(ly + k) * SPAN_W + lx]);
            gColumn[i] = int((vertical + pc.addY) >> pc.shiftY);
        } else {
            int vertical = 0;
            [unroll] for (uint k = 0; k < 5; ++k)
                vertical += kFilter[k] * gDiff[(ly + k) * SPAN_W + lx];
            gColumn[i] = (vertical + pc.addY) >> pc.shiftY;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Each thread: column local % 64, rows local / 64 + 4k.
    const uint tx = local % TILE_W, ty = local / TILE_W;
    uint distance = 0;
    [unroll] for (uint k = 0; k < TILE_H / (GROUP / TILE_W); ++k) {
        const uint ly = ty + k * (GROUP / TILE_W);
        const int x = x0 + int(tx), y = y0 + int(ly);
        if (x < pc.w && y < pc.h)
            distance += uint(abs(horizontal(ly * SPAN_W + tx)));
    }
    const uint total = WaveActiveSum(distance);  // < 2^27
    const uint wave = local / WaveGetLaneCount();
    if (WaveIsFirstLane()) gSum[wave] = total;
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        uint sum = 0;
        for (uint i = 0; i < GROUP / WaveGetLaneCount(); ++i)
            sum += gSum[i];
        if (sum != 0)
            acc_add64(acc, pc.slot, uint64_t(sum));
    }
}
