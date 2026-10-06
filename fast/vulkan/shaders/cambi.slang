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
//   MODE      filter_mode: the mode of 3, across then down; the first and
//             last rows as they were -- of the scale before's filtered
//             picture decimated (every other row and column, as is its
//             mask, which it writes too)
//   CVALUES   calculate_c_values: each masked pixel's histogram of its
//             (window x window) square counted directly -- what the CPU's
//             sliding histograms hold for it -- and c_value_pixel, with the
//             CPU's reciprocal table (its entries as written, not 1.0f/i)
//   CLEAR     zeros
//   HIST      a level of the radix select of spatial_pooling's top k: the
//             c-values' bits are ordered as the floats are (all >= 0); the
//             five scales in one dispatch (as SELECT and SUM)
//   SELECT    the bin the k-th largest value is in, at that level
//   SUM       the values above the k-th largest, exactly: each nonzero
//             c-value is at least 0.5, a multiple of 2^-24 -- an integer in
//             units of 2^-24 -- so their sum is exact in 64 bits, as the
//             CPU's double sum is while it is under 2^53 units (v1_host)
#include "common.slang"

[[vk::push_constant]] uniform uint pc[32];  // (FROM_IMAGE: the pictures' places in the last words, common.slang)

// The radix select's state (HIST, SELECT, SUM), in words: per scale the
// prefix and what is left of k (2 * scale); the zero c-values (10 +
// scale); level 0's sum of the nonzero ones, in
// units of 2^-24 (a 64-bit sum, slot 8 + scale of acc_add64) and whether one
// was out of its range (26 + scale); and whether the select is done (31 +
// scale): when level 0 finds the k-th largest among the zeros -- most
// c-values are 0 at high resolutions -- it is 0, the sum above it level 0's,
// and the other levels and SUM have nothing left to do.
#define STATE_SUM 8         // acc_add64's slot
#define STATE_INEXACT 26
#define STATE_DONE 31

// filter_mode's mode of 3 (the smallest when all differ).
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

#elif STAGE == 14  // FRONT
// PRE, DERIV, MASK and scale 0's MODE in one pass, and DECIMATE's choice of pixels (STEP 2:
// every other row and column, for cambi_high_res_speedup; 1: all): a
// workgroup takes 16 x 16 of the outputs, their 16 STEP x 16 STEP pixels of
// the picture, and works in shared memory on those and the 3 + 1 + 1 pixels
// around them that the mask's window, the derivative and the anti-dithering
// read. Each picture pixel of the tile is checked against the bit depth once.
#if FROM_IMAGE
[[vk::binding(0, 0)]] Texture2D<uint> picture;  // the distorted picture
#else
[[vk::binding(0, 0)]] StructuredBuffer<uint> picture;
#endif
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> imageOut;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> maskOut;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;
// pc: w, h, strideWords, bpc, antiDither, errorSlot, maskIndex, step, outW, outH
#define TILE 16
#define MAX_SPAN (TILE * 2 + 8)  // the pixels a tile reads across, at STEP 2
#if BPC16
#define PER_WORD 2
#else
#define PER_WORD 4
#endif
#define ROW_WORDS ((TILE * STEP + 8 + 1 + PER_WORD - 1) / PER_WORD)  // a row's words, from the sample before its first
#define LOADS ((ROW_WORDS * (TILE * STEP + 8) + 255) / 256)  // each thread's words

groupshared uint gTen[MAX_SPAN * MAX_SPAN];   // to10 of the picture; outside it, unused
groupshared uint gZeroWords[MAX_SPAN * 2];  // DERIV's flags, a bit per pixel, two words a row (0 outside the picture)

// PRE (anti_dithering_filter, if pc[4]) at local (lx, ly), picture (x, y):
// from its sample v and those right (r), below (d) and diagonally (rd).
// (Each case's value calculated and one chosen: no branches but the dispatch's.)
uint pre_of(uint v, uint r, uint d, uint rd, int lx, int ly, int x, int y)
{
    if (pc[4] == 0) return v;  // (no anti-dithering: the same for the whole dispatch)
    const int w = int(pc[0]), h = int(pc[1]), span = TILE * STEP + 8;
    const bool on = lx + 1 < span && ly + 1 < span && uint(x) < uint(w) && uint(y) < uint(h);
    const bool right = x < w - 1, down = y < h - 1;
    const uint both = (v + r + d + rd) >> 2, onlyDown = (v + d) >> 1, onlyRight = (v + r) >> 1;
    const uint filtered = right ? (down ? both : onlyRight) : (down ? onlyDown : v);
    return on ? filtered : v;
}

uint to10_of(uint v)
{
    const uint bpc = pc[3];
    if (bpc <= 8) return v << (10 - bpc);
    if (bpc == 9) return v << 1;
    if (bpc == 10) return v;
    const uint shift = bpc - 10;
    return (v + (1u << (shift - 1))) >> shift;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const int w = int(pc[0]), h = int(pc[1]), step = STEP;
    const int core = TILE * step, span = core + 8;
    // The tile's picture pixels start at (u0, v0); shared memory at (u0 - 3, v0 - 3).
    const int u0 = int(group.x) * core, v0 = int(group.y) * core;
    const int ox = u0 - 3, oy = v0 - 3;
    const uint bpc = pc[3];
    // The samples a word (2 or 4) at a time, from ox - 1 (on a word: u0 is a
    // multiple of 16): every load first, then their conversion, so the loads
    // wait for memory together (STEP is fixed, so the loops unroll).
    if (local < MAX_SPAN * 2) gZeroWords[local] = 0;
#if FROM_IMAGE
    // A sample at a time from the texture.
    for (int i = int(local); i < span * span; i += 256) {
        const int lx = i % span, ly = i / span, x = ox + lx, y = oy + ly;
        uint ten = 0;
        if (x >= 0 && x < w && y >= 0 && y < h) {
            const uint raw = image_pixel(picture, x, y, int(pc[PICTURE_DISTORTED]), int(pc[PICTURE_DISTORTED + 1]));
            if (bpc > 8 && bpc < 16 && x >= u0 && x < u0 + core && y >= v0 && y < v0 + core && raw > (1u << bpc) - 1u)
                InterlockedOr(acc[2 * pc[5]], 1u);
            ten = to10_of(raw);
        }
        gTen[ly * span + lx] = ten;
    }
#else
    const int firstWord = (ox - 1) / PER_WORD;  // (negative at the left edge: those words are not read)
    uint raws[LOADS];
    [unroll] for (int k = 0; k < LOADS; ++k) {
        const int i = int(local) + k * 256, lw = i % ROW_WORDS, ly = i / ROW_WORDS;
        const int x = ox - 1 + lw * PER_WORD, y = oy + ly;  // the word's first sample
        raws[k] = ly < span && x + PER_WORD > 0 && x < w && y >= 0 && y < h
            ? picture[y * int(pc[2]) + firstWord + lw] : 0u;
    }
    [unroll] for (int k = 0; k < LOADS; ++k) {
        const int i = int(local) + k * 256, lw = i % ROW_WORDS, ly = i / ROW_WORDS;
        if (ly >= span)
            break;
        const int y = oy + ly;
        [unroll] for (int b = 0; b < PER_WORD; ++b) {
            const int lx = lw * PER_WORD + b - 1, x = ox + lx;
            if (lx < 0 || lx >= span)
                continue;
            uint ten = 0;
            if (x >= 0 && x < w && y >= 0 && y < h) {
#if BPC16
                const uint raw = (raws[k] >> (b * 16)) & 0xFFFFu;
#else
                const uint raw = (raws[k] >> (b * 8)) & 0xFFu;
#endif
                if (bpc > 8 && bpc < 16 && x >= u0 && x < u0 + core && y >= v0 && y < v0 + core && raw > (1u << bpc) - 1u)
                    InterlockedOr(acc[2 * pc[5]], 1u);
                ten = to10_of(raw);
            }
            gTen[ly * span + lx] = ten;
        }
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    // DERIV of PRE (anti_dithering_filter of the converted picture): equal to
    // its right and lower neighbours (the last column and row: always). A
    // thread takes 8 pixels of a row, their flags a byte of the row's bits, and
    // calculates the PRE values they compare from the 3 x 10 converted
    // samples they read.
    for (int task = int(local); task < span * (span / 8); task += 256) {
        const int ly = task / (span / 8), lx0 = (task % (span / 8)) * 8, y = oy + ly;
        uint t[3][10];
        [unroll] for (int r = 0; r < 3; ++r)
            [unroll] for (int c = 0; c < 10; ++c)
                t[r][c] = lx0 + c < span && ly + r < span ? gTen[(ly + r) * span + lx0 + c] : 0u;
        uint p[2][9];
        [unroll] for (int r = 0; r < 2; ++r)
            [unroll] for (int c = 0; c < 9; ++c)
                p[r][c] = pre_of(t[r][c], t[r][c + 1], t[r + 1][c], t[r + 1][c + 1], lx0 + c, ly + r, ox + lx0 + c, y + r);
        uint bits = 0;
        [unroll] for (int k = 0; k < 8; ++k) {
            const int lx = lx0 + k, x = ox + lx;
            if (x >= 0 && x < w && y >= 0 && y < h && lx + 1 < span && ly + 1 < span) {
                const bool across = x == w - 1 || p[0][k] == p[0][k + 1];
                const bool down = y == h - 1 || p[0][k] == p[1][k];
                if (across && down) bits |= 1u << uint(k);
            }
        }
        if (bits != 0)
            InterlockedOr(gZeroWords[ly * 2 + lx0 / 32], bits << uint(lx0 % 32));
    }
    GroupMemoryBarrierWithGroupSync();
    // MASK at the outputs' pixels: more than maskIndex zero derivatives in the
    // 7 x 7 square, its rows' 7 bits counted.
    const int tx = int(local) % TILE, ty = int(local) / TILE;
    const int outX = int(group.x) * TILE + tx, outY = int(group.y) * TILE + ty;
    if (outX >= int(pc[8]) || outY >= int(pc[9])) return;
    const int lx = 3 + tx * step, ly = 3 + ty * step;  // the output's picture pixel, in shared memory
    uint sum = 0;
    [unroll] for (int r = -3; r <= 3; ++r) {
        const uint64_t row = uint64_t(gZeroWords[(ly + r) * 2]) | (uint64_t(gZeroWords[(ly + r) * 2 + 1]) << 32);
        sum += countbits(uint(row >> uint(lx - 3)) & 0x7Fu);
    }
    const uint at = uint(outY) * pc[8] + uint(outX);
    // MODE of scale 0 too: filter_mode of the outputs' PRE values, the 3 x 3
    // around it STEP pixels apart (the outputs' picture pixels), across then
    // down; the first and last rows and columns of the outputs as they are.
    uint pre[3][3];
    [unroll] for (int dy = 0; dy < 3; ++dy) {
        [unroll] for (int dx = 0; dx < 3; ++dx) {
            const int px = lx + (dx - 1) * step, py = ly + (dy - 1) * step, i = py * span + px;
            pre[dy][dx] = pre_of(gTen[i], gTen[i + 1], gTen[i + span], gTen[i + span + 1], px, py, ox + px, oy + py);
        }
    }
    const int outW = int(pc[8]), outH = int(pc[9]);
    uint across[3];
    [unroll] for (int dy = 0; dy < 3; ++dy)
        across[dy] = outX == 0 || outX == outW - 1 ? pre[dy][1] : mode3(pre[dy][0], pre[dy][1], pre[dy][2]);
    imageOut[at] = outY == 0 || outY == outH - 1 ? pre[1][1] : mode3(across[0], across[1], across[2]);
    maskOut[at] = sum > pc[6] ? 1u : 0u;
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

#elif STAGE == 5  // MODE, and DECIMATE
// A scale's image and mask are the scale before's filtered image and mask
// DECIMATEd (the pixel at twice its coordinates): read here every `step`-th
// pixel of every step-th row (step 2; 1 at scale 0, whose image and mask
// FRONT wrote), and the mode filter done in shared memory on a 16 x 16 tile
// and the pixel around it; the scale's mask is written too (step > 1).
[[vk::binding(0, 0)]] StructuredBuffer<uint> imageIn;   // the scale before's, filtered (scale 0: its own)
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> imageOut;  // the scale's, filtered
[[vk::binding(2, 0)]] StructuredBuffer<uint> maskIn;    // the scale before's (scale 0: its own)
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> maskOut;   // the scale's (step > 1)
// pc: w, h (the scale's), imageIn's width, step

// The scale's pixel (x, y): imageIn's at every step-th column and row.
uint at(int x, int y)
{
    return imageIn[(y * int(pc[3])) * int(pc[2]) + x * int(pc[3])];
}

uint across(int x, int y, int w)
{
    if (x == 0 || x == w - 1) return at(x, y);
    return mode3(at(x - 1, y), at(x, y), at(x + 1, y));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int w = int(pc[0]), h = int(pc[1]), x = int(id.x), y = int(id.y);
    if (x >= w || y >= h) return;
    const int step = int(pc[3]);
    if (step > 1)
        maskOut[y * w + x] = maskIn[(y * step) * int(pc[2]) + x * step];
    if (y == 0 || y == h - 1)
        imageOut[y * w + x] = at(x, y);
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

#elif STAGE == 13  // CVALUES_SLIDE
// CVALUES, the same counts kept as calculate_c_values keeps them: as the
// window moves down a row, the row entering it is added and the row leaving
// it taken away. A workgroup takes a tile of S columns and a band of rows;
// in shared memory it keeps, per column its windows reach, how many masked,
// in-band pixels of each value the window's rows have there (a byte: at most
// 2 * pad + 1 <= 65), and a pixel's count of a value is the sum of the bytes
// of its window's 2 * pad + 1 columns. The values are taken VALUES at a time
// (window k: the values 56k - 4 .. 56k + 59, for the pixels of value
// 56k .. 56k + 55, which read their value +- 4), only the windows the tile's
// pixels are in: one or two where the picture is flat, which is where pixels
// are masked. For pads up to 32 (CVALUES otherwise).
[[vk::binding(0, 0)]] StructuredBuffer<uint> image;
[[vk::binding(1, 0)]] StructuredBuffer<uint> mask;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> cValues;
[[vk::binding(3, 0)]] StructuredBuffer<float> reciprocal;
// pc: w, h, pad, vltLuma, bandBase, bandSize, tvi[0..3] (+ num_diffs), weights[0..3], rows per workgroup
#define NUM_DIFFS 4
#define S 64             // the tile's columns: a thread each
#ifndef MAX_PAD
#define MAX_PAD 32       // the pads (window_size / 2) this variant takes
#endif
#define COLS (S + 2 * MAX_PAD)  // the columns its windows reach: less shared memory, more workgroups at once
#define WORDS (COLS / 4)
#define VALUES 64        // a window of values
#define CORE 56          // VALUES - 2 * NUM_DIFFS
#define MAX_ROWS 128     // rows per workgroup, at most

groupshared uint gCounts[VALUES * WORDS];  // [value - the window's first][column], a byte each
groupshared uint gWindows;                 // the windows the tile's pixels are in (bits)
groupshared uint gRowWindows[MAX_ROWS];   // and each row's

// A pixel not counted (above every window's values: one compare tells).
#define NONE 0x10000

// Row r's pixel in local column lc: its value less the band's base if it is
// masked and in the band (counted), else NONE. Read ahead of its use: the
// loads are what a row's step waits for.
int pixel_compact(int r, int x0, int pad, uint lc)
{
    const int w = int(pc[0]), h = int(pc[1]);
    const int col = x0 - pad + int(lc);
    if (r < 0 || r >= h || col < 0 || col >= w || int(lc) >= S + 2 * pad) return NONE;
    const int compact = int(image[r * w + col]) - int(pc[4]);
    return mask[r * w + col] != 0 && compact >= 0 && compact < int(pc[5]) ? compact : NONE;
}

// A counted pixel of local column lc added (or taken away), if its value is
// in the window that starts at `first`.
void count_pixel(int compact, uint lc, int first, bool remove)
{
    const uint local = uint(compact - first);  // (NONE and values below the window: past VALUES too)
    if (local >= VALUES) return;
    const uint one = 1u << ((lc & 3u) * 8u);
    // (local below 64: masked so, the compiler multiplies in 24 bits)
    InterlockedAdd(gCounts[(local & 63u) * WORDS + (lc >> 2)], remove ? 0u - one : one);
}

#define MAX_WORDS ((2 * MAX_PAD + 3) / 4 + 1)  // the words a window's 2 * pad + 1 columns are in, at most

// The pixels of value first + local in the window of the tile's column t: a
// fixed number of words read (those past the window's last, inside gCounts,
// left out), so that their reads are made together.
uint window_count(int local, uint t, int pad)
{
    const uint base = (uint(local) & 63u) * WORDS + (t >> 2), last = t + uint(2 * pad);  // (local 0 .. 63: in 24 bits)
    const uint words = (last >> 2) - (t >> 2);  // the last's, from the first
    uint sum = 0;
    [unroll] for (uint k = 0; k < MAX_WORDS; ++k) {
        uint bytes = gCounts[base + k];  // (inside gCounts also past the window: no branch)
        if (k > words) bytes = 0;
        if (k == 0) bytes &= 0xFFFFFFFFu << ((t & 3u) * 8u);
        if (k == words) bytes &= 0xFFFFFFFFu >> ((3u - (last & 3u)) * 8u);
        sum += (bytes & 0x00FF00FFu) + ((bytes >> 8) & 0x00FF00FFu);
    }
    return (sum & 0xFFFFu) + (sum >> 16);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex)
{
    const int w = int(pc[0]), h = int(pc[1]), pad = int(pc[2]);
    const int bandBase = int(pc[4]), bandSize = int(pc[5]);
    const int x0 = int(group.x) * S, y0 = int(group.y * pc[14]);
    const int y1 = min(y0 + int(pc[14]), h);
    const int x = x0 + int(t);
    const int rows = y1 - y0;
    if (t == 0) gWindows = 0;
    for (int i = int(t); i < rows; i += 64) gRowWindows[i] = 0;
    GroupMemoryBarrierWithGroupSync();
    // The windows of values the tile's masked, in-band pixels are in, in all
    // and row by row; the other pixels' c-values are 0.
    uint windows = 0;
    for (int y = y0; y < y1; ++y) {
        if (x >= w) break;
        const int compact = int(image[y * w + x]) - bandBase;
        if (mask[y * w + x] != 0 && compact >= 0 && compact < bandSize) {
            const uint bit = 1u << uint(compact / CORE);
            windows |= bit;
            InterlockedOr(gRowWindows[y - y0], bit);
        } else {
            cValues[y * w + x] = 0;
        }
    }
    InterlockedOr(gWindows, windows);
    GroupMemoryBarrierWithGroupSync();
    uint todo = gWindows;
    while (todo != 0) {
        const int k = int(firstbitlow(todo));
        const uint bit = 1u << uint(k);
        todo &= todo - 1;
        const int first = k * CORE - NUM_DIFFS;
        // The window slides from the first row with pixels in it to the last.
        int ys = rows, ye = -1;
        for (int i = 0; i < rows; ++i) {
            if (gRowWindows[i] & bit) {
                ys = min(ys, i);
                ye = i;
            }
        }
        ys += y0;
        ye += y0;
        for (uint i = t; i < VALUES * WORDS; i += 64) gCounts[i] = 0;
        GroupMemoryBarrierWithGroupSync();
        // The window of row ys, eight rows' loads at a time.
        for (int r0 = ys - pad; r0 < ys + pad; r0 += 8) {
            int v[16];
            [unroll] for (int j = 0; j < 8; ++j) {
                v[j] = r0 + j < ys + pad ? pixel_compact(r0 + j, x0, pad, t) : NONE;
                v[8 + j] = r0 + j < ys + pad ? pixel_compact(r0 + j, x0, pad, t + 64) : NONE;
            }
            [unroll] for (int j = 0; j < 8; ++j) {
                count_pixel(v[j], t, first, false);
                count_pixel(v[8 + j], t + 64, first, false);
            }
        }
        // Each row's step: its entering and leaving rows' pixels, and its own
        // (the tile's column), read one step ahead.
        int add0 = pixel_compact(ys + pad, x0, pad, t), add1 = pixel_compact(ys + pad, x0, pad, t + 64);
        int remove0 = NONE, remove1 = NONE;
        int own = x < w ? pixel_compact(ys, x0, pad, t + uint(pad)) : NONE;
        for (int y = ys; y <= ye; ++y) {
            const int a0 = add0, a1 = add1, r0 = remove0, r1 = remove1, o = own;
            if (y < ye) {
                add0 = pixel_compact(y + 1 + pad, x0, pad, t);
                add1 = pixel_compact(y + 1 + pad, x0, pad, t + 64);
                remove0 = pixel_compact(y - pad, x0, pad, t);
                remove1 = pixel_compact(y - pad, x0, pad, t + 64);
                own = x < w ? pixel_compact(y + 1, x0, pad, t + uint(pad)) : NONE;
            }
            count_pixel(a0, t, first, false);
            count_pixel(a1, t + 64, first, false);
            count_pixel(r0, t, first, true);
            count_pixel(r1, t + 64, first, true);
            if ((gRowWindows[y - y0] & bit) == 0) continue;  // nothing to read: no barrier
            GroupMemoryBarrierWithGroupSync();
            if (o != NONE && o / CORE == k) {
                const int compact = o, I = o + bandBase;
                float c = 0.0;
                const int local = compact - first;  // 4 .. 59: its value +- 4 are in the window
                // The counts of its value and its value +- 1 .. 4, all read together.
                uint counts[2 * NUM_DIFFS + 1];
                [unroll] for (int m = 0; m < 2 * NUM_DIFFS + 1; ++m)
                    counts[m] = window_count(local + m - NUM_DIFFS, t, pad);
                const uint p0 = counts[NUM_DIFFS];
                const int value = I + NUM_DIFFS;
                [unroll] for (int d = 0; d < NUM_DIFFS; ++d) {
                    if (value <= int(pc[6 + d]) && value + d + 1 > int(pc[3])) {
                        // (The gate keeps compact + d + 1 inside the band, as on the CPU.)
                        const uint p1 = compact + d + 1 < bandSize ? counts[NUM_DIFFS + d + 1] : 0u;
                        const uint p2 = compact - d - 1 >= 0 ? counts[NUM_DIFFS - d - 1] : 0u;
                        const uint pm = p1 > p2 ? p1 : p2;
                        const float val = float(int(pc[10 + d]) * int(p0) * int(pm)) * reciprocal[pm + p0];
                        if (val > c) c = val;
                    }
                }
                cValues[y * w + x] = asuint(c);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        GroupMemoryBarrierWithGroupSync();
    }
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

#elif STAGE == 8 || STAGE == 10  // HIST, SUM
// The five scales at once: a row of workgroups each (group.y = the scale).
[[vk::binding(0, 0)]] StructuredBuffer<uint> cValues0;
[[vk::binding(1, 0)]] StructuredBuffer<uint> cValues1;
[[vk::binding(2, 0)]] StructuredBuffer<uint> cValues2;
[[vk::binding(3, 0)]] StructuredBuffer<uint> cValues3;
[[vk::binding(4, 0)]] StructuredBuffer<uint> cValues4;
#if STAGE == 8
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> hist;    // every scale's and level's bins
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> state;   // (above)
// pc: level, n[5], groups[5]
#else
[[vk::binding(5, 0)]] StructuredBuffer<uint> state;     // (above)
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> acc;
// pc: (unused), n[5], groups[5], the first scale's slot (sum, threshold, remaining, inexact: 4 a scale)
#endif

uint c_value(uint scale, uint i)
{
    switch (scale) {
    case 0: return cValues0[i];
    case 1: return cValues1[i];
    case 2: return cValues2[i];
    case 3: return cValues3[i];
    default: return cValues4[i];
    }
}

#if STAGE == 8
groupshared uint gHist[4096];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const uint scale = group.y, level = pc[0];
    if (group.x >= pc[6 + scale]) return;  // the scale's workgroups
    if (level > 0 && state[STATE_DONE + scale] != 0) return;  // found at level 0
    const uint bins = level < 2 ? 4096u : 256u;
    for (uint b = local; b < bins; b += 256) gHist[b] = 0;
    GroupMemoryBarrierWithGroupSync();
    const uint prefix = state[scale * 2];
    const uint stride = pc[6 + scale] * 256, n = pc[1 + scale];
    uint64_t sum = 0;  // level 0: the nonzero values', as SUM adds them
    bool inexact = false;
    for (uint i = group.x * 256 + local; i < n; i += stride) {
        const uint bits = c_value(scale, i);
        // Most c-values are 0 (pixels not masked): not counted here, every
        // workgroup on one bin; SELECT adds them to bin 0 where they belong.
        if (bits == 0) continue;
        uint bin;
        if (level == 0) {
            bin = bits >> 20;
            const uint e = bits >> 23;
            if (e >= 126 && e < 142)
                sum += uint64_t((bits & 0x7FFFFFu) | 0x800000u) << (e - 126);
            else
                inexact = true;
        } else if (level == 1) {
            if ((bits >> 20) != prefix) continue;
            bin = (bits >> 8) & 0xFFFu;
        } else {
            if ((bits >> 8) != prefix) continue;
            bin = bits & 0xFFu;
        }
        InterlockedAdd(gHist[bin], 1u);
    }
    if (level == 0) {
        const uint64_t total = WaveActiveSum(sum);
        if (WaveIsFirstLane() && total != 0) acc_add64(state, STATE_SUM + scale, total);
        if (WaveActiveAnyTrue(inexact) && WaveIsFirstLane()) InterlockedOr(state[STATE_INEXACT + scale], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint offset = (scale * 3 + level) * 4096;
    for (uint b = local; b < bins; b += 256)
        if (gHist[b] != 0) InterlockedAdd(hist[offset + b], gHist[b]);
}
#else
groupshared uint64_t gSum[256];
groupshared uint gInexact;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const uint scale = group.y;
    if (group.x >= pc[6 + scale]) return;
    if (state[STATE_DONE + scale] != 0) return;  // level 0 added it up
    if (local == 0) gInexact = 0;
    GroupMemoryBarrierWithGroupSync();
    const uint threshold = state[scale * 2];
    const uint stride = pc[6 + scale] * 256, n = pc[1 + scale];
    uint64_t sum = 0;
    for (uint i = group.x * 256 + local; i < n; i += stride) {
        const uint bits = c_value(scale, i);
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
        const uint slot = pc[11] + scale * 4;
        if (gSum[0] != 0) acc_add64(acc, slot, gSum[0]);
        if (gInexact != 0) InterlockedOr(acc[2 * (slot + 3)], 1u);
    }
}
#endif

#elif STAGE == 9  // SELECT
// A level of the radix select, a workgroup per scale: the bin the k-th
// largest value is in, the first bin from the top at which the count from
// the top reaches what is left of k (as a loop from the top would find it;
// with none, the bottom bin and k less every count). 256 threads sum 16 bins
// each (1 at the last level), the sums are scanned, and the thread whose
// bins hold it finds the bin.
[[vk::binding(0, 0)]] StructuredBuffer<uint> hist;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> state;   // prefix, remaining
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> acc;
// pc: level, k[5], the first scale's slot (sum, threshold, remaining,
// inexact: 4 a scale), n[5] (the scales' c-values). The zero c-values, which
// HIST does not count, are n less the rest (level 0, kept in state[10 +
// scale]), and are bin 0's at every level whose prefix is 0.

groupshared uint gScan[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const uint scale = group.y, level = pc[0];
    if (level > 0 && state[STATE_DONE + scale] != 0) return;  // found at level 0
    const uint bins = level < 2 ? 4096u : 256u, per = bins / 256;
    const uint at = scale * 2, offset = (scale * 3 + level) * 4096;
    const uint remaining = level == 0 ? pc[1 + scale] : state[at + 1];
    const uint prefix = level == 0 ? 0u : state[at];
    const uint zerosAt = 10 + scale;
    // The thread's bins, from the top: bins - 1 - per * local down.
    const uint top = bins - 1 - per * local;
    uint own = 0;
    for (uint i = 0; i < per; ++i) own += hist[offset + top - i];
    gScan[local] = own;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 1; step < 256; step <<= 1) {  // inclusive scan: the count from the top to the thread's last bin
        const uint add = local >= step ? gScan[local - step] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gScan[local] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    const uint before = local == 0 ? 0u : gScan[local - 1];
    // The zeros, in bin 0: the last thread's (the bottom bins' from the top).
    uint zeros = 0;
    if (level == 0)
        zeros = pc[7 + scale] - gScan[255];
    else if (prefix == 0)
        zeros = state[zerosAt];
    const uint scanned = gScan[local] + (local == 255 ? zeros : 0u);
    bool write = false;
    uint chosen = 0, left = 0;
    if (remaining == 0) {  // the top bin's count reaches it at once
        if (local == 0) {
            chosen = bins - 1;
            left = 0;
            write = true;
        }
    } else if (scanned >= remaining && before < remaining) {
        uint counted = before;
        for (uint i = 0; i < per; ++i) {
            const uint count = hist[offset + top - i] + (top - i == 0 ? zeros : 0u);
            if (counted + count >= remaining) {
                chosen = top - i;
                left = remaining - counted;
                break;
            }
            counted += count;
        }
        write = true;
    } else if (local == 255 && scanned < remaining) {  // no bin reaches it
        chosen = 0;
        left = remaining - scanned;
        write = true;
    }
    if (level == 0 && local == 255)
        state[zerosAt] = zeros;
    if (write) {
        const uint next = level == 0 ? chosen : (prefix << (level < 2 ? 12u : 8u)) | chosen;
        state[at] = next;
        state[at + 1] = left;
        const uint slot = pc[6] + scale * 4;
        if (level == 0) {
            // Among the zeros (bin 0 holds no other value: a nonzero c-value
            // is at least 0.5): the k-th largest is 0, and level 0 has added
            // up the values above it, as SUM would.
            const bool zero = chosen == 0;
            state[STATE_DONE + scale] = zero ? 1u : 0u;
            if (zero) {
                acc[2 * (slot + 1)] = 0;
                acc[2 * (slot + 2)] = left;
                const uint64_t sum = uint64_t(state[2 * (STATE_SUM + scale)])
                                     | (uint64_t(state[2 * (STATE_SUM + scale) + 1]) << 32);
                if (sum != 0) acc_add64(acc, slot, sum);
                if (state[STATE_INEXACT + scale] != 0) InterlockedOr(acc[2 * (slot + 3)], 1u);
            }
            state[2 * (STATE_SUM + scale)] = 0;  // for the next frame
            state[2 * (STATE_SUM + scale) + 1] = 0;
            state[STATE_INEXACT + scale] = 0;
        }
        if (level == 2) {
            acc[2 * (slot + 1)] = next;  // the k-th largest value's bits
            acc[2 * (slot + 2)] = left;  // how many of it are in the top k
        }
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
