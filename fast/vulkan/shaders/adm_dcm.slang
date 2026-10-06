// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// ADM's decouple and contrast masking of both gain limits (VMAF's and VMAF
// NEG's) in one pass: adm_decouple.slang's BOTH (VARIANT 0) and adm_cm.slang
// for a tile of TX x TY of contrast masking's positions. Contrast masking
// reads a position's restored and csfA images and the csfF image of its 3 x 3
// neighbourhood, all within the decouple step's region: a group decouples
// its tile and the positions around it, keeps their csfF in group memory and
// its own positions' restored and csfA in registers, and masks. The six
// images are never written to memory or read back: on an integrated GPU,
// which shares the memory's bandwidth with everything else, most of the two
// passes' time.
//
// Contrast masking rounds each row's sum once (adm_cm.slang: a group a row).
// Here a row's positions are in several tiles: each tile adds its part of
// each row's sums to rowSums (64-bit, exact in any order) and adm_rows.slang
// rounds the rows and adds them up. The integers are the two shaders'.
import glsl;  // umulExtended, imulExtended
#include "common.slang"

struct Params {
    int w, h;                       // the band images' size
    int inStride;                   // positions per band row
    int startRow, endRow, startCol, endCol;  // contrast masking's region
    uint rfactor0, rfactor1, rfactor2;
    int shiftCub0, shiftCub1, shiftCub2;
    uint rowSlot;                   // this scale's first row sum (words / 2) in rowSums
};
[[vk::push_constant]] Params pc;
// The parameters as specialization constants (constant k: Params' word k):
// the engine compiles the shader for its passes' (a pipeline for each set),
// and the compiler folds what they decide.
#if SCALE0
[vk::constant_id(0)] const int kSpec_w = 0;
#define P_w kSpec_w
[vk::constant_id(1)] const int kSpec_h = 0;
#define P_h kSpec_h
[vk::constant_id(2)] const int kSpec_inStride = 0;
#define P_inStride kSpec_inStride
[vk::constant_id(3)] const int kSpec_startRow = 0;
#define P_startRow kSpec_startRow
[vk::constant_id(4)] const int kSpec_endRow = 0;
#define P_endRow kSpec_endRow
[vk::constant_id(5)] const int kSpec_startCol = 0;
#define P_startCol kSpec_startCol
[vk::constant_id(6)] const int kSpec_endCol = 0;
#define P_endCol kSpec_endCol
[vk::constant_id(7)] const uint kSpec_rfactor0 = 0;
#define P_rfactor0 kSpec_rfactor0
[vk::constant_id(8)] const uint kSpec_rfactor1 = 0;
#define P_rfactor1 kSpec_rfactor1
[vk::constant_id(9)] const uint kSpec_rfactor2 = 0;
#define P_rfactor2 kSpec_rfactor2
[vk::constant_id(10)] const int kSpec_shiftCub0 = 0;
#define P_shiftCub0 kSpec_shiftCub0
[vk::constant_id(11)] const int kSpec_shiftCub1 = 0;
#define P_shiftCub1 kSpec_shiftCub1
[vk::constant_id(12)] const int kSpec_shiftCub2 = 0;
#define P_shiftCub2 kSpec_shiftCub2
[vk::constant_id(13)] const uint kSpec_rowSlot = 0;
#define P_rowSlot kSpec_rowSlot
#else
#define P_w pc.w
#define P_h pc.h
#define P_inStride pc.inStride
#define P_startRow pc.startRow
#define P_endRow pc.endRow
#define P_startCol pc.startCol
#define P_endCol pc.endCol
#define P_rfactor0 pc.rfactor0
#define P_rfactor1 pc.rfactor1
#define P_rfactor2 pc.rfactor2
#define P_shiftCub0 pc.shiftCub0
#define P_shiftCub1 pc.shiftCub1
#define P_shiftCub2 pc.shiftCub2
#define P_rowSlot pc.rowSlot
#endif
[[vk::binding(0, 0)]] StructuredBuffer<uint> refBands;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disBands;
[[vk::binding(2, 0)]] StructuredBuffer<uint> divTable;
// [limit][band][row - startRow], 64-bit sums as two words each.
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> rowSums;

#define COS_MANTISSA 0xFFEC0Au
#include "adm_decouple_common.slang"

#if !SCALE0
// Scales 1-3's 64-bit integer arithmetic (the CUDA code's) from 32-bit
// words. A GPU without 64-bit integers (Intel's) makes each 64-bit multiply,
// shift and comparison of several instructions; SPIR-V's 32 x 32 -> 64-bit
// multiplies give a product's two words directly. Each helper is, to the
// bit, the 64-bit expression its comment gives.

// (int64_t)a * b.
int64_t mul64(int a, int b)
{
    int hi, lo;
    imulExtended(a, b, hi, lo);
    return int64_t((uint64_t(uint(hi)) << 32) | uint(lo));
}

// (int)(((int64_t)r * v + 2^27) >> 28), r unsigned: bits 28..59 of the sum.
int rfactor_round(uint r, int v)
{
    uint hi, lo;
    umulExtended(r, uint(v), hi, lo);
    if (v < 0)
        hi -= r;  // r v = r (v + 2^32) - r 2^32
    const uint sum = lo + 134217728u;
    hi += sum < lo ? 1u : 0u;
    return int((sum >> 28) | (hi << 4));
}

// (int)(((int64_t)c * a + (int32_t)(1u << 31)) >> 32): the high word of
// c a - 2^31.
int high_round(int c, int a)
{
    int hi, lo;
    imulExtended(c, a, hi, lo);
    return hi - (uint(lo) < 0x80000000u ? 1 : 0);
}

// saturate32((int64_t)a * b).
int saturating_mul(int a, int b)
{
    int hi, lo;
    imulExtended(a, b, hi, lo);
    if (hi != (lo >> 31))
        return hi < 0 ? int(0x80000000u) : 0x7FFFFFFF;
    return lo;
}

// 2^30 / top, top in 1..32768: a float quotient (Vulkan's division is
// within 2.5 ulp: at most 320 / top off), corrected by its remainder (at
// most 320 + top in magnitude) divided the same way, to within one, then to
// the quotient. A GPU without integer division (Intel's) takes many more
// instructions for the division itself.
uint div30(uint top)
{
    uint q = uint(1073741824.0 / float(top));
    int r = int(1073741824u - q * top);
    q = uint(int(q) + int(floor(float(r) / float(top))));
    r = int(1073741824u - q * top);
    if (r < 0)
        q--;
    else if (r >= int(top))
        q++;
    return q;
}
#endif

#ifndef TX
#define TX 16               // the engine's kDcmTile
#define TY 8
#endif
#define THREADS (TX * TY)
#define HX (TX + 2)               // the tile and the positions around it
#define HY (TY + 2)
#define LIMITS 2
// The gain limits, VMAF's and VMAF NEG's (the pass is for the two together),
// and contrast masking's shifts but shiftCub, which the band width sets
// (integer_adm_cuda.c's).
#define GAIN_A 100
#define GAIN_B 1
#if SCALE0
static const int kShiftSub[3] = { 10, 10, 12 }, kShiftSq[3] = { 29, 29, 30 };
// And scale 0's rfactors (the engine's adm_rfactors).
static const uint kRfactor0[3] = { 36453, 36453, 49417 };
#define RFACTORS kRfactor0
#else
#define RFACTORS { P_rfactor0, P_rfactor1, P_rfactor2 }
static const int kShiftSq[3] = { 30, 30, 30 };
#endif
// csfF of both limits' three bands a position: 16-bit at scale 0 (three
// words, two values each), 32-bit at the others.
#if SCALE0
#define F_WORDS 3
#else
#define F_WORDS 6
#endif
groupshared uint gF[HX * HY * F_WORDS];
// Each of the tile's rows' sums, [row][limit * 3 + band][copy], 64-bit as
// two words: zeroed at the start, added to after the first barrier. A
// thread adds into copy tx % ROW_COPIES: the row's 16 threads' atomic adds
// to the same words wait on each other. At scales 1-3, four copies (Core
// Ultra 9 285K's iGPU, 4K: 0.973 -> 0.935 ms); at scale 0 they gained
// nothing.
#if SCALE0
#define ROW_COPIES 1
#else
#define ROW_COPIES 4
#endif
groupshared uint gRows[TY * 12 * ROW_COPIES];

// adm_decouple.slang's main (BOTH, VARIANT 0) for the position at `from`:
// both limits' restored, csfA and csfF of its three bands.
void decouple(int from, out int restored[LIMITS][3], out int csfA[LIMITS][3], out int csfF[LIMITS][3])
{
    int o[3], t[3];
    adm_hvd(refBands, from, SCALE0 != 0, o);
    adm_hvd(disBands, from, SCALE0 != 0, t);
#if SCALE0
    const bool aligned = within_one_degree16(o[0], o[1], t[0], t[1]);
#else
    int64_t otDp = mul64(o[0], t[0]) + mul64(o[1], t[1]);
    int64_t oMagSq = mul64(o[0], o[0]) + mul64(o[1], o[1]);
    int64_t tMagSq = mul64(t[0], t[0]) + mul64(t[1], t[1]);
    bool aligned = within_one_degree(otDp, oMagSq, tMagSq);
#endif
    const int limits[LIMITS] = { aligned ? GAIN_A : 1, aligned ? GAIN_B : 1 };
    const uint rfactor[3] = RFACTORS;
    for (int b = 0; b < 3; ++b) {
#if SCALE0
        int k = 32768;
        int quotient = reciprocal(o[b]);
        if (o[b] != 0)
            k = clamp((quotient >> 15) * t[b] + (((quotient & 32767) * t[b] + 16384) >> 15), 0, 32768);
        int product = k * o[b];
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
            int restV = ((product + 16384) >> 15) * limits[v];
            if (aligned) {
                if (product > 0) restV = min(restV, t[b]);
                if (product < 0) restV = max(restV, t[b]);
            }
            restored[v][b] = sext16(restV);
            const int additiveV = sext16(t[b] - restV);
            int filtered = sext16((int(rfactor[b] * uint(additiveV)) + kAdd[b]) >> kShift[b]);
            csfA[v][b] = filtered;
            csfF[v][b] = sext16((4369 * abs(filtered) + 2048) >> 12);
        }
#else
        uint magnitude = uint(abs(o[b]));
        int shift = 0;
        uint top = magnitude;
        if (magnitude >= 32768u) {  // get_best15_from32
            shift = msb32(magnitude) - 14;
            top = ((magnitude + (1u << (shift - 1))) >> shift) & 0xFFFFu;
        }
        // k = clamp((2^30 / top * t * sign + 2^(14 + shift)) >> (15 + shift), 0,
        // 32768), the 64-bit sum as two words: if negative, 0; if 2^32 or more
        // after the shift, 32768.
        int k = 32768;
        if (o[b] != 0) {
            int hi, lo;
            imulExtended(int(div30(top)), t[b], hi, lo);
            if (o[b] < 0) {  // the product negated
                hi = ~hi + (lo == 0 ? 1 : 0);
                lo = -lo;
            }
            const uint sum = uint(lo) + (1u << uint(14 + shift));
            hi += sum < uint(lo) ? 1 : 0;
            const int n = 15 + shift;  // 15..32
            if (hi < 0)
                k = 0;
            else if (n == 32)
                k = min(hi, 32768);
            else if ((uint(hi) >> uint(n)) != 0)
                k = 32768;
            else
                k = int(min((sum >> uint(n)) | (uint(hi) << uint(32 - n)), 32768u));
        }
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
            // (k o + 16384) >> 15, k in 0..32768: k times o's bits above the 15th
            // exactly, plus the rounded rest.
            const int rest = k * (o[b] >> 15) + ((k * (o[b] & 32767) + 16384) >> 15);
            int restV = limits[v] == 1 ? rest : saturating_mul(rest, limits[v]);
            if (aligned && k > 0) {
                if (o[b] > 0) restV = min(restV, t[b]);
                if (o[b] < 0) restV = max(restV, t[b]);
            }
            const int additiveV = t[b] - restV;
            const int filtered = rfactor_round(rfactor[b], additiveV);
            restored[v][b] = restV;
            csfA[v][b] = filtered;
            // The CUDA kernel's rounding term is (int32_t)(1u << 31): negative.
            csfF[v][b] = high_round(143165577, abs(filtered));
        }
#endif
    }
}

void store_f(uint at, int csfF[LIMITS][3])
{
#if SCALE0
    gF[at * 3] = uint(csfF[0][0] & 0xFFFF) | uint(csfF[0][1] << 16);
    gF[at * 3 + 1] = uint(csfF[0][2] & 0xFFFF) | uint(csfF[1][0] << 16);
    gF[at * 3 + 2] = uint(csfF[1][1] & 0xFFFF) | uint(csfF[1][2] << 16);
#else
    for (int v = 0; v < LIMITS; ++v)
        for (int b = 0; b < 3; ++b)
            gF[at * 6 + uint(v * 3 + b)] = uint(csfF[v][b]);
#endif
}

// The three bands' csfF at tile position `at`, added up, of each limit: its
// words read once for both.
void add_f(uint at, inout int thresholds[LIMITS])
{
#if SCALE0
    const int w0 = int(gF[at * 3]), w1 = int(gF[at * 3 + 1]), w2 = int(gF[at * 3 + 2]);
    thresholds[0] += sext16(w0) + (w0 >> 16) + sext16(w1);
    thresholds[1] += (w1 >> 16) + sext16(w2) + (w2 >> 16);
#else
    for (int v = 0; v < LIMITS; ++v)
        thresholds[v] += int(gF[at * 6 + uint(v * 3)]) + int(gF[at * 6 + uint(v * 3 + 1)])
                         + int(gF[at * 6 + uint(v * 3 + 2)]);
#endif
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const int x0 = P_startCol + int(group.x) * TX, y0 = P_startRow + int(group.y) * TY;
    const int tx = int(local % uint(TX)), ty = int(local / uint(TX));
    for (uint z = local; z < uint(TY * 12 * ROW_COPIES); z += THREADS)
        gRows[z] = 0;
    // The thread's own position first (its restored and csfA kept), then the
    // positions around the tile, each decoupled once. Positions outside the
    // band images are not; contrast masking never reads them.
    int restored[LIMITS][3], csfA[LIMITS][3], csfF[LIMITS][3];
    const int i = y0 + ty, j = x0 + tx;
    if (i < P_h && j < P_w) {
        decouple(i * P_inStride + j, restored, csfA, csfF);
        store_f(uint((ty + 1) * HX + tx + 1), csfF);
    }
    for (uint e = local; e < uint(2 * HX + 2 * TY); e += THREADS) {
        int hx, hy;
        if (e < uint(HX)) {
            hx = int(e);
            hy = 0;
        } else if (e < uint(2 * HX)) {
            hx = int(e) - HX;
            hy = HY - 1;
        } else {
            hx = (e - uint(2 * HX)) < uint(TY) ? 0 : HX - 1;
            hy = 1 + int((e - uint(2 * HX)) % uint(TY));
        }
        const int hi = y0 - 1 + hy, hj = x0 - 1 + hx;
        if (hi >= 0 && hi < P_h && hj >= 0 && hj < P_w) {
            int r[LIMITS][3], a[LIMITS][3], f[LIMITS][3];
            decouple(hi * P_inStride + hj, r, a, f);
            store_f(uint(hy * HX + hx), f);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // adm_cm.slang's masking of the thread's position, both limits.
    int64_t sums[LIMITS][3] = { { 0, 0, 0 }, { 0, 0, 0 } };
    if (i < P_endRow && j < P_endCol) {
        const int rows[3] = { min(abs(i - 1), P_h - 1), i, min(i + 1, P_h - 1) };
        const int columns[3] = { min(abs(j - 1), P_w - 1), j, min(j + 1, P_w - 1) };
        const uint rfactor[3] = RFACTORS;
        const int shiftCub[3] = { P_shiftCub0, P_shiftCub1, P_shiftCub2 };
        // The 3x3 neighbourhood's csfF of every band, but the centre's own,
        // which is csfA's: both limits' thresholds.
        int thresholds[LIMITS] = { 0, 0 };
        for (int di = 0; di < 3; ++di) {
            for (int c = 0; c < 3; ++c) {
                if (di == 1 && c == 1) {
                    [ForceUnroll]
                    for (int v = 0; v < LIMITS; ++v) {
                        for (int theta = 0; theta < 3; ++theta) {
                            int center = abs(csfA[v][theta]);
#if SCALE0
                            thresholds[v] += sext16((8738 * center + 2048) >> 12);
#else
                            thresholds[v] += high_round(286331153, center);
#endif
                        }
                    }
                } else {
                    add_f(uint((rows[di] - y0 + 1) * HX + columns[c] - x0 + 1), thresholds);
                }
            }
        }
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
            const int threshold = thresholds[v];
            for (int b = 0; b < 3; ++b) {
                int source = restored[v][b];
#if SCALE0
                int value = abs(int(rfactor[b] * uint(source))) - (threshold << kShiftSub[b]);
#else
                int value = abs(rfactor_round(rfactor[b], source)) - threshold;
#endif
                value = max(value, 0);
                int square = int((int64_t(value) * value + (1ll << (kShiftSq[b] - 1))) >> kShiftSq[b]);
                const int64_t addCub = shiftCub[b] ? (1ll << (shiftCub[b] - 1)) : 0;
                sums[v][b] = (int64_t(square) * value + addCub) >> shiftCub[b];
            }
        }
    }
    // Each row's part of the tile: added up in group memory (common.slang's
    // acc_add64, there), then into rowSums.
    for (int k = 0; k < 6; ++k) {
        const uint64_t value = uint64_t(sums[k / 3][k % 3]);
        if (value != 0) {
            const uint lo = uint(value), at = uint(((ty * 6 + k) * ROW_COPIES + tx % ROW_COPIES) * 2);
            uint old;
            InterlockedAdd(gRows[at], lo, old);
            const uint high = uint(value >> 32) + (old > ~lo ? 1u : 0u);
            if (high != 0)
                InterlockedAdd(gRows[at + 1], high);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local < uint(6 * TY)) {
        const int k = int(local) % 6, row = int(local) / 6, ri = y0 + row;
        uint64_t total = 0;
        for (int c = 0; c < ROW_COPIES; ++c) {
            const uint at = uint(((row * 6 + k) * ROW_COPIES + c) * 2);
            total += uint64_t(gRows[at]) | (uint64_t(gRows[at + 1]) << 32);
        }
        if (ri < P_endRow && total != 0)
            acc_add64(rowSums, P_rowSlot + uint(k * (P_endRow - P_startRow) + ri - P_startRow), total);
    }
}
