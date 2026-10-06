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
#include "common.slang"

struct Params {
    int w, h;                       // the band images' size
    int inStride;                   // positions per band row
    int startRow, endRow, startCol, endCol;  // contrast masking's region
    uint gainA, gainB;              // the gain limits: VMAF's, VMAF NEG's
    uint rfactor0, rfactor1, rfactor2;
    int shiftSub0, shiftSub1, shiftSub2;
    int shiftSq0, shiftSq1, shiftSq2;
    int shiftCub0, shiftCub1, shiftCub2;
    uint rowSlot;                   // this scale's first row sum (words / 2) in rowSums
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> refBands;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disBands;
[[vk::binding(2, 0)]] StructuredBuffer<uint> divTable;
// [limit][band][row - startRow], 64-bit sums as two words each.
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> rowSums;

#define COS_MANTISSA 0xFFEC0Au
#include "adm_decouple_common.slang"

#ifndef TX
#define TX 16               // the engine's kDcmTile
#define TY 8
#endif
#define THREADS (TX * TY)
#define HX (TX + 2)               // the tile and the positions around it
#define HY (TY + 2)
#define LIMITS 2
// csfF of both limits' three bands a position: 16-bit at scale 0 (three
// words, two values each), 32-bit at the others.
#if SCALE0
#define F_WORDS 3
#else
#define F_WORDS 6
#endif
groupshared uint gF[HX * HY * F_WORDS];
// Each of the tile's rows' sums, [row][limit * 3 + band], 64-bit as two
// words: zeroed at the start, added to after the first barrier.
groupshared uint gRows[TY * 12];

// adm_decouple.slang's main (BOTH, VARIANT 0) for the position at `from`:
// both limits' restored, csfA and csfF of its three bands.
void decouple(int from, out int restored[LIMITS][3], out int csfA[LIMITS][3], out int csfF[LIMITS][3])
{
    int o[3], t[3];
    adm_hvd(refBands, from, SCALE0 != 0, o);
    adm_hvd(disBands, from, SCALE0 != 0, t);
    int64_t otDp = int64_t(o[0]) * t[0] + int64_t(o[1]) * t[1];
    int64_t oMagSq = int64_t(o[0]) * o[0] + int64_t(o[1]) * o[1];
    int64_t tMagSq = int64_t(t[0]) * t[0] + int64_t(t[1]) * t[1];
    bool aligned = within_one_degree(otDp, oMagSq, tMagSq);
    const int64_t limits[LIMITS] = { aligned ? int64_t(pc.gainA) : int64_t(1),
                                     aligned ? int64_t(pc.gainB) : int64_t(1) };
    const uint rfactor[3] = { pc.rfactor0, pc.rfactor1, pc.rfactor2 };
    for (int b = 0; b < 3; ++b) {
#if SCALE0
        int k = 32768;
        int quotient = reciprocal(o[b]);
        if (o[b] != 0)
            k = clamp((quotient >> 15) * t[b] + (((quotient & 32767) * t[b] + 16384) >> 15), 0, 32768);
        int product = k * o[b];
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
            int restV = ((product + 16384) >> 15) * int(limits[v]);
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
        int64_t k = 32768;
        if (o[b] != 0) {
            int64_t sign = o[b] < 0 ? -1 : 1;
            k = (int64_t(1073741824 / int(top)) * t[b] * sign + int64_t(1 << (14 + shift))) >> (15 + shift);
            k = clamp(k, int64_t(0), int64_t(32768));
        }
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
            int restV = saturate32(int64_t(int((k * o[b] + 16384) >> 15)) * limits[v]);
            if (aligned && k > 0) {
                if (o[b] > 0) restV = min(restV, t[b]);
                if (o[b] < 0) restV = max(restV, t[b]);
            }
            const int additiveV = t[b] - restV;
            const int filtered = int((int64_t(rfactor[b]) * int64_t(additiveV) + 134217728) >> 28);
            restored[v][b] = restV;
            csfA[v][b] = filtered;
            // The CUDA kernel's rounding term is (int32_t)(1u << 31): negative.
            csfF[v][b] = int((int64_t(143165577) * abs(filtered) - 2147483648ll) >> 32);
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

// The three bands' csfF of limit v at tile position `at`, added up.
int f_sum(uint at, int v)
{
#if SCALE0
    const int w0 = int(gF[at * 3]), w1 = int(gF[at * 3 + 1]), w2 = int(gF[at * 3 + 2]);
    return v == 0 ? sext16(w0) + (w0 >> 16) + sext16(w1) : (w1 >> 16) + sext16(w2) + (w2 >> 16);
#else
    return int(gF[at * 6 + uint(v * 3)]) + int(gF[at * 6 + uint(v * 3 + 1)]) + int(gF[at * 6 + uint(v * 3 + 2)]);
#endif
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const int x0 = pc.startCol + int(group.x) * TX, y0 = pc.startRow + int(group.y) * TY;
    const int tx = int(local % uint(TX)), ty = int(local / uint(TX));
    if (local < uint(TY * 12))
        gRows[local] = 0;
    // The thread's own position first (its restored and csfA kept), then the
    // positions around the tile, each decoupled once. Positions outside the
    // band images are not; contrast masking never reads them.
    int restored[LIMITS][3], csfA[LIMITS][3], csfF[LIMITS][3];
    const int i = y0 + ty, j = x0 + tx;
    if (i < pc.h && j < pc.w) {
        decouple(i * pc.inStride + j, restored, csfA, csfF);
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
        if (hi >= 0 && hi < pc.h && hj >= 0 && hj < pc.w) {
            int r[LIMITS][3], a[LIMITS][3], f[LIMITS][3];
            decouple(hi * pc.inStride + hj, r, a, f);
            store_f(uint(hy * HX + hx), f);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // adm_cm.slang's masking of the thread's position, both limits.
    int64_t sums[LIMITS][3] = { { 0, 0, 0 }, { 0, 0, 0 } };
    if (i < pc.endRow && j < pc.endCol) {
        const int rows[3] = { min(abs(i - 1), pc.h - 1), i, min(i + 1, pc.h - 1) };
        const int columns[3] = { min(abs(j - 1), pc.w - 1), j, min(j + 1, pc.w - 1) };
        const uint rfactor[3] = { pc.rfactor0, pc.rfactor1, pc.rfactor2 };
        const int shiftSub[3] = { pc.shiftSub0, pc.shiftSub1, pc.shiftSub2 };
        const int shiftSq[3] = { pc.shiftSq0, pc.shiftSq1, pc.shiftSq2 };
        const int shiftCub[3] = { pc.shiftCub0, pc.shiftCub1, pc.shiftCub2 };
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
            // The 3x3 neighbourhood's csfF of every band, but the centre's own,
            // which is csfA's.
            int threshold = 0;
            for (int di = 0; di < 3; ++di) {
                for (int c = 0; c < 3; ++c) {
                    if (di == 1 && c == 1) {
                        for (int theta = 0; theta < 3; ++theta) {
                            int center = abs(csfA[v][theta]);
#if SCALE0
                            threshold += sext16((8738 * center + 2048) >> 12);
#else
                            threshold += int((int64_t(286331153) * center - 2147483648ll) >> 32);
#endif
                        }
                    } else {
                        threshold += f_sum(uint((rows[di] - y0 + 1) * HX + columns[c] - x0 + 1), v);
                    }
                }
            }
            for (int b = 0; b < 3; ++b) {
                int source = restored[v][b];
#if SCALE0
                int value = abs(int(rfactor[b] * uint(source))) - (threshold << shiftSub[b]);
#else
                int value = abs(int((int64_t(source) * int64_t(rfactor[b]) + 134217728) >> 28)) - threshold;
#endif
                value = max(value, 0);
                int square = int((int64_t(value) * value + (1ll << (shiftSq[b] - 1))) >> shiftSq[b]);
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
            const uint lo = uint(value), at = uint(ty * 12 + 2 * k);
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
        const uint64_t total = uint64_t(gRows[row * 12 + 2 * k]) | (uint64_t(gRows[row * 12 + 2 * k + 1]) << 32);
        if (ri < pc.endRow && total != 0)
            acc_add64(rowSums, pc.rowSlot + uint(k * (pc.endRow - pc.startRow) + ri - pc.startRow), total);
    }
}
