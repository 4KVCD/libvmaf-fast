// Derived from libvmaf (https://github.com/Netflix/vmaf): its CPU code
// (integer_adm.c), Copyright 2016-2020 Netflix, Inc., and its CUDA feature
// extractors, Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation; licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root
// of this repository.
//
// VMAF v1's ADM of a scale after its wavelet transform, in one pass: what
// adm_decouple.slang (V1), adm_csf_den.slang (ROWWISE) and adm_cm.slang (both
// images) calculate, without the six images they pass on.
//
// A workgroup takes ROWS rows of the bands and goes across them 128 columns
// at a time. Each thread decouples its column's pixels of those rows and of
// the rows above and below them (the contrast masking's neighbours) and
// keeps its own rows' values; of the neighbours only what the masking
// threshold adds up is kept, in shared memory: per pixel the three bands'
// csfF summed (and csfRF's). The threshold is a 32-bit integer sum, the same
// in any order (it is modulo 2^32 if it wraps). The rows' sums are each
// rounded as the CPU code rounds them, a row at a time: here when the whole
// row is summed.
#include "common.slang"

[[vk::push_constant]] uniform uint pc[4];  // the scale's first word in params
[[vk::binding(0, 0)]] StructuredBuffer<uint> refBands;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disBands;
[[vk::binding(2, 0)]] StructuredBuffer<uint> divTable;
[[vk::binding(3, 0)]] StructuredBuffer<int> params;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> acc;

// params, from pc[0]:
#define P(i) params[pc[0] + (i)]
#define BW P(0)
#define BH P(1)
#define IN_STRIDE P(2)
#define GAIN_LIMIT P(3)
#define RFACTOR(b) uint(P(4 + (b)))
#define COS_MANTISSA uint(P(7))
#define CM_TOP P(8)        // start_row
#define CM_BOTTOM P(9)     // end_row
#define CM_LEFT P(10)      // start_col
#define CM_RIGHT P(11)     // end_col
#define DEN_TOP P(12)
#define DEN_BOTTOM P(13)
#define DEN_LEFT P(14)
#define DEN_RIGHT P(15)
#define SHIFT_SUB(b) P(16 + (b))
#define SHIFT_SQ(b) P(19 + (b))
#define ADD_SQ(b) P(22 + (b))
#define SHIFT_CUB(b) P(25 + (b))
#define ADD_CUB(b) P(28 + (b))
#define SHIFT_INNER P(31)
#define ADD_INNER P(32)
#define DEN_SHIFT_SQ uint(P(33))
#define DEN_ADD_SQ uint(P(34))
#define DEN_SHIFT_CUB uint(P(35))
#define DEN_ADD_CUB uint(P(36))
#define DEN_SHIFT_ACCUM uint(P(37))
#define DEN_ADD_ACCUM uint(P(38))
#define SLOT_CM0 uint(P(39))   // the restored image masked by the additive one
#define SLOT_CM1 uint(P(40))   // the additive image masked by the restored one (AIM)
#define SLOT_DEN uint(P(41))
#define FIRST_ROW P(42)
#define LAST_ROW P(43)         // past it

#include "adm_decouple_common.slang"

#define ROWS 4
#define GROUP 128
#define SPAN (GROUP + 2)
// Per pixel of the rows and their neighbours: the bands' csfF summed, and csfRF's.
groupshared int gSumF[(ROWS + 2) * SPAN];
groupshared int gSumRF[(ROWS + 2) * SPAN];
groupshared int64_t gTotals[(GROUP / 32) * ROWS * 9];

// A pixel's decoupling (adm_decouple.slang with V1 and VARIANT 0): the
// restored and additive values of its three bands, and per band
// csfA / csfF of the additive and csfR / csfRF of the restored image.
struct Decoupled {
    int restored[3], additive[3], csfA[3], csfF[3], csfR[3], csfRF[3];
};

Decoupled decouple(int i, int j)
{
    Decoupled out;
    const int from = (i * IN_STRIDE + j) * 4;
    int o[3], t[3];
    for (int b = 0; b < 3; ++b) {
        o[b] = int(refBands[from + 1 + b]);
        t[b] = int(disBands[from + 1 + b]);
    }
    const int64_t otDp = int64_t(o[0]) * t[0] + int64_t(o[1]) * t[1];
    const int64_t oMagSq = int64_t(o[0]) * o[0] + int64_t(o[1]) * o[1];
    const int64_t tMagSq = int64_t(t[0]) * t[0] + int64_t(t[1]) * t[1];
    const bool aligned = within_one_degree(otDp, oMagSq, tMagSq);
    const int64_t limit = aligned ? int64_t(GAIN_LIMIT) : int64_t(1);
    for (int b = 0; b < 3; ++b) {
        const uint rfactor = RFACTOR(b);
#if SCALE0
        int k = 32768;
        const int quotient = reciprocal(o[b]);
        if (o[b] != 0)
            k = clamp((quotient >> 15) * t[b] + (((quotient & 32767) * t[b] + 16384) >> 15), 0, 32768);
        const int product = k * o[b];
        int rest = ((product + 16384) >> 15) * int(limit);
        if (aligned) {
            if (product > 0) rest = min(rest, t[b]);
            if (product < 0) rest = max(rest, t[b]);
        }
        out.restored[b] = sext16(rest);
        const int additive = sext16(t[b] - rest);
        out.additive[b] = additive;
        const int filtered = sext16((int(rfactor * uint(additive)) + kAdd[b]) >> kShift[b]);
        out.csfA[b] = filtered;
        out.csfF[b] = sext16((4369 * abs(filtered) + 2048) >> 12);
        const int filteredR = sext16((int(rfactor * uint(sext16(rest))) + kAdd[b]) >> kShift[b]);
        out.csfR[b] = filteredR;
        out.csfRF[b] = sext16((4369 * abs(filteredR) + 2048) >> 12);
#else
        const uint magnitude = uint(abs(o[b]));
        int shift = 0;
        uint top = magnitude;
        if (magnitude >= 32768u) {  // get_best15_from32
            shift = msb32(magnitude) - 14;
            top = ((magnitude + (1u << (shift - 1))) >> shift) & 0xFFFFu;
        }
        int64_t k = 32768;
        if (o[b] != 0) {
            const int64_t sign = o[b] < 0 ? -1 : 1;
            k = (int64_t(1073741824 / int(top)) * t[b] * sign + int64_t(1 << (14 + shift))) >> (15 + shift);
            k = clamp(k, int64_t(0), int64_t(32768));
        }
        int rest = saturate32(int64_t(int((k * o[b] + 16384) >> 15)) * limit);
        if (aligned && k > 0) {
            if (o[b] > 0) rest = min(rest, t[b]);
            if (o[b] < 0) rest = max(rest, t[b]);
        }
        out.restored[b] = rest;
        const int additive = t[b] - rest;
        out.additive[b] = additive;
        const int filtered = int((int64_t(rfactor) * int64_t(additive) + 134217728) >> 28);
        out.csfA[b] = filtered;
        // The CUDA kernel's rounding term is (int32_t)(1u << 31): negative.
        out.csfF[b] = int((int64_t(143165577) * abs(filtered) - 2147483648ll) >> 32);
        const int filteredR = int((int64_t(rfactor) * int64_t(rest) + 134217728) >> 28);
        out.csfR[b] = filteredR;
        out.csfRF[b] = int((int64_t(143165577) * abs(filteredR) - 2147483648ll) >> 32);
#endif
    }
    return out;
}

// What the centre adds to the masking threshold, from |csfA| (or |csfR|).
int center_term(int center)
{
#if SCALE0
    return sext16((8738 * center + 2048) >> 12);
#else
    // (int32_t)(1u << 31) again: the rounding term is negative.
    return int((int64_t(286331153) * center - 2147483648ll) >> 32);
#endif
}

// adm_cm's masked value of a band, cubed and shifted as it is summed.
int64_t masked_term(int source, int threshold, int b)
{
#if SCALE0
    int value = abs(int(RFACTOR(b) * uint(source))) - (threshold << SHIFT_SUB(b));
#else
    int value = abs(int((int64_t(source) * int64_t(RFACTOR(b)) + 134217728) >> 28)) - threshold;
#endif
    value = max(value, 0);
    const int square = int((int64_t(value) * value + ADD_SQ(b)) >> SHIFT_SQ(b));
    return (int64_t(square) * value + ADD_CUB(b)) >> SHIFT_CUB(b);
}

// adm_csf_den's term of a reference band value.
uint64_t den_term(int value)
{
#if SCALE0
    const uint64_t t = uint64_t(uint(abs(value)) & 0xFFFFu);
    return t * t * t;
#else
    const uint64_t t = uint64_t(uint(abs(value)));
    return ((((t * t + DEN_ADD_SQ) >> DEN_SHIFT_SQ) * t) + DEN_ADD_CUB) >> DEN_SHIFT_CUB;
#endif
}

bool in_cm_decouple(int i, int j)  // decoupled for the masking: its region and a pixel around it
{
    return i >= CM_TOP - 1 && i <= CM_BOTTOM && j >= CM_LEFT - 1 && j <= CM_RIGHT;
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const int r0 = FIRST_ROW + int(group.y) * ROWS;
    const int colStart = min(CM_LEFT, DEN_LEFT), colEnd = max(CM_RIGHT, DEN_RIGHT);
    // Per row of the workgroup: the cm sums of the two images' three bands, and the denominator's.
    int64_t cm[ROWS][6];
    uint64_t den[ROWS][3];
    [unroll] for (int r = 0; r < ROWS; ++r) {
        [unroll] for (int s = 0; s < 6; ++s) cm[r][s] = 0;
        for (int b = 0; b < 3; ++b) den[r][b] = 0;
    }
    for (int c0 = colStart; c0 < colEnd; c0 += GROUP) {
        const int j = c0 + int(local);
        // The thread's column: its rows' values kept, every row's sums shared.
        int restored[ROWS][3], additive[ROWS][3], centerA[ROWS], centerR[ROWS];
        [unroll] for (int rr = 0; rr < ROWS + 2; ++rr) {
            const int i = r0 - 1 + rr;
            int sumF = 0, sumRF = 0;
            if (in_cm_decouple(i, j)) {
                const Decoupled d = decouple(i, j);
                sumF = d.csfF[0] + d.csfF[1] + d.csfF[2];
                sumRF = d.csfRF[0] + d.csfRF[1] + d.csfRF[2];
                if (rr >= 1 && rr <= ROWS) {
                    const int r = rr - 1;
                    centerA[r] = center_term(abs(d.csfA[0])) + center_term(abs(d.csfA[1])) + center_term(abs(d.csfA[2]));
                    centerR[r] = center_term(abs(d.csfR[0])) + center_term(abs(d.csfR[1])) + center_term(abs(d.csfR[2]));
                    for (int b = 0; b < 3; ++b) {
                        restored[r][b] = d.restored[b];
                        additive[r][b] = d.additive[b];
                    }
                }
            }
            gSumF[rr * SPAN + int(local) + 1] = sumF;
            gSumRF[rr * SPAN + int(local) + 1] = sumRF;
            if (rr >= 1 && rr <= ROWS && i >= DEN_TOP && i < DEN_BOTTOM && j >= DEN_LEFT && j < DEN_RIGHT) {
                const int from = (i * IN_STRIDE + j) * 4;
                for (int b = 0; b < 3; ++b)
                    den[rr - 1][b] += den_term(int(refBands[from + 1 + b]));
            }
        }
        // The columns either side of the chunk.
        if (local < 2) {
            const int jj = local == 0 ? c0 - 1 : c0 + GROUP;
            const int at = local == 0 ? 0 : SPAN - 1;
            for (int rr = 0; rr < ROWS + 2; ++rr) {
                const int i = r0 - 1 + rr;
                int sumF = 0, sumRF = 0;
                if (in_cm_decouple(i, jj)) {
                    const Decoupled d = decouple(i, jj);
                    sumF = d.csfF[0] + d.csfF[1] + d.csfF[2];
                    sumRF = d.csfRF[0] + d.csfRF[1] + d.csfRF[2];
                }
                gSumF[rr * SPAN + at] = sumF;
                gSumRF[rr * SPAN + at] = sumRF;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (j >= CM_LEFT && j < CM_RIGHT) {
            [unroll] for (int r = 0; r < ROWS; ++r) {
                const int i = r0 + r;
                if (i < CM_TOP || i >= CM_BOTTOM)
                    continue;
                int thresholdA = centerA[r], thresholdR = centerR[r];
                for (int di = 0; di < 3; ++di) {
                    for (int dj = 0; dj < 3; ++dj) {
                        if (di == 1 && dj == 1)
                            continue;
                        const int at = (r + di) * SPAN + int(local) + dj;
                        thresholdA += gSumF[at];
                        thresholdR += gSumRF[at];
                    }
                }
                for (int b = 0; b < 3; ++b) {
                    cm[r][b] += masked_term(restored[r][b], thresholdA, b);
                    cm[r][3 + b] += masked_term(additive[r][b], thresholdR, b);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // Each row's sums over the workgroup (wave by wave, then the waves),
    // rounded as the CPU code rounds a row's.
    const uint wave = local / WaveGetLaneCount(), waves = GROUP / WaveGetLaneCount();
    [unroll] for (int r = 0; r < ROWS; ++r) {
        [unroll] for (int s = 0; s < 6; ++s) {
            const int64_t total = WaveActiveSum(cm[r][s]);
            if (WaveIsFirstLane()) gTotals[(wave * ROWS + uint(r)) * 9 + uint(s)] = total;
        }
        for (int b = 0; b < 3; ++b) {
            const uint64_t total = WaveActiveSum(den[r][b]);
            if (WaveIsFirstLane()) gTotals[(wave * ROWS + uint(r)) * 9 + 6 + uint(b)] = int64_t(total);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local < uint(ROWS) * 9) {
        const uint r = local / 9, s = local % 9;
        const int i = r0 + int(r);
        int64_t total = 0;
        for (uint w = 0; w < waves; ++w) total += gTotals[(w * ROWS + r) * 9 + s];
        if (s < 6) {
            if (i >= CM_TOP && i < CM_BOTTOM) {
                const int64_t sum = (total + ADD_INNER) >> SHIFT_INNER;
                if (sum != 0) acc_add64(acc, (s < 3 ? SLOT_CM0 : SLOT_CM1) + s % 3, uint64_t(sum));
            }
        } else if (i >= DEN_TOP && i < DEN_BOTTOM) {
            const uint64_t sum = (uint64_t(total) + DEN_ADD_ACCUM) >> DEN_SHIFT_ACCUM;
            if (sum != 0) acc_add64(acc, SLOT_DEN + s - 6, sum);
        }
    }
}
