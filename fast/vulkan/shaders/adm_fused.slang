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
//
// FROM_PICTURE (scale 0): the bands are not read but calculated here from the
// two pictures, as adm_dwt.slang's SCALE0 calculates them, for each pixel
// the pass reads; its workgroups cover the whole band image, and write the
// approximation bands (a) for the next scale's transform, one int32 a
// position (adm_dwt.slang's A_PLANE reads them).
#include "common.slang"

[[vk::push_constant]] uniform uint pc[4];  // the scale's first word in params
#if FROM_PICTURE
[[vk::binding(0, 0)]] StructuredBuffer<uint> refPicture;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disPicture;
#else
[[vk::binding(0, 0)]] StructuredBuffer<uint> refBands;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disBands;
#endif
[[vk::binding(2, 0)]] StructuredBuffer<uint> divTable;
[[vk::binding(3, 0)]] StructuredBuffer<int> params;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> acc;
#if FROM_PICTURE
[[vk::binding(5, 0)]] RWStructuredBuffer<int> refApprox;
[[vk::binding(6, 0)]] RWStructuredBuffer<int> disApprox;
#endif

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
// FROM_PICTURE: adm_dwt.slang's SCALE0 parameters, and the approximation bands' row stride.
#define PIC_W P(44)
#define PIC_H P(45)
#define PIC_STRIDE P(46)       // words per picture row
#define SHIFT_V P(47)
#define ADD_V P(48)
#define SHIFT_H P(49)
#define ADD_H P(50)
#define APPROX_STRIDE P(51)

#include "adm_decouple_common.slang"

#define ROWS 2
#define GROUP 128
#define SPAN (GROUP + 2)
#if SLIDE || ROWSUM
// SLIDE's sums of a row's columns, per row and workgroup (ROWSUM adds them).
#if FROM_PICTURE
[[vk::binding(7, 0)]] RWStructuredBuffer<int64_t> partial;
#else
[[vk::binding(5, 0)]] RWStructuredBuffer<int64_t> partial;
#endif
#else
// Per pixel of the rows and their neighbours: the bands' csfF summed, and csfRF's.
groupshared int gSumF[(ROWS + 2) * SPAN];
groupshared int gSumRF[(ROWS + 2) * SPAN];
groupshared int64_t gTotals[(GROUP / 32) * ROWS * 9];
#endif

// A pixel's decoupling (adm_decouple.slang with V1 and VARIANT 0): the
// restored and additive values of its three bands, and per band
// csfA / csfF of the additive and csfR / csfRF of the restored image.
struct Decoupled {
    int restored[3], additive[3], csfA[3], csfF[3], csfR[3], csfRF[3];
};

#if FROM_PICTURE
static const int kLo[4] = { 15826, 27411, 7345, -4240 };
static const int kHi[4] = { -4240, -7345, 27411, -15826 };
static const int kLoSum = 46342;

// adm_dwt.slang's calculate_indices
int4 indices(int n, int limit)
{
    int4 at = int4(2 * n - 1, 2 * n, 2 * n + 1, 2 * n + 2);
    if (n == 0)
        at = int4(1, 0, 1, 2);
    if (at.x >= limit) at.x = 2 * limit - at.x - 1;
    if (at.y >= limit) at.y = 2 * limit - at.y - 1;
    if (at.z >= limit) at.z = 2 * limit - at.z - 1;
    if (at.w >= limit) at.w = 2 * limit - at.w - 1;
    return at;
}

// adm_dwt.slang's SCALE0 transform at band position (i, j): a, and h, v, d
// in bands[0..2].
int transform(StructuredBuffer<uint> source, int i, int j, out int bands[3])
{
    const int4 columns = indices(j, PIC_W), rows = indices(i, PIC_H);
    int lo[4], hi[4];
    for (int c = 0; c < 4; ++c) {
        int sumLo = 0, sumHi = 0;
        for (int k = 0; k < 4; ++k) {
            const int sample = int(pixel(source, columns[c], rows[k], PIC_STRIDE));
            sumLo += kLo[k] * sample;
            sumHi += kHi[k] * sample;
        }
        sumLo -= kLoSum * ADD_V;
        lo[c] = sext16((sumLo + ADD_V) >> SHIFT_V);
        hi[c] = sext16((sumHi + ADD_V) >> SHIFT_V);
    }
    int a = 0, v = 0, h = 0, d = 0;
    for (int c = 0; c < 4; ++c) {
        a += kLo[c] * lo[c];
        v += kHi[c] * lo[c];
        h += kLo[c] * hi[c];
        d += kHi[c] * hi[c];
    }
    bands[0] = sext16((h + ADD_H) >> SHIFT_H);
    bands[1] = sext16((v + ADD_H) >> SHIFT_H);
    bands[2] = sext16((d + ADD_H) >> SHIFT_H);
    return (a + ADD_H) >> SHIFT_H;
}
#endif

// A pixel's three bands of each picture (h, v, d): read, or calculated.
void load_bands(int i, int j, out int o[3], out int t[3])
{
#if FROM_PICTURE
    transform(refPicture, i, j, o);
    transform(disPicture, i, j, t);
#else
    const int from = (i * IN_STRIDE + j) * 4;
    for (int b = 0; b < 3; ++b) {
        o[b] = int(refBands[from + 1 + b]);
        t[b] = int(disBands[from + 1 + b]);
    }
#endif
}

Decoupled decouple_bands(int o[3], int t[3])
{
    Decoupled out;
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

Decoupled decouple(int i, int j)
{
    int o[3], t[3];
    load_bands(i, j, o, t);
    return decouple_bands(o, t);
}

bool in_cm_decouple(int i, int j)  // decoupled for the masking: its region and a pixel around it
{
    return i >= CM_TOP - 1 && i <= CM_BOTTOM && j >= CM_LEFT - 1 && j <= CM_RIGHT;
}

#if SLIDE
// The workgroups slide down the band: 128 threads, a column each -- 126 of
// the workgroup's own and one either side, which the masking reads -- over
// BAND rows. Each row is decoupled once (the rows either side of the band
// once more), and of the three rows the masking reads only what the
// threshold sums is kept, a ring of three in shared memory. Each row's sums
// of the workgroup's columns go to `partial`; ROWSUM adds each row's columns
// and rounds the row as the CPU code does.
#define BAND 32
#define OWN (GROUP - 2)
groupshared int gRingF[3 * GROUP];
groupshared int gRingRF[3 * GROUP];
groupshared int64_t gRow[(GROUP / 32) * 9];

// Row i at the thread's column j: decoupled into the ring, and (`center`)
// what the masking of row i itself needs, and its denominator's terms.
void slide_row(int i, int j, uint local, bool own, inout int restored[3], inout int additive[3],
               inout int centerA, inout int centerR, inout uint64_t den[3])
{
    int sumF = 0, sumRF = 0;
    const bool inDen = own && i >= DEN_TOP && i < DEN_BOTTOM && j >= DEN_LEFT && j < DEN_RIGHT;
    for (int b = 0; b < 3; ++b) den[b] = 0;
#if FROM_PICTURE
    const bool approx = own && i >= 0 && i < BH && j >= 0 && j < BW;
    int o[3], t[3];
    if (approx || inDen || in_cm_decouple(i, j)) {
        const int aRef = transform(refPicture, i, j, o), aDis = transform(disPicture, i, j, t);
        if (approx) {
            refApprox[i * APPROX_STRIDE + j] = aRef;
            disApprox[i * APPROX_STRIDE + j] = aDis;
        }
    }
    if (inDen) {
        for (int b = 0; b < 3; ++b)
            den[b] = den_term(o[b]);
    }
#else
    if (inDen) {
        const int from = (i * IN_STRIDE + j) * 4;
        for (int b = 0; b < 3; ++b)
            den[b] = den_term(int(refBands[from + 1 + b]));
    }
#endif
    if (in_cm_decouple(i, j)) {
#if FROM_PICTURE
        const Decoupled d = decouple_bands(o, t);
#else
        const Decoupled d = decouple(i, j);
#endif
        sumF = d.csfF[0] + d.csfF[1] + d.csfF[2];
        sumRF = d.csfRF[0] + d.csfRF[1] + d.csfRF[2];
        centerA = center_term(abs(d.csfA[0])) + center_term(abs(d.csfA[1])) + center_term(abs(d.csfA[2]));
        centerR = center_term(abs(d.csfR[0])) + center_term(abs(d.csfR[1])) + center_term(abs(d.csfR[2]));
        for (int b = 0; b < 3; ++b) {
            restored[b] = d.restored[b];
            additive[b] = d.additive[b];
        }
    }
    const uint ring = uint((i + 3) % 3);
    gRingF[ring * GROUP + local] = sumF;
    gRingRF[ring * GROUP + local] = sumRF;
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
#if FROM_PICTURE
    const int colStart = 0;  // the approximation bands, all of them
#else
    const int colStart = min(CM_LEFT, DEN_LEFT);
#endif
    const int j = colStart + int(group.x) * OWN - 1 + int(local);
    const bool ownColumn = local >= 1 && local <= OWN;
    const int first = FIRST_ROW + int(group.y) * BAND, end = min(first + BAND, LAST_ROW);
    int restored[3] = { 0, 0, 0 }, additive[3] = { 0, 0, 0 }, centerA = 0, centerR = 0;
    int nextRestored[3] = { 0, 0, 0 }, nextAdditive[3] = { 0, 0, 0 }, nextA = 0, nextR = 0;
    uint64_t den[3], nextDen[3];
    slide_row(first - 1, j, local, false, nextRestored, nextAdditive, nextA, nextR, nextDen);
    slide_row(first, j, local, ownColumn, restored, additive, centerA, centerR, den);
    const uint wave = local / WaveGetLaneCount(), waves = GROUP / WaveGetLaneCount();
    for (int i = first; i < end; ++i) {
        slide_row(i + 1, j, local, ownColumn && i + 1 < end, nextRestored, nextAdditive, nextA, nextR, nextDen);
        GroupMemoryBarrierWithGroupSync();
        int64_t cm[6] = { 0, 0, 0, 0, 0, 0 };
        if (ownColumn && j >= CM_LEFT && j < CM_RIGHT && i >= CM_TOP && i < CM_BOTTOM) {
            int thresholdA = centerA, thresholdR = centerR;
            for (int di = -1; di <= 1; ++di) {
                const uint ring = uint((i + di + 3) % 3) * GROUP;
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0)
                        continue;
                    thresholdA += gRingF[ring + uint(int(local) + dj)];
                    thresholdR += gRingRF[ring + uint(int(local) + dj)];
                }
            }
            for (int b = 0; b < 3; ++b) {
                cm[b] = masked_term(restored[b], thresholdA, b);
                cm[3 + b] = masked_term(additive[b], thresholdR, b);
            }
        }
        // The row's sums of these columns, wave by wave, then the waves, to `partial`.
        [unroll] for (int s = 0; s < 6; ++s) {
            const int64_t total = WaveActiveSum(cm[s]);
            if (WaveIsFirstLane()) gRow[wave * 9 + uint(s)] = total;
        }
        [unroll] for (int b = 0; b < 3; ++b) {
            const int64_t total = int64_t(WaveActiveSum(ownColumn ? den[b] : 0ull));
            if (WaveIsFirstLane()) gRow[wave * 9 + 6 + uint(b)] = total;
        }
        GroupMemoryBarrierWithGroupSync();
        if (local < 9) {
            int64_t total = 0;
            for (uint w = 0; w < waves; ++w) total += gRow[w * 9 + local];
            partial[(uint(i - FIRST_ROW) * pc[1] + group.x) * 9 + local] = total;
        }
        restored = nextRestored;
        additive = nextAdditive;
        centerA = nextA;
        centerR = nextR;
        den = nextDen;
    }
}
#elif ROWSUM
// Each row's sums: its columns' (adm_fused's SLIDE), added and rounded as the
// CPU code rounds a row's, then added to the frame's slots.
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint rows = uint(LAST_ROW - FIRST_ROW);
    if (id.x >= rows * 9) return;
    const uint r = id.x / 9, s = id.x % 9;
    const int i = FIRST_ROW + int(r);
    int64_t total = 0;
    for (uint chunk = 0; chunk < pc[1]; ++chunk) total += partial[(r * pc[1] + chunk) * 9 + s];
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
#else
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const int r0 = FIRST_ROW + int(group.y) * ROWS;
#if FROM_PICTURE
    const int colStart = 0, colEnd = BW;  // the approximation bands, all of them
#else
    const int colStart = min(CM_LEFT, DEN_LEFT), colEnd = max(CM_RIGHT, DEN_RIGHT);
#endif
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
            const bool own = rr >= 1 && rr <= ROWS;
            const bool inDen = own && i >= DEN_TOP && i < DEN_BOTTOM && j >= DEN_LEFT && j < DEN_RIGHT;
#if FROM_PICTURE
            // The transform once for all three uses: the approximation bands
            // (the band image's every position), the denominator, the decoupling.
            const bool approx = own && i < BH && j < BW;
            int o[3], t[3];
            if (approx || inDen || in_cm_decouple(i, j)) {
                const int aRef = transform(refPicture, i, j, o), aDis = transform(disPicture, i, j, t);
                if (approx) {
                    refApprox[i * APPROX_STRIDE + j] = aRef;
                    disApprox[i * APPROX_STRIDE + j] = aDis;
                }
            }
            if (inDen) {
                for (int b = 0; b < 3; ++b)
                    den[rr - 1][b] += den_term(o[b]);
            }
#endif
            if (in_cm_decouple(i, j)) {
#if FROM_PICTURE
                const Decoupled d = decouple_bands(o, t);
#else
                const Decoupled d = decouple(i, j);
#endif
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
#if !FROM_PICTURE
            if (inDen) {
                const int from = (i * IN_STRIDE + j) * 4;
                for (int b = 0; b < 3; ++b)
                    den[rr - 1][b] += den_term(int(refBands[from + 1 + b]));
            }
#endif
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
#endif
