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
// the pass reads; its workgroups cover the whole band image, and do scale
// 1's wavelet transform of the approximation bands (a) too, as adm_dwt.slang
// does it, writing scale 1's bands (slide_scale1).
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
[[vk::binding(5, 0)]] RWStructuredBuffer<int> refBands1;  // scale 1's bands: a, h, v, d a position
[[vk::binding(6, 0)]] RWStructuredBuffer<int> disBands1;
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
#define BANDS1_STRIDE P(51)   // scale 1's bands' positions a row
#define SHIFT_V1 P(52)         // scale 1's transform (adm_dwt.slang's, not SCALE0)
#define ADD_V1 P(53)
#define SHIFT_H1 P(54)
#define ADD_H1 P(55)

#include "adm_decouple_common.slang"

#define GROUP 128
// SLIDE's sums of a row's columns, per row and workgroup (ROWSUM adds them).
#if FROM_PICTURE
[[vk::binding(7, 0)]] RWStructuredBuffer<int64_t> partial;
#else
[[vk::binding(5, 0)]] RWStructuredBuffer<int64_t> partial;
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

// Where the masking's neighbour at row (or column) n is, at the band's edges
// (which its region reaches when the border is 0, in small pictures): as the
// CPU code's ADM_CM_THRESH_S_0_J, _H_M_1_J, _I_0 and _I_W_M_1, the one before
// the first is the second, the one past the last is the last.
int cm_neighbour(int n, int size)
{
    return n == -1 ? 1 : (n == size ? size - 1 : n);
}

// A sum over the subgroup's lanes: in 32 bits when every lane's value is
// small enough for the sum to fit (nearly always), else in 64. Exact either way.
int64_t wave_sum64(int64_t v)
{
    const int64_t limit = int64_t(0x80000000u / WaveGetLaneCount());
    if (WaveActiveAllTrue(v >= 0 && v < limit))
        return int64_t(WaveActiveSum(uint(v)));
    return WaveActiveSum(v);
}

#if SLIDE
// The workgroups slide down the band: 128 threads, a column each -- 126 of
// the workgroup's own and one either side, which the masking reads -- over
// BAND rows. Each row is decoupled once (the rows either side of the band
// once more), and of the three rows the masking reads only what the
// threshold sums is kept, a ring of three in shared memory. Each row's sums
// of the workgroup's columns go to `partial`; ROWSUM adds each row's columns
// and rounds the row as the CPU code does.
#ifndef BAND
#define BAND 32
#endif
#define OWN (GROUP - 2)
groupshared int gRingF[3 * GROUP];
groupshared int gRingRF[3 * GROUP];
groupshared int64_t gRow[(GROUP / 32) * 9];

// Row i at the thread's column j: decoupled into the ring, and (`center`)
// what the masking of row i itself needs, and its denominator's terms.
void slide_row(int i, int j, uint local, bool own, inout int restored[3], inout int additive[3],
               inout int centerA, inout int centerR, inout uint64_t den[3], out int aRef, out int aDis)
{
    aRef = 0;
    aDis = 0;
    int sumF = 0, sumRF = 0;
    const bool inDen = own && i >= DEN_TOP && i < DEN_BOTTOM && j >= DEN_LEFT && j < DEN_RIGHT;
    for (int b = 0; b < 3; ++b) den[b] = 0;
    // Outside the band (a neighbour only): the pixel the masking reads instead.
    const int ci = cm_neighbour(i, BH), cj = cm_neighbour(j, BW);
#if FROM_PICTURE
    // Everywhere in the band (also outside the masking's region, for scale
    // 1's transform of a); rows and columns past it as the masking reads them.
    int o[3] = { 0, 0, 0 }, t[3] = { 0, 0, 0 };
    if (cj >= 0 && cj < BW) {
        aRef = transform(refPicture, ci, cj, o);
        aDis = transform(disPicture, ci, cj, t);
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
    if (in_cm_decouple(ci, cj)) {
#if FROM_PICTURE
        const Decoupled d = decouple_bands(o, t);
#else
        const Decoupled d = decouple(ci, cj);
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

#if FROM_PICTURE
groupshared int gScale1[4 * GROUP];  // per column: scale 1's vertical lo and hi, of each picture

// The band row slide step k gives (k = -1 and BH as the masking mirrors them).
int step_row(int k)
{
    return k == -1 ? 1 : (k == BH ? BH - 1 : k);
}

// Scale 1's vertical transform of the thread's column for its row I (rows
// indices(I, BH) of a, all among the last four steps' -- s - 3 .. s, a[0..3]),
// into gScale1 (before the barrier its horizontal transform waits for).
void scale1_vertical(int I, int s, int aRef[4], int aDis[4], uint local)
{
    const int4 rows = indices(I, BH);
    int64_t lo[2] = { 0, 0 }, hi[2] = { 0, 0 };
    [unroll] for (int m = 0; m < 4; ++m) {
        const int r = rows[m];
        int pick = 3;
        [unroll] for (int k = 2; k >= 0; --k)
            if (step_row(s - 3 + k) == r) pick = k;
        const int vr = pick == 0 ? aRef[0] : pick == 1 ? aRef[1] : pick == 2 ? aRef[2] : aRef[3];
        const int vd = pick == 0 ? aDis[0] : pick == 1 ? aDis[1] : pick == 2 ? aDis[2] : aDis[3];
        lo[0] += int64_t(kLo[m]) * int64_t(vr);
        hi[0] += int64_t(kHi[m]) * int64_t(vr);
        lo[1] += int64_t(kLo[m]) * int64_t(vd);
        hi[1] += int64_t(kHi[m]) * int64_t(vd);
    }
    gScale1[0 * GROUP + local] = int((lo[0] + ADD_V1) >> SHIFT_V1);
    gScale1[1 * GROUP + local] = int((hi[0] + ADD_V1) >> SHIFT_V1);
    gScale1[2 * GROUP + local] = int((lo[1] + ADD_V1) >> SHIFT_V1);
    gScale1[3 * GROUP + local] = int((hi[1] + ADD_V1) >> SHIFT_V1);
}

// Scale 1's horizontal transform at (I, J) (J's columns among the
// workgroup's, its first one c0 - 1 at local 0), its bands written.
void scale1_horizontal(int I, int J, int c0)
{
    const int4 columns = indices(J, BW);
    [unroll] for (int p = 0; p < 2; ++p) {
        int64_t a = 0, v = 0, h = 0, d = 0;
        [unroll] for (int c = 0; c < 4; ++c) {
            const uint at = uint(columns[c] - c0 + 1);
            const int lo = gScale1[uint(2 * p) * GROUP + at], hi = gScale1[uint(2 * p + 1) * GROUP + at];
            a += int64_t(kLo[c]) * lo;
            v += int64_t(kHi[c]) * lo;
            h += int64_t(kLo[c]) * hi;
            d += int64_t(kHi[c]) * hi;
        }
        const int out = (I * BANDS1_STRIDE + J) * 4;
        const int4 bands = int4(int((a + ADD_H1) >> SHIFT_H1), int((h + ADD_H1) >> SHIFT_H1),
                                int((v + ADD_H1) >> SHIFT_H1), int((d + ADD_H1) >> SHIFT_H1));
        if (p == 0) {
            refBands1[out] = bands.x; refBands1[out + 1] = bands.y; refBands1[out + 2] = bands.z; refBands1[out + 3] = bands.w;
        } else {
            disBands1[out] = bands.x; disBands1[out + 1] = bands.y; disBands1[out + 2] = bands.z; disBands1[out + 3] = bands.w;
        }
    }
}
#endif

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
    // The thread's column's a of the last four steps (s - 3 .. s), each picture's.
    int aRef[4] = { 0, 0, 0, 0 }, aDis[4] = { 0, 0, 0, 0 };
    slide_row(first - 1, j, local, false, nextRestored, nextAdditive, nextA, nextR, nextDen, aRef[2], aDis[2]);
    slide_row(first, j, local, ownColumn, restored, additive, centerA, centerR, den, aRef[3], aDis[3]);
    const uint wave = local / WaveGetLaneCount(), waves = GROUP / WaveGetLaneCount();
    for (int i = first; i < end; ++i) {
        int newRef, newDis;
        slide_row(i + 1, j, local, ownColumn && i + 1 < end, nextRestored, nextAdditive, nextA, nextR, nextDen,
                  newRef, newDis);
#if FROM_PICTURE
        aRef[0] = aRef[1]; aRef[1] = aRef[2]; aRef[2] = aRef[3]; aRef[3] = newRef;
        aDis[0] = aDis[1]; aDis[1] = aDis[2]; aDis[2] = aDis[3]; aDis[3] = newDis;
        // Scale 1's row I once the step has given its last row of a (2I + 2;
        // the band's last step for its last row when it is odd), of the I
        // whose first row 2I is the workgroup's.
        const int s = i + 1;
        int I = -1;
        if ((s & 1) == 0 && s - 2 >= first)
            I = s / 2 - 1;
        else if (s == end && end == BH && (BH & 1) != 0)
            I = (s - 1) / 2;
        if (I >= 0)
            scale1_vertical(I, s, aRef, aDis, local);
#endif
        GroupMemoryBarrierWithGroupSync();
#if FROM_PICTURE
        if (I >= 0 && local < uint(OWN / 2)) {
            const int c0 = int(group.x) * OWN, J = c0 / 2 + int(local);
            if (J < (BW + 1) / 2)
                scale1_horizontal(I, J, c0);
        }
#endif
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
            const int64_t total = wave_sum64(cm[s]);
            if (WaveIsFirstLane()) gRow[wave * 9 + uint(s)] = total;
        }
        [unroll] for (int b = 0; b < 3; ++b) {
            // (den's terms are below 2^63: as int64_t, never negative.)
            const int64_t total = wave_sum64(ownColumn ? int64_t(den[b]) : 0);
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
#endif
