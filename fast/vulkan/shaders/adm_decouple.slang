// Derived from libvmaf (https://github.com/Netflix/vmaf): its CUDA feature
// extractors, Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, and for VMAF v1 its CPU code (integer_adm.c), Copyright
// 2016-2020 Netflix, Inc.; licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root
// of this repository.
//
// ADM, decoupling and contrast sensitivity in one pass. SCALE0:
// adm_decouple_kernel (adm_decouple.cu) and adm_csf_kernel (adm_csf.cu) on
// 16-bit bands; otherwise adm_decouple_s123_kernel and i4_adm_csf_kernel.
// The CUDA code keeps the additive image (decouple_a) only to filter it in
// its csf kernel; here it is filtered where it is calculated.
//
// The enhancement gain limit is an integer here (VMAF's 100, VMAF NEG's 1);
// the CUDA code multiplies by it as a double and converts back, saturating.
#include "common.slang"

struct Params {
    int top, bottom, left, right;
    int inStride, outStride;
    int gainLimit;
#if BOTH
    int gainLimitB;    // VMAF NEG's, its outputs in restoredB, csfAB, csfFB
#endif
    uint rfactor0, rfactor1, rfactor2;  // scalars: Intel's driver reads an array here as zeros
#if V1
    uint cosMantissa;  // the CPU code's cos_1deg_sq as a float: mantissa * 2^-24
#endif
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> refBands;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disBands;
[[vk::binding(2, 0)]] RWStructuredBuffer<int> restored;  // decouple_r
[[vk::binding(3, 0)]] RWStructuredBuffer<int> csfA;
[[vk::binding(4, 0)]] RWStructuredBuffer<int> csfF;
// (int32_t)(2^30 / (float)(i - 32768)), from the host: the CUDA kernel's
// float division, which is correctly rounded there and on the host alike.
[[vk::binding(5, 0)]] StructuredBuffer<uint> divTable;
#if V1
// VMAF v1 (libvmaf's CPU integer_adm.c with ADM3's additive impairment
// measure): the additive image itself, and the restored image through the
// same contrast sensitivity filter, which masks it.
[[vk::binding(6, 0)]] RWStructuredBuffer<int> additiveOut;
[[vk::binding(7, 0)]] RWStructuredBuffer<int> csfR;
[[vk::binding(8, 0)]] RWStructuredBuffer<int> csfRF;
#define COS_MANTISSA pc.cosMantissa
#else
#define COS_MANTISSA 0xFFEC0Au
#endif
#if BOTH
// VMAF's and VMAF NEG's outputs from one pass: the two differ only in the
// gain limit, which changes nothing but where the vectors are within one
// degree (aligned). The bands, the angle and k are read and made once.
[[vk::binding(6, 0)]] RWStructuredBuffer<int> restoredB;
[[vk::binding(7, 0)]] RWStructuredBuffer<int> csfAB;
[[vk::binding(8, 0)]] RWStructuredBuffer<int> csfFB;
#define LIMITS 2
#else
#define LIMITS 1
#endif

#include "adm_decouple_common.slang"

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int i = pc.top + int(id.y), j = pc.left + int(id.x);
    if (i >= pc.bottom || j >= pc.right)
        return;
    int from = i * pc.inStride + j, to = (i * pc.outStride + j) * 4;
#if SCALE0 && VARIANT < 2
    // The outputs, stored packed once the three bands are done.
    int outRestored[LIMITS][3], outA[LIMITS][3], outF[LIMITS][3];
#if V1
    int outAdditive[3], outR[3], outRF[3];
#endif
#endif
    int o[3], t[3];
    adm_hvd(refBands, from, SCALE0 != 0, o);
    adm_hvd(disBands, from, SCALE0 != 0, t);
    int64_t otDp = int64_t(o[0]) * t[0] + int64_t(o[1]) * t[1];
    int64_t oMagSq = int64_t(o[0]) * o[0] + int64_t(o[1]) * o[1];
    int64_t tMagSq = int64_t(t[0]) * t[0] + int64_t(t[1]) * t[1];
    bool aligned = within_one_degree(otDp, oMagSq, tMagSq);
#if BOTH
    const int64_t limits[LIMITS] = { aligned ? int64_t(pc.gainLimit) : int64_t(1),
                                     aligned ? int64_t(pc.gainLimitB) : int64_t(1) };
#else
    const int64_t limits[LIMITS] = { aligned ? int64_t(pc.gainLimit) : int64_t(1) };
#endif
    const int64_t limit = limits[0];
    const uint rfactor[3] = { pc.rfactor0, pc.rfactor1, pc.rfactor2 };

    for (int b = 0; b < 3; ++b) {
        int additive;
#if SCALE0
        int k = 32768;
#if VARIANT == 1
        // As the CUDA kernel writes it, in 64 bits: what VARIANT 0 was until
        // 2026-10. Kept for fast/tests/diagnose_vmaf_vulkan.py.
        if (o[b] != 0)
            k = clamp(int((int64_t(int(divTable[o[b] + 32768])) * t[b] + 16384) >> 15), 0, 32768);
        int product = k * o[b];
        int rest = saturate32(int64_t((product + 16384) >> 15) * limit);
#else
#if VARIANT == 0
        int quotient = reciprocal(o[b]);
        // k = clamp((quotient * t + 16384) >> 15, 0, 32768) and
        // rest = ((k * o + 16384) >> 15) * limit, in 32-bit steps, which give
        // what the CUDA kernel's 64-bit expressions (VARIANT 1) give. With
        // quotient = high * 2^15 + low, low in 0..32767:
        //   (quotient * t + 16384) >> 15 = high * t + ((low * t + 16384) >> 15)
        // exactly, and neither term passes 2^30: the bands are 16-bit here.
        // rest is at most 32768 * 100, so nothing saturates.
        if (o[b] != 0)
            k = clamp((quotient >> 15) * t[b] + (((quotient & 32767) * t[b] + 16384) >> 15), 0, 32768);
        int product = k * o[b];
        int restOf[LIMITS];
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v)
            restOf[v] = ((product + 16384) >> 15) * int(limits[v]);
        int rest = restOf[0];
#else
        // The table read at o + 32768, as it was: what a Radeon 780M's
        // driver reads as 0 for a negative o (see reciprocal).
        int quotient = 0;
        if (o[b] != 0)
            quotient = int(divTable[o[b] + 32768]);
        if (o[b] != 0)
            k = clamp(int((int64_t(quotient) * t[b] + 16384) >> 15), 0, 32768);
        int product = k * o[b];
        int rest = saturate32(int64_t((product + 16384) >> 15) * limit);
#endif
#endif
#if VARIANT == 0
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
            if (aligned) {
                if (product > 0) restOf[v] = min(restOf[v], t[b]);
                if (product < 0) restOf[v] = max(restOf[v], t[b]);
            }
        }
        rest = restOf[0];
#else
        if (aligned) {
            if (product > 0) rest = min(rest, t[b]);
            if (product < 0) rest = max(rest, t[b]);
        }
#endif
#if VARIANT >= 2
        // For fast/tests/diagnose_vmaf_vulkan.py: one step of the above in
        // place of the result, to see which a driver gets wrong.
        int64_t wide = int64_t(quotient) * t[b] + 16384;
        int shown[6] = { quotient, k, int(wide >> 15), int(uint(wide)), int(uint(wide >> 32)), product };
        restored[to + b] = shown[VARIANT - 2];
#else
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
#if VARIANT == 0
            const int restV = restOf[v];
#else
            const int restV = rest;
#endif
            outRestored[v][b] = sext16(restV);
            const int additiveV = sext16(t[b] - restV);
            int filtered = sext16((int(rfactor[b] * uint(additiveV)) + kAdd[b]) >> kShift[b]);
            outA[v][b] = filtered;
            outF[v][b] = sext16((4369 * abs(filtered) + 2048) >> 12);
            if (v == 0)
                additive = additiveV;
        }
#if V1
        outAdditive[b] = additive;
        int filteredR = sext16((int(rfactor[b] * uint(sext16(rest))) + kAdd[b]) >> kShift[b]);
        outR[b] = filteredR;
        outRF[b] = sext16((4369 * abs(filteredR) + 2048) >> 12);
#endif
#endif
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
        int rest = 0;
        [ForceUnroll]
        for (int v = 0; v < LIMITS; ++v) {
            int restV = saturate32(int64_t(int((k * o[b] + 16384) >> 15)) * limits[v]);
            if (aligned && k > 0) {
                if (o[b] > 0) restV = min(restV, t[b]);
                if (o[b] < 0) restV = max(restV, t[b]);
            }
            const int additiveV = t[b] - restV;
            const int filtered = int((int64_t(rfactor[b]) * int64_t(additiveV) + 134217728) >> 28);
            // The CUDA kernel's rounding term is (int32_t)(1u << 31): negative.
            const int magnitude = int((int64_t(143165577) * abs(filtered) - 2147483648ll) >> 32);
            if (v == 0) {
                restored[to + b] = restV;
                csfA[to + b] = filtered;
                csfF[to + b] = magnitude;
                rest = restV;
                additive = additiveV;
            }
#if BOTH
            else {
                restoredB[to + b] = restV;
                csfAB[to + b] = filtered;
                csfFB[to + b] = magnitude;
            }
#endif
        }
#if V1
        additiveOut[to + b] = additive;
        int filteredR = int((int64_t(rfactor[b]) * int64_t(rest) + 134217728) >> 28);
        csfR[to + b] = filteredR;
        csfRF[to + b] = int((int64_t(143165577) * abs(filteredR) - 2147483648ll) >> 32);
#endif
#endif
    }
#if SCALE0 && VARIANT < 2
    int pixel = i * pc.outStride + j;
    store_packed(restored, pixel, outRestored[0][0], outRestored[0][1], outRestored[0][2]);
    store_packed(csfA, pixel, outA[0][0], outA[0][1], outA[0][2]);
    store_packed(csfF, pixel, outF[0][0], outF[0][1], outF[0][2]);
#if BOTH
    store_packed(restoredB, pixel, outRestored[1][0], outRestored[1][1], outRestored[1][2]);
    store_packed(csfAB, pixel, outA[1][0], outA[1][1], outA[1][2]);
    store_packed(csfFB, pixel, outF[1][0], outF[1][1], outF[1][2]);
#endif
#if V1
    store_packed(additiveOut, pixel, outAdditive[0], outAdditive[1], outAdditive[2]);
    store_packed(csfR, pixel, outR[0], outR[1], outR[2]);
    store_packed(csfRF, pixel, outRF[0], outRF[1], outRF[2]);
#endif
#endif
}
