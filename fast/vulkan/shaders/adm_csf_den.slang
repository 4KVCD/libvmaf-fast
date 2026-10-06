// Derived from libvmaf (https://github.com/Netflix/vmaf): its CUDA feature
// extractors, Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, and for VMAF v1 its CPU code (integer_adm.c), Copyright
// 2016-2020 Netflix, Inc.; licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root
// of this repository.
//
// ADM, the denominator: adm_csf_den_scale_line_kernel (SCALE0) and
// adm_csf_den_s123_line_kernel of adm_csf_den.cu, the sum of the reference
// bands' cubes.
//
// The CUDA kernels round each warp's partial sum before adding it to the
// total, so the total depends on which columns share a warp. The same
// columns share one here: a block of 128 threads takes 1024 columns of a
// row, thread t the columns t, t + 128, ..., and threads 32k..32k+31 are a
// warp.
#include "common.slang"

struct Params {
    int top, left, right;
    int stride;
    uint shiftSq, addSq;
    uint shiftCub, addCub;
    uint shiftAccum, addAccum;
    uint slot;
};
[[vk::push_constant]] Params pc;
// The parameters as specialization constants (constant k: Params' word k):
// the engine compiles the shader for its passes' (a pipeline for each set),
// and the compiler folds what they decide.
#if SCALE0
[vk::constant_id(0)] const int kSpec_top = 0;
#define P_top kSpec_top
[vk::constant_id(1)] const int kSpec_left = 0;
#define P_left kSpec_left
[vk::constant_id(2)] const int kSpec_right = 0;
#define P_right kSpec_right
[vk::constant_id(3)] const int kSpec_stride = 0;
#define P_stride kSpec_stride
[vk::constant_id(4)] const uint kSpec_shiftSq = 0;
#define P_shiftSq kSpec_shiftSq
[vk::constant_id(5)] const uint kSpec_addSq = 0;
#define P_addSq kSpec_addSq
[vk::constant_id(6)] const uint kSpec_shiftCub = 0;
#define P_shiftCub kSpec_shiftCub
[vk::constant_id(7)] const uint kSpec_addCub = 0;
#define P_addCub kSpec_addCub
[vk::constant_id(8)] const uint kSpec_shiftAccum = 0;
#define P_shiftAccum kSpec_shiftAccum
[vk::constant_id(9)] const uint kSpec_addAccum = 0;
#define P_addAccum kSpec_addAccum
[vk::constant_id(10)] const uint kSpec_slot = 0;
#define P_slot kSpec_slot
#else
#define P_top pc.top
#define P_left pc.left
#define P_right pc.right
#define P_stride pc.stride
#define P_shiftSq pc.shiftSq
#define P_addSq pc.addSq
#define P_shiftCub pc.shiftCub
#define P_addCub pc.addCub
#define P_shiftAccum pc.shiftAccum
#define P_addAccum pc.addAccum
#define P_slot pc.slot
#endif
[[vk::binding(0, 0)]] StructuredBuffer<uint> refBands;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> acc;

groupshared uint64_t gSums[128 * 3];
#if !ROWWISE
// Each warp's sums in eight parts a band: [warp][band][part].
groupshared uint64_t gParts[4 * 3 * 8];
#endif

#if ROWWISE
// VMAF v1 follows libvmaf's CPU code (adm_csf_den_scale, adm_csf_den_s123 of
// integer_adm.c), which rounds each row's sum, not each warp's: one block of
// threads takes a row.
[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    int row = (P_top + int(group.y)) * P_stride;
    uint64_t sums[3] = { 0, 0, 0 };
    for (int j = P_left + int(local); j < P_right; j += 128) {
        int bands[3];
        adm_hvd(refBands, row + j, SCALE0 != 0, bands);
        for (int b = 0; b < 3; ++b) {
            int value = bands[b];
#if SCALE0
            uint64_t t = uint64_t(uint(abs(value)) & 0xFFFFu);
            sums[b] += t * t * t;
#else
            uint64_t t = uint64_t(uint(abs(value)));
            sums[b] += ((((t * t + P_addSq) >> P_shiftSq) * t) + P_addCub) >> P_shiftCub;
#endif
        }
    }
    for (int b = 0; b < 3; ++b)
        gSums[local * 3 + b] = sums[b];
    GroupMemoryBarrierWithGroupSync();
    if ((local & 15) == 0) {
        for (int b = 0; b < 3; ++b) {
            uint64_t sum = 0;
            for (uint t = 0; t < 16; ++t)
                sum += gSums[(local + t) * 3 + b];
            gSums[local * 3 + b] = sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        for (int b = 0; b < 3; ++b) {
            uint64_t sum = 0;
            for (uint t = 0; t < 128; t += 16)
                sum += gSums[t * 3 + b];
            sum = (sum + P_addAccum) >> P_shiftAccum;
            if (sum != 0)
                acc_add64(acc, P_slot + uint(b), sum);
        }
    }
}
#else
[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    int row = (P_top + int(group.y)) * P_stride;
    int j = P_left + 1024 * int(group.x) + int(local);
    uint64_t sums[3] = { 0, 0, 0 };
    for (int i = 0; i < 8; ++i) {
        if (j < P_right) {
            int bands[3];
            adm_hvd(refBands, row + j, SCALE0 != 0, bands);
            for (int b = 0; b < 3; ++b) {
                int value = bands[b];
#if SCALE0
                uint64_t t = uint64_t(uint(abs(value)) & 0xFFFFu);
                sums[b] += t * t * t;
#else
                uint64_t t = uint64_t(uint(abs(value)));
                sums[b] += ((((t * t + P_addSq) >> P_shiftSq) * t) + P_addCub) >> P_shiftCub;
#endif
            }
        }
        j += 128;
    }
    for (int b = 0; b < 3; ++b)
        gSums[local * 3 + b] = sums[b];
    GroupMemoryBarrierWithGroupSync();
    // A warp's 32 threads' sums of a band added up in eight parts of four
    // (24 of its threads, a part each), then the parts (three threads, a
    // band each): the warp's sum, whatever the order, rounded as before.
    const uint warp = local / 32, lane = local % 32;
    if (lane < 24) {
        const uint b = lane / 8, part = lane % 8, first = warp * 32 + part * 4;
        uint64_t sum = 0;
        for (uint t = 0; t < 4; ++t)
            sum += gSums[(first + t) * 3 + b];
        gParts[(warp * 3 + b) * 8 + part] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < 3) {
        uint64_t sum = 0;
        for (uint part = 0; part < 8; ++part)
            sum += gParts[(warp * 3 + lane) * 8 + part];
        sum = (sum + P_addAccum) >> P_shiftAccum;
        if (sum != 0)
            acc_add64(acc, P_slot + lane, sum);
    }
}
#endif
