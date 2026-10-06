// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// ADM's contrast masking sums of a scale, after adm_dcm.slang: each row's sum
// rounded once, (sum + 2^(shiftInner-1)) >> shiftInner, as adm_cm.slang
// rounds it, and the rows added up -- a group for each of the two limits'
// three bands.
#include "common.slang"

struct Params {
    uint rows;          // contrast masking's rows at this scale
    int shiftInner;
    uint rowSlot;       // the scale's first row sum in rowSums
    uint slot;          // the first limit's first band's sum
    uint limitStride;   // from a limit's sums to the next one's
};
[[vk::push_constant]] Params pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> rowSums;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> acc;

#define THREADS 64
groupshared uint gTotal[2];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint local : SV_GroupIndex)
{
    const uint k = group.x;  // limit * 3 + band
    if (local < 2)
        gTotal[local] = 0;
    GroupMemoryBarrierWithGroupSync();
    const int64_t add = pc.shiftInner ? (1ll << (pc.shiftInner - 1)) : 0;
    uint64_t partial = 0;
    for (uint r = local; r < pc.rows; r += THREADS) {
        const uint at = 2 * (pc.rowSlot + k * pc.rows + r);
        const int64_t sum = int64_t(uint64_t(rowSums[at]) | (uint64_t(rowSums[at + 1]) << 32));
        partial += uint64_t((sum + add) >> pc.shiftInner);
    }
    if (partial != 0) {
        const uint lo = uint(partial);
        uint old;
        InterlockedAdd(gTotal[0], lo, old);
        const uint high = uint(partial >> 32) + (old > ~lo ? 1u : 0u);
        if (high != 0)
            InterlockedAdd(gTotal[1], high);
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        const uint64_t total = uint64_t(gTotal[0]) | (uint64_t(gTotal[1]) << 32);
        if (total != 0)
            acc_add64(acc, pc.slot + (k / 3) * pc.limitStride + k % 3, total);
    }
}
