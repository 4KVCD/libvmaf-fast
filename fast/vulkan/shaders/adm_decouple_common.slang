// Derived from libvmaf (https://github.com/Netflix/vmaf): its CUDA feature
// extractors, Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation; licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root
// of this repository.
//
// ADM decoupling's steps, shared by adm_decouple.slang and adm_fused.slang:
// the one-degree test and the division table's reciprocal. The includer
// defines COS_MANTISSA and binds divTable.

// (float)v of a non-negative integer: m * 2^e, m below 2^24.
void to_float(uint64_t v, out uint m, out int e)
{
    if (v < (1ull << 24)) {
        m = uint(v);
        e = 0;
        return;
    }
    int sh = msb64(v) - 23;
    uint64_t q = v >> sh, rem = v & ((1ull << sh) - 1), half = 1ull << (sh - 1);
    if (rem > half || (rem == half && (q & 1) != 0)) {
        q++;
        if (q == (1ull << 24)) {
            q >>= 1;
            sh++;
        }
    }
    m = uint(q);
    e = sh;
}

// A non-negative 64-bit integer as a float, to within a few units in the
// last place whatever the driver's own conversion does.
float approximately(int64_t v)
{
    return float(uint(uint64_t(v) >> 32)) * 4294967296.0 + float(uint(v));
}

// Whether the two vectors are within one degree, as the CUDA kernels decide
// it: the three integers rounded to float, then in double
//     ot >= 0 && ot * ot >= (cos_1deg_sq * o_mag_sq) * t_mag_sq
// (the common division by 4096 * 4096 changes nothing). cos_1deg_sq is the
// float 0xFFEC0A * 2^-24.
bool within_one_degree(int64_t otDp, int64_t oMagSq, int64_t tMagSq)
{
    if (otDp < 0)
        return false;
    // Nearly every vector is far from one degree: decided in float when the
    // two sides differ by more than float's error can explain, exactly
    // otherwise.
    const float ot = approximately(otDp);
    const float left_ = ot * ot;
    const float right_ = (float(COS_MANTISSA) / 16777216.0) * approximately(oMagSq) * approximately(tMagSq);
    if (left_ > right_ * 1.0001)
        return true;
    if (left_ < right_ * 0.9999)
        return false;
    uint mot, mo, mt;
    int eot, eo, et;
    to_float(uint64_t(otDp), mot, eot);
    to_float(uint64_t(oMagSq), mo, eo);
    to_float(uint64_t(tMagSq), mt, et);
    if (mo == 0 || mt == 0)
        return true;
    if (mot == 0)
        return false;
    uint64_t left = uint64_t(mot) * mot;           // exact in a double
    uint64_t right;
    int re;
    round53(mul64(uint64_t(COS_MANTISSA) * mo, uint64_t(mt)), right, re);
    re += eo + et - 24;
    int leftTop = msb64(left), topLeft = leftTop + 2 * eot, topRight = 52 + re;
    if (topLeft != topRight)
        return topLeft > topRight;
    return (left << (52 - leftTop)) >= right;
}

static const int kAdd[3] = { 16384, 16384, 65535 };
static const int kShift[3] = { 15, 15, 17 };

int saturate32(int64_t v)
{
    return int(clamp(v, int64_t(-2147483647 - 1), int64_t(2147483647)));
}

// divTable's entry for o, 2^30 / o as the CUDA kernel reads it at
// o + 32768, read at |o| + 32768 and negated for a negative o. The entry is
// a float division truncated towards zero, so the entry of -o is minus the
// entry of o exactly; -32768 has no positive twin, and its entry is -32768.
//
// A Radeon 780M's driver (32.0.21028.21 and 32.0.31041.1004 alike) reads
// divTable[o + 32768] as 0 for every o below -1, though the table in its
// memory is right (read back) and so is the SPIR-V: rest came out 0 wherever
// o was negative, and the GPU's sums failed the probe. The upper half of the
// table it reads correctly.
int reciprocal(int o)
{
    if (o == 0)
        return 0;
    if (o == -32768)
        return -32768;
    int entry = int(divTable[abs(o) + 32768]);
    return o < 0 ? -entry : entry;
}

