// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// VIF's statistics of a pixel (vif_statistic_calculation of
// vif_statistics.cuh), shared by vif_hori.slang and vif_fused.slang, which
// include it after declaring pc (gainA, gainB, epsLo, epsHi), logTable and
// SUMS.

uint log_generate(uint v) { return logTable[v & 0x7FFFu]; }

// get_best16_from64
uint best16(uint64_t v, out int x)
{
    int k = 63 - msb64(v);
    if (k > 48) {
        k -= 48;
        x = k;
        return uint(v << k) & 0xFFFFu;
    } else if (k < 47) {
        k = 48 - k;
        x = -k;
        return uint(v >> k) & 0xFFFFu;
    }
    x = 0;
    if ((v >> 16) != 0) {
        v >>= 1;
        x = -1;
    }
    return uint(v) & 0xFFFFu;
}

// g = sigma12 / (sigma1_sq + eps) as a double's mantissa and exponent:
// g = gm * 2^ge, gm in [2^52, 2^53). sigma12 > 0, sigma1_sq >= 2^17.
void gain(int sigma12, int sigma1, out uint64_t gm, out int ge)
{
#if NATIVE_F64
    double g = double(sigma12) / (double(uint(sigma1)) + asdouble(pc.epsLo, pc.epsHi));
    uint lo, hi;
    asuint(g, lo, hi);
    gm = (uint64_t(hi & 0xFFFFFu) << 32) | uint64_t(lo) | (1ull << 52);
    ge = int((hi >> 20) & 0x7FFu) - 1075;
#else
    // The divisor: sigma1_sq + eps, eps = 0x1B7CDFD9D7BDBB * 2^-70, rounded.
    int p = msb32(uint(sigma1));
    int r = 18 + p;
    const uint64_t kEps = 0x1B7CDFD9D7BDBBull;
    uint64_t q = kEps >> r, fraction = kEps & ((1ull << r) - 1), half = 1ull << (r - 1);
    uint64_t divisor = (uint64_t(sigma1) << (52 - p)) + q;
    if (fraction > half || (fraction == half && (q & 1) != 0))
        divisor++;
    int a = msb32(uint(sigma12));
    uint64_t rem = uint64_t(sigma12) << (52 - a);
    int e = a - p - 52;
    if (rem < divisor) {
        rem <<= 1;
        e -= 1;
    }
    rem -= divisor;
    uint64_t quotient = 1;
    // The other 52 bits of the quotient, 10 or 11 at a time: each group is
    // estimated in float (within one of the true value: the float error is
    // far below 1 for a group under 2^11) and then made exact from the
    // remainder. A bit at a time is the same quotient and five times slower
    // on a GPU that has no 64-bit integers of its own.
    const float reciprocal = 2048.0 / float(uint(divisor >> 29));
    for (int group = 0; group < 5; ++group) {
        const int bits = group < 2 ? 11 : 10;
        uint64_t scaled = rem << bits;  // below 2^64: rem < divisor < 2^53
        uint estimate = uint(float(uint(scaled >> 40)) * reciprocal);
        int64_t rest = int64_t(scaled - uint64_t(estimate) * divisor);
        for (int fix = 0; fix < 2; ++fix) {
            if (rest < 0) {
                estimate--;
                rest += int64_t(divisor);
            } else if (rest >= int64_t(divisor)) {
                estimate++;
                rest -= int64_t(divisor);
            }
        }
        quotient = (quotient << bits) | uint64_t(estimate);
        rem = uint64_t(rest);
    }
    uint64_t twice = rem << 1;
    if (twice > divisor || (twice == divisor && (quotient & 1) != 0)) {
        quotient++;
        if (quotient == (1ull << 53)) {
            quotient >>= 1;
            e++;
        }
    }
    gm = quotient;
    ge = e;
#endif
}

// max((int32_t)fma(-g, sigma12, sigma2_sq), 0): the exact difference rounded
// once to a double, then truncated.
int residual(int sigma2, uint64_t gm, int ge, int sigma12)
{
    U128 t = mul64(gm, uint64_t(sigma12));
    int n = -ge;  // 39..84: g < 2^14, g > 2^-32
    uint64_t whole, fracHi, fracLo;
    if (n < 64) {
        whole = (t.lo >> n) | (t.hi << (64 - n));
        fracHi = 0;
        fracLo = t.lo & ((1ull << n) - 1);
    } else if (n == 64) {
        whole = t.hi;
        fracHi = 0;
        fracLo = t.lo;
    } else {
        whole = t.hi >> (n - 64);
        fracHi = t.hi & ((1ull << (n - 64)) - 1);
        fracLo = t.lo;
    }
    if (whole >= uint64_t(sigma2))
        return 0;
    int difference = sigma2 - int(whole);
    if (fracHi == 0 && fracLo == 0)
        return difference;
    // The difference is in (floor, floor + 1): the double nearest to it is
    // floor + 1 when it is within half a unit in the last place of it.
    int floor_ = difference - 1;
    int k = n + (floor_ > 0 ? msb32(uint(floor_)) : -1) - 53;
    bool up = false;
    if (k >= 64) {
        uint64_t limit = 1ull << (k - 64);
        up = fracHi < limit || (fracHi == limit && fracLo == 0);
    } else if (k >= 0) {
        up = fracHi == 0 && fracLo <= (1ull << k);
    }
    return up ? difference : floor_;
}

// g > limit
bool above(uint64_t gm, int ge, uint limit)
{
    int top = msb32(limit);
    uint64_t km = uint64_t(limit) << (52 - top);
    int ke = top - 52;
    return ge > ke || (ge == ke && gm > km);
}

// (int64_t)(g * g * sigma1_sq) with g capped at `limit`.
int64_t gain_term(uint64_t gm, int ge, int sigma1, uint limit)
{
    if (above(gm, ge, limit))
        return int64_t(uint64_t(limit) * limit * uint64_t(sigma1));
#if NATIVE_F64
    double g = asdouble(uint(gm), (uint(gm >> 32) & 0xFFFFFu) | (uint(ge + 1075) << 20));
    return int64_t(g * g * double(uint(sigma1)));
#else
    uint64_t fm, hm;
    int fe, he;
    round53(mul64(gm, gm), fm, fe);
    round53(mul64(fm, uint64_t(sigma1)), hm, he);
    int e = he + fe + 2 * ge;
    if (e >= 0)
        return int64_t(hm << e);
    if (e <= -64)
        return 0;
    return int64_t(hm >> (-e));
#endif
}

void statistics(uint mu1, uint mu2, uint xx, uint yy, uint xy, inout int sums[SUMS])
{
    uint mu1Sq = uint((uint64_t(mu1) * mu1 + 2147483648ull) >> 32);
    uint mu2Sq = uint((uint64_t(mu2) * mu2 + 2147483648ull) >> 32);
    uint mu1Mu2 = uint((uint64_t(mu1) * mu2 + 2147483648ull) >> 32);
    int sigma1 = max(int(xx - mu1Sq), 0);
    int sigma2 = max(int(yy - mu2Sq), 0);
    int sigma12 = int(xy - mu1Mu2);
    const int kSigmaNsq = 131072;
    if (sigma1 < kSigmaNsq) {
        sums[3] += sigma2;
        sums[4] += 1;
        return;
    }
    uint denominator = uint(kSigmaNsq + sigma1);
    int k = msb32(denominator) - 15;  // 16 - clz
    sums[0] += -k;
    sums[1] += 1;
    sums[2] += int(log_generate(denominator >> k));
    if (sigma12 < 0)
        return;
    bool gained = sigma12 > 0 && sigma2 != 0;
    uint64_t gm = 0;
    int ge = 0;
    int sv = sigma2;
    if (gained) {
        gain(sigma12, sigma1, gm, ge);
        sv = residual(sigma2, gm, ge, sigma12);
    }
    uint numer1 = uint(sv + kSigmaNsq);
    int x2;
    uint denLog = log_generate(best16(uint64_t(numer1), x2));
    // The gain term is the same for both limits unless g is above the
    // lower one, and then that limit's term is just limit * limit * sigma1_sq.
    int64_t terms[2] = { 0, 0 };
    if (gained) {
        terms[0] = gain_term(gm, ge, sigma1, pc.gainA);
        terms[1] = pc.gainB == pc.gainA || !above(gm, ge, pc.gainB)
            ? terms[0] : int64_t(uint64_t(pc.gainB) * pc.gainB * uint64_t(sigma1));
    }
    for (int v = 0; v < 2; ++v) {
        int64_t numer = int64_t(numer1) + terms[v];
        int x1;
        uint numLog = log_generate(best16(uint64_t(numer), x1));
        sums[5 + 2 * v] += x2 - x1;
        sums[6 + 2 * v] += int(numLog) - int(denLog);
    }
}
