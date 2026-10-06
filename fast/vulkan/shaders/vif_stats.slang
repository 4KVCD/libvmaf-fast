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

// log_generate(32768 + i) for i = v & 0x7FFF: two 16-bit entries a word (the
// values are 30720..32768), half the table a word each would take in the
// caches.
uint log_generate(uint v)
{
    const uint i = v & 0x7FFFu;
    return (logTable[i >> 1] >> ((i & 1u) * 16)) & 0xFFFFu;
}

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

#ifndef VIF_FAST
#define VIF_FAST 1
#endif
// Verification builds: the first of three spare sums (the engine's kSlots
// raised by three) for the fast path's disagreements, refusals and pixels.
#ifndef VIF_STATS_VERIFY
#define VIF_STATS_VERIFY 0
#endif

// The fast path to what gain(), residual(), above() and gain_term() give
// a pixel with sigma12 > 0, sigma2 > 0 and sigma1 >= 2^17: the residual sv
// and the two limits' gain terms, from one exact integer division and a
// few floats -- or false where the floats cannot settle them, and then the
// exact path does.
//
// With D = sigma1 + eps as a double (the divisor gain() rounds), g = sigma12
// / D, and P = sigma12^2 / sigma1 = Q + R / sigma1 (Q, R integers, found
// exactly below):
//   g * sigma12 = P / (1 + e),   g^2 * sigma1 = P / (1 + e)^2,   e = (D - sigma1) / sigma1 < 2^-34
// so the double residual sigma2 - g * sigma12 is (sigma2 - Q) + (c - R /
// sigma1) and the double gain term Q + (R / sigma1 - c2), c = P e / (1 + e),
// c2 = c (2 + e) / (1 + e) = 2c to within c 2^-34. What is left is the
// doubles' own rounding: g to 53 bits, the residual rounded once, the gain
// term's two products -- at most (P + sigma2) 2^-51 together -- plus the
// floats' error here (each operation within a few units in the last place:
// at most (c + 1) 2^-20 for the fractions below). Their integer parts are
// decided only where the float fraction is further than twice that from an
// integer (Er, Et); elsewhere the exact path. g > limit is exactly
// sigma12 > limit * sigma1: D exceeds sigma1 by under 2^-17, far less than
// 1 / limit, and g is then at least 2^-38 (relative) away from limit, far
// more than its rounding.
bool fast_gain(int sigma12, int sigma1, int sigma2, out int sv, out int64_t terms[2])
{
    sv = 0;
    terms[0] = terms[1] = 0;
    const uint64_t S = uint64_t(uint(sigma12)) * uint(sigma12);
    const int64_t divisor = int64_t(sigma1);
    const float inv = 1.0 / float(uint(sigma1));
    const float estimate = float(S) * inv;
    if (!(estimate < 70368744177664.0))  // 2^46
        return false;
    // Q = S / sigma1 exactly: the estimate (within 2^25), two corrections
    // from the remainder (each within 2^-21 of it), then the last unit.
    int64_t q = int64_t(estimate);
    int64_t rem = int64_t(S) - q * divisor;
    for (int round = 0; round < 2; ++round) {
        const int64_t d = int64_t(floor(float(rem) * inv));
        q += d;
        rem -= d * divisor;
    }
    for (int fix = 0; fix < 2; ++fix) {
        if (rem < 0) {
            q--;
            rem += divisor;
        } else if (rem >= divisor) {
            q++;
            rem -= divisor;
        }
    }
    if (rem < 0 || rem >= divisor || q < 0)
        return false;
    const float fraction = float(uint(rem)) * inv;  // R / sigma1
    const float P = float(q) + fraction;
    const float c = P * (6.5536e-6 * inv);          // P eps / sigma1
    // The residual: max((int)(sigma2 - g * sigma12), 0).
    const float phi = c - fraction;
    const float Er = (c + 1.0) * 1.9073486e-6 + (P + float(sigma2)) * 3.5527137e-15;  // 2^-19, 2^-48
    const float whole = floor(phi);
    if (phi - whole < Er || whole + 1.0 - phi < Er)
        return false;
    const int64_t residual = int64_t(sigma2) - q + int64_t(whole);
    sv = residual > 0 ? int(residual) : 0;  // residual <= sigma2
    // The gain terms, g capped at each limit.
    if (uint64_t(uint(sigma12)) > uint64_t(pc.gainA) * uint(sigma1)) {
        terms[0] = int64_t(uint64_t(pc.gainA) * pc.gainA * uint64_t(sigma1));
    } else {
        const float psi = fraction - 2.0 * c;
        const float Et = (c + 1.0) * 1.9073486e-6 + P * 7.1054274e-15;  // 2^-19, 2^-47
        const float part = floor(psi);
        if (psi - part < Et || part + 1.0 - psi < Et)
            return false;
        terms[0] = q + int64_t(part);
    }
    terms[1] = pc.gainB == pc.gainA || uint64_t(uint(sigma12)) <= uint64_t(pc.gainB) * uint(sigma1)
        ? terms[0] : int64_t(uint64_t(pc.gainB) * pc.gainB * uint64_t(sigma1));
    return true;
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
    int sv = sigma2;
    int64_t terms[2] = { 0, 0 };
    bool fast = false;
#if VIF_FAST && !NATIVE_F64
    if (gained)
        fast = fast_gain(sigma12, sigma1, sigma2, sv, terms);
#endif
#if VIF_STATS_VERIFY
    // Verification builds: the exact path for every gained pixel too, and
    // counts of the fast path's disagreements, refusals and pixels.
    const int fastSv = sv;
    const int64_t fastTerms[2] = { terms[0], terms[1] };
    if (gained) {
#else
    if (gained && !fast) {
#endif
        uint64_t gm = 0;
        int ge = 0;
        gain(sigma12, sigma1, gm, ge);
        sv = residual(sigma2, gm, ge, sigma12);
        // The gain term is the same for both limits unless g is above the
        // lower one, and then that limit's term is just limit * limit * sigma1_sq.
        terms[0] = gain_term(gm, ge, sigma1, pc.gainA);
        terms[1] = pc.gainB == pc.gainA || !above(gm, ge, pc.gainB)
            ? terms[0] : int64_t(uint64_t(pc.gainB) * pc.gainB * uint64_t(sigma1));
    }
#if VIF_STATS_VERIFY
    if (gained) {
        InterlockedAdd(acc[2 * (VIF_STATS_VERIFY + 2)], 1u);
        if (!fast)
            InterlockedAdd(acc[2 * (VIF_STATS_VERIFY + 1)], 1u);
        else if (fastSv != sv || fastTerms[0] != terms[0] || fastTerms[1] != terms[1])
            InterlockedAdd(acc[2 * VIF_STATS_VERIFY], 1u);
    }
#endif
    uint numer1 = uint(sv + kSigmaNsq);
    int x2;
    uint denLog = log_generate(best16(uint64_t(numer1), x2));
    for (int v = 0; v < 2; ++v) {
        int64_t numer = int64_t(numer1) + terms[v];
        int x1;
        uint numLog = log_generate(best16(uint64_t(numer), x1));
        sums[5 + 2 * v] += x2 - x1;
        sums[6 + 2 * v] += int(numLog) - int(denLog);
    }
}
