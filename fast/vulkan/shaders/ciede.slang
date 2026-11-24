// Derived from libvmaf (https://github.com/Netflix/vmaf): ciede.c, Copyright
// 2016-2020 Netflix, Inc., licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root
// of this repository.
//
// CIEDE2000 (vv_v1_ciede): the colour difference dE00 of every pixel of a
// 4:2:0 frame pair, summed. A thread takes a 2x2 block of luma pixels and
// their chroma sample (at an odd width or height, the last column or row of
// blocks has one luma column or row), and each
// pixel's YUV becomes CIELAB as ciede.c's get_lab_color makes it (BT.709
// full-range-scaled RGB, sRGB's transfer, D65 XYZ), then dE00 as its
// ciede2000 (kL 0.65, kC 1, kH 4).
//
// ciede.c works in double (with float Lab colours); here every value is a
// float and log2, atan2, sin, cos and exp are this file's own, because
// Vulkan's built-in ones are only accurate to 2^-11 (sin, cos), 4096 ulp
// (atan) or 2^-21 absolute (log2 near 1: a uniform mid-grey frame's score
// 9e-5 off on an RTX 5090). exp2 is the GPU's own (the pass 13 % faster on
// a Core Ultra 9 285K's iGPU) and fused multiply-adds are allowed (32 %
// faster there); scores stay within 7e-6 of ciede.c's (tests/check_ciede.py).
// Identical pixels are skipped, so their dE00 stays exactly 0 (a fused
// multiply-add can round the two colours' Lab differently).
//
// Each pixel's dE00 is added as a whole number of 2^-20 (round to nearest),
// so the frame's 64-bit sum is the same in any order; collect() turns it
// into ciede.c's score, 45 - 20 log10(mean dE00).

#include "common.slang"

[[vk::push_constant]] uniform uint pc[32];
// pc: w, h, luma stride (words), chroma w and h ((w + 1) / 2, (h + 1) / 2),
// chroma stride (words), bit depth, the sum's slot.
[[vk::binding(0, 0)]] StructuredBuffer<uint> refLuma;
[[vk::binding(1, 0)]] StructuredBuffer<uint> disLuma;
[[vk::binding(2, 0)]] StructuredBuffer<uint> chroma;  // ref U, ref V, dis U, dis V
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> acc;

// ---------------------------------------------------------------- the maths

// log2(x), x > 0 and normal: x = m 2^e, m in [sqrt(1/2), sqrt(2)), and
// log2(m) = 2 atanh(s) / ln 2 with s = (m - 1) / (m + 1), |s| < 0.1716.
float f_log2(float x)
{
    const uint u = asuint(x);
    int e = int(u >> 23) - 127;
    float m = asfloat((u & 0x7FFFFFu) | 0x3F800000u);  // [1, 2)
    if (m > 1.41421356) {
        m *= 0.5;
        e += 1;
    }
    const float s = (m - 1.0) / (m + 1.0), z = s * s;
    // 2/ln2 (s + s^3/3 + s^5/5 + s^7/7 + s^9/9): the s^11 term < 3e-10
    float p = 0.32059889;
    p = 0.41219858 + z * p;
    p = 0.57707802 + z * p;
    p = 0.96179669 + z * p;
    p = 2.88539008 + z * p;
    return float(e) + s * p;
}

// 2^n for a whole n, -126 <= n <= 127
float pow2i(float n) { return asfloat(uint(int(n) + 127) << 23); }

// e^r, |r| <= ln2 / 2: Taylor to degree 7 (relative error < 2e-9)
float exp_kernel(float r)
{
    float p = 1.0 / 5040.0;
    p = 1.0 / 720.0 + r * p;
    p = 1.0 / 120.0 + r * p;
    p = 1.0 / 24.0 + r * p;
    p = 1.0 / 6.0 + r * p;
    p = 0.5 + r * p;
    p = 1.0 + r * p;
    return 1.0 + r * p;
}

// 2^t, -126 < t < 127
float f_exp2(float t)
{
    const float n = floor(t + 0.5);
    return exp_kernel((t - n) * 0.69314718) * pow2i(n);  // t - n exact, in [-0.5, 0.5]
}

// e^x: 0 below 2^-126
float f_exp(float x)
{
    const float n = floor(x * 1.44269504 + 0.5);
    if (n < -126.0)
        return 0.0;
    // ln2 = 0.693359375 - 2.12194440e-4: n times the first part is exact
    return exp_kernel((x - n * 0.693359375) + n * 2.12194440e-4) * pow2i(n);
}

// atan(t), 0 <= t <= 1: above tan(pi/8), pi/4 + atan((t - 1) / (t + 1)); the
// series to u^17 (|u| <= 0.4142: the next term < 3e-9)
float f_atan01(float t)
{
    const bool big = t > 0.41421356;
    const float u = big ? (t - 1.0) / (t + 1.0) : t;
    const float z = u * u;
    float p = 1.0 / 17.0;
    p = -1.0 / 15.0 + z * p;
    p = 1.0 / 13.0 + z * p;
    p = -1.0 / 11.0 + z * p;
    p = 1.0 / 9.0 + z * p;
    p = -1.0 / 7.0 + z * p;
    p = 1.0 / 5.0 + z * p;
    p = -1.0 / 3.0 + z * p;
    const float a = u + u * z * p;
    return big ? 0.78539816 + a : a;
}

// atan2(y, x), not both 0
float f_atan2(float y, float x)
{
    const float ay = abs(y), ax = abs(x);
    const bool swap = ay > ax;
    float a = f_atan01(swap ? ax / ay : ay / ax);
    if (swap)
        a = 1.57079633 - a;
    if (x < 0.0)
        a = 3.14159265 - a;
    return y < 0.0 ? -a : a;
}

// sin and cos of r, |r| <= pi/4: Taylor to r^9 and r^8
float k_sin(float r)
{
    const float z = r * r;
    return r + r * z * (-1.0 / 6.0 + z * (1.0 / 120.0 + z * (-1.0 / 5040.0 + z * (1.0 / 362880.0))));
}
float k_cos(float r)
{
    const float z = r * r;
    return 1.0 + z * (-0.5 + z * (1.0 / 24.0 + z * (-1.0 / 720.0 + z * (1.0 / 40320.0))));
}

// x - k pi/2 with pi/2 in three parts, the first two with few enough bits that
// k times them is exact (|x| < 2^11 here: hue angles of a few radians)
float reduce(float x, out int k)
{
    const float n = floor(x * 0.63661977 + 0.5);
    k = int(n);
    return ((x - n * 1.5703125) - n * 4.83751297e-4) - n * 7.54978995e-8;
}

float f_sin(float x)
{
    int k;
    const float r = reduce(x, k);
    const float s = (k & 1) != 0 ? k_cos(r) : k_sin(r);
    return (k & 2) != 0 ? -s : s;
}

// sin x and cos x from one reduction
float2 f_sincos(float x)
{
    int k;
    const float r = reduce(x, k);
    const float sr = k_sin(r), cr = k_cos(r);
    const float sn = (k & 1) != 0 ? cr : sr, cs = (k & 1) != 0 ? sr : cr;
    return float2((k & 2) != 0 ? -sn : sn, ((k + 1) & 2) != 0 ? -cs : cs);
}

float f_cos(float x)
{
    int k;
    const float r = reduce(x, k);
    const float c = (k & 1) != 0 ? k_sin(r) : k_cos(r);
    return ((k + 1) & 2) != 0 ? -c : c;
}

// ---------------------------------------------------------------- colour

// sRGB's transfer, from the gamma-encoded value to linear light
float rgb_linear(float c)
{
    return c > 10.0 / 255.0 ? exp2(2.4 * f_log2((c + 0.055) * (1.0 / 1.055))) : c * (1.0 / 12.92);
}

// CIELAB's f(t)
float lab_f(float t)
{
    return t > 216.0 / 24389.0 ? exp2(f_log2(t) * (1.0 / 3.0)) : (24389.0 / 27.0 * t + 16.0) * (1.0 / 116.0);
}

// get_lab_color: a sample's (y, u, v) at the frame's bit depth (scale = 2^(bpc - 8))
float3 lab(float y, float u, float v, float scale)
{
    y = (y - 16.0 * scale) * (1.0 / (219.0 * scale));
    u = (u - 128.0 * scale) * (1.0 / (224.0 * scale));
    v = (v - 128.0 * scale) * (1.0 / (224.0 * scale));
    const float r = rgb_linear(y + 1.28033 * v);
    const float g = rgb_linear(y - 0.21482 * u - 0.38059 * v);
    const float b = rgb_linear(y + 2.12798 * u);
    const float x = lab_f((r * 0.41245644 + g * 0.35757608 + b * 0.18043748) * (1.0 / 0.95047));
    const float yy = lab_f(r * 0.21267285 + g * 0.71515216 + b * 0.07217499);
    const float z = lab_f((r * 0.01933390 + g * 0.11919203 + b * 0.95030408) * (1.0 / 1.08883));
    return float3(116.0 * yy - 16.0, 500.0 * (x - yy), 200.0 * (yy - z));
}

// ciede.c's hue angle in [0, 2 pi): 0 for a = b = 0
float hue(float b, float a)
{
    if (b == 0.0 && a == 0.0)
        return 0.0;
    const float h = f_atan2(b, a);
    return h < 0.0 ? h + 6.28318531 : h;
}

// ciede2000 of two Lab colours (kL 0.65, kC 1, kH 4), in ciede.c's steps
float ciede2000(float3 c1, float3 c2)
{
    const float PI = 3.14159265;
    const float dl = c2.x - c1.x, lbar = (c1.x + c2.x) * 0.5;
    const float ch1 = sqrt(c1.y * c1.y + c1.z * c1.z), ch2 = sqrt(c2.y * c2.y + c2.z * c2.z);
    const float cbar = (ch1 + ch2) * 0.5;
    const float cb2 = cbar * cbar, cb7 = cb2 * cb2 * cb2 * cbar;
    const float g = 1.0 - sqrt(cb7 / (cb7 + 6103515625.0));  // 25^7
    const float ap1 = c1.y + c1.y * 0.5 * g, ap2 = c2.y + c2.y * 0.5 * g;
    const float cp1 = sqrt(ap1 * ap1 + c1.z * c1.z), cp2 = sqrt(ap2 * ap2 + c2.z * c2.z);
    const float cbp = (cp1 + cp2) * 0.5, dc = cp2 - cp1;
    const float lo = lbar - 50.0, lo2 = lo * lo;
    const float sl = 1.0 + 0.015 * lo2 / sqrt(20.0 + lo2);
    const float sc = 1.0 + 0.045 * cbp;
    const float h1 = hue(c1.z, ap1), h2 = hue(c2.z, ap2);
    float dh;
    if (ch1 == 0.0 || ch2 == 0.0)
        dh = 0.0;
    else if (abs(h1 - h2) <= PI)
        dh = h2 - h1;
    else if (h2 <= h1)
        dh = h2 - h1 + 2.0 * PI;
    else
        dh = h2 - h1 - 2.0 * PI;
    const float dH = 2.0 * sqrt(cp1 * cp2) * f_sin(dh * 0.5);
    const float H = abs(h1 - h2) > PI ? (h1 + h2 + 2.0 * PI) * 0.5 : (h1 + h2) * 0.5;
    // T's cosines of H - pi/6, 2H, 3H + pi/30 and 4H - 7pi/20, from sin H and
    // cos H by the multiple-angle and sum formulas (one reduction, not four)
    const float2 hsc = f_sincos(H);
    const float sin1 = hsc.x, cos1 = hsc.y;
    const float cos2 = 2.0 * cos1 * cos1 - 1.0, sin2 = 2.0 * sin1 * cos1;
    const float cos3 = cos1 * (4.0 * cos1 * cos1 - 3.0), sin3 = sin1 * (3.0 - 4.0 * sin1 * sin1);
    const float cos4 = 2.0 * cos2 * cos2 - 1.0, sin4 = 2.0 * sin2 * cos2;
    const float T = 1.0 - 0.17 * (cos1 * 0.86602540 + sin1 * 0.5) + 0.24 * cos2
                    + 0.32 * (cos3 * 0.99452190 - sin3 * 0.10452846) - 0.20 * (cos4 * 0.45399050 + sin4 * 0.89100652);
    const float sh = 1.0 + 0.015 * cbp * T;
    const float deg = (H * (180.0 / PI) - 275.0) * (1.0 / 25.0);
    const float cbp2 = cbp * cbp, cbp7 = cbp2 * cbp2 * cbp2 * cbp;
    const float rt = -2.0 * sqrt(cbp7 / (cbp7 + 6103515625.0)) * f_sin(60.0 * f_exp(-(deg * deg)) * (PI / 180.0));
    const float L = dl / (0.65 * sl), C = dc / sc, Hh = dH / (4.0 * sh);
    // never below 0 in exact arithmetic (|rt| <= 2); rounding could make it a hair negative
    return sqrt(max(L * L + C * C + Hh * Hh + rt * C * Hh, 0.0));
}

// ---------------------------------------------------------------- the pass

// Each subgroup's sum, at a place it takes from gSubgroups: the subgroups'
// lanes need not be consecutive SV_GroupIndex values (Intel's iGPU's are
// not: a place from the index counted every sum twice there).
groupshared uint64_t gSum[256];
groupshared uint gSubgroups;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint local : SV_GroupIndex)
{
    const int w = int(pc[0]), h = int(pc[1]), strideWords = int(pc[2]);
    const int cw = int(pc[3]), ch = int(pc[4]), chromaStride = int(pc[5]);
    const float scale = float(1u << (pc[6] - 8u));
    const int cx = int(id.x), cy = int(id.y);
    if (local == 0)
        gSubgroups = 0;
    GroupMemoryBarrierWithGroupSync();
    uint64_t sum = 0;
    if (cx < cw && cy < ch) {
        // The four planes one after another, ch rows each.
        const float refU = float(pixel(chroma, cx, cy, chromaStride));
        const float refV = float(pixel(chroma, cx, cy + ch, chromaStride));
        const float disU = float(pixel(chroma, cx, cy + 2 * ch, chromaStride));
        const float disV = float(pixel(chroma, cx, cy + 3 * ch, chromaStride));
        // (an odd width or height: the last chroma column or row has one luma column or row)
        const int x0 = 2 * cx, y0 = 2 * cy, x1 = min(x0 + 2, w), y1 = min(y0 + 2, h);
        // Identical pixels' dE00 is exactly 0 (also in ciede.c): skipped
        // (letterboxes, flat areas an encode keeps exact), which also keeps a
        // fused multiply-add from making it a hair above 0.
        const bool sameChroma = refU == disU && refV == disV;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const uint refY = pixel(refLuma, x, y, strideWords), disY = pixel(disLuma, x, y, strideWords);
                if (sameChroma && refY == disY)
                    continue;
                const float3 a = lab(float(refY), refU, refV, scale);
                const float3 b = lab(float(disY), disU, disV, scale);
                sum += uint64_t(ciede2000(a, b) * 1048576.0 + 0.5);
            }
        }
    }
    const uint64_t total = WaveActiveSum(sum);
    if (WaveIsFirstLane()) {
        uint place;
        InterlockedAdd(gSubgroups, 1u, place);
        gSum[place] = total;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        uint64_t all = 0;
        for (uint i = 0; i < gSubgroups; ++i)
            all += gSum[i];
        if (all != 0)
            acc_add64(acc, pc[7], all);
    }
}
