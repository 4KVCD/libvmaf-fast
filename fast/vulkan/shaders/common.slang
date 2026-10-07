// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// Shared by every VMAF compute shader (see ../vmaf_vulkan.cpp).
//
// The shaders are a port of libvmaf's CUDA kernels (libvmaf/src/feature/cuda,
// with the pull requests README.md lists) and are written to give the same
// integers as those kernels on any GPU. Where the CUDA code uses float or
// double, the shaders calculate its rounded result exactly in integers,
// because Vulkan leaves the rounding of floating point to the driver (which
// may also fuse a multiply and an add) and some GPUs (Intel's recent
// integrated ones) have no double at all. Float is used only for estimates
// the result does not depend on: ADM's one-degree test is decided in float
// only where float's error cannot change the answer, and VIF's division
// starts from a float estimate that is then made exact. Double is used only
// with NATIVE_F64 (vv_create's flag 1, off by default).

// A 64-bit sum kept as two 32-bit words, added to with 32-bit atomics (64-bit
// atomics are optional in Vulkan). The carry out of the low word is added to
// the high word, so the pair is the exact sum modulo 2^64 whatever the order.
void acc_add64(RWStructuredBuffer<uint> acc, uint slot, uint64_t value)
{
    uint lo = uint(value), hi = uint(value >> 32), old;
    InterlockedAdd(acc[2 * slot], lo, old);
    uint high = hi + ((old > ~lo) ? 1u : 0u);
    if (high != 0)
        InterlockedAdd(acc[2 * slot + 1], high);
}

// A picture's luma sample: rows of bytes (or 16-bit little-endian samples)
// packed into 32-bit words, each row starting on a word.
uint pixel(StructuredBuffer<uint> picture, int x, int y, int strideWords)
{
#if BPC16
    uint word = picture[y * strideWords + (x >> 1)];
    return (word >> ((x & 1) * 16)) & 0xFFFFu;
#else
    uint word = picture[y * strideWords + (x >> 2)];
    return (word >> ((x & 3) * 8)) & 0xFFu;
#endif
}

// Vulkan VMAF's pictures mode (vv_pictures, FROM_IMAGE): VMAF v1's pictures
// read where a decoder left them, its textures imported -- NV12 (8-bit) or
// P010, whose 10-bit samples are in the top bits of 16 -- a plane viewed as
// R8/R16 (luma) or R8G8/R16G16 (chroma, U and V). A picture's first sample is
// at (ox, oy) of its texture (luma; chroma at half), given each frame in the
// push constants' last words, as vv_pictures took them: the motion's previous
// picture's at 26-27, the reference's at 28-29, the distorted's at 30-31.
#ifndef FROM_IMAGE
#define FROM_IMAGE 0
#endif
#define PICTURE_PREVIOUS 26
#define PICTURE_REFERENCE 28
#define PICTURE_DISTORTED 30
#if FROM_IMAGE
#if BPC16
#define IMAGE_SHIFT 6
#else
#define IMAGE_SHIFT 0
#endif
uint image_pixel(Texture2D<uint> image, int x, int y, int ox, int oy)
{
    return image.Load(int3(x + ox, y + oy, 0)) >> IMAGE_SHIFT;
}
uint image_chroma(Texture2D<uint2> image, int x, int y, int ox, int oy, uint component)
{
    const uint2 both = image.Load(int3(x + (ox >> 1), y + (oy >> 1), 0));
    return (component == 0 ? both.x : both.y) >> IMAGE_SHIFT;
}
#endif

int sext16(int v) { return (v << 16) >> 16; }

// ADM's decoupled images at scale 0 (adm_decouple's outputs, adm_cm's
// inputs), whose values are all 16-bit: two words a pixel, bands 0 and 1 in
// the first (low and high half), band 2 in the second. Four words a pixel,
// one per band and one unused, made those two passes write and read twice
// the memory -- most of adm_decouple's time on an integrated GPU. The other
// scales' values are 32-bit: four words a pixel.
#define ADM_PACKED_WORDS 2

void store_packed(RWStructuredBuffer<int> buffer, int pixel, int v0, int v1, int v2)
{
    buffer[pixel * ADM_PACKED_WORDS] = (v0 & 0xFFFF) | (v1 << 16);
    buffer[pixel * ADM_PACKED_WORDS + 1] = v2;
}

// ADM's band images (adm_dwt's outputs): each position's h, v and d -- at
// scale 0 16-bit, h and v in the low and high half of the first word and d
// in the second; at the other scales 32-bit, three words. The approximation
// band a, which only the next scale's transform reads, is in an image of its
// own: one word a position, or at scale 0, whose a is 16-bit, two positions
// a word (adm_dwt). Four words a position, a h v d together, made the next
// scale's transform read four times the memory it needs, and scale 0's
// readers twice.
#define ADM_HVD_WORDS0 2
#define ADM_HVD_WORDS 3

void adm_hvd(StructuredBuffer<uint> bands, int position, bool scale0, out int v[3])
{
    if (scale0) {
        int first = int(bands[position * ADM_HVD_WORDS0]);
        v[0] = sext16(first);
        v[1] = first >> 16;
        v[2] = int(bands[position * ADM_HVD_WORDS0 + 1]);
    } else {
        for (int b = 0; b < 3; ++b)
            v[b] = int(bands[position * ADM_HVD_WORDS + b]);
    }
}

int msb32(uint v) { return int(firstbithigh(v)); }  // v != 0

int msb64(uint64_t v)  // v != 0
{
    uint hi = uint(v >> 32);
    return hi != 0 ? 32 + msb32(hi) : msb32(uint(v));
}

struct U128 { uint64_t hi, lo; };

U128 mul64(uint64_t a, uint64_t b)
{
    uint64_t a0 = a & 0xFFFFFFFFull, a1 = a >> 32, b0 = b & 0xFFFFFFFFull, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFull) + (p10 & 0xFFFFFFFFull);
    U128 r;
    r.lo = (p00 & 0xFFFFFFFFull) | (mid << 32);
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}

// A positive 128-bit integer rounded to a double's 53 bits, to nearest and
// ties to even: v ~ m * 2^e with m in [2^52, 2^53). v != 0.
void round53(U128 v, out uint64_t m, out int e)
{
    int top = v.hi != 0 ? 64 + msb64(v.hi) : msb64(v.lo);
    if (top <= 52) {
        m = v.lo << (52 - top);
        e = top - 52;
        return;
    }
    int sh = top - 52;
    uint64_t q;
    bool above, tie;
    if (sh < 64) {
        q = (v.lo >> sh) | (v.hi << (64 - sh));
        uint64_t rem = v.lo & ((1ull << sh) - 1), half = 1ull << (sh - 1);
        above = rem > half;
        tie = rem == half;
    } else if (sh == 64) {
        q = v.hi;
        above = v.lo > (1ull << 63);
        tie = v.lo == (1ull << 63);
    } else {
        int s = sh - 64;
        q = v.hi >> s;
        uint64_t rem = v.hi & ((1ull << s) - 1), half = 1ull << (s - 1);
        above = rem > half || (rem == half && v.lo != 0);
        tie = rem == half && v.lo == 0;
    }
    if (above || (tie && (q & 1) != 0)) {
        q++;
        if (q == (1ull << 53)) {
            q >>= 1;
            sh++;
        }
    }
    m = q;
    e = sh;
}
