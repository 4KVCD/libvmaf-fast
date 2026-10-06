// Derived from libvmaf (https://github.com/Netflix/vmaf): speed.c,
// vif_tools.c, picture_copy.c and common/convolution_avx.c, Copyright
// 2016-2025 Netflix, Inc., licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root
// of this repository.
//
// SpEED chroma's filter_and_downscale (VMAF v1), bit for bit: the four chroma
// planes of a frame pair (ref U, ref V, dis U, dis V), each
//   DEC   picture_copy's float (the sample, divided by 4, 16 or 256 above 8
//         bits, exactly, minus 128), rescaled where the model's
//         speed_prescale asks (bilinear only), the antialias filter (vif_filter1d_s:
//         down the columns, then across, each a float sum from the first tap,
//         mirrored at the edges without repeating the edge) and every 16th
//         row and column of that (vif_dec16_s). Only those are calculated:
//         each is the same sums as the whole picture's.
//   BLUR  the small filter on that picture (vif_filter1d_s again) and the
//         picture minus it (subtract_image), into the frame's buffer for
//         est_params on the CPU (v1_speed.c).
// libvmaf's AVX2 filter (convolution_f32_avx_s) adds in the same order as its
// plain one, a multiply and an add each rounded (no fused multiply-add):
// `precise` keeps the driver from fusing them here.

[[vk::push_constant]] uniform uint pc[32];  // (FROM_IMAGE: the pictures' places in the last words, common.slang)
#ifndef FROM_IMAGE
#define FROM_IMAGE 0
#endif
#define PICTURE_REFERENCE 28  // (common.slang's)
#define PICTURE_DISTORTED 30

int mirror(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - i - 2 : i); }

#if STAGE == 1  // DEC
#if FROM_IMAGE
// The planes from the pictures' chroma textures: 0, 1 the reference's U and
// V, 2, 3 the distorted's (as the shared buffers have them).
[[vk::binding(0, 0)]] Texture2D<uint2> refChroma;
[[vk::binding(4, 0)]] Texture2D<uint2> disChroma;
#else
[[vk::binding(0, 0)]] StructuredBuffer<uint> chroma;
#endif
[[vk::binding(1, 0)]] StructuredBuffer<float> filters;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> operating;
[[vk::binding(3, 0)]] StructuredBuffer<uint> scaling;
// pc: w, h (of the plane filtered: the scaled one where it is rescaled),
// strideWords, planeWords, ow, oh, taps, bpc, 1/scaler (float bits), scaled.
// scaling (when scaled): vif_tools.c's bilinear tables -- x1, x2, dx (bits)
// for the w columns, then y1, y2, dy for the h rows (v1_speed.c works them out).

float sample(uint plane, int x, int y)
{
#if FROM_IMAGE
    const uint place = plane < 2 ? PICTURE_REFERENCE : PICTURE_DISTORTED;
    const int3 at = int3(x + (int(pc[place]) >> 1), y + (int(pc[place + 1]) >> 1), 0);
    const uint2 both = plane < 2 ? refChroma.Load(at) : disChroma.Load(at);
    const uint v = ((plane & 1) == 0 ? both.x : both.y) >> (pc[7] > 8 ? 6u : 0u);
#else
    const uint base = plane * pc[3] + uint(y) * pc[2];
    uint v;
    if (pc[7] > 8) {
        v = (chroma[base + uint(x >> 1)] >> ((x & 1) * 16)) & 0xFFFFu;
    } else {
        v = (chroma[base + uint(x >> 2)] >> ((x & 3) * 8)) & 0xFFu;
    }
#endif
    precise float scaled = float(v) * asfloat(pc[8]);  // exact: v / 2^k
    precise float result = scaled + -128.0;
    return result;
}

// The plane's value the filter reads: picture_copy's, or that rescaled as
// vif_scale_frame_bilinear_apply_s does, from left to right:
// (1 - dy) * (1 - dx) * r1[x1] + (1 - dy) * dx * r1[x2] + dy * (1 - dx) * r2[x1] + dy * dx * r2[x2].
float value(uint plane, int x, int y)
{
    if (pc[9] == 0) return sample(plane, x, y);
    const uint w = pc[0], h = pc[1];
    const int x1 = int(scaling[x]), x2 = int(scaling[w + uint(x)]);
    const int y1 = int(scaling[3 * w + uint(y)]), y2 = int(scaling[3 * w + h + uint(y)]);
    const float dx = asfloat(scaling[2 * w + uint(x)]), dy = asfloat(scaling[3 * w + 2 * h + uint(y)]);
    precise float ny = 1.0 - dy;
    precise float nx = 1.0 - dx;
    precise float a = ny * nx;
    a = a * sample(plane, x1, y1);
    precise float b = ny * dx;
    b = b * sample(plane, x2, y1);
    precise float c = dy * nx;
    c = c * sample(plane, x1, y2);
    precise float d = dy * dx;
    d = d * sample(plane, x2, y2);
    precise float sum = a + b;
    sum = sum + c;
    sum = sum + d;
    return sum;
}

// A thread per output and column of its square (64 / taps outputs a
// workgroup): each column's vertical sum, then the output's thread adds the
// columns across in order -- the same sums as one thread doing all of them,
// with more threads to wait on the samples.
groupshared float gColumn[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex)
{
    const int w = int(pc[0]), h = int(pc[1]);
    const int ow = int(pc[4]), oh = int(pc[5]), taps = int(pc[6]), r = taps / 2;
    const uint per = 64u / uint(taps), o = t / uint(taps), c = t % uint(taps);
    const uint n = uint(ow * oh), out = group.x * per + o;
    const bool active = o < per && out < 4 * n;
    const uint plane = out / n, at = out % n;
    const int ox = int(at % uint(ow)), oy = int(at / uint(ow));
    const int cx = ox * 16, cy = oy * 16;
    if (active) {
        const int x = mirror(cx - r + int(c), w);
        precise float column = 0.0;
        for (int fi = 0; fi < taps; ++fi) {
            precise float product = filters[fi] * value(plane, x, mirror(cy - r + fi, h));
            column = column + product;
        }
        gColumn[t] = column;
    }
    GroupMemoryBarrierWithGroupSync();
    if (active && c == 0) {
        precise float sum = 0.0;
        for (int fj = 0; fj < taps; ++fj) {
            precise float product = filters[fj] * gColumn[o * uint(taps) + uint(fj)];
            sum = sum + product;
        }
        operating[plane * n + at] = sum;
    }
}

#elif STAGE == 2  // BLUR
[[vk::binding(0, 0)]] StructuredBuffer<float> operating;
[[vk::binding(1, 0)]] StructuredBuffer<float> filters;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> keep;
// pc: ow, oh, first tap (in filters), taps

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int x = int(id.x), y = int(id.y), ow = int(pc[0]), oh = int(pc[1]);
    const uint first = pc[2];
    const int taps = int(pc[3]), r = taps / 2;
    if (x >= ow || y >= oh) return;
    const uint base = id.z * uint(ow * oh);
    precise float sum = 0.0;
    for (int fj = 0; fj < taps; ++fj) {
        const int xx = mirror(x - r + fj, ow);
        precise float column = 0.0;
        for (int fi = 0; fi < taps; ++fi) {
            precise float product = filters[first + uint(fi)] * operating[base + uint(mirror(y - r + fi, oh) * ow + xx)];
            column = column + product;
        }
        precise float product = filters[first + uint(fj)] * column;
        sum = sum + product;
    }
    precise float difference = operating[base + uint(y * ow + x)] - sum;
    keep[base + uint(y * ow + x)] = difference;
}
#endif
