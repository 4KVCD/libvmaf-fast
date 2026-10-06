"""VIF's statistics' fast log (vif_stats.slang's FAST_LOG): the GPU's log2 rounded in place of the log
table's entry where the rounding is settled -- the bound it relies on, and the GPU's results.

The table's entries are round(log2f(32768 + i) * 2048), log2f being CUDA's (vmaf_vulkan.cpp's
cuda_log_generate, emulated here in float32 with exact fmaf). This checks, for all 32768 entries,
that CUDA's log2f is within 0.0011 (in table units) of the exact value, so that a GPU's log2 within
the 3 ulp Vulkan requires (0.0059 here) is within 0.0069 of CUDA's: less than FAST_LOG's margin,
1/64, from halfway between two integers.

With --gpu DEVICE, also the GPU's: VMAF_FAST_DIST names a build made for it -- vif_fused_0_8,
_0_16 and _0_16s built with LOG_VERIFY=73 as well, and vmaf_vulkan.cpp's kSlots raised by 3.
Each VIF scale 0 pass then compares the fast log with the table for every entry (one thread) and
for every pixel's logs, and sums 73 and 74 count the disagreements and the fast logs.
    check_vif_log.py [--gpu DEVICE]"""
import argparse
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

MARGIN = 1 / 64
COEFFICIENTS = (0x3E2FCF2A, 0xBE374E43, 0x3E520BF4, 0xBE763C8B, 0x3E93BF99, 0xBEB8AA49, 0x3EF6384A, 0xBF38AA3B)


def f32(x) -> Fraction:
    """The float32 nearest to x (ties to even)."""
    x = Fraction(x)
    if x == 0:
        return Fraction(0)
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    ulp = Fraction(2) ** (e - 23)
    n, rest = divmod(a / ulp, 1)
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return (1 if x > 0 else -1) * n * ulp


def from_bits(bits: int) -> Fraction:
    return Fraction(struct.unpack("<f", struct.pack("<I", bits))[0])


def fmaf(a, b, c) -> Fraction:
    return f32(a * b + c)


def cuda_log2(i: int) -> Fraction:
    """CUDA's log2f(i), as cuda_log_generate calculates it."""
    bits = struct.unpack("<I", struct.pack("<f", float(i)))[0]
    exponent = (bits - 0x3F3504F3) & 0xFF800000
    m = from_bits((bits - exponent) & 0xFFFFFFFF)
    e = fmaf(f32(exponent - (1 << 32) if exponent & 0x80000000 else exponent), from_bits(0x34000000), 0)
    f = f32(m - 1)
    p = fmaf(f, from_bits(0x3DC6B27F), from_bits(0xBE2C7F30))
    for coefficient in COEFFICIENTS:
        p = fmaf(p, f, from_bits(coefficient))
    return f32(e + fmaf(f, from_bits(0x3FB8AA3B), f32(f * f32(f * p))))


def host_check() -> bool:
    worst = max(abs(float(cuda_log2(v) * 2048) - math.log2(v) * 2048) for v in range(32768, 65536))
    gpu = 3 * 2.0 ** -20 * 2048  # 3 ulp of a log2 in [8, 16), in table units
    ok = worst + gpu < MARGIN
    print(f"CUDA's log2f within {worst:.6f} of the exact value; with a GPU's 3 ulp ({gpu:.6f}): "
          f"{worst + gpu:.6f}, {'below' if ok else 'NOT below'} the margin {MARGIN:.6f}")
    return ok


def gpu_check(device: int) -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from vmaf_fast import vulkan
    reference, distorted = vulkan.probe_frames(10)
    width, height = vulkan._PROBE_SIZE
    scorer = vulkan.VulkanScorer(width, height, 10, {}, device=device)
    try:
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        scorer.features()
        sums = [scorer.sums(f) for f in range(len(reference))]
    finally:
        scorer.close()
    wrong = sum(int(s[73]) for s in sums) if len(sums[0]) > 74 else 0
    fast = sum(int(s[74]) for s in sums) if len(sums[0]) > 74 else 0
    if fast == 0:
        print("no fast logs counted: VMAF_FAST_DIST is not a LOG_VERIFY build")
        return False
    print(f"device {device}: {fast:,} fast logs of the probe frames' pixels, every entry swept each frame: "
          f"{wrong} disagreements")
    return wrong == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, metavar="DEVICE", help="also the GPU's results (a LOG_VERIFY build)")
    args = parser.parse_args()
    ok = host_check()
    if args.gpu is not None:
        ok = gpu_check(args.gpu) and ok
    print("RESULT:", "the fast log is the table's" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
