"""Checks CIEDE2000 on the GPU (vv_v1_ciede, shaders/ciede.slang) against libvmaf's ciede.c.

Synthetic 4:2:0 frame pairs (8, 10 and 12 bits, odd and even sizes; noise, grey, black and white, identical
pictures) against a NumPy port of ciede.c, and, with --clips, real videos' per-frame scores against score files
libvmaf wrote (`ciede2000 <frame> <score>` lines, the first frames of the video pair, decoded by FFmpeg into a
pipe as 4:2:0 at the given bit depth). The GPU's float arithmetic is expected within 1e-4 of ciede.c's score on
every frame (about 1e-7 on real video). Needs a GPU with Vulkan and fast/dist's vmaf_vulkan.dll.

    python fast/tests/check_ciede.py [--device N] [--clips REF TEST BITS FRAMES SCORES ...]
"""
from __future__ import annotations

import argparse
import ctypes
import math
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from vmaf_fast import vulkan  # noqa: E402

TOLERANCE = 1e-4
_V1_OPTIONS = (3.0, 1080, 0, 0.03125, 1.0, 0.0, 10000.0, 0, 0)  # vv_create_v1's nine (unused: no motion, no ADM)
_ONLY_EXTRAS = (1 << 16) | (1 << 18)  # vv_create's flag bits: no motion, no ADM


def _bind(lib: ctypes.CDLL) -> None:
    lib.vv_create_v1_ex.restype = ctypes.c_int
    lib.vv_create_v1_ex.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.vv_v1_ciede.restype = ctypes.c_int
    lib.vv_v1_ciede.argtypes = [ctypes.c_void_p]
    lib.vv_v1_ciede_scores.restype = ctypes.c_int
    lib.vv_v1_ciede_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]
    lib.vv_submit_v1.restype = ctypes.c_int
    lib.vv_submit_v1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_ssize_t),
                                 ctypes.c_int]


class GpuCiede:
    """A context that calculates CIEDE2000 alone; frames are (Y, U, V) arrays of uint8 or uint16."""

    def __init__(self, lib: ctypes.CDLL, device: int, width: int, height: int, bits: int) -> None:
        self.lib, self.frames = lib, 0
        self.context = ctypes.c_void_p()
        options = (ctypes.c_double * 9)(*_V1_OPTIONS)
        vulkan._check(lib, lib.vv_create_v1_ex(ctypes.byref(self.context), device, width, height, bits,
                                               _ONLY_EXTRAS, options, 9), "vv_create_v1_ex")
        try:
            vulkan._check(lib, lib.vv_v1_ciede(self.context), "vv_v1_ciede")
        except Exception:
            lib.vv_destroy(self.context)
            raise

    def add(self, reference, distorted, score: bool = True) -> None:
        planes = [np.ascontiguousarray(p) for p in (*reference, *distorted)]
        pointers = (ctypes.c_void_p * 6)(*(p.ctypes.data for p in planes))
        strides = (ctypes.c_ssize_t * 6)(*(p.strides[0] for p in planes))
        vulkan._check(self.lib, self.lib.vv_submit_v1(self.context, pointers, strides, int(score)), "vv_submit_v1")
        self.frames += 1

    def scores(self) -> np.ndarray:
        vulkan._check(self.lib, self.lib.vv_flush(self.context), "vv_flush")
        out = np.zeros(self.frames)
        count = self.lib.vv_v1_ciede_scores(self.context, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                            self.frames)
        vulkan._check(self.lib, min(count, 0), "vv_v1_ciede_scores")
        return out

    def close(self) -> None:
        self.lib.vv_destroy(self.context)


# ------------------------------------------------- ciede.c in NumPy (float64, rounded to float where it is)

def _f32(x):
    return np.asarray(x, np.float64).astype(np.float32).astype(np.float64)


def _lab(y, u, v, bits):
    scale = float(1 << (bits - 8))
    y = (y - 16.0 * scale) * (1.0 / (219.0 * scale))
    u = (u - 128.0 * scale) * (1.0 / (224.0 * scale))
    v = (v - 128.0 * scale) * (1.0 / (224.0 * scale))
    rgb = [y + 1.28033 * v, y - 0.21482 * u - 0.38059 * v, y + 2.12798 * u]
    r, g, b = (np.where(c > 10.0 / 255.0, np.power(np.maximum(c + 0.055, 0) * (1.0 / 1.055), 2.4), c * (1.0 / 12.92))
               for c in rgb)

    def lab_map(c):
        return _f32(np.where(c > 216.0 / 24389.0, np.power(np.maximum(c, 0), 1.0 / 3.0),
                             (24389.0 / 27.0 * c + 16.0) * (1.0 / 116.0)))

    x = lab_map((r * 0.4124564390896921 + g * 0.357576077643909 + b * 0.18043748326639894) * (1.0 / 0.95047))
    yy = lab_map(r * 0.21267285140562248 + g * 0.715152155287818 + b * 0.07217499330655958)
    z = lab_map((r * 0.019333895582329317 + g * 0.119192025881303 + b * 0.9503040785363677) * (1.0 / 1.08883))
    return _f32(116.0 * yy - 16.0), _f32(500.0 * (x - yy)), _f32(200.0 * (yy - z))


def _hue(b, a):
    h = _f32(np.arctan2(b, a))
    h = _f32(np.where(h < 0, h + 2 * math.pi, h))
    return np.where((b == 0) & (a == 0), 0.0, h)


def _de00(l1, a1, b1, l2, a2, b2):
    dl = _f32(l2 - l1)
    lbar = _f32((l1 + l2) / 2)
    c1, c2 = _f32(np.sqrt(a1 * a1 + b1 * b1)), _f32(np.sqrt(a2 * a2 + b2 * b2))
    cbar = _f32((c1 + c2) / 2)
    cb7 = cbar ** 7
    g = 1 - np.sqrt(cb7 / (cb7 + 25.0 ** 7))
    ap1, ap2 = _f32(a1 + (a1 / 2) * g), _f32(a2 + (a2 / 2) * g)
    cp1, cp2 = _f32(np.sqrt(ap1 * ap1 + b1 * b1)), _f32(np.sqrt(ap2 * ap2 + b2 * b2))
    cbp, dc = _f32((cp1 + cp2) / 2), _f32(cp2 - cp1)
    lo = _f32(lbar - 50)
    lo2 = lo * lo
    sl = _f32(1 + 0.015 * lo2 / np.sqrt(20 + lo2))
    sc = _f32(1 + 0.045 * cbp)
    h1, h2 = _hue(b1, ap1), _hue(b2, ap2)
    d = h2 - h1
    dh = _f32(np.where((c1 == 0) | (c2 == 0), 0.0,
                       np.where(np.abs(h1 - h2) <= math.pi, d, np.where(h2 <= h1, d + 2 * math.pi, d - 2 * math.pi))))
    dH = _f32(2.0 * np.sqrt(cp1 * cp2) * np.sin(dh / 2.0))
    H = _f32(np.where(np.abs(h1 - h2) > math.pi, (h1 + h2 + 2 * math.pi) / 2, (h1 + h2) / 2))
    T = _f32(1 - 0.17 * np.cos(H - math.pi / 6) + 0.24 * np.cos(2 * H) + 0.32 * np.cos(3 * H + math.pi / 30)
             - 0.20 * np.cos(4 * H - 7 * math.pi / 20))
    sh = _f32(1 + 0.015 * cbp * T)
    deg = _f32((_f32(H * (180 / math.pi)) - 275) * (1 / 25))
    cbp7 = _f32(cbp ** 7)
    rt = _f32(-2 * np.sqrt(cbp7 / (cbp7 + _f32(25.0 ** 7))) * np.sin(_f32(60 * np.exp(-_f32(deg * deg))) * (math.pi / 180)))
    L, C, Hh = _f32(dl / (0.65 * sl)), _f32(dc / sc), _f32(dH / (4.0 * sh))
    return _f32(np.sqrt(L * L + C * C + Hh * Hh + rt * C * Hh))


def reference_score(reference, distorted, bits: int) -> float:
    """ciede.c's score of a 4:2:0 pair: each luma pixel with its block's chroma sample."""
    labs = []
    for y, u, v in (reference, distorted):
        h, w = y.shape
        up = lambda p: np.repeat(np.repeat(p.astype(np.float64), 2, 0), 2, 1)[:h, :w]  # noqa: E731
        labs.append(_lab(y.astype(np.float64), up(u), up(v), bits))
    mean = float(np.sum(_de00(*labs[0], *labs[1]), dtype=np.float64)) / (labs[0][0].size)
    return 45.0 - 20.0 * math.log10(mean) if mean > 0 else math.inf


# ------------------------------------------------- the frames

def synthetic_pairs(width: int, height: int, bits: int, rng: np.random.Generator):
    """(name, reference, distorted) 4:2:0 pairs."""
    top = (1 << bits) - 1
    dtype = np.uint8 if bits == 8 else np.uint16
    cw, ch = (width + 1) // 2, (height + 1) // 2

    def frame(y, u, v):
        return tuple(np.clip(p, 0, top).astype(dtype) for p in (y, u, v))

    def noise(scale):
        return rng.integers(-scale, scale + 1, (height, width)), rng.integers(-scale, scale + 1, (ch, cw)), \
            rng.integers(-scale, scale + 1, (ch, cw))

    shift = 1 << (bits - 8)
    base = frame(rng.integers(0, top + 1, (height, width)), rng.integers(0, top + 1, (ch, cw)),
                 rng.integers(0, top + 1, (ch, cw)))
    n = noise(4 * shift)
    yield "noise", base, frame(base[0] + n[0], base[1] + n[1], base[2] + n[2])
    # Near lossless (mean dE00 about 0.07): a small mean magnifies any bias of the GPU's arithmetic in the score.
    smooth = frame(np.add.outer(np.arange(height), np.arange(width)) * shift // 2 + 64 * shift,
                   np.full((ch, cw), 120 * shift), np.full((ch, cw), 136 * shift))
    flips = (rng.random((height, width)) < 0.1).astype(np.int64)
    yield "near lossless", smooth, frame(smooth[0] + flips * shift, smooth[1], smooth[2])
    grey = frame(np.full((height, width), 128 * shift), np.full((ch, cw), 128 * shift), np.full((ch, cw), 128 * shift))
    yield "grey", grey, frame(grey[0] + 3 * shift, grey[1], grey[2])
    black = frame(np.zeros((height, width)), np.zeros((ch, cw)), np.zeros((ch, cw)))
    white = frame(np.full((height, width), top), np.full((ch, cw), top), np.full((ch, cw), top))
    yield "black/white", black, white
    yield "identical", base, base


def decoded_frames(path: str, width: int, height: int, bits: int, count: int):
    """The video's first frames as (Y, U, V) arrays, decoded by FFmpeg into a pipe."""
    fmt, dtype = ("yuv420p", np.uint8) if bits == 8 else (f"yuv420p{bits}le", np.uint16)
    cw, ch = (width + 1) // 2, (height + 1) // 2
    size = (width * height + 2 * cw * ch) * np.dtype(dtype).itemsize
    command = ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", fmt, "-"]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        for _ in range(count):
            data = process.stdout.read(size)
            if len(data) < size:
                return
            flat = np.frombuffer(data, dtype)
            yield (flat[:width * height].reshape(height, width),
                   flat[width * height:width * height + cw * ch].reshape(ch, cw),
                   flat[width * height + cw * ch:].reshape(ch, cw))


def video_size(path: str) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    width, height = out.strip().split(",")[:2]
    return int(width), int(height)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--clips", nargs="*", default=[], metavar="ARG",
                        help="groups of five: reference, distorted, bit depth, frames, libvmaf's score file")
    args = parser.parse_args()
    lib = vulkan._load()
    _bind(lib)
    rng = np.random.default_rng(1)
    worst, failed = 0.0, False
    for width, height, bits in ((33, 35, 8), (64, 48, 10), (37, 32, 12), (320, 180, 10)):
        pairs = list(synthetic_pairs(width, height, bits, rng))
        gpu = GpuCiede(lib, args.device, width, height, bits)
        try:
            for _name, ref, dis in pairs:
                gpu.add(ref, dis)
            scores = gpu.scores()
        finally:
            gpu.close()
        for (name, ref, dis), score in zip(pairs, scores):
            expected = reference_score(ref, dis, bits)
            diff = 0.0 if expected == score else abs(score - expected)
            ok = diff <= TOLERANCE
            worst = max(worst, diff if math.isfinite(diff) else math.inf)
            failed |= not ok
            print(f"{width}x{height} {bits}-bit {name:12s} GPU {score:.9f} ciede.c {expected:.9f}"
                  f"{'' if ok else '  FAILED'}")
    groups = [args.clips[i:i + 5] for i in range(0, len(args.clips), 5)]
    for ref_path, dis_path, bits_text, count_text, score_path in groups:
        bits, count = int(bits_text), int(count_text)
        width, height = video_size(ref_path)
        expected = [float(line.split()[-1]) for line in open(score_path) if line.strip()][:count]
        gpu = GpuCiede(lib, args.device, width, height, bits)
        try:
            for ref, dis in zip(decoded_frames(ref_path, width, height, bits, count),
                                decoded_frames(dis_path, width, height, bits, count)):
                gpu.add(ref, dis)
            scores = gpu.scores()
        finally:
            gpu.close()
        diffs = np.abs(scores[:len(expected)] - np.array(expected))
        worst = max(worst, float(diffs.max()))
        failed |= bool(diffs.max() > TOLERANCE) or len(scores) != len(expected)
        print(f"{pathlib.Path(dis_path).name}: {len(scores)} frames, max |GPU - libvmaf| {diffs.max():.3g}, "
              f"mean {diffs.mean():.3g}")
    print(f"worst difference {worst:.3g}: {'FAILED' if failed else 'ok'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
