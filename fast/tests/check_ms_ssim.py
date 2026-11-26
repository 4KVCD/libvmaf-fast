"""Checks MS-SSIM on the GPU (vv_v1_ms_ssim, shaders/ms_ssim.slang) against libvmaf's float_ms_ssim.

The same 4:2:0 frame pairs go to both: synthetic ones (8, 10 and 12 bits, odd and even sizes, down to the
smallest MS-SSIM takes; noise, a near-lossless gradient, flat, black against white, identical pictures) and, with
--clips, the first frames of real video pairs, decoded by FFmpeg into a pipe. Each frame's MS-SSIM of Y, Cb and
Cr (vv_submit_v1), and of Y alone (vv_submit), is expected within 1e-6 of float_ms_ssim's (enable_chroma). Needs a GPU with Vulkan, fast/dist's
vmaf_vulkan.dll and a libvmaf.dll with float_ms_ssim's enable_chroma (--libvmaf).

    python fast/tests/check_ms_ssim.py [--device N] [--libvmaf DLL] [--clips REF TEST BITS FRAMES ...]
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
from vmaf_fast import DIST, vulkan  # noqa: E402

TOLERANCE = 1e-6
_V1_OPTIONS = (3.0, 1080, 0, 0.03125, 1.0, 0.0, 10000.0, 0, 0)  # vv_create_v1's nine (unused: no motion, no ADM)
_ONLY_EXTRAS = (1 << 16) | (1 << 18)  # vv_create's flag bits: no motion, no ADM
_PLANES = ("float_ms_ssim", "float_ms_ssim_cb", "float_ms_ssim_cr")


def _bind(lib: ctypes.CDLL) -> None:
    lib.vv_create_v1_ex.restype = ctypes.c_int
    lib.vv_create_v1_ex.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.vv_v1_ms_ssim.restype = ctypes.c_int
    lib.vv_v1_ms_ssim.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vv_v1_ms_ssim_scores.restype = ctypes.c_int
    lib.vv_v1_ms_ssim_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint,
                                         ctypes.c_int]
    lib.vv_submit_v1.restype = ctypes.c_int
    lib.vv_submit_v1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_ssize_t),
                                 ctypes.c_int]
    lib.vv_submit.restype = ctypes.c_int
    lib.vv_submit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_ssize_t,
                              ctypes.c_int]


class GpuMsSsim:
    """A context that calculates MS-SSIM alone: of Y, Cb and Cr from vv_submit_v1 (`planes` 3), or of Y from
    vv_submit (1); frames are (Y, U, V) arrays."""

    def __init__(self, lib: ctypes.CDLL, device: int, width: int, height: int, bits: int, planes: int) -> None:
        self.lib, self.frames, self.planes = lib, 0, planes
        self.context = ctypes.c_void_p()
        options = (ctypes.c_double * 9)(*_V1_OPTIONS)
        vulkan._check(lib, lib.vv_create_v1_ex(ctypes.byref(self.context), device, width, height, bits,
                                               _ONLY_EXTRAS, options, 9), "vv_create_v1_ex")
        try:
            vulkan._check(lib, lib.vv_v1_ms_ssim(self.context, planes), "vv_v1_ms_ssim")
        except Exception:
            lib.vv_destroy(self.context)
            raise

    def add(self, reference, distorted) -> None:
        planes = [np.ascontiguousarray(p) for p in (*reference, *distorted)]
        if self.planes == 1:
            ref, dis = planes[0], planes[3]
            code = self.lib.vv_submit(self.context, ref.ctypes.data, ref.strides[0], dis.ctypes.data,
                                      dis.strides[0], 1)
        else:
            pointers = (ctypes.c_void_p * 6)(*(p.ctypes.data for p in planes))
            strides = (ctypes.c_ssize_t * 6)(*(p.strides[0] for p in planes))
            code = self.lib.vv_submit_v1(self.context, pointers, strides, 1)
        vulkan._check(self.lib, code, "vv_submit")
        self.frames += 1

    def scores(self) -> np.ndarray:
        """[frame, plane]"""
        vulkan._check(self.lib, self.lib.vv_flush(self.context), "vv_flush")
        out = np.zeros((self.planes, self.frames))
        for plane in range(self.planes):
            count = self.lib.vv_v1_ms_ssim_scores(self.context, out[plane].ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)), self.frames, plane)
            vulkan._check(self.lib, min(count, 0), "vv_v1_ms_ssim_scores")
        return out.T

    def close(self) -> None:
        self.lib.vv_destroy(self.context)


# ------------------------------------------------- float_ms_ssim on the CPU (libvmaf's C API)

class _Configuration(ctypes.Structure):
    _fields_ = [("log_level", ctypes.c_int), ("n_threads", ctypes.c_uint), ("n_subsample", ctypes.c_uint),
                ("cpumask", ctypes.c_uint64), ("gpumask", ctypes.c_uint64)]


class _Picture(ctypes.Structure):
    _fields_ = [("pix_fmt", ctypes.c_int), ("bpc", ctypes.c_uint), ("w", ctypes.c_uint * 3),
                ("h", ctypes.c_uint * 3), ("stride", ctypes.c_ssize_t * 3), ("data", ctypes.c_void_p * 3),
                ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p), ("color", ctypes.c_int * 4)]


def cpu_scores(path: pathlib.Path, pairs, width: int, height: int, bits: int) -> np.ndarray:
    """float_ms_ssim's Y, Cb and Cr of each pair, [frame, plane]."""
    lib = ctypes.CDLL(str(path))
    handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    lib.vmaf_init.argtypes = [pointer, _Configuration]
    lib.vmaf_feature_dictionary_set.argtypes = [pointer, ctypes.c_char_p, ctypes.c_char_p]
    lib.vmaf_use_feature.argtypes = [handle, ctypes.c_char_p, handle]
    lib.vmaf_picture_alloc.argtypes = [ctypes.POINTER(_Picture), ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint]
    lib.vmaf_read_pictures.argtypes = [handle, ctypes.POINTER(_Picture), ctypes.POINTER(_Picture), ctypes.c_uint]
    lib.vmaf_feature_score_at_index.argtypes = [handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double),
                                                ctypes.c_uint]
    lib.vmaf_close.argtypes = [handle]
    vmaf = ctypes.c_void_p()
    if lib.vmaf_init(ctypes.byref(vmaf), _Configuration(1, 1, 1, 0, 0)):
        raise RuntimeError("vmaf_init failed")
    options = ctypes.c_void_p()
    lib.vmaf_feature_dictionary_set(ctypes.byref(options), b"enable_chroma", b"true")
    if lib.vmaf_use_feature(vmaf, b"float_ms_ssim", options):
        raise RuntimeError("float_ms_ssim with enable_chroma is not in this libvmaf")
    for index, (ref, dis) in enumerate(pairs):
        pictures = [_Picture(), _Picture()]
        for picture, frame in zip(pictures, (ref, dis)):
            if lib.vmaf_picture_alloc(ctypes.byref(picture), 1, bits, width, height):  # VMAF_PIX_FMT_YUV420P
                raise RuntimeError("vmaf_picture_alloc failed")
            for p, plane in enumerate(frame):  # (an odd size's chroma: libvmaf's w / 2 x h / 2 of it)
                rows = np.ctypeslib.as_array(ctypes.cast(picture.data[p], ctypes.POINTER(ctypes.c_uint8)),
                                             (picture.h[p], picture.stride[p]))
                part = np.ascontiguousarray(plane[:picture.h[p], :picture.w[p]]).view(np.uint8)
                rows[:, :part.shape[1]] = part
        if lib.vmaf_read_pictures(vmaf, ctypes.byref(pictures[0]), ctypes.byref(pictures[1]), index):
            raise RuntimeError("vmaf_read_pictures failed")
    lib.vmaf_read_pictures(vmaf, None, None, 0)  # flush
    out = np.zeros((len(pairs), 3))
    score = ctypes.c_double()
    for index in range(len(pairs)):
        for p, name in enumerate(_PLANES):
            if lib.vmaf_feature_score_at_index(vmaf, name.encode(), ctypes.byref(score), index):
                raise RuntimeError(f"no {name} for frame {index}")
            out[index, p] = score.value
    lib.vmaf_close(vmaf)
    return out


# ------------------------------------------------- the frames

def synthetic_pairs(width: int, height: int, bits: int, rng: np.random.Generator):
    """(name, reference, distorted) 4:2:0 pairs."""
    top = (1 << bits) - 1
    dtype = np.uint8 if bits == 8 else np.uint16
    cw, ch = (width + 1) // 2, (height + 1) // 2
    shift = 1 << (bits - 8)

    def frame(y, u, v):
        return tuple(np.clip(p, 0, top).astype(dtype) for p in (y, u, v))

    def noise(scale):
        return tuple(rng.integers(-scale, scale + 1, shape) for shape in ((height, width), (ch, cw), (ch, cw)))

    base = frame(*(rng.integers(0, top + 1, shape) for shape in ((height, width), (ch, cw), (ch, cw))))
    n = noise(8 * shift)
    yield "noise", base, frame(*(b + d for b, d in zip(base, n)))
    gradient = frame(np.add.outer(np.arange(height), np.arange(width)) * shift // 2 + 16 * shift,
                     np.add.outer(np.arange(ch), np.arange(cw)) * shift + 32 * shift,
                     np.add.outer(np.arange(ch), -np.arange(cw)) * shift + 200 * shift)
    flips = tuple((rng.random(shape) < 0.1).astype(np.int64) * shift for shape in ((height, width), (ch, cw), (ch, cw)))
    yield "near lossless", gradient, frame(*(g + f for g, f in zip(gradient, flips)))
    grey = frame(*(np.full(shape, 128 * shift) for shape in ((height, width), (ch, cw), (ch, cw))))
    yield "flat", grey, frame(grey[0] + 3 * shift, grey[1] - 2 * shift, grey[2])
    black = frame(*(np.zeros(shape) for shape in ((height, width), (ch, cw), (ch, cw))))
    white = frame(*(np.full(shape, top) for shape in ((height, width), (ch, cw), (ch, cw))))
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


def compare(lib, args, label: str, names, pairs, width: int, height: int, bits: int) -> float:
    """Prints each frame's scores (Y, Cb, Cr, then Y alone); returns the largest difference (inf where one failed
    to match)."""
    got = []
    for planes in (3, 1):
        gpu = GpuMsSsim(lib, args.device, width, height, bits, planes)
        try:
            for ref, dis in pairs:
                gpu.add(ref, dis)
            got.append(gpu.scores())
        finally:
            gpu.close()
    got = np.concatenate(got, axis=1)
    expected = cpu_scores(args.libvmaf, pairs, width, height, bits)
    expected = np.concatenate([expected, expected[:, :1]], axis=1)
    worst = 0.0
    for name, g, e in zip(names, got, expected):
        diff = np.where(g == e, 0.0, np.abs(g - e))
        frame_worst = float(np.max(diff)) if np.all(np.isfinite(diff)) else math.inf
        worst = max(worst, frame_worst)
        print(f"{label} {name:14s} GPU {' '.join(f'{v:.9f}' for v in g)}  CPU {' '.join(f'{v:.9f}' for v in e)}"
              f"  max diff {frame_worst:.2e}{'' if frame_worst <= TOLERANCE else '  FAILED'}")
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--libvmaf", type=pathlib.Path, default=DIST / "libvmaf" / "libvmaf.dll")
    parser.add_argument("--clips", nargs="*", default=[], metavar="ARG",
                        help="groups of four: reference, distorted, bit depth, frames")
    args = parser.parse_args()
    lib = vulkan._load()
    _bind(lib)
    rng = np.random.default_rng(1)
    worst = 0.0
    # (the smallest: chroma 176 x 176, MS-SSIM's least at five scales; libvmaf's chroma is w / 2 x h / 2)
    for width, height, bits in ((353, 352, 8), (401, 377, 10), (512, 384, 12), (1001, 577, 8)):
        pairs = list(synthetic_pairs(width, height, bits, rng))
        worst = max(worst, compare(lib, args, f"{width}x{height} {bits}-bit", [p[0] for p in pairs],
                                   [(p[1], p[2]) for p in pairs], width, height, bits))
    for i in range(0, len(args.clips), 4):
        ref_path, dis_path, bits_text, count_text = args.clips[i:i + 4]
        bits, count = int(bits_text), int(count_text)
        width, height = video_size(ref_path)
        pairs = list(zip(decoded_frames(ref_path, width, height, bits, count),
                         decoded_frames(dis_path, width, height, bits, count)))
        worst = max(worst, compare(lib, args, pathlib.Path(dis_path).name, [f"frame {n}" for n in range(len(pairs))],
                                   pairs, width, height, bits))
    print(f"largest difference {worst:.3e} ({'ok' if worst <= TOLERANCE else 'FAILED'}, tolerance {TOLERANCE:g})")
    return 0 if worst <= TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
