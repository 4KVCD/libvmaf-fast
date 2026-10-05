"""libvmaf's CUDA build (fast/scripts/build_libvmaf_cuda.ps1): the binding to
its C API and GpuScorer, VMAF and VMAF NEG of frame pairs on an NVIDIA GPU.
Also the libvmaf the Vulkan scorers predict with and compare against."""
from __future__ import annotations

import ctypes
import os
from collections.abc import Callable

import numpy as np

from . import DIST

LIBRARY_PATH = DIST / "libvmaf" / "libvmaf.dll"

_VMAF_PIX_FMT_YUV420P = 1
_VMAF_LOG_LEVEL_ERROR = 1
#: Pictures libvmaf holds in its pool: enough for the feeder to stay ahead of
#: the GPU without holding more host memory than it needs (25 MB each at
#: 4K 10-bit).
_PICTURES = 8

class VmafGpuError(RuntimeError):
    """VMAF could not be calculated on the GPU; the CPU calculates it."""


# ------------------------------------------------------------------ binding

class _Configuration(ctypes.Structure):
    _fields_ = [("log_level", ctypes.c_int), ("n_threads", ctypes.c_uint), ("n_subsample", ctypes.c_uint),
                ("cpumask", ctypes.c_uint64), ("gpumask", ctypes.c_uint64)]


class _Picture(ctypes.Structure):
    _fields_ = [("pix_fmt", ctypes.c_int), ("bpc", ctypes.c_uint), ("w", ctypes.c_uint * 3),
                ("h", ctypes.c_uint * 3), ("stride", ctypes.c_ssize_t * 3), ("data", ctypes.c_void_p * 3),
                # VmafColor (range, primaries, transfer, matrix), in the structure since upstream
                # 0497a0f2 (vmaf_picture_convert). Without it libvmaf writes behind the picture.
                ("color", ctypes.c_int * 4),
                ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p)]


class _PictureParameters(ctypes.Structure):
    _fields_ = [("w", ctypes.c_uint), ("h", ctypes.c_uint), ("bpc", ctypes.c_uint), ("pix_fmt", ctypes.c_int)]


class _PictureConfiguration(ctypes.Structure):
    _fields_ = [("pic_params", _PictureParameters), ("pic_cnt", ctypes.c_uint)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("flags", ctypes.c_uint64)]


class _CudaConfiguration(ctypes.Structure):
    _fields_ = [("cu_ctx", ctypes.c_void_p)]


class _CudaPictureConfiguration(ctypes.Structure):
    _fields_ = [("pic_params", _PictureParameters), ("pic_prealloc_method", ctypes.c_int)]


#: VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE: libvmaf's own pictures in
#: GPU memory, which the decoded frames are copied into on the GPU.
_PREALLOCATE_ON_DEVICE = 1


_library: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _library
    if _library is None:
        lib = ctypes.CDLL(str(LIBRARY_PATH))
        handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        for name, restype, argtypes in (
            ("vmaf_version", ctypes.c_char_p, []),
            ("vmaf_init", ctypes.c_int, [pointer, _Configuration]),
            ("vmaf_cuda_state_init", ctypes.c_int, [pointer, _CudaConfiguration]),
            ("vmaf_cuda_import_state", ctypes.c_int, [handle, handle]),
            ("vmaf_model_load", ctypes.c_int, [pointer, ctypes.POINTER(_ModelConfig), ctypes.c_char_p]),
            ("vmaf_use_features_from_model", ctypes.c_int, [handle, handle]),
            ("vmaf_preallocate_pictures", ctypes.c_int, [handle, _PictureConfiguration]),
            ("vmaf_fetch_preallocated_picture", ctypes.c_int, [handle, ctypes.POINTER(_Picture)]),
            ("vmaf_cuda_preallocate_pictures", ctypes.c_int, [handle, _CudaPictureConfiguration]),
            ("vmaf_cuda_fetch_preallocated_picture", ctypes.c_int, [handle, ctypes.POINTER(_Picture)]),
            ("vmaf_read_pictures", ctypes.c_int,
             [handle, ctypes.POINTER(_Picture), ctypes.POINTER(_Picture), ctypes.c_uint]),
            ("vmaf_score_at_index", ctypes.c_int, [handle, handle, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]),
            ("vmaf_picture_unref", ctypes.c_int, [ctypes.POINTER(_Picture)]),
            ("vmaf_model_destroy", None, [handle]),
            ("vmaf_close", ctypes.c_int, [handle]),
        ):
            function = getattr(lib, name)
            function.restype, function.argtypes = restype, argtypes
        _library = lib
    return _library


def _check(error: int, what: str) -> None:
    if error:
        raise VmafGpuError(f"{what} failed (libvmaf error {error})")


def path_bytes(path) -> bytes:
    """An existing file's path as libvmaf opens it: with fopen(), which reads
    the name in Windows' ANSI code page, not UTF-8. UTF-8 bytes of a folder
    with an accented letter (a user's own name in their profile and its Temp
    folder) named a file that is not there. A name the code page has no
    letters for (Chinese on a western Windows) goes by its short 8.3 path
    where the drive keeps those; VmafGpuError where it does not."""
    text = str(path)
    for attempt in range(2):
        try:
            return text.encode("mbcs" if os.name == "nt" else "utf-8", errors="strict")
        except UnicodeEncodeError:
            if attempt:
                break
            short = ctypes.create_unicode_buffer(32768)
            if not ctypes.windll.kernel32.GetShortPathNameW(text, short, len(short)):
                break
            text = short.value
    raise VmafGpuError(f"libvmaf cannot open {path}: its name has letters outside this PC's code page")


class GpuScorer:
    """One libvmaf context on the GPU, given frame pairs in order: the luma
    and chroma planes of 4:2:0 frames, packed as FFmpeg's rawvideo writes
    them, at `bit_depth` bits (16-bit little-endian samples above 8) --
    or, with `on_device`, pictures in GPU memory that add_on_device has
    filled (their luma: VMAF reads nothing else)."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1,
                 on_device: bool = False):
        self._lib = lib = _load()
        self._on_device = on_device
        self._step = max(1, n_subsample)
        self._context = ctypes.c_void_p()
        self._models: dict[str, ctypes.c_void_p] = {}
        self._count = 0
        configuration = _Configuration(_VMAF_LOG_LEVEL_ERROR, 0, self._step, 0, 0)
        _check(lib.vmaf_init(ctypes.byref(self._context), configuration), "Starting libvmaf")
        try:
            state = ctypes.c_void_p()
            _check(lib.vmaf_cuda_state_init(ctypes.byref(state), _CudaConfiguration(None)), "Starting CUDA")
            _check(lib.vmaf_cuda_import_state(self._context, state), "Starting CUDA")
            for name, version in models.items():
                model = ctypes.c_void_p()
                config = _ModelConfig(name.encode(), 0)
                _check(lib.vmaf_model_load(ctypes.byref(model), ctypes.byref(config), version.encode()),
                       f"Loading the {version} model")
                self._models[name] = model
                _check(lib.vmaf_use_features_from_model(self._context, model), f"Setting up {version}")
            parameters = _PictureParameters(width, height, bit_depth, _VMAF_PIX_FMT_YUV420P)
            if on_device:
                pictures = _CudaPictureConfiguration(parameters, _PREALLOCATE_ON_DEVICE)
                _check(lib.vmaf_cuda_preallocate_pictures(self._context, pictures), "Allocating pictures")
            else:
                _check(lib.vmaf_preallocate_pictures(self._context, _PictureConfiguration(parameters, _PICTURES)),
                       "Allocating pictures")
        except BaseException:
            self.close()
            raise
        self._sample = sample = 1 if bit_depth <= 8 else 2
        chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
        #: (plane offset in a frame, rows, bytes per row) for Y, U and V.
        self._planes = [(0, height, width * sample)]
        offset = width * height * sample
        for _ in range(2):
            self._planes.append((offset, chroma_h, chroma_w * sample))
            offset += chroma_w * chroma_h * sample
        self.frame_bytes = offset

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        """Scores one more pair (frame index = how many came before)."""
        self._add(lambda picture: self._fill(picture, reference), lambda picture: self._fill(picture, distorted))

    def add_on_device(self, reference: Callable[[int, int], None], distorted: Callable[[int, int], None]) -> None:
        """Scores one more pair of pictures in GPU memory: `reference` and
        `distorted` are each given a picture's luma plane (address, pitch)
        to fill, and return once it is filled."""
        self._add(lambda picture: reference(picture.data[0], picture.stride[0]),
                  lambda picture: distorted(picture.data[0], picture.stride[0]))

    def _add(self, fill_reference, fill_distorted) -> None:
        fetch = (self._lib.vmaf_cuda_fetch_preallocated_picture if self._on_device
                 else self._lib.vmaf_fetch_preallocated_picture)
        ref, dist = _Picture(), _Picture()
        _check(fetch(self._context, ctypes.byref(ref)), "Taking a picture")
        try:
            _check(fetch(self._context, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            self._lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        try:
            fill_reference(ref)
            fill_distorted(dist)
        except BaseException:
            # Pictures taken from the pool and never handed over keep
            # vmaf_close waiting for them.
            self._lib.vmaf_picture_unref(ctypes.byref(ref))
            self._lib.vmaf_picture_unref(ctypes.byref(dist))
            raise
        # libvmaf takes both pictures, also when it fails (pull request 1652).
        _check(self._lib.vmaf_read_pictures(self._context, ctypes.byref(ref), ctypes.byref(dist), self._count),
               f"Scoring frame {self._count}")
        self._count += 1

    def _fill(self, picture: _Picture, frame: bytearray) -> None:
        source = np.frombuffer(frame, dtype=np.uint8)
        for plane, (offset, rows, row_bytes) in enumerate(self._planes):
            stride = picture.stride[plane]
            # Of an odd size FFmpeg rounds the chroma planes up and libvmaf
            # down (w >> 1, h >> 1): FFmpeg's last row and last sample have
            # no place in the picture. VMAF reads the luma plane only.
            kept_rows = min(rows, picture.h[plane])
            kept_bytes = min(row_bytes, picture.w[plane] * self._sample)
            target = np.ctypeslib.as_array(
                ctypes.cast(picture.data[plane], ctypes.POINTER(ctypes.c_uint8)), shape=(kept_rows, stride))
            plane_rows = source[offset:offset + rows * row_bytes].reshape(rows, row_bytes)
            target[:, :kept_bytes] = plane_rows[:kept_rows, :kept_bytes]

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """The frame numbers scored (every n_subsample-th) and each model's
        scores for them, as libvmaf's JSON log rounds them: six decimals.
        Nothing when no pair came (the caller refuses that)."""
        if not self._count:
            return np.zeros(0, dtype=np.int32), {name: np.zeros(0) for name in self._models}
        _check(self._lib.vmaf_read_pictures(self._context, None, None, 0), "Finishing")
        frames = np.arange(0, self._count, self._step, dtype=np.int32)
        scores = {}
        value = ctypes.c_double()
        for name, model in self._models.items():
            column = np.empty(len(frames), dtype=np.float64)
            for slot, frame in enumerate(frames):
                _check(self._lib.vmaf_score_at_index(self._context, model, ctypes.byref(value), int(frame)),
                       f"Reading {name} for frame {frame}")
                column[slot] = float(f"{value.value:.6f}")
            scores[name] = column
        return frames, scores

    def close(self) -> None:
        for model in self._models.values():
            self._lib.vmaf_model_destroy(model)
        self._models.clear()
        if self._context:
            self._lib.vmaf_close(self._context)
            self._context = ctypes.c_void_p()


def probe() -> tuple[bool, str]:
    """Whether libvmaf can score on this GPU: starts CUDA and scores three
    small frame pairs (the kernels load only then; a GPU older than the
    build supports fails here)."""
    if not LIBRARY_PATH.is_file():
        return False, "libvmaf with CUDA is not built"
    try:
        lib = _load()
        version = (lib.vmaf_version() or b"").decode()
        width, height = 256, 144
        scorer = GpuScorer(width, height, 8, {"vmaf": "vmaf_v0.6.1"})
        try:
            rng = np.random.default_rng(1)
            base = rng.integers(16, 235, scorer.frame_bytes, dtype=np.uint8)
            for shift in range(3):
                reference = bytearray(np.roll(base, shift).tobytes())
                distorted = bytearray(np.clip(np.roll(base, shift).astype(np.int16) + 3, 0, 255).astype(np.uint8))
                scorer.add(reference, distorted)
            _frames, scores = scorer.finish()
        finally:
            scorer.close()
        if not np.all(np.isfinite(scores["vmaf"])):
            return False, "libvmaf's GPU test scored nothing"
        return True, f"libvmaf {version}"
    except (OSError, VmafGpuError) as error:
        return False, str(error)
