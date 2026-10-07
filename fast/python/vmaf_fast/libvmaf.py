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
            function = getattr(lib, name, None)
            if function is None and name.startswith("vmaf_cuda_"):
                # libvmaf built without CUDA (build_libvmaf_cuda.ps1 -NoCuda): its CPU code and the
                # predictions vulkan and v1 make with it work; a GpuScorer fails on the missing function.
                continue
            if function is None:
                raise AttributeError(f"{LIBRARY_PATH.name} has no {name}")
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


# ------------------------------------------------- frames from system memory

class _Copy2D(ctypes.Structure):
    """CUDA_MEMCPY2D."""

    _fields_ = [("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t), ("srcMemoryType", ctypes.c_uint),
                ("srcHost", ctypes.c_void_p), ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
                ("srcPitch", ctypes.c_size_t),
                ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t), ("dstMemoryType", ctypes.c_uint),
                ("dstHost", ctypes.c_void_p), ("dstDevice", ctypes.c_uint64), ("dstArray", ctypes.c_void_p),
                ("dstPitch", ctypes.c_size_t),
                ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t)]


_CU_MEMORYTYPE_HOST, _CU_MEMORYTYPE_DEVICE = 1, 2
_CU_MEMHOSTALLOC_PORTABLE = 1
_CU_STREAM_NON_BLOCKING = 1

_cuda_library: ctypes.CDLL | None = None


def _cuda() -> ctypes.CDLL:
    """CUDA's driver API (nvcuda.dll, the NVIDIA driver's), which libvmaf loads too."""
    global _cuda_library
    if _cuda_library is None:
        lib = ctypes.CDLL("nvcuda.dll")
        handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        for name, argtypes in (
            ("cuInit", [ctypes.c_uint]),
            ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
            ("cuDevicePrimaryCtxRetain", [pointer, ctypes.c_int]),
            ("cuDevicePrimaryCtxRelease_v2", [ctypes.c_int]),
            ("cuCtxPushCurrent_v2", [handle]),
            ("cuCtxPopCurrent_v2", [pointer]),
            ("cuMemHostAlloc", [pointer, ctypes.c_size_t, ctypes.c_uint]),
            ("cuMemFreeHost", [handle]),
            ("cuStreamCreate", [pointer, ctypes.c_uint]),
            ("cuStreamDestroy_v2", [handle]),
            ("cuMemcpy2DAsync_v2", [ctypes.POINTER(_Copy2D), handle]),
            ("cuStreamSynchronize", [handle]),
        ):
            function = getattr(lib, name)
            function.restype, function.argtypes = ctypes.c_int, argtypes
        _cuda_library = lib
    return _cuda_library


def _cuda_check(error: int, what: str) -> None:
    if error:
        raise VmafGpuError(f"{what} failed (CUDA error {error})")


class _HostUpload:
    """Copies frames' luma from system memory into libvmaf's pictures on the
    GPU, through one page-locked buffer a side, allocated once, on a stream of
    its own (as VideoMetricsLab does). libvmaf's own way with frames in system
    memory, a pool of CPU pictures each uploaded from pageable memory when it
    is read, uploads the chroma planes VMAF does not read too: about 160 4K
    10-bit pairs a second on an RTX 5090. In libvmaf's CUDA context, the
    primary one of GPU 0, made current around each use."""

    def __init__(self, width: int, height: int, sample: int):
        self._cu = cu = _cuda()
        self._row_bytes, self._rows = width * sample, height
        self.luma_bytes = self._row_bytes * height
        self._device = ctypes.c_int()
        self._context = ctypes.c_void_p()
        self._stream = ctypes.c_void_p()
        self._staging = [ctypes.c_void_p(), ctypes.c_void_p()]
        _cuda_check(cu.cuInit(0), "Starting CUDA")
        _cuda_check(cu.cuDeviceGet(ctypes.byref(self._device), 0), "Finding the GPU")
        _cuda_check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(self._context), self._device), "Taking the GPU")
        try:
            with self:
                # Non-blocking: waited for by itself, not with libvmaf's own work.
                _cuda_check(cu.cuStreamCreate(ctypes.byref(self._stream), _CU_STREAM_NON_BLOCKING),
                            "Starting the upload")
                for buffer in self._staging:
                    _cuda_check(cu.cuMemHostAlloc(ctypes.byref(buffer), self.luma_bytes, _CU_MEMHOSTALLOC_PORTABLE),
                                "Allocating the upload's memory")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> _HostUpload:  # noqa: PYI034 (the class is private)
        _cuda_check(self._cu.cuCtxPushCurrent_v2(self._context), "Taking the GPU")
        return self

    def __exit__(self, *_exc) -> None:
        self._cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))

    def send(self, side: int, frame, address: int, pitch: int) -> None:
        """Starts the copy of `frame`'s luma to the picture at `address` (rows
        `pitch` bytes apart); wait() says when `side` may be sent again."""
        source = np.frombuffer(frame, dtype=np.uint8)
        if source.size < self.luma_bytes:
            raise VmafGpuError(f"a frame of {source.size} bytes, where its luma alone is {self.luma_bytes}")
        staging = self._staging[side].value
        ctypes.memmove(staging, source.ctypes.data, self.luma_bytes)
        copy = _Copy2D(srcMemoryType=_CU_MEMORYTYPE_HOST, srcHost=staging, srcPitch=self._row_bytes,
                       dstMemoryType=_CU_MEMORYTYPE_DEVICE, dstDevice=address, dstPitch=pitch,
                       WidthInBytes=self._row_bytes, Height=self._rows)
        _cuda_check(self._cu.cuMemcpy2DAsync_v2(ctypes.byref(copy), self._stream), "Uploading a frame")

    def wait(self) -> None:
        _cuda_check(self._cu.cuStreamSynchronize(self._stream), "Uploading a frame")

    def close(self) -> None:
        if not self._context:
            return
        cu = self._cu
        if cu.cuCtxPushCurrent_v2(self._context) == 0:
            if self._stream:
                cu.cuStreamSynchronize(self._stream)
                cu.cuStreamDestroy_v2(self._stream)
            for buffer in self._staging:
                if buffer:
                    cu.cuMemFreeHost(buffer)
            cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        self._stream = ctypes.c_void_p()
        self._staging = [ctypes.c_void_p(), ctypes.c_void_p()]
        cu.cuDevicePrimaryCtxRelease_v2(self._device)
        self._context = ctypes.c_void_p()


class GpuScorer:
    """One libvmaf context on the GPU, given frame pairs in order: 4:2:0
    frames packed as FFmpeg's rawvideo writes them, at `bit_depth` bits
    (16-bit little-endian samples above 8), whose luma is uploaded (add;
    VMAF reads nothing else) -- or, with `on_device`, pictures in GPU memory
    that add_on_device has filled, and nothing allocated for uploads."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1,
                 on_device: bool = False):
        self._lib = lib = _load()
        self._upload: _HostUpload | None = None
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
            # libvmaf's pictures in GPU memory, whoever fills them: _HostUpload
            # from frames in system memory, or the caller (add_on_device).
            parameters = _PictureParameters(width, height, bit_depth, _VMAF_PIX_FMT_YUV420P)
            pictures = _CudaPictureConfiguration(parameters, _PREALLOCATE_ON_DEVICE)
            _check(lib.vmaf_cuda_preallocate_pictures(self._context, pictures), "Allocating pictures")
            if not on_device:
                self._upload = _HostUpload(width, height, 1 if bit_depth <= 8 else 2)
        except BaseException:
            self.close()
            raise
        sample = 1 if bit_depth <= 8 else 2
        #: A frame as FFmpeg writes it: the luma, then the chroma planes, which FFmpeg rounds up for an odd size.
        self.frame_bytes = (width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)) * sample

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        """Scores one more pair (frame index = how many came before). The
        frames are the caller's again when it returns."""
        upload = self._upload
        if upload is None:
            raise VmafGpuError("this scorer takes pictures in GPU memory only")

        def fill_distorted(picture: _Picture) -> None:
            # Both copies are under way together; the reference's is waited
            # for too when this one could not be started.
            try:
                upload.send(1, distorted, picture.data[0], picture.stride[0])
            finally:
                upload.wait()

        with upload:
            self._add(lambda picture: upload.send(0, reference, picture.data[0], picture.stride[0]), fill_distorted)

    def add_on_device(self, reference: Callable[[int, int], None], distorted: Callable[[int, int], None]) -> None:
        """Scores one more pair of pictures in GPU memory: `reference` and
        `distorted` are each given a picture's luma plane (address, pitch)
        to fill, and return once it is filled."""
        self._add(lambda picture: reference(picture.data[0], picture.stride[0]),
                  lambda picture: distorted(picture.data[0], picture.stride[0]))

    def _add(self, fill_reference, fill_distorted) -> None:
        fetch = self._lib.vmaf_cuda_fetch_preallocated_picture
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
        if self._upload is not None:
            self._upload.close()
            self._upload = None


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
