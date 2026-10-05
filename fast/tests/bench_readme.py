"""The README's speed tables: VMAF v0.6.1 with VMAF NEG, and VMAF v1, on the
CPU, with libvmaf's CUDA code and with Vulkan, for frames decoded on the CPU
(in system memory) and frames decoded on the GPU (already in its memory, as
NVIDIA's hardware decoder leaves them).

    python fast/tests/bench_readme.py REFERENCE DISTORTED [--size WxH] [--bits 10]
        [--frames 48] [--start 2] [--pairs 480] [--vmaf-only] [--only NAME...]

The README's numbers: HoneyBee at 4K (the defaults), and --size 1920x1080
--pairs 960; each also with --vmaf-only (VMAF v0.6.1 without NEG). --only
runs the routes whose names start with the given words ("v0 cpu", "v1
hybrid", ...).

Frames are decoded by FFmpeg before anything is timed, then given to each
scorer in a loop (--pairs pairs, after a warm-up). Each scorer gets the best
path it has:

- CPU: libvmaf with pictures it allocates once and reuses
  (vmaf_preallocate_pictures), at 4, 8, 12, 16 and 24 threads.
- CUDA from system memory: libvmaf's pinned host pictures
  (VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED), which it uploads.
  Before this fork's fix (commit 8ebd5f5d) libvmaf allocated, page-locked
  and zeroed a new one for every frame: about 60 fps at 4K.
- CUDA from GPU memory: libvmaf's device pictures, filled by copies on the
  GPU from frames held in GPU memory.
- Vulkan from system memory: vv_submit (the luma copied into the engine's
  host-visible staging memory).
- Vulkan from GPU memory: the engine's input buffers imported into CUDA
  (vv_export, cuImportExternalMemory) and filled by copies on the GPU.
- VMAF v1 with the GPU: as Vulkan above for ADM3 and motion3; the planes
  CAMBI and SpEED read are copied into libvmaf's pictures from system memory,
  or downloaded into them from GPU memory (page-locked).

VMAF v0.6.1 and NEG read only the luma, and only the luma is given to their
scorers; VMAF v1's CAMBI and SpEED read the chroma too. Speed is pairs a
second; cores is the process's CPU time over the same span.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import compare_vmaf_vulkan as frames_of

from vmaf_fast import MODELS, v1
from vmaf_fast import libvmaf as lv
from vmaf_fast import vulkan as vk

V0_MODELS = {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}
V1_MODEL = MODELS / "vmaf_v1.0.16" / "vmaf_v1.0.16_3d0h.json"
_HOST_PINNED, _DEVICE = 3, 1


# ------------------------------------------------------------------ CUDA

class _Memcpy2D(ctypes.Structure):
    _fields_ = [("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t), ("srcMemoryType", ctypes.c_int),
                ("srcHost", ctypes.c_void_p), ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
                ("srcPitch", ctypes.c_size_t), ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t),
                ("dstMemoryType", ctypes.c_int), ("dstHost", ctypes.c_void_p), ("dstDevice", ctypes.c_uint64),
                ("dstArray", ctypes.c_void_p), ("dstPitch", ctypes.c_size_t), ("WidthInBytes", ctypes.c_size_t),
                ("Height", ctypes.c_size_t)]


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _ExternalMemoryDesc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("win32", _Win32Handle), ("size", ctypes.c_uint64),
                ("flags", ctypes.c_uint), ("reserved", ctypes.c_uint * 16)]


class _ExternalBufferDesc(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_uint64), ("size", ctypes.c_uint64), ("flags", ctypes.c_uint),
                ("reserved", ctypes.c_uint * 16)]


class Cuda:
    """The NVIDIA driver's primary context (libvmaf's too), for frames held in
    GPU memory and the copies a decoder would make."""

    def __init__(self):
        self.cu = cu = ctypes.WinDLL("nvcuda.dll")
        device, self.context = ctypes.c_int(), ctypes.c_void_p()
        self.check(cu.cuInit(0))
        self.check(cu.cuDeviceGet(ctypes.byref(device), 0))
        self.check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), device))
        self.check(cu.cuCtxPushCurrent_v2(self.context))
        # The copies' own stream: waiting for it waits for them alone, not
        # for libvmaf's CUDA work (as a decoder's output stream).
        self.stream = ctypes.c_void_p()
        self.check(cu.cuStreamCreate(ctypes.byref(self.stream), 1))  # CU_STREAM_NON_BLOCKING

    @staticmethod
    def check(result):
        if result:
            raise RuntimeError(f"CUDA error {result}")

    def upload(self, frame: bytearray) -> int:
        pointer = ctypes.c_uint64()
        self.check(self.cu.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(len(frame))))
        source = (ctypes.c_char * len(frame)).from_buffer(frame)
        self.check(self.cu.cuMemcpyHtoD_v2(pointer, source, ctypes.c_size_t(len(frame))))
        return pointer.value

    def copy(self, source: int, source_pitch: int, target: int, target_pitch: int, width: int, height: int,
             to_host: bool = False):
        c = _Memcpy2D(srcMemoryType=2, srcDevice=source, srcPitch=source_pitch, WidthInBytes=width, Height=height,
                      dstPitch=target_pitch)
        if to_host:
            c.dstMemoryType, c.dstHost = 1, target
        else:
            c.dstMemoryType, c.dstDevice = 2, target
        self.check(self.cu.cuMemcpy2DAsync_v2(ctypes.byref(c), self.stream))
        # The frame must be there before libvmaf or Vulkan reads it (a
        # decoder waits for its copy the same way).
        self.check(self.cu.cuStreamSynchronize(self.stream))


class GpuFrames:
    """Frames held in GPU memory, packed as in system memory (Y, U, V), with
    what vmaf_fast's shared scorers ask of a decoder: copy_luma,
    download_planes, pin, import_memory."""

    def __init__(self, cuda: Cuda, frames: list[bytearray], width: int, height: int, sample: int):
        self.cuda, self.width, self.height, self.sample = cuda, width, height, sample
        self.pointers = [cuda.upload(frame) for frame in frames]
        cw, ch = (width + 1) // 2, (height + 1) // 2
        self.planes = [(0, width * sample, height), (width * height * sample, cw * sample, ch),
                       (width * height * sample + cw * ch * sample, cw * sample, ch)]

    def copy_luma(self, slot, address, pitch):
        self.cuda.copy(self.pointers[slot], self.width * self.sample, address, pitch, self.width * self.sample,
                       self.height)

    def download_planes(self, slot, addresses, pitches):
        for (offset, row, rows), address, pitch in zip(self.planes, addresses, pitches, strict=True):
            if address:
                self.cuda.copy(self.pointers[slot] + offset, row, address, pitch, row, rows, to_host=True)

    def pin(self, address, size):
        return self.cuda.cu.cuMemHostRegister_v2(ctypes.c_void_p(address), ctypes.c_size_t(size), 0) == 0

    def unpin(self, address):
        self.cuda.cu.cuMemHostUnregister(ctypes.c_void_p(address))

    def import_memory(self, handle, size):
        memory = ctypes.c_void_p()
        description = _ExternalMemoryDesc(type=2, size=size, flags=1)  # opaque Win32 handle, dedicated
        description.win32.handle = handle
        if self.cuda.cu.cuImportExternalMemory(ctypes.byref(memory), ctypes.byref(description)):
            return None
        address = ctypes.c_uint64()
        buffer = _ExternalBufferDesc(offset=0, size=size)
        if self.cuda.cu.cuExternalMemoryGetMappedBuffer(ctypes.byref(address), memory, ctypes.byref(buffer)):
            return None
        return address.value, memory.value

    def unimport(self, memory):
        self.cuda.cu.cuDestroyExternalMemory(ctypes.c_void_p(memory))


# --------------------------------------------------------------- scorers

class LibvmafScorer:
    """libvmaf on the CPU (`cuda` None) or with its CUDA code, given pictures
    from its own pool: host (CPU), pinned host or device (CUDA)."""

    def __init__(self, width, height, bits, models, threads=0, cuda=None, frames=None, v1_model=None):
        self.lib = lib = v1._libvmaf()
        self.context, self.models, self.count = ctypes.c_void_p(), [], 0
        self.width, self.height, self.bits, self.frames = width, height, bits, frames
        lv._check(lib.vmaf_init(ctypes.byref(self.context), lv._Configuration(lv._VMAF_LOG_LEVEL_ERROR, threads, 1, 0, 0)),
                  "init")
        if cuda:
            state = ctypes.c_void_p()
            lv._check(lib.vmaf_cuda_state_init(ctypes.byref(state), lv._CudaConfiguration(None)), "CUDA")
            lv._check(lib.vmaf_cuda_import_state(self.context, state), "CUDA")
        for name, version in models.items():
            model = ctypes.c_void_p()
            lv._check(lib.vmaf_model_load(ctypes.byref(model), ctypes.byref(lv._ModelConfig(name.encode(), 0)),
                                          version.encode()), "model")
            lv._check(lib.vmaf_use_features_from_model(self.context, model), "features")
            self.models.append(model)
        if v1_model is not None:
            model = ctypes.c_void_p()
            lv._check(lib.vmaf_model_load_from_path(ctypes.byref(model), ctypes.byref(lv._ModelConfig(b"vmaf", 0)),
                                                    str(v1_model).encode()), "model")
            lv._check(lib.vmaf_use_features_from_model(self.context, model), "features")
            self.models.append(model)
        parameters = lv._PictureParameters(width, height, bits, lv._VMAF_PIX_FMT_YUV420P)
        self.cuda = cuda
        if cuda:
            lv._check(lib.vmaf_cuda_preallocate_pictures(self.context, lv._CudaPictureConfiguration(parameters, cuda)),
                      "pictures")
            self.fetch = lib.vmaf_cuda_fetch_preallocated_picture
        else:
            lv._check(lib.vmaf_preallocate_pictures(self.context, lv._PictureConfiguration(
                parameters, 2 * (max(threads, 1) + 2))), "pictures")
            self.fetch = lib.vmaf_fetch_preallocated_picture
        self.layout = v1.V1Scorer.__new__(v1.V1Scorer)
        self.layout._set_layout(width, height, bits)
        self.all_planes = v1_model is not None

    def add(self, ref, dis, ref_slot=None, dis_slot=None):
        pictures = lv._Picture(), lv._Picture()
        ref, dis = (ref, dis) if ref is not None else ("ref", "dis")
        for picture, frame, slot in zip(pictures, (ref, dis), (ref_slot, dis_slot), strict=True):
            lv._check(self.fetch(self.context, ctypes.byref(picture)), "picture")
            if self.cuda == _DEVICE:  # frames: (the reference's, the distorted's) in GPU memory
                (self.frames[0] if frame is ref else self.frames[1]).copy_luma(slot, picture.data[0], picture.stride[0])
            else:
                self.layout._fill(picture, frame, (0, 1, 2) if self.all_planes else (0,))
        lv._check(self.lib.vmaf_read_pictures(self.context, ctypes.byref(pictures[0]), ctypes.byref(pictures[1]),
                                              self.count), "read")
        self.count += 1

    def finish(self):
        lv._check(self.lib.vmaf_read_pictures(self.context, None, None, 0), "flush")
        value, scores = ctypes.c_double(), []
        for model in self.models:
            for frame in range(self.count):
                lv._check(self.lib.vmaf_score_at_index(self.context, model, ctypes.byref(value), frame), "score")
                scores.append(float(f"{value.value:.6f}"))  # as libvmaf's log, and the GPU scorers, round them
        return np.array(scores)

    def close(self):
        for model in self.models:
            self.lib.vmaf_model_destroy(model)
        self.lib.vmaf_close(self.context)


# ----------------------------------------------------------------- runs

def measure(make, feed, pairs, warmup=16):
    scorer = make()
    try:
        for index in range(warmup):
            feed(scorer, index)
        cpu, wall = time.process_time(), time.perf_counter()
        for index in range(warmup, warmup + pairs):
            feed(scorer, index)
        result = scorer.finish()
        cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    finally:
        scorer.close()
    return pairs / wall, cpu / wall, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("distorted")
    parser.add_argument("--size")
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--pairs", type=int, default=480)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--vmaf-only", action="store_true", help="VMAF v0.6.1 without NEG")
    arguments = parser.parse_args()
    global V0_MODELS
    if arguments.vmaf_only:
        V0_MODELS = {"vmaf": "vmaf_v0.6.1"}
    size = tuple(int(part) for part in arguments.size.split("x")) if arguments.size else None
    bits = arguments.bits
    ref, w, h = frames_of.decode(arguments.reference, arguments.start, arguments.frames, size, bits)
    dis, _, _ = frames_of.decode(arguments.distorted, arguments.start, arguments.frames, (w, h), bits)
    n = min(len(ref), len(dis))
    sample = 1 if bits == 8 else 2
    devices = {d.index: d.name for d in vk.devices() if d.usable and d.kind in (1, 2)}
    nvidia = next((i for i, name in devices.items() if "NVIDIA" in name), None)
    cuda = Cuda() if nvidia is not None else None
    gpu_ref = GpuFrames(cuda, ref[:n], w, h, sample) if cuda else None
    gpu_dis = GpuFrames(cuda, dis[:n], w, h, sample) if cuda else None
    print(f"{Path(arguments.distorted).name} against {Path(arguments.reference).name}: {w}x{h} {bits}-bit, "
          f"{n} frames in memory, {arguments.pairs} pairs timed")

    def host(scorer, index):
        scorer.add(ref[index % n], dis[index % n])

    def device(scorer, index):
        scorer.add(None, None, index % n, index % n)

    def shared_v0(scorer, index):
        slot = index % n
        scorer.add_shared(lambda a, p: gpu_ref.copy_luma(slot, a, p), lambda a, p: gpu_dis.copy_luma(slot, a, p))

    def shared_v1(scorer, index):
        scorer.add_decoded(gpu_ref, index % n, gpu_dis, index % n)

    v0 = "VMAF" if arguments.vmaf_only else "VMAF + NEG"
    runs = []
    for threads in (4, 8, 12, 16, 24):
        runs.append((f"v0 cpu {threads}", f"{v0}, libvmaf on the CPU, {threads} threads",
                     lambda t=threads: LibvmafScorer(w, h, bits, V0_MODELS, t), host))
    if cuda:
        runs.append(("v0 cuda host", f"{v0}, libvmaf's CUDA code, CPU-decoded frames",
                     lambda: LibvmafScorer(w, h, bits, V0_MODELS, cuda=_HOST_PINNED), host))
        runs.append(("v0 cuda gpu", f"{v0}, libvmaf's CUDA code, GPU-decoded frames",
                     lambda: LibvmafScorer(w, h, bits, V0_MODELS, cuda=_DEVICE, frames=(gpu_ref, gpu_dis)), device))
    for index, name in devices.items():
        runs.append((f"v0 vulkan host {index}", f"{v0}, Vulkan on {name}, CPU-decoded frames",
                     lambda i=index: vk.VulkanScorer(w, h, bits, V0_MODELS, device=i), host))
    if cuda:
        runs.append(("v0 vulkan gpu", f"{v0}, Vulkan on {devices[nvidia]}, GPU-decoded frames",
                     lambda: vk.VulkanScorer(w, h, bits, V0_MODELS, device=nvidia, shared=gpu_dis), shared_v0))
    for threads in (4, 8, 12, 16, 24):
        runs.append((f"v1 cpu {threads}", f"VMAF v1, libvmaf on the CPU, {threads} threads",
                     lambda t=threads: LibvmafScorer(w, h, bits, {}, t, v1_model=V1_MODEL), host))
    for index, name in devices.items():
        runs.append((f"v1 hybrid host {index}", f"VMAF v1, CPU + Vulkan on {name}, CPU-decoded frames",
                     lambda i=index: v1.V1Scorer(w, h, bits, V1_MODEL, device=i), host))
    if cuda:
        runs.append(("v1 hybrid gpu", f"VMAF v1, CPU + Vulkan on {devices[nvidia]}, GPU-decoded frames",
                     lambda: v1.V1Scorer(w, h, bits, V1_MODEL, device=nvidia, shared=gpu_dis), shared_v1))

    scores = {}
    for key, label, make, feed in runs:
        if arguments.only and not any(key.startswith(prefix) for prefix in arguments.only):
            continue
        fps, cores, result = measure(make, feed, arguments.pairs)
        # Against the first of the same kind: the CPU's for the CPU routes,
        # libvmaf's CUDA code for VMAF v0.6.1's GPU routes (whose features
        # the Vulkan engine reproduces; the CPU differs from both by a little
        # in motion), the CPU for VMAF v1.
        family = key.split()[0] + (" gpu" if key.startswith("v0") and "cpu" not in key else "")
        values = result if isinstance(result, np.ndarray) else np.concatenate(
            [np.asarray(v, dtype=np.float64) for v in result[1].values()])
        mean = float(np.mean(values))
        first = scores.setdefault(family, values)
        if first is values:
            check = "first of its kind"
        elif np.array_equal(first, values):
            check = "scores identical"
        else:
            check = f"scores differ by at most {np.max(np.abs(first - values)):.6f}"
        print(f"  {label:76s} {fps:7.1f} fps   {cores:5.1f} cores   mean {mean:.6f}, {check}", flush=True)


if __name__ == "__main__":
    main()
