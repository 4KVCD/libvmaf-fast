"""Times VMAF v1 with the GPU (vmaf_fast.v1.V1Scorer) on frames held in
memory, and shows where the time goes (the engine's vv_profile: each GPU
pass, summed by shader, and the CPU's steps), per frame.

    python fast/tests/bench_vmaf_v1.py REFERENCE DISTORTED [--frames 24] [--count 240]
        [--model 3d0h] [--bits 10] [--engine | --shared] [--no-profile] [--dll PATH]

--engine: after the first frames, the frames are committed as a decoder
hands them over (vv_staging / vv_commit, what is in the slots' memory) with
no CPU copy: the engine's own speed. --shared: the same with the context
the app makes for a decoder that hands its pictures over (the slots in the
GPU's memory, each filled once by vv_test_fill_slot): the app's path. Without
either, every frame goes through V1Scorer.add (its copies included).
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import compare_vmaf_vulkan as frames_of
from compare_vmaf_v1 import model_path
from vmaf_fast import v1 as vmaf_v1_gpu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("distorted")
    parser.add_argument("--frames", type=int, default=24, help="frames decoded (and repeated)")
    parser.add_argument("--count", type=int, default=240, help="frames scored")
    parser.add_argument("--start", type=float, default=60.0)
    parser.add_argument("--model", default="3d0h")
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--dll", help="another build of vmaf_vulkan.dll (an earlier commit's, to compare)")
    arguments = parser.parse_args()
    if arguments.dll:
        vmaf_v1_gpu.vmaf_vulkan.LIBRARY_PATH = Path(arguments.dll)

    reference, width, height = frames_of.decode(arguments.reference, arguments.start, arguments.frames, None,
                                                arguments.bits)
    distorted, _, _ = frames_of.decode(arguments.distorted, arguments.start, arguments.frames, (width, height),
                                       arguments.bits)
    count = min(len(reference), len(distorted))
    shared = None
    if arguments.shared:  # a context as for a decoder handing over, without the decoder
        vmaf_v1_gpu.vmaf_vulkan.SharedLumas = lambda *_args: None
        shared = object()
    scorer = vmaf_v1_gpu.V1Scorer(width, height, arguments.bits, model_path(arguments.model), device=arguments.device,
                                  shared=shared)
    lib = scorer._vulkan
    check = vmaf_v1_gpu.vmaf_vulkan._check
    sample = 1 if arguments.bits <= 8 else 2
    luma = width * height * sample
    chroma_bytes = (width // 2) * sample
    try:
        if not arguments.no_profile:
            lib.vv_profile.restype = ctypes.c_int
            lib.vv_profile.argtypes = [ctypes.c_void_p]
            if lib.vv_profile(scorer._gpu) != 0:
                print("no profile:", lib.vv_error().decode() if hasattr(lib, "vv_error") else "")
        warm = 0 if arguments.shared else min(count, 8)
        for index in range(warm):
            scorer.add(reference[index], distorted[index])
        filled = set()
        if arguments.shared:
            lib.vv_test_fill_slot.restype = ctypes.c_int
            lib.vv_test_fill_slot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
            for slot in range(3):  # V1Scorer's frames in flight
                check(lib, lib.vv_test_fill_slot(scorer._gpu, slot, bytes(reference[slot % count][:luma]),
                                                 bytes(distorted[slot % count][:luma])), "filling a slot")
        started = time.perf_counter()
        for index in range(warm, warm + arguments.count):
            if arguments.shared:
                slot, stride, plane = ctypes.c_int(), ctypes.c_uint32(), ctypes.c_uint32()
                check(lib, lib.vv_shared_next(scorer._gpu, ctypes.byref(slot), ctypes.byref(stride), ctypes.byref(plane)),
                      "next")
                if vmaf_v1_gpu._SPEED in scorer._on_gpu:
                    chroma = (ctypes.c_void_p * 4)()
                    check(lib, lib.vv_chroma_staging(scorer._gpu, chroma, ctypes.byref(stride)), "chroma")
                    if slot.value not in filled:  # the slot's frame's chroma, once (a decoder downloads it each time)
                        filled.add(slot.value)
                        for i, (frame, plane_index) in enumerate(((reference, 1), (reference, 2), (distorted, 1),
                                                                   (distorted, 2))):
                            offset, rows, row_bytes = scorer._planes[plane_index]
                            source = frame[slot.value % count]
                            for row in range(height // 2):
                                ctypes.memmove(chroma[i] + row * stride.value,
                                               bytes(source[offset + row * row_bytes:offset + row * row_bytes + chroma_bytes]),
                                               chroma_bytes)
                check(lib, lib.vv_commit(scorer._gpu, 1), "commit")
                scorer._count += 1
            elif arguments.engine:
                ref, dis, stride = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_uint32()
                vmaf_v1_gpu.vmaf_vulkan._check(lib, lib.vv_staging(scorer._gpu, ctypes.byref(ref), ctypes.byref(dis),
                                                                   ctypes.byref(stride)), "staging")
                if vmaf_v1_gpu._SPEED in scorer._on_gpu:
                    chroma = (ctypes.c_void_p * 4)()
                    vmaf_v1_gpu.vmaf_vulkan._check(lib, lib.vv_chroma_staging(scorer._gpu, chroma,
                                                                              ctypes.byref(stride)), "chroma")
                vmaf_v1_gpu.vmaf_vulkan._check(lib, lib.vv_commit(scorer._gpu, 1), "commit")
                scorer._count += 1
            else:
                scorer.add(reference[index % count], distorted[index % count])
        scorer.features()
        elapsed = time.perf_counter() - started
        mode = "shared, the app's path" if arguments.shared else "engine only" if arguments.engine else "V1Scorer.add"
        print(f"{width}x{height} {arguments.bits}-bit, {arguments.model}, on the GPU: "
              f"{sorted(name.split('_')[0] for name in scorer._on_gpu)}; "
              f"{mode}: "
              f"{arguments.count / elapsed:.1f} fps ({1000 * elapsed / arguments.count:.2f} ms/frame)")
        if not arguments.no_profile:
            lib.vv_profile_text.restype = ctypes.c_int
            lib.vv_profile_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            text = ctypes.create_string_buffer(65536)
            lib.vv_profile_text(scorer._gpu, text, len(text))
            rows, gpu_total = [], 0.0
            for line in text.value.decode().splitlines():
                kind, name, ms, times = line.split()
                frames = scorer._count
                rows.append((kind, name, float(ms) / frames, int(times)))
                gpu_total += float(ms) / frames if kind == "gpu" else 0.0
            for kind, name, per_frame, times in sorted(rows, key=lambda row: (row[0], -row[2])):
                print(f"  {kind} {name:<22} {per_frame:8.3f} ms/frame  ({times} times)")
            print(f"  gpu total              {gpu_total:8.3f} ms/frame  -> {1000 / gpu_total:.0f} fps GPU-bound")
    finally:
        scorer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
