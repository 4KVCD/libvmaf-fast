"""Times VMAF v1 with the GPU (vmaf_fast.v1.V1Scorer) on frames held in
memory, and shows where the time goes (the engine's vv_profile: each GPU
pass, summed by shader, and the CPU's steps), per frame.

    python fast/tests/bench_vmaf_v1.py REFERENCE DISTORTED [--frames 24] [--count 240]
        [--model 3d0h] [--bits 10] [--engine] [--no-profile]

--engine: after the first frames, the frames are committed as a decoder
hands them over (vv_staging / vv_commit, what is in the slots' memory) with
no CPU copy: the engine's own speed. Without it, every frame goes through
V1Scorer.add (its copies into the slots included).
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
    parser.add_argument("--no-profile", action="store_true")
    arguments = parser.parse_args()

    reference, width, height = frames_of.decode(arguments.reference, arguments.start, arguments.frames, None,
                                                arguments.bits)
    distorted, _, _ = frames_of.decode(arguments.distorted, arguments.start, arguments.frames, (width, height),
                                       arguments.bits)
    count = min(len(reference), len(distorted))
    scorer = vmaf_v1_gpu.V1Scorer(width, height, arguments.bits, model_path(arguments.model), device=arguments.device)
    lib = scorer._vulkan
    try:
        if not arguments.no_profile:
            lib.vv_profile.restype = ctypes.c_int
            lib.vv_profile.argtypes = [ctypes.c_void_p]
            if lib.vv_profile(scorer._gpu) != 0:
                print("no profile:", lib.vv_error().decode() if hasattr(lib, "vv_error") else "")
        warm = min(count, 8)
        for index in range(warm):
            scorer.add(reference[index], distorted[index])
        started = time.perf_counter()
        for index in range(warm, warm + arguments.count):
            if arguments.engine:
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
        _frames, values = scorer.features()
        elapsed = time.perf_counter() - started
        print(f"{width}x{height} {arguments.bits}-bit, {arguments.model}, on the GPU: "
              f"{sorted(name.split('_')[0] for name in scorer._on_gpu)}; "
              f"{'engine only' if arguments.engine else 'V1Scorer.add'}: "
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
