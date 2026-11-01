"""Checks vv_shared_host_plane: a context made for frames from GPU memory (flag bit 19) that is given both of a
pair's luma planes from host memory instead gives every frame the same features, bit for bit, as a context given
them by vv_submit. (One plane from host beside one a decoder copies is checked by the app's runs.) Needs a GPU with
Vulkan and fast/dist's vmaf_vulkan.dll; takes a few seconds.

    python fast/tests/check_shared_host_plane.py [--device N]
"""
import argparse
import ctypes
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from vmaf_fast import vulkan

FRAMES = 6


def host_features(device, bits, reference, distorted):
    """Every frame's features from a context given its planes by vv_submit."""
    scorer = vulkan.VulkanScorer(*vulkan._PROBE_SIZE, bits, {}, device=device)
    try:
        for ref, dis in zip(reference, distorted):
            scorer.add(ref, dis)
        return scorer.features()[1]
    finally:
        scorer.close()


def shared_features(lib, device, bits, reference, distorted):
    """Every frame's features from a shared context given its planes by vv_shared_host_plane."""
    width, height = vulkan._PROBE_SIZE
    sample = 1 if bits == 8 else 2
    context = ctypes.c_void_p()
    if lib.vv_create(ctypes.byref(context), device, width, height, bits, (3 << 8) | vulkan.SHARED_FLAG) < 0:
        raise RuntimeError(lib.vv_error().decode())
    try:
        for ref, dis in zip(reference, distorted):
            slot, stride, offset = ctypes.c_int(), ctypes.c_uint32(), ctypes.c_uint32()
            if lib.vv_shared_next(context, ctypes.byref(slot), ctypes.byref(stride), ctypes.byref(offset)) < 0:
                raise RuntimeError(lib.vv_error().decode())
            for which, frame in ((0, ref), (1, dis)):
                plane, pitch = ctypes.c_void_p(), ctypes.c_uint32()
                if lib.vv_shared_host_plane(context, which, ctypes.byref(plane), ctypes.byref(pitch)) < 0:
                    raise RuntimeError(lib.vv_error().decode())
                rows = np.ctypeslib.as_array(ctypes.cast(plane.value, ctypes.POINTER(ctypes.c_uint8)),
                                             shape=(height, pitch.value))
                rows[:, :width * sample] = np.frombuffer(frame, np.uint8, width * height * sample).reshape(height, -1)
            if lib.vv_commit(context, 1) < 0:
                raise RuntimeError(lib.vv_error().decode())
        if lib.vv_flush(context) < 0:
            raise RuntimeError(lib.vv_error().decode())
        rows = np.zeros((len(reference), vulkan.FEATURE_COUNT))
        for frame in range(len(reference)):
            if lib.vv_features(context, frame, rows[frame].ctypes.data_as(ctypes.POINTER(ctypes.c_double))) < 0:
                raise RuntimeError(lib.vv_error().decode())
        return rows
    finally:
        lib.vv_destroy(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=int, default=None)
    device = parser.parse_args().device
    if device is None:
        device = vulkan.best_device().index
    lib = vulkan._load()
    lib.vv_shared_host_plane.restype = ctypes.c_int
    lib.vv_shared_host_plane.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.POINTER(ctypes.c_uint32)]
    failures = 0
    for bits in (8, 10):
        reference, distorted = vulkan.probe_frames(bits, FRAMES)
        expected = host_features(device, bits, reference, distorted)
        got = shared_features(lib, device, bits, reference, distorted)
        same = expected.tobytes() == got.tobytes()
        failures += not same
        print(f"{bits}-bit, {FRAMES} frames: {'the same' if same else 'DIFFERENT'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
