"""libvmaf-fast from Python: libvmaf's CUDA build (libvmaf), VMAF v0.6.1 and NEG
on any GPU with Vulkan (vulkan), and VMAF v1 with the GPU (v1). The libraries
are the ones fast/scripts builds into fast/dist, or those in the directory
the VMAF_FAST_DIST environment variable names (libvmaf/libvmaf.dll and
vmaf_vulkan/vmaf_vulkan.dll in it)."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
DIST = Path(os.environ.get("VMAF_FAST_DIST") or REPOSITORY / "fast" / "dist")
#: libvmaf's model files: VMAF v1's are read from here.
MODELS = Path(os.environ.get("VMAF_FAST_MODELS") or REPOSITORY / "model")
