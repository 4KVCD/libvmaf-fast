# libvmaf-fast

An unofficial fork of [Netflix/vmaf](https://github.com/Netflix/vmaf) (libvmaf)
with faster GPU paths and fixes. It is not affiliated with or endorsed by
Netflix. Everything upstream is unchanged and still here; what this fork adds
lives under `fast/`.

What it adds:

- **VMAF on any GPU with Vulkan.** A port of libvmaf's CUDA feature extractors
  (VIF, ADM, motion) to Vulkan compute shaders, for VMAF v0.6.1 and VMAF NEG.
  Its features are bit-identical to libvmaf's CUDA code on every GPU it has
  been tested on, NVIDIA's included.
- **VMAF v1 with the GPU.** The same engine in a second mode follows libvmaf's
  CPU code for VMAF v1's ADM3 and motion3, bit-identical to it. CAMBI and SpEED
  stay on libvmaf's CPU extractors, and libvmaf predicts the score, so the
  score is libvmaf's CPU score exactly.
- **libvmaf's CUDA backend with 11 open upstream fixes merged** (and one
  since merged upstream), and a Windows build script for it.
- **Python bindings and accuracy tests** that check all of the above against
  libvmaf bit for bit.

Windows x64 only for now.

## Speed

Frames held in memory (no decoding), 10-bit, on an RTX 5090, an Intel Core
Ultra 9 285K (24 cores) and its integrated Intel GPU. CPU numbers are
libvmaf's own CPU code from the same DLL.

**VMAF v0.6.1 and VMAF NEG together** (HoneyBee, 4K and scaled to 1080p):

| | 4K | 1080p |
|---|---|---|
| libvmaf on the CPU, 24 threads | 21 fps | 106 fps |
| libvmaf's CUDA code, RTX 5090 | 46 fps | 154 fps |
| **Vulkan, RTX 5090** | **312 fps** | **1126 fps** |
| Vulkan, Intel integrated GPU | 14 fps | 54 fps |

The CUDA row is fed from host memory and calculates VMAF's and NEG's features
separately; the Vulkan engine calculates what the two share once. With
pictures already on the GPU (a hardware decoder feeding it), CUDA reaches
250-330 fps at 4K in VideoMetricsLab's runs.

**VMAF v1** (The Beekeeper, 3840x1608 film):

| | 4K |
|---|---|
| libvmaf on the CPU, 12 / 24 threads | 67 / 51 fps |
| **CPU + Vulkan, RTX 5090** | **124 fps** |
| CPU + Vulkan, Intel integrated GPU | 55 fps |

VMAF v1's GPU half alone runs at 655 fps on the RTX 5090; CAMBI and SpEED on
the CPU set the pace. On a weak integrated GPU the hybrid is about the CPU's
speed. More libvmaf threads are not faster here: 24 threads used twice the CPU
of 12 and were slower.

## Accuracy

Every result below is from the scripts in `fast/tests`, which compare bit
patterns, not tolerances:

- `compare_vmaf_vulkan.py --matrix`: Vulkan's VIF, ADM and motion features and
  the VMAF and NEG scores against libvmaf's CUDA code, over 45 cases (sizes
  from the smallest to 8K, odd sizes, 8 and 10 bits, black, white, noise,
  extremes, still and sharpened frames). All identical on an RTX 5090 and an
  Intel integrated GPU. On an AMD Radeon 780M, where CUDA does not run, the
  probe's 219 sums and every pass of `diagnose_vmaf_vulkan.py` match those
  GPUs'.
- `compare_vmaf_v1.py --matrix`: VMAF v1's four features and the score against
  libvmaf on the CPU, over 71 cases (all eight bundled v1.0.16 models, 640x480
  to 4K, 8 and 10 bits, frame skipping, one to three frames). All identical on
  the RTX 5090, the Intel GPU and the Radeon 780M.
- On real video (4K 10-bit and 1080p 8-bit): identical on the RTX 5090 and the
  Intel GPU. VMAF v0.6.1 and NEG over a whole film (151,919 frames, in
  VideoMetricsLab): identical to CUDA on both.

libvmaf's CUDA code is within about 0.001 of its own CPU code for VMAF v0.6.1
(motion's blur rounds in a different order: [Netflix/vmaf#1562](https://github.com/Netflix/vmaf/issues/1562)),
so the Vulkan engine is too. VMAF v1 has no such difference.

### Drivers

GPU drivers do miscompile shaders. The shaders use only integer arithmetic
(64-bit sums as pairs of 32-bit words, the CUDA kernels' few float and double
steps emulated exactly), and work around two faults found so far:

- Intel: a push-constant array indexed by a loop variable reads as zeros.
- AMD (Radeon 780M, drivers 32.0.21028.21 and 32.0.31041.1004): a table read
  at a negative index reads zero. The table is read at |o| instead.

A program using the engine should check a GPU before trusting it, as
`vmaf_fast.vulkan.probe()` does: it scores fixed frames and compares the sums
with known SHA-256 hashes. `diagnose_vmaf_vulkan.py` finds the first pass that
differs on a GPU that fails.

## Building

Needs Windows x64, Visual Studio 2022 Build Tools (C++), git, and for libvmaf
also the CUDA Toolkit 13.x, Python with meson and ninja
(`pip install meson ninja`), and nasm, cmake and xxd on PATH (Git for
Windows has xxd in `usr\bin`). Slang, the shader compiler, and the Vulkan
headers are downloaded at pinned versions.

```powershell
powershell -ExecutionPolicy Bypass -File fast\scripts\build_libvmaf_cuda.ps1 -Python <python with meson> -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
powershell -ExecutionPolicy Bypass -File fast\scripts\build_vmaf_vulkan.ps1
```

They write `fast/dist/libvmaf/libvmaf.dll` and
`fast/dist/vmaf_vulkan/vmaf_vulkan.dll`, each with its licences. Both link the
C runtime statically and need only Windows and a GPU driver at run time. Both
builds are reproducible: the same commit gives the same bytes. libvmaf reports
the commit it was built from as its version.

`fast\scripts\package.ps1 -Version <version>` makes the release archive from
them, with a SHA-256 of every file.

## Testing

With the two DLLs built, and numpy:

```powershell
python fast\tests\compare_vmaf_vulkan.py --matrix [--device N]
python fast\tests\compare_vmaf_v1.py --matrix [--device N]
python fast\tests\diagnose_vmaf_vulkan.py [--device N]
python fast\tests\bench_vmaf_vulkan.py REFERENCE DISTORTED --size 3840x2160 --bits 10
```

`--device` is Vulkan's GPU number (the scripts list them). `VMAF_FAST_DIST`
points the bindings at DLLs elsewhere.

## Using it

The Vulkan engine is a DLL with a small C API, not yet one of libvmaf's
feature extractors: a program gives it frames and gets VMAF's features back,
then predicts the score with libvmaf (`vmaf_import_feature_score`,
`vmaf_score_at_index`), as `fast/python/vmaf_fast` does:

- `vulkan.py`: `VulkanScorer`, VMAF v0.6.1 and NEG.
- `v1.py`: `V1Scorer`, VMAF v1 (the GPU's ADM3 and motion3 with libvmaf's CAMBI
  and SpEED).
- `libvmaf.py`: the binding to libvmaf's C API, and `GpuScorer`, its CUDA code.

The engine's exports are in `fast/vulkan/vmaf_vulkan.cpp`'s API section:
`vv_create` / `vv_create_v1`, `vv_submit` (or `vv_staging` and `vv_commit` to
write frames in place), `vv_flush`, `vv_features` / `vv_features_v1`,
`vv_destroy`, and `vv_shared_next` / `vv_export` to share its input buffers
with a hardware decoder's CUDA, so decoded pictures never leave the GPU.

[VideoMetricsLab](https://github.com/4KVCD/VideoMetricsLab), for which this
was written, uses a pinned release of it.

## What differs from upstream

The `fast` branch is upstream at
[cea2b4d8](https://github.com/Netflix/vmaf/commit/cea2b4d832a105116a3f16f56d6f5d953421952c)
(2026-10-01) with these merged, each as a merge commit at the commit that was
tested:

| Pull request | Author | Fixes |
|---|---|---|
| [#1477](https://github.com/Netflix/vmaf/pull/1477) | StormBytePP | Native MSVC build (merged upstream since) |
| [#1573](https://github.com/Netflix/vmaf/pull/1573) | StormBytePP | CUDA build fixes; a crash in pinned pictures |
| [#1583](https://github.com/Netflix/vmaf/pull/1583) | jmsether | CUDA end-of-stream double flush; the motion SAD reset racing the kernel |
| [#1644](https://github.com/Netflix/vmaf/pull/1644) | lusoris | CUDA motion: the CPU's reflect-101 edge mirror |
| [#1612](https://github.com/Netflix/vmaf/pull/1612) | BardieJoensen | CUDA motion: uninitialized previous blur on the first frame |
| [#1614](https://github.com/Netflix/vmaf/pull/1614) | BardieJoensen | CUDA VIF: accumulator reset racing scale 0 (NaN scores) |
| [#1647](https://github.com/Netflix/vmaf/pull/1647)-[#1651](https://github.com/Netflix/vmaf/pull/1651) | lusoris | CUDA ADM: five differences from the CPU code |
| [#1652](https://github.com/Netflix/vmaf/pull/1652) | lusoris | A picture leaked when `vmaf_read_pictures` fails |

Most of the CUDA fixes are their authors' work, not this fork's. Some of those
pull requests have changed since; the fork keeps the tested commits until the
new ones are tested.

Everything else is new and under `fast/`: the Vulkan engine (`fast/vulkan`),
the build and packaging scripts (`fast/scripts`), the Python bindings
(`fast/python`) and the tests (`fast/tests`). The `master` branch follows
upstream unchanged.

## Licence

BSD-2-Clause-Patent, as libvmaf ([LICENSE](../LICENSE)). The Vulkan engine is
a derived work of libvmaf's CUDA and CPU feature extractors and keeps their
licence and Netflix's and NVIDIA's copyright notices. The release archive
carries the licences of what is compiled in: libvmaf, pthreads4w and
nv-codec-headers (MIT) in `libvmaf/`, libvmaf's in `vmaf_vulkan/`. Slang and
the Vulkan headers are build tools.

VMAF is Netflix's; questions about VMAF itself belong upstream. Problems with
anything under `fast/` belong here.
