# libvmaf-fast

An unofficial fork of [Netflix/vmaf](https://github.com/Netflix/vmaf) (libvmaf),
not affiliated with or endorsed by Netflix. It adds GPU code that gives
libvmaf's own numbers:

- **VMAF v0.6.1 and VMAF NEG on any GPU with Vulkan.** The features are
  bit-identical to libvmaf's CUDA code, which is upstream's only GPU code and
  runs on NVIDIA GPUs only. libvmaf predicts the score from them on the CPU,
  as it does with CUDA.
- **VMAF v1 partly on the GPU.** ADM3 and motion3, about two thirds of its
  CPU work, are calculated on the GPU, bit-identical to libvmaf's CPU code.
  CAMBI and SpEED stay on libvmaf's CPU code. The score is libvmaf's CPU
  score, bit for bit.
- **libvmaf's CUDA code with 11 open upstream pull requests merged.** They fix
  its build, a crash, races and leaks, and bring it closer to the CPU code.
  Upstream's native MSVC build is merged too.

**Ready-made tool:** [VideoMetricsLab](https://github.com/4KVCD/VideoMetricsLab)
is a Windows app for scoring and comparing video encodes. From version 1.5 it
calculates VMAF, VMAF NEG and VMAF v1 on the GPU with these libraries (on its
`release/v1.5` branch for now). The current release, 1.4, has GPU VMAF on
NVIDIA only.

Windows x64 only; not built or tested on Linux. The Vulkan engine needs
Vulkan 1.1 and 64-bit integers in shaders. It is a DLL with a small C API
beside libvmaf, not a libvmaf feature extractor, so FFmpeg's libvmaf filter
and `vmaf.exe` cannot use it. A program feeds it frames and predicts with
libvmaf, as [fast/python](fast/python) does; see
[fast/README.md](fast/README.md) for building, testing and using it. For VMAF
itself, see [Netflix/vmaf](https://github.com/Netflix/vmaf). The `master`
branch here is upstream, unchanged.

## Speed

These were measured with [fast/tests/bench_readme.py](fast/tests/bench_readme.py)
on one PC: an RTX 5090, a Core Ultra 9 285K (24 cores) and the 285K's
integrated GPU. The video is HoneyBee, 3840x2160 10-bit, against an x265
encode of it; the 1080p numbers use the same frames scaled down. Frames are
decoded before timing, so decoding is not counted. Each number is the mean of
two runs, rounded to the nearest 10 above 100. The two runs differed by up to
21%, so small differences are noise.

A GPU's speed depends on where the decoder leaves the frames:

- **CPU-decoded:** a software decoder leaves frames in system memory, and
  each one is uploaded to the GPU. Frames from Intel's and AMD's hardware
  decoders reach the engine this way too.
- **GPU-decoded:** NVIDIA's hardware decoder leaves frames in GPU memory, and
  they are copied on the GPU. The hand-over uses CUDA, so this is NVIDIA
  only.

### VMAF v0.6.1 and VMAF NEG

Frames per second. CUDA is libvmaf's own CUDA code (with the fixes below), on
the RTX 5090.

| | VMAF, 4K | VMAF, 1080p | VMAF + NEG, 4K | VMAF + NEG, 1080p |
|---|--:|--:|--:|--:|
| CPU, 24 threads | 52 | 210 | 24 | 110 |
| CUDA, CPU-decoded | 330 | 840 | 260 | 670 |
| CUDA, GPU-decoded | 470 | 1250 | 300 | 850 |
| Vulkan, RTX 5090, CPU-decoded | 310 | 1130 | 310 | 1120 |
| Vulkan, RTX 5090, GPU-decoded | 380 | 1070 | 380 | 1030 |
| Vulkan, Intel iGPU, CPU-decoded | 14 | 54 | 14 | 54 |

- **GPU-decoded, VMAF alone: libvmaf's CUDA code is the fastest.** With NEG
  as well, Vulkan is faster (380 against 300 fps at 4K). Vulkan calculates
  what VMAF and NEG share once, while libvmaf calculates VIF and ADM twice.
- **CPU-decoded: Vulkan is as fast or faster.** For VMAF alone at 4K the two
  are about even; CUDA's two runs gave 300 and 370 fps. In the other three
  columns Vulkan is faster.
- **libvmaf's own host pictures are slow.** In the CUDA CPU-decoded row, the
  program uploads each frame into libvmaf's GPU pictures through one reused
  page-locked buffer, as the Vulkan engine does. Handing libvmaf host
  pictures instead gives only about 60 fps at 4K. For every picture, libvmaf
  allocates, page-locks and zeroes a new 25 MB buffer, taking about 9 ms per
  4K frame pair.
- **The Intel integrated GPU is slower than the CPU.**

### VMAF v1

Model `vmaf_v1.0.16_3d0h`, frames per second:

| | 4K | 1080p |
|---|--:|--:|
| CPU, 12 threads | 55 | 270 |
| CPU + Vulkan, RTX 5090, CPU-decoded | 100 | 460 |
| CPU + Vulkan, RTX 5090, GPU-decoded | 130 | 670 |
| CPU + Vulkan, Intel iGPU, CPU-decoded | 41 | 160 |

- **The RTX 5090 makes VMAF v1 1.7 to 2.5 times as fast** as the CPU alone.
- **With the Intel GPU, VMAF v1 is slower** than on the CPU alone.
- On the CPU alone, 16 and 24 threads are no faster than 12.
- With the GPU, CAMBI and SpEED run on 16 CPU threads.

## How it was done

### VMAF and NEG on Vulkan

`fast/vulkan` ports libvmaf's CUDA feature extractors for VIF, ADM and motion
(`integer_vif_cuda.c`, `integer_adm_cuda.c`, `integer_motion_cuda.c` and
their kernels) to Vulkan compute shaders. The shaders are written in Slang,
compiled to SPIR-V and embedded in `vmaf_vulkan.dll`.

The goal was CUDA's exact values. Vulkan lets each driver round floating
point its own way, and some GPUs, such as this PC's Intel GPU, have no double
precision. So:

- **Double arithmetic is done in integers.** VIF's double steps are each
  rounded to 53 bits; ADM's one-degree angle test uses integers up to 128
  bits.
- **Float results come from tables made on the host.** These are ADM's
  reciprocals, and VIF's logarithms from CUDA's own `log2f` polynomial (read
  from the compiled kernel). Float remains on the GPU only in estimates that
  cannot change the result.
- **Sums round as CUDA's do.** They are 64-bit, kept as pairs of 32-bit
  atomics. Partial sums are rounded where CUDA rounds them.
- **libvmaf's expressions make the features.** `vmaf_vulkan.cpp` uses them,
  and libvmaf predicts the score (`vmaf_import_feature_score`).

NEG differs from VMAF only in the enhancement gain limit of VIF and ADM (1
instead of 100), applied partway through both. So everything the limit does
not affect is calculated once for both. libvmaf, given both models,
calculates VIF and ADM twice.

Two driver miscompilations were found and worked around:

- **Intel:** the driver read a push-constant array indexed by a loop variable
  as zeros. The shaders take scalars instead.
- **AMD:** the Radeon 780M's driver (32.0.21028.21 and 32.0.31041.1004) reads
  the entries of ADM's reciprocal table for values below -1 as zero. The
  shader reads the entry for the absolute value and negates it.

A miscompiled shader gives wrong numbers, not an error. So a program should
run the Python bindings' self-tests before trusting a GPU:

- `vulkan.probe()` checks 219 sums at 8 and 10 bits against known SHA-256
  hashes.
- `v1.probe()` checks VMAF v1's features against libvmaf's CPU code.

### VMAF v1 with the GPU

VMAF v1 is predicted from ADM3, motion3, CAMBI and SpEED (chroma). On one CPU
thread at 4K, libvmaf takes 55, 6, 12 and 16 ms a frame for them, so ADM3
and motion3 are 68% of the work (63% at 1080p).

libvmaf has no CUDA code for ADM3 or motion3. So the engine's VMAF v1 mode
follows libvmaf's CPU code (`integer_adm.c`, `integer_motion.c`), in the same
integer arithmetic. That includes ADM3's `adm_csf_mode` 2 and the HFR models'
five-frame motion window. libvmaf's own CPU extractors calculate CAMBI and
SpEED on a thread pool beside the GPU. SpEED is floating point throughout,
which would make a bit-exact GPU port much harder.

### GPU-decoded frames

The engine's input buffers can be exported (`VK_KHR_external_memory_win32`)
and imported into CUDA (`cuImportExternalMemory`). So frames from NVIDIA's
decoder are copied in on the GPU (`vv_export`). For VMAF v1, the planes CAMBI
and SpEED read are downloaded straight into libvmaf's page-locked pictures.
CPU-decoded frames go through `vv_submit`.

### libvmaf's CUDA code

The `fast` branch is upstream at
[cea2b4d8](https://github.com/Netflix/vmaf/commit/cea2b4d832a105116a3f16f56d6f5d953421952c)
(2026-10-01) with these pull requests merged, each at the commit that was
tested. They are their authors' work:

| Pull request | Author | Fixes |
|---|---|---|
| [#1477](https://github.com/Netflix/vmaf/pull/1477) | StormBytePP | Native MSVC build (merged upstream on 2026-10-02) |
| [#1573](https://github.com/Netflix/vmaf/pull/1573) | StormBytePP | CUDA builds outside the source tree; a crash in pinned pictures |
| [#1583](https://github.com/Netflix/vmaf/pull/1583) | jmsether | A double flush that fails threaded CUDA runs at the end; a race in motion |
| [#1644](https://github.com/Netflix/vmaf/pull/1644) | lusoris | CUDA motion: the CPU's edge mirror |
| [#1612](https://github.com/Netflix/vmaf/pull/1612) | BardieJoensen | CUDA motion: an uninitialized first-frame blur; a leak |
| [#1614](https://github.com/Netflix/vmaf/pull/1614) | BardieJoensen | CUDA VIF: a race giving NaN scores |
| [#1647](https://github.com/Netflix/vmaf/pull/1647)-[#1651](https://github.com/Netflix/vmaf/pull/1651) | lusoris | CUDA ADM: five differences from the CPU code |
| [#1652](https://github.com/Netflix/vmaf/pull/1652) | lusoris | `vmaf_read_pictures` leaking its pictures on failure |

Seven of them (#1644 and #1647 to #1652) have new commits since. The fork
keeps the tested commits until the new ones are tested.

CUDA still differs slightly from the CPU in motion. The CPU blurs the
difference of two frames; CUDA blurs each frame and subtracts, which rounds
differently. This is the second cause in
[Netflix/vmaf#1562](https://github.com/Netflix/vmaf/issues/1562); #1644
fixes the first.

On the benchmark video at 4K, VIF and ADM are identical, and motion2 differs
by at most 0.000029. Per-frame VMAF and NEG scores differ by at most
0.000033. Vulkan, reproducing CUDA, differs from the CPU in the same way.
VMAF v1's GPU half follows the CPU code and does not differ.

## How it is checked

Every "identical" is an exact comparison: features bit for bit, scores by
equality. The scripts are in `fast/tests`. On the RTX 5090 and the Intel
iGPU, with the release's DLLs:

- **VMAF and NEG** (`compare_vmaf_vulkan.py --matrix`): Vulkan is identical
  to CUDA in 45 cases. They cover sizes from 33x35 to 8K, 8 and 10 bits,
  special frames such as black, noise and sharpened, and frame skipping.
- **VMAF v1** (`compare_vmaf_v1.py --matrix`): identical to libvmaf's CPU
  code in 71 cases. They cover all eight v1.0.16 models, 640x480 to 4K, 8 and
  10 bits, special frames and frame skipping.
- **Real video:** 48 frames of HoneyBee at 4K 10-bit and 1080p 8-bit. VMAF
  and NEG are identical to CUDA, and VMAF v1 to the CPU.
- **A whole film:** all 151,919 frames of a 4K 10-bit film (3840x1608),
  NVIDIA-decoded, scored with VMAF (4K model) and NEG. Every frame is
  identical to CUDA. On the RTX 5090 this ran with the release's DLLs; on the
  Intel GPU it ran on 2026-10-04 with an earlier build, from before the AMD
  workaround.

On a Radeon 780M, on another PC without CUDA, the self-test's sums and all 32
passes of `diagnose_vmaf_vulkan.py` match the reference. VMAF v1 over 48
frames of 4K film is identical to libvmaf's CPU code there; the 71-case
matrix was not run.

## Releases and licence

Releases carry `libvmaf.dll` (with CUDA) and `vmaf_vulkan.dll`, with their
licences and checksums. The licence is BSD-2-Clause-Patent, as for libvmaf
([LICENSE](LICENSE)). The Vulkan engine, a derived work of libvmaf's feature
extractors, keeps their licence and Netflix's and NVIDIA's copyright notices.
Questions about VMAF itself belong upstream; problems with anything under
`fast/` belong here.
