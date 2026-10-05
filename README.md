# libvmaf-fast

An unofficial fork of [Netflix/vmaf](https://github.com/Netflix/vmaf) (libvmaf),
not affiliated with or endorsed by Netflix. It makes VMAF faster on GPUs,
with the same scores as libvmaf's own code:

- **VMAF v0.6.1 and VMAF NEG on any GPU with Vulkan**, fully on the GPU, with
  features bit-identical to libvmaf's CUDA code. Upstream's only GPU code is
  CUDA, so NVIDIA only.
- **VMAF v1 partly on the GPU.** Its two heaviest features, ADM3 and motion3
  (about two thirds of its work), run on the GPU, bit-identical to libvmaf's
  CPU code. Its other two, CAMBI and SpEED, still run on libvmaf's CPU code.
  The score is libvmaf's CPU score exactly.
- **libvmaf's CUDA backend with 11 open upstream pull requests merged**, which
  fix its build and make it agree with the CPU code, and a native Windows
  build.

Windows x64 only for now. For VMAF itself (what it measures, its models, the
Python tools), see [Netflix/vmaf](https://github.com/Netflix/vmaf); the
`master` branch here is upstream, unchanged. Building, testing and using this
fork's additions: [fast/README.md](fast/README.md).

## What got faster

4K 10-bit frames held in memory (no decoding), on an RTX 5090, an Intel Core
Ultra 9 285K (24 cores) and its integrated GPU. CPU numbers are libvmaf's own
CPU code from the same build.

| VMAF v0.6.1 and NEG together | 4K | 1080p |
|---|---|---|
| libvmaf on the CPU, 24 threads | 21 fps | 106 fps |
| libvmaf's CUDA code, RTX 5090 | 46 fps | 154 fps |
| **Vulkan, RTX 5090** | **312 fps** | **1126 fps** |
| Vulkan, Intel integrated GPU | 14 fps | 54 fps |

| VMAF v1 (3840x1608 film) | 4K |
|---|---|
| libvmaf on the CPU, 12 / 24 threads | 67 / 51 fps |
| **CPU + Vulkan, RTX 5090** | **124 fps** |
| CPU + Vulkan, Intel integrated GPU | 55 fps |

The CUDA row is fed from host memory and calculates VMAF's and NEG's features
twice; with pictures already on the GPU it reaches 250-330 fps at 4K. VMAF v1
is limited by the CAMBI and SpEED left on the CPU: its GPU half alone runs at
655 fps on the RTX 5090. On a weak integrated GPU, VMAF v1 runs at about the
CPU's speed but frees most of the CPU (2.9 cores busy instead of 7.7 in one
measurement).

## How it was done

### VMAF on Vulkan

`fast/vulkan` is a port of libvmaf's CUDA feature extractors (VIF, ADM and
motion: `integer_vif_cuda.c`, `integer_adm_cuda.c`, `integer_motion_cuda.c`
and their kernels) to Vulkan compute shaders, written in Slang, compiled to
SPIR-V and embedded in `vmaf_vulkan.dll`, which loads the Vulkan driver at run
time.

"Close to CUDA" was not the goal; the same bits were. Floating-point results
on a GPU depend on the driver's compiler, so the shaders use **integer
arithmetic only**:

- The CUDA kernels' few float and double steps are emulated exactly in
  integers: VIF's one fused multiply-add, ADM's one-degree angle test (done in
  double by CUDA, here with 128-bit integer arithmetic and the same rounding),
  and the logarithm table, built on the host as CUDA computes it.
- Sums are 64-bit, kept as pairs of 32-bit atomics (64-bit atomics are
  optional in Vulkan), and rounded per row and per block in CUDA's order.
- The engine turns the GPU's sums into features with libvmaf's own
  expressions, in the same types and order, and libvmaf itself predicts the
  score from them (`vmaf_import_feature_score`), so the model code is
  libvmaf's.

It is also faster than the CUDA code it ports: VMAF and NEG differ only in
the enhancement gain limit, so motion, VIF's filtering, ADM's wavelet
transform and its denominators are calculated once for both, and only the
limited parts twice. libvmaf runs the two feature sets separately.

GPU drivers do miscompile shaders. Two faults were found and worked around:
Intel's driver reads a push-constant array indexed by a loop variable as
zeros, and AMD's (Radeon 780M) reads a table at a negative index as zero. A
start-up self-test scores fixed frames and compares the sums with known
SHA-256 hashes, so a driver that gets something wrong is caught instead of
trusted.

### VMAF v1 with the GPU

VMAF v1 is predicted from four features. Measured on one CPU thread on 4K
film, ADM3 takes 40 ms a frame, motion3 4, CAMBI 12 and SpEED 13: ADM and
motion are 64% of the work.

The same engine has a second mode that follows libvmaf's **CPU** code for
those two (`integer_adm.c`, `integer_motion.c`), not its CUDA code: ADM3 with
its additive impairment measure and the blended contrast sensitivity of
`adm_csf_mode` 2, motion3 with its five-frame window and moving average, all in
the CPU code's integer arithmetic. CAMBI and SpEED are calculated by libvmaf's
own CPU extractors on a thread pool beside the GPU, and libvmaf predicts the
score from all four. Nothing is approximated: the score is libvmaf's CPU
score.

Porting CAMBI and SpEED as well is possible; SpEED is floating point
throughout, which makes bit-exact results much harder.

### Frames without a CPU copy

Moving a 4K frame pair through system memory costs the CPU more than the GPU
half of VMAF v1 saves. The engine's input buffers can be exported
(`VK_KHR_external_memory_win32`) and imported by a hardware decoder's CUDA
(`cuImportExternalMemory`), which then copies decoded pictures into them on
the GPU; the planes libvmaf's CPU extractors read can be downloaded straight
into libvmaf's pictures, page-locked, so the GPU writes them itself. In
VideoMetricsLab (4K, RTX 5090) this cut the CPU time spent moving a frame
pair from 9.7 ms to 0.6 ms, and VMAF v1 went from 85 to 146 fps.

### libvmaf's CUDA backend

The `fast` branch is upstream at
[cea2b4d8](https://github.com/Netflix/vmaf/commit/cea2b4d832a105116a3f16f56d6f5d953421952c)
(2026-10-01) with these open pull requests merged, each as a merge commit at
the commit that was tested. They are their authors' work:

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

Some have changed since; the fork keeps the tested commits until the new
ones are tested.

## How it is checked

Every claim of "identical" above is a comparison of bit patterns, not a
tolerance, by the scripts in `fast/tests`:

- Vulkan's features and VMAF and NEG scores against libvmaf's CUDA code over
  45 cases (sizes from the smallest to 8K, odd sizes, 8 and 10 bits, black,
  white, noise, extremes, still and sharpened frames): identical on an RTX 5090
  and an Intel integrated GPU. On an AMD Radeon 780M, where CUDA does not run,
  the self-test's 219 sums and every pass of the diagnosis match theirs.
- VMAF v1's four features and score against libvmaf on the CPU over 71 cases
  (all eight v1.0.16 models, 640x480 to 4K, 8 and 10 bits, frame skipping,
  one to three frames): identical on all three GPUs.
- Real 4K 10-bit and 1080p 8-bit video: identical. VMAF v0.6.1 and NEG over a
  whole film (151,919 frames, in VideoMetricsLab): identical to CUDA.

libvmaf's CUDA code is itself within about 0.001 of its CPU code for VMAF
v0.6.1 (motion's blur rounds in another order:
[Netflix/vmaf#1562](https://github.com/Netflix/vmaf/issues/1562)), so the
Vulkan engine is too. VMAF v1 has no such difference.

## Limits

- Windows x64 only; not built or tested on Linux.
- The Vulkan engine is a DLL beside libvmaf with a small C API, not one of
  libvmaf's feature extractors, so FFmpeg's libvmaf filter and `vmaf.exe`
  cannot use it. A program gives it frames and predicts with libvmaf, as
  [fast/python](fast/python) does.
- VMAF v1's CAMBI and SpEED stay on the CPU.

## Releases and licence

Releases carry `libvmaf.dll` (with CUDA) and `vmaf_vulkan.dll` with their
licences and checksums; [VideoMetricsLab](https://github.com/4KVCD/VideoMetricsLab),
which this was written for, uses them.

BSD-2-Clause-Patent, as libvmaf ([LICENSE](LICENSE)). The Vulkan engine is a
derived work of libvmaf's feature extractors and keeps their licence and
Netflix's and NVIDIA's copyright notices. VMAF is Netflix's; questions about
VMAF itself belong upstream, problems with anything under `fast/` here.
