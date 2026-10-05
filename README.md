# libvmaf-fast

An unofficial fork of [Netflix/vmaf](https://github.com/Netflix/vmaf) (libvmaf),
not affiliated with or endorsed by Netflix. It adds GPU code to libvmaf that
gives libvmaf's own numbers:

- **VMAF v0.6.1 and VMAF NEG features on GPUs of any vendor, with Vulkan**,
  bit-identical to libvmaf's CUDA code. Upstream's only GPU code is CUDA,
  which runs on NVIDIA GPUs only. As with CUDA, libvmaf predicts the score
  from the features on the CPU.
- **VMAF v1 partly on the GPU.** Its two heaviest features, ADM3 and motion3
  (about two thirds of its CPU work), are calculated on the GPU, bit-identical
  to libvmaf's CPU code. The other two, CAMBI and SpEED, are still
  calculated by libvmaf's CPU code. The score is libvmaf's CPU score, bit for
  bit.
- **libvmaf's CUDA code with 11 open upstream pull requests merged**, which
  fix bugs in it (its build, a crash, races, leaks) and bring it closer to the
  CPU code, and upstream's native Windows (MSVC) build.

Windows x64 only for now. The Vulkan code needs Vulkan 1.1 and 64-bit
integers in shaders. For VMAF itself (what it measures, its models, the
Python tools), see [Netflix/vmaf](https://github.com/Netflix/vmaf); the
`master` branch here is upstream, unchanged. Building, testing and using this
fork's additions: [fast/README.md](fast/README.md).

## Speed

Measured with [fast/tests/bench_readme.py](fast/tests/bench_readme.py) on one
PC: an NVIDIA RTX 5090, an Intel Core Ultra 9 285K (24 cores) and the 285K's
integrated GPU. The video is HoneyBee (3840x2160, 10-bit) against an H.265
encode of it. 48 frame pairs are decoded into memory first; decoding is not
timed. Each route then scores them over and over: 480 pairs at 4K, and 960
at 1080p (the same frames scaled to 1920x1080).

- *From system memory*: the frames are in RAM, as a software decoder leaves
  them.
- *From GPU memory*: the frames are already in the NVIDIA GPU's memory, as a
  hardware decoder leaves them, and the GPU copies them to the scorer. This
  route uses CUDA, so it is NVIDIA only.
- *Cores busy*: the process's CPU time divided by the run's length.

### VMAF v0.6.1 and VMAF NEG

Models `vmaf_v0.6.1` and `vmaf_v0.6.1neg`, frames per second:

| | VMAF, 4K | VMAF, 1080p | VMAF + NEG, 4K | VMAF + NEG, 1080p |
|---|--:|--:|--:|--:|
| libvmaf on the CPU, 24 threads | 53 | 214 | 23 | 95 |
| libvmaf's CUDA code, RTX 5090, from system memory | 63 | 223 | 58 | 199 |
| libvmaf's CUDA code, RTX 5090, from GPU memory | 465 | 1300 | 304 | 860 |
| Vulkan, RTX 5090, from system memory | 323 | 1130 | 311 | 1113 |
| Vulkan, RTX 5090, from GPU memory | 382 | 1075 | 372 | 1101 |
| Vulkan, Intel integrated GPU, from system memory | 14 | 55 | 14 | 55 |

The CPU keeps 21 to 22 cores busy. Every GPU route keeps at most one core
busy, and the Intel GPU's under 0.1.

- **For VMAF alone, libvmaf's CUDA code is the fastest** when the frames are
  already in GPU memory: 465 fps at 4K, against Vulkan's 382.
- **For VMAF and NEG together, Vulkan is faster**: 372 against 304 fps at 4K.
  Vulkan calculates what VMAF and NEG share once (see below); libvmaf
  calculates VIF and ADM twice.
- **From system memory, Vulkan is about five times as fast as libvmaf's CUDA
  code** (311 to 323 against 58 to 63 fps at 4K). libvmaf's CUDA code reads
  frames from system memory at 58 to 65 fps at 4K whatever libvmaf's thread
  count (0 to 8), with about one core busy.
- **The Intel integrated GPU is slower than the CPU**: 14 fps at 4K, against
  23 to 53. It leaves the CPU free, though.
- The CPU is fastest with the most threads measured, 24. VMAF + NEG at 4K
  runs at 12.6 fps with 8 threads, 18.5 with 12, 18.9 with 16 and 23.1 with
  24.

### VMAF v1

Model `vmaf_v1.0.16_3d0h`:

| | 4K fps | 4K cores busy | 1080p fps | 1080p cores busy |
|---|--:|--:|--:|--:|
| libvmaf on the CPU, 4 threads | 36 | 4.1 | 156 | 4.1 |
| libvmaf on the CPU, 8 threads | 51 | 8.0 | 239 | 8.1 |
| libvmaf on the CPU, 12 threads | 58 | 12.0 | 272 | 12.0 |
| CPU + Vulkan, RTX 5090, from system memory | 101 | 10.5 | 424 | 6.8 |
| CPU + Vulkan, RTX 5090, from GPU memory | 132 | 15.8 | 683 | 13.3 |
| CPU + Vulkan, Intel integrated GPU, from system memory | 41 | 2.7 | 168 | 2.2 |

- **With the RTX 5090, VMAF v1 is 1.6 to 2.5 times as fast as on the CPU
  alone.** From system memory it also keeps fewer cores busy than the CPU
  alone at its fastest: 10.5 against 12.0 at 4K, and 6.8 against 12.0 at
  1080p.
- **With the Intel integrated GPU, VMAF v1 keeps few cores busy**: 2.7 at
  41 fps at 4K. The CPU alone keeps 4.1 cores busy at 36 fps, and 8 to 12 at
  51 to 58 fps.
- On the CPU alone, 16 or 24 threads are no faster than 12 (58 fps at 4K, 272
  at 1080p); they only keep more cores busy (15.5 to 19.9).
- With the GPU, CAMBI and SpEED run on a pool of 16 threads (the bindings'
  default on this CPU).

## How it was done

### VMAF v0.6.1 and NEG on Vulkan

`fast/vulkan` is a port of libvmaf's CUDA feature extractors to Vulkan
compute shaders. It covers VIF, ADM and motion: `integer_vif_cuda.c`,
`integer_adm_cuda.c`, `integer_motion_cuda.c` and their kernels. The shaders
are written in Slang and compiled to SPIR-V. They are embedded in
`vmaf_vulkan.dll`, which loads Vulkan (`vulkan-1.dll`, the Vulkan loader) at
run time.

The aim was the CUDA code's exact values, not values close to them. Two
things stand in the way: Vulkan lets each driver round floating point its
own way (and fuse a multiply and an add), and some GPUs, such as this PC's
Intel integrated GPU, have no double precision in Vulkan. So:

- **The CUDA kernels' floating point on each pixel is reproduced exactly.**
  - VIF's gain is a double division in CUDA. It is followed by a fused
    multiply-add, made by the CUDA compiler, and two multiplications. The
    shaders do these in integers and round each result to 53 bits, as CUDA's
    double arithmetic does.
  - ADM's one-degree angle test, in float and double in CUDA, is done with
    integer arithmetic up to 128 bits.
  - CUDA's ADM divides 2^30 by a value in float. The engine reads the result
    from a table of the same float divisions, made on the host, where float
    division rounds the same.
  - CUDA builds VIF's logarithm table on the GPU with its own `log2f`. The
    engine builds it on the host with the same polynomial, read from the
    compiled kernel, because the host's `log2f` differs in the last bit.
  - Float is used on the GPU only for estimates that cannot change the
    result. The angle test is decided in float only where float's error
    cannot change the answer, and VIF's division starts from a float estimate
    that is then corrected to the exact value.
- **Sums round the same way.** 64-bit sums are kept as pairs of 32-bit words
  added with 32-bit atomics, because 64-bit atomics are optional in Vulkan.
  Where the CUDA kernels round partial sums, the shaders round the same ones:
  ADM's denominator per warp of 32 threads over the same columns, and its
  contrast masking once per row.
- **Features and scores as libvmaf calculates them.** `vmaf_vulkan.cpp`
  turns the GPU's sums into features with libvmaf's expressions, in the same
  types and order. libvmaf itself then predicts the score from them
  (`vmaf_import_feature_score`, `vmaf_score_at_index`).

VMAF and NEG are calculated in one pass. NEG's features differ from VMAF's
only in the enhancement gain limit for VIF and ADM: 1 instead of 100. The
limit is applied partway through VIF and ADM. What comes before it, or does
not depend on it, is calculated once for both: motion, VIF's filtering, and
ADM's wavelet transform and denominator. Only the parts the limit changes
are calculated twice. libvmaf, given both models, runs motion once but VIF
and ADM twice.

GPU drivers do miscompile shaders. Two faults were found and worked around:

- **Intel:** the driver read a push-constant array indexed by a loop variable
  as zeros. The shaders take those values as separate scalars.
- **AMD:** the Radeon 780M's driver (32.0.21028.21 and 32.0.31041.1004) reads
  the entries of ADM's reciprocal table for values below -1 as zero,
  although the table in GPU memory is correct. The shader reads the entry for
  the absolute value and negates it, which gives the same number for this
  table.

A driver that compiles a shader wrongly gives wrong numbers, not an error.
So the Python bindings have self-tests, which a program should run before
trusting a GPU:

- `vulkan.probe()` scores fixed frames at 8 and 10 bits. It accepts a GPU
  only if the SHA-256 of its 219 sums at each depth is the known one.
- `v1.probe()` also compares VMAF v1's features with libvmaf's CPU code.

### VMAF v1 with the GPU

VMAF v1 is predicted from four features: ADM3, motion3, CAMBI and SpEED
(chroma). On one CPU thread, on the 4K benchmark video above, libvmaf takes
per frame:

| Feature | CPU time |
|---|--:|
| ADM3 | 55 ms |
| motion3 | 6 ms |
| CAMBI | 12 ms |
| SpEED | 16 ms |

ADM3 and motion3 are 68% of the work (63% at 1080p).

libvmaf has no CUDA code for ADM3 or motion3. The engine's VMAF v1 mode
follows libvmaf's CPU code for them (`integer_adm.c`, `integer_motion.c`),
using the CPU code's integer arithmetic:

- ADM3 combines ADM's detail loss with its additive impairment measure, and
  uses the Barten-Watson contrast sensitivity of `adm_csf_mode` 2.
- motion3 includes the HFR models' five-frame window and moving average.

libvmaf's own CPU extractors calculate CAMBI and SpEED on a pool of threads
beside the GPU. libvmaf then predicts the score from all four features.
Nothing is approximated: the score is libvmaf's CPU score.

Porting CAMBI and SpEED to the GPU is possible. SpEED is floating point
throughout, including an eigenvalue calculation, which makes bit-exact
results much harder.

### Frames from GPU memory

The engine takes frames in one of two ways:

- **From system memory:** `vv_submit` copies the luma into memory the GPU
  reads.
- **From GPU memory:** the engine's input buffers are exported
  (`VK_KHR_external_memory_win32`) and imported into CUDA
  (`cuImportExternalMemory`). A program holding pictures in CUDA memory,
  such as those from NVIDIA's hardware decoder, then copies them in on the
  GPU (`vv_shared_next`, `vv_export`).

For VMAF v1, the planes CAMBI and SpEED read are also needed on the CPU. They
are downloaded from GPU memory straight into libvmaf's pictures, which are
page-locked so the GPU's copy writes them directly.

In the benchmark this raises VMAF v1 on the RTX 5090 from 101 to 132 fps at
4K, and from 424 to 683 at 1080p. It raises VMAF + NEG from 311 to 372 fps at
4K, and leaves 1080p about the same. The route needs CUDA, so it is NVIDIA
only.

### libvmaf's CUDA backend

The `fast` branch is upstream at
[cea2b4d8](https://github.com/Netflix/vmaf/commit/cea2b4d832a105116a3f16f56d6f5d953421952c)
(2026-10-01) with these pull requests merged. Each is a merge commit at the
commit that was tested. They are their authors' work:

| Pull request | Author | Fixes |
|---|---|---|
| [#1477](https://github.com/Netflix/vmaf/pull/1477) | StormBytePP | Native MSVC build (merged upstream on 2026-10-02) |
| [#1573](https://github.com/Netflix/vmaf/pull/1573) | StormBytePP | CUDA builds outside the source tree; a crash in pinned pictures; a test's build order |
| [#1583](https://github.com/Netflix/vmaf/pull/1583) | jmsether | CUDA: a double flush that fails every threaded run at its end; the motion SAD reset racing the kernel |
| [#1644](https://github.com/Netflix/vmaf/pull/1644) | lusoris | CUDA motion: the CPU's reflect-101 edge mirror |
| [#1612](https://github.com/Netflix/vmaf/pull/1612) | BardieJoensen | CUDA motion: the first frame's uninitialized previous blur; a leaked readback buffer |
| [#1614](https://github.com/Netflix/vmaf/pull/1614) | BardieJoensen | CUDA VIF: the accumulator reset racing scale 0 (NaN scores) |
| [#1647](https://github.com/Netflix/vmaf/pull/1647)-[#1651](https://github.com/Netflix/vmaf/pull/1651) | lusoris | CUDA ADM: five differences from the CPU code |
| [#1652](https://github.com/Netflix/vmaf/pull/1652) | lusoris | `vmaf_read_pictures` leaking both pictures when it fails, or `vmaf_close` hanging with preallocated ones |

Seven of them (#1644 and #1647 to #1652) have new commits since. The fork
keeps the tested commits until the new ones are tested.

With these merged, libvmaf's CUDA code still differs a little from its CPU
code in motion. The CPU code blurs the difference of two frames; the CUDA
code blurs each frame and subtracts, which rounds differently. This is the
second cause in [Netflix/vmaf#1562](https://github.com/Netflix/vmaf/issues/1562);
#1644 fixes the first.

On the benchmark video, motion2 differs by at most 0.000029 at 4K, while VIF
and ADM (for VMAF and NEG) are identical. Per-frame VMAF and NEG scores
differ by at most 0.000033 at 4K and 0.000006 at 1080p. The Vulkan engine
reproduces the CUDA code, so it has the same difference. VMAF v1's GPU half
follows the CPU code and has none.

## How it is checked

Every "identical" here is an exact comparison (features bit for bit, scores
by equality), not a tolerance. The scripts are in `fast/tests`.

On the RTX 5090 and the Intel integrated GPU, with the release's DLLs:

- **VMAF and NEG** (`compare_vmaf_vulkan.py --matrix`): Vulkan's features and
  VMAF and NEG scores are identical to libvmaf's CUDA code in all 45 cases:
  - 11 sizes from 33x35 to 4096x2160 at 8 and 10 bits, and 8K at 10 bits;
  - black, white, black-white, identical, noise, extremes, still, sharpened
    and inverted frames at 8 and 10 bits;
  - every 2nd and every 5th frame, one frame and two frames.
- **VMAF v1** (`compare_vmaf_v1.py --matrix`): the four features and the
  score are identical to libvmaf's CPU code in all 71 cases:
  - all eight v1.0.16 models at six sizes from 640x480 to 3840x2160, three
    at 8 bits and three at 10;
  - the nine special frame kinds above at 8 and 10 bits;
  - every 2nd and every 5th frame, and one to three frames.
- **Real video:** 48 frames of HoneyBee at 4K 10-bit and at 1080p 8-bit.
  VMAF and NEG are identical to CUDA, and VMAF v1 to the CPU.
- **A whole film:** VMAF (4K model) and NEG over all 151,919 frames of a 4K
  10-bit film (3840x1608), against an H.265 encode, decoded by NVIDIA's
  decoder: every feature and score is identical to CUDA. This ran on the
  RTX 5090 with the release's DLLs. On the Intel GPU it ran on 2026-10-04
  with an earlier build, from before the AMD workaround in ADM's decouple
  shader, and was identical too.

On an AMD Radeon 780M, on another PC, where CUDA does not run:

- the self-test's 219 sums at 8 and at 10 bits, and all 32 passes of
  `diagnose_vmaf_vulkan.py`, match the reference;
- VMAF v1 over 48 frames of 4K 10-bit film is identical to libvmaf's CPU
  code.

The 71-case matrix was not run there.

## Limits

- Windows x64 only; not built or tested on Linux.
- The Vulkan engine is a DLL beside libvmaf with a small C API, not one of
  libvmaf's feature extractors. FFmpeg's libvmaf filter and `vmaf.exe` cannot
  use it. A program gives it frames and predicts with libvmaf, as
  [fast/python](fast/python) does.
- VMAF v1's CAMBI and SpEED stay on the CPU.
- Taking frames from GPU memory needs CUDA, so NVIDIA GPUs only.

## Releases and licence

Releases carry `libvmaf.dll` (with CUDA) and `vmaf_vulkan.dll`, with their
licences and checksums.
[VideoMetricsLab](https://github.com/4KVCD/VideoMetricsLab), which this fork
was written for, uses them.

The licence is BSD-2-Clause-Patent, as for libvmaf ([LICENSE](LICENSE)). The
Vulkan engine is a derived work of libvmaf's feature extractors. It keeps
their licence and Netflix's and NVIDIA's copyright notices. VMAF is
Netflix's: questions about VMAF itself belong upstream, and problems with
anything under `fast/` belong here.
