# libvmaf-fast

[Netflix's libvmaf](https://github.com/Netflix/vmaf), faster, with the same
scores. An unofficial fork, not affiliated with or endorsed by Netflix.

- **VMAF and VMAF NEG on any GPU, bit-exact with libvmaf's CUDA code: up to
  28x as fast as libvmaf on the CPU, and 4.8x to 6.4x as fast as its CUDA
  code** (RTX 5090). libvmaf's CUDA runs only on NVIDIA; this runs on Vulkan.
- **VMAF v1 on any GPU with Vulkan, bit-exact with libvmaf's CPU code: 2.6x
  to 9.4x as fast** (Intel's iGPU: 0.8x).
- **PSNR and SSIM, bit-exact: 1.9x to 4.1x as fast.**
- **XPSNR, new in libvmaf, bit-exact with FFmpeg's `xpsnr` filter: 5.2x to
  16x as fast.**
- **libvmaf's CUDA code fixed**: 13 upstream pull requests merged, 35% less
  GPU memory, scores closer to the CPU's.

**Ready-made tool:** [VideoMetricsLab](https://github.com/4KVCD/VideoMetricsLab)
is a Windows app for scoring and comparing video encodes. Calculate VMAF and VMAF NEG on NVIDIA GPUs, and SSIMULACRA2, Butteraugli and ColorVideo VDP on NVIDIA, AMD and Intel GPUs, alongside PSNR, SSIM and XPSNR. Compare encodes with frame-exact playback that switches between the source and each encode instantly to easily spot differences.

Windows x64 only. Releases carry `libvmaf.dll`, libvmaf with CUDA, and
`vmaf_vulkan.dll`, the Vulkan engine (needs Vulkan 1.1 with 64-bit shader
integers). The engine is not a libvmaf feature extractor, so FFmpeg's libvmaf
filter and `vmaf.exe` cannot use it: a program gives it frames and has
libvmaf predict the score, as [fast/python](fast/python) does. Building,
testing and using both: [fast/README.md](fast/README.md). The `master`
branch is upstream, unchanged.

## Speed

Frames per second, official libvmaf → libvmaf-fast:

| 4K | RTX 5090, Core Ultra 9 285K | Intel iGPU, Core Ultra 9 285K | Radeon 8060S, Ryzen AI Max+ 395 | Radeon 780M, Ryzen 7 8845HS |
|---|--:|--:|--:|--:|
| VMAF + NEG, CPU → Vulkan | 30 → 855 (**28x**) | 30 → 71 (**2.4x**) | 42 → 401 (**9.6x**) | 9 → 92 (**10x**) |
| VMAF + NEG, CUDA → Vulkan | 177 → 855 (**4.8x**) | | | |
| VMAF + NEG, CUDA → Optimized CUDA | 177 → 269 (**1.5x**) | | | |
| VMAF v1, CPU → GPU | 60 → 567 (**9.4x**) | 60 → 51 (**0.8x**) | 118 → 313 (**2.6x**) | 33 → 143 (**4.3x**) |
| PSNR | 170 → 452 (**2.7x**) | | 135 → 422 (**3.1x**) | 174 → 340 (**2.0x**) |
| SSIM | 120 → 451 (**3.8x**) | | 220 → 421 (**1.9x**) | 73 → 304 (**4.1x**) |
| XPSNR, FFmpeg's filter → libvmaf-fast | 58 → 415 (**7.2x**) | | 29 → 457 (**16x**) | 38 → 293 (**7.7x**) |

| 1080p | RTX 5090, Core Ultra 9 285K | Intel iGPU, Core Ultra 9 285K | Radeon 8060S, Ryzen AI Max+ 395 | Radeon 780M, Ryzen 7 8845HS |
|---|--:|--:|--:|--:|
| VMAF + NEG, CPU → Vulkan | 114 → 2933 (**26x**) | 114 → 268 (**2.3x**) | 243 → 1700 (**7.0x**) | 60 → 341 (**5.7x**) |
| VMAF + NEG, CUDA → Vulkan | 461 → 2933 (**6.4x**) | | | |
| VMAF + NEG, CUDA → Optimized CUDA | 461 → 674 (**1.5x**) | | | |
| VMAF v1, CPU → GPU | 251 → 1960 (**7.8x**) | 251 → 192 (**0.8x**) | 495 → 1811 (**3.7x**) | 135 → 533 (**4.0x**) |
| PSNR | 578 → 1167 (**2.0x**) | | 831 → 1849 (**2.2x**) | 582 → 1131 (**1.9x**) |
| SSIM | 418 → 945 (**2.3x**) | | 655 → 1289 (**2.0x**) | 247 → 570 (**2.3x**) |
| XPSNR, FFmpeg's filter → libvmaf-fast | 272 → 1422 (**5.2x**) | | 108 → 1477 (**14x**) | 89 → 972 (**11x**) |

- The Intel iGPU is in the RTX 5090's PC: its CPU rows are in that column.
- VMAF and NEG on the CPU are libvmaf's own code, not sped up.

Measured with the [release benchmark](fast/BENCHMARK.md): VideoQ's HDR10
test clip (3840x2160, 10-bit) against an x265 CRF 22 encode and, scaled to
1080p, against a 1080p one. 48 frame pairs are decoded into memory first,
so decoding is not counted; GPUs get them from system memory, and the CPU
code uses all threads. Each number is the median of three rounds of 10
seconds or more. Official is Netflix's libvmaf master at
[acdd9376](https://github.com/Netflix/vmaf/commit/acdd9376) (release 3.2.1
does not build with Visual Studio). With CUDA, official libvmaf uploads the
CPU pictures itself; libvmaf-fast's bindings upload only the luma.

## How it works

**VMAF and NEG on Vulkan.** `fast/vulkan` ports libvmaf's CUDA code for VIF,
ADM and motion to compute shaders (Slang, embedded in `vmaf_vulkan.dll`).
Drivers round floats their own way and some GPUs, such as Intel's
integrated ones, have no doubles. To give CUDA's exact values anyway:

- double arithmetic is done in integers (VIF's steps rounded to 53 bits,
  ADM's angle test up to 128 bits), with a float shortcut where the result
  is certain;
- float results come from tables made on the host: ADM's reciprocals, and
  VIF's logarithms from CUDA's own `log2f`;
- sums are 64-bit and rounded where CUDA rounds them.

libvmaf's own expressions make the features from the sums, and libvmaf
predicts the score. NEG differs from VMAF only in VIF's and ADM's
enhancement gain limit, so everything the limit does not affect is
calculated once for both (libvmaf calculates VIF and ADM twice). Passes are
fused (VIF's filters and statistics; ADM's decoupling and masking, per
scale), and frames from the CPU go straight into GPU memory where the GPU
has Resizable BAR.

**VMAF v1 on the GPU.** libvmaf has no GPU code for VMAF v1's features, so
the engine follows its CPU code: ADM3 and motion3 in the same integer
arithmetic (with the HFR models' five-frame motion window), CAMBI, and
SpEED's chroma filtering, its float multiplies and adds kept unfused to
round as libvmaf's do. SpEED's last step (a covariance, eigenvalues, a log)
is libvmaf's own code, on CPU threads.

**PSNR, SSIM and XPSNR** in `libvmaf.dll`: PSNR runs on libvmaf's thread
pool, and uses XPSNR's squared errors when both are asked for. SSIM is
decimated straight from the picture's samples, not from two full-size float
copies. XPSNR is FFmpeg's `xpsnr` filter, ported to a libvmaf feature
extractor.

**Frames from a hardware decoder** need not leave the GPU: CUDA (for
NVIDIA's decoder) or another Vulkan device (for AMD's) writes into the
engine's input buffers, and VMAF v1 can read the decoder's Direct3D 11
textures.

**Driver bugs.** Intel's driver read a push-constant array indexed in a
loop as zeros; AMD's (Radeon 780M, 8060S) read ADM's reciprocal table's
entries for values below -1 as zero. The shaders avoid both. A miscompiled
shader gives wrong numbers, not an error, so a program should run the
bindings' self-tests, `vulkan.probe()` and `v1.probe()`, before trusting a
GPU.

## libvmaf's CUDA code

Upstream master at
[b41d2340](https://github.com/Netflix/vmaf/commit/b41d2340a881c69682efb08fbffd0856485c57b9),
with these pull requests merged, each at the commit tested. They are their
authors' work:

| Pull request | Author | Fixes |
|---|---|---|
| [#1573](https://github.com/Netflix/vmaf/pull/1573) | StormBytePP | CUDA builds outside the source tree; a crash in pinned pictures |
| [#1583](https://github.com/Netflix/vmaf/pull/1583) | jmsether | A double flush that fails threaded CUDA runs at the end; a race in motion |
| [#1644](https://github.com/Netflix/vmaf/pull/1644) | lusoris | CUDA motion: the CPU's edge mirror |
| [#1612](https://github.com/Netflix/vmaf/pull/1612) | BardieJoensen | CUDA motion: an uninitialized first-frame blur; a leak |
| [#1614](https://github.com/Netflix/vmaf/pull/1614) | BardieJoensen | CUDA VIF: a race giving NaN scores |
| [#1647](https://github.com/Netflix/vmaf/pull/1647)-[#1651](https://github.com/Netflix/vmaf/pull/1651) | lusoris | CUDA ADM: five differences from the CPU code |
| [#1652](https://github.com/Netflix/vmaf/pull/1652) | lusoris | `vmaf_read_pictures` leaking its pictures on failure |
| [#1382](https://github.com/Netflix/vmaf/pull/1382) | shin.han | `cuMemFreeAsync` replaced by `cuMemFree` in `vmaf_cuda_picture_free` |
| [#1645](https://github.com/Netflix/vmaf/pull/1645) | lusoris | The CUDA extractors' kernel modules unloaded and streams destroyed |

The fork's own fixes:

- `vmaf_cuda_fetch_preallocated_picture` reuses host pictures from a pool
  instead of allocating, page-locking and zeroing one per frame: 4K VMAF
  from system memory about 5.5x as fast (64 to 350 fps).
- ADM's and VIF's scratch buffers are the size their kernels use: 4K VMAF +
  NEG in 35% less GPU memory (2644 to 1708 MB).

CUDA's motion still differs slightly from the CPU's: the CPU blurs the
difference of two frames, CUDA blurs each frame and subtracts, which rounds
differently (the second cause in
[Netflix/vmaf#1562](https://github.com/Netflix/vmaf/issues/1562); #1644 fixes
the first). On the benchmark's video, CUDA's VMAF and NEG, and so Vulkan's,
differ from the CPU's by at most 0.000035 a frame. VMAF v1 on the GPU does
not differ.

## How it is checked

Comparisons are exact (features bit for bit, scores by equality); the
scripts are in `fast/tests`.

- On the RTX 5090 and the Intel iGPU, Vulkan VMAF and NEG equal CUDA's in
  45 cases: 33x35 to 8K, 8 and 10 bits, special frames (black, noise,
  sharpened...) and frame skipping.
- On those and the Radeon 8060S, VMAF v1 on the GPU equals libvmaf's CPU
  code in 73 cases: all eight v1.0.16 models, 640x480 to 4K, 8 and 10 bits,
  special frames and frame skipping.
- On the 8060S, which has no CUDA, every sum of `diagnose_vmaf_vulkan.py`
  equals the reference.
- The release benchmark checks every score on all three PCs: PSNR and SSIM
  equal official libvmaf's, XPSNR FFmpeg's.
- libvmaf's own C tests pass.
- The builds are reproducible: the 8060S PC's DLLs, built with a Chinese
  Visual Studio, are the main PC's byte for byte.

## Licence

BSD-2-Clause-Patent, as libvmaf ([LICENSE](LICENSE)), except XPSNR:
`libvmaf/src/feature/xpsnr.c` and `xpsnr_template.c` are ported from FFmpeg
and stay under the LGPL 2.1 or later, so `libvmaf.dll`, which includes them,
is under it too for that code (`licenses/LICENSE.xpsnr.txt` in the release).
The Vulkan engine, derived from libvmaf's feature extractors, keeps their
licence and Netflix's and NVIDIA's copyright notices. Questions about VMAF
itself belong upstream; problems with anything under `fast/` belong here.
