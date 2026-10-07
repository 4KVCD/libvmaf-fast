# Release benchmark

The speed numbers for libvmaf-fast's README and release notes: this build
against its last release (3.2.0-fast.1) and against official libvmaf, on each
computer, with `fast/tests/bench_release.py`. The main PC (RTX 5090, Core
Ultra 9 285K and its iGPU) runs it, and so does the AMD laptop (Radeon 780M,
Ryzen 7 8845HS). Scoring only: VideoMetricsLab's own benchmark covers whole
runs with decoding.

## What it measures

| Metric | Baselines | New (this checkout) |
|---|---|---|
| VMAF + NEG | official libvmaf on the CPU and with CUDA; 3.2.0-fast.1 with CUDA and Vulkan | CPU, CUDA (NVIDIA), Vulkan (every GPU) |
| VMAF v1 | official libvmaf on the CPU; 3.2.0-fast.1 (GPU + CPU) | all on the GPU |
| PSNR, SSIM | official libvmaf; 3.2.0-fast.1 | CPU |
| XPSNR | FFmpeg's `xpsnr` filter (official libvmaf has none) | CPU |
| GPU memory | official libvmaf CUDA, 4K VMAF + NEG | CUDA |

- **Builds** (`bench_release.py build`, in `%TEMP%\vmaf-fast-bench`):
  official libvmaf is Netflix's master at acdd9376 (2026-10-05; release
  3.2.1 does not build with Visual Studio), built with this checkout's
  `build_libvmaf_cuda.ps1`; new is this checkout's last commit, libvmaf and
  Vulkan engine, built with its scripts; 3.2.0-fast.1 is its GitHub release
  (SHA-256 checked). libvmaf with CUDA where the CUDA Toolkit is installed,
  without it elsewhere (`-NoCuda`): the CPU code is the same. Each build is
  driven by the Python bindings of its own commit.
- **Frames**: `VideoQ_HDR10_UHD_120fps_4m00s.mp4` (HEVC Main 10, HDR10, 120
  fps), frames 0-47, against `VideoQ HDR10 4K H.265 CRF 22 medium.mkv` at 4K
  and, scaled to 1080p (lanczos), against `VideoQ HDR10 1080p H.265 CRF 22
  medium.mkv`. 10-bit, decoded once into memory. VMAF v1 with
  VideoMetricsLab's models: `vmaf_v1.0.16_3d0h` at 1080p,
  `vmaf_v1.0.16_1d5h_2160` at 4K. The CPU code gets 4:2:0 pictures, as
  FFmpeg's libvmaf filter gives it, and all of the CPU's threads.
- **Method**: first a warm-up (20 s of the new build on the CPU, then on each
  GPU). Then, metric by metric and device by device (the CPU, CUDA, each
  GPU), its implementations there are set up, each in a process of its own,
  set up once and given the 48 frame pairs once untimed (the scores
  checked), and they take turns: round 1 of each, round 2 of each, round 3
  of each, back to back. (A whole metric's processes alive at once, 8 at
  4K, held enough memory to slow the iGPU by a quarter; a device's few do
  not.) A round fills the scorer's queue
  with a pass over the frames, then loops over them for at least 10 s,
  timed from the end of one pass to the end of another (a scorer takes a
  pair only when it has room for it, so its queue is as full at both ends),
  then waits for its queue to empty, so none of its work runs into the next
  one's round. Its number is the median of its 3 rounds. Taking turns puts
  what the PC does from one minute to the next (the CPU's speed drifted
  about 10% between minutes on the main PC) on all the implementations
  that are close to each other alike. Each round waits until no other benchmark runs,
  no other process uses an NVIDIA GPU and the CPU is under 15% busy. The
  frames are shared between the processes (mapped, not copied).
- **Score checks**, per frame, against the baselines. Expected: the new CPU
  VMAF + NEG, PSNR and SSIM identical to official libvmaf's; Vulkan VMAF +
  NEG identical to libvmaf CUDA's; VMAF v1 on the GPU identical to official
  libvmaf on the CPU; XPSNR identical to FFmpeg's (4 decimals). GPU VMAF + NEG
  differs from the CPU's by up to about 1e-4 (CUDA's motion, libvmaf issue
  1562), official CUDA's more (the CUDA fixes this fork carries). Anything
  else is a bug: report it and do not publish the numbers.

## Running it

The videos are not in git: Brian brings them (from the main PC's
`E:\Video encodings`).

| File | Bytes |
|---|---|
| `VideoQ_HDR10_UHD_120fps_4m00s.mp4` | 1,055,015,060 |
| `VideoQ HDR10 4K H.265 CRF 22 medium.mkv` | 425,144,502 |
| `VideoQ HDR10 1080p H.265 CRF 22 medium.mkv` | 162,407,007 |

Needs what `fast\scripts\build_libvmaf_cuda.ps1` and `build_vmaf_vulkan.ps1`
need (Visual Studio 2022 Build Tools with C++, Git for Windows, nasm, a
Python with meson and ninja; the CUDA Toolkit only for CUDA), FFmpeg 9 on
PATH, and a Python with numpy and psutil for the benchmark.

1. Check out `fast` at the commit Brian names (or later), clean.
2. Build (about 15 minutes; fetches Netflix's master into this clone and the
   3.2.0-fast.1 release from GitHub):

       python fast\tests\bench_release.py build --python D:\path\python-with-meson.exe

   It ends with each build's libvmaf version: official `acdd9376`, release
   `3ddc3d84`, new the commit checked out.
3. Decode the frames (about 1.5 GB in `%TEMP%\vmaf-fast-bench`):

       python fast\tests\bench_release.py prepare ^
           --reference "D:\path\VideoQ_HDR10_UHD_120fps_4m00s.mp4" ^
           --distorted-2160 "D:\path\VideoQ HDR10 4K H.265 CRF 22 medium.mkv" ^
           --distorted-1080 "D:\path\VideoQ HDR10 1080p H.265 CRF 22 medium.mkv"

   The frames' SHA-256 it prints must be these, the main PC's; otherwise the
   computers did not score the same frames:

   | Frames | SHA-256 |
   |---|---|
   | 4K reference | `8db8ec6a057c909892026b26b137f6c8081f03412387e3bd4b7ee1635bf67abc` |
   | 4K distorted | `d43e9a764104630377ca8cc238f962c0cad5a2eb6efd4c3320f05a6ccd2470eb` |
   | 1080p reference | `665d8cbdccbcbb13f7ae5e3576099461ed276481285e20edf2f1905ff3c33aef` |
   | 1080p distorted | `267138e8abd55cf2f59112addca77ff3074fd04019fa6cb84ac3eb02091d1abd` |

4. Plug a laptop's charger in and close what you can. Do not change
   Windows' power or display settings; the script only records the power
   plan and whether it ran on battery.
5. Run it (about 30-45 minutes; a line per implementation, results written
   after each):

       python fast\tests\bench_release.py run

   CUDA rows run only where there is an NVIDIA GPU.
6. Check `%TEMP%\vmaf-fast-bench\results\<computer>.md`: no "Failed"
   section, no implementation started before the PC was quiet, the Scores
   column as expected above.
7. Send back `<computer>.json` and `<computer>.md` (to Brian, or committed as
   `fast/benchmark-results/<computer>.*` on a branch of their own if he says
   so).

`bench_release.py report A.json B.json --out all.md` puts several computers'
results in one file.
