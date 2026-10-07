"""The release benchmark (fast/BENCHMARK.md): how fast VMAF + NEG, VMAF v1,
PSNR, SSIM and XPSNR score frames held in memory with libvmaf-fast's new
build (this checkout), its last release (3.2.0-fast.1), official libvmaf
(Netflix's master, built here as this checkout is) and FFmpeg's xpsnr filter.
Every implementation scores the same frames; each one's per-frame scores are
checked against its baseline's.

    python fast/tests/bench_release.py build --python PY_WITH_MESON [--no-cuda] [--release-dir DIR]
    python fast/tests/bench_release.py prepare --reference REF --distorted-2160 D4K --distorted-1080 D1080
    python fast/tests/bench_release.py run [--rounds 3] [--seconds 10] [--cooldown 2]
    python fast/tests/bench_release.py report RESULTS.json [MORE.json ...]

build makes the three builds in --work (default %TEMP%\\vmaf-fast-bench):
official libvmaf and this checkout's libvmaf and Vulkan engine from source,
3.2.0-fast.1 from its GitHub release; prepare decodes the frames there; run
times every implementation this PC has and writes results\\<computer>.json
and .md there; report puts several computers' results in one markdown file.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SIZES = {"1080": (1920, 1080), "2160": (3840, 2160)}
SIZE_NAMES = {"1080": "1080p", "2160": "4K"}
BITS = 10
FRAME_RATE = 120
#: VideoMetricsLab's own VMAF v1 models for these sizes (its defaults).
V1_MODELS = {"1080": "vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json", "2160": "vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json"}
METRICS = {"vmaf_neg": "VMAF + NEG", "vmaf_v1": "VMAF v1", "psnr": "PSNR", "ssim": "SSIM", "xpsnr": "XPSNR"}
BUILD_NAMES = {"official": "official libvmaf", "release": "libvmaf-fast 3.2.0-fast.1", "new": "libvmaf-fast new",
               "ffmpeg": "FFmpeg"}
#: libvmaf's CPU features: the extractor, the score read, its decimals (as FFmpeg's filters write them).
CPU_FEATURES = {"psnr": ("psnr", "psnr_y", 6), "ssim": ("float_ssim", "float_ssim", 6), "xpsnr": ("xpsnr", "xpsnr_y", 4)}
#: Other sessions' benchmarks, which this one must not time over (nor they over it).
FOREIGN = re.compile(r"bench_|vship_runs|compare_vmaf|diagnose_vmaf|clean_runs|sweep_")
#: Official libvmaf: Netflix's master of 2026-10-05 (release 3.2.1 does not build with Visual Studio).
OFFICIAL_COMMIT = "acdd9376e978b9eea13bbf304cd3f24b2bd2ef32"
NETFLIX = "https://github.com/Netflix/vmaf.git"
RELEASE_TAG = "v3.2.0-fast.1"
RELEASE_ZIP = ("https://github.com/4KVCD/libvmaf-fast/releases/download/v3.2.0-fast.1/"
               "libvmaf-fast-3.2.0-fast.1-windows-x64.zip")
RELEASE_SHA256 = "3ad8ab2b0fd26f0c89389d55e086f4cad1d0e059d63845ecf88e5aaed6c38f9c"


def frame_bytes(size: str) -> int:
    width, height = SIZES[size]
    return (width * height + 2 * (width // 2) * (height // 2)) * 2


def unchecked(command, **options) -> subprocess.CompletedProcess:
    """A command whose exit code the caller looks at itself."""
    return subprocess.run(command, check=False, **options)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


# -------------------------------------------------------------------- build

def build(args) -> int:
    """The three builds, each a folder as VMAF_FAST_DIST takes it (libvmaf\\,
    vmaf_vulkan\\) in --work\\kit: official (Netflix's master, its libvmaf
    built with this checkout's build script) and new (this checkout's HEAD,
    libvmaf and Vulkan engine) from source, with the same compiler and
    options; release (3.2.0-fast.1) from its GitHub release. libvmaf with CUDA
    where the CUDA Toolkit is (CUDA_PATH), else without: the CPU code is the
    same. Worktrees of the three commits in --work\\src: the sources, and the
    Python bindings each build is driven with."""
    work = Path(args.work)
    kit, src = work / "kit", work / "src"
    python = args.python or sys.executable
    if unchecked([python, "-c", "import mesonbuild"], capture_output=True).returncode:
        raise SystemExit(f"{python} has no meson: install meson and ninja in a Python (pip install meson ninja) "
                         "and give it with --python")
    cuda_path = os.environ.get("CUDA_PATH", "")
    cuda = not args.no_cuda and bool(cuda_path) and (Path(cuda_path) / "bin" / "nvcc.exe").is_file()
    repo = str(REPOSITORY)
    if git("-C", repo, "status", "--porcelain", "--untracked-files=no"):
        print("Note: this checkout has changes; the new build is its last commit, without them.", flush=True)
    if unchecked(["git", "-C", repo, "cat-file", "-e", f"{OFFICIAL_COMMIT}^{{commit}}"],
                      capture_output=True).returncode:
        print(f"Fetching Netflix's master from {NETFLIX}", flush=True)
        git("-C", repo, "fetch", "--quiet", NETFLIX, "master")
    if unchecked(["git", "-C", repo, "cat-file", "-e", f"{RELEASE_TAG}^{{commit}}"],
                      capture_output=True).returncode:
        git("-C", repo, "fetch", "--quiet", "origin", "tag", RELEASE_TAG)
    trees = {"official": OFFICIAL_COMMIT, "new": git("-C", repo, "rev-parse", "HEAD"), "release": RELEASE_TAG}
    for name, commit in trees.items():
        tree = src / name
        if tree.exists():
            git("-C", str(tree), "checkout", "--quiet", "--detach", "--force", commit)
        else:
            git("-C", repo, "worktree", "add", "--quiet", "--detach", str(tree), commit)
    # pthread-win32 (a submodule of both trees built) from this checkout's own copy of it where it has one,
    # else the build script fetches it from GitHub. Only for this command: nothing goes into git's config.
    pthreads = REPOSITORY / "libvmaf" / "subprojects" / "pthread-win32"
    if (pthreads / ".git").exists():
        for name in ("official", "new"):
            tree = str(src / name)
            module = git("-C", tree, "config", "-f", ".gitmodules", "--get-regexp", r"submodule\..*\.path",
                         "libvmaf/subprojects/pthread-win32").split()[0][len("submodule."):-len(".path")]
            git("-C", tree, "-c", f"submodule.{module}.url={pthreads}", "-c", "protocol.file.allow=always",
                "submodule", "update", "--quiet", "--init", "libvmaf/subprojects/pthread-win32")
    script = Path("fast") / "scripts" / "build_libvmaf_cuda.ps1"
    (src / "official" / script).parent.mkdir(parents=True, exist_ok=True)
    (src / "official" / script).write_bytes((src / "new" / script).read_bytes())  # Netflix's tree has none
    powershell_file = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    for name in ("official", "new"):
        tree = src / name
        print(f"Building {BUILD_NAMES[name]}'s libvmaf ({trees[name][:8]}) {'with' if cuda else 'without'} CUDA",
              flush=True)
        # The build folder inside libvmaf\: upstream's CUDA build finds its headers only from there.
        command = [*powershell_file, str(tree / script), "-Python", python, "-WorkDirectory", str(tree / "libvmaf"),
                   "-OutputDirectory", str(kit / name / "libvmaf")]
        command += ["-CudaPath", cuda_path] if cuda else ["-NoCuda"]
        if unchecked(command).returncode:
            raise SystemExit(f"building {name}'s libvmaf failed")
    print(f"Building {BUILD_NAMES['new']}'s Vulkan engine", flush=True)
    if unchecked([*powershell_file, str(src / "new" / "fast" / "scripts" / "build_vmaf_vulkan.ps1"),
                       "-OutputDirectory", str(kit / "new" / "vmaf_vulkan")]).returncode:
        raise SystemExit("building the Vulkan engine failed")
    release = kit / "release"
    if release.exists():
        shutil.rmtree(release)
    if args.release_dir:  # the release's two folders, unpacked before (VideoMetricsLab's vmaf_app\tools has them)
        for folder in ("libvmaf", "vmaf_vulkan"):
            shutil.copytree(Path(args.release_dir) / folder, release / folder)
    else:
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "release.zip"
            print(f"Downloading {RELEASE_ZIP}", flush=True)
            urllib.request.urlretrieve(RELEASE_ZIP, archive)
            if sha256(archive) != RELEASE_SHA256:
                raise SystemExit("the 3.2.0-fast.1 release's archive is not the one released (SHA-256)")
            with zipfile.ZipFile(archive) as packed:
                packed.extractall(release)
        for line in (release / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            if sha256(release / name) != expected:
                raise SystemExit(f"{name} does not match the release's SHA256SUMS")
    for name in ("official", "release", "new"):
        dll = kit / name / "libvmaf" / "libvmaf.dll"
        lib = ctypes.CDLL(str(dll))
        lib.vmaf_version.restype = ctypes.c_char_p
        print(f"{name}: libvmaf {lib.vmaf_version().decode()}, SHA-256 {sha256(dll)}")
    return 0


# ------------------------------------------------------------------ prepare

def prepare(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
    meta = {"reference": Path(args.reference).name, "start": args.start, "sizes": {}}
    count = args.frames
    for size, distorted in (("2160", args.distorted_2160), ("1080", args.distorted_1080)):
        width, height = SIZES[size]
        pick = f"select='between(n\\,{args.start}\\,{args.start + count - 1})'"
        entry = {"distorted": Path(distorted).name, "frames": count}
        for name, path, chain in (
                ("ref", args.reference,
                 pick if size == "2160" else f"{pick},scale={width}:{height}:flags=lanczos+accurate_rnd+bitexact"),
                ("dis", distorted, pick)):
            out = work / f"{name}_{size}.yuv"
            print(f"{out.name}: frames {args.start}-{args.start + count - 1} of {Path(path).name}", flush=True)
            subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(path), "-map", "0:v:0",
                            "-vf", chain, "-fps_mode", "passthrough", "-frames:v", str(count), "-pix_fmt",
                            "yuv420p10le", "-f", "rawvideo", str(out)], check=True)
            if out.stat().st_size != count * frame_bytes(size):
                raise SystemExit(f"{out.name}: {out.stat().st_size} bytes, not {count} frames of {width}x{height}")
            entry[name] = sha256(out)
        meta["sizes"][size] = entry
    (work / "frames.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


# -------------------------------------------------------------------- child

class _Configuration(ctypes.Structure):
    _fields_ = [("log_level", ctypes.c_int), ("n_threads", ctypes.c_uint), ("n_subsample", ctypes.c_uint),
                ("cpumask", ctypes.c_uint64), ("gpumask", ctypes.c_uint64)]


_PICTURE = [("pix_fmt", ctypes.c_int), ("bpc", ctypes.c_uint), ("w", ctypes.c_uint * 3), ("h", ctypes.c_uint * 3),
            ("stride", ctypes.c_ssize_t * 3), ("data", ctypes.c_void_p * 3)]


class _PlainPicture(ctypes.Structure):  # libvmaf before upstream's vmaf_picture_convert (3.2.0-fast.1)
    _fields_ = [*_PICTURE, ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p)]


class _ColorPicture(ctypes.Structure):  # since it: a VmafColor (four enums) between data and ref
    _fields_ = [*_PICTURE, ("color", ctypes.c_int * 4), ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p)]


class _PictureParameters(ctypes.Structure):
    _fields_ = [("w", ctypes.c_uint), ("h", ctypes.c_uint), ("bpc", ctypes.c_uint), ("pix_fmt", ctypes.c_int)]


class _PictureConfiguration(ctypes.Structure):
    _fields_ = [("pic_params", _PictureParameters), ("pic_cnt", ctypes.c_uint)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("flags", ctypes.c_uint64)]


class LibvmafCpu:
    """libvmaf's CPU code on 4:2:0 pictures (as FFmpeg's libvmaf filter hands
    them over), from all threads: models (built-in versions or .json paths)
    or one of CPU_FEATURES."""

    def __init__(self, dll: Path, width: int, height: int, threads: int, models: dict[str, str] | None = None,
                 feature: str | None = None):
        import numpy as np
        self._np = np
        self._lib = lib = ctypes.CDLL(str(dll))
        self._picture = _ColorPicture if hasattr(lib, "vmaf_picture_convert") else _PlainPicture
        pointer, handle = ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        for name, argtypes in (
                ("vmaf_init", [pointer, _Configuration]), ("vmaf_close", [handle]),
                ("vmaf_model_load", [pointer, ctypes.POINTER(_ModelConfig), ctypes.c_char_p]),
                ("vmaf_model_load_from_path", [pointer, ctypes.POINTER(_ModelConfig), ctypes.c_char_p]),
                ("vmaf_use_features_from_model", [handle, handle]), ("vmaf_use_feature", [handle, ctypes.c_char_p, handle]),
                ("vmaf_feature_dictionary_set", [pointer, ctypes.c_char_p, ctypes.c_char_p]),
                ("vmaf_preallocate_pictures", [handle, _PictureConfiguration]),
                ("vmaf_fetch_preallocated_picture", [handle, ctypes.POINTER(self._picture)]),
                ("vmaf_read_pictures", [handle, ctypes.POINTER(self._picture), ctypes.POINTER(self._picture),
                                        ctypes.c_uint]),
                ("vmaf_score_at_index", [handle, handle, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]),
                ("vmaf_feature_score_at_index", [handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double),
                                                 ctypes.c_uint])):
            getattr(lib, name).restype, getattr(lib, name).argtypes = ctypes.c_int, argtypes
        lib.vmaf_model_destroy.restype, lib.vmaf_model_destroy.argtypes = None, [handle]
        self._context = ctypes.c_void_p()
        self._models: dict[str, ctypes.c_void_p] = {}
        self._feature = feature
        self._count = 0
        self._check(lib.vmaf_init(ctypes.byref(self._context), _Configuration(1, threads, 1, 0, 0)), "Starting libvmaf")
        for name, model in (models or {}).items():
            loaded, config = ctypes.c_void_p(), _ModelConfig(name.encode(), 0)
            if model.endswith(".json"):
                error = lib.vmaf_model_load_from_path(ctypes.byref(loaded), ctypes.byref(config), str(model).encode())
            else:
                error = lib.vmaf_model_load(ctypes.byref(loaded), ctypes.byref(config), model.encode())
            self._check(error, f"Loading {model}")
            self._models[name] = loaded
            self._check(lib.vmaf_use_features_from_model(self._context, loaded), f"Setting up {model}")
        if feature:
            options = ctypes.c_void_p()
            if feature == "xpsnr":  # weighted by the reference, as FFmpeg's filter by its first input
                self._check(lib.vmaf_feature_dictionary_set(ctypes.byref(options), b"frame_rate",
                                                            str(FRAME_RATE).encode()), "xpsnr's frame rate")
            self._check(lib.vmaf_use_feature(self._context, CPU_FEATURES[feature][0].encode(), options),
                        f"Starting {feature}")
        self._check(lib.vmaf_preallocate_pictures(self._context, _PictureConfiguration(
            _PictureParameters(width, height, BITS, 1), 2 * (max(1, threads) + 2))), "Allocating pictures")
        luma, chroma = width * height * 2, (width // 2) * (height // 2) * 2
        #: (offset, rows, row bytes) of each plane in a packed frame.
        self._planes = ((0, height, width * 2), (luma, height // 2, width), (luma + chroma, height // 2, width))

    @staticmethod
    def _check(error: int, what: str) -> None:
        if error:
            raise RuntimeError(f"{what} failed (libvmaf error {error})")

    def add(self, reference, distorted) -> None:
        np, lib = self._np, self._lib
        pictures = (self._picture(), self._picture())
        for picture, frame in zip(pictures, (reference, distorted), strict=True):
            self._check(lib.vmaf_fetch_preallocated_picture(self._context, ctypes.byref(picture)), "Taking a picture")
            source = np.frombuffer(frame, dtype=np.uint8)
            for plane, (offset, rows, row_bytes) in enumerate(self._planes):
                target = np.ctypeslib.as_array(ctypes.cast(picture.data[plane], ctypes.POINTER(ctypes.c_uint8)),
                                               shape=(rows, picture.stride[plane]))
                target[:, :row_bytes] = source[offset:offset + rows * row_bytes].reshape(rows, row_bytes)
        self._check(lib.vmaf_read_pictures(self._context, ctypes.byref(pictures[0]), ctypes.byref(pictures[1]),
                                           self._count), f"Scoring frame {self._count}")
        self._count += 1

    def finish(self):
        np, lib = self._np, self._lib
        self._check(lib.vmaf_read_pictures(self._context, None, None, 0), "Finishing")
        value, scores = ctypes.c_double(), {}
        for name, model in self._models.items():
            column = np.empty(self._count)
            for frame in range(self._count):
                self._check(lib.vmaf_score_at_index(self._context, model, ctypes.byref(value), frame), "Reading")
                column[frame] = float(f"{value.value:.6f}")
            scores[name] = column
        if self._feature:
            _extractor, score, decimals = CPU_FEATURES[self._feature]
            column = np.empty(self._count)
            for frame in range(self._count):
                self._check(lib.vmaf_feature_score_at_index(self._context, score.encode(), ctypes.byref(value), frame),
                            "Reading")
                column[frame] = float(f"{value.value:.{decimals}f}")
            scores[self._feature] = column
        return np.arange(self._count), scores

    def close(self) -> None:
        for model in self._models.values():
            self._lib.vmaf_model_destroy(model)
        self._models = {}
        if self._context:
            self._lib.vmaf_close(self._context)
            self._context = ctypes.c_void_p()


def load_frames(work: Path, size: str) -> tuple[list[memoryview], list[memoryview]]:
    """The frames, mapped from prepare's files copy-on-write: every implementation of a metric holds them at
    once, and they share the same memory (written by none). Writable, as the bindings' from_buffer wants."""
    import mmap
    each = frame_bytes(size)
    pairs = []
    for name in ("ref", "dis"):
        with open(work / f"{name}_{size}.yuv", "rb") as file:
            mapped = memoryview(mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY))
        pairs.append([mapped[start:start + each] for start in range(0, len(mapped) - each + 1, each)])
    return pairs[0], pairs[1]


def ffmpeg_xpsnr(ffmpeg: str, work: Path, size: str, frames: int, loops: int) -> tuple[float, list[float]]:
    """FFmpeg's xpsnr filter over the raw frames, `loops` + 1 times: the seconds FFmpeg's -benchmark gives
    for the run (rtime: from its first frame read, not the process's start), the first pass's XPSNR y."""
    width, height = SIZES[size]
    raw = ["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{width}x{height}", "-r", str(FRAME_RATE)]
    with tempfile.TemporaryDirectory() as folder:
        stats = Path(folder) / "xpsnr.txt"
        # The filter's own escaping of a Windows path: forward slashes, the drive's colon escaped.
        target = str(stats).replace("\\", "/").replace(":", "\\:")
        process = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-benchmark", "-stream_loop",
                                  str(loops), *raw, "-i", str(work / f"ref_{size}.yuv"), "-stream_loop", str(loops),
                                  *raw, "-i", str(work / f"dis_{size}.yuv"), "-lavfi",
                                  f"[0:v][1:v]xpsnr=stats_file='{target}'", "-f", "null", "-"],
                                 capture_output=True, text=True, check=True)
        seconds = float(re.findall(r"rtime=([0-9.]+)s", process.stderr)[-1])
        values = [float(match) for match in re.findall(r"XPSNR y: *([0-9.]+|inf)", stats.read_text())]
    return seconds, values[:frames]


def reply(message: dict) -> None:
    print(json.dumps(message), flush=True)


def child(args) -> int:
    """An implementation at a size, in a process of its own, run by the parent
    a line on stdin at a time, so the implementations of a metric take turns
    while each is set up once. Set up, it scores the frames once (not timed:
    these scores are checked) and says READY. Each "round": a pass over the
    frames to fill its queue again, then passes for --seconds, timed from the
    end of one pass to the end of another (every scorer takes a pair only when
    it has room for it, so its queue is as full at both ends and the frames
    counted are the frames scored in that time), then a wait for its queue to
    empty, so none of its work runs into the next one's round. "finish":
    the checked scores, and exit. The build's bindings come from --tree
    (fast\\python), its DLLs from --dist (VMAF_FAST_DIST, set by the parent)."""
    work, size, dist = Path(args.work), args.size, Path(args.dist)
    if args.kind == "ffmpeg":
        frames = json.loads((work / "frames.json").read_text())["sizes"][size]["frames"]
        first, values = ffmpeg_xpsnr(args.ffmpeg, work, size, frames, 0)
        loops = max(1, round(args.seconds / first)) if first > 0 else 1
        reply({"ready": os.getpid()})
        for command in sys.stdin:
            if command.strip() != "round":
                break
            seconds, _ = ffmpeg_xpsnr(args.ffmpeg, work, size, frames, loops)
            reply({"fps": frames * (loops + 1) / seconds, "frames": frames * (loops + 1), "seconds": seconds})
        reply({"scores": {"xpsnr": values}})
        return 0
    sys.path.insert(0, str(Path(args.tree) / "fast" / "python"))
    width, height = SIZES[size]
    threads = os.cpu_count() or 1
    reference, distorted = load_frames(work, size)
    models = {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}
    v1_model = Path(args.tree) / "model" / V1_MODELS[size]
    dll = dist / "libvmaf" / "libvmaf.dll"
    if args.kind == "cpu" and args.metric == "vmaf_neg":
        scorer = LibvmafCpu(dll, width, height, threads, models=models)
    elif args.kind == "cpu" and args.metric == "vmaf_v1":
        scorer = LibvmafCpu(dll, width, height, threads, models={"vmaf_v1": str(v1_model)})
    elif args.kind == "cpu":
        scorer = LibvmafCpu(dll, width, height, threads, feature=args.metric)
    elif args.kind == "cuda":
        from vmaf_fast import libvmaf
        scorer = libvmaf.GpuScorer(width, height, BITS, models)
    elif args.kind == "vulkan":
        from vmaf_fast import vulkan
        scorer = vulkan.VulkanScorer(width, height, BITS, models, device=args.device)
    elif args.kind == "v1gpu":
        from vmaf_fast import v1
        scorer = v1.V1Scorer(width, height, BITS, v1_model, device=args.device)
    else:
        raise SystemExit(f"no implementation {args.kind}")
    # The most frame pairs a scorer has in flight: libvmaf's CPU code one per thread and two more (its
    # pictures, preallocated two a pair); the GPU scorers a few.
    in_flight = threads + 2 if isinstance(scorer, LibvmafCpu) else 4
    try:
        count = len(reference)

        def one_pass() -> None:
            for index in range(count):
                scorer.add(reference[index], distorted[index])

        one_pass()
        reply({"ready": os.getpid()})  # the parent reads a CUDA process's GPU memory now, set up and scoring
        for command in sys.stdin:
            if command.strip() != "round":
                break
            one_pass()  # its queue full again (it emptied while the others had their rounds)
            timed = 0
            started = time.perf_counter()
            while True:
                one_pass()
                timed += count
                seconds = time.perf_counter() - started
                if seconds >= args.seconds:
                    break
            time.sleep(max(0.2, 1.5 * in_flight * seconds / timed))  # its queue's last pairs done
            reply({"fps": timed / seconds, "frames": timed, "seconds": seconds})
        _frames, scores = scorer.finish()
    finally:
        scorer.close()
    keep = ("vmaf", "vmaf_neg") if args.metric == "vmaf_neg" else None
    out = {}
    for name, column in scores.items():
        if keep is None or name in keep:
            out[name if keep else args.metric] = [float(value) for value in column[:count]]
    version = ctypes.CDLL(str(dll))
    version.vmaf_version.restype = ctypes.c_char_p
    reply({"scores": out, "libvmaf": version.vmaf_version().decode()})
    return 0


# ---------------------------------------------------------------------- run

def powershell(script: str) -> str:
    return unchecked(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True).stdout


def machine() -> dict:
    """What the results were measured on (read only: nothing here is changed)."""
    info = {"computer": platform.node(), "os": platform.platform(), "python": platform.python_version()}
    cpu = powershell("$p = Get-CimInstance Win32_Processor | Select-Object -First 1; "
                     "'{0}|{1}|{2}' -f $p.Name.Trim(), $p.NumberOfCores, $p.NumberOfLogicalProcessors").strip()
    if cpu.count("|") == 2:
        name, cores, logical = cpu.split("|")
        info.update(cpu=name, cores=int(cores), threads=int(logical))
    memory = powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory").strip()
    if memory.isdigit():
        info["memory_gb"] = round(int(memory) / 2 ** 30)
    info["gpus"] = [line.strip() for line in powershell(
        "Get-CimInstance Win32_VideoController | ForEach-Object { '{0} (driver {1})' -f $_.Name, $_.DriverVersion }"
    ).splitlines() if line.strip()]
    with contextlib.suppress(OSError):
        info["nvidia_driver"] = unchecked(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip()
    info["power_scheme"] = unchecked(["powercfg", "/getactivescheme"], capture_output=True,
                                          text=True).stdout.strip()
    battery = powershell("Get-CimInstance Win32_Battery | ForEach-Object { $_.BatteryStatus }").strip()
    if battery:
        info["on_ac_power"] = battery.splitlines()[0].strip() == "2"
    return info


def others_running() -> bool:
    import psutil
    me = psutil.Process()
    # This run's own processes: its children, and the python.exe launcher of a venv it runs under.
    mine = {me.pid, *(child.pid for child in me.children(recursive=True)), *(parent.pid for parent in me.parents())}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (process.info["pid"] not in mine and (process.info["name"] or "").lower().startswith("python")
                    and FOREIGN.search(" ".join(process.info["cmdline"] or []))):
                return True
        except psutil.Error:
            continue
    return False


def nvidia_busy() -> float:
    """Other processes' use of NVIDIA GPUs (percent of the SMs), 0 without nvidia-smi."""
    try:
        out = unchecked(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True).stdout
    except OSError:
        return 0.0
    busy = 0.0
    for line in out.splitlines():
        parts = line.split()
        if line.startswith("#") or len(parts) < 4 or parts[3] == "-":
            continue
        if parts[-1].lower().startswith(("dwm", "claude")):
            continue
        with contextlib.suppress(ValueError):
            busy += float(parts[3])
    return busy


def wait_quiet(cpu_limit: float, longest: float = 1800.0) -> bool:
    """Until no other benchmark runs, other processes leave the NVIDIA GPU and
    the CPU idle (two samples running): True, or False after `longest`."""
    import psutil
    started, quiet = time.monotonic(), 0
    while time.monotonic() - started < longest:
        if not others_running() and nvidia_busy() < 5 and psutil.cpu_percent(interval=1.0) < cpu_limit:
            quiet += 1
            if quiet >= 2:
                return True
        else:
            quiet = 0
            time.sleep(10)
    return False


def dedicated_gpu_memory(pid: int) -> int:
    out = powershell(f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage').CounterSamples | "
                     "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum").strip()
    try:
        return int(float(out))
    except ValueError:
        return 0


def builds(args) -> dict[str, tuple[Path, Path]]:
    """Each build: (the tree whose bindings drive it, its DLLs' folder)."""
    work = Path(args.work)
    kit, src = work / "kit", work / "src"
    return {"official": (src / "new", kit / "official"), "release": (src / "release", kit / "release"),
            "new": (src / "new", kit / "new"), "ffmpeg": (src / "new", kit / "new")}


def plan_rows(args) -> tuple[list[dict], list[dict]]:
    """The rows this PC can run, and its Vulkan GPUs."""
    tree, dist = builds(args)["new"]
    os.environ["VMAF_FAST_DIST"] = str(dist)
    sys.path.insert(0, str(tree / "fast" / "python"))
    from vmaf_fast import vulkan
    found = vulkan.devices()
    gpus = [{"index": device.index, "name": device.name, "vendor": device.vendor} for device in found
            if device.usable and device.kind in (vulkan._TYPE_INTEGRATED, vulkan._TYPE_DISCRETE)]
    nvidia = any(gpu["vendor"] == 0x10DE for gpu in gpus)
    rows = [{"metric": "vmaf_neg", "build": build, "kind": "cpu"} for build in ("official", "new")]
    if nvidia:
        rows += [{"metric": "vmaf_neg", "build": build, "kind": "cuda"} for build in ("official", "release", "new")]
    for gpu in gpus:
        rows += [{"metric": "vmaf_neg", "build": build, "kind": "vulkan", "device": gpu["index"]}
                 for build in ("release", "new")]
    rows.append({"metric": "vmaf_v1", "build": "official", "kind": "cpu"})
    for gpu in gpus:
        rows += [{"metric": "vmaf_v1", "build": build, "kind": "v1gpu", "device": gpu["index"]}
                 for build in ("release", "new")]
    for metric in ("psnr", "ssim"):
        rows += [{"metric": metric, "build": build, "kind": "cpu"} for build in ("official", "release", "new")]
    rows += [{"metric": "xpsnr", "build": "ffmpeg", "kind": "ffmpeg"}, {"metric": "xpsnr", "build": "new", "kind": "cpu"}]
    names = {gpu["index"]: gpu["name"] for gpu in gpus}
    vendors = {gpu["index"]: gpu["vendor"] for gpu in gpus}
    nvidia_name = next((gpu["name"] for gpu in gpus if gpu["vendor"] == 0x10DE), "NVIDIA")
    for row in rows:
        kind, device = row["kind"], row.get("device")
        if kind in ("cpu", "ffmpeg"):
            where = "CPU"
        elif kind == "cuda":
            where = f"CUDA, {nvidia_name}"
        elif kind == "vulkan":
            where = f"Vulkan, {names[device]}"
        else:  # VMAF v1: 3.2.0-fast.1 left CAMBI and SpEED to the CPU
            where = f"{'GPU + CPU' if row['build'] == 'release' else 'GPU'}, {names[device]}"
        build = BUILD_NAMES[row["build"]] + (" xpsnr filter" if kind == "ffmpeg" else "")
        row["label"] = f"{build} ({where})"
        # On the NVIDIA GPU: compared with libvmaf's CUDA there too.
        row["nvidia"] = kind == "cuda" or (device is not None and vendors[device] == 0x10DE)
        row["id"] = "/".join(str(part) for part in (row["metric"], row["build"], kind, "" if device is None else device))
    return rows, gpus


class Worker:
    """An implementation's child process at a size, driven a line at a time."""

    def __init__(self, args, row: dict, size: str, seconds: float):
        tree, dist = builds(args)[row["build"]]
        command = [sys.executable, str(Path(__file__).resolve()), "child", "--work", args.work, "--size", size,
                   "--metric", row["metric"], "--kind", row["kind"], "--tree", str(tree), "--dist", str(dist),
                   "--seconds", str(seconds), "--ffmpeg", args.ffmpeg]
        if "device" in row:
            command += ["--device", str(row["device"])]
        self.row = row
        self._log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")  # noqa: SIM115 (closed in close)
        self._process = subprocess.Popen(command, env=dict(os.environ, VMAF_FAST_DIST=str(dist)),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True)

    def read(self) -> dict:
        """The child's next reply (its other output, what a library prints, skipped); RuntimeError if it died."""
        while line := self._process.stdout.readline():
            if line.startswith("{"):
                return json.loads(line)
        self._process.wait()
        self._log.seek(0)
        raise RuntimeError(self._log.read().strip()[-2000:] or f"exit code {self._process.returncode}")

    def send(self, command: str) -> dict:
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()
        return self.read()

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._log.close()


def run(args) -> int:
    work = Path(args.work)
    args.ffmpeg = args.ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
    frames = json.loads((work / "frames.json").read_text(encoding="utf-8"))
    for name, (tree, dist) in builds(args).items():
        if not (dist / "libvmaf" / "libvmaf.dll").is_file() or not (tree / "fast" / "python").is_dir():
            raise SystemExit(f"the {name} build is missing in {work} (run build first)")
    rows, gpus = plan_rows(args)
    sizes = [size for size in ("1080", "2160") if size in frames["sizes"]]
    out_dir = work / "results"
    out_dir.mkdir(exist_ok=True)
    name = args.name or platform.node()
    results = {"machine": machine(), "gpus": gpus, "frames": frames, "rows": rows, "runs": [], "scores": {},
               "memory": {}, "settings": {"rounds": args.rounds, "seconds": args.seconds, "cooldown": args.cooldown,
                                          "threads": os.cpu_count()},
               "builds": builds_info(args)}
    path = out_dir / f"{name}.json"

    def save() -> None:
        results["summary"] = summarize(results)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1), encoding="utf-8")
        temporary.replace(path)
        (out_dir / f"{name}.md").write_text(markdown(results), encoding="utf-8")

    def failed(row: dict, size: str, error: Exception) -> None:
        results["runs"].append({"id": row["id"], "size": size, "round": 0, "quiet": True, "error": str(error)})

    # One warm-up before anything is timed, so the CPU and every GPU are at the clocks and temperatures they keep
    # for the rest of the run, which then goes on back to back: the new build at 4K, --warmup seconds on the CPU,
    # then as long on each GPU (not counted).
    if args.warmup > 0:
        print(f"Warming up: {args.warmup:g} s on the CPU and on each GPU", flush=True)
        for row in rows:
            if row["build"] == "new" and row["metric"] == "vmaf_neg" and row["kind"] in ("cpu", "vulkan"):
                worker = Worker(args, row, sizes[-1], args.warmup)
                with contextlib.suppress(RuntimeError):
                    worker.read()
                    worker.send("round")
                worker.close()
    # Each metric at each size, a device at a time (its CPU implementations, its CUDA ones, each GPU's): those
    # set up first, each in its process, then their rounds taking turns (round 1 of each, round 2 of each, ...),
    # so what the PC does from one minute to the next falls on all of them alike. A device's: the comparisons
    # that are close are between those; and a whole metric's processes alive at once (8 at 4K) held enough
    # memory to slow the iGPU (3.2.0-fast.1 at 4K 15.3 -> 11.2 fps), where no device's few did.
    def place(row: dict) -> str:
        return {"cpu": "cpu", "ffmpeg": "cpu", "cuda": "cuda"}.get(row["kind"], f"gpu {row.get('device')}")

    groups = [(size, metric, here) for size in sizes for metric in METRICS
              for here in dict.fromkeys(place(row) for row in rows if row["metric"] == metric)]
    for size, metric, here in groups:
        workers = []
        for row in (row for row in rows if row["metric"] == metric and place(row) == here):
            worker = Worker(args, row, size, args.seconds)
            try:
                ready = worker.read()
            except RuntimeError as error:
                failed(row, size, error)
                worker.close()
                continue
            if row["kind"] == "cuda" and size == "2160":  # libvmaf CUDA's GPU memory, set up and scoring
                results["memory"][row["id"]] = dedicated_gpu_memory(int(ready["ready"]))
            workers.append(worker)
        for number in range(1, args.rounds + 1):
            quiet = wait_quiet(args.cpu_quiet)
            for worker in list(workers):
                try:
                    measured = worker.send("round")
                except RuntimeError as error:
                    failed(worker.row, size, error)
                    workers.remove(worker)
                    worker.close()
                    continue
                results["runs"].append({"id": worker.row["id"], "size": size, "round": number, "quiet": quiet,
                                        **measured})
                time.sleep(args.cooldown)
        for worker in workers:
            try:
                final = worker.send("finish")
                results["scores"].setdefault(worker.row["id"], {})[size] = final["scores"]
                for record in results["runs"]:
                    if record["id"] == worker.row["id"] and record["size"] == size:
                        record["libvmaf"] = final.get("libvmaf", "")
            except RuntimeError as error:
                failed(worker.row, size, error)
            worker.close()
            fps = [record["fps"] for record in results["runs"]
                   if record["id"] == worker.row["id"] and record["size"] == size and "fps" in record]
            print(f"{SIZE_NAMES[size]:5} {METRICS[metric]:10} {worker.row['label']:58} "
                  + " ".join(f"{value:8.1f}" for value in fps) + " fps", flush=True)
        save()
    save()
    print(f"\n{path}\n{path.with_suffix('.md')}")
    return 0


def builds_info(args) -> dict:
    info = {"ffmpeg": unchecked([args.ffmpeg, "-version"], capture_output=True, text=True).stdout.split("\n")[0]}
    for name, (tree, dist) in builds(args).items():
        if name == "ffmpeg":
            continue
        dlls = {folder: sha256(dist / folder / f"{folder}.dll") for folder in ("libvmaf", "vmaf_vulkan")
                if (dist / folder / f"{folder}.dll").is_file()}
        commit = "" if name == "official" else git("-C", str(tree), "log", "-1", "--format=%h %s")[:100]
        info[name] = {"dlls": dlls, "bindings": commit}
    return info


# ------------------------------------------------------------ summary, report

def compare(values: dict, against: dict) -> str:
    """'identical', or the largest difference of any score, per frame."""
    worst = 0.0
    for name, column in values.items():
        other = against.get(name)
        if other is None or len(other) != len(column):
            return "not compared"
        for a, b in zip(column, other, strict=True):
            if a != b:
                worst = max(worst, math.inf if math.isinf(a) or math.isinf(b) else abs(a - b))
    return "identical" if worst == 0 else f"differs by up to {worst:.6g}"


def summarize(results: dict) -> list[dict]:
    rows = {row["id"]: row for row in results["rows"]}
    summary = []
    for size in results["frames"]["sizes"]:
        fps: dict[str, list[float]] = {}
        for record in results["runs"]:
            if record["size"] == size and record["round"] > 0 and "fps" in record:
                fps.setdefault(record["id"], []).append(record["fps"])
        medians = {key: statistics.median(values) for key, values in fps.items()}
        scores = {key: value[size] for key, value in results["scores"].items() if size in value}
        for key, row in rows.items():
            if key not in medians:
                continue
            entry = {"size": size, "id": key, "metric": row["metric"], "label": row["label"],
                     "fps": medians[key], "runs": len(fps[key]), "min": min(fps[key]), "max": max(fps[key]),
                     "speedup": {}, "scores": {}}
            for other, title in baselines(row, rows):
                if other in medians:
                    entry["speedup"][title] = medians[key] / medians[other]
                if key in scores and other in scores:
                    entry["scores"][title] = compare(scores[key], scores[other])
            summary.append(entry)
    return summary


def baselines(row: dict, rows: dict) -> list[tuple[str, str]]:
    """The rows a row is measured against: (id, title)."""
    metric, build, kind, device = row["metric"], row["build"], row["kind"], row.get("device", "")
    found = []

    def add(other_build: str, other_kind: str, other_device="", title: str = "") -> None:
        key = f"{metric}/{other_build}/{other_kind}/{other_device}"
        if key in rows and key != row["id"]:
            found.append((key, title or rows[key]["label"]))

    if build == "ffmpeg" or (build == "official" and kind == "cpu"):
        return found
    if metric == "xpsnr":
        add("ffmpeg", "ffmpeg")
    elif metric == "vmaf_v1":
        add("official", "cpu")
        if build == "new":
            add("release", kind, device)
    elif metric == "vmaf_neg":
        add("official", "cpu")
        if row.get("nvidia"):
            add("official", "cuda")
            if kind == "vulkan":
                add(build, "cuda")
        if build == "new":
            add("release", kind, device)
    else:
        add("official", "cpu")
        if build == "new":
            add("release", "cpu")
    return found


def markdown(results: dict) -> str:
    info = results["machine"]
    settings = results["settings"]
    lines = [f"# {info.get('computer', '')}: {info.get('cpu', '')}, {', '.join(info.get('gpus', []))}", "",
             f"{info.get('cores', '?')} cores / {info.get('threads', '?')} threads, {info.get('memory_gb', '?')} GB; "
             f"{info.get('os', '')}; NVIDIA driver {info.get('nvidia_driver') or '-'}; "
             f"{info.get('power_scheme', '')}" + ("" if info.get("on_ac_power", True) else "; ON BATTERY"), "",
             (f"Frames: {results['frames']['reference']} from frame {results['frames']['start']}, 10-bit; each "
              "implementation in a process of its own, set up once and given the frames once untimed; then "
              f"{settings['rounds']} rounds of {settings['seconds']} s or more each, a metric's implementations on "
              "the same device taking turns round by round: each number the rounds' median."), ""]
    for size in results["frames"]["sizes"]:
        entries = [entry for entry in results.get("summary", []) if entry["size"] == size]
        if not entries:
            continue
        width, height = SIZES[size]
        lines += [f"## {SIZE_NAMES[size]} ({width}x{height})", "",
                  "| Metric | Implementation | Frames/s | Range | Faster than | Scores |", "|---|---|---|---|---|---|"]
        for entry in entries:
            faster = "; ".join(f"{value:.2f}x {title}" for title, value in entry["speedup"].items())
            checks = "; ".join(f"{value} ({title})" for title, value in entry["scores"].items())
            lines.append(f"| {METRICS[entry['metric']]} | {entry['label']} | {entry['fps']:.1f} | "
                         f"{entry['min']:.1f}-{entry['max']:.1f} | {faster} | {checks} |")
        lines.append("")
    if results.get("memory"):
        lines += ["## GPU memory, 4K VMAF + NEG (libvmaf CUDA, dedicated)", ""]
        rows = {row["id"]: row for row in results["rows"]}
        for key, value in results["memory"].items():
            lines.append(f"- {rows[key]['label']}: {value / 2 ** 20:.0f} MB")
        lines.append("")
    failures = [record for record in results["runs"] if "error" in record]
    if failures:
        lines += ["## Failed", ""] + [f"- {record['id']} {SIZE_NAMES[record['size']]}: "
                                      f"{record['error'].splitlines()[-1] if record['error'] else '?'}"
                                      for record in failures] + [""]
    loud = {(record["id"], record["size"]) for record in results["runs"] if not record.get("quiet", True)}
    if loud:
        lines += [f"{len(loud)} implementations started before the PC was quiet (30 minutes waited).", ""]
    return "\n".join(lines)


def report(args) -> int:
    text = "\n\n".join(markdown(json.loads(Path(path).read_text(encoding="utf-8"))) for path in args.results)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    default_work = str(Path(tempfile.gettempdir()) / "vmaf-fast-bench")
    p = commands.add_parser("build")
    p.add_argument("--python", help="a Python with meson and ninja (default: this one)")
    p.add_argument("--no-cuda", action="store_true", help="libvmaf without CUDA even where the CUDA Toolkit is")
    p.add_argument("--release-dir", help="3.2.0-fast.1's libvmaf\\ and vmaf_vulkan\\ already unpacked (default: "
                                         "downloaded from its GitHub release)")
    p.add_argument("--work", default=default_work)
    p = commands.add_parser("prepare")
    p.add_argument("--reference", required=True)
    p.add_argument("--distorted-2160", required=True)
    p.add_argument("--distorted-1080", required=True)
    p.add_argument("--start", type=int, default=0, help="first frame (by number) of both videos")
    p.add_argument("--frames", type=int, default=48, help="frames held in memory, at each size")
    p.add_argument("--work", default=default_work)
    p.add_argument("--ffmpeg")
    p = commands.add_parser("run")
    p.add_argument("--work", default=default_work)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--warmup", type=float, default=20.0, help="seconds of warm-up on the CPU and on each GPU first")
    p.add_argument("--cooldown", type=float, default=0.0, help="seconds of rest after each implementation")
    p.add_argument("--cpu-quiet", type=float, default=15.0, help="CPU use (percent) a run waits to be under")
    p.add_argument("--name", help="the results' file name (default: the computer's name)")
    p.add_argument("--ffmpeg")
    p = commands.add_parser("child")
    for option in ("--work", "--size", "--metric", "--kind", "--tree", "--dist", "--ffmpeg"):
        p.add_argument(option)
    p.add_argument("--device", type=int)
    p.add_argument("--seconds", type=float, default=10.0)
    p = commands.add_parser("report")
    p.add_argument("results", nargs="+")
    p.add_argument("--out")
    args = parser.parse_args()
    return {"build": build, "prepare": prepare, "run": run, "child": child, "report": report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
