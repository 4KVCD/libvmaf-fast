# Building, testing and using libvmaf-fast

What this fork adds and how it works is on the [landing page](../README.md).
Everything the fork adds lives under `fast/`:

- `fast/vulkan`: the Vulkan engine (Slang compute shaders and
  `vmaf_vulkan.cpp`), built into `vmaf_vulkan.dll`.
- `fast/scripts`: the build and packaging scripts.
- `fast/python/vmaf_fast`: Python bindings, the reference for using the DLLs.
- `fast/tests`: the bit-exact comparisons, the driver diagnosis and the
  benchmarks.

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
`fast/dist/vmaf_vulkan/vmaf_vulkan.dll`, each with its licences (libvmaf is
built in `fast/build/libvmaf`). Both link the C runtime statically and need
only Windows and a GPU driver at run time. Both builds are reproducible: the
same commit, built with the same versions of the tools (Visual Studio, the
CUDA Toolkit, meson, Slang), gives the same bytes, wherever the checkout is
and on any PC (checked on two). Other tool versions give equivalent code,
not identical bytes. libvmaf reports the commit it was built from as its
version, and the build stops without AVX-512 (nasm older than 2.14): the
Vulkan engine's SpEED uses libvmaf's AVX-512 code where the CPU has it.

## Testing

With the two DLLs built, and numpy:

```powershell
python fast\tests\compare_vmaf_vulkan.py --matrix [--device N]
python fast\tests\compare_vmaf_v1.py --matrix [--device N]
python fast\tests\diagnose_vmaf_vulkan.py [--device N]
python fast\tests\bench_vmaf_vulkan.py REFERENCE DISTORTED --size 3840x2160 --bits 10
python fast\tests\bench_readme.py REFERENCE DISTORTED [--size 1920x1080 --pairs 960] [--vmaf-only]
```

- `compare_vmaf_vulkan.py --matrix`: Vulkan's features and VMAF and NEG scores
  against libvmaf's CUDA code, 45 cases. Needs an NVIDIA GPU for the CUDA side.
- `compare_vmaf_v1.py --matrix`: VMAF v1 with the GPU against libvmaf on the
  CPU, 73 cases, on any GPU.
- `diagnose_vmaf_vulkan.py`: on a GPU whose self-test fails, finds the first
  pass whose sums differ from the reference (`vmaf_vulkan_reference.json`) and,
  for ADM's scale-0 decouple, which step.
- Each prints IDENTICAL per case; `--matrix` ends with ALL IDENTICAL or SOME
  DIFFER. Run all three on every GPU (`--device`, Vulkan's GPU number, which
  the scripts list) after any change to the engine.
- `bench_readme.py`: the landing page's speed tables, every route on every
  GPU it finds (its docstring lists the runs). The GPU-decoded routes need an
  NVIDIA GPU.

`VMAF_FAST_DIST` points the bindings at DLLs elsewhere, `VMAF_FAST_MODELS` at
the model files (by default `model/` of this repository).

## Using it

The Vulkan engine is a DLL with a small C API, not one of libvmaf's feature
extractors: a program gives it frames and gets VMAF's features back, then
predicts the score with libvmaf (`vmaf_import_feature_score`,
`vmaf_score_at_index`), as `fast/python/vmaf_fast` does:

- `vulkan.py`: `VulkanScorer`, VMAF v0.6.1 and NEG; `probe()`, the self-test a
  program should run before trusting a GPU.
- `v1.py`: `V1Scorer`, VMAF v1's four features with the engine (SpEED's last
  step on CPU threads; libvmaf's CPU extractors only for CAMBI or SpEED
  options the engine does not take); `probe()`.
- `libvmaf.py`: the binding to libvmaf's C API, and `GpuScorer`, its CUDA code.

The engine's exports are in `fast/vulkan/vmaf_vulkan.cpp`'s API section:
`vv_create` / `vv_create_v1` (with `vv_v1_cambi` and `vv_v1_speed` for VMAF
v1's CAMBI and SpEED options), `vv_submit` / `vv_submit_v1` (or `vv_staging`
and `vv_commit` to write frames in place), `vv_flush`, `vv_features` /
`vv_features_v1`, `vv_destroy`; and to take a hardware decoder's pictures
without them leaving the GPU: `vv_shared_next` / `vv_export` /
`vv_shared_device` (and `vv_shared_chroma` for VMAF v1) to share the input
buffers with its CUDA or Vulkan device, `vv_import_timeline` /
`vv_commit_after` to wait for its copies on the GPU, and `vv_pictures` to read
VMAF v1's pictures where it left them (Direct3D 11 textures).

## Releasing

Build both DLLs from a clean checkout of the commit to release, run the tests
above on every GPU available, then:

```powershell
powershell -ExecutionPolicy Bypass -File fast\scripts\package.ps1 -Version <version> -Python <any python>
```

It writes `fast/release/libvmaf-fast-<version>-windows-x64.zip` with a
`SHA256SUMS` of every file and a `.sha256` of the archive, and refuses a
checkout with changes, a DLL built from another commit, a libvmaf without
CUDA, and any file in `fast/dist` the release does not carry. Tag the commit
`libvmaf-fast-<version>` (e.g. `-Version v1`, tag `libvmaf-fast-v1`) and
attach both files to the GitHub release.

## Keeping up with upstream

`master` follows Netflix/vmaf unchanged. To take in upstream changes, merge
`master` into `fast`; keep this fork's `README.md` where they conflict. When a
merged pull request changes upstream, take its new commits only after the
tests above pass with them.
