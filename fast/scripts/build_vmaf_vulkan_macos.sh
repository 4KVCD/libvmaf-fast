#!/bin/bash
# Builds VMAF's features for any GPU with Vulkan on macOS (arm64), as
# build_vmaf_vulkan.ps1 builds vmaf_vulkan.dll on Windows: vmaf_vulkan.dylib,
# the same shaders (build_vmaf_vulkan_shaders.py compiles that script's list
# with the same Slang release and flags) and the same engine.
#
# macOS has no Vulkan of its own: the library runs on MoltenVK (Vulkan on
# Metal) through Khronos' loader, which it loads by its rpath,
# @loader_path/../vulkan (libvulkan.1.dylib, as an application bundles it),
# else as the system finds it. Whoever loads it points the loader at
# MoltenVK (VK_DRIVER_FILES, or MoltenVK installed).
#
# clang without contraction (-ffp-contract=off), as MSVC compiles the DLL:
# the host's float and double arithmetic rounds as Windows'. The SpEED
# covariance kernels are libvmaf's x86/speed_avx2.c and v1_speed_cov.c's
# AVX2 sums, their intrinsics in plain C (fast/vulkan/portable/immintrin.h),
# lane by lane the same operations: a Mac's VMAF v1 is that of an x86 CPU
# with AVX2.
#
#     fast/scripts/build_vmaf_vulkan_macos.sh [OUTPUT_FOLDER]   (default fast/dist/vmaf_vulkan)
#
# Needs git, curl, python3 and the Command Line Tools' clang. Slang and
# Vulkan-Headers are downloaded.
set -euo pipefail

slang_version="2026.10.2"  # as build_vmaf_vulkan.ps1
slang_sha256="a0e438467aa4a5936546bc3d1ebaa677de5174b52c7a2dd9d50bce230b15df58"  # its macos-aarch64 archive
vulkan_headers_commit="3c65a01745e4a1134d32b9c2c456472212dba16d"  # 2026-09-25, as build_vmaf_vulkan.ps1
deployment_target="14.4"

repository="$(cd "$(dirname "$0")/../.." && pwd)"
output="${1:-$repository/fast/dist/vmaf_vulkan}"
work="$(mktemp -d "${TMPDIR:-/tmp}/vmaf-vulkan-macos-XXXXXX")"
trap 'rm -rf "$work"' EXIT

curl -sSfL -o "$work/slang.tar.gz" \
    "https://github.com/shader-slang/slang/releases/download/v$slang_version/slang-$slang_version-macos-aarch64.tar.gz"
actual="$(shasum -a 256 "$work/slang.tar.gz" | cut -d' ' -f1)"
if [[ "$actual" != "$slang_sha256" ]]; then
    echo "Slang $slang_version has SHA-256 $actual, expected $slang_sha256" >&2
    exit 1
fi
mkdir -p "$work/slang" && tar -xzf "$work/slang.tar.gz" -C "$work/slang"
git -c advice.detachedHead=false clone --quiet --filter=blob:none --no-checkout \
    https://github.com/KhronosGroup/Vulkan-Headers.git "$work/Vulkan-Headers"
git -C "$work/Vulkan-Headers" -c advice.detachedHead=false checkout --quiet --force "$vulkan_headers_commit"

# Metal has no 64-bit subgroup arithmetic: those sums in 32-bit pieces
# (shaders/common.slang's wave_sum_u64), the same bits.
python3 "$repository/fast/scripts/build_vmaf_vulkan_shaders.py" "$work/slang/bin/slangc" "$work/build" WAVE_INT64=0

# The commit it is built from, which vv_version() reports.
commit="$(git -C "$repository" rev-parse --short=8 HEAD)"
if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=no)" ]]; then
    commit+="-dirty"
fi

flags=(-O2 -arch arm64 -mmacosx-version-min="$deployment_target" -ffp-contract=off -fPIC
       -I "$repository/fast/vulkan/portable")
cd "$repository"
mkdir -p "$output"
for file in libvmaf/src/feature/x86/speed_avx2.c fast/vulkan/v1_speed_cov.c fast/vulkan/v1_host.c fast/vulkan/v1_speed.c; do
    clang "${flags[@]}" -std=c11 -w -c "$file" -o "$work/build/$(basename "$file" .c).o"
done
clang++ "${flags[@]}" -std=c++17 -w -fvisibility=hidden -DVV_COMMIT="\"$commit\"" -I "$work/build" \
    -I "$work/Vulkan-Headers/include" fast/vulkan/vmaf_vulkan.cpp "$work/build"/*.o -shared \
    -install_name @rpath/vmaf_vulkan.dylib -Wl,-rpath,@loader_path/../vulkan -o "$output/vmaf_vulkan.dylib"
codesign --force --sign - "$output/vmaf_vulkan.dylib" 2>/dev/null
if otool -L "$output/vmaf_vulkan.dylib" | tail -n +2 | awk '{print $1}' | grep -v -E '^(@rpath/|/usr/lib/|/System/Library/)'; then
    echo "vmaf_vulkan.dylib links a library that is not the system's" >&2
    exit 1
fi

# The notice that ships with it, as on Windows.
rm -rf "$output/licenses"
mkdir -p "$output/licenses"
cp LICENSE "$output/licenses/LICENSE.libvmaf.txt"

echo "VMAF for Vulkan ($commit): $output/vmaf_vulkan.dylib"
echo "SHA-256 $(shasum -a 256 "$output/vmaf_vulkan.dylib" | cut -d' ' -f1), $(stat -f %z "$output/vmaf_vulkan.dylib") bytes"
