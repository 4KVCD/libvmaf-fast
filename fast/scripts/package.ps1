# Packages fast/dist as a release archive: libvmaf/ (build_libvmaf_cuda.ps1)
# and vmaf_vulkan/ (build_vmaf_vulkan.ps1), each with its licences, a
# BUILD.txt saying what they were built from, and SHA256SUMS of every file.
# Writes fast/release/libvmaf-fast-<version>-windows-x64.zip and its
# .sha256 beside it.
#
# Build both from a clean checkout of the commit to be released first:
# libvmaf reports the commit it was built from as its version, and this
# script refuses a checkout with changes or a libvmaf built from another
# commit.
param(
    [Parameter(Mandatory = $true)][string]$Version,  # e.g. 3.2.0-fast.1
    [string]$Python = 'python',                       # any Python 3: reads libvmaf's version
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
$repository = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$dist = Join-Path $repository 'fast/dist'
$out = if ($OutputDirectory) { $OutputDirectory } else { Join-Path $repository 'fast/release' }

$changes = git -C $repository status --porcelain --untracked-files=no
if ($changes) { throw "The checkout has uncommitted changes; build and package a commit:`n$changes" }
$commit = (git -C $repository rev-parse HEAD).Trim()
foreach ($file in 'libvmaf/libvmaf.dll', 'vmaf_vulkan/vmaf_vulkan.dll') {
    if (-not (Test-Path (Join-Path $dist $file))) { throw "fast/dist/$file is not built" }
}
$library = Join-Path $dist 'libvmaf/libvmaf.dll'
$reported = (& $Python -c "import ctypes, sys; lib = ctypes.CDLL(sys.argv[1]); lib.vmaf_version.restype = ctypes.c_char_p; print(lib.vmaf_version().decode())" $library).Trim()
if (-not $commit.StartsWith($reported)) {
    throw "libvmaf.dll was built from $reported, not from $($commit.Substring(0, 8)): build it again"
}

$name = "libvmaf-fast-$Version-windows-x64"
$stage = Join-Path $out $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
Copy-Item -Recurse (Join-Path $dist 'libvmaf') $stage
Copy-Item -Recurse (Join-Path $dist 'vmaf_vulkan') $stage
Set-Content -Path (Join-Path $stage 'BUILD.txt') -Encoding ascii -Value @(
    "libvmaf-fast $Version",
    "Built from https://github.com/4KVCD/libvmaf-fast/commit/$commit",
    "libvmaf.dll reports version $reported.",
    '',
    'libvmaf/libvmaf.dll      libvmaf with CUDA (fast/scripts/build_libvmaf_cuda.ps1)',
    'vmaf_vulkan/vmaf_vulkan.dll  VMAF features with Vulkan (fast/scripts/build_vmaf_vulkan.ps1)',
    'Both need only Windows x64 and a GPU driver; licences are in each folder.')
$sums = Get-ChildItem -Recurse -File $stage | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1).Replace('\', '/')
    "$((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $relative"
}
Set-Content -Path (Join-Path $stage 'SHA256SUMS') -Encoding ascii -Value $sums

$archive = Join-Path $out "$name.zip"
if (Test-Path $archive) { Remove-Item -Force $archive }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $archive
$hash = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -Path "$archive.sha256" -Encoding ascii -Value "$hash  $name.zip"
Remove-Item -Recurse -Force $stage
Write-Host "libvmaf-fast $Version ($($commit.Substring(0, 8))): $archive"
Write-Host "SHA-256 $hash"
$sums | ForEach-Object { Write-Host "  $_" }
