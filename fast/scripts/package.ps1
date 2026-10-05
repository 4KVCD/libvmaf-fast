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
# Text files with LF line ends, which sha256sum -c and other systems read.
function Write-Lines([string]$path, [string[]]$lines) {
    [System.IO.File]::WriteAllText($path, (($lines -join "`n") + "`n"), [System.Text.Encoding]::ASCII)
}
Write-Lines (Join-Path $stage 'BUILD.txt') @(
    "libvmaf-fast $Version",
    "Built from https://github.com/4KVCD/libvmaf-fast/commit/$commit",
    "libvmaf.dll reports version $reported.",
    '',
    'libvmaf/libvmaf.dll      libvmaf with CUDA (fast/scripts/build_libvmaf_cuda.ps1)',
    'vmaf_vulkan/vmaf_vulkan.dll  VMAF features with Vulkan (fast/scripts/build_vmaf_vulkan.ps1)',
    'Both need only Windows x64 and a GPU driver; licences are in each folder.')
$files = Get-ChildItem -Recurse -File $stage | Sort-Object FullName | ForEach-Object {
    [pscustomobject]@{ Path = $_.FullName; Name = $_.FullName.Substring($stage.Length + 1).Replace('\', '/') }
}
$sums = $files | ForEach-Object { "$((Get-FileHash $_.Path -Algorithm SHA256).Hash.ToLowerInvariant())  $($_.Name)" }
Write-Lines (Join-Path $stage 'SHA256SUMS') $sums

# The archive written entry by entry: Windows PowerShell's Compress-Archive
# names entries with backslashes, which the zip format does not allow and
# other systems unpack as part of the file name.
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$archive = Join-Path $out "$name.zip"
if (Test-Path $archive) { Remove-Item -Force $archive }
$zip = [System.IO.Compression.ZipFile]::Open($archive, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in @($files) + [pscustomobject]@{ Path = (Join-Path $stage 'SHA256SUMS'); Name = 'SHA256SUMS' }) {
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $file.Path, $file.Name, [System.IO.Compression.CompressionLevel]::Optimal)
    }
}
finally {
    $zip.Dispose()
}
$hash = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Lines "$archive.sha256" @("$hash  $name.zip")
Remove-Item -Recurse -Force $stage
Write-Host "libvmaf-fast $Version ($($commit.Substring(0, 8))): $archive"
Write-Host "SHA-256 $hash"
$sums | ForEach-Object { Write-Host "  $_" }
