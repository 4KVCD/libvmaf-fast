"""VIF's statistics: the fast path (vif_stats.slang's fast_gain) against the exact path, pixel by
pixel, on a build made for it -- vif_stats.slang's VIF_STATS_VERIFY set to 73 and vmaf_vulkan.cpp's
kSlots raised by 3 (sums 73, 74, 75: disagreements, refusals, gained pixels; every gained pixel's
final values are then the exact path's). VMAF_FAST_DIST names that build. For each pair of sources
(this machine's test videos; CASESET=gain for pairs made to have large gains), frames decoded by
FFmpeg at WxH, counted over all four scales.
    verify_vif_fast_stats.py WxH FRAMES [DEVICE]"""
import subprocess, sys
import numpy as np
sys.path.insert(0, "C:/Users/Brian/Documents/libvmaf-fast-intel/fast/python")
sys.path.insert(0, "C:/Users/Brian/OneDrive/Code/VMAF calc")
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_fast import vulkan
W, H = (int(x) for x in sys.argv[1].split("x")); N = int(sys.argv[2])
SRC = "C:/The.Beekeeper.2024.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.HYBRID.REMUX-FraMeSToR.mkv"
ENC = "E:/Video encodings/"
CASES = [  # (name, ref, ref seconds, dis, dis seconds, bits, ref filter, dis filter)
    ("Beekeeper vs AV1 grain 30", SRC, 600, ENC + "The.Beekeeper AV1 3000kbps preset 5 film grain 30.mkv", 600, 10, "", ""),
    ("Beekeeper vs x265 2000k", SRC, 1500, ENC + "The.Beekeeper 4k x265 slower 2000kbps.mkv", 1500, 10, "", ""),
    ("Beekeeper vs x264 4000k 8-bit", SRC, 3000, ENC + "The.Beekeeper 4k x264 veryslow 4000kbps.mkv", 3000, 8, "", ""),
    ("HoneyBee vs H264 CRF20", ENC + "HoneyBee reference 4K 120fps 10-bit.mkv", 2, ENC + "HoneyBee H264 slow CRF20.mkv", 2, 10, "", ""),
    ("Conan QP30 vs QP32", ENC + "Conan 26 QP 30 faster.mkv", 400, ENC + "Conan 26 QP 32 medium.mkv", 400, 10, "", ""),
    ("Beekeeper vs itself, noise", SRC, 2000, SRC, 2000, 10, "", "noise=alls=20:allf=t"),
    ("Beekeeper vs itself, blur", SRC, 4000, SRC, 4000, 10, "", "gblur=sigma=3"),
    ("Beekeeper vs other scene", SRC, 900, SRC, 5000, 10, "", ""),
    ("noise vs noise", "noise", 0, "noise", 1, 10, "", ""),
    ("8-bit grain vs sharpen", SRC, 3500, SRC, 3500, 8, "noise=alls=8:allf=t", "unsharp=7:7:2.5"),
    ("contrast vs gamma", SRC, 2500, SRC, 2500, 10, "eq=contrast=1.8", "eq=gamma=0.5"),
]
import os
if os.environ.get("CASESET") == "gain":
    CASES = [  # large gains: a nearly flat reference against a contrasty distorted frame
        ("flat ref vs contrast x2", SRC, 1200, SRC, 1200, 10, "eq=contrast=0.04", "eq=contrast=2"),
        ("flat ref vs contrast x1", SRC, 2600, SRC, 2600, 10, "eq=contrast=0.1", ""),
        ("flat ref vs noise", SRC, 3300, SRC, 3300, 10, "eq=contrast=0.03", "noise=alls=40:allf=t"),
        ("flat 8-bit vs sharpen", SRC, 4100, SRC, 4100, 8, "eq=contrast=0.06", "unsharp=13:13:5"),
        ("flat ref vs inverted", SRC, 1800, SRC, 1800, 10, "eq=contrast=0.05", "negate"),
        ("dim ref vs bright", SRC, 700, SRC, 700, 10, "eq=contrast=0.2:brightness=-0.3", "eq=contrast=1.5:brightness=0.2"),
    ]
def frames(path, seconds, bits, vf, n):
    fmt = "yuv420p10le" if bits > 8 else "yuv420p"
    if path == "noise":
        rng = np.random.default_rng(seconds)
        top = 1023 if bits > 8 else 255
        out = []
        for i in range(n):
            y = rng.integers(0, top + 1, size=(H, W)).astype("<u2" if bits > 8 else "u1")
            y[:, : W // 3] = (y[:, : W // 3] // 64) * 64  # flat-ish stripes, gradients
            y[: H // 4] = np.linspace(0, top, W).astype(y.dtype)[None, :]
            c = np.full(((H // 2) * (W // 2) * 2,), top // 2, dtype=y.dtype)
            out.append(bytearray(y.tobytes() + c.tobytes()))
        return out
    chain = f"scale={W}:{H}:flags=bicubic" + (f",{vf}" if vf else "")
    data = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-ss", str(seconds), "-i", path, "-map", "0:V:0",
                           "-frames:v", str(n), "-vf", chain, "-pix_fmt", fmt, "-f", "rawvideo", "pipe:1"],
                          capture_output=True, check=True).stdout
    size = W * H * 3 // 2 * (2 if bits > 8 else 1)
    return [bytearray(data[a:a + size]) for a in range(0, len(data) - size + 1, size)]
total = np.zeros(3, dtype=np.int64)
for name, ref_path, ref_s, dis_path, dis_s, bits, ref_vf, dis_vf in CASES:
    ref = frames(ref_path, ref_s, bits, ref_vf, N)
    dis = frames(dis_path, dis_s, bits, dis_vf, N)
    n = min(len(ref), len(dis))
    s = vulkan.VulkanScorer(W, H, bits, {}, device=int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    try:
        for i in range(n):
            s.add(ref[i], dis[i])
        s.features()
        counts = np.zeros(3, dtype=np.int64)
        for f in range(n):
            sums = s.sums(f)
            counts += [int(sums[75]), int(sums[74]), int(sums[73])]
    finally:
        s.close()
    total += counts
    print(f"{name:32s} {W}x{H} {bits:2d}-bit {n:3d} frames: {counts[0]:>12,} gained pixels, "
          f"{counts[1]:>9,} refused ({counts[1] / max(counts[0], 1):.2e}), {counts[2]} DISAGREE")
print(f"TOTAL: {total[0]:,} gained pixels, {total[1]:,} refused ({total[1] / max(total[0], 1):.2e}), "
      f"{total[2]} disagreements -> {'ALL AGREE' if total[2] == 0 else 'DISAGREEMENTS'}")
