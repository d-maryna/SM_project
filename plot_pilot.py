"""
Maps around the pilot variant: ref, alt, alt-ref, alt-ref for a same-position null and a random-position null,
plus the insulation profile. Reads only files saved by pilot_variant.py, no model needed.
Run (WSL, from the folder with pilot_variant.py):
    python plot_pilot.py --dir pilot_out2 --tag chr21_40686217
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BIN, HALF, W_INS = 4000, 500_000, 10

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="pilot_out2")
ap.add_argument("--tag", default="chr21_40686217")
ap.add_argument("--r", type=int, default=50, help="zoom radius in bins (50 bins = 200 kb)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

ref = np.load(os.path.join(a.dir, f"{a.tag}_ref.npy"))
alt = np.load(os.path.join(a.dir, f"{a.tag}_alt.npy"))
n, b0 = ref.shape[0], HALF // BIN
lo, hi = max(0, b0 - a.r), min(n, b0 + a.r + 1)


def zoom(M, b):
    """Square of +-r bins around bin b."""
    s, e = max(0, b - a.r), min(n, b + a.r + 1)
    return M[s:e, s:e]


def insulation(M):
    out = np.full(n, np.nan)
    for i in range(W_INS, n - W_INS):
        out[i] = np.nanmean(M[i - W_INS : i, i + 1 : i + 1 + W_INS])
    return out


diff = zoom(alt - ref, b0)
vmax = np.nanpercentile(np.abs(diff), 99.5)  # one colour scale for all difference panels
panels = [("ref (H1-ESC)", zoom(ref, b0), 1.5), ("alt", zoom(alt, b0), 1.5), ("alt - ref (variant)", diff, vmax)]

for prefix, label in [("same", "alt - ref (same-position null)"), ("loc", "alt - ref (random-position null)")]:
    files = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}_{prefix}*_o*.npy")))
    if files:  # first null of each kind
        o = int(re.search(r"_o(\d+)\.npy$", files[0]).group(1))
        panels.append((f"{label}\noffset {o - HALF:+d} bp", zoom(np.load(files[0]) - ref, o // BIN), vmax))

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
axes = axes.ravel()
for ax, (title, M, vm) in zip(axes, panels):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("bins of 4 kb")
    plt.colorbar(im, ax=ax, fraction=0.046)

# the panel after the maps: insulation score of ref and alt around the variant
ax = axes[min(len(panels), len(axes) - 1)]
x = (np.arange(lo, hi) - b0) * BIN / 1000
ax.plot(x, insulation(ref)[lo:hi], label="ref")
ax.plot(x, insulation(alt)[lo:hi], label="alt", linestyle="--")
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_title(f"insulation score (w={W_INS} bins), 4 kb bins", fontsize=10)
ax.set_xlabel("kb from the variant")
ax.legend()
for ax in axes[len(panels) + 1 :]:
    ax.axis("off")

fig.suptitle(f"{a.tag}: zoom +-{a.r * BIN // 1000} kb; difference panels share one colour scale (+-{vmax:.3f})", fontsize=11)
fig.tight_layout()
out = a.out or os.path.join(a.dir, f"{a.tag}_maps.png")
fig.savefig(out, dpi=130)
print("saved", out)