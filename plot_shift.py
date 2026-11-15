import numpy as np, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sorted(glob.glob("control_shift/*.npy"))
fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 5))
for ax, f in zip(axes, files):
    m = np.load(f)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_title(f.split("/")[-1].replace(".npy", ""), fontsize=8)
fig.colorbar(im, ax=axes, shrink=0.7)
plt.savefig("control_shift/shift_comparison.png", dpi=120, bbox_inches="tight")