import numpy as np, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in sorted(glob.glob("control_hg38/*.npy")):
    m = np.load(f)
    print(f, m.shape, "min/max:", float(m.min()), float(m.max()))
    plt.figure(figsize=(4, 4))
    plt.imshow(m, cmap="RdBu_r")
    plt.colorbar(shrink=0.7)
    plt.title(f.split("/")[-1].replace(".npy", ""), fontsize=7)
    plt.savefig(f.replace(".npy", ".png"), dpi=120, bbox_inches="tight")
    plt.close()