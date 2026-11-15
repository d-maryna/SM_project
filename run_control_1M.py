import sys, os
sys.path.append(os.getcwd())
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import orca_predict
orca_predict.load_resources(models=["1M"], use_cuda=False)
from orca_predict import *
torch.set_num_threads(2)
loci = [
    ("chr21", 30_000_000),
    ("chr21", 35_000_000),
    ("chr21", 40_000_000),
]
os.makedirs("control_hg38", exist_ok=True)
for chrom, start in loci:
    end = start + 1_000_000
    seq = hg38.get_encoding_from_coords(chrom, start, end)[None, :, :]   # (1, 1_000_000, 4)
    # скільки позицій без нуклеотиду (N)
    n_frac = float((seq.sum(axis=2) == 0).mean())
    print(f"{chrom}:{start}-{end}  shape={seq.shape}  частка N={n_frac:.4f}")
    x = torch.FloatTensor(seq).transpose(1, 2)
    with torch.no_grad():
        for name, model in [("h1esc", h1esc_1m), ("hff", hff_1m)]:
            pred = model(x).squeeze().detach().cpu().numpy()
            print(f"   {name}: output shape = {pred.shape}")
            np.save(f"control_hg38/{chrom}_{start}_{end}_{name}.npy", pred)