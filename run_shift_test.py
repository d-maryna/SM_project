import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys; sys.path.append(os.getcwd())
import numpy as np, torch
import orca_predict
orca_predict.load_resources(models=["1M"], use_cuda=False)
from orca_predict import *
torch.set_num_threads(2)

chrom, base = "chr21", 35_000_000
os.makedirs("control_shift", exist_ok=True)
for shift in [0, 100_000, 200_000]:      # зсув у bp
    s = base + shift
    seq = hg38.get_encoding_from_coords(chrom, s, s + 1_000_000)[None, :, :]
    x = torch.FloatTensor(seq).transpose(1, 2)
    with torch.no_grad():
        pred = h1esc_1m(x).squeeze().detach().cpu().numpy()
    np.save(f"control_shift/{chrom}_{s}_h1esc.npy", pred)
    print(shift, pred.shape)