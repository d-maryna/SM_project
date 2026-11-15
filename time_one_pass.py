import os, sys, time
sys.path.append(os.getcwd())
import torch
import orca_predict
orca_predict.load_resources(models=["1M"], use_cuda=False)
from orca_predict import *
torch.set_num_threads(2)

seq = hg38.get_encoding_from_coords("chr21", 30_000_000, 31_000_000)[None, :, :]
x = torch.FloatTensor(seq).transpose(1, 2)

with torch.no_grad():
    for name, model in [("h1esc", h1esc_1m), ("hff", hff_1m)]:
        model(x)  # прогрів
        times = []
        for _ in range(3):
            t = time.time()
            model(x)
            times.append(time.time() - t)
        print(name, "секунд на один прогін:", [round(t, 1) for t in times])