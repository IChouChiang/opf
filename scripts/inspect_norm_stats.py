import numpy as np
import torch

data = np.load("gcnn_opf_01/data_matlab_npz/norm_stats.npz")
print("Keys:", data.files)
for k in data.files:
    print(f"{k}: shape={data[k].shape}, dtype={data[k].dtype}")
    print(f"   Sample values: {data[k].flatten()[:5]}")
