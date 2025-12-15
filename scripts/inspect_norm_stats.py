import numpy as np
from pathlib import Path

# Support running from project root
script_dir = Path(__file__).parent.parent
data_path = script_dir / "legacy/gcnn_opf_01/data_matlab_npz/norm_stats.npz"

data = np.load(data_path)
print("Keys:", data.files)
for k in data.files:
    print(f"{k}: shape={data[k].shape}, dtype={data[k].dtype}")
    print(f"   Sample values: {data[k].flatten()[:5]}")
