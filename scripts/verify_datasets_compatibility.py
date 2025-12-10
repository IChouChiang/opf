
import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

def verify_model_01():
    print("\n--- Verifying Model 01 (GCNN) Dataset ---")
    try:
        from gcnn_opf_01.dataset import OPFDataset
        
        data_dir = Path("gcnn_opf_01/data_matlab_npz")
        dataset = OPFDataset(
            data_path = data_dir / "samples_train.npz",
            topo_operators_path = data_dir / "topology_operators.npz",
            norm_stats_path = data_dir / "norm_stats.npz",
            normalize=True,
            split="train",
            feature_iterations=10
        )
        
        print(f"Successfully initialized OPFDataset (Model 01). Samples: {len(dataset)}")
        
        # Try getting an item
        sample = dataset[0]
        print(f"Sample 0 keys: {list(sample.keys())}")
        print(f"Feature shape (e_0_k): {sample['e_0_k'].shape}")
        
        # Try dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print("DataLoader batch successful.")
        return True
        
    except Exception as e:
        print(f"FAILED Model 01: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_model_03():
    print("\n--- Verifying Model 03 (MLP) Dataset ---")
    try:
        from dnn_opf_03.dataset_03 import OPFDataset03
        
        data_dir = Path("gcnn_opf_01/data_matlab_npz")
        dataset = OPFDataset03(
            data_path = data_dir / "samples_train.npz",
            topo_operators_path = data_dir / "topology_operators.npz",
            norm_stats_path = data_dir / "norm_stats.npz",
            normalize=True,
            split="train"
        )
        
        print(f"Successfully initialized OPFDataset03 (Model 03). Samples: {len(dataset)}")
        
        # Try getting an item
        sample = dataset[0]
        print(f"Sample 0 keys: {list(sample.keys())}")
        print(f"Input shape: {sample['input'].shape}") 
        # Expected: 2*39 + 2*(39*39) = 78 + 3042 = 3120?
        # N=39. 2*N + 2*N^2 = 78 + 2*1521 = 3120.
        
        # Try dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print("DataLoader batch successful.")
        return True
        
    except Exception as e:
        print(f"FAILED Model 03: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    ok1 = verify_model_01()
    ok3 = verify_model_03()
    
    if ok1 and ok3:
        print("\n✅ SUCCESS: Both datasets verified compatible!")
    else:
        print("\n❌ FAILED: Compatibility issues found.")
