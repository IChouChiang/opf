
import numpy as np
import os

def check_data():
    base_dir = 'gcnn_opf_01/data_matlab_npz'
    
    # 1. Check Operators
    op_path = os.path.join(base_dir, 'topology_operators.npz')
    ops = np.load(op_path)
    print(f"\n--- Topology Operators ---")
    print(f"N_BUS: {ops['N_BUS']}, N_GEN: {ops['N_GEN']}")
    print(f"G_ndiag shape: {ops['g_ndiag'].shape}")
    print(f"gen_bus_map: {ops['gen_bus_map']}")
    
    # 2. Check Train Data
    train_path = os.path.join(base_dir, 'samples_train.npz')
    train = np.load(train_path)
    print(f"\n--- Training Data ---")
    print(f"Samples: {train['pd'].shape[0]}")
    print(f"Features (e_0_k) shape: {train['e_0_k'].shape}")
    print(f"Labels (pg_labels) shape: {train['pg_labels'].shape}")
    print(f"Topology IDs: {np.unique(train['topo_id'])}")
    
    # 3. Check Test Data
    test_path = os.path.join(base_dir, 'samples_test.npz')
    test = np.load(test_path)
    print(f"\n--- Test Data ---")
    print(f"Samples: {test['pd'].shape[0]}")
    print(f"Topology IDs: {np.unique(test['topo_id'])}")
    
    # Range check
    print(f"\n--- Sanity Checks ---")
    print(f"PD range: {train['pd'].min():.4f} to {train['pd'].max():.4f}")
    print(f"PG range: {train['pg_labels'].min():.4f} to {train['pg_labels'].max():.4f}")
    
    if np.isnan(train['e_0_k']).any():
        print("!! WARNING: NaNs found in features !!")
    else:
        print("Features clean (no NaNs).")

if __name__ == "__main__":
    check_data()
