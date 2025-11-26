import numpy as np
from pathlib import Path
import sys

def main():
    # Paths
    REPO_ROOT = Path(__file__).parent.parent
    DATA_DIR = REPO_ROOT / "gcnn_opf_01" / "data"
    OUTPUT_DIR = Path(__file__).parent / "data"
    
    train_path = DATA_DIR / "samples_train.npz"
    output_path = OUTPUT_DIR / "topology_voltage_means.npz"
    
    print(f"Loading training data from {train_path}...")
    data = np.load(train_path)
    
    e_0_k = data['e_0_k']  # [N_SAMPLES, N_BUS, K]
    f_0_k = data['f_0_k']  # [N_SAMPLES, N_BUS, K]
    topo_id = data['topo_id']  # [N_SAMPLES]
    
    # Use the last iteration (k=-1) for the voltage proxy
    e_last = e_0_k[:, :, -1]
    f_last = f_0_k[:, :, -1]
    
    # Find unique topologies
    unique_topos = np.unique(topo_id)
    print(f"Found topologies: {unique_topos}")
    
    # Compute mean voltage profile (e and f) for each topology
    n_topos = len(unique_topos)
    n_bus = e_last.shape[1]
    
    e_means = np.zeros((n_topos, n_bus), dtype=np.float32)
    f_means = np.zeros((n_topos, n_bus), dtype=np.float32)
    
    for t_id in unique_topos:
        mask = (topo_id == t_id)
        e_means[t_id] = np.mean(e_last[mask], axis=0)
        f_means[t_id] = np.mean(f_last[mask], axis=0)
        
        # Calculate mean magnitude for checking
        v_mag_mean = np.sqrt(e_means[t_id]**2 + f_means[t_id]**2)
        print(f"Topology {t_id}: {np.sum(mask)} samples")
        print(f"  Mean e range: [{e_means[t_id].min():.4f}, {e_means[t_id].max():.4f}]")
        print(f"  Mean f range: [{f_means[t_id].min():.4f}, {f_means[t_id].max():.4f}]")
        
    # Save results
    print(f"Saving mean voltage profiles to {output_path}...")
    np.savez(output_path, e_means=e_means, f_means=f_means)
    print("Done.")

if __name__ == "__main__":
    main()
