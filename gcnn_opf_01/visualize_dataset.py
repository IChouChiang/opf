import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path to handle imports if run from root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def visualize_dataset(data_dir):
    data_path = Path(data_dir) / "samples_train.npz"
    print(f"Loading dataset from {data_path}...")

    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    # Extract data
    # Keys usually: 'pd', 'qd', 'gen_label' (PG, VG)
    # Or 'pg_labels', 'vg_labels'

    pd = data["pd"]  # [N_samples, N_bus]
    qd = data["qd"]  # [N_samples, N_bus]

    if "gen_label" in data:
        gen_label = data["gen_label"]  # [N_samples, N_gen, 2]
        pg = gen_label[:, :, 0]
        vg = gen_label[:, :, 1]
    elif "pg_labels" in data and "vg_labels" in data:
        pg = data["pg_labels"]
        vg = data["vg_labels"]
    else:
        print(f"Error: Could not find generation labels. Keys: {list(data.keys())}")
        return

    print(f"Dataset shape: {pd.shape[0]} samples")
    print(f"Buses: {pd.shape[1]}, Generators: {pg.shape[1]}")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Dataset Visualization: {data_path.name}", fontsize=16)

    # 1. Active Demand (Pd)
    axes[0, 0].hist(pd.flatten(), bins=50, color="blue", alpha=0.7)
    axes[0, 0].set_title("Active Demand (Pd) Distribution")
    axes[0, 0].set_xlabel("p.u.")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Reactive Demand (Qd)
    axes[0, 1].hist(qd.flatten(), bins=50, color="orange", alpha=0.7)
    axes[0, 1].set_title("Reactive Demand (Qd) Distribution")
    axes[0, 1].set_xlabel("p.u.")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Active Generation (Pg)
    axes[1, 0].hist(pg.flatten(), bins=50, color="green", alpha=0.7)
    axes[1, 0].set_title("Active Generation (Pg) Distribution")
    axes[1, 0].set_xlabel("p.u.")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Voltage Setpoints (Vg)
    axes[1, 1].hist(vg.flatten(), bins=50, color="red", alpha=0.7)
    axes[1, 1].set_title("Voltage Setpoints (Vg) Distribution")
    axes[1, 1].set_xlabel("p.u.")
    axes[1, 1].grid(True, alpha=0.3)

    # Save plot
    output_path = Path(data_dir) / "dataset_viz.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

    # Statistical Summary
    print("\nStatistical Summary:")
    print(
        f"Pd: Mean={pd.mean():.4f}, Std={pd.std():.4f}, Min={pd.min():.4f}, Max={pd.max():.4f}"
    )
    print(
        f"Qd: Mean={qd.mean():.4f}, Std={qd.std():.4f}, Min={qd.min():.4f}, Max={qd.max():.4f}"
    )
    print(
        f"Pg: Mean={pg.mean():.4f}, Std={pg.std():.4f}, Min={pg.min():.4f}, Max={pg.max():.4f}"
    )
    print(
        f"Vg: Mean={vg.mean():.4f}, Std={vg.std():.4f}, Min={vg.min():.4f}, Max={vg.max():.4f}"
    )

    # Check Power Balance (Roughly)
    # Sum(Pg) should be > Sum(Pd) due to losses
    total_gen = pg.sum(axis=1)
    total_load = pd.sum(axis=1)
    losses = total_gen - total_load

    print("\nPower Balance Check (per sample):")
    print(f"Avg Total Gen: {total_gen.mean():.4f} p.u.")
    print(f"Avg Total Load: {total_load.mean():.4f} p.u.")
    print(
        f"Avg Losses: {losses.mean():.4f} p.u. ({losses.mean()/total_load.mean()*100:.2f}%)"
    )

    if losses.mean() < 0:
        print("WARNING: Average generation is less than average load! (Infeasible?)")
    elif losses.mean() > 0.1 * total_load.mean():
        print("WARNING: Losses seem very high (>10%)")
    else:
        print("Power balance looks reasonable (Gen > Load).")


if __name__ == "__main__":
    # Assuming the data is in gcnn_opf_01/data_matlab_npz
    # Use absolute path for safety
    root_dir = Path("e:/DOCUMENT/Learn_Py/opf")
    data_dir = root_dir / "gcnn_opf_01/data_matlab_npz"
    visualize_dataset(data_dir)
