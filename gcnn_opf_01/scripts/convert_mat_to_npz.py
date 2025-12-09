import numpy as np
import scipy.io as sio
import os


def convert_mat_to_npz(mat_dir, out_dir):
    """
    Convert MATLAB generated .mat datasets to Python .npz format.
    Handles 1-based indexing conversion and type casting.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    print(f"Converting data from {mat_dir} to {out_dir}...")

    # 1. Convert Topology Operators
    op_path = os.path.join(mat_dir, "topology_operators.mat")
    if os.path.exists(op_path):
        print("  Processing topology_operators.mat...")
        ops = sio.loadmat(op_path)

        # Convert gen_bus_map to 0-based
        gen_bus_map = ops["gen_bus_map"].squeeze().astype(np.int32) - 1

        np.savez_compressed(
            os.path.join(out_dir, "topology_operators.npz"),
            g_ndiag=ops["g_ndiag"].astype(np.float32),
            b_ndiag=ops["b_ndiag"].astype(np.float32),
            g_diag=ops["g_diag"].astype(np.float32),
            b_diag=ops["b_diag"].astype(np.float32),
            gen_bus_map=gen_bus_map,
            N_BUS=ops["N_BUS"].item(),
            N_GEN=ops["N_GEN"].item(),
        )
    else:
        print("  Warning: topology_operators.mat not found!")

    # 2. Convert Normalization Stats
    stat_path = os.path.join(mat_dir, "norm_stats.mat")
    if os.path.exists(stat_path):
        print("  Processing norm_stats.mat...")
        stats = sio.loadmat(stat_path)
        np.savez(
            os.path.join(out_dir, "norm_stats.npz"),
            pd_mean=stats["pd_mean"].item(),
            pd_std=stats["pd_std"].item(),
            qd_mean=stats["qd_mean"].item(),
            qd_std=stats["qd_std"].item(),
            pg_mean=stats["pg_mean"].item(),
            pg_std=stats["pg_std"].item(),
            vg_mean=stats["vg_mean"].item(),
            vg_std=stats["vg_std"].item(),
        )
    else:
        print("  Warning: norm_stats.mat not found!")

    # 3. Convert Datasets (Train/Test/Unseen)
    for split in ["train", "test", "unseen"]:
        fname = f"samples_{split}.mat"
        fpath = os.path.join(mat_dir, fname)

        if not os.path.exists(fpath):
            print(f"  Warning: {fname} not found!")
            continue

        print(f"  Processing {fname}...")
        data = sio.loadmat(fpath)

        np.savez_compressed(
            os.path.join(out_dir, f"samples_{split}.npz"),
            e_0_k=data["e_0_k"].astype(np.float32),
            f_0_k=data["f_0_k"].astype(np.float32),
            pd=data["pd"].astype(np.float32),
            qd=data["qd"].astype(np.float32),
            topo_id=data["topo_id"].squeeze().astype(np.int32),
            pg_labels=data["pg_labels"].astype(np.float32),
            vg_labels=data["vg_labels"].astype(np.float32),
        )

    # 4. Convert Unseen Topology Operators
    op_unseen_path = os.path.join(mat_dir, "topology_operators_unseen.mat")
    if os.path.exists(op_unseen_path):
        print("  Processing topology_operators_unseen.mat...")
        ops = sio.loadmat(op_unseen_path)

        # Convert gen_bus_map to 0-based
        gen_bus_map = ops["gen_bus_map"].squeeze().astype(np.int32) - 1

        np.savez_compressed(
            os.path.join(out_dir, "topology_operators_unseen.npz"),
            g_ndiag=ops["g_ndiag"].astype(np.float32),
            b_ndiag=ops["b_ndiag"].astype(np.float32),
            g_diag=ops["g_diag"].astype(np.float32),
            b_diag=ops["b_diag"].astype(np.float32),
            gen_bus_map=gen_bus_map,
            N_BUS=ops["N_BUS"].item(),
            N_GEN=ops["N_GEN"].item(),
        )

    print("Conversion complete.")


if __name__ == "__main__":
    # Assumes run from project root
    MAT_DIR = "gcnn_opf_01/data_matlab"
    OUT_DIR = "gcnn_opf_01/data_matlab_npz"
    convert_mat_to_npz(MAT_DIR, OUT_DIR)
