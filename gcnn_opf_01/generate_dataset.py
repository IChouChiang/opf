"""
Generate training and test datasets for PG-GCNN (case6ww).

This script:
1. Precomputes topology operators for all 5 topologies
2. Generates 10,000 training samples + 2,000 test samples
3. For each sample:
   - Samples topology_id uniformly from {0,1,2,3,4}
   - Generates RES + load scenario (pd, qd)
   - Constructs features (e_0_k, f_0_k) via k-iteration
   - Solves AC-OPF to get labels (pg_labels, vg_labels)
4. Computes normalization statistics from training set
5. Saves to NPZ files

Output files (gcnn_opf_01/data/):
  - samples_train.npz (10k samples)
  - samples_test.npz (2k samples)
  - topology_operators.npz (5 topologies)
  - norm_stats.npz (z-score statistics)

Estimated runtime: ~2.5 hours (case6ww @ 0.5-1s per AC-OPF solve)
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing
from datetime import datetime

# Set device (GPU if available, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Add paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "gcnn_opf_01"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from sample_config_model_01 import (
    load_case6ww_int,
    get_res_bus_indices,
    apply_topology,
    build_G_B_operators,
    extract_gen_limits,
    SIGMA_REL_LOAD,
    LAM_WIND,
    K_WIND,
    V_CUT_IN,
    V_RATED,
    V_CUT_OUT,
    ALPHA_PV,
    BETA_PV,
    G_STC,
)
from sample_generator_model_01 import SampleGeneratorModel01
from feature_construction_model_01 import construct_features
from helpers_ac_opf import solve_ac_opf
from pyomo.environ import value

# ============================================================================
# Configuration
# ============================================================================

N_TRAIN = 10000  # Training samples
N_TEST = 2000  # Test samples
N_TOPOLOGIES = 5  # Base + 4 N-1 contingencies
K_FEATURES = 8  # Feature construction iterations
BATCH_SIZE = 10  # Mini-batch size (for reference, not used in generation)
RNG_SEED_TRAIN = 42  # Random seed for training set
RNG_SEED_TEST = 123  # Random seed for test set (different from train)
CHECKPOINT_INTERVAL = 500  # Save checkpoint every N samples
OUTPUT_DIR = Path(__file__).parent / "data"

# Solver configuration
SOLVER_TIME_LIMIT = 180  # seconds
SOLVER_MIP_GAP = 0.03
SOLVER_THREADS = multiprocessing.cpu_count()  # Use all CPU threads

# ============================================================================
# Helper Functions
# ============================================================================


def precompute_topology_operators():
    """
    Precompute physics operators for all 5 topologies.

    Returns:
        dict with keys:
            'g_ndiag': [5, N_BUS, N_BUS]
            'b_ndiag': [5, N_BUS, N_BUS]
            'g_diag': [5, N_BUS]
            'b_diag': [5, N_BUS]
            'gen_bus_map': [N_GEN]
            'N_BUS': int
            'N_GEN': int
    """
    print("\n" + "=" * 70)
    print("PRECOMPUTING TOPOLOGY OPERATORS")
    print("=" * 70)

    ppc_base, baseMVA, bus, gen, branch, N_BUS, N_GEN, N_BRANCH = load_case6ww_int()
    gen_bus_map = gen[:, 0].astype(np.int32)

    # Storage
    g_ndiag_all = np.zeros((N_TOPOLOGIES, N_BUS, N_BUS), dtype=np.float32)
    b_ndiag_all = np.zeros((N_TOPOLOGIES, N_BUS, N_BUS), dtype=np.float32)
    g_diag_all = np.zeros((N_TOPOLOGIES, N_BUS), dtype=np.float32)
    b_diag_all = np.zeros((N_TOPOLOGIES, N_BUS), dtype=np.float32)

    for topo_id in range(N_TOPOLOGIES):
        print(f"  Computing operators for topology {topo_id}...")
        ppc_topo = apply_topology(ppc_base, topo_id)
        G, B, g_diag, b_diag, g_ndiag, b_ndiag = build_G_B_operators(ppc_topo)

        # Convert torch tensors to numpy arrays
        # Use .detach().cpu() to move to CPU, then convert via list for NumPy 2.x compatibility
        g_ndiag_all[topo_id] = np.array(g_ndiag.detach().cpu().tolist(), dtype=np.float32)
        b_ndiag_all[topo_id] = np.array(b_ndiag.detach().cpu().tolist(), dtype=np.float32)
        g_diag_all[topo_id] = np.array(g_diag.detach().cpu().tolist(), dtype=np.float32)
        b_diag_all[topo_id] = np.array(b_diag.detach().cpu().tolist(), dtype=np.float32)

    operators = {
        "g_ndiag": g_ndiag_all,
        "b_ndiag": b_ndiag_all,
        "g_diag": g_diag_all,
        "b_diag": b_diag_all,
        "gen_bus_map": gen_bus_map,
        "N_BUS": np.int32(N_BUS),
        "N_GEN": np.int32(N_GEN),
    }

    print(f"  ✓ Operators computed for all {N_TOPOLOGIES} topologies")
    print(f"    Shapes: g_ndiag={g_ndiag_all.shape}, g_diag={g_diag_all.shape}")
    return operators, ppc_base, baseMVA, bus, gen, N_BUS, N_GEN


def extract_opf_labels(instance, gen_bus_map, n_gen):
    """Extract PG and VG labels from solved AC-OPF instance."""
    pg_labels = np.zeros(n_gen, dtype=np.float32)
    vg_labels = np.zeros(n_gen, dtype=np.float32)

    for g in range(n_gen):
        pg_labels[g] = value(instance.PG[g])

        bus_idx = int(gen_bus_map[g])
        e_val = value(instance.e[bus_idx])
        f_val = value(instance.f[bus_idx])
        vg_labels[g] = np.sqrt(e_val**2 + f_val**2)

    return pg_labels, vg_labels


def generate_sample(
    sample_idx,
    gen_obj,
    ppc_base,
    baseMVA,
    operators,
    gen_limits,
    rng,
    N_BUS,
    N_GEN,
):
    """
    Generate one training/test sample.

    Returns:
        dict with keys: e_0_k, f_0_k, pd, qd, topo_id, pg_labels, vg_labels
        OR None if AC-OPF fails
    """
    # 1. Sample topology uniformly
    topo_id = rng.choice(N_TOPOLOGIES)

    # 2. Generate scenario (pd, qd)
    scenario = gen_obj.sample_scenario(topology_id=topo_id)
    pd_np = scenario["pd"]
    qd_np = scenario["qd"]

    # Convert to torch tensors and move to device
    pd_torch = torch.tensor(pd_np, dtype=torch.float32).to(device)
    qd_torch = torch.tensor(qd_np, dtype=torch.float32).to(device)

    # 3. Get topology operators and move to device
    g_ndiag = torch.tensor(operators["g_ndiag"][topo_id], dtype=torch.float32).to(device)
    b_ndiag = torch.tensor(operators["b_ndiag"][topo_id], dtype=torch.float32).to(device)
    g_diag = torch.tensor(operators["g_diag"][topo_id], dtype=torch.float32).to(device)
    b_diag = torch.tensor(operators["b_diag"][topo_id], dtype=torch.float32).to(device)

    # Also need full G, B for feature construction (reconstruct on device)
    G_full = g_ndiag + torch.diag(g_diag)
    B_full = b_ndiag + torch.diag(b_diag)

    # 4. Construct features (k-iteration voltage estimation)
    # Move gen_limits to device as well
    gen_limits_device = {
        "gen_bus_indices": gen_limits["gen_bus_indices"],  # NumPy array, stays on CPU
        "PG_min": gen_limits["PG_min"].to(device),
        "PG_max": gen_limits["PG_max"].to(device),
        "QG_min": gen_limits["QG_min"].to(device),
        "QG_max": gen_limits["QG_max"].to(device),
        "gen_mask": gen_limits["gen_mask"].to(device),
    }
    
    e_0_k, f_0_k = construct_features(
        pd=pd_torch,
        qd=qd_torch,
        G=G_full,
        B=B_full,
        g_ndiag=g_ndiag,
        b_ndiag=b_ndiag,
        g_diag=g_diag,
        b_diag=b_diag,
        gen_bus_indices=gen_limits_device["gen_bus_indices"],
        PG_min=gen_limits_device["PG_min"],
        PG_max=gen_limits_device["PG_max"],
        QG_min=gen_limits_device["QG_min"],
        QG_max=gen_limits_device["QG_max"],
        gen_mask=gen_limits_device["gen_mask"],
        k=K_FEATURES,
    )

    # 5. Build scenario ppc for AC-OPF
    ppc_topo = apply_topology(ppc_base, topo_id)
    ppc_scenario = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ppc_topo.items()}
    ppc_scenario["bus"][:, 2] = pd_np * baseMVA  # PD in MW
    ppc_scenario["bus"][:, 3] = qd_np * baseMVA  # QD in MVAr

    # 6. Solve AC-OPF
    try:
        instance, result = solve_ac_opf(
            ppc_scenario,
            verbose=False,
            time_limit=SOLVER_TIME_LIMIT,
            mip_gap=SOLVER_MIP_GAP,
            threads=SOLVER_THREADS,
        )

        # Check if optimal
        term_cond = str(result.solver.termination_condition)
        if term_cond != "optimal":
            return None  # Skip non-optimal solutions

        # 7. Extract labels
        pg_labels, vg_labels = extract_opf_labels(
            instance, operators["gen_bus_map"], N_GEN
        )

        # 8. Return sample dict (convert tensors back to numpy via detach().cpu().tolist())
        return {
            "e_0_k": np.array(e_0_k.detach().cpu().tolist(), dtype=np.float32),  # [N_BUS, K_FEATURES]
            "f_0_k": np.array(f_0_k.detach().cpu().tolist(), dtype=np.float32),
            "pd": pd_np,  # [N_BUS]
            "qd": qd_np,
            "topo_id": np.int32(topo_id),
            "pg_labels": pg_labels,  # [N_GEN]
            "vg_labels": vg_labels,
        }

    except Exception as e:
        print(f"\n  WARNING: Sample {sample_idx} failed with error: {e}")
        return None


def generate_dataset(n_samples, rng_seed, split_name, operators, ppc_base, baseMVA, bus, gen, N_BUS, N_GEN):
    """
    Generate a dataset (training or test).

    Returns:
        dict with arrays ready for np.savez()
    """
    print("\n" + "=" * 70)
    print(f"GENERATING {split_name.upper()} SET ({n_samples} samples)")
    print("=" * 70)

    # Initialize RNG and scenario generator
    rng = np.random.default_rng(rng_seed)
    wind_idx, pv_idx = get_res_bus_indices(ppc_base)
    PD_base = bus[:, 2] / baseMVA
    QD_base = bus[:, 3] / baseMVA

    gen_obj = SampleGeneratorModel01(
        PD_base=PD_base,
        QD_base=QD_base,
        penetration_target=0.30,
        res_bus_idx_wind=wind_idx,
        res_bus_idx_pv=pv_idx,
        rng_seed=rng_seed,
        sigma_rel=SIGMA_REL_LOAD,
        lam_wind=LAM_WIND,
        k_wind=K_WIND,
        v_cut_in=V_CUT_IN,
        v_rated=V_RATED,
        v_cut_out=V_CUT_OUT,
        alpha_pv=ALPHA_PV,
        beta_pv=BETA_PV,
        g_stc=G_STC,
        allow_negative_pd=False,
    )

    # Extract generator limits (constant across samples)
    gen_bus_indices, PG_min, PG_max, QG_min, QG_max, gen_mask = extract_gen_limits(
        ppc_base
    )
    gen_limits = {
        "gen_bus_indices": gen_bus_indices,
        "PG_min": PG_min,
        "PG_max": PG_max,
        "QG_min": QG_min,
        "QG_max": QG_max,
        "gen_mask": gen_mask,
    }

    # Storage for samples
    e_0_k_list = []
    f_0_k_list = []
    pd_list = []
    qd_list = []
    topo_id_list = []
    pg_labels_list = []
    vg_labels_list = []

    # Progress tracking
    n_success = 0
    n_failed = 0
    max_consecutive_failures = 100  # Stop if too many consecutive failures
    consecutive_failures = 0
    pbar = tqdm(total=n_samples, desc=f"{split_name} samples", unit="sample")

    sample_idx = 0
    while n_success < n_samples:
        # Safety check: avoid infinite loop
        if consecutive_failures >= max_consecutive_failures:
            print(f"\n  ERROR: {consecutive_failures} consecutive failures. Stopping generation.")
            print(f"  This might indicate a systematic issue with the scenarios or solver.")
            break
            
        sample = generate_sample(
            sample_idx,
            gen_obj,
            ppc_base,
            baseMVA,
            operators,
            gen_limits,
            rng,
            N_BUS,
            N_GEN,
        )

        if sample is not None:
            # Store sample
            e_0_k_list.append(sample["e_0_k"])
            f_0_k_list.append(sample["f_0_k"])
            pd_list.append(sample["pd"])
            qd_list.append(sample["qd"])
            topo_id_list.append(sample["topo_id"])
            pg_labels_list.append(sample["pg_labels"])
            vg_labels_list.append(sample["vg_labels"])

            n_success += 1
            pbar.update(1)

            # Checkpoint
            if n_success % CHECKPOINT_INTERVAL == 0:
                print(
                    f"\n  Checkpoint: {n_success}/{n_samples} samples generated "
                    f"({n_failed} failed, {n_failed/(n_success+n_failed)*100:.1f}% failure rate)"
                )
        else:
            n_failed += 1

        sample_idx += 1

    pbar.close()

    print(f"\n  ✓ Generation complete: {n_success} success, {n_failed} failed")
    print(f"    Success rate: {n_success/(n_success+n_failed)*100:.1f}%")

    # Convert lists to arrays
    dataset = {
        "e_0_k": np.stack(e_0_k_list, axis=0),  # [n_samples, N_BUS, K_FEATURES]
        "f_0_k": np.stack(f_0_k_list, axis=0),
        "pd": np.stack(pd_list, axis=0),  # [n_samples, N_BUS]
        "qd": np.stack(qd_list, axis=0),
        "topo_id": np.array(topo_id_list, dtype=np.int32),  # [n_samples]
        "pg_labels": np.stack(pg_labels_list, axis=0),  # [n_samples, N_GEN]
        "vg_labels": np.stack(vg_labels_list, axis=0),
    }

    # Print shapes
    print(f"\n  Dataset shapes:")
    for key, val in dataset.items():
        print(f"    {key:12s}: {val.shape}")

    return dataset


def compute_normalization_stats(train_data):
    """
    Compute z-score normalization statistics from training set.

    Returns:
        dict with mean/std for pd, qd, pg_labels, vg_labels
    """
    print("\n" + "=" * 70)
    print("COMPUTING NORMALIZATION STATISTICS")
    print("=" * 70)

    stats = {
        "pd_mean": np.float32(train_data["pd"].mean()),
        "pd_std": np.float32(train_data["pd"].std()),
        "qd_mean": np.float32(train_data["qd"].mean()),
        "qd_std": np.float32(train_data["qd"].std()),
        "pg_mean": np.float32(train_data["pg_labels"].mean()),
        "pg_std": np.float32(train_data["pg_labels"].std()),
        "vg_mean": np.float32(train_data["vg_labels"].mean()),
        "vg_std": np.float32(train_data["vg_labels"].std()),
    }

    print(f"  pd: mean={stats['pd_mean']:.4f}, std={stats['pd_std']:.4f}")
    print(f"  qd: mean={stats['qd_mean']:.4f}, std={stats['qd_std']:.4f}")
    print(f"  pg: mean={stats['pg_mean']:.4f}, std={stats['pg_std']:.4f}")
    print(f"  vg: mean={stats['vg_mean']:.4f}, std={stats['vg_std']:.4f}")

    return stats


# ============================================================================
# Main Script
# ============================================================================


def main():
    print("\n" + "=" * 70)
    print("GCNN-OPF DATASET GENERATION (case6ww)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Training samples: {N_TRAIN}")
    print(f"  Test samples: {N_TEST}")
    print(f"  Topologies: {N_TOPOLOGIES}")
    print(f"  Feature iterations: {K_FEATURES}")
    print(f"  RNG seeds: train={RNG_SEED_TRAIN}, test={RNG_SEED_TEST}")
    print(f"  Solver: time_limit={SOLVER_TIME_LIMIT}s, mip_gap={SOLVER_MIP_GAP}, threads={SOLVER_THREADS}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")

    # Step 1: Precompute topology operators
    operators, ppc_base, baseMVA, bus, gen, N_BUS, N_GEN = (
        precompute_topology_operators()
    )

    # Step 2: Generate training set
    train_data = generate_dataset(
        N_TRAIN, RNG_SEED_TRAIN, "train", operators, ppc_base, baseMVA, bus, gen, N_BUS, N_GEN
    )

    # Step 3: Generate test set
    test_data = generate_dataset(
        N_TEST, RNG_SEED_TEST, "test", operators, ppc_base, baseMVA, bus, gen, N_BUS, N_GEN
    )

    # Step 4: Compute normalization statistics
    norm_stats = compute_normalization_stats(train_data)

    # Step 5: Save all data
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)

    train_path = OUTPUT_DIR / "samples_train.npz"
    test_path = OUTPUT_DIR / "samples_test.npz"
    operators_path = OUTPUT_DIR / "topology_operators.npz"
    stats_path = OUTPUT_DIR / "norm_stats.npz"

    print(f"  Saving {train_path}...")
    np.savez_compressed(train_path, **train_data)
    print(f"    Size: {train_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"  Saving {test_path}...")
    np.savez_compressed(test_path, **test_data)
    print(f"    Size: {test_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"  Saving {operators_path}...")
    np.savez_compressed(operators_path, **operators)
    print(f"    Size: {operators_path.stat().st_size / 1024:.2f} KB")

    print(f"  Saving {stats_path}...")
    np.savez(stats_path, **norm_stats)
    print(f"    Size: {stats_path.stat().st_size} bytes")

    # Summary
    print("\n" + "=" * 70)
    print("✓ DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nGenerated files:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {operators_path}")
    print(f"  {stats_path}")
    print(f"\nTotal storage: {(train_path.stat().st_size + test_path.stat().st_size) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
