"""
Generate UNSEEN test dataset for PG-GCNN (case6ww) with 3 specific N-1 topologies.

This script:
1. Precomputes topology operators for the 3 unseen topologies.
2. Generates 400 samples for each topology (Total 1200).
3. Uses multiprocessing to speed up generation.
4. Saves to gcnn_opf_01/data_unseen/samples_test.npz

Topologies (removed lines):
- Case 0: Line 3-5
- Case 1: Line 1-5
- Case 2: Line 2-4
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing
from datetime import datetime
from functools import partial

# Set device (GPU if available, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Add paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "gcnn_opf_01"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import the NEW config for unseen topologies
import topo_vary_sample_config_model_01 as config

# Import shared modules
from sample_config_model_01 import (
    load_case6ww_int,
    get_res_bus_indices,
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

SAMPLES_PER_TOPO = 400
N_TOPOLOGIES = len(config.topology_pairs)  # Should be 3
TOTAL_SAMPLES = SAMPLES_PER_TOPO * N_TOPOLOGIES
K_FEATURES = 10
OUTPUT_DIR = Path(__file__).parent / "data_unseen"

# Solver configuration
SOLVER_TIME_LIMIT = 180
SOLVER_MIP_GAP = 0.03
# For parallel workers, we usually want 1 thread per solver instance to avoid oversubscription
SOLVER_THREADS = 1
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ============================================================================
# Helper Functions
# ============================================================================


def apply_unseen_topology(ppc_base, topo_idx):
    """Apply the specific N-1 topology from config."""
    ppc_topo = ppc_base.copy()
    # Deep copy branch array
    ppc_topo["branch"] = ppc_base["branch"].copy()

    removed_pairs = config.topology_pairs[topo_idx]

    # Find branch indices to remove
    # Note: config pairs are 1-based (from, to)
    branch = ppc_topo["branch"]
    removed_indices = []

    for f, t in removed_pairs:
        f_int, t_int = f - 1, t - 1
        # Find matching row
        match = np.where(
            ((branch[:, 0] == f_int) & (branch[:, 1] == t_int))
            | ((branch[:, 0] == t_int) & (branch[:, 1] == f_int))
        )[0]

        if len(match) > 0:
            removed_indices.append(match[0])

    if removed_indices:
        ppc_topo["branch"] = np.delete(ppc_topo["branch"], removed_indices, axis=0)

    return ppc_topo


def precompute_topology_operators():
    """Precompute physics operators for the 3 unseen topologies."""
    print("\n" + "=" * 70)
    print("PRECOMPUTING TOPOLOGY OPERATORS (UNSEEN)")
    print("=" * 70)

    ppc_base = config.get_base_case()
    # Extract dimensions from base case
    bus = ppc_base["bus"]
    gen = ppc_base["gen"]
    N_BUS = bus.shape[0]
    N_GEN = gen.shape[0]
    baseMVA = ppc_base["baseMVA"]

    gen_bus_map = gen[:, 0].astype(np.int32)

    # Storage
    g_ndiag_all = np.zeros((N_TOPOLOGIES, N_BUS, N_BUS), dtype=np.float32)
    b_ndiag_all = np.zeros((N_TOPOLOGIES, N_BUS, N_BUS), dtype=np.float32)
    g_diag_all = np.zeros((N_TOPOLOGIES, N_BUS), dtype=np.float32)
    b_diag_all = np.zeros((N_TOPOLOGIES, N_BUS), dtype=np.float32)

    for topo_id in range(N_TOPOLOGIES):
        print(
            f"  Computing operators for topology {topo_id} (Removed: {config.topology_pairs[topo_id]})..."
        )
        ppc_topo = apply_unseen_topology(ppc_base, topo_id)
        G, B, g_diag, b_diag, g_ndiag, b_ndiag = build_G_B_operators(ppc_topo)

        # Convert torch tensors to numpy arrays
        g_ndiag_all[topo_id] = np.array(
            g_ndiag.detach().cpu().tolist(), dtype=np.float32
        )
        b_ndiag_all[topo_id] = np.array(
            b_ndiag.detach().cpu().tolist(), dtype=np.float32
        )
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


def worker_generate_chunk(
    topo_id,
    n_samples,
    seed_offset,
    ppc_base,
    baseMVA,
    operators_dict,
    gen_limits_dict,
    N_BUS,
    N_GEN,
):
    """
    Worker function to generate a chunk of samples for a specific topology.
    """
    # Re-initialize RNG
    rng = np.random.default_rng(seed_offset)

    # Re-initialize Generator
    wind_idx, pv_idx = get_res_bus_indices(ppc_base)
    PD_base = ppc_base["bus"][:, 2] / baseMVA
    QD_base = ppc_base["bus"][:, 3] / baseMVA

    gen_obj = SampleGeneratorModel01(
        PD_base=PD_base,
        QD_base=QD_base,
        penetration_target=0.30,
        res_bus_idx_wind=wind_idx,
        res_bus_idx_pv=pv_idx,
        rng_seed=seed_offset,
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

    # Prepare topology operators (convert back to torch if needed, or keep numpy)
    # For construct_features, we need torch tensors
    # We'll create them once per worker to save overhead
    g_ndiag = torch.tensor(operators_dict["g_ndiag"][topo_id], dtype=torch.float32)
    b_ndiag = torch.tensor(operators_dict["b_ndiag"][topo_id], dtype=torch.float32)
    g_diag = torch.tensor(operators_dict["g_diag"][topo_id], dtype=torch.float32)
    b_diag = torch.tensor(operators_dict["b_diag"][topo_id], dtype=torch.float32)

    G_full = g_ndiag + torch.diag(g_diag)
    B_full = b_ndiag + torch.diag(b_diag)

    # Prepare gen limits tensors
    PG_min = gen_limits_dict["PG_min"]
    PG_max = gen_limits_dict["PG_max"]
    QG_min = gen_limits_dict["QG_min"]
    QG_max = gen_limits_dict["QG_max"]
    gen_mask = gen_limits_dict["gen_mask"]
    gen_bus_indices = gen_limits_dict["gen_bus_indices"]

    # Prepare ppc for this topology
    ppc_topo = apply_unseen_topology(ppc_base, topo_id)

    samples = []
    attempts = 0
    max_attempts = n_samples * 20  # Allow 20x attempts

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        # 1. Generate scenario
        scenario = gen_obj.sample_scenario(topology_id=topo_id)
        pd_np = scenario["pd"]
        qd_np = scenario["qd"]

        pd_torch = torch.tensor(pd_np, dtype=torch.float32)
        qd_torch = torch.tensor(qd_np, dtype=torch.float32)

        # 2. Construct features
        e_0_k, f_0_k = construct_features(
            pd=pd_torch,
            qd=qd_torch,
            G=G_full,
            B=B_full,
            g_ndiag=g_ndiag,
            b_ndiag=b_ndiag,
            g_diag=g_diag,
            b_diag=b_diag,
            gen_bus_indices=gen_bus_indices,
            PG_min=PG_min,
            PG_max=PG_max,
            QG_min=QG_min,
            QG_max=QG_max,
            gen_mask=gen_mask,
            k=K_FEATURES,
        )

        # 3. Solve AC-OPF
        ppc_solve = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in ppc_topo.items()
        }
        ppc_solve["bus"][:, 2] = pd_np * baseMVA
        ppc_solve["bus"][:, 3] = qd_np * baseMVA

        try:
            instance, result = solve_ac_opf(
                ppc_solve,
                verbose=False,
                time_limit=SOLVER_TIME_LIMIT,
                mip_gap=SOLVER_MIP_GAP,
                threads=SOLVER_THREADS,
            )

            if str(result.solver.termination_condition) == "optimal":
                pg_labels, vg_labels = extract_opf_labels(
                    instance, operators_dict["gen_bus_map"], N_GEN
                )

                samples.append(
                    {
                        "e_0_k": np.array(e_0_k.detach().numpy(), dtype=np.float32),
                        "f_0_k": np.array(f_0_k.detach().numpy(), dtype=np.float32),
                        "pd": pd_np,
                        "qd": qd_np,
                        "topo_id": np.int32(topo_id),
                        "pg_labels": pg_labels,
                        "vg_labels": vg_labels,
                    }
                )
        except Exception:
            pass

    return samples


def main():
    print("\n" + "=" * 70)
    print("UNSEEN DATASET GENERATION (case6ww)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Topologies: {N_TOPOLOGIES}")
    print(f"Samples per topology: {SAMPLES_PER_TOPO}")
    print(f"Total samples: {TOTAL_SAMPLES}")
    print(f"Workers: {NUM_WORKERS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Precompute operators
    operators, ppc_base, baseMVA, bus, gen, N_BUS, N_GEN = (
        precompute_topology_operators()
    )

    # 2. Prepare generator limits (shared)
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

    # 3. Prepare tasks
    tasks = []
    chunk_size = 20  # Smaller chunks for better load balancing

    for topo_id in range(N_TOPOLOGIES):
        n_chunks = SAMPLES_PER_TOPO // chunk_size
        for i in range(n_chunks):
            seed = 10000 + topo_id * 1000 + i  # Unique seed
            tasks.append(
                (
                    topo_id,
                    chunk_size,
                    seed,
                    ppc_base,
                    baseMVA,
                    operators,
                    gen_limits,
                    N_BUS,
                    N_GEN,
                )
            )

    print(f"\nProcessing {len(tasks)} tasks with {NUM_WORKERS} workers...")

    all_samples = []

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # Use starmap_async to allow progress bar
        results = []
        for result in tqdm(
            pool.starmap(worker_generate_chunk, tasks),
            total=len(tasks),
            desc="Generating",
            unit="chunk",
        ):
            results.extend(result)

    print(f"\nTotal valid samples generated: {len(results)}")

    # 4. Save Data
    if not results:
        print("ERROR: No samples generated!")
        return

    # Stack data
    dataset = {
        "e_0_k": np.stack([s["e_0_k"] for s in results], axis=0),
        "f_0_k": np.stack([s["f_0_k"] for s in results], axis=0),
        "pd": np.stack([s["pd"] for s in results], axis=0),
        "qd": np.stack([s["qd"] for s in results], axis=0),
        "topo_id": np.stack([s["topo_id"] for s in results], axis=0),
        "pg_labels": np.stack([s["pg_labels"] for s in results], axis=0),
        "vg_labels": np.stack([s["vg_labels"] for s in results], axis=0),
    }

    save_path = OUTPUT_DIR / "samples_test.npz"
    ops_path = OUTPUT_DIR / "topology_operators.npz"

    print(f"Saving to {save_path}...")
    np.savez_compressed(save_path, **dataset)

    print(f"Saving to {ops_path}...")
    np.savez_compressed(ops_path, **operators)

    print("\nDone!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
