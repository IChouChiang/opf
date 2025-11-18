import sys
from pathlib import Path

# Make gcnn_opf_01 importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "gcnn_opf_01"))

import numpy as np  # noqa: E402
from sample_config_model_01 import (  # noqa: E402
    load_case39_int,
    get_res_bus_indices,
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
from sample_generator_model_01 import SampleGeneratorModel01  # noqa: E402


def main():
    print("=== Sample Generator Test ===\n")

    # Load case39
    ppc_int, baseMVA, bus, gen, branch, N_BUS, N_GEN, N_BRANCH = load_case39_int()
    print(f"Loaded case39: {N_BUS} buses, {N_GEN} gens, {N_BRANCH} branches")

    # Get RES bus indices
    wind_idx, pv_idx = get_res_bus_indices(ppc_int)
    print(f"Wind buses (internal indices): {wind_idx}")
    print(f"PV buses (internal indices): {pv_idx}")

    # Extract base PD/QD in p.u.
    PD_base = bus[:, 2] / baseMVA  # MW -> p.u.
    QD_base = bus[:, 3] / baseMVA  # MVAr -> p.u.

    print(
        f"\nBase total load: PD={PD_base.sum():.2f} p.u., QD={QD_base.sum():.2f} p.u."
    )

    # Create generator with 50.9% RES penetration and allow negative PD
    gen_obj = SampleGeneratorModel01(
        PD_base=PD_base,
        QD_base=QD_base,
        penetration_target=0.509,
        res_bus_idx_wind=wind_idx,
        res_bus_idx_pv=pv_idx,
        rng_seed=42,
        sigma_rel=SIGMA_REL_LOAD,
        lam_wind=LAM_WIND,
        k_wind=K_WIND,
        v_cut_in=V_CUT_IN,
        v_rated=V_RATED,
        v_cut_out=V_CUT_OUT,
        alpha_pv=ALPHA_PV,
        beta_pv=BETA_PV,
        g_stc=G_STC,
        allow_negative_pd=True,
    )

    print(f"\nGenerator initialized with:")
    print(f"  - {len(wind_idx)} wind buses")
    print(f"  - {len(pv_idx)} PV buses")
    print(f"  - Target penetration: 50.9%")

    # Generate 3 samples for topology 0 (base case)
    print("\n=== Generating 3 samples for topology 0 ===")
    for i in range(3):
        sample = gen_obj.sample_scenario(topology_id=0)

        pd = sample["pd"]
        qd = sample["qd"]
        P_res = sample["P_res_avail"]
        pd_raw = sample["pd_raw"]

        total_load = pd.sum()
        total_res_avail = P_res.sum()
        # Method 1 (injected penetration): fraction of raw load offset by RES injection
        injected_res = pd_raw.sum() - pd.sum()
        penetration_injected = injected_res / (pd_raw.sum() + 1e-8)

        neg_mask = pd < 0.0
        n_neg = neg_mask.sum()
        min_pd = pd.min()
        neg_total = pd[neg_mask].sum() if n_neg > 0 else 0.0

        print(f"\nSample {i+1}:")
        print(f"  Total load (after RES): {total_load:.3f} p.u.")
        print(f"  Total RES available (raw): {total_res_avail:.3f} p.u.")
        print(f"  Injected RES (scaled): {injected_res:.3f} p.u.")
        print(f"  Penetration (method 1): {penetration_injected*100:.1f}%")
        print(f"  Wind contribution: {P_res[wind_idx].sum():.3f} p.u.")
        print(f"  PV contribution: {P_res[pv_idx].sum():.3f} p.u.")
        print(
            f"  Negative PD buses: {n_neg} (min pd = {min_pd:.3f} p.u., sum negative = {neg_total:.3f} p.u.)"
        )

    print("\n=== Test PASSED ===")


if __name__ == "__main__":
    main()
