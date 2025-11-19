"""Quick test for feature_construction_model_01.py with case6ww."""

import sys
from pathlib import Path

# Add gcnn_opf_01 to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "gcnn_opf_01"))

import torch
from sample_config_model_01 import load_case6ww_int, build_G_B_operators
from feature_construction_model_01 import construct_features_from_ppc


def main():
    print("=" * 60)
    print("Feature Construction Test (case6ww)")
    print("=" * 60)

    # Load case6ww
    ppc_int, baseMVA, bus, gen, branch, N_BUS, N_GEN, N_BRANCH = load_case6ww_int()
    print(f"\nSystem: {N_BUS} buses, {N_GEN} generators")

    # Build operators
    G, B, g_diag, b_diag, g_ndiag, b_ndiag = build_G_B_operators(ppc_int)
    print(f"Operator shapes: G={G.shape}, g_diag={g_diag.shape}")

    # Extract demands (base case)
    pd = torch.tensor(bus[:, 2] / baseMVA, dtype=torch.float32)  # PD in p.u.
    qd = torch.tensor(bus[:, 3] / baseMVA, dtype=torch.float32)  # QD in p.u.

    print(f"\nTotal demand: PD={pd.sum():.3f} p.u., QD={qd.sum():.3f} p.u.")

    # Construct features with k=8 iterations
    k = 8
    print(f"\nRunning feature construction with k={k} iterations...")

    e_0_k, f_0_k = construct_features_from_ppc(
        ppc_int=ppc_int,
        pd=pd,
        qd=qd,
        G=G,
        B=B,
        g_ndiag=g_ndiag,
        b_ndiag=b_ndiag,
        g_diag=g_diag,
        b_diag=b_diag,
        k=k,
    )

    print(f"\n✓ Feature construction complete!")
    print(f"  e_0_k shape: {e_0_k.shape}")  # Expected: [6, 8]
    print(f"  f_0_k shape: {f_0_k.shape}")  # Expected: [6, 8]

    # Check voltage magnitudes (should be ~1.0 after normalization)
    v_mag = torch.sqrt(e_0_k**2 + f_0_k**2)
    print(f"\nVoltage magnitudes (should be ~1.0 per iteration):")
    for i in range(k):
        print(
            f"  Iter {i+1}: mean={v_mag[:, i].mean():.6f}, std={v_mag[:, i].std():.6f}"
        )

    # Check initial and final features
    print(f"\nFirst iteration features (e¹, f¹):")
    print(f"  e¹: min={e_0_k[:, 0].min():.4f}, max={e_0_k[:, 0].max():.4f}")
    print(f"  f¹: min={f_0_k[:, 0].min():.4f}, max={f_0_k[:, 0].max():.4f}")

    print(f"\nLast iteration features (e⁸, f⁸):")
    print(f"  e⁸: min={e_0_k[:, -1].min():.4f}, max={e_0_k[:, -1].max():.4f}")
    print(f"  f⁸: min={f_0_k[:, -1].min():.4f}, max={f_0_k[:, -1].max():.4f}")

    print(f"\n{'='*60}")
    print("✓ Test PASSED: Features constructed successfully!")
    print("='*60}")


if __name__ == "__main__":
    main()
