"""Test GCNN model forward pass shapes."""

import sys
from pathlib import Path

import torch
import numpy as np

# Add gcnn_opf_01 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gcnn_opf_01"))

from model_01 import GCNN_OPF_01
from config_model_01 import ModelConfig


def test_model_forward_shapes():
    """Test that model outputs correct shapes for single sample and batch."""

    config = ModelConfig()
    model = GCNN_OPF_01()
    model.eval()

    N_BUS = config.n_bus
    N_GEN = config.n_gen
    k = config.channels_gc_in

    print("=" * 60)
    print("GCNN Model Forward Pass Shape Test")
    print("=" * 60)
    print(f"Configuration: N_BUS={N_BUS}, N_GEN={N_GEN}, k={k}")

    # Create topology operators (shared across all samples)
    g_ndiag = torch.randn(N_BUS, N_BUS)
    b_ndiag = torch.randn(N_BUS, N_BUS)
    g_diag = torch.randn(N_BUS)
    b_diag = torch.randn(N_BUS)

    # Create sample inputs (single sample, batch dim = 1)
    e_0_k = torch.randn(1, N_BUS, k)
    f_0_k = torch.randn(1, N_BUS, k)
    pd = torch.randn(1, N_BUS)
    qd = torch.randn(1, N_BUS)

    print("\n--- Single Sample Test ---")
    print(f"Input shapes:")
    print(f"  e_0_k: {e_0_k.shape}")
    print(f"  f_0_k: {f_0_k.shape}")
    print(f"  pd: {pd.shape}")
    print(f"  qd: {qd.shape}")

    with torch.no_grad():
        gen_out, v_out = model(e_0_k, f_0_k, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)

    print(f"\nOutput shapes:")
    print(f"  gen_out: {gen_out.shape} (expected: [1, {N_GEN}, 2])")
    print(f"  v_out: {v_out.shape} (expected: [1, {N_BUS}, 2])")

    assert gen_out.shape == (
        1,
        N_GEN,
        2,
    ), f"gen_out shape mismatch: {gen_out.shape} != (1, {N_GEN}, 2)"
    assert v_out.shape == (
        1,
        N_BUS,
        2,
    ), f"v_out shape mismatch: {v_out.shape} != (1, {N_BUS}, 2)"
    print("[OK] Single sample shapes correct")

    # Test batch
    batch_size = 16
    e_0_k_batch = torch.randn(batch_size, N_BUS, k)
    f_0_k_batch = torch.randn(batch_size, N_BUS, k)
    pd_batch = torch.randn(batch_size, N_BUS)
    qd_batch = torch.randn(batch_size, N_BUS)
    # Topology operators are shared across all samples in batch (not batched)
    # Reuse same operators from single sample test

    print(f"\n--- Batch Test (batch_size={batch_size}) ---")
    print(f"Input shapes:")
    print(f"  e_0_k: {e_0_k_batch.shape}")
    print(f"  pd: {pd_batch.shape}")

    with torch.no_grad():
        gen_out_batch, v_out_batch = model(
            e_0_k_batch,
            f_0_k_batch,
            pd_batch,
            qd_batch,
            g_ndiag,  # Same topology operators for all samples
            b_ndiag,
            g_diag,
            b_diag,
        )

    print(f"\nOutput shapes:")
    print(f"  gen_out: {gen_out_batch.shape} (expected: [{batch_size}, {N_GEN}, 2])")
    print(f"  v_out: {v_out_batch.shape} (expected: [{batch_size}, {N_BUS}, 2])")

    assert gen_out_batch.shape == (
        batch_size,
        N_GEN,
        2,
    ), f"gen_out batch shape mismatch: {gen_out_batch.shape} != ({batch_size}, {N_GEN}, 2)"
    assert v_out_batch.shape == (
        batch_size,
        N_BUS,
        2,
    ), f"v_out batch shape mismatch: {v_out_batch.shape} != ({batch_size}, {N_BUS}, 2)"
    print("[OK] Batch shapes correct")

    # Test output ranges (tanh activation)
    print(f"\n--- Output Range Test ---")
    print(f"gen_out range: [{gen_out_batch.min():.3f}, {gen_out_batch.max():.3f}]")
    print(f"v_out range: [{v_out_batch.min():.3f}, {v_out_batch.max():.3f}]")

    # Tanh outputs should be in [-1, 1]
    assert (
        gen_out_batch.min() >= -1.1 and gen_out_batch.max() <= 1.1
    ), "gen_out range issue (should be ~[-1,1])"
    assert (
        v_out_batch.min() >= -1.1 and v_out_batch.max() <= 1.1
    ), "v_out range issue (should be ~[-1,1])"
    print("[OK] Output ranges reasonable (tanh activation)")

    print("\n" + "=" * 60)
    print("[OK] ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_model_forward_shapes()
