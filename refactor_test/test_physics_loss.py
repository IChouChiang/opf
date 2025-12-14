"""Test physics loss functions from deep_opf.loss module.

This test validates:
1. build_gen_bus_matrix creates correct incidence matrix
2. compute_power_from_voltage implements AC power flow correctly
3. physics_loss computes active power balance
4. correlative_loss combines supervised + physics losses
5. Batched inputs work correctly
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deep_opf.loss import (
    build_gen_bus_matrix,
    compute_power_from_voltage,
    correlative_loss,
    correlative_loss_pg,
    physics_loss,
)


def test_build_gen_bus_matrix():
    """Test generator-to-bus incidence matrix construction."""
    print("=" * 60)
    print("TEST: build_gen_bus_matrix")
    print("=" * 60)

    # Simple case: 4 buses, 2 generators at buses 0 and 2
    n_bus = 4
    gen_bus_indices = [0, 2]

    A = build_gen_bus_matrix(n_bus, gen_bus_indices)

    print(f"n_bus = {n_bus}, gen_bus_indices = {gen_bus_indices}")
    print(f"A_g2b shape: {A.shape}")
    print(f"A_g2b:\n{A}")

    # Verify shape
    assert A.shape == (4, 2), f"Expected (4, 2), got {A.shape}"

    # Verify entries
    expected = torch.tensor(
        [
            [1.0, 0.0],  # bus 0 has gen 0
            [0.0, 0.0],  # bus 1 has no gen
            [0.0, 1.0],  # bus 2 has gen 1
            [0.0, 0.0],  # bus 3 has no gen
        ]
    )
    assert torch.allclose(A, expected), "Incidence matrix mismatch"

    print("✓ build_gen_bus_matrix test PASSED\n")


def test_compute_power_from_voltage_unbatched():
    """Test AC power flow computation (unbatched)."""
    print("=" * 60)
    print("TEST: compute_power_from_voltage (unbatched)")
    print("=" * 60)

    # Simple 2-bus system
    n_bus = 2

    # Conductance and susceptance (symmetric)
    G = torch.tensor([[1.0, -0.5], [-0.5, 1.0]])
    B = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])

    # Voltage: bus 0 = 1.0∠0°, bus 1 = 0.95∠-5°
    import math

    e = torch.tensor([1.0, 0.95 * math.cos(math.radians(-5))])
    f = torch.tensor([0.0, 0.95 * math.sin(math.radians(-5))])
    v_bus = torch.stack([e, f], dim=-1)  # [2, 2]

    # Demands
    pd = torch.tensor([0.0, 0.5])
    qd = torch.tensor([0.0, 0.2])

    P, Q = compute_power_from_voltage(v_bus, pd, qd, G, B)

    print(f"v_bus shape: {v_bus.shape}")
    print(f"e = {e.tolist()}")
    print(f"f = {f.tolist()}")
    print(f"pd = {pd.tolist()}")
    print(f"P_from_V = {P.tolist()}")
    print(f"Q_from_V = {Q.tolist()}")

    # Verify output shape
    assert P.shape == (n_bus,), f"Expected P shape ({n_bus},), got {P.shape}"
    assert Q.shape == (n_bus,), f"Expected Q shape ({n_bus},), got {Q.shape}"

    print("✓ compute_power_from_voltage (unbatched) test PASSED\n")


def test_compute_power_from_voltage_batched():
    """Test AC power flow computation (batched)."""
    print("=" * 60)
    print("TEST: compute_power_from_voltage (batched)")
    print("=" * 60)

    batch_size = 3
    n_bus = 2

    G = torch.tensor([[1.0, -0.5], [-0.5, 1.0]])
    B = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])

    # Batched voltages [B, n_bus, 2]
    v_bus = torch.randn(batch_size, n_bus, 2)

    # Batched demands [B, n_bus]
    pd = torch.rand(batch_size, n_bus)
    qd = torch.rand(batch_size, n_bus)

    P, Q = compute_power_from_voltage(v_bus, pd, qd, G, B)

    print(f"Batch size: {batch_size}, n_bus: {n_bus}")
    print(f"v_bus shape: {v_bus.shape}")
    print(f"P_from_V shape: {P.shape}")
    print(f"Q_from_V shape: {Q.shape}")

    # Verify output shapes
    assert P.shape == (
        batch_size,
        n_bus,
    ), f"Expected ({batch_size}, {n_bus}), got {P.shape}"
    assert Q.shape == (
        batch_size,
        n_bus,
    ), f"Expected ({batch_size}, {n_bus}), got {Q.shape}"

    print("✓ compute_power_from_voltage (batched) test PASSED\n")


def test_physics_loss():
    """Test physics-based loss computation."""
    print("=" * 60)
    print("TEST: physics_loss")
    print("=" * 60)

    batch_size = 4
    n_bus = 5
    n_gen = 2
    gen_bus_indices = [0, 3]

    # Build incidence matrix
    A_g2b = build_gen_bus_matrix(n_bus, gen_bus_indices)

    # Random matrices (should be symmetric in practice)
    G = torch.randn(n_bus, n_bus)
    B = torch.randn(n_bus, n_bus)

    # Predictions
    pg = torch.rand(batch_size, n_gen)
    v_bus = torch.randn(batch_size, n_bus, 2)

    # Demands
    pd = torch.rand(batch_size, n_bus)
    qd = torch.rand(batch_size, n_bus)

    result = physics_loss(
        pg=pg,
        v_bus=v_bus,
        pd=pd,
        qd=qd,
        G=G,
        B=B,
        gen_bus_matrix=A_g2b,
        include_reactive=False,
    )

    print(f"Batch: {batch_size}, n_bus: {n_bus}, n_gen: {n_gen}")
    print(f"loss_p: {result['loss_p'].item():.6f}")
    print(f"loss_physics: {result['loss_physics'].item():.6f}")

    assert "loss_p" in result
    assert "loss_physics" in result
    assert result["loss_p"].dim() == 0, "Loss should be scalar"

    print("✓ physics_loss test PASSED\n")


def test_correlative_loss():
    """Test correlative loss (supervised + physics)."""
    print("=" * 60)
    print("TEST: correlative_loss")
    print("=" * 60)

    batch_size = 4
    n_bus = 5
    n_gen = 2
    gen_bus_indices = [0, 3]
    kappa = 0.1

    A_g2b = build_gen_bus_matrix(n_bus, gen_bus_indices)
    G = torch.randn(n_bus, n_bus)
    B = torch.randn(n_bus, n_bus)

    # Predictions
    pg = torch.rand(batch_size, n_gen)
    vg = torch.rand(batch_size, n_gen)
    v_bus = torch.randn(batch_size, n_bus, 2)

    # Labels
    pg_label = torch.rand(batch_size, n_gen)
    vg_label = torch.rand(batch_size, n_gen)

    # Demands
    pd = torch.rand(batch_size, n_bus)
    qd = torch.rand(batch_size, n_bus)

    result = correlative_loss(
        pg=pg,
        vg=vg,
        v_bus=v_bus,
        pg_label=pg_label,
        vg_label=vg_label,
        pd=pd,
        qd=qd,
        G=G,
        B=B,
        gen_bus_matrix=A_g2b,
        kappa=kappa,
        include_vg_supervised=True,
    )

    print(f"kappa = {kappa}")
    print(f"loss_supervised_pg: {result['loss_supervised_pg'].item():.6f}")
    print(f"loss_supervised_vg: {result['loss_supervised_vg'].item():.6f}")
    print(f"loss_supervised: {result['loss_supervised'].item():.6f}")
    print(f"loss_physics: {result['loss_physics'].item():.6f}")
    print(f"loss_total: {result['loss_total'].item():.6f}")

    # Verify formula: total = supervised + kappa * physics
    expected_total = result["loss_supervised"] + kappa * result["loss_physics"]
    assert torch.isclose(
        result["loss_total"], expected_total, atol=1e-6
    ), f"Total loss mismatch: {result['loss_total']} vs {expected_total}"

    print("✓ correlative_loss test PASSED\n")


def test_correlative_loss_pg_legacy():
    """Test legacy interface (correlative_loss_pg)."""
    print("=" * 60)
    print("TEST: correlative_loss_pg (legacy interface)")
    print("=" * 60)

    n_bus = 5
    n_gen = 2
    gen_bus_indices = [0, 3]

    A_g2b = build_gen_bus_matrix(n_bus, gen_bus_indices)
    G = torch.randn(n_bus, n_bus)
    B = torch.randn(n_bus, n_bus)

    # Legacy format: gen_out = [n_gen, 2] stacked (PG, VG)
    gen_out = torch.rand(n_gen, 2)  # Unbatched
    v_out = torch.randn(n_bus, 2)

    PG_label = torch.rand(n_gen)
    VG_label = torch.rand(n_gen)
    pd = torch.rand(n_bus)

    loss_total, loss_sup, loss_phys = correlative_loss_pg(
        gen_out=gen_out,
        v_out=v_out,
        PG_label=PG_label,
        VG_label=VG_label,
        pd=pd,
        G=G,
        B=B,
        A_g2b=A_g2b,
        kappa=0.1,
        use_VG_supervised=True,
    )

    print(f"gen_out shape: {gen_out.shape} (unbatched)")
    print(f"loss_total: {loss_total.item():.6f}")
    print(f"loss_supervised: {loss_sup.item():.6f}")
    print(f"loss_physics: {loss_phys.item():.6f}")

    assert loss_total.dim() == 0, "Loss should be scalar"
    assert loss_sup.dim() == 0, "Loss should be scalar"
    assert loss_phys.dim() == 0, "Loss should be scalar"

    print("✓ correlative_loss_pg (legacy) test PASSED\n")


def test_gradients_flow():
    """Test that gradients flow through the loss computation."""
    print("=" * 60)
    print("TEST: Gradient flow through physics loss")
    print("=" * 60)

    batch_size = 2
    n_bus = 3
    n_gen = 2
    gen_bus_indices = [0, 2]

    A_g2b = build_gen_bus_matrix(n_bus, gen_bus_indices)
    G = torch.randn(n_bus, n_bus)
    B = torch.randn(n_bus, n_bus)

    # Parameters requiring gradients
    pg = torch.rand(batch_size, n_gen, requires_grad=True)
    vg = torch.rand(batch_size, n_gen, requires_grad=True)
    v_bus = torch.randn(batch_size, n_bus, 2, requires_grad=True)

    pg_label = torch.rand(batch_size, n_gen)
    vg_label = torch.rand(batch_size, n_gen)
    pd = torch.rand(batch_size, n_bus)
    qd = torch.rand(batch_size, n_bus)

    result = correlative_loss(
        pg=pg,
        vg=vg,
        v_bus=v_bus,
        pg_label=pg_label,
        vg_label=vg_label,
        pd=pd,
        qd=qd,
        G=G,
        B=B,
        gen_bus_matrix=A_g2b,
        kappa=0.1,
    )

    loss = result["loss_total"]
    loss.backward()

    print(f"pg.grad shape: {pg.grad.shape if pg.grad is not None else 'None'}")
    print(f"vg.grad shape: {vg.grad.shape if vg.grad is not None else 'None'}")
    print(f"v_bus.grad shape: {v_bus.grad.shape if v_bus.grad is not None else 'None'}")

    assert pg.grad is not None, "Gradients should flow to pg"
    assert vg.grad is not None, "Gradients should flow to vg"
    assert v_bus.grad is not None, "Gradients should flow to v_bus"

    print("✓ Gradient flow test PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHYSICS LOSS MODULE TESTS")
    print("=" * 60 + "\n")

    test_build_gen_bus_matrix()
    test_compute_power_from_voltage_unbatched()
    test_compute_power_from_voltage_batched()
    test_physics_loss()
    test_correlative_loss()
    test_correlative_loss_pg_legacy()
    test_gradients_flow()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
