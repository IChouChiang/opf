"""Test GCNN model with unified data loader."""

import sys
from pathlib import Path

# Add src to path for unified data loader
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import legacy model
sys.path.insert(0, str(Path(__file__).parent.parent / "legacy" / "gcnn_opf_01"))

import torch
from deep_opf.data.datamodule import OPFDataModule
from model_01 import GCNN_OPF_01


def test_gcnn_forward():
    """Test GCNN forward pass with unified data loader."""
    print("Testing GCNN forward pass with unified data loader...")

    # Create data module with graph features
    dm = OPFDataModule(
        data_dir="legacy/gcnn_opf_01/data_matlab_npz",
        batch_size=4,
        feature_type="graph",
        normalize=True,
        num_workers=0,
        feature_params={"feature_iterations": 3},
    )

    # Setup the data module
    dm.setup(stage="fit")

    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Batch keys: {list(batch.keys())}")
    print(f"e_0_k shape: {batch['e_0_k'].shape}")
    print(f"f_0_k shape: {batch['f_0_k'].shape}")

    # Create model config
    n_bus = dm.n_bus
    n_gen = dm.n_gen
    gen_bus_map = dm.get_gen_bus_map()

    print(f"\nCreating GCNN model...")
    print(f"n_bus: {n_bus}, n_gen: {n_gen}")
    print(f"gen_bus_map shape: {gen_bus_map.shape}")

    # Create ModelConfig
    from config_model_01 import ModelConfig

    config = ModelConfig(
        n_bus=n_bus,
        n_gen=n_gen,
        feature_iterations=3,
        channels_gc_in=3,  # Should match feature_iterations
        channels_gc_out=8,
        neurons_fc=256,
        n_gc_layers=2,
        n_fc_layers=1,
        dropout=0.1,
    )

    model = GCNN_OPF_01(config)

    # Test forward pass
    print("\nTesting forward pass...")
    e_0_k = batch["e_0_k"]
    f_0_k = batch["f_0_k"]
    pd = batch["pd"]
    qd = batch["qd"]

    # Get operators for the first sample in batch (all samples have same topology in this batch)
    operators = batch["operators"]
    g_ndiag = operators["g_ndiag"][0]  # [39, 39]
    b_ndiag = operators["b_ndiag"][0]  # [39, 39]
    g_diag = operators["g_diag"][0]  # [39]
    b_diag = operators["b_diag"][0]  # [39]

    print(f"Input shapes: e_0_k={e_0_k.shape}, f_0_k={f_0_k.shape}")
    print(f"pd shape: {pd.shape}, qd shape: {qd.shape}")
    print(f"g_ndiag shape: {g_ndiag.shape}, b_ndiag shape: {b_ndiag.shape}")

    with torch.no_grad():
        gen_out, v_out = model(e_0_k, f_0_k, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)

    print(f"Output shapes: gen_out={gen_out.shape}, v_out={v_out.shape}")
    print(f"Expected PG shape: {batch['pg_label'].shape}")
    print(f"Expected VG shape: {batch['vg_label'].shape}")

    # The model outputs [batch, n_gen, 2] where last dim is [PG, VG]
    # Extract PG and VG predictions
    pg_pred = gen_out[..., 0]  # PG is first element
    vg_pred = gen_out[..., 1]  # VG is second element

    print(f"Extracted PG shape: {pg_pred.shape}")
    print(f"Extracted VG shape: {vg_pred.shape}")

    # Check if outputs match expected shapes
    assert (
        pg_pred.shape == batch["pg_label"].shape
    ), f"PG shape mismatch: {pg_pred.shape} != {batch['pg_label'].shape}"
    assert (
        vg_pred.shape == batch["vg_label"].shape
    ), f"VG shape mismatch: {vg_pred.shape} != {batch['vg_label'].shape}"

    print("\n✅ GCNN forward pass test passed!")
    return True


def test_training_step():
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("Testing training step...")
    print("=" * 60)

    # Create data module
    dm = OPFDataModule(
        data_dir="legacy/gcnn_opf_01/data_matlab_npz",
        batch_size=4,
        feature_type="graph",
        normalize=True,
        num_workers=0,
        feature_params={"feature_iterations": 3},
    )

    dm.setup(stage="fit")

    # Create model config
    n_bus = dm.n_bus
    n_gen = dm.n_gen
    gen_bus_map = dm.get_gen_bus_map()

    # Create ModelConfig
    from config_model_01 import ModelConfig

    config = ModelConfig(
        n_bus=n_bus,
        n_gen=n_gen,
        feature_iterations=3,
        channels_gc_in=3,  # Should match feature_iterations
        channels_gc_out=8,
        neurons_fc=256,
        n_gc_layers=2,
        n_fc_layers=1,
        dropout=0.1,
    )

    model = GCNN_OPF_01(config)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    # Training step
    model.train()
    optimizer.zero_grad()

    e_0_k = batch["e_0_k"]
    f_0_k = batch["f_0_k"]
    pd = batch["pd"]
    qd = batch["qd"]
    pg_label = batch["pg_label"]
    vg_label = batch["vg_label"]

    # Get operators for the first sample in batch (all samples have same topology in this batch)
    operators = batch["operators"]
    g_ndiag = operators["g_ndiag"][0]  # [39, 39]
    b_ndiag = operators["b_ndiag"][0]  # [39, 39]
    g_diag = operators["g_diag"][0]  # [39]
    b_diag = operators["b_diag"][0]  # [39]

    # Forward pass
    gen_out, v_out = model(e_0_k, f_0_k, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)

    # Extract PG and VG predictions from gen_out [batch, n_gen, 2]
    pg_pred = gen_out[..., 0]  # PG is first element
    vg_pred = gen_out[..., 1]  # VG is second element

    # Compute loss (simple MSE for testing)
    loss_pg = torch.nn.functional.mse_loss(pg_pred, pg_label)
    loss_vg = torch.nn.functional.mse_loss(vg_pred, vg_label)
    loss = loss_pg + loss_vg

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.6f}")
    print(f"Loss PG: {loss_pg.item():.6f}")
    print(f"Loss VG: {loss_vg.item():.6f}")

    print("\n✅ Training step test passed!")
    return True


def main():
    """Run all tests."""
    try:
        if not test_gcnn_forward():
            print("❌ GCNN forward test failed")
            return False

        if not test_training_step():
            print("❌ Training step test failed")
            return False

        print("\n" + "=" * 60)
        print("✅ All GCNN tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
