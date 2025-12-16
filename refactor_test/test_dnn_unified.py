"""Test DNN model with unified data loader."""

import sys
from pathlib import Path

# Add src to path for unified data loader
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import legacy model
sys.path.insert(0, str(Path(__file__).parent.parent / "legacy" / "dnn_opf_03"))

import torch
from deep_opf.data.datamodule import OPFDataModule
from model_03 import AdmittanceDNN
from config_03 import ModelConfig


def test_dnn_forward():
    """Test DNN forward pass with unified data loader."""
    print("Testing DNN forward pass with unified data loader...")

    # Create data module with flat features
    dm = OPFDataModule(
        data_dir="legacy/gcnn_opf_01/data_matlab_npz",
        batch_size=4,
        feature_type="flat",
        normalize=True,
        num_workers=0,
    )

    # Setup the data module
    dm.setup(stage="fit")

    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Input shape: {batch['input'].shape}")
    print(f"PG label shape: {batch['pg_label'].shape}")
    print(f"VG label shape: {batch['vg_label'].shape}")

    # Create model config
    n_bus = dm.n_bus
    n_gen = dm.n_gen
    input_dim = dm.input_dim

    print(f"\nCreating DNN model...")
    print(f"n_bus: {n_bus}, n_gen: {n_gen}")
    print(f"input_dim: {input_dim}")

    # Create ModelConfig
    config = ModelConfig(
        n_bus=n_bus,
        n_gen=n_gen,
        input_dim=input_dim,
        hidden_dim=128,
        n_hidden_layers=3,
        hidden_layers=None,  # Use default hidden_dim * n_hidden_layers
    )

    model = AdmittanceDNN(config)

    # Test forward pass
    print("\nTesting forward pass...")
    x = batch["input"]

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        gen_out, v_out = model(x)

    print(f"Output shapes: gen_out={gen_out.shape}, v_out={v_out.shape}")
    print(f"Expected PG shape: {batch['pg_label'].shape}")
    print(f"Expected VG shape: {batch['vg_label'].shape}")

    # Extract PG and VG predictions from gen_out [batch, n_gen, 2]
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

    print("\n✅ DNN forward pass test passed!")
    return True


def test_training_step():
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("Testing DNN training step...")
    print("=" * 60)

    # Create data module
    dm = OPFDataModule(
        data_dir="legacy/gcnn_opf_01/data_matlab_npz",
        batch_size=4,
        feature_type="flat",
        normalize=True,
        num_workers=0,
    )

    dm.setup(stage="fit")

    # Create model config
    n_bus = dm.n_bus
    n_gen = dm.n_gen
    input_dim = dm.input_dim

    config = ModelConfig(
        n_bus=n_bus,
        n_gen=n_gen,
        input_dim=input_dim,
        hidden_dim=128,
        n_hidden_layers=3,
        hidden_layers=None,
    )

    model = AdmittanceDNN(config)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    # Training step
    model.train()
    optimizer.zero_grad()

    x = batch["input"]
    pg_label = batch["pg_label"]
    vg_label = batch["vg_label"]

    # Forward pass
    gen_out, v_out = model(x)

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

    print("\n✅ DNN training step test passed!")
    return True


def main():
    """Run all tests."""
    try:
        if not test_dnn_forward():
            print("❌ DNN forward test failed")
            return False

        if not test_training_step():
            print("❌ DNN training step test failed")
            return False

        print("\n" + "=" * 60)
        print("✅ All DNN tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
