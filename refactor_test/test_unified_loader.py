"""Test script for unified data loader.

This script tests the new unified data loader (datamodule.py and dataset.py)
with both 'flat' and 'graph' feature types.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from deep_opf.data.datamodule import OPFDataModule


def test_flat_features():
    """Test the unified data loader with 'flat' feature type (for DNN)."""
    print("Testing 'flat' feature type...")

    # Create data module with flat features
    dm = OPFDataModule(
        data_dir="legacy/gcnn_opf_01/data_matlab_npz",  # Using existing data
        batch_size=4,
        feature_type="flat",
        normalize=True,
        num_workers=0,
    )

    # Setup the data module
    dm.setup(stage="fit")

    # Get a batch from train dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Input shape: {batch['input'].shape}")
    print(f"PG label shape: {batch['pg_label'].shape}")
    print(f"VG label shape: {batch['vg_label'].shape}")
    print(f"Topo ID shape: {batch['topo_id'].shape}")

    # Check that operators are present (operators is a dict with batched tensors)
    print(f"Operators type: {type(batch['operators'])}")
    print(f"Operators keys: {list(batch['operators'].keys())}")
    if "g_ndiag" in batch["operators"]:
        print(f"g_ndiag shape: {batch['operators']['g_ndiag'].shape}")

    return True


def test_graph_features():
    """Test the unified data loader with 'graph' feature type (for GCNN)."""
    print("\nTesting 'graph' feature type...")

    # Create data module with graph features
    dm = OPFDataModule(
        data_dir="legacy/gcnn_opf_01/data_matlab_npz",  # Using existing data
        batch_size=4,
        feature_type="graph",
        normalize=True,
        num_workers=0,
        feature_params={"feature_iterations": 3},  # Use first 3 iterations
    )

    # Setup the data module
    dm.setup(stage="fit")

    # Get a batch from train dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Batch keys: {list(batch.keys())}")
    print(f"e_0_k shape: {batch['e_0_k'].shape}")
    print(f"f_0_k shape: {batch['f_0_k'].shape}")
    print(f"PG label shape: {batch['pg_label'].shape}")
    print(f"VG label shape: {batch['vg_label'].shape}")
    print(f"Topo ID shape: {batch['topo_id'].shape}")

    # Check that operators are present (operators is a dict with batched tensors)
    print(f"Operators type: {type(batch['operators'])}")
    print(f"Operators keys: {list(batch['operators'].keys())}")
    if "g_ndiag" in batch["operators"]:
        print(f"g_ndiag shape: {batch['operators']['g_ndiag'].shape}")

    return True


def test_properties():
    """Test the properties of the data module."""
    print("\nTesting data module properties...")

    dm = OPFDataModule(
        data_dir="legacy/gcnn_opf_01/data_matlab_npz",
        batch_size=4,
        feature_type="flat",
        normalize=True,
        num_workers=0,
    )

    dm.setup(stage="fit")

    print(f"Number of buses (n_bus): {dm.n_bus}")
    print(f"Number of generators (n_gen): {dm.n_gen}")

    if dm.feature_type == "flat":
        print(f"Input dimension: {dm.input_dim}")
    else:
        print(f"Feature iterations: {dm.feature_iterations}")

    # Get normalization stats
    norm_stats = dm.get_norm_stats()
    if norm_stats:
        print(f"Normalization stats keys: {list(norm_stats.keys())}")

    # Get generator bus map
    gen_bus_map = dm.get_gen_bus_map()
    print(f"Generator bus map shape: {gen_bus_map.shape}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Unified Data Loader")
    print("=" * 60)

    try:
        # Test flat features
        if not test_flat_features():
            print("❌ Flat features test failed")
            return False

        # Test graph features
        if not test_graph_features():
            print("❌ Graph features test failed")
            return False

        # Test properties
        if not test_properties():
            print("❌ Properties test failed")
            return False

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
