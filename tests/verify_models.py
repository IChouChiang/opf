"""Test script to verify DNN and GCNN model implementations.

This script:
1. Instantiates AdmittanceDNN with dummy arguments
2. Tests forward pass with random input tensor
3. Instantiates GCNN with dummy arguments
4. Tests forward pass with dummy batch dictionary
5. Verifies output shapes match between models
6. Prints success message and exits with code 0

Usage:
    python tests/verify_models.py
"""

import sys
from pathlib import Path

# Add src to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from deep_opf.models.dnn import AdmittanceDNN
from deep_opf.models.gcnn import GCNN


def test_dnn():
    """Test AdmittanceDNN model."""
    print("Testing AdmittanceDNN...")

    # Dummy parameters for a small test case
    n_bus = 6
    n_gen = 3
    input_dim = 2 * n_bus + 2 * n_bus * n_bus  # 2*6 + 2*36 = 84

    # Instantiate model
    model = AdmittanceDNN(
        input_dim=input_dim,
        hidden_dim=32,
        num_layers=2,
        n_gen=n_gen,
        n_bus=n_bus,
        dropout=0.1,
    )

    # Test 1: Batched input
    print("  Testing batched input...")
    batch_size = 2
    x = torch.randn(batch_size, input_dim)
    output = model(x)

    # Verify output keys
    expected_keys = {"pg", "vg", "v_bus"}
    actual_keys = set(output.keys())
    assert (
        actual_keys == expected_keys
    ), f"Expected keys {expected_keys}, got {actual_keys}"

    # Verify output shapes
    assert output["pg"].shape == (batch_size, n_gen), f"pg shape: {output['pg'].shape}"
    assert output["vg"].shape == (batch_size, n_gen), f"vg shape: {output['vg'].shape}"
    assert output["v_bus"].shape == (
        batch_size,
        n_bus,
        2,
    ), f"v_bus shape: {output['v_bus'].shape}"

    print(f"    ✓ Batched forward pass successful")

    # Test 2: Single sample (non-batched)
    print("  Testing single sample input...")
    x_single = torch.randn(input_dim)
    output_single = model(x_single)

    # Verify output shapes for single sample
    # DNN may return [1, n_gen] or [n_gen] depending on implementation
    # Both are acceptable for now
    pg_shape = output_single["pg"].shape
    vg_shape = output_single["vg"].shape
    v_bus_shape = output_single["v_bus"].shape

    # Accept either [n_gen] or [1, n_gen]
    assert pg_shape in [(n_gen,), (1, n_gen)], f"Single pg shape: {pg_shape}"
    assert vg_shape in [(n_gen,), (1, n_gen)], f"Single vg shape: {vg_shape}"
    # Accept either [n_bus, 2] or [1, n_bus, 2]
    assert v_bus_shape in [
        (n_bus, 2),
        (1, n_bus, 2),
    ], f"Single v_bus shape: {v_bus_shape}"

    print(f"    ✓ Single sample forward pass successful")

    print(f"  ✓ DNN tests completed")
    print(f"    - Batched pg shape: {output['pg'].shape}")
    print(f"    - Single pg shape: {output_single['pg'].shape}")

    return model, output


def test_gcnn():
    """Test GCNN model."""
    print("\nTesting GCNN...")

    # Dummy parameters matching DNN test
    n_bus = 6
    n_gen = 3
    feature_iterations = 5  # k iterations for e_0_k, f_0_k

    # Instantiate model
    model = GCNN(
        n_bus=n_bus,
        n_gen=n_gen,
        in_channels=feature_iterations,  # e_0_k and f_0_k have k channels
        hidden_channels=16,
        n_layers=2,
        fc_hidden_dim=32,
        n_fc_layers=2,
        dropout=0.1,
    )

    # Test 1: Batched input
    print("  Testing batched input...")
    batch_size = 2

    # e_0_k and f_0_k: [batch_size, n_bus, feature_iterations]
    e_0_k = torch.randn(batch_size, n_bus, feature_iterations)
    f_0_k = torch.randn(batch_size, n_bus, feature_iterations)

    # pd and qd: [batch_size, n_bus]
    pd = torch.randn(batch_size, n_bus)
    qd = torch.randn(batch_size, n_bus)

    # Operators: create dummy G/B matrices
    # Note: In real usage, these would come from topology_operators.npz
    g_ndiag = torch.randn(n_bus, n_bus)
    b_ndiag = torch.randn(n_bus, n_bus)
    g_diag = torch.randn(n_bus)
    b_diag = torch.randn(n_bus)

    # Create batch dictionary matching dataset format
    batch = {
        "e_0_k": e_0_k,
        "f_0_k": f_0_k,
        "pd": pd,
        "qd": qd,
        "operators": {
            "g_ndiag": g_ndiag,
            "b_ndiag": b_ndiag,
            "g_diag": g_diag,
            "b_diag": b_diag,
        },
    }

    # Forward pass
    output = model(batch)

    # Verify output keys
    expected_keys = {"pg", "vg", "v_bus"}
    actual_keys = set(output.keys())
    assert (
        actual_keys == expected_keys
    ), f"Expected keys {expected_keys}, got {actual_keys}"

    # Verify output shapes
    assert output["pg"].shape == (batch_size, n_gen), f"pg shape: {output['pg'].shape}"
    assert output["vg"].shape == (batch_size, n_gen), f"vg shape: {output['vg'].shape}"
    assert output["v_bus"].shape == (
        batch_size,
        n_bus,
        2,
    ), f"v_bus shape: {output['v_bus'].shape}"

    print(f"    ✓ Batched forward pass successful")

    # Test 2: Single sample (non-batched)
    print("  Testing single sample input...")
    # Single sample: remove batch dimension
    e_0_k_single = e_0_k[0]  # [n_bus, feature_iterations]
    f_0_k_single = f_0_k[0]
    pd_single = pd[0]  # [n_bus]
    qd_single = qd[0]

    batch_single = {
        "e_0_k": e_0_k_single,
        "f_0_k": f_0_k_single,
        "pd": pd_single,
        "qd": qd_single,
        "operators": {
            "g_ndiag": g_ndiag,
            "b_ndiag": b_ndiag,
            "g_diag": g_diag,
            "b_diag": b_diag,
        },
    }

    output_single = model(batch_single)

    # Verify output shapes for single sample
    assert output_single["pg"].shape == (
        n_gen,
    ), f"Single pg shape: {output_single['pg'].shape}"
    assert output_single["vg"].shape == (
        n_gen,
    ), f"Single vg shape: {output_single['vg'].shape}"
    assert output_single["v_bus"].shape == (
        n_bus,
        2,
    ), f"Single v_bus shape: {output_single['v_bus'].shape}"

    print(f"    ✓ Single sample forward pass successful")

    print(f"  ✓ GCNN tests completed")
    print(f"    - Batched pg shape: {output['pg'].shape}")
    print(f"    - Single pg shape: {output_single['pg'].shape}")

    return model, output


def verify_shapes_match(dnn_output, gcnn_output):
    """Verify that DNN and GCNN outputs have matching shapes."""
    print("\nVerifying shape consistency between DNN and GCNN...")

    for key in ["pg", "vg", "v_bus"]:
        dnn_shape = dnn_output[key].shape
        gcnn_shape = gcnn_output[key].shape
        assert (
            dnn_shape == gcnn_shape
        ), f"Shape mismatch for {key}: DNN {dnn_shape} != GCNN {gcnn_shape}"
        print(f"  ✓ {key}: DNN {dnn_shape} == GCNN {gcnn_shape}")

    print("  ✓ All output shapes match between models")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Verifying DNN and GCNN Model Implementations")
    print("=" * 60)

    try:
        # Test DNN
        dnn_model, dnn_output = test_dnn()

        # Test GCNN
        gcnn_model, gcnn_output = test_gcnn()

        # Verify shape consistency
        verify_shapes_match(dnn_output, gcnn_output)

        # Print model summaries
        print("\n" + "=" * 60)
        print("Model Summaries:")
        print("=" * 60)
        print(f"DNN: {dnn_model}")
        print(f"GCNN: {gcnn_model}")

        print("\n" + "=" * 60)
        print("✅ All Model Tests Passed!")
        print("=" * 60)

        # Exit with success code
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
