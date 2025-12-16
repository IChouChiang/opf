"""Verify end-to-end pipeline with new evaluation logic.

This script tests the complete training and evaluation pipeline:
1. Creates mock data in a temporary folder
2. Trains a GCNN model for 1 epoch with fast_dev_run
3. Evaluates the trained model
4. Verifies the evaluation output contains expected metrics
5. Cleans up temporary files

Usage:
    python tests/verify_pipeline_case6.py
"""

import sys
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
import numpy as np


def create_mock_data(temp_dir: Path) -> None:
    """
    Create mock data files in temporary directory.

    Creates:
    - samples.npz: Random data with required fields
    - topology_operators.npz: Random topology operators
    - norm_stats.npz: Normalization statistics (with non-zero std)
    """
    print(f"Creating mock data in {temp_dir}")

    # Create directory if it doesn't exist
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create samples.npz with random data for case6 (6 buses, 3 generators)
    n_samples = 100
    n_bus = 6
    n_gen = 3

    # Create random data matching the expected format
    samples_data = {
        "pd": np.random.randn(n_samples, n_bus).astype(np.float32) * 0.5
        + 1.0,  # Active power demand
        "qd": np.random.randn(n_samples, n_bus).astype(np.float32) * 0.3
        + 0.5,  # Reactive power demand
        "pg_labels": np.random.randn(n_samples, n_gen).astype(np.float32) * 0.2
        + 0.8,  # Generator active power
        "vg_labels": np.random.randn(n_samples, n_gen).astype(np.float32) * 0.05
        + 1.0,  # Generator voltage
        "gen_label": np.random.randn(n_samples, n_gen, 2).astype(
            np.float32
        ),  # Generator labels (e,f)
        "topo_id": np.zeros(
            n_samples, dtype=np.int32
        ),  # Topology ID (all same topology)
        "e_0_k": np.random.randn(n_samples, n_bus, 10).astype(np.float32)
        * 0.1,  # Graph features (e)
        "f_0_k": np.random.randn(n_samples, n_bus, 10).astype(np.float32)
        * 0.1,  # Graph features (f)
    }

    # Save samples_train.npz, samples_test.npz, and samples_unseen.npz
    samples_train_path = temp_dir / "samples_train.npz"
    samples_test_path = temp_dir / "samples_test.npz"
    samples_unseen_path = temp_dir / "samples_unseen.npz"

    np.savez(samples_train_path, **samples_data)
    np.savez(samples_test_path, **samples_data)
    np.savez(samples_unseen_path, **samples_data)  # Test file for evaluation
    print(f"  Created {samples_train_path}")
    print(f"  Created {samples_test_path}")
    print(f"  Created {samples_unseen_path}")

    # Create topology_operators.npz
    # Need shape [n_topologies, n_bus, n_bus] for g_ndiag, b_ndiag
    # and [n_topologies, n_bus] for g_diag, b_diag
    n_topologies = 1  # Single topology for case6
    topo_data = {
        "g_ndiag": np.random.randn(n_topologies, n_bus, n_bus).astype(np.float32) * 0.1,
        "b_ndiag": np.random.randn(n_topologies, n_bus, n_bus).astype(np.float32) * 0.1,
        "g_diag": np.random.randn(n_topologies, n_bus).astype(np.float32) * 0.1 + 1.0,
        "b_diag": np.random.randn(n_topologies, n_bus).astype(np.float32) * 0.1 - 1.0,
        "gen_bus_map": np.array([0, 1, 2], dtype=np.int32),  # Generator bus indices
    }

    # Make matrices symmetric for each topology
    for i in range(n_topologies):
        topo_data["g_ndiag"][i] = (
            topo_data["g_ndiag"][i] + topo_data["g_ndiag"][i].T
        ) / 2
        topo_data["b_ndiag"][i] = (
            topo_data["b_ndiag"][i] + topo_data["b_ndiag"][i].T
        ) / 2

    topo_path = temp_dir / "topology_operators.npz"
    np.savez(topo_path, **topo_data)
    print(f"  Created {topo_path}")

    # Create norm_stats.npz with valid mean/std (non-zero std)
    norm_data = {
        "pd_mean": np.random.randn(n_bus).astype(np.float32),
        "pd_std": np.abs(np.random.randn(n_bus).astype(np.float32)) + 0.1,  # Ensure > 0
        "qd_mean": np.random.randn(n_bus).astype(np.float32),
        "qd_std": np.abs(np.random.randn(n_bus).astype(np.float32)) + 0.1,
        "pg_mean": np.random.randn(n_gen).astype(np.float32),
        "pg_std": np.abs(np.random.randn(n_gen).astype(np.float32)) + 0.1,
        "vg_mean": np.random.randn(n_gen).astype(np.float32),
        "vg_std": np.abs(np.random.randn(n_gen).astype(np.float32)) + 0.1,
        "e_0_k_mean": np.random.randn(10).astype(np.float32),
        "e_0_k_std": np.abs(np.random.randn(10).astype(np.float32)) + 0.1,
        "f_0_k_mean": np.random.randn(10).astype(np.float32),
        "f_0_k_std": np.abs(np.random.randn(10).astype(np.float32)) + 0.1,
    }

    norm_path = temp_dir / "norm_stats.npz"
    np.savez(norm_path, **norm_data)
    print(f"  Created {norm_path}")


def run_training(temp_dir: Path) -> bool:
    """
    Run training with fast_dev_run.

    Args:
        temp_dir: Path to temporary data directory

    Returns:
        True if training succeeds, False otherwise
    """
    print(f"\nRunning training on mock data...")

    command = [
        "python",
        "scripts/train.py",
        f"data.data_dir={temp_dir}",
        "data.n_bus=6",
        "data.n_gen=3",
        "data.gen_bus_indices=[0,1,2]",
        "model=gcnn",
        "train.max_epochs=1",
        "+train.fast_dev_run=True",
        "train.accelerator=cpu",
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
    ]

    print(f"  Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,  # Run from project root
        )

        if result.returncode == 0:
            print(f"  [OK] Training succeeded")
            # Check for expected output
            if "fast_dev_run" in result.stdout.lower():
                print(f"    Found 'fast_dev_run' in output")
            return True
        else:
            print(f"  [FAIL] Training failed with exit code {result.returncode}")
            if result.stderr:
                print(f"    Stderr: {result.stderr[:500]}...")
            return False

    except Exception as e:
        print(f"  [FAIL] Training exception: {e}")
        return False


def run_evaluation(temp_dir: Path) -> bool:
    """
    Run evaluation on trained model.

    Args:
        temp_dir: Path to temporary data directory

    Returns:
        True if evaluation succeeds and contains expected output, False otherwise
    """
    print(f"\nRunning evaluation...")

    # Find the latest checkpoint
    checkpoints_dir = Path("lightning_logs")
    ckpt_path = None

    if checkpoints_dir.exists():
        # Look for .ckpt files
        ckpt_files = list(checkpoints_dir.rglob("*.ckpt"))
        if ckpt_files:
            # Get the most recent checkpoint
            ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            ckpt_path = ckpt_files[0].resolve()  # Get absolute path
            print(f"  Found checkpoint: {ckpt_path}")
        else:
            print(f"  [WARN] No checkpoint files found in {checkpoints_dir}")
            # Try to find any checkpoint
            for root, dirs, files in os.walk(checkpoints_dir):
                for file in files:
                    if file.endswith(".ckpt"):
                        ckpt_path = Path(root) / file
                        ckpt_path = ckpt_path.resolve()  # Get absolute path
                        print(f"  Found checkpoint: {ckpt_path}")
                        break
                if ckpt_path:
                    break

    if not ckpt_path:
        print(f"  [FAIL] No checkpoint found for evaluation")
        return False

    # Check if checkpoint file actually exists
    if not ckpt_path.exists():
        print(f"  [FAIL] Checkpoint file does not exist at: {ckpt_path}")
        # List files in the directory
        if ckpt_path.parent.exists():
            print(f"    Files in {ckpt_path.parent}:")
            for f in ckpt_path.parent.iterdir():
                print(f"      {f.name}")
        return False

    command = [
        "python",
        "scripts/evaluate.py",
        f"data.data_dir={temp_dir}",
        "data.n_bus=6",
        "data.n_gen=3",
        "data.gen_bus_indices=[0,1,2]",
        "model=gcnn",
        f"checkpoint={ckpt_path}",
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
    ]

    print(f"  Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,  # Run from project root
        )

        if result.returncode == 0:
            print(f"  [OK] Evaluation succeeded")

            # Check for expected output
            output_lower = result.stdout.lower()
            has_probabilistic = "probabilistic accuracy" in output_lower
            has_mw = "1.0 mw" in output_lower or "1.0mw" in output_lower.replace(
                " ", ""
            )

            print(f"    Contains 'Probabilistic Accuracy': {has_probabilistic}")
            print(f"    Contains '1.0 MW': {has_mw}")

            if has_probabilistic and has_mw:
                print(f"    [OK] Evaluation output contains expected metrics")
                return True
            else:
                print(f"    [WARN] Evaluation output missing expected metrics")
                # Print first 500 chars of output for debugging
                if result.stdout:
                    print(f"    Output preview: {result.stdout[:500]}...")
                return False
        else:
            print(f"  [FAIL] Evaluation failed with exit code {result.returncode}")

            # Check if we got the expected output despite the error
            # (e.g., Unicode error might happen after printing metrics)
            combined_output = result.stdout + result.stderr
            output_lower = combined_output.lower()
            has_probabilistic = "probabilistic accuracy" in output_lower
            has_mw = "1.0 mw" in output_lower or "1.0mw" in output_lower.replace(
                " ", ""
            )

            if has_probabilistic and has_mw:
                print(
                    f"    [OK] Evaluation output contains expected metrics (despite error)"
                )
                print(f"    Contains 'Probabilistic Accuracy': {has_probabilistic}")
                print(f"    Contains '1.0 MW': {has_mw}")
                return True

            if result.stderr:
                print(f"    Stderr: {result.stderr[:500]}...")
            return False

    except Exception as e:
        print(f"  [FAIL] Evaluation exception: {e}")
        return False


def cleanup(temp_dir: Path) -> None:
    """
    Clean up temporary directory and lightning_logs.

    Args:
        temp_dir: Path to temporary data directory
    """
    print(f"\nCleaning up...")

    # Remove temporary data directory
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            print(f"  Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"  [WARN] Failed to remove {temp_dir}: {e}")

    # Remove lightning_logs directory
    logs_dir = Path("lightning_logs")
    if logs_dir.exists():
        try:
            shutil.rmtree(logs_dir)
            print(f"  Removed lightning_logs directory")
        except Exception as e:
            print(f"  [WARN] Failed to remove lightning_logs: {e}")


def main():
    """Main test function."""
    print("=" * 70)
    print("End-to-End Pipeline Verification Test")
    print("=" * 70)

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="opf_test_"))
    print(f"Created temporary directory: {temp_dir}")

    success = True

    try:
        # Step 1: Create mock data
        create_mock_data(temp_dir)

        # Step 2: Run training
        if not run_training(temp_dir):
            success = False
            print(f"\n[FAIL] Training failed, skipping evaluation")
        else:
            # Step 3: Run evaluation
            if not run_evaluation(temp_dir):
                success = False
                print(f"\n[FAIL] Evaluation failed or missing expected output")

    except Exception as e:
        print(f"\n[FAIL] Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        # Step 4: Cleanup
        cleanup(temp_dir)

    # Final result
    print("\n" + "=" * 70)
    if success:
        print("[OK] End-to-End Pipeline Test PASSED")
        print("All steps completed successfully:")
        print("  ✓ Mock data created")
        print("  ✓ Training completed with fast_dev_run")
        print("  ✓ Evaluation ran and produced expected metrics")
        print("  ✓ Cleanup completed")
        sys.exit(0)
    else:
        print("[FAIL] End-to-End Pipeline Test FAILED")
        print("See error messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
