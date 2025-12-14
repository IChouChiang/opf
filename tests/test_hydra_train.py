"""Test Hydra configuration system integration.

This script verifies that the Hydra configuration system works correctly
by running training with fast development mode for both DNN and GCNN models.

Usage:
    python tests/test_hydra_train.py
"""

import subprocess
import sys
from pathlib import Path


def run_hydra_command(command: str) -> bool:
    """
    Run a Hydra command and check if it succeeds.
    
    Args:
        command: Full command string to execute
        
    Returns:
        True if command succeeds (exit code 0), False otherwise
    """
    print(f"Running: {command}")
    
    try:
        # Run command with subprocess
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,  # Run from project root
        )
        
        if result.returncode == 0:
            print(f"  [OK] Command succeeded")
            # Print first few lines of output for debugging
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[:5]:  # Show first 5 lines
                    if line.strip():
                        print(f"    {line}")
                if len(lines) > 5:
                    print(f"    ... ({len(lines) - 5} more lines)")
            return True
        else:
            print(f"  [FAIL] Command failed with exit code {result.returncode}")
            if result.stderr:
                print(f"  Error output:\n{result.stderr[:500]}")  # Show first 500 chars
            return False
            
    except Exception as e:
        print(f"  ❌ Exception running command: {e}")
        return False


def test_dnn_hydra() -> bool:
    """Test DNN training with Hydra configuration."""
    print("\n" + "=" * 60)
    print("Testing DNN Hydra Configuration")
    print("=" * 60)
    
    command = (
        "python scripts/train.py "
        "model=dnn "
        "data=case39 "  # Use case39 data (actual data files)
        "train.max_epochs=1 "
        "+train.fast_dev_run=True "  # Use + to append to config
        "train.accelerator=cpu "  # Use CPU for consistent testing
        "hydra.job.chdir=False "  # Don't change directory
        "hydra.output_subdir=null "  # Don't create output subdirectory
        "hydra.run.dir=."  # Run in current directory
    )
    
    return run_hydra_command(command)


def test_gcnn_hydra() -> bool:
    """Test GCNN training with Hydra configuration."""
    print("\n" + "=" * 60)
    print("Testing GCNN Hydra Configuration")
    print("=" * 60)
    
    command = (
        "python scripts/train.py "
        "model=gcnn "
        "data=case39 "  # Use case39 data (actual data files)
        "train.max_epochs=1 "
        "+train.fast_dev_run=True "  # Use + to append to config
        "train.accelerator=cpu "  # Use CPU for consistent testing
        "hydra.job.chdir=False "  # Don't change directory
        "hydra.output_subdir=null "  # Don't create output subdirectory
        "hydra.run.dir=."  # Run in current directory
    )
    
    return run_hydra_command(command)


def test_config_overrides() -> bool:
    """Test that Hydra configuration overrides work correctly."""
    print("\n" + "=" * 60)
    print("Testing Hydra Configuration Overrides")
    print("=" * 60)
    
    # Test multiple overrides at once
    command = (
        "python scripts/train.py "
        "model=dnn "
        "data=case39 "  # Switch to case39
        "model.task.lr=0.0005 "
        "train.batch_size=16 "
        "train.max_epochs=1 "
        "+train.fast_dev_run=True "  # Use + to append to config
        "train.accelerator=cpu "
        "hydra.job.chdir=False "
        "hydra.output_subdir=null "
        "hydra.run.dir=."
    )
    
    return run_hydra_command(command)


def main():
    """Run all Hydra integration tests."""
    print("=" * 60)
    print("Hydra Configuration System Integration Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: DNN with Hydra
    dnn_success = test_dnn_hydra()
    success = success and dnn_success
    
    # Test 2: GCNN with Hydra
    gcnn_success = test_gcnn_hydra()
    success = success and gcnn_success
    
    # Test 3: Configuration overrides
    override_success = test_config_overrides()
    success = success and override_success
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if success:
        print("[OK] Hydra Configuration System Functional")
        print("\nAll tests passed:")
        print("  - DNN training with Hydra configuration")
        print("  - GCNN training with Hydra configuration")
        print("  - Configuration overrides work correctly")
        print("\nHydra integration is working properly!")
        sys.exit(0)
    else:
        print("[FAIL] Hydra Configuration System Tests Failed")
        print("\nSome tests failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
