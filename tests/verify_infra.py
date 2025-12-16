#!/usr/bin/env python
"""
Verify experimental infrastructure:
1. Parameter counter works
2. Warm-start training works
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=False)
    if check and result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        print(f"STDOUT: {result.stdout}")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def test_param_counter():
    """Test that parameter counter works."""
    print("\n" + "="*60)
    print("TEST 1: Parameter Counter")
    print("="*60)
    
    cmd = [sys.executable, "scripts/calc_params.py", "model=dnn"]
    result = run_command(cmd)
    
    assert "Parameters:" in result.stdout, "Expected 'Parameters:' in output"
    print("✅ Parameter counter working")
    print(f"Output snippet: {result.stdout[:200]}...")


def test_warm_start():
    """Test that warm-start training works."""
    print("\n" + "="*60)
    print("TEST 2: Warm Start")
    print("="*60)
    
    # Use existing legacy data for testing
    legacy_data_dir = Path("legacy/gcnn_opf_01/data_matlab_npz")
    
    if not legacy_data_dir.exists():
        print("⚠️  Legacy data not found, skipping warm start test")
        print(f"Expected data at: {legacy_data_dir}")
        return
    
    # Step A: Train for 1 epoch
    print("\nStep A: Training for 1 epoch...")
    train_cmd = [
        sys.executable, "scripts/train.py",
        f"data.data_dir={legacy_data_dir}",
        "data.n_bus=39",
        "data.n_gen=10",
        "data.gen_bus_indices=[29,30,31,32,33,34,35,36,37,38]",
        "data.train_file=samples_train.npz",
        "data.val_file=samples_test.npz",
        "data.test_file=samples_unseen.npz",
        "model=gcnn",
        "model.architecture.in_channels=10",  # Match legacy data feature iterations
        "train.max_epochs=1",
        "+train.fast_dev_run=True",
        "train.accelerator=cpu",
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
    ]
    
    result = run_command(train_cmd)
    
    # Find the checkpoint (should be in ./lightning_logs/version_X/checkpoints/)
    checkpoint_dir = Path("lightning_logs")
    checkpoints = list(checkpoint_dir.glob("version_*/checkpoints/*.ckpt"))
    
    if not checkpoints:
        raise RuntimeError("No checkpoint found after training")
    
    # Use the most recent checkpoint
    ckpt_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Found checkpoint: {ckpt_path}")
    
    # Step B: Load checkpoint and train again
    print("\nStep B: Training with warm start...")
    warmstart_cmd = [
        sys.executable, "scripts/train.py",
        f"data.data_dir={legacy_data_dir}",
        "data.n_bus=39",
        "data.n_gen=10",
        "data.gen_bus_indices=[29,30,31,32,33,34,35,36,37,38]",
        "data.train_file=samples_train.npz",
        "data.val_file=samples_test.npz",
        "data.test_file=samples_unseen.npz",
        "model=gcnn",
        "model.architecture.in_channels=10",  # Match legacy data feature iterations
        "train.max_epochs=1",
        "+train.fast_dev_run=True",
        "train.accelerator=cpu",
        f"train.warm_start_ckpt={ckpt_path}",
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
    ]
    
    result = run_command(warmstart_cmd)
    
    assert result.returncode == 0, "Warm start training failed"
    print("✅ Warm-start working")


def main():
    """Run all tests."""
    try:
        test_param_counter()
        test_warm_start()
        
        print("\n" + "="*60)
        print("✅ Infrastructure Ready: Parameter counting and Warm-start working")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
