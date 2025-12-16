"""
Batch size tuning script for GCNN-OPF model.

Features:
- Automatically detects and skips already-tested batch sizes
- Loads cached results from previous runs
- Generates comparison plots across all tested configurations

Usage:
    python gcnn_opf_01/tune_batch_size.py                    # Skip existing, test new only
    python gcnn_opf_01/tune_batch_size.py --force            # Re-run all tests
    python gcnn_opf_01/tune_batch_size.py --batch_sizes 4 8 12  # Test custom batch sizes
"""

import subprocess
import time
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add gcnn_opf_01 to path
sys.path.insert(0, str(Path(__file__).parent))
from config_model_01 import ModelConfig

# Configuration
PYTHON_EXE = r"E:\DevTools\anaconda3\envs\opf311\python.exe"
TRAIN_SCRIPT = "gcnn_opf_01/train.py"
BATCH_SIZES = [2, 4, 6, 8, 10, 12, 16, 24, 32, 64]  # Expanded batch sizes to test
EPOCHS = 5  # Reduced epochs for tuning
DATA_DIR = "gcnn_opf_01/data"
BASE_RESULTS_DIR = "gcnn_opf_01/results/tuning"  # Universal folder

# Get current model configuration
CURRENT_CONFIG = ModelConfig()
CURRENT_NEURONS = CURRENT_CONFIG.neurons_fc


def load_existing_results(base_dir: Path) -> dict:
    """Load results from all existing batch size folders."""
    existing = {}
    if not base_dir.exists():
        return existing

    for bs_dir in base_dir.glob("bs_*"):
        history_path = bs_dir / "training_history.npz"
        if history_path.exists():
            try:
                # Parse folder name: bs_{bs} or bs_{bs}_n{neurons}
                parts = bs_dir.name.split("_")
                bs = int(parts[1])

                # Detect neuron count
                neurons = 128  # Default for legacy folders
                for part in parts:
                    if part.startswith("n"):
                        try:
                            neurons = int(part[1:])
                        except ValueError:
                            pass

                data = np.load(history_path)
                key = (bs, neurons)
                existing[key] = {
                    "batch_size": bs,
                    "neurons": neurons,
                    "final_train_loss": float(data["train_loss"][-1]),
                    "final_val_loss": float(data["val_loss"][-1]),
                    "epochs": len(data["val_loss"]),
                    "path": str(bs_dir),
                }
                print(
                    f"  [cached] bs={bs}, n={neurons}: val_loss={existing[key]['final_val_loss']:.6f}"
                )
            except Exception as e:
                print(f"  Warning: Could not load {history_path}: {e}")

    return existing


def run_tuning(force_rerun=False):
    """Run batch size tuning, skipping already-tested values unless force_rerun=True."""
    results = []
    base_dir = Path(BASE_RESULTS_DIR)

    print("=" * 70)
    print("BATCH SIZE TUNING FOR GCNN-OPF")
    print("=" * 70)
    print(f"Target batch sizes: {BATCH_SIZES}")
    print(f"Current Configuration: {CURRENT_NEURONS} neurons")
    print(f"Epochs per test: {EPOCHS}")
    print(f"Results directory: {base_dir}")
    print("-" * 70)

    # Load existing results
    print("\nChecking for existing results...")
    existing_results = load_existing_results(base_dir)

    if existing_results:
        print(f"\nFound {len(existing_results)} cached results.")
    else:
        print("\nNo existing results found.")

    # Determine which batch sizes need testing
    to_test = []
    skipped = []

    if force_rerun:
        to_test = BATCH_SIZES
        print(f"\n[FORCE MODE] Will re-run all: {to_test}")
    else:
        for bs in BATCH_SIZES:
            key = (bs, CURRENT_NEURONS)
            if key in existing_results:
                skipped.append(bs)
            else:
                to_test.append(bs)

        if skipped:
            print(f"\n[SKIP] Already tested for n={CURRENT_NEURONS}: {skipped}")
        if to_test:
            print(f"[NEW] Will test: {to_test}")
        else:
            print(f"\n✓ All batch sizes already tested for n={CURRENT_NEURONS}!")
            print(
                "  Use --force to re-test, or --batch_sizes to test different values."
            )

    # Run new tests
    for bs in to_test:
        print(f"\n{'='*50}")
        print(f"Testing Batch Size: {bs} (Neurons: {CURRENT_NEURONS})")
        print(f"{'='*50}")

        # Create specific folder name with neuron count
        run_dir = base_dir / f"bs_{bs}_n{CURRENT_NEURONS}"

        cmd = [
            PYTHON_EXE,
            TRAIN_SCRIPT,
            "--batch_size",
            str(bs),
            "--epochs",
            str(EPOCHS),
            "--data_dir",
            DATA_DIR,
            "--results_dir",
            str(run_dir),
            "--log_interval",
            "10",
        ]

        start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
            total_time = time.time() - start_time
            avg_time_per_epoch = total_time / EPOCHS

            history_path = run_dir / "training_history.npz"
            if history_path.exists():
                data = np.load(history_path)
                existing_results[(bs, CURRENT_NEURONS)] = {
                    "batch_size": bs,
                    "neurons": CURRENT_NEURONS,
                    "time_per_epoch": avg_time_per_epoch,
                    "final_train_loss": float(data["train_loss"][-1]),
                    "final_val_loss": float(data["val_loss"][-1]),
                    "total_time": total_time,
                    "epochs": len(data["val_loss"]),
                    "path": str(run_dir),
                }
                print(
                    f"  -> Finished in {total_time:.2f}s ({avg_time_per_epoch:.2f}s/epoch)"
                )
                print(
                    f"  -> Val Loss: {existing_results[(bs, CURRENT_NEURONS)]['final_val_loss']:.6f}"
                )

        except subprocess.CalledProcessError as e:
            print(f"  -> Training failed for batch size {bs}: {e}")

    # Print Summary
    print_summary(existing_results, to_test)

    # Plot comparison (all tested batch sizes)
    plot_comparison(existing_results)


def print_summary(all_results, newly_tested):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("BATCH SIZE TUNING RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Batch':<8} | {'Neurons':<8} | {'Epochs':<7} | {'Train Loss':<12} | {'Val Loss':<12} | {'Status':<10}"
    )
    print("-" * 80)

    best_bs = None
    min_loss = float("inf")

    # Sort by batch size, then neurons
    sorted_keys = sorted(all_results.keys(), key=lambda x: (x[0], x[1]))

    for key in sorted_keys:
        bs, neurons = key
        r = all_results[key]
        status = (
            "NEW" if bs in newly_tested and neurons == CURRENT_NEURONS else "cached"
        )
        print(
            f"{r['batch_size']:<8} | {r.get('neurons', 'N/A'):<8} | {r.get('epochs', 'N/A'):<7} | {r['final_train_loss']:<12.6f} | {r['final_val_loss']:<12.6f} | {status:<10}"
        )

        # Only consider current configuration for "Best Batch Size" recommendation
        if neurons == CURRENT_NEURONS and r["final_val_loss"] < min_loss:
            min_loss = r["final_val_loss"]
            best_bs = r["batch_size"]

    print("-" * 80)
    if best_bs is not None:
        print(
            f"Best Batch Size (for n={CURRENT_NEURONS}): {best_bs} (val_loss={min_loss:.6f})"
        )
    else:
        print(f"No results for current configuration (n={CURRENT_NEURONS}).")
    print(f"Total configurations: {len(all_results)}")


def plot_comparison(results_dict):
    """Generate comparison plot for all tested batch sizes."""
    if not results_dict:
        return

    # Group by neurons
    by_neurons = {}
    for r in results_dict.values():
        n = r.get("neurons", 128)
        if n not in by_neurons:
            by_neurons[n] = []
        by_neurons[n].append(r)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot a line for each neuron count
    for n, res_list in sorted(by_neurons.items()):
        # Sort by batch size
        res_list.sort(key=lambda x: x["batch_size"])

        batch_sizes = [r["batch_size"] for r in res_list]
        losses = [r["final_val_loss"] for r in res_list]

        label = f"Neurons={n}"
        marker = "o"
        if n == CURRENT_NEURONS:
            label += " (Current)"
            marker = "s"
            linewidth = 2.5
        else:
            linewidth = 1.5

        ax.plot(batch_sizes, losses, marker=marker, linewidth=linewidth, label=label)

        # Add labels
        for bs, loss in zip(batch_sizes, losses):
            ax.text(bs, loss, f"{loss:.4f}", fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Final Validation Loss", fontsize=12)
    ax.set_title("Batch Size Tuning Results", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = Path(BASE_RESULTS_DIR) / "tuning_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nComparison plot saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch size tuning for GCNN-OPF")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all batch sizes (ignore cached results)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        help="Custom batch sizes to test (e.g., --batch_sizes 4 8 12 16)",
    )
    args = parser.parse_args()

    if args.batch_sizes:
        BATCH_SIZES = args.batch_sizes

    run_tuning(force_rerun=args.force)
