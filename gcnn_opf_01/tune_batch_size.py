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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Configuration
PYTHON_EXE = r"E:\DevTools\anaconda3\envs\opf311\python.exe"
TRAIN_SCRIPT = "gcnn_opf_01/train.py"
BATCH_SIZES = [6, 8, 16, 32]  # Default batch sizes to test
EPOCHS = 5  # Reduced epochs for tuning
DATA_DIR = "gcnn_opf_01/data"
BASE_RESULTS_DIR = "gcnn_opf_01/results/tuning"


def load_existing_results(base_dir: Path) -> dict:
    """Load results from all existing batch size folders."""
    existing = {}
    if not base_dir.exists():
        return existing
    
    for bs_dir in base_dir.glob("bs_*"):
        history_path = bs_dir / "training_history.npz"
        if history_path.exists():
            try:
                bs = int(bs_dir.name.split("_")[1])
                data = np.load(history_path)
                existing[bs] = {
                    "batch_size": bs,
                    "final_train_loss": float(data["train_loss"][-1]),
                    "final_val_loss": float(data["val_loss"][-1]),
                    "epochs": len(data["val_loss"]),
                    "path": str(bs_dir),
                }
                print(f"  [cached] bs={bs}: val_loss={existing[bs]['final_val_loss']:.6f} ({existing[bs]['epochs']} epochs)")
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
    if force_rerun:
        to_test = BATCH_SIZES
        print(f"\n[FORCE MODE] Will re-run all: {to_test}")
    else:
        to_test = [bs for bs in BATCH_SIZES if bs not in existing_results]
        skipped = [bs for bs in BATCH_SIZES if bs in existing_results]
        if skipped:
            print(f"\n[SKIP] Already tested: {skipped}")
        if to_test:
            print(f"[NEW] Will test: {to_test}")
        else:
            print("\n✓ All batch sizes already tested!")
            print("  Use --force to re-test, or --batch_sizes to test different values.")
    
    # Run new tests
    for bs in to_test:
        print(f"\n{'='*50}")
        print(f"Testing Batch Size: {bs}")
        print(f"{'='*50}")

        run_dir = base_dir / f"bs_{bs}"

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
                existing_results[bs] = {
                    "batch_size": bs,
                    "time_per_epoch": avg_time_per_epoch,
                    "final_train_loss": float(data["train_loss"][-1]),
                    "final_val_loss": float(data["val_loss"][-1]),
                    "total_time": total_time,
                    "epochs": len(data["val_loss"]),
                    "path": str(run_dir),
                }
                print(f"  -> Finished in {total_time:.2f}s ({avg_time_per_epoch:.2f}s/epoch)")
                print(f"  -> Val Loss: {existing_results[bs]['final_val_loss']:.6f}")

        except subprocess.CalledProcessError as e:
            print(f"  -> Training failed for batch size {bs}: {e}")

    # Print Summary
    print_summary(existing_results, to_test)
    
    # Plot comparison (all tested batch sizes)
    plot_comparison(existing_results)


def print_summary(all_results, newly_tested):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("BATCH SIZE TUNING RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Batch':<8} | {'Epochs':<7} | {'Train Loss':<12} | {'Val Loss':<12} | {'Status':<10}")
    print("-" * 70)

    best_bs = None
    min_loss = float("inf")

    for bs in sorted(all_results.keys()):
        r = all_results[bs]
        status = "NEW" if bs in newly_tested else "cached"
        print(f"{r['batch_size']:<8} | {r.get('epochs', 'N/A'):<7} | {r['final_train_loss']:<12.6f} | {r['final_val_loss']:<12.6f} | {status:<10}")

        if r["final_val_loss"] < min_loss:
            min_loss = r["final_val_loss"]
            best_bs = r["batch_size"]

    print("-" * 70)
    print(f"Best Batch Size: {best_bs} (val_loss={min_loss:.6f})")
    print(f"Total configurations: {len(all_results)}")


def plot_comparison(results_dict):
    """Generate comparison plot for all tested batch sizes."""
    if not results_dict:
        return

    # Sort by batch size
    results = sorted(results_dict.values(), key=lambda x: x["batch_size"])
    
    batch_sizes = [r["batch_size"] for r in results]
    losses = [r["final_val_loss"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["steelblue"] * len(batch_sizes)
    best_idx = losses.index(min(losses))
    colors[best_idx] = "green"

    bars = ax.bar(range(len(batch_sizes)), losses, color=colors, alpha=0.8)
    
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Final Validation Loss", fontsize=12)
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    
    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, losses)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{loss:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label=f'Best: bs={batch_sizes[best_idx]}'),
        Patch(facecolor='steelblue', alpha=0.8, label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"Batch Size Tuning Results ({len(results)} configurations)", fontsize=14)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out_path = Path(BASE_RESULTS_DIR) / "tuning_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nComparison plot saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch size tuning for GCNN-OPF")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-run all batch sizes (ignore cached results)")
    parser.add_argument("--batch_sizes", type=int, nargs="+", 
                        help="Custom batch sizes to test (e.g., --batch_sizes 4 8 12 16)")
    args = parser.parse_args()
    
    if args.batch_sizes:
        BATCH_SIZES = args.batch_sizes
    
    run_tuning(force_rerun=args.force)
