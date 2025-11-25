import subprocess
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Configuration
PYTHON_EXE = r"E:\DevTools\anaconda3\envs\opf311\python.exe"
TRAIN_SCRIPT = "gcnn_opf_01/train.py"
BATCH_SIZES = [2, 4, 6, 8]
EPOCHS = 5  # Reduced epochs for tuning
DATA_DIR = "gcnn_opf_01/data"
BASE_RESULTS_DIR = "gcnn_opf_01/results/tuning_ultra_fine"


def run_tuning():
    results = []

    print(f"Starting batch size tuning with sizes: {BATCH_SIZES}")
    print(f"Training for {EPOCHS} epochs each...")

    for bs in BATCH_SIZES:
        print(f"\n{'='*40}")
        print(f"Testing Batch Size: {bs}")
        print(f"{'='*40}")

        run_dir = Path(BASE_RESULTS_DIR) / f"bs_{bs}"

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
            # Run training
            subprocess.run(cmd, check=True)
            total_time = time.time() - start_time
            avg_time_per_epoch = total_time / EPOCHS

            # Load results
            history_path = run_dir / "training_history.npz"
            if history_path.exists():
                data = np.load(history_path)
                final_train_loss = data["train_loss"][-1]
                final_val_loss = data["val_loss"][-1]

                results.append(
                    {
                        "batch_size": bs,
                        "time_per_epoch": avg_time_per_epoch,
                        "final_train_loss": final_train_loss,
                        "final_val_loss": final_val_loss,
                        "total_time": total_time,
                    }
                )
                print(
                    f"  -> Finished in {total_time:.2f}s ({avg_time_per_epoch:.2f}s/epoch)"
                )
                print(f"  -> Final Val Loss: {final_val_loss:.6f}")
            else:
                print(f"  -> Error: Results file not found at {history_path}")

        except subprocess.CalledProcessError as e:
            print(f"  -> Training failed for batch size {bs}")
            print(e)

    # Print Summary
    print("\n" + "=" * 60)
    print("BATCH SIZE TUNING RESULTS")
    print("=" * 60)
    print(
        f"{'Batch Size':<12} | {'Time/Epoch (s)':<15} | {'Final Val Loss':<15} | {'Total Time (s)':<15}"
    )
    print("-" * 60)

    best_bs_time = None
    best_bs_loss = None
    min_time = float("inf")
    min_loss = float("inf")

    for r in results:
        print(
            f"{r['batch_size']:<12} | {r['time_per_epoch']:<15.2f} | {r['final_val_loss']:<15.6f} | {r['total_time']:<15.2f}"
        )

        if r["time_per_epoch"] < min_time:
            min_time = r["time_per_epoch"]
            best_bs_time = r["batch_size"]

        if r["final_val_loss"] < min_loss:
            min_loss = r["final_val_loss"]
            best_bs_loss = r["batch_size"]

    print("-" * 60)
    print(f"Fastest Batch Size: {best_bs_time}")
    print(f"Best Loss Batch Size: {best_bs_loss}")

    # Plot comparison
    plot_comparison(results)


def plot_comparison(results):
    if not results:
        return

    batch_sizes = [r["batch_size"] for r in results]
    times = [r["time_per_epoch"] for r in results]
    losses = [r["final_val_loss"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time per Epoch (s)", color=color)
    ax1.plot(batch_sizes, times, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(batch_sizes)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Final Validation Loss", color=color)
    ax2.plot(batch_sizes, losses, color=color, marker="s")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"Batch Size Tuning ({EPOCHS} epochs)")
    plt.grid(True, alpha=0.3)

    out_path = Path(BASE_RESULTS_DIR) / "tuning_comparison.png"
    plt.savefig(out_path)
    print(f"\nComparison plot saved to {out_path}")


if __name__ == "__main__":
    run_tuning()
