"""Loss curve plotting utilities.

Generates training/validation loss curves from PyTorch Lightning metrics.csv files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curves(
    metrics_csv_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> Path | None:
    """
    Plot training and validation loss curves from metrics.csv.

    Args:
        metrics_csv_path: Path to metrics.csv file from PyTorch Lightning
        output_path: Path to save the PNG file. If None, saves next to metrics.csv
        title: Plot title. If None, uses the version folder name
        show: Whether to display the plot interactively

    Returns:
        Path to the saved PNG file, or None if failed
    """
    metrics_path = Path(metrics_csv_path)
    if not metrics_path.exists():
        print(f"[WARN] metrics.csv not found: {metrics_path}")
        return None

    # Read metrics
    df = pd.read_csv(metrics_path)

    if df.empty:
        print(f"[WARN] Empty metrics.csv: {metrics_path}")
        return None

    # Group by epoch and get the values (train and val are on separate rows)
    # Train rows have train/* columns filled, val rows have val/* columns filled
    train_df = df[df["train/loss"].notna()][
        ["epoch", "train/loss", "train/sup", "train/phys"]
    ].copy()
    val_df = df[df["val/loss"].notna()][
        ["epoch", "val/loss", "val/sup", "val/phys"]
    ].copy()

    if train_df.empty and val_df.empty:
        print(f"[WARN] No loss data in metrics.csv: {metrics_path}")
        return None

    # Determine output path
    if output_path is None:
        output_path = metrics_path.parent / "loss_curves.png"
    else:
        output_path = Path(output_path)

    # Determine title
    if title is None:
        # Use version folder name (e.g., "version_50")
        title = metrics_path.parent.name

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Total Loss
    ax1 = axes[0]
    if not train_df.empty:
        ax1.plot(
            train_df["epoch"],
            train_df["train/loss"],
            "b-",
            label="Train Loss",
            linewidth=2,
        )
    if not val_df.empty:
        ax1.plot(
            val_df["epoch"], val_df["val/loss"], "r-", label="Val Loss", linewidth=2
        )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"{title} - Total Loss", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Loss Components (Supervised vs Physics)
    ax2 = axes[1]
    if not train_df.empty:
        ax2.plot(
            train_df["epoch"],
            train_df["train/sup"],
            "b--",
            label="Train Supervised",
            linewidth=1.5,
            alpha=0.8,
        )
        ax2.plot(
            train_df["epoch"],
            train_df["train/phys"],
            "b:",
            label="Train Physics",
            linewidth=1.5,
            alpha=0.8,
        )
    if not val_df.empty:
        ax2.plot(
            val_df["epoch"],
            val_df["val/sup"],
            "r--",
            label="Val Supervised",
            linewidth=1.5,
            alpha=0.8,
        )
        ax2.plot(
            val_df["epoch"],
            val_df["val/phys"],
            "r:",
            label="Val Physics",
            linewidth=1.5,
            alpha=0.8,
        )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title(f"{title} - Loss Components", fontsize=14)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Loss curves saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_loss_curves_from_log_dir(log_dir: str | Path, **kwargs) -> Path | None:
    """
    Plot loss curves from a lightning log directory.

    Args:
        log_dir: Path to lightning log directory (e.g., lightning_logs/version_50)
        **kwargs: Additional arguments passed to plot_loss_curves

    Returns:
        Path to the saved PNG file, or None if failed
    """
    log_dir = Path(log_dir)
    metrics_csv = log_dir / "metrics.csv"
    return plot_loss_curves(metrics_csv, **kwargs)


def batch_plot_loss_curves(
    base_dir: str | Path = "lightning_logs",
    version_min: int | None = None,
    version_max: int | None = None,
) -> list[Path]:
    """
    Batch generate loss curve plots for multiple versions.

    Args:
        base_dir: Base directory containing version folders
        version_min: Minimum version number to process (inclusive)
        version_max: Maximum version number to process (inclusive)

    Returns:
        List of paths to generated PNG files
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return []

    generated = []

    # Find all version directories
    version_dirs = sorted(base_dir.glob("version_*"))

    for version_dir in version_dirs:
        # Extract version number
        try:
            version_num = int(version_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Filter by version range
        if version_min is not None and version_num < version_min:
            continue
        if version_max is not None and version_num > version_max:
            continue

        # Check if metrics.csv exists
        metrics_csv = version_dir / "metrics.csv"
        if not metrics_csv.exists():
            print(f"[SKIP] No metrics.csv in {version_dir.name}")
            continue

        # Check if loss_curves.png already exists
        output_path = version_dir / "loss_curves.png"
        if output_path.exists():
            print(f"[SKIP] Already exists: {output_path}")
            continue

        # Generate plot
        result = plot_loss_curves(metrics_csv)
        if result:
            generated.append(result)

    print(f"\nGenerated {len(generated)} loss curve plots")
    return generated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot loss curves from metrics.csv")
    parser.add_argument(
        "--version",
        type=int,
        help="Specific version number to plot",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process all versions",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=None,
        help="Minimum version number for batch processing",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum version number for batch processing",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively",
    )

    args = parser.parse_args()

    if args.batch:
        batch_plot_loss_curves(version_min=args.min, version_max=args.max)
    elif args.version:
        log_dir = Path("lightning_logs") / f"version_{args.version}"
        plot_loss_curves_from_log_dir(log_dir, show=args.show)
    else:
        print("Usage:")
        print("  python -m deep_opf.utils.plot_loss --version 50")
        print("  python -m deep_opf.utils.plot_loss --batch --min 44")
        print("  python -m deep_opf.utils.plot_loss --batch")
