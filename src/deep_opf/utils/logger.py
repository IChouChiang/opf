"""Experiment logging utilities for deep_opf.

This module provides CSV-based experiment tracking without external dependencies.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_experiment_to_csv(
    cfg: DictConfig,
    model: nn.Module,
    best_loss: float,
    duration: float,
    csv_path: Union[str, Path] = "experiments_log.csv",
    extra_metrics: Optional[dict[str, Any]] = None,
) -> None:
    """
    Log experiment results to a CSV file for tracking.

    Args:
        cfg: Hydra configuration object
        model: Trained model (for parameter count)
        best_loss: Best validation loss achieved
        duration: Training duration in seconds
        csv_path: Path to CSV file (created if doesn't exist)
        extra_metrics: Optional dict of additional metrics to log

    Example:
        >>> log_experiment_to_csv(cfg, model, best_loss=0.0012, duration=120.5)
        # Appends row to experiments_log.csv
    """
    csv_path = Path(csv_path)

    # Calculate model parameters
    total_params = count_parameters(model)

    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract config values safely
    model_name = cfg.model.get("name", "unknown")
    dataset = cfg.data.get("name", "unknown")
    n_bus = cfg.data.get("n_bus", 0)
    n_gen = cfg.data.get("n_gen", 0)

    # Model architecture params (handle different model types)
    arch = cfg.model.get("architecture", {})
    if model_name == "gcnn":
        hidden_dim = arch.get("fc_hidden_dim", 0)
        channels = arch.get("hidden_channels", 0)
        in_channels = arch.get("in_channels", 0)
        layers = arch.get("n_layers", 0)
    elif model_name == "dnn":
        hidden_dim = arch.get("hidden_dim", 0)
        channels = 0  # N/A for DNN
        in_channels = 0  # N/A for DNN
        layers = arch.get("num_layers", 0)
    else:
        hidden_dim = arch.get("hidden_dim", arch.get("fc_hidden_dim", 0))
        channels = arch.get("hidden_channels", 0)
        in_channels = arch.get("in_channels", 0)
        layers = arch.get("n_layers", arch.get("num_layers", 0))

    # Training params
    lr = cfg.model.task.get("lr", cfg.train.get("lr", 0))
    kappa = cfg.model.task.get("kappa", 0)
    weight_decay = cfg.model.task.get("weight_decay", 0)
    batch_size = cfg.train.get("batch_size", 0)
    max_epochs = cfg.train.get("max_epochs", 0)

    # Warm start info
    warm_start = cfg.train.get("warm_start_ckpt") is not None

    # Format duration
    duration_str = _format_duration(duration)

    # Build row data
    row_data = {
        "timestamp": timestamp,
        "model": model_name,
        "dataset": dataset,
        "n_bus": n_bus,
        "n_gen": n_gen,
        "params": total_params,
        "hidden_dim": hidden_dim,
        "channels": channels,
        "in_channels": in_channels,
        "layers": layers,
        "lr": lr,
        "kappa": kappa,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "warm_start": warm_start,
        "best_loss": f"{best_loss:.6f}",
        "duration": duration_str,
        "duration_sec": f"{duration:.1f}",
    }

    # Add extra metrics if provided
    if extra_metrics:
        for key, value in extra_metrics.items():
            if isinstance(value, float):
                row_data[key] = f"{value:.6f}"
            else:
                row_data[key] = value

    # Check if file exists to determine if header is needed
    file_exists = csv_path.exists()

    # Write to CSV
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

    print(f"Experiment logged to: {csv_path}")


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def load_experiment_log(
    csv_path: Union[str, Path] = "experiments_log.csv",
) -> list[dict]:
    """
    Load experiment log from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of experiment records as dicts
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def print_experiment_summary(
    csv_path: Union[str, Path] = "experiments_log.csv",
) -> None:
    """Print a summary table of all experiments."""
    records = load_experiment_log(csv_path)

    if not records:
        print("No experiments logged yet.")
        return

    print(f"\n{'='*80}")
    print(f"Experiment Log: {csv_path} ({len(records)} experiments)")
    print(f"{'='*80}")

    # Print header
    header = f"{'Model':<8} {'Dataset':<10} {'Params':<10} {'Loss':<10} {'Duration':<12} {'Timestamp':<20}"
    print(header)
    print("-" * 80)

    # Print rows
    for r in records[-10:]:  # Show last 10
        row = (
            f"{r.get('model', 'N/A'):<8} "
            f"{r.get('dataset', 'N/A'):<10} "
            f"{r.get('params', 'N/A'):<10} "
            f"{r.get('best_loss', 'N/A'):<10} "
            f"{r.get('duration', 'N/A'):<12} "
            f"{r.get('timestamp', 'N/A'):<20}"
        )
        print(row)

    print(f"{'='*80}\n")
