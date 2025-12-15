"""Experiment logging utilities for deep_opf.

This module provides CSV-based experiment tracking without external dependencies.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig


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
    log_dir: Optional[Union[str, Path]] = None,
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
        log_dir: Path to Hydra output directory (for finding loss curves later)

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
        "phase": "Training",
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
        # Evaluation metrics (N/A for training rows)
        "R2_PG": "",
        "R2_VG": "",
        "Pacc_PG": "",
        "Pacc_VG": "",
        "RMSE_PG": "",
        "RMSE_VG": "",
        "MAE_PG": "",
        "MAE_VG": "",
        "Physics_Violation_MW": "",
        # Log directory for finding artifacts
        "log_dir": str(Path(log_dir).resolve()) if log_dir else "",
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


def log_evaluation_to_csv(
    model_name: str,
    dataset_name: str,
    metrics: dict[str, Any],
    csv_path: Union[str, Path] = "experiments_log.csv",
    ckpt_path: Optional[Union[str, Path]] = None,
    n_samples: Optional[int] = None,
) -> None:
    """
    Log evaluation results to the experiment CSV file.

    This function appends an evaluation row that includes metrics like R², Pacc,
    RMSE, MAE, and physics violation from the evaluate.py script.

    Args:
        model_name: Name of the model (e.g., "gcnn", "dnn")
        dataset_name: Name of the dataset (e.g., "case6ww", "case39")
        metrics: Dictionary containing evaluation metrics:
            - R2_PG, R2_VG: R-squared for PG and VG
            - Pacc_PG, Pacc_VG: Probabilistic accuracy (%)
            - RMSE_PG, RMSE_VG: Root mean squared error
            - MAE_PG, MAE_VG: Mean absolute error
            - Physics_Violation_MW: Physics violation in MW
        csv_path: Path to CSV file
        ckpt_path: Path to the checkpoint that was evaluated
        n_samples: Number of test samples evaluated

    Example:
        >>> metrics = {
        ...     "R2_PG": 0.9987, "R2_VG": 0.9999,
        ...     "Pacc_PG": 98.73, "Pacc_VG": 100.0,
        ...     "RMSE_PG": 0.0012, "RMSE_VG": 0.0001,
        ...     "MAE_PG": 0.0008, "MAE_VG": 0.00005,
        ...     "Physics_Violation_MW": 0.25,
        ... }
        >>> log_evaluation_to_csv("gcnn", "case6ww", metrics)
    """
    csv_path = Path(csv_path)

    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build row data with same columns as training log
    row_data = {
        "timestamp": timestamp,
        "phase": "Evaluation",
        "model": model_name,
        "dataset": dataset_name,
        "n_bus": "",  # N/A for evaluation
        "n_gen": "",
        "params": "",
        "hidden_dim": "",
        "channels": "",
        "in_channels": "",
        "layers": "",
        "lr": "",
        "kappa": "",
        "weight_decay": "",
        "batch_size": "",
        "max_epochs": "",
        "warm_start": "",
        "best_loss": "",
        "duration": "",
        "duration_sec": "",
        # Evaluation metrics
        "R2_PG": _format_metric(metrics.get("R2_PG")),
        "R2_VG": _format_metric(metrics.get("R2_VG")),
        "Pacc_PG": _format_metric(metrics.get("Pacc_PG"), decimals=2),
        "Pacc_VG": _format_metric(metrics.get("Pacc_VG"), decimals=2),
        "RMSE_PG": _format_metric(metrics.get("RMSE_PG")),
        "RMSE_VG": _format_metric(metrics.get("RMSE_VG")),
        "MAE_PG": _format_metric(metrics.get("MAE_PG")),
        "MAE_VG": _format_metric(metrics.get("MAE_VG")),
        "Physics_Violation_MW": _format_metric(
            metrics.get("Physics_Violation_MW"), decimals=4
        ),
        # Log directory: use checkpoint parent dir if available
        "log_dir": str(Path(ckpt_path).parent.resolve()) if ckpt_path else "",
    }

    # Add n_samples to extra info if provided
    if n_samples is not None:
        row_data["n_samples"] = n_samples

    # Check if file exists to determine if header is needed
    file_exists = csv_path.exists()

    # If file exists, read existing headers to ensure compatibility
    if file_exists:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_headers = next(reader, None)

        # Add any missing columns from existing file
        if existing_headers:
            for header in existing_headers:
                if header not in row_data:
                    row_data[header] = ""

    # Write to CSV
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

    print(f"Evaluation logged to: {csv_path}")


def _format_metric(value: Any, decimals: int = 6) -> str:
    """Format a metric value for CSV output."""
    if value is None:
        return ""
    if isinstance(value, float):
        if decimals == 2:
            return f"{value:.2f}"
        elif decimals == 4:
            return f"{value:.4f}"
        return f"{value:.6f}"
    return str(value)


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
    header = (
        f"{'Phase':<12} {'Model':<8} {'Dataset':<10} "
        f"{'Loss/R2_PG':<12} {'Duration':<12} {'Timestamp':<20}"
    )
    print(header)
    print("-" * 80)

    # Print rows
    for r in records[-10:]:  # Show last 10
        phase = r.get("phase", "N/A")
        # Show best_loss for Training, R2_PG for Evaluation
        if phase == "Evaluation":
            metric = r.get("R2_PG", "N/A")
        else:
            metric = r.get("best_loss", "N/A")

        row = (
            f"{phase:<12} "
            f"{r.get('model', 'N/A'):<8} "
            f"{r.get('dataset', 'N/A'):<10} "
            f"{metric:<12} "
            f"{r.get('duration', 'N/A'):<12} "
            f"{r.get('timestamp', 'N/A'):<20}"
        )
        print(row)

    print(f"{'='*80}\n")
