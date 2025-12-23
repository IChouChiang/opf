"""Experiment logging utilities for automated experiment pipeline.

This module provides model-specific CSV logging with unified schemas for
GCNN and DNN experiments, combining training and evaluation results.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch.nn as nn


# GCNN CSV Schema - includes train, seen and unseen evaluation metrics in one row
GCNN_CSV_COLUMNS = [
    # Metadata
    "timestamp",
    "dataset",
    "phase",  # 1 or 2 for two-phase training
    "params",
    # Architecture (display columns)
    "channels",  # in_channels = hidden_channels
    "n_layers",
    "fc_hidden_dim",
    "n_fc_layers",
    # Training hyperparameters
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "dropout",
    "max_epochs",
    "kappa",
    # Training results
    "best_val_loss",
    "duration_sec",
    # Evaluation on TRAIN dataset (10000 samples)
    "R2_PG_train",
    "R2_VG_train",
    "Pacc_PG_train",
    "Pacc_VG_train",
    "Physics_MW_train",
    "PG_Viol_Rate_train",
    "VG_Viol_Rate_train",
    # Evaluation on SEEN dataset (2000 samples)
    "R2_PG_seen",
    "R2_VG_seen",
    "Pacc_PG_seen",
    "Pacc_VG_seen",
    "Physics_MW_seen",
    "PG_Viol_Rate_seen",
    "VG_Viol_Rate_seen",
    # Evaluation on UNSEEN dataset (1200 samples)
    "R2_PG_unseen",
    "R2_VG_unseen",
    "Pacc_PG_unseen",
    "Pacc_VG_unseen",
    "Physics_MW_unseen",
    "PG_Viol_Rate_unseen",
    "VG_Viol_Rate_unseen",
    # Paths
    "ckpt_path",
    "log_dir",
]

# DNN CSV Schema
DNN_CSV_COLUMNS = [
    # Metadata
    "timestamp",
    "dataset",
    "params",
    # Architecture (display columns)
    "hidden_dim",
    "num_layers",
    # Training hyperparameters
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "dropout",
    "max_epochs",
    # Training results
    "best_val_loss",
    "duration_sec",
    # Evaluation on TRAIN dataset (10000 samples)
    "R2_PG_train",
    "R2_VG_train",
    "Pacc_PG_train",
    "Pacc_VG_train",
    "Physics_MW_train",
    "PG_Viol_Rate_train",
    "VG_Viol_Rate_train",
    # Evaluation on SEEN dataset (2000 samples)
    "R2_PG_seen",
    "R2_VG_seen",
    "Pacc_PG_seen",
    "Pacc_VG_seen",
    "Physics_MW_seen",
    "PG_Viol_Rate_seen",
    "VG_Viol_Rate_seen",
    # Evaluation on UNSEEN dataset (1200 samples)
    "R2_PG_unseen",
    "R2_VG_unseen",
    "Pacc_PG_unseen",
    "Pacc_VG_unseen",
    "Physics_MW_unseen",
    "PG_Viol_Rate_unseen",
    "VG_Viol_Rate_unseen",
    # Paths
    "ckpt_path",
    "log_dir",
]


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _format_metric(value: Any, decimals: int = 4) -> str:
    """Format a metric value for CSV output."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


class ExperimentRow:
    """Helper class to build an experiment result row."""

    def __init__(self, model_type: str = "gcnn"):
        """
        Initialize experiment row.

        Args:
            model_type: 'gcnn' or 'dnn'
        """
        if model_type not in ("gcnn", "dnn"):
            raise ValueError(f"model_type must be 'gcnn' or 'dnn', got {model_type}")

        self.model_type = model_type
        self.columns = GCNN_CSV_COLUMNS if model_type == "gcnn" else DNN_CSV_COLUMNS
        self.data: dict[str, Any] = {col: "" for col in self.columns}
        self.data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_architecture(
        self,
        params: int,
        # GCNN specific
        channels: Optional[int] = None,
        n_layers: Optional[int] = None,
        fc_hidden_dim: Optional[int] = None,
        n_fc_layers: Optional[int] = None,
        # DNN specific
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> "ExperimentRow":
        """Set architecture parameters."""
        self.data["params"] = params

        if self.model_type == "gcnn":
            self.data["channels"] = channels or ""
            self.data["n_layers"] = n_layers or ""
            self.data["fc_hidden_dim"] = fc_hidden_dim or ""
            self.data["n_fc_layers"] = n_fc_layers or ""
        else:  # dnn
            self.data["hidden_dim"] = hidden_dim or ""
            self.data["num_layers"] = num_layers or ""

        return self

    def set_training_config(
        self,
        dataset: str,
        batch_size: int,
        lr: float,
        weight_decay: float,
        patience: int,
        dropout: float,
        max_epochs: int,
        kappa: Optional[float] = None,  # GCNN only
        phase: Optional[int] = None,  # GCNN only
    ) -> "ExperimentRow":
        """Set training hyperparameters."""
        self.data["dataset"] = dataset
        self.data["batch_size"] = batch_size
        self.data["lr"] = lr
        self.data["weight_decay"] = weight_decay
        self.data["patience"] = patience
        self.data["dropout"] = dropout
        self.data["max_epochs"] = max_epochs

        if self.model_type == "gcnn":
            self.data["kappa"] = kappa if kappa is not None else ""
            self.data["phase"] = phase if phase is not None else ""

        return self

    def set_training_results(
        self,
        best_val_loss: float,
        duration_sec: float,
        ckpt_path: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
    ) -> "ExperimentRow":
        """Set training results."""
        self.data["best_val_loss"] = _format_metric(best_val_loss, decimals=6)
        self.data["duration_sec"] = _format_metric(duration_sec, decimals=1)
        self.data["ckpt_path"] = str(ckpt_path) if ckpt_path else ""
        self.data["log_dir"] = str(log_dir) if log_dir else ""
        return self

    def set_eval_train(
        self,
        r2_pg: float,
        r2_vg: float,
        pacc_pg: float,
        pacc_vg: float,
        physics_mw: float,
        pg_viol_rate: Optional[float] = None,
        vg_viol_rate: Optional[float] = None,
    ) -> "ExperimentRow":
        """Set evaluation results on train dataset (10000 samples)."""
        self.data["R2_PG_train"] = _format_metric(r2_pg, decimals=4)
        self.data["R2_VG_train"] = _format_metric(r2_vg, decimals=4)
        self.data["Pacc_PG_train"] = _format_metric(pacc_pg, decimals=2)
        self.data["Pacc_VG_train"] = _format_metric(pacc_vg, decimals=2)
        self.data["Physics_MW_train"] = _format_metric(physics_mw, decimals=2)
        self.data["PG_Viol_Rate_train"] = (
            _format_metric(pg_viol_rate, decimals=2) if pg_viol_rate is not None else ""
        )
        self.data["VG_Viol_Rate_train"] = (
            _format_metric(vg_viol_rate, decimals=2) if vg_viol_rate is not None else ""
        )
        return self

    def set_eval_seen(
        self,
        r2_pg: float,
        r2_vg: float,
        pacc_pg: float,
        pacc_vg: float,
        physics_mw: float,
        pg_viol_rate: Optional[float] = None,
        vg_viol_rate: Optional[float] = None,
    ) -> "ExperimentRow":
        """Set evaluation results on seen (test) dataset."""
        self.data["R2_PG_seen"] = _format_metric(r2_pg, decimals=4)
        self.data["R2_VG_seen"] = _format_metric(r2_vg, decimals=4)
        self.data["Pacc_PG_seen"] = _format_metric(pacc_pg, decimals=2)
        self.data["Pacc_VG_seen"] = _format_metric(pacc_vg, decimals=2)
        self.data["Physics_MW_seen"] = _format_metric(physics_mw, decimals=2)
        self.data["PG_Viol_Rate_seen"] = (
            _format_metric(pg_viol_rate, decimals=2) if pg_viol_rate is not None else ""
        )
        self.data["VG_Viol_Rate_seen"] = (
            _format_metric(vg_viol_rate, decimals=2) if vg_viol_rate is not None else ""
        )
        return self

    def set_eval_unseen(
        self,
        r2_pg: float,
        r2_vg: float,
        pacc_pg: float,
        pacc_vg: float,
        physics_mw: float,
        pg_viol_rate: Optional[float] = None,
        vg_viol_rate: Optional[float] = None,
    ) -> "ExperimentRow":
        """Set evaluation results on unseen dataset."""
        self.data["R2_PG_unseen"] = _format_metric(r2_pg, decimals=4)
        self.data["R2_VG_unseen"] = _format_metric(r2_vg, decimals=4)
        self.data["Pacc_PG_unseen"] = _format_metric(pacc_pg, decimals=2)
        self.data["Pacc_VG_unseen"] = _format_metric(pacc_vg, decimals=2)
        self.data["Physics_MW_unseen"] = _format_metric(physics_mw, decimals=2)
        self.data["PG_Viol_Rate_unseen"] = (
            _format_metric(pg_viol_rate, decimals=2) if pg_viol_rate is not None else ""
        )
        self.data["VG_Viol_Rate_unseen"] = (
            _format_metric(vg_viol_rate, decimals=2) if vg_viol_rate is not None else ""
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return row data as dict."""
        return self.data.copy()


def log_gcnn_experiment(
    row: ExperimentRow,
    csv_path: Union[str, Path] = "outputs/gcnn_experiments.csv",
) -> None:
    """
    Log a GCNN experiment to CSV.

    Args:
        row: ExperimentRow with GCNN data
        csv_path: Path to GCNN experiments CSV
    """
    if row.model_type != "gcnn":
        raise ValueError("Expected GCNN experiment row")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GCNN_CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row.to_dict())

    print(f"GCNN experiment logged to: {csv_path}")


def log_dnn_experiment(
    row: ExperimentRow,
    csv_path: Union[str, Path] = "outputs/dnn_experiments.csv",
) -> None:
    """
    Log a DNN experiment to CSV.

    Args:
        row: ExperimentRow with DNN data
        csv_path: Path to DNN experiments CSV
    """
    if row.model_type != "dnn":
        raise ValueError("Expected DNN experiment row")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DNN_CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row.to_dict())

    print(f"DNN experiment logged to: {csv_path}")


def load_gcnn_experiments(
    csv_path: Union[str, Path] = "outputs/gcnn_experiments.csv",
) -> list[dict]:
    """Load all GCNN experiments from CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_dnn_experiments(
    csv_path: Union[str, Path] = "outputs/dnn_experiments.csv",
) -> list[dict]:
    """Load all DNN experiments from CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
