#!/usr/bin/env python
"""Automated experiment runner for deep_opf models.

Chains: parameter counting → training → evaluation (seen + unseen) → CSV logging

Supports GCNN two-phase training (Phase 1: supervised, Phase 2: physics-informed).

Usage:
    # GCNN single phase
    python scripts/run_experiment.py gcnn case39 \\
        --channels 8 --n_layers 2 --fc_hidden_dim 256 --n_fc_layers 1 \\
        --batch_size 32 --lr 1e-3 --patience 20 --max_epochs 100

    # GCNN two-phase  
    python scripts/run_experiment.py gcnn case39 \\
        --channels 8 --n_layers 2 --fc_hidden_dim 256 --n_fc_layers 1 \\
        --two-phase --phase1_epochs 50 --phase2_epochs 100 --kappa 0.1

    # DNN
    python scripts/run_experiment.py dnn case39 \\
        --hidden_dim 128 --num_layers 3 \\
        --batch_size 64 --lr 1e-3 --max_epochs 100
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deep_opf.models import GCNN, AdmittanceDNN
from deep_opf.utils.experiment_logger import (
    ExperimentRow,
    count_parameters,
    log_dnn_experiment,
    log_gcnn_experiment,
)


# =============================================================================
# Configuration Dataclasses
# =============================================================================
@dataclass
class GCNNConfig:
    """GCNN experiment configuration."""

    # Dataset
    dataset: str = "case39"

    # Architecture (channels = in_channels = hidden_channels)
    channels: int = 8
    n_layers: int = 2
    fc_hidden_dim: int = 256
    n_fc_layers: int = 1

    # Training
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 20
    dropout: float = 0.0

    # Single phase training
    max_epochs: int = 100

    # Two-phase training
    two_phase: bool = False
    phase1_epochs: int = 50
    phase2_epochs: int = 100
    kappa: float = 0.1  # Physics loss weight for phase 2

    # GPU
    gpu: int = 0


@dataclass
class DNNConfig:
    """DNN experiment configuration."""

    # Dataset
    dataset: str = "case39"

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 3

    # Training
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 20
    dropout: float = 0.1
    max_epochs: int = 100

    # GPU
    gpu: int = 0


# Dataset info
DATASET_INFO = {
    "case39": {"n_bus": 39, "n_gen": 10, "config_name": "case39"},
    "case6ww": {"n_bus": 6, "n_gen": 3, "config_name": "case6"},
}


# =============================================================================
# Parameter Counting
# =============================================================================
def count_gcnn_params(cfg: GCNNConfig) -> int:
    """Instantiate GCNN and count parameters."""
    info = DATASET_INFO[cfg.dataset]
    model = GCNN(
        n_bus=info["n_bus"],
        n_gen=info["n_gen"],
        in_channels=cfg.channels,
        hidden_channels=cfg.channels,
        n_layers=cfg.n_layers,
        fc_hidden_dim=cfg.fc_hidden_dim,
        n_fc_layers=cfg.n_fc_layers,
        dropout=cfg.dropout,
    )
    return count_parameters(model)


def count_dnn_params(cfg: DNNConfig) -> int:
    """Instantiate DNN and count parameters."""
    info = DATASET_INFO[cfg.dataset]
    n_bus = info["n_bus"]
    input_dim = 2 * n_bus + 2 * (n_bus * n_bus)
    model = AdmittanceDNN(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        n_gen=info["n_gen"],
        n_bus=n_bus,
        dropout=cfg.dropout,
    )
    return count_parameters(model)


# =============================================================================
# Command Generation
# =============================================================================
def generate_gcnn_train_command(
    cfg: GCNNConfig,
    max_epochs: int,
    kappa: float,
    warm_start_ckpt: Optional[str] = None,
) -> str:
    """Generate Hydra command for GCNN training."""
    data_cfg = DATASET_INFO[cfg.dataset]["config_name"]

    cmd_parts = [
        "python",
        "scripts/train.py",
        "model=gcnn",
        f"data={data_cfg}",
        f"model.architecture.in_channels={cfg.channels}",
        f"model.architecture.hidden_channels={cfg.channels}",
        f"model.architecture.n_layers={cfg.n_layers}",
        f"model.architecture.fc_hidden_dim={cfg.fc_hidden_dim}",
        f"model.architecture.n_fc_layers={cfg.n_fc_layers}",
        f"model.architecture.dropout={cfg.dropout}",
        f"train.batch_size={cfg.batch_size}",
        f"train.lr={cfg.lr}",
        f"model.task.weight_decay={cfg.weight_decay}",
        f"train.patience={cfg.patience}",
        f"train.max_epochs={max_epochs}",
        f"model.task.kappa={kappa}",
    ]

    if warm_start_ckpt:
        cmd_parts.append(f'"+train.warm_start_ckpt={warm_start_ckpt}"')

    return " ".join(cmd_parts)


def generate_dnn_train_command(cfg: DNNConfig) -> str:
    """Generate Hydra command for DNN training."""
    data_cfg = DATASET_INFO[cfg.dataset]["config_name"]

    cmd_parts = [
        "python",
        "scripts/train.py",
        "model=dnn",
        f"data={data_cfg}",
        f"model.architecture.hidden_dim={cfg.hidden_dim}",
        f"model.architecture.num_layers={cfg.num_layers}",
        f"model.architecture.dropout={cfg.dropout}",
        f"train.batch_size={cfg.batch_size}",
        f"train.lr={cfg.lr}",
        f"model.task.weight_decay={cfg.weight_decay}",
        f"train.patience={cfg.patience}",
        f"train.max_epochs={cfg.max_epochs}",
    ]

    return " ".join(cmd_parts)


def generate_eval_command(
    model_type: str,
    dataset: str,
    ckpt_path: str,
    test_file: str = "samples_test.npz",
    # GCNN architecture params
    channels: int | None = None,
    n_layers: int | None = None,
    fc_hidden_dim: int | None = None,
    n_fc_layers: int | None = None,
    # DNN architecture params
    hidden_dim: int | None = None,
    num_layers: int | None = None,
) -> str:
    """Generate Hydra command for evaluation."""
    data_cfg = DATASET_INFO[dataset]["config_name"]

    # For Windows, wrap the ckpt_path with single quotes inside the Hydra override
    # The whole argument gets double quotes, inner path gets single quotes
    cmd_parts = [
        "python",
        "scripts/evaluate.py",
        f"model={model_type}",
        f"data={data_cfg}",
        f"\"+ckpt_path='{ckpt_path}'\"",
        f"data.test_file={test_file}",
    ]

    # Pass architecture params for GCNN
    if model_type == "gcnn":
        if channels is not None:
            cmd_parts.append(f"model.architecture.in_channels={channels}")
            cmd_parts.append(f"model.architecture.hidden_channels={channels}")
        if n_layers is not None:
            cmd_parts.append(f"model.architecture.n_layers={n_layers}")
        if fc_hidden_dim is not None:
            cmd_parts.append(f"model.architecture.fc_hidden_dim={fc_hidden_dim}")
        if n_fc_layers is not None:
            cmd_parts.append(f"model.architecture.n_fc_layers={n_fc_layers}")

    # Pass architecture params for DNN
    elif model_type == "dnn":
        if hidden_dim is not None:
            cmd_parts.append(f"model.architecture.hidden_dim={hidden_dim}")
        if num_layers is not None:
            cmd_parts.append(f"model.architecture.num_layers={num_layers}")

    return " ".join(cmd_parts)


# =============================================================================
# Subprocess Execution with Interrupt Handling
# =============================================================================
def run_command(cmd: str, gpu: int = 0) -> tuple[int, str]:
    """
    Run a command with GPU selection and interrupt handling.

    Returns:
        (return_code, best_checkpoint_path or empty string)
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"\n{'='*70}")
    print(f"Running: {cmd}")
    print(f"GPU: cuda:{gpu}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            env=env,
            capture_output=False,
            text=True,
        )
        return result.returncode, ""
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user.")
        print("Continuing to evaluation with available checkpoint...")
        return -1, ""


def find_latest_checkpoint(search_dirs: list[Path] | None = None) -> Optional[Path]:
    """Find the most recent checkpoint in outputs/ or lightning_logs/."""
    if search_dirs is None:
        search_dirs = [Path("outputs"), Path("lightning_logs")]

    all_ckpt_files = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Find all best checkpoint files (not last.ckpt)
        ckpt_files = [p for p in search_dir.rglob("*.ckpt") if "last" not in p.name]
        all_ckpt_files.extend(ckpt_files)

    if not all_ckpt_files:
        # Fall back to any checkpoint including last.ckpt
        for search_dir in search_dirs:
            if search_dir.exists():
                all_ckpt_files.extend(search_dir.rglob("*.ckpt"))

    if not all_ckpt_files:
        return None

    # Sort by modification time (most recent first)
    all_ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_ckpt_files[0]


def extract_best_loss_from_ckpt(ckpt_path: Path) -> float:
    """Extract val_loss from checkpoint filename."""
    # Pattern: epoch=X-val_loss=Y.ckpt
    name = ckpt_path.stem
    if "val_loss=" in name:
        try:
            loss_str = name.split("val_loss=")[1]
            return float(loss_str)
        except (IndexError, ValueError):
            pass
    return float("nan")


# =============================================================================
# Evaluation Runner
# =============================================================================
def run_evaluation(
    model_type: str,
    dataset: str,
    ckpt_path: str,
    test_file: str,
    gpu: int,
    # GCNN architecture
    channels: int | None = None,
    n_layers: int | None = None,
    fc_hidden_dim: int | None = None,
    n_fc_layers: int | None = None,
    # DNN architecture
    hidden_dim: int | None = None,
    num_layers: int | None = None,
) -> dict:
    """
    Run evaluation and return metrics.

    Returns:
        Dict with R2_PG, R2_VG, Pacc_PG, Pacc_VG, Physics_Violation_MW
    """
    cmd = generate_eval_command(
        model_type,
        dataset,
        ckpt_path,
        test_file,
        channels=channels,
        n_layers=n_layers,
        fc_hidden_dim=fc_hidden_dim,
        n_fc_layers=n_fc_layers,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"\n{'='*70}")
    print(f"Evaluating: {test_file}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")

    result = subprocess.run(
        cmd,
        shell=True,
        env=env,
        capture_output=True,
        text=True,
    )

    # Parse metrics from output
    metrics = {
        "R2_PG": 0.0,
        "R2_VG": 0.0,
        "Pacc_PG": 0.0,
        "Pacc_VG": 0.0,
        "Physics_MW": 0.0,
    }

    output = result.stdout + result.stderr

    # Debug: print output if something went wrong
    if result.returncode != 0:
        print(f"[WARN] Evaluation returned code {result.returncode}")
        print(output[:500] if output else "(no output)")

    # Parse simple key=value format from evaluate.py output
    lines = output.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("R2_PG="):
            try:
                metrics["R2_PG"] = float(line.split("=")[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("R2_VG="):
            try:
                metrics["R2_VG"] = float(line.split("=")[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("Pacc_PG="):
            try:
                metrics["Pacc_PG"] = float(line.split("=")[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("Pacc_VG="):
            try:
                metrics["Pacc_VG"] = float(line.split("=")[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("Physics_MW="):
            val = line.split("=")[1]
            if val != "N/A":
                try:
                    metrics["Physics_MW"] = float(val)
                except (ValueError, IndexError):
                    pass

    # Debug: print parsed metrics
    print(
        f"  Parsed: R2_PG={metrics['R2_PG']:.4f}, Pacc_PG={metrics['Pacc_PG']:.2f}%, Physics={metrics['Physics_MW']:.2f} MW"
    )

    return metrics


# =============================================================================
# Main Experiment Runners
# =============================================================================
def run_gcnn_experiment(cfg: GCNNConfig) -> None:
    """Run complete GCNN experiment pipeline."""

    # Count parameters
    n_params = count_gcnn_params(cfg)

    # Display configuration
    print("\n" + "=" * 70)
    print("               EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Model:      GCNN")
    print(f"Dataset:    {cfg.dataset} ({DATASET_INFO[cfg.dataset]['n_bus']} bus)")
    print(f"Parameters: {n_params:,}")
    print()
    print("Architecture:")
    print(
        f"  channels={cfg.channels}, n_layers={cfg.n_layers}, "
        f"fc_hidden_dim={cfg.fc_hidden_dim}, n_fc_layers={cfg.n_fc_layers}"
    )
    print()
    print("Training:")
    print(
        f"  batch_size={cfg.batch_size}, lr={cfg.lr}, "
        f"weight_decay={cfg.weight_decay}, patience={cfg.patience}, dropout={cfg.dropout}"
    )

    if cfg.two_phase:
        print(f"  Phase 1: max_epochs={cfg.phase1_epochs}, kappa=0.0")
        print(
            f"  Phase 2: max_epochs={cfg.phase2_epochs}, kappa={cfg.kappa} (warm-start)"
        )
    else:
        print(f"  max_epochs={cfg.max_epochs}, kappa={cfg.kappa}")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(cfg.gpu)
        print(f"\nGPU: cuda:{cfg.gpu} ({gpu_name})")
    else:
        print("\nGPU: CPU mode (CUDA not available)")

    print("=" * 70)

    # Confirm
    response = input("\nProceed? [Y/n]: ").strip().lower()
    if response not in ("", "y", "yes"):
        print("Aborted.")
        return

    # Generate commands
    if cfg.two_phase:
        # Phase 1
        print("\n" + "=" * 70)
        print("PHASE 1: Supervised Training (kappa=0.0)")
        print("=" * 70)

        cmd1 = generate_gcnn_train_command(cfg, cfg.phase1_epochs, kappa=0.0)
        print(f"Command: {cmd1}")

        start_time = time.time()
        ret_code, _ = run_command(cmd1, cfg.gpu)
        duration1 = time.time() - start_time

        # Find checkpoint
        ckpt_path1 = find_latest_checkpoint()
        if ckpt_path1 is None:
            print("[ERROR] No checkpoint found after Phase 1 training")
            return

        print(f"Phase 1 checkpoint: {ckpt_path1}")
        best_loss1 = extract_best_loss_from_ckpt(ckpt_path1)

        # Evaluate Phase 1
        metrics_seen1 = run_evaluation(
            "gcnn",
            cfg.dataset,
            str(ckpt_path1),
            "samples_test.npz",
            cfg.gpu,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )
        metrics_unseen1 = run_evaluation(
            "gcnn",
            cfg.dataset,
            str(ckpt_path1),
            "samples_unseen.npz",
            cfg.gpu,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )

        # Log Phase 1
        row1 = ExperimentRow("gcnn")
        row1.set_architecture(
            params=n_params,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )
        row1.set_training_config(
            dataset=cfg.dataset,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            dropout=cfg.dropout,
            max_epochs=cfg.phase1_epochs,
            kappa=0.0,
            phase=1,
        )
        row1.set_training_results(
            best_val_loss=best_loss1,
            duration_sec=duration1,
            ckpt_path=ckpt_path1,
            log_dir=ckpt_path1.parent.parent,
        )
        row1.set_eval_seen(
            r2_pg=metrics_seen1["R2_PG"],
            r2_vg=metrics_seen1["R2_VG"],
            pacc_pg=metrics_seen1["Pacc_PG"],
            pacc_vg=metrics_seen1["Pacc_VG"],
            physics_mw=metrics_seen1["Physics_MW"],
        )
        row1.set_eval_unseen(
            r2_pg=metrics_unseen1["R2_PG"],
            r2_vg=metrics_unseen1["R2_VG"],
            pacc_pg=metrics_unseen1["Pacc_PG"],
            pacc_vg=metrics_unseen1["Pacc_VG"],
            physics_mw=metrics_unseen1["Physics_MW"],
        )
        log_gcnn_experiment(row1)

        # Phase 2
        print("\n" + "=" * 70)
        print(f"PHASE 2: Physics-Informed Training (kappa={cfg.kappa})")
        print("=" * 70)

        cmd2 = generate_gcnn_train_command(
            cfg, cfg.phase2_epochs, kappa=cfg.kappa, warm_start_ckpt=str(ckpt_path1)
        )
        print(f"Command: {cmd2}")

        start_time = time.time()
        ret_code, _ = run_command(cmd2, cfg.gpu)
        duration2 = time.time() - start_time

        # Find Phase 2 checkpoint
        ckpt_path2 = find_latest_checkpoint()
        if ckpt_path2 is None:
            print("[ERROR] No checkpoint found after Phase 2 training")
            return

        print(f"Phase 2 checkpoint: {ckpt_path2}")
        best_loss2 = extract_best_loss_from_ckpt(ckpt_path2)

        # Evaluate Phase 2
        metrics_seen2 = run_evaluation(
            "gcnn",
            cfg.dataset,
            str(ckpt_path2),
            "samples_test.npz",
            cfg.gpu,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )
        metrics_unseen2 = run_evaluation(
            "gcnn",
            cfg.dataset,
            str(ckpt_path2),
            "samples_unseen.npz",
            cfg.gpu,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )

        # Log Phase 2
        row2 = ExperimentRow("gcnn")
        row2.set_architecture(
            params=n_params,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )
        row2.set_training_config(
            dataset=cfg.dataset,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            dropout=cfg.dropout,
            max_epochs=cfg.phase2_epochs,
            kappa=cfg.kappa,
            phase=2,
        )
        row2.set_training_results(
            best_val_loss=best_loss2,
            duration_sec=duration2,
            ckpt_path=ckpt_path2,
            log_dir=ckpt_path2.parent.parent,
        )
        row2.set_eval_seen(
            r2_pg=metrics_seen2["R2_PG"],
            r2_vg=metrics_seen2["R2_VG"],
            pacc_pg=metrics_seen2["Pacc_PG"],
            pacc_vg=metrics_seen2["Pacc_VG"],
            physics_mw=metrics_seen2["Physics_MW"],
        )
        row2.set_eval_unseen(
            r2_pg=metrics_unseen2["R2_PG"],
            r2_vg=metrics_unseen2["R2_VG"],
            pacc_pg=metrics_unseen2["Pacc_PG"],
            pacc_vg=metrics_unseen2["Pacc_VG"],
            physics_mw=metrics_unseen2["Physics_MW"],
        )
        log_gcnn_experiment(row2)

    else:
        # Single phase training
        cmd = generate_gcnn_train_command(cfg, cfg.max_epochs, kappa=cfg.kappa)
        print(f"\nCommand: {cmd}")

        start_time = time.time()
        ret_code, _ = run_command(cmd, cfg.gpu)
        duration = time.time() - start_time

        # Find checkpoint
        ckpt_path = find_latest_checkpoint()
        if ckpt_path is None:
            print("[ERROR] No checkpoint found after training")
            return

        print(f"Checkpoint: {ckpt_path}")
        best_loss = extract_best_loss_from_ckpt(ckpt_path)

        # Evaluate on seen dataset
        metrics_seen = run_evaluation(
            "gcnn",
            cfg.dataset,
            str(ckpt_path),
            "samples_test.npz",
            cfg.gpu,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )

        # Evaluate on unseen dataset
        metrics_unseen = run_evaluation(
            "gcnn",
            cfg.dataset,
            str(ckpt_path),
            "samples_unseen.npz",
            cfg.gpu,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )

        # Log experiment
        row = ExperimentRow("gcnn")
        row.set_architecture(
            params=n_params,
            channels=cfg.channels,
            n_layers=cfg.n_layers,
            fc_hidden_dim=cfg.fc_hidden_dim,
            n_fc_layers=cfg.n_fc_layers,
        )
        row.set_training_config(
            dataset=cfg.dataset,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            dropout=cfg.dropout,
            max_epochs=cfg.max_epochs,
            kappa=cfg.kappa,
            phase=1,
        )
        row.set_training_results(
            best_val_loss=best_loss,
            duration_sec=duration,
            ckpt_path=ckpt_path,
            log_dir=ckpt_path.parent.parent,
        )
        row.set_eval_seen(
            r2_pg=metrics_seen["R2_PG"],
            r2_vg=metrics_seen["R2_VG"],
            pacc_pg=metrics_seen["Pacc_PG"],
            pacc_vg=metrics_seen["Pacc_VG"],
            physics_mw=metrics_seen["Physics_MW"],
        )
        row.set_eval_unseen(
            r2_pg=metrics_unseen["R2_PG"],
            r2_vg=metrics_unseen["R2_VG"],
            pacc_pg=metrics_unseen["Pacc_PG"],
            pacc_vg=metrics_unseen["Pacc_VG"],
            physics_mw=metrics_unseen["Physics_MW"],
        )
        log_gcnn_experiment(row)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("Results logged to: outputs/gcnn_experiments.csv")
    print("=" * 70)


def run_dnn_experiment(cfg: DNNConfig) -> None:
    """Run complete DNN experiment pipeline."""

    # Count parameters
    n_params = count_dnn_params(cfg)

    # Display configuration
    print("\n" + "=" * 70)
    print("               EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Model:      DNN")
    print(f"Dataset:    {cfg.dataset} ({DATASET_INFO[cfg.dataset]['n_bus']} bus)")
    print(f"Parameters: {n_params:,}")
    print()
    print("Architecture:")
    print(f"  hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers}")
    print()
    print("Training:")
    print(
        f"  batch_size={cfg.batch_size}, lr={cfg.lr}, "
        f"weight_decay={cfg.weight_decay}, patience={cfg.patience}, dropout={cfg.dropout}"
    )
    print(f"  max_epochs={cfg.max_epochs}")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(cfg.gpu)
        print(f"\nGPU: cuda:{cfg.gpu} ({gpu_name})")
    else:
        print("\nGPU: CPU mode (CUDA not available)")

    print("=" * 70)

    # Confirm
    response = input("\nProceed? [Y/n]: ").strip().lower()
    if response not in ("", "y", "yes"):
        print("Aborted.")
        return

    # Generate and run command
    cmd = generate_dnn_train_command(cfg)
    print(f"\nCommand: {cmd}")

    start_time = time.time()
    ret_code, _ = run_command(cmd, cfg.gpu)
    duration = time.time() - start_time

    # Find checkpoint
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        print("[ERROR] No checkpoint found after training")
        return

    print(f"Checkpoint: {ckpt_path}")
    best_loss = extract_best_loss_from_ckpt(ckpt_path)

    # Evaluate on seen dataset
    metrics_seen = run_evaluation(
        "dnn",
        cfg.dataset,
        str(ckpt_path),
        "samples_test.npz",
        cfg.gpu,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )

    # Evaluate on unseen dataset (DNN may not have unseen data)
    try:
        metrics_unseen = run_evaluation(
            "dnn",
            cfg.dataset,
            str(ckpt_path),
            "samples_unseen.npz",
            cfg.gpu,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
        )
    except Exception:
        metrics_unseen = {
            "R2_PG": 0.0,
            "R2_VG": 0.0,
            "Pacc_PG": 0.0,
            "Pacc_VG": 0.0,
            "Physics_MW": 0.0,
        }

    # Log experiment
    row = ExperimentRow("dnn")
    row.set_architecture(
        params=n_params,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )
    row.set_training_config(
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        dropout=cfg.dropout,
        max_epochs=cfg.max_epochs,
    )
    row.set_training_results(
        best_val_loss=best_loss,
        duration_sec=duration,
        ckpt_path=ckpt_path,
        log_dir=ckpt_path.parent.parent,
    )
    row.set_eval_seen(
        r2_pg=metrics_seen["R2_PG"],
        r2_vg=metrics_seen["R2_VG"],
        pacc_pg=metrics_seen["Pacc_PG"],
        pacc_vg=metrics_seen["Pacc_VG"],
        physics_mw=metrics_seen["Physics_MW"],
    )
    row.set_eval_unseen(
        r2_pg=metrics_unseen["R2_PG"],
        r2_vg=metrics_unseen["R2_VG"],
        pacc_pg=metrics_unseen["Pacc_PG"],
        pacc_vg=metrics_unseen["Pacc_VG"],
        physics_mw=metrics_unseen["Physics_MW"],
    )
    log_dnn_experiment(row)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("Results logged to: outputs/dnn_experiments.csv")
    print("=" * 70)


# =============================================================================
# CLI Argument Parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated experiment runner for deep_opf models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GCNN single phase
  python scripts/run_experiment.py gcnn case39 --channels 8 --n_layers 2 --max_epochs 100

  # GCNN two-phase
  python scripts/run_experiment.py gcnn case39 --channels 8 --two-phase --phase1_epochs 50 --kappa 0.1

  # DNN
  python scripts/run_experiment.py dnn case39 --hidden_dim 128 --num_layers 3
        """,
    )

    parser.add_argument(
        "model",
        choices=["gcnn", "dnn"],
        help="Model type to train",
    )
    parser.add_argument(
        "dataset",
        choices=["case39", "case6ww"],
        help="Dataset to use",
    )

    # GPU selection
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )

    # Common training params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=100)

    # GCNN-specific params
    parser.add_argument(
        "--channels",
        type=int,
        default=8,
        help="GCNN: in_channels and hidden_channels (same value)",
    )
    parser.add_argument(
        "--n_layers", type=int, default=2, help="GCNN: number of GraphConv layers"
    )
    parser.add_argument(
        "--fc_hidden_dim", type=int, default=256, help="GCNN: FC trunk hidden dimension"
    )
    parser.add_argument(
        "--n_fc_layers", type=int, default=1, help="GCNN: number of FC layers"
    )

    # GCNN two-phase params
    parser.add_argument(
        "--two-phase", action="store_true", help="GCNN: Enable two-phase training"
    )
    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=50,
        help="GCNN two-phase: Phase 1 max epochs",
    )
    parser.add_argument(
        "--phase2_epochs",
        type=int,
        default=100,
        help="GCNN two-phase: Phase 2 max epochs",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.1,
        help="GCNN: Physics loss weight (0.0 for phase 1, custom for phase 2)",
    )

    # DNN-specific params
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="DNN: hidden layer dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="DNN: number of hidden layers"
    )

    # Dry run option
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and parameter count without running",
    )

    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    args = parse_args()

    if args.model == "gcnn":
        cfg = GCNNConfig(
            dataset=args.dataset,
            channels=args.channels,
            n_layers=args.n_layers,
            fc_hidden_dim=args.fc_hidden_dim,
            n_fc_layers=args.n_fc_layers,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            dropout=args.dropout,
            max_epochs=args.max_epochs,
            two_phase=args.two_phase,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            kappa=args.kappa,
            gpu=args.gpu,
        )
        if args.dry_run:
            dry_run_gcnn(cfg)
        else:
            run_gcnn_experiment(cfg)

    elif args.model == "dnn":
        cfg = DNNConfig(
            dataset=args.dataset,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            dropout=args.dropout,
            max_epochs=args.max_epochs,
            gpu=args.gpu,
        )
        if args.dry_run:
            dry_run_dnn(cfg)
        else:
            run_dnn_experiment(cfg)


def dry_run_gcnn(cfg: GCNNConfig) -> None:
    """Display GCNN configuration without running."""
    n_params = count_gcnn_params(cfg)

    print("\n" + "=" * 70)
    print("               EXPERIMENT CONFIGURATION (DRY RUN)")
    print("=" * 70)
    print(f"Model:      GCNN")
    print(f"Dataset:    {cfg.dataset} ({DATASET_INFO[cfg.dataset]['n_bus']} bus)")
    print(f"Parameters: {n_params:,}")
    print()
    print("Architecture:")
    print(
        f"  channels={cfg.channels}, n_layers={cfg.n_layers}, "
        f"fc_hidden_dim={cfg.fc_hidden_dim}, n_fc_layers={cfg.n_fc_layers}"
    )
    print()
    print("Training:")
    print(
        f"  batch_size={cfg.batch_size}, lr={cfg.lr}, "
        f"weight_decay={cfg.weight_decay}, patience={cfg.patience}, dropout={cfg.dropout}"
    )

    if cfg.two_phase:
        print(f"  Phase 1: max_epochs={cfg.phase1_epochs}, kappa=0.0")
        print(
            f"  Phase 2: max_epochs={cfg.phase2_epochs}, kappa={cfg.kappa} (warm-start)"
        )
        cmd1 = generate_gcnn_train_command(cfg, cfg.phase1_epochs, kappa=0.0)
        cmd2 = generate_gcnn_train_command(
            cfg, cfg.phase2_epochs, kappa=cfg.kappa, warm_start_ckpt="<phase1_best>"
        )
        print()
        print("Phase 1 Command:")
        print(f"  {cmd1}")
        print()
        print("Phase 2 Command:")
        print(f"  {cmd2}")
    else:
        print(f"  max_epochs={cfg.max_epochs}, kappa={cfg.kappa}")
        cmd = generate_gcnn_train_command(cfg, cfg.max_epochs, kappa=cfg.kappa)
        print()
        print("Command:")
        print(f"  {cmd}")

    print("=" * 70)


def dry_run_dnn(cfg: DNNConfig) -> None:
    """Display DNN configuration without running."""
    n_params = count_dnn_params(cfg)

    print("\n" + "=" * 70)
    print("               EXPERIMENT CONFIGURATION (DRY RUN)")
    print("=" * 70)
    print(f"Model:      DNN")
    print(f"Dataset:    {cfg.dataset} ({DATASET_INFO[cfg.dataset]['n_bus']} bus)")
    print(f"Parameters: {n_params:,}")
    print()
    print("Architecture:")
    print(f"  hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers}")
    print()
    print("Training:")
    print(
        f"  batch_size={cfg.batch_size}, lr={cfg.lr}, "
        f"weight_decay={cfg.weight_decay}, patience={cfg.patience}, dropout={cfg.dropout}"
    )
    print(f"  max_epochs={cfg.max_epochs}")

    cmd = generate_dnn_train_command(cfg)
    print()
    print("Command:")
    print(f"  {cmd}")
    print("=" * 70)


if __name__ == "__main__":
    main()
