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
import itertools
import os
import re
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
# Sweep Utilities
# =============================================================================
def parse_sweep_param(value_str: str, param_type: type = float) -> list:
    """
    Parse a potentially comma-separated parameter string into a list of values.

    Args:
        value_str: String like "8" or "8,16,32" or "8, 16, 32"
        param_type: Type to convert each value (int or float)

    Returns:
        List of parsed values

    Raises:
        ValueError: If any value cannot be parsed
    """
    # Split by comma and optional whitespace
    parts = re.split(r"\s*,\s*", value_str.strip())
    values = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            values.append(param_type(part))
        except ValueError:
            raise ValueError(f"Invalid {param_type.__name__} value: '{part}'")
    return values if values else [param_type(value_str)]


def is_sweep_value(value_str: str) -> bool:
    """Check if a parameter string contains multiple values (comma-separated)."""
    return "," in str(value_str)


def expand_combinations(params_dict: dict) -> list[dict]:
    """
    Expand a dict of potentially multi-value params into all combinations.

    Args:
        params_dict: {"channels": [8,16], "batch_size": [32,64]}

    Returns:
        [{"channels": 8, "batch_size": 32}, {"channels": 8, "batch_size": 64}, ...]
    """
    keys = list(params_dict.keys())
    value_lists = [
        params_dict[k] if isinstance(params_dict[k], list) else [params_dict[k]]
        for k in keys
    ]

    combinations = list(itertools.product(*value_lists))
    return [dict(zip(keys, combo)) for combo in combinations]


def format_sweep_progress(current: int, total: int, params: dict) -> str:
    """Format progress string for sweep mode."""
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"[{current}/{total}] {param_str}"


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

    # Phase 2 only mode (resume from existing checkpoint)
    phase2_only: bool = False
    warm_start_ckpt: str = ""

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
    dropout: float | None = None,
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
        if dropout is not None:
            cmd_parts.append(f"model.architecture.dropout={dropout}")

    return " ".join(cmd_parts)


# =============================================================================
# Subprocess Execution with Interrupt Handling
# =============================================================================
def get_latest_version_number(log_dir: Path = Path("lightning_logs")) -> int:
    """Get the highest version number in lightning_logs directory."""
    if not log_dir.exists():
        return -1

    version_dirs = list(log_dir.glob("version_*"))
    if not version_dirs:
        return -1

    version_nums = []
    for d in version_dirs:
        try:
            num = int(d.name.split("_")[1])
            version_nums.append(num)
        except (IndexError, ValueError):
            continue

    return max(version_nums) if version_nums else -1


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


def find_checkpoint_in_version(
    version_num: int, log_dir: Path = Path("lightning_logs")
) -> Optional[Path]:
    """
    Find the best checkpoint in a specific version directory.

    This is more robust than find_latest_checkpoint when running multiple
    training jobs in parallel.

    Args:
        version_num: The version number to look for
        log_dir: Base lightning_logs directory

    Returns:
        Path to checkpoint, or None if not found
    """
    version_dir = log_dir / f"version_{version_num}"
    if not version_dir.exists():
        return None

    ckpt_dir = version_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Find best checkpoint (not last.ckpt)
    ckpt_files = [p for p in ckpt_dir.glob("*.ckpt") if "last" not in p.name]

    if not ckpt_files:
        # Fall back to last.ckpt
        last_ckpt = ckpt_dir / "last.ckpt"
        return last_ckpt if last_ckpt.exists() else None

    # If multiple, sort by modification time (newest first)
    ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpt_files[0]


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
    dropout: float | None = None,
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
        dropout=dropout,
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

    if cfg.phase2_only:
        print(
            f"  Phase 2 Only: max_epochs={cfg.max_epochs}, kappa={cfg.kappa} (warm-start)"
        )
        print(f"  Warm start: {cfg.warm_start_ckpt}")
    elif cfg.two_phase:
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
    if cfg.phase2_only:
        # Phase 2 only - use provided checkpoint
        if not cfg.warm_start_ckpt:
            print("[ERROR] --warm_start_ckpt required for --phase2-only mode")
            return

        ckpt_path1 = Path(cfg.warm_start_ckpt)
        if not ckpt_path1.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_path1}")
            return

        print("\n" + "=" * 70)
        print(f"PHASE 2 ONLY: Physics-Informed Training (kappa={cfg.kappa})")
        print("=" * 70)
        print(f"Using checkpoint: {ckpt_path1}")

        cmd2 = generate_gcnn_train_command(
            cfg, cfg.max_epochs, kappa=cfg.kappa, warm_start_ckpt=str(ckpt_path1)
        )
        print(f"Command: {cmd2}")

        # Track version number before training
        version_before = get_latest_version_number()

        start_time = time.time()
        ret_code, _ = run_command(cmd2, cfg.gpu)
        duration2 = time.time() - start_time

        # Find Phase 2 checkpoint in expected version
        expected_version = version_before + 1
        ckpt_path2 = find_checkpoint_in_version(expected_version)
        if ckpt_path2 is None:
            print(
                f"[WARN] Checkpoint not found in version_{expected_version}, falling back"
            )
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

        # Log Phase 2 Only
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
            max_epochs=cfg.max_epochs,
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

    elif cfg.two_phase:
        # Phase 1
        print("\n" + "=" * 70)
        print("PHASE 1: Supervised Training (kappa=0.0)")
        print("=" * 70)

        cmd1 = generate_gcnn_train_command(cfg, cfg.phase1_epochs, kappa=0.0)
        print(f"Command: {cmd1}")

        # Track version number before training
        version_before_p1 = get_latest_version_number()

        start_time = time.time()
        ret_code, _ = run_command(cmd1, cfg.gpu)
        duration1 = time.time() - start_time

        # Find Phase 1 checkpoint in expected version
        expected_version_p1 = version_before_p1 + 1
        ckpt_path1 = find_checkpoint_in_version(expected_version_p1)
        if ckpt_path1 is None:
            print(
                f"[WARN] Checkpoint not found in version_{expected_version_p1}, falling back"
            )
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

        # Track version number before Phase 2 training
        version_before_p2 = get_latest_version_number()

        start_time = time.time()
        ret_code, _ = run_command(cmd2, cfg.gpu)
        duration2 = time.time() - start_time

        # Find Phase 2 checkpoint in expected version
        expected_version_p2 = version_before_p2 + 1
        ckpt_path2 = find_checkpoint_in_version(expected_version_p2)
        if ckpt_path2 is None:
            print(
                f"[WARN] Checkpoint not found in version_{expected_version_p2}, falling back"
            )
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

        # Track version number before training
        version_before = get_latest_version_number()

        start_time = time.time()
        ret_code, _ = run_command(cmd, cfg.gpu)
        duration = time.time() - start_time

        # Find checkpoint in the new version directory (version_before + 1)
        expected_version = version_before + 1
        ckpt_path = find_checkpoint_in_version(expected_version)

        # Fallback to global search if version-specific search fails
        if ckpt_path is None:
            print(
                f"[WARN] Checkpoint not found in version_{expected_version}, falling back to global search"
            )
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

    # Track version number before training
    version_before = get_latest_version_number()

    start_time = time.time()
    ret_code, _ = run_command(cmd, cfg.gpu)
    duration = time.time() - start_time

    # Find checkpoint in expected version
    expected_version = version_before + 1
    ckpt_path = find_checkpoint_in_version(expected_version)
    if ckpt_path is None:
        print(
            f"[WARN] Checkpoint not found in version_{expected_version}, falling back"
        )
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
        dropout=cfg.dropout,
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
            dropout=cfg.dropout,
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
# No-Confirm Versions for Sweep Mode
# =============================================================================
def run_gcnn_experiment_no_confirm(cfg: GCNNConfig) -> None:
    """Run GCNN experiment without confirmation prompt (for sweep mode)."""
    n_params = count_gcnn_params(cfg)

    print(f"Model: GCNN, Dataset: {cfg.dataset}, Params: {n_params:,}")
    print(
        f"  Architecture: channels={cfg.channels}, n_layers={cfg.n_layers}, "
        f"fc_hidden_dim={cfg.fc_hidden_dim}, n_fc_layers={cfg.n_fc_layers}"
    )
    print(
        f"  Training: batch_size={cfg.batch_size}, max_epochs={cfg.max_epochs}, kappa={cfg.kappa}"
    )

    if cfg.phase2_only:
        # Phase 2 only - use provided checkpoint
        if not cfg.warm_start_ckpt:
            print("[ERROR] --warm_start_ckpt required for --phase2-only mode")
            return

        ckpt_path1 = Path(cfg.warm_start_ckpt)
        if not ckpt_path1.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_path1}")
            return

        print(f"\n  Phase 2 Only: Using checkpoint {ckpt_path1}")
        cmd2 = generate_gcnn_train_command(
            cfg, cfg.max_epochs, kappa=cfg.kappa, warm_start_ckpt=str(ckpt_path1)
        )

        # Track version number before training
        version_before = get_latest_version_number()

        start_time = time.time()
        run_command(cmd2, cfg.gpu)
        duration2 = time.time() - start_time

        # Find checkpoint in expected version
        expected_version = version_before + 1
        ckpt_path2 = find_checkpoint_in_version(expected_version)
        if ckpt_path2 is None:
            print(
                f"[WARN] Checkpoint not found in version_{expected_version}, falling back"
            )
            ckpt_path2 = find_latest_checkpoint()
        if ckpt_path2 is None:
            print("[ERROR] No checkpoint found after Phase 2")
            return

        best_loss2 = extract_best_loss_from_ckpt(ckpt_path2)
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
            max_epochs=cfg.max_epochs,
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

    elif cfg.two_phase:
        # Phase 1
        print(f"\n  Phase 1: Supervised (kappa=0.0)")
        cmd1 = generate_gcnn_train_command(cfg, cfg.phase1_epochs, kappa=0.0)

        # Track version number before training
        version_before_p1 = get_latest_version_number()

        start_time = time.time()
        run_command(cmd1, cfg.gpu)
        duration1 = time.time() - start_time

        # Find checkpoint in expected version
        expected_version_p1 = version_before_p1 + 1
        ckpt_path1 = find_checkpoint_in_version(expected_version_p1)
        if ckpt_path1 is None:
            print(
                f"[WARN] Checkpoint not found in version_{expected_version_p1}, falling back"
            )
            ckpt_path1 = find_latest_checkpoint()
        if ckpt_path1 is None:
            print("[ERROR] No checkpoint found after Phase 1")
            return

        best_loss1 = extract_best_loss_from_ckpt(ckpt_path1)
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
        print(f"\n  Phase 2: Physics-Informed (kappa={cfg.kappa})")
        cmd2 = generate_gcnn_train_command(
            cfg, cfg.phase2_epochs, kappa=cfg.kappa, warm_start_ckpt=str(ckpt_path1)
        )

        # Track version number before Phase 2
        version_before_p2 = get_latest_version_number()

        start_time = time.time()
        run_command(cmd2, cfg.gpu)
        duration2 = time.time() - start_time

        # Find checkpoint in expected version
        expected_version_p2 = version_before_p2 + 1
        ckpt_path2 = find_checkpoint_in_version(expected_version_p2)
        if ckpt_path2 is None:
            print(
                f"[WARN] Checkpoint not found in version_{expected_version_p2}, falling back"
            )
            ckpt_path2 = find_latest_checkpoint()
        if ckpt_path2 is None:
            print("[ERROR] No checkpoint found after Phase 2")
            return

        best_loss2 = extract_best_loss_from_ckpt(ckpt_path2)
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
        # Single phase
        cmd = generate_gcnn_train_command(cfg, cfg.max_epochs, kappa=cfg.kappa)

        # Track version number before training
        version_before = get_latest_version_number()

        start_time = time.time()
        run_command(cmd, cfg.gpu)
        duration = time.time() - start_time

        # Find checkpoint in expected version
        expected_version = version_before + 1
        ckpt_path = find_checkpoint_in_version(expected_version)
        if ckpt_path is None:
            print(
                f"[WARN] Checkpoint not found in version_{expected_version}, falling back"
            )
            ckpt_path = find_latest_checkpoint()
        if ckpt_path is None:
            print("[ERROR] No checkpoint found")
            return

        best_loss = extract_best_loss_from_ckpt(ckpt_path)
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

    print("  ✓ Logged to CSV")


def run_dnn_experiment_no_confirm(cfg: DNNConfig) -> None:
    """Run DNN experiment without confirmation prompt (for sweep mode)."""
    n_params = count_dnn_params(cfg)

    print(f"Model: DNN, Dataset: {cfg.dataset}, Params: {n_params:,}")
    print(f"  Architecture: hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers}")
    print(f"  Training: batch_size={cfg.batch_size}, max_epochs={cfg.max_epochs}")

    cmd = generate_dnn_train_command(cfg)

    # Track version number before training
    version_before = get_latest_version_number()

    start_time = time.time()
    run_command(cmd, cfg.gpu)
    duration = time.time() - start_time

    # Find checkpoint in expected version
    expected_version = version_before + 1
    ckpt_path = find_checkpoint_in_version(expected_version)
    if ckpt_path is None:
        print(
            f"[WARN] Checkpoint not found in version_{expected_version}, falling back"
        )
        ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        print("[ERROR] No checkpoint found")
        return

    best_loss = extract_best_loss_from_ckpt(ckpt_path)
    metrics_seen = run_evaluation(
        "dnn",
        cfg.dataset,
        str(ckpt_path),
        "samples_test.npz",
        cfg.gpu,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    try:
        metrics_unseen = run_evaluation(
            "dnn",
            cfg.dataset,
            str(ckpt_path),
            "samples_unseen.npz",
            cfg.gpu,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
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
        params=n_params, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers
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

    print("  ✓ Logged to CSV")


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

    # Common training params (sweepable - accept strings)
    parser.add_argument(
        "--batch_size", type=str, default="32", help="Batch size (sweep: 32,64,128)"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--max_epochs", type=str, default="100", help="Max epochs (sweep: 50,100,200)"
    )

    # GCNN-specific params (sweepable - accept strings)
    parser.add_argument(
        "--channels",
        type=str,
        default="8",
        help="GCNN: in_channels and hidden_channels (sweep: 4,8,16)",
    )
    parser.add_argument(
        "--n_layers",
        type=str,
        default="2",
        help="GCNN: number of GraphConv layers (sweep: 1,2,3)",
    )
    parser.add_argument(
        "--fc_hidden_dim",
        type=str,
        default="256",
        help="GCNN: FC trunk hidden dimension (sweep: 128,256,512)",
    )
    parser.add_argument(
        "--n_fc_layers",
        type=str,
        default="1",
        help="GCNN: number of FC layers (sweep: 1,2)",
    )

    # GCNN two-phase params
    parser.add_argument(
        "--two-phase", action="store_true", help="GCNN: Enable two-phase training"
    )
    parser.add_argument(
        "--phase2-only",
        action="store_true",
        help="GCNN: Run only Phase 2 with warm start checkpoint",
    )
    parser.add_argument(
        "--warm_start_ckpt",
        type=str,
        default="",
        help="GCNN: Path to Phase 1 checkpoint for Phase 2 warm start",
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
        type=str,
        default="0.1",
        help="GCNN: Physics loss weight (sweep: 0.1,0.5,1.0)",
    )

    # DNN-specific params (sweepable - accept strings)
    parser.add_argument(
        "--hidden_dim",
        type=str,
        default="128",
        help="DNN: hidden layer dimension (sweep: 64,128,256)",
    )
    parser.add_argument(
        "--num_layers",
        type=str,
        default="3",
        help="DNN: number of hidden layers (sweep: 2,3,4)",
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
def check_sweep_mode(args) -> bool:
    """Check if any sweepable parameter contains multiple values."""
    sweepable = [
        args.channels,
        args.n_layers,
        args.fc_hidden_dim,
        args.n_fc_layers,
        args.batch_size,
        args.max_epochs,
        args.kappa,
        args.hidden_dim,
        args.num_layers,
    ]
    return any(is_sweep_value(str(v)) for v in sweepable)


def run_gcnn_sweep(args) -> None:
    """Run GCNN experiments with sweep over parameter combinations."""
    # Parse sweep parameters
    params_dict = {
        "channels": parse_sweep_param(args.channels, int),
        "n_layers": parse_sweep_param(args.n_layers, int),
        "fc_hidden_dim": parse_sweep_param(args.fc_hidden_dim, int),
        "n_fc_layers": parse_sweep_param(args.n_fc_layers, int),
        "batch_size": parse_sweep_param(args.batch_size, int),
        "max_epochs": parse_sweep_param(args.max_epochs, int),
        "kappa": parse_sweep_param(args.kappa, float),
    }

    combinations = expand_combinations(params_dict)
    n_runs = len(combinations)

    print("\n" + "=" * 70)
    print(f"🔄 SWEEP MODE: {n_runs} experiments")
    print("=" * 70)
    for i, combo in enumerate(combinations, 1):
        print(f"  [{i}] {combo}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for i, combo in enumerate(combinations, 1):
            cfg = GCNNConfig(
                dataset=args.dataset,
                channels=combo["channels"],
                n_layers=combo["n_layers"],
                fc_hidden_dim=combo["fc_hidden_dim"],
                n_fc_layers=combo["n_fc_layers"],
                batch_size=combo["batch_size"],
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                dropout=args.dropout,
                max_epochs=combo["max_epochs"],
                two_phase=args.two_phase,
                phase1_epochs=args.phase1_epochs,
                phase2_epochs=args.phase2_epochs,
                kappa=combo["kappa"],
                phase2_only=getattr(args, "phase2_only", False),
                warm_start_ckpt=getattr(args, "warm_start_ckpt", ""),
                gpu=args.gpu,
            )
            print(
                f"\n[{i}/{n_runs}] channels={combo['channels']}, n_layers={combo['n_layers']}, "
                f"fc_hidden_dim={combo['fc_hidden_dim']}, batch_size={combo['batch_size']}, kappa={combo['kappa']}"
            )
            n_params = count_gcnn_params(cfg)
            print(f"  Parameters: {n_params:,}")
        return

    # Confirm
    response = input(f"\nRun {n_runs} experiments? [Y/n]: ").strip().lower()
    if response not in ("", "y", "yes"):
        print("Aborted.")
        return

    # Run each combination
    for i, combo in enumerate(combinations, 1):
        print("\n" + "=" * 70)
        print(f"🔄 {format_sweep_progress(i, n_runs, combo)}")
        print("=" * 70)

        cfg = GCNNConfig(
            dataset=args.dataset,
            channels=combo["channels"],
            n_layers=combo["n_layers"],
            fc_hidden_dim=combo["fc_hidden_dim"],
            n_fc_layers=combo["n_fc_layers"],
            batch_size=combo["batch_size"],
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            dropout=args.dropout,
            max_epochs=combo["max_epochs"],
            two_phase=args.two_phase,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            kappa=combo["kappa"],
            phase2_only=getattr(args, "phase2_only", False),
            warm_start_ckpt=getattr(args, "warm_start_ckpt", ""),
            gpu=args.gpu,
        )

        # Run without confirmation prompt (already confirmed above)
        run_gcnn_experiment_no_confirm(cfg)

    print("\n" + "=" * 70)
    print(f"✅ SWEEP COMPLETE: {n_runs} experiments logged")
    print("=" * 70)


def run_dnn_sweep(args) -> None:
    """Run DNN experiments with sweep over parameter combinations."""
    # Parse sweep parameters
    params_dict = {
        "hidden_dim": parse_sweep_param(args.hidden_dim, int),
        "num_layers": parse_sweep_param(args.num_layers, int),
        "batch_size": parse_sweep_param(args.batch_size, int),
        "max_epochs": parse_sweep_param(args.max_epochs, int),
    }

    combinations = expand_combinations(params_dict)
    n_runs = len(combinations)

    print("\n" + "=" * 70)
    print(f"🔄 SWEEP MODE: {n_runs} experiments")
    print("=" * 70)
    for i, combo in enumerate(combinations, 1):
        print(f"  [{i}] {combo}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for i, combo in enumerate(combinations, 1):
            cfg = DNNConfig(
                dataset=args.dataset,
                hidden_dim=combo["hidden_dim"],
                num_layers=combo["num_layers"],
                batch_size=combo["batch_size"],
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                dropout=args.dropout,
                max_epochs=combo["max_epochs"],
                gpu=args.gpu,
            )
            print(
                f"\n[{i}/{n_runs}] hidden_dim={combo['hidden_dim']}, "
                f"num_layers={combo['num_layers']}, batch_size={combo['batch_size']}"
            )
            n_params = count_dnn_params(cfg)
            print(f"  Parameters: {n_params:,}")
        return

    # Confirm
    response = input(f"\nRun {n_runs} experiments? [Y/n]: ").strip().lower()
    if response not in ("", "y", "yes"):
        print("Aborted.")
        return

    # Run each combination
    for i, combo in enumerate(combinations, 1):
        print("\n" + "=" * 70)
        print(f"🔄 {format_sweep_progress(i, n_runs, combo)}")
        print("=" * 70)

        cfg = DNNConfig(
            dataset=args.dataset,
            hidden_dim=combo["hidden_dim"],
            num_layers=combo["num_layers"],
            batch_size=combo["batch_size"],
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            dropout=args.dropout,
            max_epochs=combo["max_epochs"],
            gpu=args.gpu,
        )

        # Run without confirmation prompt (already confirmed above)
        run_dnn_experiment_no_confirm(cfg)

    print("\n" + "=" * 70)
    print(f"✅ SWEEP COMPLETE: {n_runs} experiments logged")
    print("=" * 70)


def main():
    args = parse_args()

    # Check for sweep mode
    is_sweep = check_sweep_mode(args)

    if args.model == "gcnn":
        if is_sweep:
            run_gcnn_sweep(args)
        else:
            # Single run - parse single values
            cfg = GCNNConfig(
                dataset=args.dataset,
                channels=int(args.channels),
                n_layers=int(args.n_layers),
                fc_hidden_dim=int(args.fc_hidden_dim),
                n_fc_layers=int(args.n_fc_layers),
                batch_size=int(args.batch_size),
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                dropout=args.dropout,
                max_epochs=int(args.max_epochs),
                two_phase=args.two_phase,
                phase1_epochs=args.phase1_epochs,
                phase2_epochs=args.phase2_epochs,
                kappa=float(args.kappa),
                phase2_only=getattr(args, "phase2_only", False),
                warm_start_ckpt=getattr(args, "warm_start_ckpt", ""),
                gpu=args.gpu,
            )
            if args.dry_run:
                dry_run_gcnn(cfg)
            else:
                run_gcnn_experiment(cfg)

    elif args.model == "dnn":
        if is_sweep:
            run_dnn_sweep(args)
        else:
            # Single run - parse single values
            cfg = DNNConfig(
                dataset=args.dataset,
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.num_layers),
                batch_size=int(args.batch_size),
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                dropout=args.dropout,
                max_epochs=int(args.max_epochs),
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

    if cfg.phase2_only:
        print(
            f"  Phase 2 Only: max_epochs={cfg.max_epochs}, kappa={cfg.kappa} (warm-start)"
        )
        print(f"  Warm start: {cfg.warm_start_ckpt}")
        cmd = generate_gcnn_train_command(
            cfg, cfg.max_epochs, kappa=cfg.kappa, warm_start_ckpt=cfg.warm_start_ckpt
        )
        print()
        print("Command:")
        print(f"  {cmd}")
    elif cfg.two_phase:
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
