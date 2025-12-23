"""Recover and re-log experiments v72-v75 that were lost during CSV cleanup.

These experiments had their checkpoints preserved but CSV rows were corrupted.
This script re-evaluates them on train/seen/unseen and properly logs to CSV.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
GCNN_CSV = PROJECT_ROOT / "outputs" / "gcnn_experiments.csv"
PYTHON = r"E:\DevTools\anaconda3\envs\opf311\python.exe"

# Experiments to recover with their version numbers
EXPERIMENTS_TO_RECOVER = [72, 73, 74, 75]


def get_architecture_from_checkpoint(version: int) -> dict:
    """Extract architecture params from checkpoint state_dict."""
    ckpt_dir = PROJECT_ROOT / "lightning_logs" / f"version_{version}" / "checkpoints"

    # Find best checkpoint
    ckpts = list(ckpt_dir.glob("epoch*.ckpt"))
    if not ckpts:
        return {}

    ckpt_path = ckpts[0]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # State dict is nested under 'state_dict' key in Lightning checkpoints
    state_dict = ckpt.get("state_dict", ckpt)

    # Extract from gc_layers (graph conv layers)
    gc_layers = [k for k in state_dict.keys() if "gc_layers" in k]
    gc_indices = set()
    for k in gc_layers:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "gc_layers" and i + 1 < len(parts):
                try:
                    gc_indices.add(int(parts[i + 1]))
                except ValueError:
                    pass
    n_layers = len(gc_indices)

    # Get hidden channels from first gc layer W1 weight
    hidden_ch = 8
    for k, v in state_dict.items():
        if "gc_layers.0.W1.weight" in k:
            hidden_ch = v.shape[0]
            break

    # Get FC layers
    fc_layers = [k for k in state_dict.keys() if "fc_layers" in k]
    fc_indices = set()
    for k in fc_layers:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "fc_layers" and i + 1 < len(parts):
                try:
                    fc_indices.add(int(parts[i + 1]))
                except ValueError:
                    pass
    n_fc_layers = len(fc_indices)

    # Get FC hidden dim from first fc layer
    fc_hidden_dim = 512
    for k, v in state_dict.items():
        if "fc_layers.0.weight" in k:
            fc_hidden_dim = v.shape[0]
            break

    return {
        "channels": hidden_ch,
        "n_layers": n_layers,
        "fc_hidden_dim": fc_hidden_dim,
        "n_fc_layers": n_fc_layers,
        "ckpt_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
    }


def get_hparams(version: int) -> dict:
    """Load hyperparameters from hparams.yaml."""
    hparams_path = (
        PROJECT_ROOT / "lightning_logs" / f"version_{version}" / "hparams.yaml"
    )
    if hparams_path.exists():
        with open(hparams_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def run_evaluation(ckpt_path: str, test_file: str, arch: dict) -> dict:
    """Run evaluation and parse metrics."""
    cmd_parts = [
        PYTHON,
        "scripts/evaluate.py",
        "model=gcnn",
        "data=case39",
        f'"+ckpt_path={ckpt_path}"',
        f"data.test_file={test_file}",
        f"++model.architecture.in_channels={arch['channels']}",
        f"++model.architecture.hidden_channels={arch['channels']}",
        f"++model.architecture.n_layers={arch['n_layers']}",
        f"++model.architecture.fc_hidden_dim={arch['fc_hidden_dim']}",
        f"++model.architecture.n_fc_layers={arch['n_fc_layers']}",
    ]

    cmd = " ".join(cmd_parts)
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    metrics = {}
    output = result.stdout + result.stderr

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("R2_PG="):
            try:
                metrics["R2_PG"] = float(line.split("=")[1])
            except ValueError:
                pass
        elif line.startswith("R2_VG="):
            try:
                metrics["R2_VG"] = float(line.split("=")[1])
            except ValueError:
                pass
        elif line.startswith("Pacc_PG="):
            try:
                metrics["Pacc_PG"] = float(line.split("=")[1])
            except ValueError:
                pass
        elif line.startswith("Pacc_VG="):
            try:
                metrics["Pacc_VG"] = float(line.split("=")[1])
            except ValueError:
                pass
        elif line.startswith("Physics_MW="):
            val = line.split("=")[1]
            if val != "N/A":
                try:
                    metrics["Physics_MW"] = float(val)
                except ValueError:
                    pass
        elif line.startswith("PG_Violation_Rate="):
            val = line.split("=")[1].rstrip("%")
            try:
                metrics["PG_Violation_Rate"] = float(val)
            except ValueError:
                pass
        elif line.startswith("VG_Violation_Rate="):
            val = line.split("=")[1].rstrip("%")
            try:
                metrics["VG_Violation_Rate"] = float(val)
            except ValueError:
                pass

    return metrics


def recover_experiment(version: int) -> dict | None:
    """Recover a single experiment by re-evaluating on all test sets."""
    print(f"\n{'='*60}")
    print(f"Recovering version_{version}")
    print("=" * 60)

    arch = get_architecture_from_checkpoint(version)
    if not arch:
        print(f"  [ERROR] Could not extract architecture from v{version}")
        return None

    hparams = get_hparams(version)
    ckpt_path = str(PROJECT_ROOT / arch["ckpt_path"])

    print(
        f"  Architecture: channels={arch['channels']}, n_layers={arch['n_layers']}, "
        f"fc_hidden_dim={arch['fc_hidden_dim']}, n_fc_layers={arch['n_fc_layers']}"
    )
    print(f"  Checkpoint: {arch['ckpt_path']}")

    # Evaluate on all test sets
    print("\n  Evaluating on samples_train.npz...")
    train_metrics = run_evaluation(ckpt_path, "samples_train.npz", arch)
    print(
        f"    R2_PG={train_metrics.get('R2_PG', 'N/A')}, Pacc_PG={train_metrics.get('Pacc_PG', 'N/A')}%"
    )

    print("  Evaluating on samples_test.npz (seen)...")
    seen_metrics = run_evaluation(ckpt_path, "samples_test.npz", arch)
    print(
        f"    R2_PG={seen_metrics.get('R2_PG', 'N/A')}, Pacc_PG={seen_metrics.get('Pacc_PG', 'N/A')}%"
    )

    print("  Evaluating on samples_unseen.npz...")
    unseen_metrics = run_evaluation(ckpt_path, "samples_unseen.npz", arch)
    print(
        f"    R2_PG={unseen_metrics.get('R2_PG', 'N/A')}, Pacc_PG={unseen_metrics.get('Pacc_PG', 'N/A')}%"
    )

    # Build row data (matching GCNN_CSV_COLUMNS order)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "case39",
        "model_type": "GCNN",
        "channels": arch["channels"],
        "n_layers": arch["n_layers"],
        "fc_hidden_dim": arch["fc_hidden_dim"],
        "n_fc_layers": arch["n_fc_layers"],
        "lr": hparams.get("lr", 0.001),
        "kappa": hparams.get("kappa", 0.0),
        "two_phase": hparams.get("two_phase", False),
        "epochs": (
            int(arch["ckpt_path"].split("epoch")[1].split("-")[0])
            if "epoch" in arch["ckpt_path"]
            else 0
        ),
        "patience": hparams.get("patience", 100),
        "lr_scheduler": hparams.get("lr_scheduler", "none"),
        # Train metrics
        "R2_PG_train": train_metrics.get("R2_PG", ""),
        "R2_VG_train": train_metrics.get("R2_VG", ""),
        "Pacc_PG_train": train_metrics.get("Pacc_PG", ""),
        "Pacc_VG_train": train_metrics.get("Pacc_VG", ""),
        "Physics_MW_train": train_metrics.get("Physics_MW", ""),
        "PG_Viol_Rate_train": train_metrics.get("PG_Violation_Rate", ""),
        "VG_Viol_Rate_train": train_metrics.get("VG_Violation_Rate", ""),
        # Seen metrics
        "R2_PG_seen": seen_metrics.get("R2_PG", ""),
        "R2_VG_seen": seen_metrics.get("R2_VG", ""),
        "Pacc_PG_seen": seen_metrics.get("Pacc_PG", ""),
        "Pacc_VG_seen": seen_metrics.get("Pacc_VG", ""),
        "Physics_MW_seen": seen_metrics.get("Physics_MW", ""),
        "PG_Viol_Rate_seen": seen_metrics.get("PG_Violation_Rate", ""),
        "VG_Viol_Rate_seen": seen_metrics.get("VG_Violation_Rate", ""),
        # Unseen metrics
        "R2_PG_unseen": unseen_metrics.get("R2_PG", ""),
        "R2_VG_unseen": unseen_metrics.get("R2_VG", ""),
        "Pacc_PG_unseen": unseen_metrics.get("Pacc_PG", ""),
        "Pacc_VG_unseen": unseen_metrics.get("Pacc_VG", ""),
        "Physics_MW_unseen": unseen_metrics.get("Physics_MW", ""),
        "PG_Viol_Rate_unseen": unseen_metrics.get("PG_Violation_Rate", ""),
        "VG_Viol_Rate_unseen": unseen_metrics.get("VG_Violation_Rate", ""),
        # Checkpoint path
        "ckpt_path": arch["ckpt_path"],
        "notes": f"Recovered from v{version}",
    }

    return row


def main():
    print("=" * 70)
    print("Recovering Lost Experiments (v72-v75)")
    print("=" * 70)

    recovered_rows = []
    for version in EXPERIMENTS_TO_RECOVER:
        row = recover_experiment(version)
        if row:
            recovered_rows.append(row)

    if not recovered_rows:
        print("\nNo experiments recovered.")
        return

    # Load existing CSV and append
    if GCNN_CSV.exists():
        df = pd.read_csv(GCNN_CSV)
    else:
        df = pd.DataFrame()

    # Append new rows
    new_df = pd.DataFrame(recovered_rows)
    df = pd.concat([df, new_df], ignore_index=True)

    # Save
    df.to_csv(GCNN_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Recovered {len(recovered_rows)} experiments")
    print(f"Saved to: {GCNN_CSV}")
    print("=" * 70)

    # Print summary
    print("\nRecovered Experiments Summary:")
    print("-" * 70)
    for row in recovered_rows:
        print(
            f"  v{row['notes'].split('v')[1]}: "
            f"R2_PG_seen={row.get('R2_PG_seen', 'N/A')}, "
            f"Pacc_PG_seen={row.get('Pacc_PG_seen', 'N/A')}%, "
            f"lr_scheduler={row.get('lr_scheduler', 'none')}"
        )


if __name__ == "__main__":
    main()
