"""Backfill train set evaluation metrics for existing CSV rows.

This script re-evaluates all experiments on the training set (10k samples)
to populate the new train columns (R2_PG_train, R2_VG_train, etc.).
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
GCNN_CSV = PROJECT_ROOT / "outputs" / "gcnn_experiments.csv"
DNN_CSV = PROJECT_ROOT / "outputs" / "dnn_experiments.csv"
PYTHON = r"E:\DevTools\anaconda3\envs\opf311\python.exe"

# New train columns to add
TRAIN_COLUMNS = [
    "R2_PG_train",
    "R2_VG_train",
    "Pacc_PG_train",
    "Pacc_VG_train",
    "Physics_MW_train",
    "PG_Viol_Rate_train",
    "VG_Viol_Rate_train",
]


def run_evaluation(
    model_type: str,
    dataset: str,
    ckpt_path: str,
    test_file: str,
    **arch_params,
) -> dict:
    """Run evaluation and parse ALL metrics."""
    ckpt_path_obj = Path(ckpt_path)

    # Use last.ckpt if the path contains = (which breaks Hydra parsing)
    if "=" in ckpt_path and ckpt_path_obj.parent.exists():
        last_ckpt = ckpt_path_obj.parent / "last.ckpt"
        if last_ckpt.exists():
            print(f"  [INFO] Using last.ckpt instead of {ckpt_path_obj.name}")
            ckpt_path = str(last_ckpt)
            ckpt_path_obj = last_ckpt

    if not ckpt_path_obj.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return {}

    # Map dataset name to Hydra config name
    hydra_data_name = "case6" if dataset == "case6ww" else dataset

    cmd_parts = [
        PYTHON,
        "scripts/evaluate.py",
        f"model={model_type}",
        f"data={hydra_data_name}",
        f'"+ckpt_path={ckpt_path}"',
        f"data.test_file={test_file}",
    ]

    # Add architecture params
    if model_type == "gcnn":
        cmd_parts.append(f"++model.architecture.in_channels={arch_params.get('channels', 8)}")
        cmd_parts.append(f"++model.architecture.hidden_channels={arch_params.get('channels', 8)}")
        cmd_parts.append(f"++model.architecture.n_layers={arch_params.get('n_layers', 3)}")
        cmd_parts.append(f"++model.architecture.fc_hidden_dim={arch_params.get('fc_hidden_dim', 512)}")
        cmd_parts.append(f"++model.architecture.n_fc_layers={arch_params.get('n_fc_layers', 3)}")
    else:  # dnn
        cmd_parts.append(f"++model.architecture.hidden_dim={arch_params.get('hidden_dim', 128)}")
        cmd_parts.append(f"++model.architecture.num_layers={arch_params.get('num_layers', 3)}")
        cmd_parts.append(f"++model.architecture.dropout={arch_params.get('dropout', 0.0)}")

    cmd = " ".join(cmd_parts)

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    # Parse ALL metrics from output
    metrics = {}
    output = result.stdout + result.stderr

    for line in output.split("\n"):
        line = line.strip()
        # R2 metrics
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
        # Pacc metrics
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
        # Physics
        elif line.startswith("Physics_MW="):
            val = line.split("=")[1]
            if val != "N/A":
                try:
                    metrics["Physics_MW"] = float(val)
                except ValueError:
                    pass
        # Violations
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


def backfill_gcnn():
    """Backfill GCNN experiments with train metrics."""
    if not GCNN_CSV.exists():
        print("No GCNN CSV found")
        return

    df = pd.read_csv(GCNN_CSV)

    # Add new train columns if missing
    for col in TRAIN_COLUMNS:
        if col not in df.columns:
            df[col] = None

    updated = 0
    for idx, row in df.iterrows():
        ckpt_path = row.get("ckpt_path", "")
        if not ckpt_path or pd.isna(ckpt_path):
            continue

        # Skip if already has train data
        if pd.notna(row.get("R2_PG_train")) and row.get("R2_PG_train") != "":
            print(f"  [SKIP] Row {idx} already has train data")
            continue

        dataset = row.get("dataset", "case39")
        channels = int(row.get("channels", 8))
        n_layers = int(row.get("n_layers", 3))
        fc_hidden_dim = int(row.get("fc_hidden_dim", 512))
        n_fc_layers = int(row.get("n_fc_layers", 3))

        print(f"\n[{idx+1}/{len(df)}] Evaluating GCNN on train: {ckpt_path}")

        # Eval on train
        print("  Evaluating on samples_train.npz...")
        metrics_train = run_evaluation(
            "gcnn",
            dataset,
            str(PROJECT_ROOT / ckpt_path),
            "samples_train.npz",
            channels=channels,
            n_layers=n_layers,
            fc_hidden_dim=fc_hidden_dim,
            n_fc_layers=n_fc_layers,
        )

        if metrics_train:
            df.at[idx, "R2_PG_train"] = metrics_train.get("R2_PG", "")
            df.at[idx, "R2_VG_train"] = metrics_train.get("R2_VG", "")
            df.at[idx, "Pacc_PG_train"] = metrics_train.get("Pacc_PG", "")
            df.at[idx, "Pacc_VG_train"] = metrics_train.get("Pacc_VG", "")
            df.at[idx, "Physics_MW_train"] = metrics_train.get("Physics_MW", "")
            df.at[idx, "PG_Viol_Rate_train"] = metrics_train.get("PG_Violation_Rate", "")
            df.at[idx, "VG_Viol_Rate_train"] = metrics_train.get("VG_Violation_Rate", "")
            print(
                f"  Train: R2_PG={metrics_train.get('R2_PG', 'N/A'):.4f}, "
                f"Pacc_PG={metrics_train.get('Pacc_PG', 'N/A'):.2f}%, "
                f"Physics={metrics_train.get('Physics_MW', 'N/A'):.2f} MW"
            )
            updated += 1

    # Save updated CSV
    df.to_csv(GCNN_CSV, index=False)
    print(f"\nUpdated {updated} GCNN rows. Saved to {GCNN_CSV}")


def backfill_dnn():
    """Backfill DNN experiments with train metrics."""
    if not DNN_CSV.exists():
        print("No DNN CSV found")
        return

    df = pd.read_csv(DNN_CSV)

    # Add new train columns if missing
    for col in TRAIN_COLUMNS:
        if col not in df.columns:
            df[col] = None

    updated = 0
    for idx, row in df.iterrows():
        ckpt_path = row.get("ckpt_path", "")
        if not ckpt_path or pd.isna(ckpt_path):
            continue

        # Skip if already has train data
        if pd.notna(row.get("R2_PG_train")) and row.get("R2_PG_train") != "":
            print(f"  [SKIP] Row {idx} already has train data")
            continue

        dataset = row.get("dataset", "case39")
        hidden_dim = int(row.get("hidden_dim", 128))
        num_layers = int(row.get("num_layers", 3))
        dropout = float(row.get("dropout", 0.0))

        print(f"\n[{idx+1}/{len(df)}] Evaluating DNN on train: {ckpt_path}")

        # Eval on train
        print("  Evaluating on samples_train.npz...")
        metrics_train = run_evaluation(
            "dnn",
            dataset,
            str(PROJECT_ROOT / ckpt_path),
            "samples_train.npz",
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        if metrics_train:
            df.at[idx, "R2_PG_train"] = metrics_train.get("R2_PG", "")
            df.at[idx, "R2_VG_train"] = metrics_train.get("R2_VG", "")
            df.at[idx, "Pacc_PG_train"] = metrics_train.get("Pacc_PG", "")
            df.at[idx, "Pacc_VG_train"] = metrics_train.get("Pacc_VG", "")
            df.at[idx, "Physics_MW_train"] = metrics_train.get("Physics_MW", "")
            df.at[idx, "PG_Viol_Rate_train"] = metrics_train.get("PG_Violation_Rate", "")
            df.at[idx, "VG_Viol_Rate_train"] = metrics_train.get("VG_Violation_Rate", "")
            print(
                f"  Train: R2_PG={metrics_train.get('R2_PG', 'N/A'):.4f}, "
                f"Pacc_PG={metrics_train.get('Pacc_PG', 'N/A'):.2f}%, "
                f"Physics={metrics_train.get('Physics_MW', 'N/A'):.2f} MW"
            )
            updated += 1

    # Save updated CSV
    df.to_csv(DNN_CSV, index=False)
    print(f"\nUpdated {updated} DNN rows. Saved to {DNN_CSV}")


if __name__ == "__main__":
    print("=" * 70)
    print("Backfilling Train Set Evaluation Metrics")
    print("=" * 70)

    print("\n--- GCNN Experiments ---")
    backfill_gcnn()

    print("\n--- DNN Experiments ---")
    backfill_dnn()

    print("\nDone!")
