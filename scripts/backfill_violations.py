"""Backfill constraint violation metrics for existing CSV rows.

This script re-evaluates all experiments in the CSV files to compute
the new PG/VG violation rate metrics.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
GCNN_CSV = PROJECT_ROOT / "outputs" / "gcnn_experiments.csv"
DNN_CSV = PROJECT_ROOT / "outputs" / "dnn_experiments.csv"
PYTHON = r"E:\DevTools\anaconda3\envs\opf311\python.exe"


def run_evaluation(
    model_type: str,
    dataset: str,
    ckpt_path: str,
    test_file: str,
    **arch_params,
) -> dict:
    """Run evaluation and parse metrics."""
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
    # case6ww -> case6 (config file is case6.yaml)
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

    # Parse output
    metrics = {}
    output = result.stdout + result.stderr
    
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("PG_Violation_Rate="):
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
    """Backfill GCNN experiments."""
    if not GCNN_CSV.exists():
        print("No GCNN CSV found")
        return

    df = pd.read_csv(GCNN_CSV)
    
    # Add new columns if missing
    for col in ["PG_Viol_Rate_seen", "VG_Viol_Rate_seen", "PG_Viol_Rate_unseen", "VG_Viol_Rate_unseen"]:
        if col not in df.columns:
            df[col] = None

    updated = 0
    for idx, row in df.iterrows():
        ckpt_path = row.get("ckpt_path", "")
        if not ckpt_path or pd.isna(ckpt_path):
            continue

        # Skip if already has violation data
        if pd.notna(row.get("PG_Viol_Rate_seen")) and row.get("PG_Viol_Rate_seen") != "":
            print(f"  [SKIP] Row {idx} already has violation data")
            continue

        dataset = row.get("dataset", "case39")
        channels = int(row.get("channels", 8))
        n_layers = int(row.get("n_layers", 3))
        fc_hidden_dim = int(row.get("fc_hidden_dim", 512))
        n_fc_layers = int(row.get("n_fc_layers", 3))

        print(f"\n[{idx+1}/{len(df)}] Evaluating GCNN: {ckpt_path}")
        
        # Determine test files based on dataset
        if dataset == "case6ww":
            seen_file = "samples_test.npz"
            unseen_file = "samples_unseen.npz"
        else:
            seen_file = "samples_test.npz"
            unseen_file = "samples_unseen.npz"

        # Eval on seen
        print(f"  Evaluating on {seen_file}...")
        metrics_seen = run_evaluation(
            "gcnn", dataset, str(PROJECT_ROOT / ckpt_path),
            seen_file,
            channels=channels, n_layers=n_layers,
            fc_hidden_dim=fc_hidden_dim, n_fc_layers=n_fc_layers,
        )
        
        if metrics_seen:
            df.at[idx, "PG_Viol_Rate_seen"] = metrics_seen.get("PG_Violation_Rate", "")
            df.at[idx, "VG_Viol_Rate_seen"] = metrics_seen.get("VG_Violation_Rate", "")
            print(f"  Seen: PG_Viol={metrics_seen.get('PG_Violation_Rate', 'N/A')}%, VG_Viol={metrics_seen.get('VG_Violation_Rate', 'N/A')}%")

        # Eval on unseen
        print(f"  Evaluating on {unseen_file}...")
        metrics_unseen = run_evaluation(
            "gcnn", dataset, str(PROJECT_ROOT / ckpt_path),
            unseen_file,
            channels=channels, n_layers=n_layers,
            fc_hidden_dim=fc_hidden_dim, n_fc_layers=n_fc_layers,
        )
        
        if metrics_unseen:
            df.at[idx, "PG_Viol_Rate_unseen"] = metrics_unseen.get("PG_Violation_Rate", "")
            df.at[idx, "VG_Viol_Rate_unseen"] = metrics_unseen.get("VG_Violation_Rate", "")
            print(f"  Unseen: PG_Viol={metrics_unseen.get('PG_Violation_Rate', 'N/A')}%, VG_Viol={metrics_unseen.get('VG_Violation_Rate', 'N/A')}%")

        if metrics_seen or metrics_unseen:
            updated += 1

    # Save updated CSV
    df.to_csv(GCNN_CSV, index=False)
    print(f"\nUpdated {updated} GCNN rows. Saved to {GCNN_CSV}")


def backfill_dnn():
    """Backfill DNN experiments."""
    if not DNN_CSV.exists():
        print("No DNN CSV found")
        return

    df = pd.read_csv(DNN_CSV)
    
    # Add new columns if missing
    for col in ["PG_Viol_Rate_seen", "VG_Viol_Rate_seen", "PG_Viol_Rate_unseen", "VG_Viol_Rate_unseen"]:
        if col not in df.columns:
            df[col] = None

    updated = 0
    for idx, row in df.iterrows():
        ckpt_path = row.get("ckpt_path", "")
        if not ckpt_path or pd.isna(ckpt_path):
            continue

        # Skip if already has violation data
        if pd.notna(row.get("PG_Viol_Rate_seen")) and row.get("PG_Viol_Rate_seen") != "":
            print(f"  [SKIP] Row {idx} already has violation data")
            continue

        dataset = row.get("dataset", "case39")
        hidden_dim = int(row.get("hidden_dim", 128))
        num_layers = int(row.get("num_layers", 3))
        dropout = float(row.get("dropout", 0.0))

        print(f"\n[{idx+1}/{len(df)}] Evaluating DNN: {ckpt_path}")
        
        seen_file = "samples_test.npz"
        unseen_file = "samples_unseen.npz"

        # Eval on seen
        print(f"  Evaluating on {seen_file}...")
        metrics_seen = run_evaluation(
            "dnn", dataset, str(PROJECT_ROOT / ckpt_path),
            seen_file,
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
        )
        
        if metrics_seen:
            df.at[idx, "PG_Viol_Rate_seen"] = metrics_seen.get("PG_Violation_Rate", "")
            df.at[idx, "VG_Viol_Rate_seen"] = metrics_seen.get("VG_Violation_Rate", "")
            print(f"  Seen: PG_Viol={metrics_seen.get('PG_Violation_Rate', 'N/A')}%, VG_Viol={metrics_seen.get('VG_Violation_Rate', 'N/A')}%")

        # Eval on unseen
        print(f"  Evaluating on {unseen_file}...")
        metrics_unseen = run_evaluation(
            "dnn", dataset, str(PROJECT_ROOT / ckpt_path),
            unseen_file,
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
        )
        
        if metrics_unseen:
            df.at[idx, "PG_Viol_Rate_unseen"] = metrics_unseen.get("PG_Violation_Rate", "")
            df.at[idx, "VG_Viol_Rate_unseen"] = metrics_unseen.get("VG_Violation_Rate", "")
            print(f"  Unseen: PG_Viol={metrics_unseen.get('PG_Violation_Rate', 'N/A')}%, VG_Viol={metrics_unseen.get('VG_Violation_Rate', 'N/A')}%")

        if metrics_seen or metrics_unseen:
            updated += 1

    # Save updated CSV
    df.to_csv(DNN_CSV, index=False)
    print(f"\nUpdated {updated} DNN rows. Saved to {DNN_CSV}")


if __name__ == "__main__":
    print("=" * 70)
    print("Backfilling Constraint Violation Metrics")
    print("=" * 70)
    
    print("\n--- GCNN Experiments ---")
    backfill_gcnn()
    
    print("\n--- DNN Experiments ---")
    backfill_dnn()
    
    print("\nDone!")
