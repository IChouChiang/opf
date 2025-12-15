#!/usr/bin/env python
"""Visualize and analyze experiment results from experiments_log.csv.

Features:
- Filter by phase (Training/Evaluation), model, dataset
- Sort by any metric
- Highlight best results
- Export to HTML with styling
- Summary statistics
- Compare models side-by-side

Usage:
    # Show all evaluation results
    python scripts/show_results.py

    # Filter by model
    python scripts/show_results.py --model gcnn

    # Filter by dataset
    python scripts/show_results.py --dataset case39

    # Show training results
    python scripts/show_results.py --phase Training

    # Sort by R² (descending)
    python scripts/show_results.py --sort R2_PG --desc

    # Export to HTML
    python scripts/show_results.py --html results.html

    # Show summary statistics
    python scripts/show_results.py --summary

    # Compare models
    python scripts/show_results.py --compare
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# =============================================================================
# Constants
# =============================================================================
CSV_PATH = Path(__file__).parent.parent / "experiments_log.csv"

# Column groups for different views
EVAL_COLS = [
    "timestamp",
    "model",
    "dataset",
    "n_samples",
    "R2_PG",
    "R2_VG",
    "Pacc_PG",
    "Pacc_VG",
    "RMSE_PG",
    "RMSE_VG",
    "Physics_Violation_MW",
    "ckpt_path",
]

TRAIN_COLS = [
    "timestamp",
    "model",
    "dataset",
    "params",
    "hidden_dim",
    "channels",
    "in_channels",
    "layers",
    "lr",
    "kappa",
    "max_epochs",
    "warm_start",
    "best_loss",
    "duration",
]

COMPACT_EVAL_COLS = [
    "model",
    "dataset",
    "n_samples",
    "R2_PG",
    "R2_VG",
    "Pacc_PG",
    "Pacc_VG",
    "Physics_Violation_MW",
]

COMPARE_COLS = [
    "model",
    "R2_PG",
    "R2_VG",
    "Pacc_PG",
    "Pacc_VG",
    "Physics_Violation_MW",
]


# =============================================================================
# Data Loading & Cleaning
# =============================================================================
def load_experiments(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load experiments CSV and clean up column names.

    Handles multiple CSV formats from different logger versions.
    """
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    # Read raw lines to handle inconsistent columns
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return pd.DataFrame()

    # Check if new format (has 'phase' column)
    header = lines[0].strip().split(",")
    has_phase_col = "phase" in header

    all_rows = []

    for i, line in enumerate(lines[1:], 1):
        values = line.strip().split(",")

        if has_phase_col:
            # New format - parse directly
            row = dict(zip(header, values))
        else:
            # Old format - detect row type
            # Old Training rows: timestamp,Training,model,dataset,...
            # Old Evaluation rows: timestamp,Evaluation,model,dataset,...,R2_PG,R2_VG,...
            if len(values) > 1 and values[1] in ["Training", "Evaluation"]:
                phase = values[1]
                # Shift values: timestamp, phase, model, dataset, ...
                row = {
                    "timestamp": values[0],
                    "phase": phase,
                    "model": values[2] if len(values) > 2 else "",
                    "dataset": values[3] if len(values) > 3 else "",
                }

                if phase == "Training":
                    # Old training row format
                    row.update(
                        {
                            "n_bus": values[4] if len(values) > 4 else "",
                            "n_gen": values[5] if len(values) > 5 else "",
                            "params": values[6] if len(values) > 6 else "",
                            "hidden_dim": values[7] if len(values) > 7 else "",
                            "channels": values[8] if len(values) > 8 else "",
                            "in_channels": values[9] if len(values) > 9 else "",
                            "layers": values[10] if len(values) > 10 else "",
                            "lr": values[11] if len(values) > 11 else "",
                            "kappa": values[12] if len(values) > 12 else "",
                            "weight_decay": values[13] if len(values) > 13 else "",
                            "batch_size": values[14] if len(values) > 14 else "",
                            "max_epochs": values[15] if len(values) > 15 else "",
                            "warm_start": values[16] if len(values) > 16 else "",
                            "best_loss": values[17] if len(values) > 17 else "",
                            "duration": values[18] if len(values) > 18 else "",
                            "duration_sec": values[19] if len(values) > 19 else "",
                            "log_dir": values[-1] if len(values) > 20 else "",
                        }
                    )
                else:
                    # Old evaluation row format
                    # timestamp,Evaluation,model,dataset,<16 empty>,R2_PG,R2_VG,Pacc_PG,Pacc_VG,RMSE_PG,RMSE_VG,MAE_PG,MAE_VG,Physics,ckpt_path,n_samples
                    # Actual positions: R2_PG at [20], R2_VG at [21], etc.
                    row.update(
                        {
                            "R2_PG": values[20] if len(values) > 20 else "",
                            "R2_VG": values[21] if len(values) > 21 else "",
                            "Pacc_PG": values[22] if len(values) > 22 else "",
                            "Pacc_VG": values[23] if len(values) > 23 else "",
                            "RMSE_PG": values[24] if len(values) > 24 else "",
                            "RMSE_VG": values[25] if len(values) > 25 else "",
                            "MAE_PG": values[26] if len(values) > 26 else "",
                            "MAE_VG": values[27] if len(values) > 27 else "",
                            "Physics_Violation_MW": (
                                values[28] if len(values) > 28 else ""
                            ),
                            "ckpt_path": values[29] if len(values) > 29 else "",
                            "n_samples": values[30] if len(values) > 30 else "",
                        }
                    )
            else:
                # Very old format without phase marker - assume Training
                row = dict(zip(header, values))
                row["phase"] = "Training"

        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Convert numeric columns
    numeric_cols = [
        "n_bus",
        "n_gen",
        "params",
        "hidden_dim",
        "channels",
        "in_channels",
        "layers",
        "lr",
        "kappa",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "best_loss",
        "duration_sec",
        "R2_PG",
        "R2_VG",
        "Pacc_PG",
        "Pacc_VG",
        "RMSE_PG",
        "RMSE_VG",
        "MAE_PG",
        "MAE_VG",
        "Physics_Violation_MW",
        "n_samples",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect phase from first column pattern
    # Old format: model column contains phase (Training/Evaluation)
    # New format: separate phase column
    if (
        "phase" not in df.columns
        and df["model"].str.contains("Training|Evaluation", na=False).any()
    ):
        # Old format - extract phase from model column
        mask_train = df["model"] == "Training"
        mask_eval = df["model"] == "Evaluation"

        # For Training rows, model is in 'dataset' column
        # For Evaluation rows, model is in 'dataset' column
        df.loc[mask_train, "phase"] = "Training"
        df.loc[mask_eval, "phase"] = "Evaluation"

        # Shift columns for old format rows
        df.loc[mask_train | mask_eval, "model"] = df.loc[
            mask_train | mask_eval, "dataset"
        ]
        df.loc[mask_train | mask_eval, "dataset"] = df.loc[
            mask_train | mask_eval, "n_bus"
        ]

    # Fill missing phase
    if "phase" not in df.columns:
        df["phase"] = "Training"  # Default for old format without phase marker

    return df


def filter_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for evaluation rows only."""
    if "phase" in df.columns:
        return df[df["phase"] == "Evaluation"].copy()
    return df


def filter_training(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for training rows only."""
    if "phase" in df.columns:
        return df[df["phase"] == "Training"].copy()
    return df


# =============================================================================
# Formatting
# =============================================================================
def format_dataframe(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Format numeric columns to specified decimal places."""
    df = df.copy()

    # Round numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns
    for col in numeric_cols:
        df[col] = df[col].round(decimals)

    # Format specific columns
    if "Pacc_PG" in df.columns:
        df["Pacc_PG"] = df["Pacc_PG"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else ""
        )
    if "Pacc_VG" in df.columns:
        df["Pacc_VG"] = df["Pacc_VG"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else ""
        )
    if "Physics_Violation_MW" in df.columns:
        df["Physics_Violation_MW"] = df["Physics_Violation_MW"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )

    # Shorten timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m-%d %H:%M")

    # Shorten checkpoint path
    if "ckpt_path" in df.columns:
        df["ckpt_path"] = df["ckpt_path"].apply(
            lambda x: Path(x).parent.name if pd.notna(x) and x else ""
        )

    # Format params with K/M suffix
    if "params" in df.columns:
        df["params"] = df["params"].apply(format_params)

    return df


def format_params(x) -> str:
    """Format parameter count with K/M suffix."""
    if pd.isna(x) or x == 0:
        return ""
    x = int(x)
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    elif x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return str(x)


def highlight_best(
    df: pd.DataFrame, col: str, higher_better: bool = True
) -> pd.DataFrame:
    """Add marker to best value in column."""
    if col not in df.columns or df[col].isna().all():
        return df

    df = df.copy()

    # Get numeric values, handling both numeric and string types
    if df[col].dtype == object:
        # Handle formatted strings like "34.27%" or empty strings
        numeric_vals = pd.to_numeric(
            df[col].str.replace("%", "").str.replace("★ ", ""), errors="coerce"
        )
    else:
        numeric_vals = df[col]

    # Skip if all NaN
    if numeric_vals.isna().all():
        return df

    if higher_better:
        best_idx = numeric_vals.idxmax()
    else:
        best_idx = numeric_vals.idxmin()

    # Skip if best is NaN
    if pd.isna(best_idx) or pd.isna(numeric_vals[best_idx]):
        return df

    # Convert column to string for marking
    df[col] = df[col].astype(str)
    df.loc[best_idx, col] = f"★ {df.loc[best_idx, col]}"
    return df


# =============================================================================
# Display Functions
# =============================================================================
def print_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "",
    tablefmt: str = "grid",
) -> None:
    """Print DataFrame as formatted table."""
    if df.empty:
        print("[INFO] No data to display")
        return

    if columns:
        # Filter to existing columns only
        columns = [c for c in columns if c in df.columns]
        df = df[columns]

    if title:
        print(f"\n{'=' * 70}")
        print(f" {title}")
        print(f"{'=' * 70}")

    print(tabulate(df, headers="keys", tablefmt=tablefmt, showindex=False))
    print()


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for evaluation results."""
    print("\n" + "=" * 70)
    print(" SUMMARY STATISTICS")
    print("=" * 70)

    # Group by model
    if "model" in df.columns:
        print("\n[By Model]")
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            print(f"\n  {model.upper()}:")
            print(f"    Experiments: {len(model_df)}")

            for metric in [
                "R2_PG",
                "R2_VG",
                "Pacc_PG",
                "Pacc_VG",
                "Physics_Violation_MW",
            ]:
                if metric in model_df.columns:
                    vals = model_df[metric].dropna()
                    if len(vals) > 0:
                        # Handle string formatted values
                        if vals.dtype == object:
                            vals = (
                                vals.str.replace("%", "")
                                .str.replace("★ ", "")
                                .astype(float)
                            )
                        print(
                            f"    {metric}: mean={vals.mean():.4f}, best={vals.max():.4f}"
                        )

    # Best overall results
    print("\n[Best Results]")
    metrics_higher = ["R2_PG", "R2_VG", "Pacc_PG", "Pacc_VG"]
    metrics_lower = ["RMSE_PG", "RMSE_VG", "Physics_Violation_MW"]

    for metric in metrics_higher:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                if vals.dtype == object:
                    vals = vals.str.replace("%", "").str.replace("★ ", "").astype(float)
                best_idx = vals.idxmax()
                best_val = vals.max()
                best_model = df.loc[best_idx, "model"] if "model" in df.columns else "?"
                print(f"  {metric}: {best_val:.4f} ({best_model})")

    for metric in metrics_lower:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                if vals.dtype == object:
                    vals = vals.str.replace("★ ", "").astype(float)
                best_idx = vals.idxmin()
                best_val = vals.min()
                best_model = df.loc[best_idx, "model"] if "model" in df.columns else "?"
                print(f"  {metric}: {best_val:.4f} ({best_model})")


def compare_models(df: pd.DataFrame) -> None:
    """Compare models side by side."""
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON")
    print("=" * 70)

    if "model" not in df.columns:
        print("[ERROR] No model column found")
        return

    models = df["model"].unique()
    if len(models) < 2:
        print("[INFO] Need at least 2 models to compare")
        return

    # Get best result per model (highest R2_PG)
    comparison_data = []
    for model in models:
        model_df = df[df["model"] == model]
        if "R2_PG" in model_df.columns:
            # Convert R2_PG to numeric if needed
            r2_vals = model_df["R2_PG"]
            if r2_vals.dtype == object:
                r2_vals = r2_vals.str.replace("★ ", "").astype(float)
            best_idx = r2_vals.idxmax()
            best_row = model_df.loc[best_idx].copy()
            comparison_data.append(best_row)

    if comparison_data:
        compare_df = pd.DataFrame(comparison_data)
        cols = [c for c in COMPARE_COLS if c in compare_df.columns]
        print_table(compare_df, columns=cols, title="Best Result Per Model")


# =============================================================================
# HTML Export
# =============================================================================
def export_html(
    df: pd.DataFrame, output_path: Path, title: str = "Experiment Results"
) -> None:
    """Export DataFrame to styled HTML file."""
    # CSS styling
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 8px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .best {
            background-color: #e8f5e9 !important;
            font-weight: bold;
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
        }
        .metric-good {
            color: #2e7d32;
        }
        .metric-bad {
            color: #c62828;
        }
        .footer {
            margin-top: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
    """

    # Convert DataFrame to HTML
    html_table = df.to_html(index=False, classes="results-table", escape=False)

    # Build full HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        {css}
    </head>
    <body>
        <h1>{title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total experiments: {len(df)}</p>
        {html_table}
        <p class="footer">
            Metrics: R² (higher=better), Pacc (higher=better), RMSE/Physics (lower=better)
        </p>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")
    print(f"[OK] HTML exported to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize experiment results from experiments_log.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Filters
    parser.add_argument(
        "--phase",
        choices=["Training", "Evaluation", "all"],
        default="Evaluation",
        help="Filter by phase (default: Evaluation)",
    )
    parser.add_argument(
        "--model", type=str, help="Filter by model name (e.g., gcnn, dnn)"
    )
    parser.add_argument("--dataset", type=str, help="Filter by dataset (e.g., case39)")
    parser.add_argument(
        "--n-samples", type=int, help="Filter by number of test samples"
    )

    # Sorting
    parser.add_argument("--sort", type=str, help="Sort by column name")
    parser.add_argument(
        "--desc", action="store_true", help="Sort descending (default: ascending)"
    )
    parser.add_argument("--asc", action="store_true", help="Sort ascending")

    # Display options
    parser.add_argument("--compact", action="store_true", help="Show compact view")
    parser.add_argument("--full", action="store_true", help="Show all columns")
    parser.add_argument("--last", type=int, default=0, help="Show only last N results")

    # Analysis
    parser.add_argument(
        "--summary", action="store_true", help="Show summary statistics"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare models side-by-side"
    )
    parser.add_argument("--best", action="store_true", help="Highlight best values")

    # Export
    parser.add_argument("--html", type=str, help="Export to HTML file")
    parser.add_argument("--csv", type=str, help="Export filtered results to CSV")

    # Input
    parser.add_argument(
        "--input",
        type=str,
        default=str(CSV_PATH),
        help=f"Input CSV file (default: {CSV_PATH})",
    )

    args = parser.parse_args()

    # Load data
    df = load_experiments(Path(args.input))

    if df.empty:
        print("[ERROR] No data found")
        return

    # Filter by phase
    if args.phase == "Evaluation":
        df = filter_evaluations(df)
        columns = COMPACT_EVAL_COLS if args.compact else EVAL_COLS
        title = "EVALUATION RESULTS"
    elif args.phase == "Training":
        df = filter_training(df)
        columns = TRAIN_COLS
        title = "TRAINING RESULTS"
    else:
        columns = None
        title = "ALL EXPERIMENTS"

    # Apply filters
    if args.model:
        df = df[df["model"].str.lower() == args.model.lower()]
    if args.dataset:
        df = df[df["dataset"].str.lower() == args.dataset.lower()]
    if args.n_samples:
        df = df[df["n_samples"] == args.n_samples]

    if df.empty:
        print("[INFO] No results match the filters")
        return

    # Sort
    if args.sort:
        if args.sort in df.columns:
            ascending = not args.desc if not args.asc else True
            df = df.sort_values(args.sort, ascending=ascending)
        else:
            print(f"[WARN] Sort column '{args.sort}' not found")

    # Limit results
    if args.last > 0:
        df = df.tail(args.last)

    # Format
    df = format_dataframe(df)

    # Highlight best
    if args.best:
        df = highlight_best(df, "R2_PG", higher_better=True)
        df = highlight_best(df, "R2_VG", higher_better=True)
        df = highlight_best(df, "Physics_Violation_MW", higher_better=False)

    # Show full columns if requested
    if args.full:
        columns = None

    # Print results
    print_table(df, columns=columns, title=title)

    # Summary
    if args.summary:
        print_summary(df)

    # Compare
    if args.compare:
        compare_models(df)

    # Export HTML
    if args.html:
        export_html(df, Path(args.html), title=title)

    # Export CSV
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"[OK] CSV exported to: {args.csv}")


if __name__ == "__main__":
    main()
