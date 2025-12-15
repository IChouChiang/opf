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
    """Load experiments CSV with proper column handling.

    The CSV has a consistent format:
    - Header with all columns
    - Training rows: have architecture params, no metrics
    - Evaluation rows: have metrics, no architecture params
    """
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    # Read CSV with pandas - handles all parsing
    df = pd.read_csv(csv_path, dtype=str)  # Read all as string first

    if df.empty:
        return df

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

    # Convert boolean columns to proper strings
    if "warm_start" in df.columns:
        df["warm_start"] = df["warm_start"].apply(
            lambda x: (
                "Yes"
                if str(x).lower() == "true"
                else ("No" if str(x).lower() == "false" else "")
            )
        )

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
    """Export DataFrame to interactive HTML file with filtering and sorting.

    Uses DataTables.js for client-side interactivity:
    - Column sorting (click headers)
    - Global search
    - Per-column filters
    - Pagination
    - Export buttons (CSV, Excel, PDF)
    """
    import json

    # Prepare data for DataTables
    df_display = df.copy()

    # Drop columns that are completely empty
    empty_cols = [
        col
        for col in df_display.columns
        if df_display[col].replace("", pd.NA).isna().all()
    ]
    df_display = df_display.drop(columns=empty_cols)

    # Convert DataFrame to JSON for JavaScript
    columns_json = json.dumps(
        [{"data": i, "title": col} for i, col in enumerate(df_display.columns)]
    )
    data_json = json.dumps(df_display.fillna("").values.tolist())
    col_names_json = json.dumps(list(df_display.columns))

    # Build table headers
    header_row = "".join(f"<th>{col}</th>" for col in df_display.columns)
    filter_row = "".join(
        '<th><input type="text" class="column-filter" placeholder="Filter..." /></th>'
        for _ in df_display.columns
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
    
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px 40px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #2e7d32, #4CAF50);
            color: white;
            padding: 20px 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 1.8em; }}
        .header p {{ margin: 5px 0; opacity: 0.9; }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        table.dataTable {{ width: 100% !important; border-collapse: collapse; }}
        table.dataTable thead th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px 8px;
            border-bottom: 2px solid #2e7d32;
            white-space: nowrap;
        }}
        table.dataTable tbody td {{
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
            white-space: nowrap;
        }}
        table.dataTable tbody tr:hover {{ background-color: #e8f5e9 !important; }}
        table.dataTable tfoot th {{ padding: 8px 4px; }}
        .best-high {{
            background-color: #c8e6c9 !important;
            font-weight: bold;
            color: #2e7d32;
        }}
        .best-low {{
            background-color: #e3f2fd !important;
            font-weight: bold;
            color: #1565c0;
        }}
        .column-filter {{
            width: 100%;
            padding: 4px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .column-filter:focus {{ outline: none; border-color: #4CAF50; }}
        .stats-row {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            min-width: 150px;
        }}
        .stat-card .label {{ font-size: 0.85em; color: #666; margin-bottom: 5px; }}
        .stat-card .value {{ font-size: 1.5em; font-weight: bold; color: #2e7d32; }}
        .dt-buttons {{ margin-bottom: 15px; }}
        .dt-button {{
            background: #4CAF50 !important;
            border: none !important;
            color: white !important;
            padding: 8px 16px !important;
            border-radius: 4px !important;
            margin-right: 5px !important;
        }}
        .dt-button:hover {{ background: #2e7d32 !important; }}
        .dataTables_filter input {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-left: 10px;
        }}
        .dataTables_filter input:focus {{ outline: none; border-color: #4CAF50; }}
        .footer {{
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 8px;
            color: #666;
            font-size: 0.9em;
        }}
        .legend {{ display: flex; gap: 20px; margin-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-box {{ width: 20px; height: 20px; border-radius: 4px; }}
        .legend-box.high {{ background-color: #c8e6c9; }}
        .legend-box.low {{ background-color: #e3f2fd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 {title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total experiments: {len(df)}</p>
    </div>
    
    <div class="stats-row" id="statsRow"></div>
    
    <div class="container">
        <table id="resultsTable" class="display nowrap" style="width:100%">
            <thead>
                <tr>{header_row}</tr>
            </thead>
            <tbody></tbody>
            <tfoot>
                <tr>{filter_row}</tr>
            </tfoot>
        </table>
    </div>
    
    <div class="footer">
        <strong>Metrics Guide:</strong>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-box high"></div>
                <span>R², Pacc: Higher is better</span>
            </div>
            <div class="legend-item">
                <div class="legend-box low"></div>
                <span>RMSE, Physics Violation: Lower is better</span>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.print.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.2.7/pdfmake.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.2.7/vfs_fonts.js"></script>
    
    <script>
        const tableData = {data_json};
        const columns = {columns_json};
        const colNames = {col_names_json};
        const higherBetter = ['R2_PG', 'R2_VG', 'Pacc_PG', 'Pacc_VG'];
        const lowerBetter = ['RMSE_PG', 'RMSE_VG', 'MAE_PG', 'MAE_VG', 'Physics_Violation_MW'];
        
        const colIndex = {{}};
        colNames.forEach((name, idx) => colIndex[name] = idx);
        
        $(document).ready(function() {{
            const table = $('#resultsTable').DataTable({{
                data: tableData,
                columns: columns,
                pageLength: 25,
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                order: [[0, 'desc']],
                scrollX: true,
                dom: 'Blfrtip',
                buttons: [
                    {{ extend: 'copy', text: '📋 Copy' }},
                    {{ extend: 'csv', text: '📄 CSV' }},
                    {{ extend: 'excel', text: '📊 Excel' }},
                    {{ extend: 'pdf', text: '📑 PDF', orientation: 'landscape' }},
                    {{ extend: 'print', text: '🖨️ Print' }}
                ],
                createdRow: function(row, data, dataIndex) {{
                    higherBetter.forEach(function(colName) {{
                        if (colIndex[colName] !== undefined) {{
                            const idx = colIndex[colName];
                            const val = parseFloat(String(data[idx]).replace('%', '').replace('★ ', ''));
                            if (!isNaN(val)) {{
                                const maxVal = Math.max(...tableData.map(r => 
                                    parseFloat(String(r[idx]).replace('%', '').replace('★ ', '')) || -Infinity
                                ));
                                if (val === maxVal) $('td', row).eq(idx).addClass('best-high');
                            }}
                        }}
                    }});
                    lowerBetter.forEach(function(colName) {{
                        if (colIndex[colName] !== undefined) {{
                            const idx = colIndex[colName];
                            const val = parseFloat(String(data[idx]).replace('★ ', ''));
                            if (!isNaN(val)) {{
                                const minVal = Math.min(...tableData.map(r => {{
                                    const v = parseFloat(String(r[idx]).replace('★ ', ''));
                                    return isNaN(v) ? Infinity : v;
                                }}));
                                if (val === minVal) $('td', row).eq(idx).addClass('best-low');
                            }}
                        }}
                    }});
                }}
            }});
            
            $('#resultsTable tfoot .column-filter').on('keyup change', function() {{
                table.column($(this).parent().index()).search(this.value).draw();
            }});
            
            // Stats
            const models = [...new Set(tableData.map(r => r[colIndex['model']]))].filter(Boolean);
            let statsHtml = `<div class="stat-card"><div class="label">Total</div><div class="value">${{tableData.length}}</div></div>`;
            statsHtml += `<div class="stat-card"><div class="label">Models</div><div class="value">${{models.join(', ')}}</div></div>`;
            if (colIndex['R2_PG'] !== undefined) {{
                const vals = tableData.map(r => parseFloat(String(r[colIndex['R2_PG']]).replace('★ ', ''))).filter(v => !isNaN(v));
                if (vals.length) statsHtml += `<div class="stat-card"><div class="label">Best R² PG</div><div class="value">${{Math.max(...vals).toFixed(4)}}</div></div>`;
            }}
            if (colIndex['Physics_Violation_MW'] !== undefined) {{
                const vals = tableData.map(r => parseFloat(String(r[colIndex['Physics_Violation_MW']]).replace('★ ', ''))).filter(v => !isNaN(v));
                if (vals.length) statsHtml += `<div class="stat-card"><div class="label">Best Physics</div><div class="value">${{Math.min(...vals).toFixed(2)}} MW</div></div>`;
            }}
            $('#statsRow').html(statsHtml);
        }});
    </script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"[OK] Interactive HTML exported to: {output_path}")


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
