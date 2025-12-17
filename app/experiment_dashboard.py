"""
Experiment Dashboard for Deep OPF

A Streamlit web app for configuring experiments and viewing results.

Usage:
    streamlit run app/experiment_dashboard.py

Or via the helper script:
    python app/run_dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deep_opf.models import GCNN, AdmittanceDNN
from deep_opf.utils.experiment_logger import count_parameters

# =============================================================================
# Constants
# =============================================================================
DATASET_INFO = {
    "case39": {"n_bus": 39, "n_gen": 10, "description": "IEEE 39-bus (10k/2k/1.2k)"},
    "case6ww": {
        "n_bus": 6,
        "n_gen": 3,
        "description": "Wood & Wollenberg 6-bus (10k/2k)",
    },
}

GCNN_CSV_PATH = PROJECT_ROOT / "outputs" / "gcnn_experiments.csv"
DNN_CSV_PATH = PROJECT_ROOT / "outputs" / "dnn_experiments.csv"


# =============================================================================
# Parameter Counting Functions
# =============================================================================
def count_gcnn_params(
    n_bus: int,
    n_gen: int,
    channels: int,
    n_layers: int,
    fc_hidden_dim: int,
    n_fc_layers: int,
) -> int:
    """Instantiate GCNN and count parameters."""
    model = GCNN(
        n_bus=n_bus,
        n_gen=n_gen,
        in_channels=channels,
        hidden_channels=channels,
        n_layers=n_layers,
        fc_hidden_dim=fc_hidden_dim,
        n_fc_layers=n_fc_layers,
        dropout=0.0,
    )
    return count_parameters(model)


def count_dnn_params(
    n_bus: int,
    n_gen: int,
    hidden_dim: int,
    num_layers: int,
) -> int:
    """Instantiate DNN and count parameters."""
    input_dim = 2 * n_bus + 2 * (n_bus * n_bus)
    model = AdmittanceDNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_gen=n_gen,
        n_bus=n_bus,
        dropout=0.0,
    )
    return count_parameters(model)


# =============================================================================
# Sweep Utilities
# =============================================================================
def parse_sweep_value(value_str: str, param_type: type = int) -> tuple[bool, list, str]:
    """
    Parse a potentially comma-separated parameter string.

    Returns:
        (is_valid, values_list, error_message)
    """
    import re

    parts = re.split(r"\s*,\s*", value_str.strip())
    values = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            values.append(param_type(part))
        except ValueError:
            return False, [], f"Invalid value: '{part}'"

    if not values:
        return False, [], "No values provided"

    return True, values, ""


def is_sweep_value(value_str: str) -> bool:
    """Check if value contains multiple comma-separated values."""
    return "," in str(value_str)


def count_combinations(*value_strings: str) -> int:
    """Count total combinations from multiple sweep value strings."""
    import re

    total = 1
    for vs in value_strings:
        parts = [p.strip() for p in re.split(r"\s*,\s*", str(vs).strip()) if p.strip()]
        total *= max(1, len(parts))
    return total


# =============================================================================
# Command Generation Functions
# =============================================================================
def format_command(cmd_parts: list[str], shell: str = "bash") -> str:
    """Format command parts with appropriate line continuation for shell type."""
    if shell == "powershell":
        return " `\n    ".join(cmd_parts)
    else:  # bash
        return " \\\n    ".join(cmd_parts)


def generate_gcnn_command(
    dataset: str,
    channels: str,  # Now string for sweep support
    n_layers: str,
    fc_hidden_dim: str,
    n_fc_layers: str,
    batch_size: str,
    lr: float,
    weight_decay: float,
    patience: int,
    dropout: float,
    max_epochs: str,  # Now string for sweep support
    two_phase: bool,
    phase1_epochs: int,
    phase2_epochs: int,
    kappa: str,  # Now string for sweep support
    gpu: int,
    shell: str = "bash",
    phase2_only: bool = False,
    warm_start_ckpt: str = "",
) -> str:
    """Generate CLI command for GCNN experiment."""
    cmd_parts = [
        "python scripts/run_experiment.py gcnn",
        dataset,
        f"--channels {channels}",
        f"--n_layers {n_layers}",
        f"--fc_hidden_dim {fc_hidden_dim}",
        f"--n_fc_layers {n_fc_layers}",
        f"--batch_size {batch_size}",
        f"--lr {lr}",
        f"--weight_decay {weight_decay}",
        f"--patience {patience}",
        f"--dropout {dropout}",
        f"--gpu {gpu}",
    ]

    if phase2_only and warm_start_ckpt:
        # Phase 2 only mode: single phase with warm start
        cmd_parts.extend(
            [
                "--phase2-only",
                f'--warm_start_ckpt "{warm_start_ckpt}"',
                f"--max_epochs {max_epochs}",
                f"--kappa {kappa}",
            ]
        )
    elif two_phase:
        cmd_parts.extend(
            [
                "--two-phase",
                f"--phase1_epochs {phase1_epochs}",
                f"--phase2_epochs {phase2_epochs}",
                f"--kappa {kappa}",
            ]
        )
    else:
        cmd_parts.append(f"--max_epochs {max_epochs}")
        cmd_parts.append(f"--kappa {kappa}")

    return format_command(cmd_parts, shell)


def generate_dnn_command(
    dataset: str,
    hidden_dim: str,  # Now string for sweep support
    num_layers: str,  # Now string for sweep support
    batch_size: str,  # Now string for sweep support
    lr: float,
    weight_decay: float,
    patience: int,
    dropout: float,
    max_epochs: str,  # Now string for sweep support
    gpu: int,
    shell: str = "bash",
) -> str:
    """Generate CLI command for DNN experiment."""
    cmd_parts = [
        "python scripts/run_experiment.py dnn",
        dataset,
        f"--hidden_dim {hidden_dim}",
        f"--num_layers {num_layers}",
        f"--batch_size {batch_size}",
        f"--lr {lr}",
        f"--weight_decay {weight_decay}",
        f"--patience {patience}",
        f"--dropout {dropout}",
        f"--max_epochs {max_epochs}",
        f"--gpu {gpu}",
    ]

    return format_command(cmd_parts, shell)


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_gcnn_results() -> pd.DataFrame:
    """Load GCNN experiment results from CSV."""
    if not GCNN_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(GCNN_CSV_PATH)


def load_dnn_results() -> pd.DataFrame:
    """Load DNN experiment results from CSV."""
    if not DNN_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DNN_CSV_PATH)


def format_params(n: float) -> str:
    """Format parameter count as K or M."""
    if pd.isna(n):
        return "N/A"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# =============================================================================
# Streamlit UI - Settings Tab
# =============================================================================
def render_settings_tab():
    """Render the Settings tab content."""
    st.header("⚙️ Experiment Settings")

    # Model Type Selection
    model_type = st.radio(
        "Model Type",
        ["GCNN", "DNN"],
        horizontal=True,
        help="Select the model architecture to train",
    )

    # Dataset Selection
    dataset = st.selectbox(
        "Dataset",
        list(DATASET_INFO.keys()),
        format_func=lambda x: f"{x} - {DATASET_INFO[x]['description']}",
    )

    dataset_info = DATASET_INFO[dataset]
    n_bus = dataset_info["n_bus"]
    n_gen = dataset_info["n_gen"]

    st.divider()

    # Architecture Parameters
    st.subheader("📐 Architecture")
    st.caption("💡 Use commas for sweep mode: `8,16,32`")

    if model_type == "GCNN":
        col1, col2 = st.columns(2)
        with col1:
            channels = st.text_input(
                "Channels (in_channels = hidden_channels)",
                value="8",
                help="Sweep: 4,8,16",
            )
            n_layers = st.text_input(
                "GraphConv Layers",
                value="2",
                help="Sweep: 1,2,3",
            )
        with col2:
            fc_hidden_dim = st.text_input(
                "FC Hidden Dim",
                value="256",
                help="Sweep: 128,256,512",
            )
            n_fc_layers = st.text_input(
                "FC Layers",
                value="1",
                help="Sweep: 1,2",
            )

        # Validate and calculate params (use first value for display)
        valid_ch, ch_vals, ch_err = parse_sweep_value(channels, int)
        valid_nl, nl_vals, nl_err = parse_sweep_value(n_layers, int)
        valid_fc, fc_vals, fc_err = parse_sweep_value(fc_hidden_dim, int)
        valid_nfc, nfc_vals, nfc_err = parse_sweep_value(n_fc_layers, int)

        if valid_ch and valid_nl and valid_fc and valid_nfc:
            n_params = count_gcnn_params(
                n_bus, n_gen, ch_vals[0], nl_vals[0], fc_vals[0], nfc_vals[0]
            )
        else:
            n_params = 0
            errors = [e for e in [ch_err, nl_err, fc_err, nfc_err] if e]
            if errors:
                st.error(f"Invalid input: {'; '.join(errors)}")

    else:  # DNN
        col1, col2 = st.columns(2)
        with col1:
            hidden_dim = st.text_input(
                "Hidden Dim",
                value="128",
                help="Sweep: 64,128,256",
            )
        with col2:
            num_layers = st.text_input(
                "Number of Layers",
                value="3",
                help="Sweep: 2,3,4",
            )

        # Validate and calculate params
        valid_hd, hd_vals, hd_err = parse_sweep_value(hidden_dim, int)
        valid_numl, numl_vals, numl_err = parse_sweep_value(num_layers, int)

        if valid_hd and valid_numl:
            n_params = count_dnn_params(n_bus, n_gen, hd_vals[0], numl_vals[0])
        else:
            n_params = 0
            errors = [e for e in [hd_err, numl_err] if e]
            if errors:
                st.error(f"Invalid input: {'; '.join(errors)}")

    st.divider()

    # Training Hyperparameters
    st.subheader("🎯 Training")

    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.text_input(
            "Batch Size",
            value="32" if model_type == "GCNN" else "64",
            help="Sweep: 32,64,128",
        )
        patience = st.number_input(
            "Patience",
            min_value=5,
            max_value=2000,
            value=20,
            step=5,
            help="Early stopping patience",
        )
    with col2:
        lr = st.number_input(
            "Learning Rate",
            min_value=1e-5,
            max_value=1e-1,
            value=1e-3,
            step=1e-4,
            format="%.5f",
        )
        dropout = st.number_input(
            "Dropout",
            min_value=0.0,
            max_value=0.5,
            value=0.0 if model_type == "GCNN" else 0.1,
            step=0.05,
            format="%.2f",
        )
    with col3:
        weight_decay = st.number_input(
            "Weight Decay",
            min_value=0.0,
            max_value=0.1,
            value=0.0,
            step=0.001,
            format="%.4f",
        )
        max_epochs = st.text_input(
            "Max Epochs",
            value="100",
            help="Sweep: 50,100,200",
        )

    # GCNN-specific: Two-phase training
    two_phase = False
    phase1_epochs = 50
    phase2_epochs = 100
    kappa = "0.1"
    phase2_only = False
    warm_start_ckpt = ""

    if model_type == "GCNN":
        st.divider()
        st.subheader("🔬 Physics-Informed Training")

        two_phase = st.checkbox(
            "Enable Two-Phase Training",
            value=False,
            help="Phase 1: Supervised only (kappa=0). Phase 2: Add physics loss.",
        )

        # Phase 2 Only mode - resume from existing checkpoint
        phase2_only = st.checkbox(
            "Phase 2 Only (Resume from checkpoint)",
            value=False,
            help="Run only Phase 2 with an existing Phase 1 checkpoint",
        )

        warm_start_ckpt = ""
        if phase2_only:
            warm_start_ckpt = st.text_input(
                "Phase 1 Checkpoint Path",
                value="",
                placeholder="lightning_logs/version_X/checkpoints/epochXX-val_lossX.XXXX.ckpt",
                help="Path to Phase 1 checkpoint for warm start",
            )
            if not warm_start_ckpt:
                st.warning("⚠️ Please provide a checkpoint path for Phase 2 training")

        col1, col2, col3 = st.columns(3)
        if two_phase and not phase2_only:
            with col1:
                phase1_epochs = st.number_input(
                    "Phase 1 Epochs",
                    min_value=0,
                    value=50,
                    step=10,
                )
            with col2:
                phase2_epochs = st.number_input(
                    "Phase 2 Epochs",
                    min_value=0,
                    value=100,
                    step=10,
                )
        with col3 if two_phase else col1:
            kappa = st.text_input(
                "Kappa (Physics Loss Weight)",
                value="0.1",
                help="Sweep: 0.1,0.5,1.0",
            )

    st.divider()

    # Hardware
    st.subheader("🖥️ Hardware")
    gpu = st.number_input(
        "GPU Device",
        min_value=0,
        max_value=3,
        value=0,
        help="CUDA device index (0 for first GPU)",
    )

    st.divider()

    # Parameter Count Display
    st.metric(
        label="📊 Total Parameters",
        value=format_params(n_params),
        help=f"Exact: {n_params:,}",
    )

    st.divider()

    # Command Generation
    st.subheader("📝 Generated Command")

    # Detect sweep mode
    if model_type == "GCNN":
        sweep_params = {
            "channels": channels,
            "n_layers": n_layers,
            "fc_hidden_dim": fc_hidden_dim,
            "n_fc_layers": n_fc_layers,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "kappa": kappa,
        }
    else:
        sweep_params = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
        }

    # Count sweep combinations
    sweep_counts = {}
    for name, val in sweep_params.items():
        if is_sweep_value(val):
            _, values, _ = parse_sweep_value(val)
            if values:
                sweep_counts[name] = len(values)

    total_runs = 1
    for count in sweep_counts.values():
        total_runs *= count

    # Display sweep indicator
    if total_runs > 1:
        sweep_details = " × ".join(
            f"{name}({count})" for name, count in sweep_counts.items()
        )
        st.warning(f"⚠️ **Sweep Mode**: {sweep_details} = **{total_runs} experiments**")

    # Shell format selector
    shell_format = st.radio(
        "Shell Format",
        ["PowerShell", "Bash"],
        horizontal=True,
        help="Select your shell for correct line continuation syntax",
    )
    shell = "powershell" if shell_format == "PowerShell" else "bash"

    if model_type == "GCNN":
        command = generate_gcnn_command(
            dataset=dataset,
            channels=channels,
            n_layers=n_layers,
            fc_hidden_dim=fc_hidden_dim,
            n_fc_layers=n_fc_layers,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            dropout=dropout,
            max_epochs=max_epochs,
            two_phase=two_phase,
            phase1_epochs=phase1_epochs,
            phase2_epochs=phase2_epochs,
            kappa=kappa,
            gpu=gpu,
            shell=shell,
            phase2_only=phase2_only,
            warm_start_ckpt=warm_start_ckpt,
        )
    else:
        command = generate_dnn_command(
            dataset=dataset,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            dropout=dropout,
            max_epochs=max_epochs,
            gpu=gpu,
            shell=shell,
        )

    st.code(command, language="powershell" if shell == "powershell" else "bash")

    # Copy button hint
    if total_runs > 1:
        st.caption(
            "💡 Copy the command above. The script will automatically run all sweep combinations."
        )
    else:
        st.caption("💡 Click the copy icon in the code block above to copy the command")


# =============================================================================
# Streamlit UI - Results Tab
# =============================================================================
def render_results_tab():
    """Render the Results tab content."""
    st.header("📊 Experiment Results")

    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        model_filter = st.selectbox(
            "Model",
            ["All", "GCNN", "DNN"],
            label_visibility="collapsed",
        )

    # Load data
    gcnn_df = load_gcnn_results()
    dnn_df = load_dnn_results()

    # Add model type column
    if not gcnn_df.empty:
        gcnn_df["model"] = "GCNN"
        gcnn_df["arch"] = gcnn_df.apply(
            lambda r: f"{int(r['channels'])}-{int(r['n_layers'])}-{int(r['fc_hidden_dim'])}",
            axis=1,
        )

    if not dnn_df.empty:
        dnn_df["model"] = "DNN"
        dnn_df["arch"] = dnn_df.apply(
            lambda r: f"{int(r['hidden_dim'])}-{int(r['num_layers'])}", axis=1
        )

    # Combine and filter
    if model_filter == "GCNN":
        df = gcnn_df
    elif model_filter == "DNN":
        df = dnn_df
    else:
        # Combine with common columns
        common_cols = [
            "timestamp",
            "model",
            "dataset",
            "params",
            "arch",
            "batch_size",
            "lr",
            "patience",
            "dropout",
            "max_epochs",
            "kappa",
            "phase",
            "R2_PG_seen",
            "R2_VG_seen",
            "Pacc_PG_seen",
            "Pacc_VG_seen",
            "Physics_MW_seen",
            "R2_PG_unseen",
            "R2_VG_unseen",
            "Pacc_PG_unseen",
            "Pacc_VG_unseen",
            "Physics_MW_unseen",
            "best_val_loss",
            "duration_sec",
            "ckpt_path",
        ]

        dfs = []
        if not gcnn_df.empty:
            gcnn_subset = gcnn_df[
                [c for c in common_cols if c in gcnn_df.columns]
            ].copy()
            dfs.append(gcnn_subset)
        if not dnn_df.empty:
            dnn_subset = dnn_df[[c for c in common_cols if c in dnn_df.columns]].copy()
            dfs.append(dnn_subset)

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if df.empty:
        st.info("No experiment results found. Run some experiments first!")
        return

    # Sort by timestamp descending
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    # Display summary table
    st.subheader(f"Found {len(df)} experiments")

    # Format columns for display
    display_cols = [
        "timestamp",
        "model",
        "dataset",
        "params",
        "arch",
        "R2_PG_seen",
        "Pacc_PG_seen",
        "Physics_MW_seen",
        "R2_PG_unseen",
        "Pacc_PG_unseen",
        "Physics_MW_unseen",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    # Create display dataframe with formatting
    display_df = df[display_cols].copy()
    if "params" in display_df.columns:
        display_df["params"] = display_df["params"].apply(format_params)
    if "timestamp" in display_df.columns:
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime(
            "%m-%d %H:%M"
        )

    # Rename columns for display
    column_names = {
        "timestamp": "Time",
        "model": "Model",
        "dataset": "Data",
        "params": "Params",
        "arch": "Arch",
        "R2_PG_seen": "R²(P) Seen",
        "Pacc_PG_seen": "Pacc(P) Seen",
        "Physics_MW_seen": "Phys Seen",
        "R2_PG_unseen": "R²(P) Unseen",
        "Pacc_PG_unseen": "Pacc(P) Unseen",
        "Physics_MW_unseen": "Phys Unseen",
    }
    display_df = display_df.rename(columns=column_names)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # Detailed view with expanders
    st.subheader("📋 Detailed Results")

    for idx, row in df.iterrows():
        timestamp = row.get("timestamp", "N/A")
        model = row.get("model", "N/A")
        dataset = row.get("dataset", "N/A")
        arch = row.get("arch", "N/A")

        with st.expander(f"**{timestamp}** | {model} | {dataset} | {arch}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Training**")
                st.text(f"Best Val Loss: {row.get('best_val_loss', 'N/A')}")
                st.text(f"Duration: {row.get('duration_sec', 'N/A')}s")
                st.text(f"Batch Size: {row.get('batch_size', 'N/A')}")
                st.text(f"LR: {row.get('lr', 'N/A')}")
                st.text(f"Patience: {row.get('patience', 'N/A')}")
                if model == "GCNN":
                    st.text(f"Kappa: {row.get('kappa', 'N/A')}")
                    st.text(f"Phase: {row.get('phase', 'N/A')}")

            with col2:
                st.markdown("**Evaluation (Seen / Unseen)**")
                st.text(
                    f"R²(PG): {row.get('R2_PG_seen', 'N/A')} / {row.get('R2_PG_unseen', 'N/A')}"
                )
                st.text(
                    f"R²(VG): {row.get('R2_VG_seen', 'N/A')} / {row.get('R2_VG_unseen', 'N/A')}"
                )
                st.text(
                    f"Pacc(PG): {row.get('Pacc_PG_seen', 'N/A')}% / {row.get('Pacc_PG_unseen', 'N/A')}%"
                )
                st.text(
                    f"Pacc(VG): {row.get('Pacc_VG_seen', 'N/A')}% / {row.get('Pacc_VG_unseen', 'N/A')}%"
                )
                st.text(
                    f"Physics: {row.get('Physics_MW_seen', 'N/A')} / {row.get('Physics_MW_unseen', 'N/A')} MW"
                )

            st.markdown("**Checkpoint**")
            st.code(row.get("ckpt_path", "N/A"), language=None)


# =============================================================================
# Main App
# =============================================================================
def main():
    st.set_page_config(
        page_title="Deep OPF Experiment Dashboard",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ Deep OPF Experiment Dashboard")

    # Create tabs
    tab1, tab2 = st.tabs(["⚙️ Settings", "📊 Results"])

    with tab1:
        render_settings_tab()

    with tab2:
        render_results_tab()


if __name__ == "__main__":
    main()
