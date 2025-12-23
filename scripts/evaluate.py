"""Evaluation script for deep_opf models using Hydra configuration.

Implements metrics from Gao et al. paper (IEEE Trans. Power Systems, 2024):
- Probabilistic Accuracy (Pacc, Eq. 37): % of samples within threshold
  - PG: |pred - true| < 0.01 p.u. (i.e., 1 MW assuming BaseMVA=100)
  - VG: |pred - true| < 0.001 p.u.
- Standard metrics: R^2, RMSE, MAE (on denormalized p.u. values)
- Physics violation: Mean active power mismatch (Eq. 8)

CRITICAL: Model outputs are Z-score normalized. Must denormalize before metrics.

Usage:
    # Evaluate with specific checkpoint
    python scripts/evaluate.py ckpt_path=/path/to/model.ckpt

    # Auto-find best checkpoint in outputs/
    python scripts/evaluate.py

    # Override data/model config
    python scripts/evaluate.py model=gcnn data=case39 ckpt_path=best
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deep_opf.data import OPFDataModule
from deep_opf.loss import build_gen_bus_matrix, physics_loss
from deep_opf.models import GCNN, AdmittanceDNN
from deep_opf.task import OPFTask
from deep_opf.utils.logger import log_evaluation_to_csv


# =============================================================================
# Constants (from Gao et al. paper Section V-A)
# =============================================================================
BASE_MVA = 100.0  # Standard base power for p.u. conversion
PG_THRESHOLD_PU = 0.01  # 1 MW = 0.01 p.u. (assuming BaseMVA=100)
VG_THRESHOLD_PU = 0.001  # 0.001 p.u. for voltage

# Physical limits from PYPOWER case files (in MW and p.u.)
# These are loaded dynamically based on case, but defaults for case39:
CASE39_PG_LIMITS_MW = {
    # gen_idx: (Pmin, Pmax)
    0: (0, 1040), 1: (0, 646), 2: (0, 725), 3: (0, 652), 4: (0, 508),
    5: (0, 687), 6: (0, 580), 7: (0, 564), 8: (0, 865), 9: (0, 1100),
}
CASE39_V_LIMITS_PU = (0.94, 1.06)  # (Vmin, Vmax) for all buses


def get_physical_limits(case_name: str) -> dict:
    """Get physical limits for a given case."""
    if case_name == "case39":
        return {
            "pg_min_pu": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / BASE_MVA,
            "pg_max_pu": np.array([1040, 646, 725, 652, 508, 687, 580, 564, 865, 1100]) / BASE_MVA,
            "vg_min_pu": 0.94,
            "vg_max_pu": 1.06,
        }
    elif case_name == "case6ww" or case_name == "case6":
        # Wood & Wollenberg 6-bus case
        # Generator VG setpoints from PYPOWER: [1.05, 1.05, 1.07]
        # Use per-generator limits with ±0.02 tolerance around setpoints
        return {
            "pg_min_pu": np.array([0, 0, 0]) / BASE_MVA,
            "pg_max_pu": np.array([200, 150, 180]) / BASE_MVA,
            # Per-generator voltage limits based on setpoints
            "vg_min_pu": np.array([1.03, 1.03, 1.05]),  # setpoints - 0.02
            "vg_max_pu": np.array([1.07, 1.07, 1.09]),  # setpoints + 0.02
        }
    else:
        # Default conservative limits
        return {
            "pg_min_pu": None,
            "pg_max_pu": None,
            "vg_min_pu": 0.90,
            "vg_max_pu": 1.10,
        }


# =============================================================================
# Model Instantiation (reused from train.py)
# =============================================================================
def instantiate_model(cfg: DictConfig, n_bus: int, n_gen: int):
    """Instantiate the appropriate model based on config."""
    model_name = cfg.model.name

    if model_name == "dnn":
        input_dim = 2 * n_bus + 2 * (n_bus * n_bus)
        model = AdmittanceDNN(
            input_dim=input_dim,
            hidden_dim=cfg.model.architecture.hidden_dim,
            num_layers=cfg.model.architecture.num_layers,
            n_gen=n_gen,
            n_bus=n_bus,
            dropout=cfg.model.architecture.dropout,
        )
    elif model_name == "gcnn":
        model = GCNN(
            n_bus=n_bus,
            n_gen=n_gen,
            in_channels=cfg.model.architecture.in_channels,
            hidden_channels=cfg.model.architecture.hidden_channels,
            n_layers=cfg.model.architecture.n_layers,
            fc_hidden_dim=cfg.model.architecture.fc_hidden_dim,
            n_fc_layers=cfg.model.architecture.n_fc_layers,
            dropout=cfg.model.architecture.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# =============================================================================
# Checkpoint Discovery
# =============================================================================
def find_best_checkpoint(search_dir: Path) -> Path | None:
    """
    Find the most recent .ckpt file in outputs/ directory.

    Args:
        search_dir: Directory to search (typically outputs/)

    Returns:
        Path to most recent checkpoint, or None if not found
    """
    if not search_dir.exists():
        return None

    # Find all .ckpt files recursively
    ckpt_files = list(search_dir.rglob("*.ckpt"))
    if not ckpt_files:
        return None

    # Sort by modification time (most recent first)
    ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpt_files[0]


# =============================================================================
# Denormalization
# =============================================================================
def denormalize_tensor(
    values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Revert Z-score normalization: y_orig = y_norm * (std + eps) + mean

    Args:
        values: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        eps: Small value added to std during normalization

    Returns:
        Denormalized tensor in original scale (p.u.)
    """
    return values * (std + eps) + mean


# =============================================================================
# Metrics Computation
# =============================================================================
def probabilistic_accuracy(
    pred: np.ndarray, true: np.ndarray, threshold: float
) -> float:
    """
    Compute probabilistic accuracy (Eq. 37 from paper).

    Pacc = P(|pred - true| < threshold) * 100%

    Args:
        pred: Predicted values (any shape, will be flattened)
        true: True values (same shape as pred)
        threshold: Acceptance threshold in p.u.

    Returns:
        Percentage of samples within threshold (0-100)
    """
    errors = np.abs(pred.flatten() - true.flatten())
    return float(np.mean(errors < threshold) * 100.0)


def compute_regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """
    Compute standard regression metrics (R^2, RMSE, MAE).

    Args:
        pred: Predicted values
        true: True values

    Returns:
        Dict with R2, RMSE, MAE
    """
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    return {
        "R2": float(r2_score(true_flat, pred_flat)),
        "RMSE": float(np.sqrt(mean_squared_error(true_flat, pred_flat))),
        "MAE": float(mean_absolute_error(true_flat, pred_flat)),
    }


def compute_constraint_violations(
    pg_pred: np.ndarray,
    vg_pred: np.ndarray,
    limits: dict,
) -> dict[str, float]:
    """
    Compute constraint violation metrics for PG and VG predictions.

    Checks if predictions violate physical limits:
    - PG: Generator power output limits [Pmin, Pmax] per generator
    - VG: Voltage magnitude limits [Vmin, Vmax] for all generators

    Args:
        pg_pred: Predicted PG [N, n_gen] in p.u.
        vg_pred: Predicted VG [N, n_gen] in p.u.
        limits: Dict with pg_min_pu, pg_max_pu, vg_min_pu, vg_max_pu

    Returns:
        Dict with violation metrics:
        - pg_violation_rate: % of PG predictions outside limits
        - pg_violation_avg_mw: Average violation magnitude (MW) when violated
        - pg_violation_max_mw: Maximum violation (MW)
        - vg_violation_rate: % of VG predictions outside limits
        - vg_violation_avg_pu: Average violation magnitude (p.u.) when violated
        - vg_violation_max_pu: Maximum violation (p.u.)
    """
    result = {}

    # PG violations
    pg_min = limits.get("pg_min_pu")
    pg_max = limits.get("pg_max_pu")

    if pg_min is not None and pg_max is not None:
        # Compute violations for each generator
        # pg_pred: [N, n_gen], pg_min/pg_max: [n_gen]
        below_min = pg_pred < pg_min[None, :]  # [N, n_gen]
        above_max = pg_pred > pg_max[None, :]  # [N, n_gen]
        pg_violated = below_min | above_max

        # Violation rate (% of all predictions)
        result["pg_violation_rate"] = float(np.mean(pg_violated) * 100.0)

        # Violation magnitude (only for violated predictions)
        violation_below = np.where(below_min, pg_min[None, :] - pg_pred, 0)
        violation_above = np.where(above_max, pg_pred - pg_max[None, :], 0)
        violation_magnitude = violation_below + violation_above  # [N, n_gen]

        if np.any(pg_violated):
            result["pg_violation_avg_mw"] = float(
                np.mean(violation_magnitude[pg_violated]) * BASE_MVA
            )
            result["pg_violation_max_mw"] = float(
                np.max(violation_magnitude) * BASE_MVA
            )
        else:
            result["pg_violation_avg_mw"] = 0.0
            result["pg_violation_max_mw"] = 0.0
    else:
        result["pg_violation_rate"] = float("nan")
        result["pg_violation_avg_mw"] = float("nan")
        result["pg_violation_max_mw"] = float("nan")

    # VG violations
    vg_min = limits.get("vg_min_pu")
    vg_max = limits.get("vg_max_pu")

    if vg_min is not None and vg_max is not None:
        # Handle both scalar and per-generator array limits
        if np.isscalar(vg_min):
            below_min = vg_pred < vg_min
            above_max = vg_pred > vg_max
            violation_below = np.where(below_min, vg_min - vg_pred, 0)
            violation_above = np.where(above_max, vg_pred - vg_max, 0)
        else:
            # Per-generator limits: vg_min/vg_max are [n_gen] arrays
            below_min = vg_pred < vg_min[None, :]  # [N, n_gen]
            above_max = vg_pred > vg_max[None, :]  # [N, n_gen]
            violation_below = np.where(below_min, vg_min[None, :] - vg_pred, 0)
            violation_above = np.where(above_max, vg_pred - vg_max[None, :], 0)

        vg_violated = below_min | above_max
        result["vg_violation_rate"] = float(np.mean(vg_violated) * 100.0)

        violation_magnitude = violation_below + violation_above

        if np.any(vg_violated):
            result["vg_violation_avg_pu"] = float(
                np.mean(violation_magnitude[vg_violated])
            )
            result["vg_violation_max_pu"] = float(np.max(violation_magnitude))
        else:
            result["vg_violation_avg_pu"] = 0.0
            result["vg_violation_max_pu"] = 0.0
    else:
        result["vg_violation_rate"] = float("nan")
        result["vg_violation_avg_pu"] = float("nan")
        result["vg_violation_max_pu"] = float("nan")

    return result


def compute_physics_violation(
    pg_pred: torch.Tensor,
    v_bus_pred: torch.Tensor,
    pd: torch.Tensor,
    qd: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    gen_bus_matrix: torch.Tensor,
) -> float:
    """
    Compute physics violation using AC power flow equations (Eq. 8).

    Measures mismatch between predicted PG and PG computed from
    predicted voltages via power flow equations.

    Args:
        pg_pred: Predicted active power [N, n_gen] in p.u.
        v_bus_pred: Predicted voltages [N, n_bus, 2] (e, f components)
        pd, qd: Power demands [N, n_bus] in p.u.
        G, B: Admittance matrices [n_bus, n_bus]
        gen_bus_matrix: Generator-bus incidence [n_bus, n_gen]

    Returns:
        Mean physics violation in MW (RMSE * BaseMVA)
    """
    with torch.no_grad():
        result = physics_loss(
            pg=pg_pred,
            v_bus=v_bus_pred,
            pd=pd,
            qd=qd,
            G=G,
            B=B,
            gen_bus_matrix=gen_bus_matrix,
            include_reactive=False,
        )
        mse_pu = result["loss_p"].item()
        rmse_mw = np.sqrt(mse_pu) * BASE_MVA

    return rmse_mw


# =============================================================================
# Inference Loop
# =============================================================================
def run_inference(
    task: OPFTask,
    datamodule: OPFDataModule,
    device: torch.device,
    feature_type: str = "graph",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    """
    Run inference on test set and collect all predictions/labels.

    Args:
        task: Loaded OPFTask
        datamodule: DataModule with test data
        device: Torch device
        feature_type: 'flat' for DNN, 'graph' for GCNN

    Returns:
        Tuple of (predictions, labels, auxiliary_data)
        - predictions: {'pg': [N, n_gen], 'vg': [N, n_gen], 'v_bus': [N, n_bus, 2]}
        - labels: {'pg': [N, n_gen], 'vg': [N, n_gen]}
        - auxiliary_data: {'pd': [N, n_bus], 'qd': [N, n_bus], 'operators': {...}}
    """
    task.eval()
    task.to(device)

    all_pg_pred, all_vg_pred, all_v_bus_pred = [], [], []
    all_pg_label, all_vg_label = [], []
    all_pd, all_qd = [], []
    first_operators = None

    dataloader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in dataloader:
            # Move tensors to device
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                elif isinstance(v, dict):
                    batch_device[k] = {
                        kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                        for kk, vv in v.items()
                    }
                else:
                    batch_device[k] = v

            # Forward pass through model
            # DNN expects flat tensor input, GCNN expects batch dict
            if feature_type == "flat":
                model_input = batch_device["input"]
            else:
                model_input = batch_device
            preds = task.model(model_input)

            # Collect predictions
            all_pg_pred.append(preds["pg"].cpu())
            all_vg_pred.append(preds["vg"].cpu())
            all_v_bus_pred.append(preds["v_bus"].cpu())

            # Collect labels
            all_pg_label.append(batch["pg_label"].cpu())
            all_vg_label.append(batch["vg_label"].cpu())

            # Collect auxiliary data for physics loss
            all_pd.append(batch["pd"].cpu())
            all_qd.append(batch["qd"].cpu())

            # Store first batch operators (same topology assumed)
            if first_operators is None and "operators" in batch:
                first_operators = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in batch["operators"].items()
                }

    predictions = {
        "pg": torch.cat(all_pg_pred, dim=0),
        "vg": torch.cat(all_vg_pred, dim=0),
        "v_bus": torch.cat(all_v_bus_pred, dim=0),
    }
    labels = {
        "pg": torch.cat(all_pg_label, dim=0),
        "vg": torch.cat(all_vg_label, dim=0),
    }
    auxiliary = {
        "pd": torch.cat(all_pd, dim=0),
        "qd": torch.cat(all_qd, dim=0),
        "operators": first_operators,
    }

    return predictions, labels, auxiliary


# =============================================================================
# Main Evaluation Function
# =============================================================================
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function.

    Args:
        cfg: Hydra configuration (expects optional 'ckpt_path' key)
    """
    print("=" * 70)
    print("Model Evaluation - deep_opf")
    print("=" * 70)

    # Get original working directory
    original_cwd = Path(hydra.utils.get_original_cwd())

    # ==========================================================================
    # 1. Resolve checkpoint path
    # ==========================================================================
    ckpt_path_str = cfg.get("ckpt_path", "best")

    if ckpt_path_str == "best" or ckpt_path_str is None:
        # Search for most recent checkpoint in outputs/ and lightning_logs/
        print("\nSearching for best checkpoint...")
        ckpt_path = find_best_checkpoint(original_cwd / "outputs")
        if ckpt_path is None:
            ckpt_path = find_best_checkpoint(original_cwd / "lightning_logs")
        if ckpt_path is None:
            print("[ERROR] No checkpoint found in outputs/ or lightning_logs/")
            print("Please provide ckpt_path=/path/to/checkpoint.ckpt")
            return
        print(f"  Found: {ckpt_path}")
    else:
        # Use provided path (resolve relative to original cwd)
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.is_absolute():
            ckpt_path = original_cwd / ckpt_path
        if not ckpt_path.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_path}")
            return
        print(f"\nUsing checkpoint: {ckpt_path}")

    # ==========================================================================
    # 2. Set up DataModule
    # ==========================================================================
    print("\nSetting up DataModule...")
    pl.seed_everything(cfg.seed, workers=True)

    data_dir = original_cwd / cfg.data.data_dir
    feature_type = cfg.data.feature_type

    # Build feature_params for graph feature slicing (same as train.py)
    feature_params: dict | None = None
    if feature_type == "graph" and cfg.model.name == "gcnn":
        in_channels = cfg.model.architecture.in_channels
        feature_params = {"feature_iterations": in_channels}

    datamodule = OPFDataModule(
        data_dir=str(data_dir),
        train_file=cfg.data.train_file,
        val_file=cfg.data.val_file,
        test_file=cfg.data.get("test_file"),
        batch_size=cfg.train.batch_size,
        num_workers=0,  # Use 0 for evaluation to avoid multiprocessing issues
        feature_type=feature_type,
        feature_params=feature_params,
        pin_memory=False,
    )

    # Setup for test stage (loads test data and norm_stats from training data)
    datamodule.setup(stage="fit")  # Load training data (for norm_stats)
    datamodule.setup(stage="test")  # Load test data
    print(f"  Data dir: {data_dir}")
    print(f"  Feature type: {feature_type}")
    print(f"  Test samples: {len(datamodule.val_dataset)}")

    # ==========================================================================
    # 3. Instantiate model and load checkpoint
    # ==========================================================================
    print("\nLoading model from checkpoint...")
    n_bus = cfg.data.n_bus
    n_gen = cfg.data.n_gen
    gen_bus_indices = list(cfg.data.gen_bus_indices)

    # Instantiate base model (required for OPFTask.load_from_checkpoint)
    model = instantiate_model(cfg, n_bus, n_gen)

    # Load OPFTask from checkpoint
    task = OPFTask.load_from_checkpoint(
        str(ckpt_path),
        model=model,
        lr=cfg.model.task.lr,
        kappa=cfg.model.task.kappa,
        weight_decay=cfg.model.task.weight_decay,
        gen_bus_indices=gen_bus_indices,
        n_bus=n_bus,
    )
    print(f"  Model: {cfg.model.name}")
    print(f"  Checkpoint loaded successfully")

    # ==========================================================================
    # 4. Run inference
    # ==========================================================================
    print("\nRunning inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    predictions, labels, auxiliary = run_inference(
        task, datamodule, device, feature_type=feature_type
    )
    n_samples = predictions["pg"].shape[0]
    print(f"  Processed {n_samples} samples")

    # ==========================================================================
    # 5. Denormalize predictions and labels
    # ==========================================================================
    print("\nDenormalizing outputs...")

    # Get norm_stats from dataset
    norm_stats = datamodule.train_dataset.norm_stats

    if norm_stats is not None:
        pg_mean = norm_stats["pg_mean"]
        pg_std = norm_stats["pg_std"]
        vg_mean = norm_stats["vg_mean"]
        vg_std = norm_stats["vg_std"]

        # Denormalize predictions
        pg_pred_pu = denormalize_tensor(predictions["pg"], pg_mean, pg_std)
        vg_pred_pu = denormalize_tensor(predictions["vg"], vg_mean, vg_std)

        # Denormalize labels
        pg_label_pu = denormalize_tensor(labels["pg"], pg_mean, pg_std)
        vg_label_pu = denormalize_tensor(labels["vg"], vg_mean, vg_std)

        print(f"  PG: mean={pg_mean.mean():.4f}, std={pg_std.mean():.4f}")
        print(f"  VG: mean={vg_mean.mean():.4f}, std={vg_std.mean():.4f}")
    else:
        print("  [WARN] No norm_stats found, assuming data is already in p.u.")
        pg_pred_pu = predictions["pg"]
        vg_pred_pu = predictions["vg"]
        pg_label_pu = labels["pg"]
        vg_label_pu = labels["vg"]

    # Convert to numpy for metrics
    pg_pred_np = pg_pred_pu.numpy()
    vg_pred_np = vg_pred_pu.numpy()
    pg_label_np = pg_label_pu.numpy()
    vg_label_np = vg_label_pu.numpy()

    # ==========================================================================
    # 6. Compute metrics
    # ==========================================================================
    print("\nComputing metrics...")

    # Probabilistic Accuracy (Eq. 37)
    pacc_pg = probabilistic_accuracy(pg_pred_np, pg_label_np, PG_THRESHOLD_PU)
    pacc_vg = probabilistic_accuracy(vg_pred_np, vg_label_np, VG_THRESHOLD_PU)

    # Standard regression metrics
    pg_metrics = compute_regression_metrics(pg_pred_np, pg_label_np)
    vg_metrics = compute_regression_metrics(vg_pred_np, vg_label_np)

    # Physics violation (Eq. 8)
    physics_violation_mw = float("nan")
    if auxiliary["operators"] is not None:
        print("  Computing physics violation...")
        operators = auxiliary["operators"]

        # Reconstruct G, B matrices
        if "G" in operators:
            G = operators["G"]
            B = operators["B"]
        else:
            g_ndiag = operators["g_ndiag"]
            b_ndiag = operators["b_ndiag"]
            g_diag = operators["g_diag"]
            b_diag = operators["b_diag"]

            # Handle batched operators (use first topology)
            if g_ndiag.dim() == 3:
                g_ndiag = g_ndiag[0]
                b_ndiag = b_ndiag[0]
                g_diag = g_diag[0]
                b_diag = b_diag[0]

            G = g_ndiag + torch.diag(g_diag)
            B = b_ndiag + torch.diag(b_diag)

        # Build generator-bus matrix
        gen_bus_matrix = build_gen_bus_matrix(n_bus, gen_bus_indices)

        # Compute physics violation
        physics_violation_mw = compute_physics_violation(
            pg_pred=pg_pred_pu,
            v_bus_pred=predictions["v_bus"],  # v_bus not normalized
            pd=auxiliary["pd"],
            qd=auxiliary["qd"],
            G=G,
            B=B,
            gen_bus_matrix=gen_bus_matrix,
        )

    # Constraint violations (physical limits)
    print("  Computing constraint violations...")
    limits = get_physical_limits(cfg.data.name)
    constraint_violations = compute_constraint_violations(
        pg_pred=pg_pred_np,
        vg_pred=vg_pred_np,
        limits=limits,
    )

    # ==========================================================================
    # 7. Print results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Model: {cfg.model.name.upper()}")
    print(f"Dataset: {cfg.data.name}")
    print(f"Test samples: {n_samples}")
    print(f"Checkpoint: {ckpt_path.name}")
    print("-" * 70)

    # Simple key=value format for easy parsing
    print(f"R2_PG={pg_metrics['R2']:.4f}")
    print(f"R2_VG={vg_metrics['R2']:.4f}")
    print(f"Pacc_PG={pacc_pg:.2f}")
    print(f"Pacc_VG={pacc_vg:.2f}")
    print(f"RMSE_PG={pg_metrics['RMSE']:.6f}")
    print(f"RMSE_VG={vg_metrics['RMSE']:.6f}")
    print(f"MAE_PG={pg_metrics['MAE']:.6f}")
    print(f"MAE_VG={vg_metrics['MAE']:.6f}")
    if not np.isnan(physics_violation_mw):
        print(f"Physics_MW={physics_violation_mw:.4f}")
    else:
        print("Physics_MW=N/A")

    # Constraint violations
    print("-" * 70)
    print("CONSTRAINT VIOLATIONS (Physical Limits)")
    print("-" * 70)
    pg_viol_rate = constraint_violations.get("pg_violation_rate", float("nan"))
    vg_viol_rate = constraint_violations.get("vg_violation_rate", float("nan"))
    if not np.isnan(pg_viol_rate):
        print(f"PG_Violation_Rate={pg_viol_rate:.2f}%")
        print(f"PG_Violation_Avg_MW={constraint_violations['pg_violation_avg_mw']:.2f}")
        print(f"PG_Violation_Max_MW={constraint_violations['pg_violation_max_mw']:.2f}")
    else:
        print("PG_Violation=N/A (no limits defined)")
    if not np.isnan(vg_viol_rate):
        print(f"VG_Violation_Rate={vg_viol_rate:.2f}%")
        print(f"VG_Violation_Avg_PU={constraint_violations['vg_violation_avg_pu']:.4f}")
        print(f"VG_Violation_Max_PU={constraint_violations['vg_violation_max_pu']:.4f}")
    else:
        print("VG_Violation=N/A (no limits defined)")

    print("=" * 70)

    # ==========================================================================
    # 8. Log evaluation results to CSV
    # ==========================================================================
    def safe_val(v):
        """Convert nan to None for CSV logging."""
        return None if (isinstance(v, float) and np.isnan(v)) else v

    metrics = {
        "R2_PG": pg_metrics["R2"],
        "R2_VG": vg_metrics["R2"],
        "Pacc_PG": pacc_pg,
        "Pacc_VG": pacc_vg,
        "RMSE_PG": pg_metrics["RMSE"],
        "RMSE_VG": vg_metrics["RMSE"],
        "MAE_PG": pg_metrics["MAE"],
        "MAE_VG": vg_metrics["MAE"],
        "Physics_Violation_MW": safe_val(physics_violation_mw),
        # Constraint violations
        "PG_Violation_Rate": safe_val(constraint_violations.get("pg_violation_rate")),
        "PG_Violation_Avg_MW": safe_val(constraint_violations.get("pg_violation_avg_mw")),
        "PG_Violation_Max_MW": safe_val(constraint_violations.get("pg_violation_max_mw")),
        "VG_Violation_Rate": safe_val(constraint_violations.get("vg_violation_rate")),
        "VG_Violation_Avg_PU": safe_val(constraint_violations.get("vg_violation_avg_pu")),
        "VG_Violation_Max_PU": safe_val(constraint_violations.get("vg_violation_max_pu")),
    }

    log_evaluation_to_csv(
        model_name=cfg.model.name,
        dataset_name=cfg.data.name,
        metrics=metrics,
        csv_path=original_cwd / "experiments_log.csv",
        ckpt_path=ckpt_path,
        n_samples=n_samples,
    )


if __name__ == "__main__":
    main()
