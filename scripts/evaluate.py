"""Evaluation script for deep_opf models using Hydra configuration.

Implements metrics from Gao et al. paper:
- Probabilistic Accuracy (Pacc): % of samples within threshold
  - PG: |pred - true| < 1.0 MW (threshold from paper Section V-A)
  - VG: |pred - true| < 0.001 p.u.
- Standard metrics: R², RMSE, MAE
- Physics violation: Mean active power mismatch

CRITICAL: Model outputs are Z-score normalized. Must denormalize before metrics.

Usage:
    python scripts/evaluate.py model=gcnn data=case39 checkpoint=/path/to/best.ckpt

    # Or use the best checkpoint from a training run
    python scripts/evaluate.py model=gcnn data=case39 checkpoint=outputs/2024-01-01/checkpoints/best.ckpt
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


# Constants from paper
BASE_MVA = 100.0  # Standard base power for p.u. conversion
PG_THRESHOLD_MW = 1.0  # Threshold for PG probabilistic accuracy (MW)
VG_THRESHOLD_PU = 0.001  # Threshold for VG probabilistic accuracy (p.u.)


def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Denormalize Z-score normalized values back to original scale.

    Formula: y_original = y_normalized * std + mean

    If std == 0, returns values + mean (per paper Eq. 36).

    Args:
        values: Normalized values
        mean: Mean used for normalization
        std: Standard deviation used for normalization

    Returns:
        Denormalized values in original scale (p.u.)
    """
    if std == 0 or std < 1e-10:
        return values + mean
    return values * std + mean


def probabilistic_accuracy(
    pred: np.ndarray, true: np.ndarray, threshold: float
) -> float:
    """
    Compute probabilistic accuracy (Eq. 37 from paper).

    p = P(|T - T_hat| < threshold)

    Args:
        pred: Predicted values
        true: True values
        threshold: Acceptance threshold

    Returns:
        Percentage of samples within threshold (0-100)
    """
    errors = np.abs(pred - true)
    within_threshold = np.sum(errors < threshold)
    total = errors.size
    return float((within_threshold / total) * 100.0)


def compute_metrics(pred: np.ndarray, true: np.ndarray, name: str) -> dict[str, float]:
    """
    Compute standard regression metrics.

    Args:
        pred: Predicted values (flattened)
        true: True values (flattened)
        name: Variable name for logging

    Returns:
        Dict with R², RMSE, MAE
    """
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    r2 = r2_score(true_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
    mae = mean_absolute_error(true_flat, pred_flat)

    return {
        f"{name}_R2": r2,
        f"{name}_RMSE_pu": rmse,
        f"{name}_MAE_pu": mae,
    }


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
    Compute mean physics violation (active power mismatch).

    Uses physics_loss to compute MSE between predicted PG and
    PG calculated from predicted voltage via power flow equations.

    Args:
        pg_pred: Predicted PG [N_samples, n_gen] in p.u.
        v_bus_pred: Predicted bus voltages [N_samples, n_bus, 2] in p.u.
        pd, qd: Power demands [N_samples, n_bus] in p.u.
        G, B: Admittance matrices [n_bus, n_bus]
        gen_bus_matrix: Generator-bus incidence [n_bus, n_gen]

    Returns:
        Mean physics violation in MW (sqrt(MSE) * BASE_MVA)
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
        # Convert MSE to RMSE in MW
        mse_pu = result["loss_p"].item()
        rmse_pu = np.sqrt(mse_pu)
        rmse_mw = rmse_pu * BASE_MVA

    return rmse_mw


def load_checkpoint(checkpoint_path: str, task: OPFTask) -> OPFTask:
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    task.load_state_dict(checkpoint["state_dict"])
    return task


def run_inference(
    task: OPFTask,
    datamodule: OPFDataModule,
    device: torch.device,
) -> tuple[dict, dict, dict]:
    """
    Run inference on test set and collect predictions.

    Returns:
        Tuple of (predictions, labels, batch_data) dicts
    """
    task.eval()
    task.to(device)

    all_pg_pred = []
    all_vg_pred = []
    all_v_bus_pred = []
    all_pg_label = []
    all_vg_label = []
    all_pd = []
    all_qd = []
    all_operators = []

    # Setup datamodule for training first (to get norm_stats), then test
    datamodule.setup(stage="fit")  # Load norm_stats from training data
    datamodule.setup(stage="test")  # Load test data
    dataloader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            if "operators" in batch and isinstance(batch["operators"], dict):
                batch["operators"] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch["operators"].items()
                }

            # Forward pass
            preds = task._forward(batch)

            # Collect predictions
            all_pg_pred.append(preds["pg"].cpu())
            all_vg_pred.append(preds["vg"].cpu())
            all_v_bus_pred.append(preds["v_bus"].cpu())

            # Collect labels
            all_pg_label.append(batch["pg_label"].cpu())  # type: ignore[union-attr]
            all_vg_label.append(batch["vg_label"].cpu())  # type: ignore[union-attr]

            # Collect batch data for physics loss
            all_pd.append(batch["pd"].cpu())  # type: ignore[union-attr]
            all_qd.append(batch["qd"].cpu())  # type: ignore[union-attr]
            if "operators" in batch and isinstance(batch["operators"], dict):
                all_operators.append(
                    {k: v.cpu() for k, v in batch["operators"].items()}
                )

    # Concatenate all batches
    predictions = {
        "pg": torch.cat(all_pg_pred, dim=0),
        "vg": torch.cat(all_vg_pred, dim=0),
        "v_bus": torch.cat(all_v_bus_pred, dim=0),
    }
    labels = {
        "pg": torch.cat(all_pg_label, dim=0),
        "vg": torch.cat(all_vg_label, dim=0),
    }
    batch_data = {
        "pd": torch.cat(all_pd, dim=0),
        "qd": torch.cat(all_qd, dim=0),
    }

    # Stack operators (use first batch's topology matrices)
    if all_operators:
        batch_data["operators"] = all_operators[0]

    return predictions, labels, batch_data


def evaluate_model(
    predictions: dict,
    labels: dict,
    batch_data: dict,
    norm_stats: dict | None,
    gen_bus_indices: list[int],
    n_bus: int,
) -> dict[str, float]:
    """
    Evaluate model predictions with denormalization.

    Args:
        predictions: Dict with pg, vg, v_bus tensors
        labels: Dict with pg, vg tensors
        batch_data: Dict with pd, qd, operators
        norm_stats: Normalization statistics (mean, std for each variable)
        gen_bus_indices: Generator bus indices
        n_bus: Number of buses

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # Convert to numpy
    pg_pred = predictions["pg"].numpy()
    vg_pred = predictions["vg"].numpy()
    v_bus_pred = predictions["v_bus"].numpy()
    pg_label = labels["pg"].numpy()
    vg_label = labels["vg"].numpy()

    # Denormalize if norm_stats available
    if norm_stats is not None:
        print("\nDenormalizing predictions using norm_stats...")

        # Get normalization parameters (may be torch.Tensor or scalar)
        pg_mean = norm_stats.get("pg_mean", 0.0)
        pg_std = norm_stats.get("pg_std", 1.0)
        vg_mean = norm_stats.get("vg_mean", 0.0)
        vg_std = norm_stats.get("vg_std", 1.0)

        # Convert torch tensors to numpy
        if isinstance(pg_mean, torch.Tensor):
            pg_mean = pg_mean.numpy()
        if isinstance(pg_std, torch.Tensor):
            pg_std = pg_std.numpy()
        if isinstance(vg_mean, torch.Tensor):
            vg_mean = vg_mean.numpy()
        if isinstance(vg_std, torch.Tensor):
            vg_std = vg_std.numpy()

        # Denormalize: y_orig = y_norm * std + mean
        # Note: Dataset applies: y_norm = (y_orig - mean) / (std + 1e-8)
        pg_pred_pu = pg_pred * (pg_std + 1e-8) + pg_mean
        vg_pred_pu = vg_pred * (vg_std + 1e-8) + vg_mean
        pg_label_pu = pg_label * (pg_std + 1e-8) + pg_mean
        vg_label_pu = vg_label * (vg_std + 1e-8) + vg_mean

        print(f"  PG: mean={pg_mean}, std={pg_std}")
        print(f"  VG: mean={vg_mean}, std={vg_std}")
    else:
        print("\nNo norm_stats found. Assuming outputs are already in p.u.")
        pg_pred_pu = pg_pred
        vg_pred_pu = vg_pred
        pg_label_pu = pg_label
        vg_label_pu = vg_label

    # Convert PG to MW for probabilistic accuracy
    pg_pred_mw = pg_pred_pu * BASE_MVA
    pg_label_mw = pg_label_pu * BASE_MVA

    # =========================================================================
    # Probabilistic Accuracy (Eq. 37 from paper)
    # =========================================================================
    pacc_pg = probabilistic_accuracy(pg_pred_mw, pg_label_mw, PG_THRESHOLD_MW)
    pacc_vg = probabilistic_accuracy(vg_pred_pu, vg_label_pu, VG_THRESHOLD_PU)

    metrics["Pacc_PG_%"] = pacc_pg
    metrics["Pacc_VG_%"] = pacc_vg

    # =========================================================================
    # Standard Metrics (R², RMSE, MAE) in p.u.
    # =========================================================================
    pg_metrics = compute_metrics(pg_pred_pu, pg_label_pu, "PG")
    vg_metrics = compute_metrics(vg_pred_pu, vg_label_pu, "VG")
    metrics.update(pg_metrics)
    metrics.update(vg_metrics)

    # =========================================================================
    # Physics Violation (Active Power Mismatch)
    # =========================================================================
    if "operators" in batch_data:
        print("\nComputing physics violation...")

        # Build gen-bus matrix
        gen_bus_matrix = build_gen_bus_matrix(n_bus, gen_bus_indices)

        # Get G, B matrices from operators
        operators = batch_data["operators"]
        if "G" in operators:
            G = operators["G"]
            B = operators["B"]
        else:
            # Reconstruct from diagonal and off-diagonal
            g_ndiag = operators["g_ndiag"]
            b_ndiag = operators["b_ndiag"]
            g_diag = operators["g_diag"]
            b_diag = operators["b_diag"]

            # Handle batched (use first sample)
            if g_ndiag.dim() == 3:
                g_ndiag = g_ndiag[0]
                b_ndiag = b_ndiag[0]
                g_diag = g_diag[0]
                b_diag = b_diag[0]

            G = g_ndiag + torch.diag(g_diag)
            B = b_ndiag + torch.diag(b_diag)

        # Convert predictions back to tensors (denormalized, in p.u.)
        pg_tensor = torch.from_numpy(pg_pred_pu).float()
        v_bus_tensor = predictions["v_bus"]  # v_bus may not be normalized

        # Compute physics violation
        phys_violation_mw = compute_physics_violation(
            pg_pred=pg_tensor,
            v_bus_pred=v_bus_tensor,
            pd=batch_data["pd"],
            qd=batch_data["qd"],
            G=G,
            B=B,
            gen_bus_matrix=gen_bus_matrix,
        )
        metrics["Physics_Violation_MW"] = phys_violation_mw
    else:
        metrics["Physics_Violation_MW"] = float("nan")

    return metrics


def print_results_table(metrics: dict, cfg: DictConfig) -> None:
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Model: {cfg.model.name}")
    print(f"Dataset: {cfg.data.name}")
    print("=" * 70)

    # Probabilistic Accuracy Table
    pacc_data = [
        ["PG", f"{metrics['Pacc_PG_%']:.2f}%", f"< {PG_THRESHOLD_MW} MW"],
        ["VG", f"{metrics['Pacc_VG_%']:.2f}%", f"< {VG_THRESHOLD_PU} p.u."],
    ]
    print("\n[Probabilistic Accuracy] (Paper Eq. 37):")
    print(
        tabulate(
            pacc_data, headers=["Variable", "Accuracy", "Threshold"], tablefmt="grid"
        )
    )

    # Standard Metrics Table
    std_data = [
        [
            "PG",
            f"{metrics['PG_R2']:.4f}",
            f"{metrics['PG_RMSE_pu']:.6f}",
            f"{metrics['PG_MAE_pu']:.6f}",
        ],
        [
            "VG",
            f"{metrics['VG_R2']:.4f}",
            f"{metrics['VG_RMSE_pu']:.6f}",
            f"{metrics['VG_MAE_pu']:.6f}",
        ],
    ]
    print("\n[Standard Metrics] (in p.u.):")
    print(
        tabulate(std_data, headers=["Variable", "R^2", "RMSE", "MAE"], tablefmt="grid")
    )

    # Physics Violation
    print("\n[Physics Consistency]:")
    if not np.isnan(metrics.get("Physics_Violation_MW", float("nan"))):
        print(
            f"   Active Power Mismatch (RMSE): {metrics['Physics_Violation_MW']:.4f} MW"
        )
    else:
        print("   Physics violation not computed (missing operators)")

    print("\n" + "=" * 70)


def cleanup_lightning_logs(keep_versions: int = 5) -> None:
    """
    Clean up old lightning_logs versions, keeping only the most recent ones.

    Args:
        keep_versions: Number of recent versions to keep
    """
    logs_dir = Path("lightning_logs")
    if not logs_dir.exists():
        return

    # Get all version directories
    versions = sorted(logs_dir.glob("version_*"), key=lambda p: p.stat().st_mtime)

    # Remove old versions
    if len(versions) > keep_versions:
        for old_version in versions[:-keep_versions]:
            print(f"Removing old log: {old_version}")
            import shutil

            shutil.rmtree(old_version)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function.

    Args:
        cfg: Hydra configuration object (must include 'checkpoint' key)
    """
    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    # Check for checkpoint path
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path is None:
        print("\n❌ ERROR: No checkpoint specified!")
        print("Usage: python scripts/evaluate.py checkpoint=/path/to/model.ckpt")
        return

    # Resolve checkpoint path relative to original cwd
    original_cwd = Path(hydra.utils.get_original_cwd())
    checkpoint_path = original_cwd / checkpoint_path
    if not checkpoint_path.exists():
        print(f"\n❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return

    print(f"\n[OK] Loading checkpoint: {checkpoint_path}")

    # Set seed
    pl.seed_everything(cfg.seed, workers=True)

    # Extract data parameters
    n_bus = cfg.data.n_bus
    n_gen = cfg.data.n_gen
    gen_bus_indices = list(cfg.data.gen_bus_indices)

    # Instantiate datamodule
    data_dir = original_cwd / cfg.data.data_dir
    feature_type = cfg.data.feature_type

    datamodule = OPFDataModule(
        data_dir=str(data_dir),
        train_file=cfg.data.train_file,
        val_file=cfg.data.val_file,
        test_file=cfg.data.get("test_file"),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.get("num_workers", 0),  # Use 0 for evaluation
        feature_type=feature_type,
        pin_memory=False,
    )

    print(f"[OK] DataModule: {data_dir}, feature_type={feature_type}")

    # Instantiate model
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

    print(f"[OK] Model: {model_name}")

    # Create task and load checkpoint
    task = OPFTask(
        model=model,
        lr=cfg.model.task.lr,
        kappa=cfg.model.task.kappa,
        weight_decay=cfg.model.task.weight_decay,
        gen_bus_indices=gen_bus_indices,
        n_bus=n_bus,
    )

    task = load_checkpoint(str(checkpoint_path), task)
    print("[OK] Checkpoint loaded")

    # Run inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Device: {device}")

    predictions, labels, batch_data = run_inference(task, datamodule, device)
    print(f"[OK] Inference complete: {predictions['pg'].shape[0]} samples")

    # Get normalization stats
    norm_stats = datamodule.get_norm_stats()

    # Evaluate
    metrics = evaluate_model(
        predictions=predictions,
        labels=labels,
        batch_data=batch_data,
        norm_stats=norm_stats,
        gen_bus_indices=gen_bus_indices,
        n_bus=n_bus,
    )

    # Print results
    print_results_table(metrics, cfg)

    # Cleanup old lightning logs
    cleanup_lightning_logs(keep_versions=5)
    print("\n[OK] Cleaned up old lightning_logs (kept last 5 versions)")


if __name__ == "__main__":
    main()
