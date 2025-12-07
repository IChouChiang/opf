"""
Evaluate trained GCNN model on test dataset.

Usage:
    python gcnn_opf_01/evaluate.py --model_path gcnn_opf_01/results/best_model.pth
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_03 import OPFDataset03
from model_03 import AdmittanceDNN
from config_03 import ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepOPF-FT Baseline")
    parser.add_argument(
        "--model_path",
        type=str,
        default="dnn_opf_03/results/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="gcnn_opf_01/data",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default=None,
        help="Path to normalization stats (default: data_dir/norm_stats.npz)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=None, help="Override hidden dimension"
    )
    parser.add_argument(
        "--n_hidden_layers",
        type=int,
        default=None,
        help="Override number of hidden layers",
    )
    return parser.parse_args()


def compute_metrics(predictions, labels):
    """Compute evaluation metrics.

    Args:
        predictions: [N, N_GEN, 2] (PG, VG)
        labels: [N, N_GEN, 2] (PG, VG)

    Returns:
        dict of metrics
    """
    # Separate PG and VG
    pg_pred = predictions[:, :, 0]  # [N, N_GEN]
    vg_pred = predictions[:, :, 1]
    pg_label = labels[:, :, 0]
    vg_label = labels[:, :, 1]

    # MSE
    mse_pg = np.mean((pg_pred - pg_label) ** 2)
    mse_vg = np.mean((vg_pred - vg_label) ** 2)
    mse_total = (mse_pg + mse_vg) / 2

    # RMSE
    rmse_pg = np.sqrt(mse_pg)
    rmse_vg = np.sqrt(mse_vg)

    # MAE
    mae_pg = np.mean(np.abs(pg_pred - pg_label))
    mae_vg = np.mean(np.abs(vg_pred - vg_label))

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    epsilon = 1e-8
    mape_pg = np.mean(np.abs((pg_pred - pg_label) / (pg_label + epsilon))) * 100
    mape_vg = np.mean(np.abs((vg_pred - vg_label) / (vg_label + epsilon))) * 100

    # R² score (coefficient of determination)
    ss_res_pg = np.sum((pg_label - pg_pred) ** 2)
    ss_tot_pg = np.sum((pg_label - np.mean(pg_label)) ** 2)
    r2_pg = 1 - (ss_res_pg / (ss_tot_pg + epsilon))

    ss_res_vg = np.sum((vg_label - vg_pred) ** 2)
    ss_tot_vg = np.sum((vg_label - np.mean(vg_label)) ** 2)
    r2_vg = 1 - (ss_res_vg / (ss_tot_vg + epsilon))

    # Max error
    max_error_pg = np.max(np.abs(pg_pred - pg_label))
    max_error_vg = np.max(np.abs(vg_pred - vg_label))

    # Probabilistic accuracy (Formula 37)
    # P(|T - T_hat| < thr) - probability of error below threshold
    # Thresholds: 1 MW for PG (assumes labels are in p.u., convert threshold)
    # Note: If baseMVA=100, 1MW = 0.01 p.u.; adjust if needed
    thr_pg = 0.01  # 1 MW in p.u. (assuming baseMVA=100)
    thr_vg = 0.001  # 0.001 p.u. for voltage

    error_pg = np.abs(pg_pred - pg_label)
    error_vg = np.abs(vg_pred - vg_label)

    p_pg = np.mean(error_pg < thr_pg) * 100  # Percentage
    p_vg = np.mean(error_vg < thr_vg) * 100  # Percentage

    return {
        "mse_pg": mse_pg,
        "mse_vg": mse_vg,
        "mse_total": mse_total,
        "rmse_pg": rmse_pg,
        "rmse_vg": rmse_vg,
        "mae_pg": mae_pg,
        "mae_vg": mae_vg,
        "mape_pg": mape_pg,
        "mape_vg": mape_vg,
        "r2_pg": r2_pg,
        "r2_vg": r2_vg,
        "max_error_pg": max_error_pg,
        "max_error_vg": max_error_vg,
        "p_pg": p_pg,
        "p_vg": p_vg,
    }


def evaluate_model(model, test_loader, device):
    """Run evaluation on test set."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_vec = batch["input"].to(device)
            gen_labels = batch["gen_label"].to(device)

            # Forward pass
            gen_out, v_out = model(input_vec)

            # Store results
            all_predictions.append(gen_out.cpu().numpy())
            all_labels.append(gen_labels.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # [N_test, N_GEN, 2]
    labels = np.concatenate(all_labels, axis=0)

    return predictions, labels


def main():
    args = parse_args()

    print("=" * 80)
    print("DeepOPF-FT Baseline EVALUATION")
    print("=" * 80)

    # Setup device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Load test dataset
    data_dir = Path(args.data_dir)
    print(f"\nLoading test dataset from: {data_dir}")

    # Determine norm stats path
    norm_stats_path = (
        Path(args.norm_stats_path)
        if args.norm_stats_path
        else data_dir / "norm_stats.npz"
    )
    print(f"Using normalization stats from: {norm_stats_path}")

    test_dataset = OPFDataset03(
        data_path=data_dir / "samples_test.npz",
        topo_operators_path=data_dir / "topology_operators.npz",
        norm_stats_path=norm_stats_path,
        normalize=True,
        split="test",
    )

    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize model
    config = ModelConfig()

    # Apply overrides
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
        print(f"Overriding hidden_dim: {config.hidden_dim}")

    if args.n_hidden_layers is not None:
        config.n_hidden_layers = args.n_hidden_layers
        print(f"Overriding n_hidden_layers: {config.n_hidden_layers}")

    model = AdmittanceDNN(config)

    # Load model weights
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Evaluate
    print("\nRunning evaluation...")
    predictions, labels = evaluate_model(model, test_loader, device)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")

    # Denormalize if needed
    if test_dataset.norm_stats is not None:
        print("\nDenormalizing predictions and labels...")
        stats = test_dataset.norm_stats

        # Extract stats as numpy arrays
        pg_mean = stats["pg_mean"].numpy()
        pg_std = stats["pg_std"].numpy()
        vg_mean = stats["vg_mean"].numpy()
        vg_std = stats["vg_std"].numpy()

        # Denormalize PG (index 0)
        predictions[:, :, 0] = predictions[:, :, 0] * (pg_std + 1e-8) + pg_mean
        labels[:, :, 0] = labels[:, :, 0] * (pg_std + 1e-8) + pg_mean

        # Denormalize VG (index 1)
        predictions[:, :, 1] = predictions[:, :, 1] * (vg_std + 1e-8) + vg_mean
        labels[:, :, 1] = labels[:, :, 1] * (vg_std + 1e-8) + vg_mean

    # Compute metrics
    metrics = compute_metrics(predictions, labels)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\n--- Generator Active Power (PG) ---")
    print(f"  MSE:        {metrics['mse_pg']:.6f}")
    print(f"  RMSE:       {metrics['rmse_pg']:.6f}")
    print(f"  MAE:        {metrics['mae_pg']:.6f}")
    print(f"  MAPE:       {metrics['mape_pg']:.2f}%")
    print(f"  R²:         {metrics['r2_pg']:.6f}")
    print(f"  Max Error:  {metrics['max_error_pg']:.6f}")

    print("\n--- Generator Voltage (VG) ---")
    print(f"  MSE:        {metrics['mse_vg']:.6f}")
    print(f"  RMSE:       {metrics['rmse_vg']:.6f}")
    print(f"  MAE:        {metrics['mae_vg']:.6f}")
    print(f"  MAPE:       {metrics['mape_vg']:.2f}%")
    print(f"  R²:         {metrics['r2_vg']:.6f}")
    print(f"  Max Error:  {metrics['max_error_vg']:.6f}")

    print("\n--- Overall ---")
    print(f"  Total MSE:  {metrics['mse_total']:.6f}")

    print("\n--- Probabilistic Accuracy (Formula 37) ---")
    print(f"  P_PG:       {metrics['p_pg']:.2f}% (errors < 0.01 p.u. / 1 MW)")
    print(f"  P_VG:       {metrics['p_vg']:.2f}% (errors < 0.001 p.u.)")

    # Per-generator statistics
    print("\n--- Per-Generator Statistics (PG) ---")
    config = ModelConfig()
    for g in range(config.n_gen):
        pg_pred_g = predictions[:, g, 0]
        pg_label_g = labels[:, g, 0]

        mse_g = np.mean((pg_pred_g - pg_label_g) ** 2)
        mae_g = np.mean(np.abs(pg_pred_g - pg_label_g))

        print(
            f"  Gen {g}: MSE={mse_g:.6f}, MAE={mae_g:.6f}, "
            f"Mean_pred={pg_pred_g.mean():.4f}, Mean_true={pg_label_g.mean():.4f}"
        )

    # Save results
    results_path = Path(args.model_path).parent / "evaluation_results.npz"
    np.savez(
        results_path,
        predictions=predictions,
        labels=labels,
        metrics=metrics,
    )
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
