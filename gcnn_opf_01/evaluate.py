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

from dataset import OPFDataset
from model_01 import GCNN_OPF_01
from config_model_01 import ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GCNN OPF model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="gcnn_opf_01/results/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="gcnn_opf_01/data",
        help="Directory containing test data",
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
    }


def evaluate_model(model, test_loader, device):
    """Run evaluation on test set."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            e_0_k = batch["e_0_k"].to(device)
            f_0_k = batch["f_0_k"].to(device)
            pd = batch["pd"].to(device)
            qd = batch["qd"].to(device)
            gen_labels = batch["gen_label"].to(device)

            # Get operators
            ops = batch["operators"]
            g_ndiag = ops["g_ndiag"][0].to(device)
            b_ndiag = ops["b_ndiag"][0].to(device)
            g_diag = ops["g_diag"][0].to(device)
            b_diag = ops["b_diag"][0].to(device)

            # Forward pass
            gen_out, v_out = model(
                e_0_k, f_0_k, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag
            )

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
    print("GCNN OPF MODEL EVALUATION")
    print("=" * 80)

    # Setup device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Load test dataset
    data_dir = Path(args.data_dir)
    print(f"\nLoading test dataset from: {data_dir}")

    test_dataset = OPFDataset(
        data_path=data_dir / "samples_test.npz",
        topo_operators_path=data_dir / "topology_operators.npz",
        norm_stats_path=data_dir / "norm_stats.npz",
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

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = GCNN_OPF_01().to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Evaluate
    print("\nRunning evaluation...")
    predictions, labels = evaluate_model(model, test_loader, device)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")

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
