"""
Evaluate trained Hasan DNN (M8) model on test dataset.

Usage:
    python dnn_opf_02/evaluate_02.py --model_path dnn_opf_02/results/best_model.pth
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dnn_opf_02.dataset_02 import OPFDataset02
from dnn_opf_02.model_02 import HasanDNN

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hasan DNN OPF model")
    parser.add_argument("--model_path", type=str, default="dnn_opf_02/results/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="gcnn_opf_01/data", help="Directory containing test data")
    parser.add_argument("--means_dir", type=str, default="dnn_opf_02/data", help="Directory containing topology means")
    parser.add_argument("--norm_stats_path", type=str, default=None, help="Path to normalization stats")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def compute_metrics(predictions, labels):
    """Compute evaluation metrics."""
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

    # MAPE
    epsilon = 1e-8
    mape_pg = np.mean(np.abs((pg_pred - pg_label) / (pg_label + epsilon))) * 100
    mape_vg = np.mean(np.abs((vg_pred - vg_label) / (vg_label + epsilon))) * 100

    # R² score
    ss_res_pg = np.sum((pg_label - pg_pred) ** 2)
    ss_tot_pg = np.sum((pg_label - np.mean(pg_label)) ** 2)
    r2_pg = 1 - (ss_res_pg / (ss_tot_pg + epsilon))

    ss_res_vg = np.sum((vg_label - vg_pred) ** 2)
    ss_tot_vg = np.sum((vg_label - np.mean(vg_label)) ** 2)
    r2_vg = 1 - (ss_res_vg / (ss_tot_vg + epsilon))

    # Max error
    max_error_pg = np.max(np.abs(pg_pred - pg_label))
    max_error_vg = np.max(np.abs(vg_pred - vg_label))

    # Probabilistic accuracy
    thr_pg = 0.01  # 1 MW in p.u.
    thr_vg = 0.001  # 0.001 p.u.

    error_pg = np.abs(pg_pred - pg_label)
    error_vg = np.abs(vg_pred - vg_label)

    p_pg = np.mean(error_pg < thr_pg) * 100
    p_vg = np.mean(error_vg < thr_vg) * 100

    return {
        "mse_pg": mse_pg, "mse_vg": mse_vg, "mse_total": mse_total,
        "rmse_pg": rmse_pg, "rmse_vg": rmse_vg,
        "mae_pg": mae_pg, "mae_vg": mae_vg,
        "mape_pg": mape_pg, "mape_vg": mape_vg,
        "r2_pg": r2_pg, "r2_vg": r2_vg,
        "max_error_pg": max_error_pg, "max_error_vg": max_error_vg,
        "p_pg": p_pg, "p_vg": p_vg,
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            gen_labels = batch["gen_label"].to(device)

            gen_out, _ = model(x)

            all_predictions.append(gen_out.cpu().numpy())
            all_labels.append(gen_labels.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return predictions, labels

def main():
    args = parse_args()
    print("=" * 80)
    print("HASAN DNN OPF MODEL EVALUATION")
    print("=" * 80)

    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    data_dir = Path(args.data_dir)
    means_dir = Path(args.means_dir)
    norm_stats_path = Path(args.norm_stats_path) if args.norm_stats_path else data_dir / "norm_stats.npz"

    test_dataset = OPFDataset02(
        data_path=data_dir / "samples_test.npz",
        topo_means_path=means_dir / "topology_voltage_means.npz",
        topo_operators_path=data_dir / "topology_operators.npz",
        norm_stats_path=norm_stats_path,
        normalize=True,
        split="test"
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    sample = test_dataset[0]
    n_bus = len(sample["pd"])
    n_gen = len(sample["pg_label"])
    
    model = HasanDNN(n_bus=n_bus, n_gen=n_gen)
    
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print("\nRunning evaluation...")
    predictions, labels = evaluate_model(model, test_loader, device)

    # Denormalize
    if test_dataset.norm_stats is not None:
        print("\nDenormalizing predictions and labels...")
        stats = test_dataset.norm_stats
        pg_mean = stats["pg_mean"].numpy()
        pg_std = stats["pg_std"].numpy()
        vg_mean = stats["vg_mean"].numpy()
        vg_std = stats["vg_std"].numpy()

        predictions[:, :, 0] = predictions[:, :, 0] * (pg_std + 1e-8) + pg_mean
        labels[:, :, 0] = labels[:, :, 0] * (pg_std + 1e-8) + pg_mean
        predictions[:, :, 1] = predictions[:, :, 1] * (vg_std + 1e-8) + vg_mean
        labels[:, :, 1] = labels[:, :, 1] * (vg_std + 1e-8) + vg_mean

    metrics = compute_metrics(predictions, labels)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print("\n--- Generator Active Power (PG) ---")
    print(f"  MSE:        {metrics['mse_pg']:.6f}")
    print(f"  RMSE:       {metrics['rmse_pg']:.6f}")
    print(f"  MAE:        {metrics['mae_pg']:.6f}")
    print(f"  MAPE:       {metrics['mape_pg']:.2f}%")
    print(f"  R²:         {metrics['r2_pg']:.6f}")
    print(f"  P_PG:       {metrics['p_pg']:.2f}%")

    print("\n--- Generator Voltage (VG) ---")
    print(f"  MSE:        {metrics['mse_vg']:.6f}")
    print(f"  RMSE:       {metrics['rmse_vg']:.6f}")
    print(f"  MAE:        {metrics['mae_vg']:.6f}")
    print(f"  MAPE:       {metrics['mape_vg']:.2f}%")
    print(f"  R²:         {metrics['r2_vg']:.6f}")
    print(f"  P_VG:       {metrics['p_vg']:.2f}%")

    # Save results
    results_path = Path(args.model_path).parent / "evaluation_results.npz"
    np.savez(results_path, predictions=predictions, labels=labels, metrics=metrics)
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
