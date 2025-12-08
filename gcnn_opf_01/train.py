"""
Training script for GCNN OPF Model 01.

Usage:
    python gcnn_opf_01/train.py --config gcnn_opf_01/configs/base.json
"""

import sys
from pathlib import Path
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add gcnn_opf_01 to path
sys.path.insert(0, str(Path(__file__).parent))

from model_01 import GCNN_OPF_01
from model_nodewise import GCNN_OPF_NodeWise
from loss_model_01 import correlative_loss_pg
from dataset import OPFDataset
from config_model_01 import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train GCNN OPF Model")

    # Config path
    parser.add_argument(
        "--config",
        type=str,
        default="gcnn_opf_01/configs/base.json",
        help="Path to JSON configuration file",
    )

    # Data paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="gcnn_opf_01/data",
        help="Directory containing dataset files",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="gcnn_opf_01/results",
        help="Directory to save results",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Batch size for training (paper uses 10)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )

    # Loss weights
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.1,
        help="Physics loss weight (correlative term)",
    )
    parser.add_argument(
        "--use_physics_loss",
        action="store_true",
        default=True,
        help="Use physics-informed correlative loss",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (epochs)"
    )

    # Two-stage training
    parser.add_argument(
        "--two_stage",
        action="store_true",
        help="Enable two-stage training (Phase 1: Supervised, Phase 2: Physics-Informed)",
    )
    parser.add_argument(
        "--phase1_epochs", type=int, default=25, help="Epochs for Phase 1 (Supervised)"
    )
    parser.add_argument(
        "--phase2_epochs", type=int, default=25, help="Epochs for Phase 2 (Physics)"
    )
    parser.add_argument(
        "--phase2_lr", type=float, default=1e-4, help="Learning rate for Phase 2"
    )
    parser.add_argument(
        "--phase2_kappa",
        type=float,
        default=1.0,
        help="Physics loss weight for Phase 2",
    )

    # Logging
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log every N batches"
    )

    return parser.parse_args()


def run_training_phase(
    phase_name,
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    A_g2b,
    epochs,
    kappa,
    use_physics,
    results_dir,
    log_interval,
    patience,
    start_epoch=1,
):
    """Run a single training phase."""
    print(f"\n{'='*80}")
    print(f"STARTING PHASE: {phase_name}")
    print(f"Epochs: {epochs}, Kappa: {kappa}, LR: {optimizer.param_groups[0]['lr']}")
    print(f"{'='*80}")

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{start_epoch + epochs - 1} ({phase_name})")
        print("-" * 80)

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            A_g2b,
            kappa=kappa,
            use_physics=use_physics,
            log_interval=log_interval,
        )

        # Validate
        val_metrics = validate(
            model,
            val_loader,
            device,
            A_g2b,
            kappa=kappa,
            use_physics=use_physics,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"\n  Train: Loss={train_metrics['loss']:.6f} "
            f"(Sup={train_metrics['sup_loss']:.6f}, Phys={train_metrics['phys_loss']:.6f})"
        )
        print(
            f"  Val:   Loss={val_metrics['loss']:.6f} "
            f"(Sup={val_metrics['sup_loss']:.6f}, Phys={val_metrics['phys_loss']:.6f})"
        )
        print(f"  Time: {epoch_time:.1f}s")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Save best model for this phase
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                model.state_dict(), results_dir / f"best_model_{phase_name.lower()}.pth"
            )
            print(f"  → Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n Early stopping triggered in {phase_name}")
                break

    return history


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    A_g2b,
    kappa=0.1,
    use_physics=True,
    log_interval=10,
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_sup_loss = 0.0
    total_phys_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        e_0_k = batch["e_0_k"].to(device)  # [B, N_BUS, k]
        f_0_k = batch["f_0_k"].to(device)
        pd = batch["pd"].to(device)  # [B, N_BUS]
        qd = batch["qd"].to(device)
        gen_labels = batch["gen_label"].to(device)  # [B, N_GEN, 2]

        # Get operators for this batch (assume same topology for simplicity, or handle mixed)
        # For now, use first sample's operators (all samples in batch should have same topology)
        ops = batch["operators"]
        g_ndiag = ops["g_ndiag"][0].to(device)  # [N_BUS, N_BUS]
        b_ndiag = ops["b_ndiag"][0].to(device)
        g_diag = ops["g_diag"][0].to(device)  # [N_BUS]
        b_diag = ops["b_diag"][0].to(device)

        # Reconstruct full G and B for physics loss (G = g_diag + g_ndiag)
        G = g_ndiag + torch.diag(g_diag)
        B = b_ndiag + torch.diag(b_diag)

        # Forward pass
        optimizer.zero_grad()
        gen_out, v_out = model(e_0_k, f_0_k, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)

        # Supervised loss (MSE on PG and VG)
        sup_loss = nn.functional.mse_loss(gen_out, gen_labels)

        # Physics loss (optional)
        if use_physics:
            phys_loss_total = 0.0
            # Compute physics loss for each sample in batch
            for i in range(e_0_k.size(0)):
                loss_t, loss_s, loss_p = correlative_loss_pg(
                    gen_out[i],  # [N_GEN, 2] (squeeze batch dim)
                    v_out[i],  # [N_BUS, 2]
                    gen_labels[i, :, 0],  # [N_GEN] PG labels
                    gen_labels[i, :, 1],  # [N_GEN] VG labels
                    pd[i],  # [N_BUS]
                    G,
                    B,  # [N_BUS, N_BUS]
                    A_g2b,  # [N_BUS, N_GEN]
                    kappa=0.0,  # Don't double-apply kappa, we scale manually
                )
                phys_loss_total += loss_p.item()
            phys_loss = phys_loss_total / e_0_k.size(0)  # Average

            loss = sup_loss + kappa * torch.tensor(phys_loss, device=device)
        else:
            phys_loss = 0.0
            loss = sup_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate stats
        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_phys_loss += (
            phys_loss.item() if isinstance(phys_loss, torch.Tensor) else phys_loss
        )
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_sup = total_sup_loss / n_batches
            avg_phys = total_phys_loss / n_batches
            print(
                f"  Batch {batch_idx+1}/{len(train_loader)}: "
                f"Loss={avg_loss:.6f} (Sup={avg_sup:.6f}, Phys={avg_phys:.6f})"
            )

    return {
        "loss": total_loss / n_batches,
        "sup_loss": total_sup_loss / n_batches,
        "phys_loss": total_phys_loss / n_batches,
    }


def validate(model, val_loader, device, A_g2b, kappa=0.1, use_physics=True):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_sup_loss = 0.0
    total_phys_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            e_0_k = batch["e_0_k"].to(device)
            f_0_k = batch["f_0_k"].to(device)
            pd = batch["pd"].to(device)
            qd = batch["qd"].to(device)
            gen_labels = batch["gen_label"].to(device)

            ops = batch["operators"]
            g_ndiag = ops["g_ndiag"][0].to(device)
            b_ndiag = ops["b_ndiag"][0].to(device)
            g_diag = ops["g_diag"][0].to(device)
            b_diag = ops["b_diag"][0].to(device)

            # Reconstruct full G and B
            G = g_ndiag + torch.diag(g_diag)
            B = b_ndiag + torch.diag(b_diag)

            gen_out, v_out = model(
                e_0_k, f_0_k, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag
            )

            sup_loss = nn.functional.mse_loss(gen_out, gen_labels)

            if use_physics:
                phys_loss_total = 0.0
                for i in range(e_0_k.size(0)):
                    loss_t, loss_s, loss_p = correlative_loss_pg(
                        gen_out[i],  # [N_GEN, 2]
                        v_out[i],  # [N_BUS, 2]
                        gen_labels[i, :, 0],  # [N_GEN]
                        gen_labels[i, :, 1],  # [N_GEN]
                        pd[i],
                        G,
                        B,
                        A_g2b,
                        kappa=0.0,
                    )
                    phys_loss_total += loss_p.item()
                phys_loss = phys_loss_total / e_0_k.size(0)
                loss = sup_loss + kappa * torch.tensor(phys_loss, device=device)
            else:
                phys_loss = 0.0
                loss = sup_loss

            total_loss += loss.item()
            total_sup_loss += sup_loss.item()
            total_phys_loss += phys_loss if isinstance(phys_loss, float) else 0.0
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "sup_loss": total_sup_loss / n_batches,
        "phys_loss": total_phys_loss / n_batches,
    }


def main():
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    model_config, training_config = load_config(args.config)

    # Override args with config values
    args.batch_size = training_config.batch_size
    args.lr = training_config.learning_rate
    args.weight_decay = training_config.weight_decay
    args.epochs = training_config.epochs
    args.patience = training_config.early_stopping_patience
    args.kappa = training_config.kappa
    args.use_physics_loss = training_config.use_physics_loss
    args.two_stage = training_config.two_stage

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load datasets
    data_dir = Path(args.data_dir)
    print("Loading datasets...")
    train_dataset = OPFDataset(
        data_path=data_dir / "samples_train.npz",
        topo_operators_path=data_dir / "topology_operators.npz",
        norm_stats_path=data_dir / "norm_stats.npz",
        normalize=True,
        split="train",
        feature_iterations=model_config.feature_iterations,
    )

    val_dataset = OPFDataset(
        data_path=data_dir / "samples_test.npz",
        topo_operators_path=data_dir / "topology_operators.npz",
        norm_stats_path=data_dir / "norm_stats.npz",
        normalize=True,
        split="test",
        feature_iterations=model_config.feature_iterations,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    print(f"Initializing model ({model_config.model_type})...")
    if model_config.model_type == "nodewise":
        model = GCNN_OPF_NodeWise(config=model_config).to(device)
    else:
        model = GCNN_OPF_01(config=model_config).to(device)

    print(
        f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Load generator bus mapping for physics loss
    gen_bus_map = train_dataset.gen_bus_map.numpy()
    N_BUS = model_config.n_bus
    N_GEN = model_config.n_gen

    A_g2b = np.zeros((N_BUS, N_GEN), dtype=np.float32)
    for g in range(N_GEN):
        bus_idx = gen_bus_map[g]
        A_g2b[bus_idx, g] = 1.0
    A_g2b = torch.from_numpy(A_g2b).to(device)

    # Training Logic
    if args.two_stage:
        # --- PHASE 1: Supervised Pre-training ---
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        hist1 = run_training_phase(
            "Phase1",
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            A_g2b,
            epochs=args.phase1_epochs,
            kappa=0.0,  # Pure supervised
            use_physics=False,  # Disable physics calculation for speed/purity
            results_dir=results_dir,
            log_interval=args.log_interval,
            patience=args.patience,
            start_epoch=1,
        )

        # Load best model from Phase 1
        best_p1_path = results_dir / "best_model_phase1.pth"
        if best_p1_path.exists():
            model.load_state_dict(torch.load(best_p1_path))
            print("\nLoaded best model from Phase 1 for Phase 2.")

        # --- PHASE 2: Physics Fine-tuning ---
        # Re-initialize optimizer with lower LR
        optimizer = optim.Adam(
            model.parameters(), lr=args.phase2_lr, weight_decay=args.weight_decay
        )

        hist2 = run_training_phase(
            "Phase2",
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            A_g2b,
            epochs=args.phase2_epochs,
            kappa=args.phase2_kappa,
            use_physics=True,
            results_dir=results_dir,
            log_interval=args.log_interval,
            patience=args.patience,
            start_epoch=args.phase1_epochs + 1,
        )

        # Combine histories
        train_history = hist1["train"] + hist2["train"]
        val_history = hist1["val"] + hist2["val"]

        # Save final refined model
        torch.save(model.state_dict(), results_dir / "final_model_refined.pth")
        # Also save as best_model.pth for evaluation script compatibility
        best_p2_path = results_dir / "best_model_phase2.pth"
        if best_p2_path.exists():
            import shutil

            shutil.copy(best_p2_path, results_dir / "best_model.pth")
            print("Copied best_model_phase2.pth to best_model.pth for evaluation.")

    else:
        # --- Standard Single-Stage Training (Original Logic) ---
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        hist = run_training_phase(
            "Standard",
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            A_g2b,
            epochs=args.epochs,
            kappa=args.kappa,
            use_physics=args.use_physics_loss,
            results_dir=results_dir,
            log_interval=args.log_interval,
            patience=args.patience,
        )
        train_history = hist["train"]
        val_history = hist["val"]

        # Ensure best_model.pth exists (created by run_training_phase as best_model_standard.pth)
        best_std_path = results_dir / "best_model_standard.pth"
        if best_std_path.exists():
            import shutil

            shutil.copy(best_std_path, results_dir / "best_model.pth")

    # --- Post-Training Saving & Plotting (Shared) ---

    # Save training history as CSV
    import csv

    with open(results_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_sup",
                "train_phys",
                "val_loss",
                "val_sup",
                "val_phys",
            ]
        )
        for i, (train_m, val_m) in enumerate(zip(train_history, val_history), 1):
            writer.writerow(
                [
                    i,
                    train_m["loss"],
                    train_m["sup_loss"],
                    train_m["phys_loss"],
                    val_m["loss"],
                    val_m["sup_loss"],
                    val_m["phys_loss"],
                ]
            )

    # Save training curves as NPZ
    np.savez(
        results_dir / "training_history.npz",
        train_loss=[m["loss"] for m in train_history],
        train_sup_loss=[m["sup_loss"] for m in train_history],
        train_phys_loss=[m["phys_loss"] for m in train_history],
        val_loss=[m["loss"] for m in val_history],
        val_sup_loss=[m["sup_loss"] for m in val_history],
        val_phys_loss=[m["phys_loss"] for m in val_history],
    )

    # Plot training curves
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(train_history) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Total loss
        axes[0].plot(
            epochs_range, [m["loss"] for m in train_history], "b-", label="Train"
        )
        axes[0].plot(epochs_range, [m["loss"] for m in val_history], "r-", label="Val")
        axes[0].set_title("Total Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Supervised loss
        axes[1].plot(
            epochs_range, [m["sup_loss"] for m in train_history], "b-", label="Train"
        )
        axes[1].plot(
            epochs_range, [m["sup_loss"] for m in val_history], "r-", label="Val"
        )
        axes[1].set_title("Supervised Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Physics loss
        axes[2].plot(
            epochs_range, [m["phys_loss"] for m in train_history], "b-", label="Train"
        )
        axes[2].plot(
            epochs_range, [m["phys_loss"] for m in val_history], "r-", label="Val"
        )
        axes[2].set_title("Physics Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(results_dir / "training_curves.png", dpi=150)
        print(f"  Training curves saved to: {results_dir / 'training_curves.png'}")
        plt.close()
    except Exception as e:
        print(f"  Warning: Could not plot training curves: {e}")

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
