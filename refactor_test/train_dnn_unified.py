"""Modified DNN training script using unified data loader.

This is a simplified version of the original train_03.py that uses
the new unified data loader (OPFDataModule) instead of the legacy
dataset implementation.
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

# Add src to path for unified data loader
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import unified data loader
from deep_opf.data.datamodule import OPFDataModule

# Import legacy model (for testing compatibility)
sys.path.insert(0, str(Path(__file__).parent.parent / "legacy" / "dnn_opf_03"))
from model_03 import AdmittanceDNN
from loss_model_03 import correlative_loss_pg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DNN OPF Model with Unified Data Loader"
    )

    # Data paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="legacy/gcnn_opf_01/data_matlab_npz",
        help="Directory containing dataset files",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="refactor_test/results_dnn",
        help="Directory to save results",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (reduced for testing)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )

    # Model architecture
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="128,64,32",
        help="Comma-separated hidden layer dimensions",
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
        "--patience", type=int, default=5, help="Early stopping patience (epochs)"
    )

    # Logging
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log every N batches"
    )

    return parser.parse_args()


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    gen_bus_map,
    kappa=0.1,
    use_physics=True,
    log_interval=10,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_physics = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        inputs = batch["input"].to(device)
        pg_label = batch["pg_label"].to(device)
        vg_label = batch["vg_label"].to(device)
        topo_id = batch["topo_id"].to(device)

        # Get operators for physics loss
        operators = batch["operators"]

        # Forward pass
        pg_pred, vg_pred = model(inputs)

        # MSE loss
        mse_loss = nn.functional.mse_loss(pg_pred, pg_label) + nn.functional.mse_loss(
            vg_pred, vg_label
        )

        # Physics loss (if enabled)
        physics_loss = 0.0
        if use_physics:
            # Convert batch operators to device
            g_ndiag = torch.stack([op["g_ndiag"].to(device) for op in operators])
            b_ndiag = torch.stack([op["b_ndiag"].to(device) for op in operators])
            g_diag = torch.stack([op["g_diag"].to(device) for op in operators])
            b_diag = torch.stack([op["b_diag"].to(device) for op in operators])

            # Get demands
            pd = batch["pd"].to(device)
            qd = batch["qd"].to(device)

            physics_loss = correlative_loss_pg(
                pg_pred,
                vg_pred,
                pd,
                qd,
                g_ndiag,
                b_ndiag,
                g_diag,
                b_diag,
                gen_bus_map.to(device),
            )

        # Total loss
        loss = mse_loss + kappa * physics_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_physics += physics_loss.item() if use_physics else 0.0
        n_batches += 1

        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "mse": f"{mse_loss.item():.4f}",
                    "phys": f"{physics_loss.item():.4f}" if use_physics else "0.0",
                }
            )

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "physics": total_physics / n_batches if use_physics else 0.0,
    }


def validate(model, val_loader, device, gen_bus_map, kappa=0.1, use_physics=True):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_physics = 0.0
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for batch in pbar:
            # Move data to device
            inputs = batch["input"].to(device)
            pg_label = batch["pg_label"].to(device)
            vg_label = batch["vg_label"].to(device)
            topo_id = batch["topo_id"].to(device)

            # Get operators for physics loss
            operators = batch["operators"]

            # Forward pass
            pg_pred, vg_pred = model(inputs)

            # MSE loss
            mse_loss = nn.functional.mse_loss(
                pg_pred, pg_label
            ) + nn.functional.mse_loss(vg_pred, vg_label)

            # Physics loss (if enabled)
            physics_loss = 0.0
            if use_physics:
                # Convert batch operators to device
                g_ndiag = torch.stack([op["g_ndiag"].to(device) for op in operators])
                b_ndiag = torch.stack([op["b_ndiag"].to(device) for op in operators])
                g_diag = torch.stack([op["g_diag"].to(device) for op in operators])
                b_diag = torch.stack([op["b_diag"].to(device) for op in operators])

                # Get demands
                pd = batch["pd"].to(device)
                qd = batch["qd"].to(device)

                physics_loss = correlative_loss_pg(
                    pg_pred,
                    vg_pred,
                    pd,
                    qd,
                    g_ndiag,
                    b_ndiag,
                    g_diag,
                    b_diag,
                    gen_bus_map.to(device),
                )

            # Total loss
            loss = mse_loss + kappa * physics_loss

            # Update metrics
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_physics += physics_loss.item() if use_physics else 0.0
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "physics": total_physics / n_batches if use_physics else 0.0,
    }


def main():
    args = parse_args()

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create unified data module
    print("\nSetting up unified data loader...")
    dm = OPFDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        feature_type="flat",  # DNN uses flat features
        normalize=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
    )

    # Setup data module
    dm.setup(stage="fit")

    # Get data dimensions from data module
    n_bus = dm.n_bus
    n_gen = dm.n_gen
    input_dim = dm.input_dim

    print(f"Data dimensions:")
    print(f"  - Number of buses: {n_bus}")
    print(f"  - Number of generators: {n_gen}")
    print(f"  - Input dimension: {input_dim}")

    # Get generator bus map
    gen_bus_map = dm.get_gen_bus_map()
    print(f"  - Generator bus map shape: {gen_bus_map.shape}")

    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]

    # Create model
    print("\nCreating DNN model...")
    model = AdmittanceDNN(
        input_dim=input_dim,
        output_dim=n_gen * 2,  # PG and VG for each generator
        hidden_dims=hidden_dims,
        dropout=0.1,
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Architecture: Input({input_dim}) -> {hidden_dims} -> Output({n_gen * 2})")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Get dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(f"\nTraining setup:")
    print(f"  - Training samples: {len(dm.train_dataset)}")
    print(f"  - Validation samples: {len(dm.val_dataset)}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Physics loss weight (kappa): {args.kappa}")
    print(f"  - Use physics loss: {args.use_physics_loss}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gen_bus_map,
            kappa=args.kappa,
            use_physics=args.use_physics_loss,
            log_interval=args.log_interval,
        )

        # Validate
        val_metrics = validate(
            model,
            val_loader,
            device,
            gen_bus_map,
            kappa=args.kappa,
            use_physics=args.use_physics_loss,
        )

        epoch_time = time.time() - epoch_start

        # Print metrics
        print(
            f"Train - Loss: {train_metrics['loss']:.6f}, "
            f"MSE: {train_metrics['mse']:.6f}, "
            f"Physics: {train_metrics['physics']:.6f}"
        )
        print(
            f"Val   - Loss: {val_metrics['loss']:.6f}, "
            f"MSE: {val_metrics['mse']:.6f}, "
            f"Physics: {val_metrics['physics']:.6f}"
        )
        print(f"Time: {epoch_time:.2f}s")

        # Save history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Check for improvement
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best model
            model_path = results_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "best_val_loss": best_val_loss,
                },
                model_path,
            )
            print(f"✓ Saved best model to {model_path}")
        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/{args.patience})")

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    # Save final model
    final_model_path = results_dir / "final_model.pth"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        final_model_path,
    )
    print(f"\nSaved final model to {final_model_path}")

    # Save training history
    history_path = results_dir / "training_history.npz"
    np.savez(
        history_path,
        train_loss=[m["loss"] for m in history["train"]],
        train_mse=[m["mse"] for m in history["train"]],
        train_physics=[m["physics"] for m in history["train"]],
        val_loss=[m["loss"] for m in history["val"]],
        val_mse=[m["mse"] for m in history["val"]],
        val_physics=[m["physics"] for m in history["val"]],
    )
    print(f"Saved training history to {history_path}")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
