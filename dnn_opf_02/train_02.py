"""Training script for Hasan DNN (M8) OPF Model."""

import sys
from pathlib import Path
import argparse
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add repo root to path to allow imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import from local folder
from model_02 import HasanDNN
from dataset_02 import OPFDataset02

# Import from gcnn_opf_01 (reusing loss and config if needed)
# We need correlative_loss_pg
sys.path.insert(0, str(REPO_ROOT / "gcnn_opf_01"))
from loss_model_01 import correlative_loss_pg

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hasan DNN OPF Model")

    # Data paths
    parser.add_argument("--data_dir", type=str, default="gcnn_opf_01/data", help="Directory containing samples")
    parser.add_argument("--means_dir", type=str, default="dnn_opf_02/data", help="Directory containing topology means")
    parser.add_argument("--results_dir", type=str, default="dnn_opf_02/results", help="Directory to save results")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    
    # Two-stage training
    parser.add_argument("--epochs_stage1", type=int, default=25, help="Epochs for Stage 1 (Supervised)")
    parser.add_argument("--epochs_stage2", type=int, default=25, help="Epochs for Stage 2 (Physics)")
    parser.add_argument("--kappa_stage2", type=float, default=1.0, help="Physics loss weight for Stage 2")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)

    return parser.parse_args()

def train_epoch(model, train_loader, optimizer, device, A_g2b, kappa, log_interval):
    model.train()
    total_loss = 0.0
    total_sup_loss = 0.0
    total_phys_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        x = batch["x"].to(device)
        pd = batch["pd"].to(device)
        gen_labels = batch["gen_label"].to(device)
        
        # Operators for physics loss
        ops = batch["operators"]
        g_ndiag = ops["g_ndiag"][0].to(device)
        b_ndiag = ops["b_ndiag"][0].to(device)
        g_diag = ops["g_diag"][0].to(device)
        b_diag = ops["b_diag"][0].to(device)
        
        G = g_ndiag + torch.diag(g_diag)
        B = b_ndiag + torch.diag(b_diag)
        
        optimizer.zero_grad()
        gen_out, v_out = model(x)
        
        # Supervised loss
        sup_loss = nn.functional.mse_loss(gen_out, gen_labels)
        
        # Physics loss
        if kappa > 0:
            phys_loss_total = 0.0
            for i in range(x.size(0)):
                _, _, loss_p = correlative_loss_pg(
                    gen_out[i], v_out[i], 
                    gen_labels[i,:,0], gen_labels[i,:,1],
                    pd[i], G, B, A_g2b, kappa=0.0
                )
                phys_loss_total += loss_p
            phys_loss = phys_loss_total / x.size(0)
            loss = sup_loss + kappa * phys_loss
        else:
            phys_loss = torch.tensor(0.0, device=device)
            loss = sup_loss
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_phys_loss += phys_loss.item()
        n_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return {
        "loss": total_loss / n_batches,
        "sup_loss": total_sup_loss / n_batches,
        "phys_loss": total_phys_loss / n_batches
    }

def validate(model, val_loader, device, A_g2b, kappa):
    model.eval()
    total_loss = 0.0
    total_sup_loss = 0.0
    total_phys_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            x = batch["x"].to(device)
            pd = batch["pd"].to(device)
            gen_labels = batch["gen_label"].to(device)
            
            ops = batch["operators"]
            g_ndiag = ops["g_ndiag"][0].to(device)
            b_ndiag = ops["b_ndiag"][0].to(device)
            g_diag = ops["g_diag"][0].to(device)
            b_diag = ops["b_diag"][0].to(device)
            
            G = g_ndiag + torch.diag(g_diag)
            B = b_ndiag + torch.diag(b_diag)
            
            gen_out, v_out = model(x)
            
            sup_loss = nn.functional.mse_loss(gen_out, gen_labels)
            
            if kappa > 0:
                phys_loss_total = 0.0
                for i in range(x.size(0)):
                    _, _, loss_p = correlative_loss_pg(
                        gen_out[i], v_out[i], 
                        gen_labels[i,:,0], gen_labels[i,:,1],
                        pd[i], G, B, A_g2b, kappa=0.0
                    )
                    phys_loss_total += loss_p
                phys_loss = phys_loss_total / x.size(0)
                loss = sup_loss + kappa * phys_loss
            else:
                phys_loss = torch.tensor(0.0, device=device)
                loss = sup_loss
                
            total_loss += loss.item()
            total_sup_loss += sup_loss.item()
            total_phys_loss += phys_loss.item()
            n_batches += 1
            
    return {
        "loss": total_loss / n_batches,
        "sup_loss": total_sup_loss / n_batches,
        "phys_loss": total_phys_loss / n_batches
    }

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Datasets
    print("Loading datasets...")
    data_dir = Path(args.data_dir)
    means_dir = Path(args.means_dir)
    
    train_dataset = OPFDataset02(
        data_path=data_dir / "samples_train.npz",
        topo_means_path=means_dir / "topology_voltage_means.npz",
        topo_operators_path=data_dir / "topology_operators.npz",
        norm_stats_path=data_dir / "norm_stats.npz",
        normalize=True,
        split="train"
    )
    
    val_dataset = OPFDataset02(
        data_path=data_dir / "samples_test.npz",
        topo_means_path=means_dir / "topology_voltage_means.npz",
        topo_operators_path=data_dir / "topology_operators.npz",
        norm_stats_path=data_dir / "norm_stats.npz",
        normalize=True,
        split="test"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    # Assume case6ww: N_BUS=6, N_GEN=3
    # We can get this from dataset
    sample = train_dataset[0]
    n_bus = len(sample["pd"])
    n_gen = len(sample["pg_label"])
    
    model = HasanDNN(n_bus=n_bus, n_gen=n_gen).to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # A_g2b for physics loss
    gen_bus_map = train_dataset.gen_bus_map.numpy()
    A_g2b = np.zeros((n_bus, n_gen), dtype=np.float32)
    for g in range(n_gen):
        bus_idx = gen_bus_map[g]
        A_g2b[bus_idx, g] = 1.0
    A_g2b = torch.from_numpy(A_g2b).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    history = []
    
    # Training Loop
    total_epochs = args.epochs_stage1 + args.epochs_stage2
    
    # Overall progress bar
    epoch_pbar = tqdm(range(1, total_epochs + 1), desc="Total Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        # Determine stage and kappa
        if epoch <= args.epochs_stage1:
            stage = 1
            kappa = 0.0
        else:
            stage = 2
            kappa = args.kappa_stage2
            
        # Update description
        epoch_pbar.set_description(f"Epoch {epoch}/{total_epochs} [Stage {stage}]")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device, A_g2b, kappa, args.log_interval)
        val_metrics = validate(model, val_loader, device, A_g2b, kappa)
        
        # Print metrics (tqdm compatible)
        tqdm.write(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f} (Sup={train_metrics['sup_loss']:.4f}), Val Loss={val_metrics['loss']:.4f}")
        
        history.append({
            "epoch": epoch,
            "stage": stage,
            "train": train_metrics,
            "val": val_metrics
        })
        
        # Save best model (based on val loss)
        # Note: Val loss definition changes between stages (includes kappa), so be careful comparing across stages
        # We'll save best model for each stage separately
        if epoch == 1 or val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), results_dir / f"best_model_stage{stage}.pth")
            
        # Reset best_val_loss at start of stage 2
        if epoch == args.epochs_stage1:
            best_val_loss = float("inf")
            
    # Save final model
    torch.save(model.state_dict(), results_dir / "final_model.pth")
    
    # Save history
    np.savez(results_dir / "training_history.npz", history=history)
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
