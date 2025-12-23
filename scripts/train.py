"""Main training script for deep_opf models using Hydra configuration.

Usage:
    # Default: DNN on case6
    python scripts/train.py

    # GCNN on case39
    python scripts/train.py model=gcnn data=case39

    # Override hyperparameters
    python scripts/train.py model.task.lr=0.0005 train.max_epochs=100

    # Multi-run with different seeds
    python scripts/train.py --multirun seed=42,123,456

    # Two-stage training (Phase 2 with warm start from Phase 1):
    # Phase 1: Train with supervised loss only (kappa=0)
    python scripts/train.py model.task.kappa=0.0 train.max_epochs=50

    # Phase 2: Load Phase 1 weights, train with physics loss (kappa=1.0)
    python scripts/train.py model.task.kappa=1.0 train.max_epochs=25 \\
        train.warm_start_ckpt=outputs/2025-01-01/12-00-00/lightning_logs/version_0/checkpoints/best.ckpt

    # SSH mode (lightweight progress bar, no TQDM, logs to CSV)
    python scripts/train.py train.mode=ssh
"""

import sys
import time
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

# Enable Tensor Cores on supported GPUs (RTX 20xx, 30xx, 40xx, etc.)
# 'medium' uses TF32 for ~3x speedup with minimal precision loss
torch.set_float32_matmul_precision("medium")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deep_opf.data import OPFDataModule
from deep_opf.models import GCNN, AdmittanceDNN
from deep_opf.task import OPFTask
from deep_opf.utils.callbacks import LiteProgressBar
from deep_opf.utils.logger import log_experiment_to_csv
from deep_opf.utils.plot_loss import plot_loss_curves_from_log_dir


def instantiate_model(cfg: DictConfig, n_bus: int, n_gen: int) -> pl.LightningModule:
    """
    Instantiate the appropriate model based on config.

    Args:
        cfg: Full Hydra config
        n_bus: Number of buses (from data config)
        n_gen: Number of generators (from data config)

    Returns:
        Instantiated model (AdmittanceDNN or GCNN)
    """
    model_name = cfg.model.name

    if model_name == "dnn":
        # Calculate input dimension for DNN
        # Flat feature: [Pd, Qd, G_flat, B_flat] = n_bus + n_bus + n_bus^2 + n_bus^2
        input_dim = 2 * n_bus + 2 * (n_bus * n_bus)

        model = AdmittanceDNN(
            input_dim=input_dim,
            hidden_dim=cfg.model.architecture.hidden_dim,
            num_layers=cfg.model.architecture.num_layers,
            n_gen=n_gen,
            n_bus=n_bus,
            dropout=cfg.model.architecture.dropout,
        )
        print(
            f"Instantiated AdmittanceDNN: input_dim={input_dim}, "
            f"hidden_dim={cfg.model.architecture.hidden_dim}, "
            f"num_layers={cfg.model.architecture.num_layers}"
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
        print(
            f"Instantiated GCNN: n_bus={n_bus}, n_gen={n_gen}, "
            f"in_channels={cfg.model.architecture.in_channels}, "
            f"hidden_channels={cfg.model.architecture.hidden_channels}"
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}. Expected 'dnn' or 'gcnn'.")

    return model


def instantiate_datamodule(cfg: DictConfig) -> OPFDataModule:
    """
    Instantiate OPFDataModule from config.

    Args:
        cfg: Full Hydra config

    Returns:
        Configured OPFDataModule
    """
    # Resolve data paths relative to original working directory
    original_cwd = Path(hydra.utils.get_original_cwd())
    data_dir = original_cwd / cfg.data.data_dir

    # Feature type from resolved config (interpolation: ${model.feature_type})
    feature_type = cfg.data.feature_type

    # Build feature_params for graph feature slicing
    feature_params: dict | None = None
    if feature_type == "graph" and cfg.model.name == "gcnn":
        # Use in_channels as feature_iterations (sync model input with data)
        in_channels = cfg.model.architecture.in_channels
        feature_params = {"feature_iterations": in_channels}

    datamodule = OPFDataModule(
        data_dir=str(data_dir),
        train_file=cfg.data.train_file,
        val_file=cfg.data.val_file,
        test_file=cfg.data.get("test_file"),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.get("num_workers", 4),
        feature_type=feature_type,
        feature_params=feature_params,
        pin_memory=cfg.data.get("pin_memory", True),
        persistent_workers=cfg.train.get("persistent_workers", False),
    )

    print(
        f"Instantiated OPFDataModule: data_dir={data_dir}, "
        f"feature_type={feature_type}, batch_size={cfg.train.batch_size}"
    )
    if feature_params:
        print(f"  feature_params: {feature_params}")

    return datamodule


def instantiate_task(
    model: pl.LightningModule,
    cfg: DictConfig,
    gen_bus_indices: list[int],
    n_bus: int,
) -> OPFTask:
    """
    Instantiate OPFTask wrapping the model.

    Args:
        model: The neural network model
        cfg: Full Hydra config
        gen_bus_indices: List of bus indices for each generator
        n_bus: Number of buses

    Returns:
        Configured OPFTask
    """
    # Get scheduler params with defaults for backward compatibility
    lr_scheduler = getattr(cfg.model.task, "lr_scheduler", None)
    lr_scheduler_patience = getattr(cfg.model.task, "lr_scheduler_patience", 50)
    lr_scheduler_factor = getattr(cfg.model.task, "lr_scheduler_factor", 0.5)
    min_lr = getattr(cfg.model.task, "min_lr", 1e-6)

    task = OPFTask(
        model=model,
        lr=cfg.model.task.lr,
        kappa=cfg.model.task.kappa,
        weight_decay=cfg.model.task.weight_decay,
        gen_bus_indices=gen_bus_indices,
        n_bus=n_bus,
        lr_scheduler=lr_scheduler,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
        min_lr=min_lr,
    )

    scheduler_info = f", lr_scheduler={lr_scheduler}" if lr_scheduler else ""
    print(
        f"Instantiated OPFTask: lr={cfg.model.task.lr}, "
        f"kappa={cfg.model.task.kappa}, weight_decay={cfg.model.task.weight_decay}"
        f"{scheduler_info}"
    )

    return task


def setup_callbacks(cfg: DictConfig, ssh_mode: bool = False) -> list:
    """
    Set up training callbacks.

    Args:
        cfg: Full Hydra config
        ssh_mode: If True, add LiteProgressBar for SSH-friendly output

    Returns:
        List of callbacks
    """
    callbacks = []

    # Model checkpoint
    # Note: auto_insert_metric_name=False prevents 'epoch=' and 'val_loss=' prefixes
    # This avoids Hydra parsing issues with '=' in filenames
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        auto_insert_metric_name=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    print(
        f"Added ModelCheckpoint: monitor={cfg.checkpoint.monitor}, "
        f"save_top_k={cfg.checkpoint.save_top_k}"
    )

    # Early stopping (use train.patience if available, else early_stopping.patience)
    if cfg.early_stopping.enabled:
        patience = cfg.train.get("patience", cfg.early_stopping.patience)
        early_stop_callback = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=patience,
            mode=cfg.early_stopping.mode,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        print(
            f"Added EarlyStopping: monitor={cfg.early_stopping.monitor}, "
            f"patience={patience}"
        )

    # Progress bar configuration
    if ssh_mode:
        # SSH mode: Add lightweight progress bar (epoch-level only)
        callbacks.append(LiteProgressBar())
        print("Added LiteProgressBar (SSH mode)")
    else:
        # Standard mode: TQDM with reduced refresh rate (update every N batches)
        # refresh_rate=0 means update only at epoch end
        refresh_rate = cfg.train.get("progress_refresh_rate", 0)
        callbacks.append(TQDMProgressBar(refresh_rate=refresh_rate))
        if refresh_rate == 0:
            print("Added TQDMProgressBar: epoch-level updates only (refresh_rate=0)")
        else:
            print(f"Added TQDMProgressBar: refresh_rate={refresh_rate}")

    return callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Hydra configuration object
    """
    # Print config
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Extract data parameters
    n_bus = cfg.data.n_bus
    n_gen = cfg.data.n_gen
    gen_bus_indices = list(cfg.data.gen_bus_indices)

    # Instantiate components
    print("\n" + "=" * 60)
    print("Instantiating components...")
    print("=" * 60)

    # Check training mode
    train_mode = cfg.train.get("mode", "standard")
    ssh_mode = train_mode == "ssh"
    if ssh_mode:
        print("SSH mode enabled: Using LiteProgressBar, disabling TQDM")

    datamodule = instantiate_datamodule(cfg)
    model = instantiate_model(cfg, n_bus, n_gen)
    task = instantiate_task(model, cfg, gen_bus_indices, n_bus)
    callbacks = setup_callbacks(cfg, ssh_mode=ssh_mode)

    # Warm start: Load weights from checkpoint if provided
    warm_start_ckpt = cfg.train.get("warm_start_ckpt")
    if warm_start_ckpt:
        warm_start_path = Path(warm_start_ckpt)
        if not warm_start_path.is_absolute():
            # Resolve relative paths from original working directory
            original_cwd = Path(hydra.utils.get_original_cwd())
            warm_start_path = original_cwd / warm_start_path

        if warm_start_path.exists():
            print(f"\nWarm starting from: {warm_start_path}")
            checkpoint = torch.load(warm_start_path, map_location="cpu")

            # Handle PyTorch Lightning checkpoint format (contains 'state_dict' key)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Load weights only (strict=False allows partial loading)
            missing, unexpected = task.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
            print("  Weights loaded successfully. Optimizer will be fresh.")
        else:
            raise FileNotFoundError(
                f"Warm start checkpoint not found: {warm_start_path}"
            )

    # Create trainer
    print("\n" + "=" * 60)
    print("Creating Trainer...")
    print("=" * 60)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=callbacks,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=True,
        enable_progress_bar=True,  # Required when using TQDMProgressBar callback
    )

    print(
        f"Trainer: max_epochs={cfg.train.max_epochs}, "
        f"accelerator={cfg.train.accelerator}, devices={cfg.train.devices}"
    )

    # Train with timing
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    start_time = time.time()
    trainer.fit(task, datamodule=datamodule)
    duration = time.time() - start_time

    # Get best score
    best_score = None
    if trainer.checkpoint_callback:
        best_score = trainer.checkpoint_callback.best_model_score
        if hasattr(best_score, "item"):
            best_score = best_score.item()

    # Print best checkpoint path
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
    if trainer.checkpoint_callback:
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best score: {best_score}")

    # Generate loss curves plot
    if trainer.log_dir:
        plot_loss_curves_from_log_dir(trainer.log_dir)

    # Log experiment to CSV
    if best_score is not None:
        original_cwd = Path(hydra.utils.get_original_cwd())
        csv_path = original_cwd / "experiments_log.csv"
        # Get Hydra output directory (cwd since Hydra changes directory)
        # Alternative: HydraConfig.get().runtime.output_dir
        hydra_output_dir = Path.cwd()
        log_experiment_to_csv(
            cfg=cfg,
            model=model,
            best_loss=best_score,
            duration=duration,
            csv_path=csv_path,
            log_dir=hydra_output_dir,
        )


if __name__ == "__main__":
    main()
