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
"""

import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deep_opf.data import OPFDataModule
from deep_opf.models import GCNN, AdmittanceDNN
from deep_opf.task import OPFTask


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

    datamodule = OPFDataModule(
        data_dir=str(data_dir),
        train_file=cfg.data.train_file,
        val_file=cfg.data.val_file,
        test_file=cfg.data.get("test_file"),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.get("num_workers", 4),
        feature_type=feature_type,
        pin_memory=cfg.data.get("pin_memory", True),
    )

    print(
        f"Instantiated OPFDataModule: data_dir={data_dir}, "
        f"feature_type={feature_type}, batch_size={cfg.train.batch_size}"
    )

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
    task = OPFTask(
        model=model,
        lr=cfg.model.task.lr,
        kappa=cfg.model.task.kappa,
        weight_decay=cfg.model.task.weight_decay,
        gen_bus_indices=gen_bus_indices,
        n_bus=n_bus,
    )

    print(
        f"Instantiated OPFTask: lr={cfg.model.task.lr}, "
        f"kappa={cfg.model.task.kappa}, weight_decay={cfg.model.task.weight_decay}"
    )

    return task


def setup_callbacks(cfg: DictConfig) -> list:
    """
    Set up training callbacks.

    Args:
        cfg: Full Hydra config

    Returns:
        List of callbacks
    """
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        filename="{epoch}-{val_loss:.4f}",
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

    datamodule = instantiate_datamodule(cfg)
    model = instantiate_model(cfg, n_bus, n_gen)
    task = instantiate_task(model, cfg, gen_bus_indices, n_bus)
    callbacks = setup_callbacks(cfg)

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
    )

    print(
        f"Trainer: max_epochs={cfg.train.max_epochs}, "
        f"accelerator={cfg.train.accelerator}, devices={cfg.train.devices}"
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.fit(task, datamodule=datamodule)

    # Print best checkpoint path
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    if trainer.checkpoint_callback:
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best score: {trainer.checkpoint_callback.best_model_score}")


if __name__ == "__main__":
    main()
