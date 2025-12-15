"""Custom PyTorch Lightning callbacks for experiment tracking.

This module provides lightweight callbacks for monitoring training progress
without heavy dependencies like W&B or MLflow.
"""

# pyright: reportAttributeAccessIssue=false

import time
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class LiteProgressBar(pl.Callback):
    """Lightweight progress callback for training monitoring.

    Provides:
    - Formatted epoch progress to stdout
    - Current status file for external monitoring (e.g., `cat current_status.txt`)

    Usage:
        from deep_opf.utils.callbacks import LiteProgressBar

        trainer = pl.Trainer(
            callbacks=[LiteProgressBar(status_file="current_status.txt")],
            ...
        )
    """

    def __init__(self, status_file: Optional[str] = "current_status.txt"):
        """
        Initialize LiteProgressBar.

        Args:
            status_file: Path to status file (relative to working directory).
                        Set to None to disable file writing.
        """
        super().__init__()
        self.status_file: Optional[Path] = Path(status_file) if status_file else None
        self._epoch_start_time: float = 0.0
        self._train_start_time: float = 0.0

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Record training start time."""
        self._train_start_time = time.time()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Record epoch start time."""
        self._epoch_start_time = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log epoch metrics and update status file."""
        # Calculate epoch time
        epoch_time = time.time() - self._epoch_start_time
        total_time = time.time() - self._train_start_time

        # Get metrics from callback_metrics
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train/loss", metrics.get("train_loss", float("nan")))
        val_loss = metrics.get("val/loss", metrics.get("val_loss", float("nan")))

        # Convert tensors to floats if necessary
        if hasattr(train_loss, "item"):
            train_loss = train_loss.item()
        if hasattr(val_loss, "item"):
            val_loss = val_loss.item()

        # Format time strings
        epoch_time_str = self._format_time(epoch_time)
        total_time_str = self._format_time(total_time)

        # Build status string
        epoch = trainer.current_epoch + 1  # 1-indexed for display
        max_epochs = trainer.max_epochs

        status = (
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Time: {epoch_time_str} (Total: {total_time_str})"
        )

        # Print to stdout
        print(status)

        # Write to status file (overwrite mode)
        if self.status_file:
            self._write_status(status, epoch, max_epochs, total_time_str)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log training completion."""
        total_time = time.time() - self._train_start_time
        total_time_str = self._format_time(total_time)

        # Get best metrics if available
        best_score: Optional[float] = None
        best_path: Optional[str] = None
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback and isinstance(ckpt_callback, ModelCheckpoint):
            best_score = (
                ckpt_callback.best_model_score.item()
                if ckpt_callback.best_model_score is not None
                else None
            )
            best_path = ckpt_callback.best_model_path

        completion_msg = f"\n{'='*60}\nTraining Complete! Total Time: {total_time_str}"
        if best_score is not None:
            completion_msg += f"\nBest Val Loss: {best_score:.4f}"
        if best_path:
            completion_msg += f"\nBest Model: {best_path}"
        completion_msg += f"\n{'='*60}"

        print(completion_msg)

        # Update status file with completion info
        if self.status_file:
            with open(self.status_file, "w", encoding="utf-8") as f:
                f.write(f"COMPLETED | Total Time: {total_time_str}")
                if best_score is not None:
                    f.write(f" | Best Loss: {best_score:.4f}")
                f.write("\n")

    def _write_status(
        self, status: str, epoch: int, max_epochs: int, total_time: str
    ) -> None:
        """Write current status to file for external monitoring."""
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                f.write(f"{status}\n")
                f.write(
                    f"Progress: {epoch}/{max_epochs} ({100*epoch/max_epochs:.1f}%)\n"
                )
                f.write(f"Elapsed: {total_time}\n")
        except OSError as e:
            # Don't crash training if status file can't be written
            print(f"Warning: Could not write status file: {e}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
