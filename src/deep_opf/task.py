"""PyTorch Lightning task module for OPF model training.

Provides a unified training interface for both DNN and GCNN models
with physics-informed loss computation.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import build_gen_bus_matrix, physics_loss


class OPFTask(pl.LightningModule):
    """
    PyTorch Lightning module for OPF model training.

    Supports both DNN (flat input) and GCNN (graph input) models with
    automatic input format detection. Combines supervised loss with
    physics-informed loss for training.

    Args:
        model: Neural network model (AdmittanceDNN or GCNN)
        lr: Learning rate for Adam optimizer
        kappa: Weight for physics loss term (loss = sup + kappa * phys)
        weight_decay: L2 regularization weight for optimizer
        gen_bus_indices: List of bus indices for each generator (required for physics loss)
        n_bus: Number of buses (required for physics loss)

    Example:
        >>> from deep_opf.models import AdmittanceDNN, GCNN
        >>> from deep_opf.task import OPFTask
        >>>
        >>> # For DNN
        >>> model = AdmittanceDNN(input_dim=1000, hidden_dim=256, num_layers=3,
        ...                       n_gen=10, n_bus=39)
        >>> task = OPFTask(model, lr=1e-3, kappa=0.1, weight_decay=1e-4,
        ...                gen_bus_indices=[0, 1, 2, ...], n_bus=39)
        >>>
        >>> # For GCNN
        >>> model = GCNN(n_bus=39, n_gen=10)
        >>> task = OPFTask(model, lr=1e-3, kappa=0.1, weight_decay=1e-4,
        ...                gen_bus_indices=[0, 1, 2, ...], n_bus=39)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        kappa: float = 0.1,
        weight_decay: float = 1e-4,
        gen_bus_indices: list[int] | None = None,
        n_bus: int | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_patience: int = 50,
        lr_scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.kappa = kappa
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.min_lr = min_lr

        # Store for physics loss computation
        self.gen_bus_indices = gen_bus_indices
        self.n_bus = n_bus

        # Pre-build generator-bus matrix if indices provided
        self._gen_bus_matrix: torch.Tensor | None = None
        if gen_bus_indices is not None and n_bus is not None:
            self._gen_bus_matrix = build_gen_bus_matrix(n_bus, gen_bus_indices)

        # Save hyperparameters (excluding model to avoid duplication)
        self.save_hyperparameters(ignore=["model"])

    def _get_gen_bus_matrix(self, device: torch.device) -> torch.Tensor:
        """Get generator-bus matrix, moving to correct device if needed."""
        if self._gen_bus_matrix is None:
            raise ValueError(
                "gen_bus_indices and n_bus must be provided for physics loss. "
                "Pass them to OPFTask.__init__()."
            )
        if self._gen_bus_matrix.device != device:
            self._gen_bus_matrix = self._gen_bus_matrix.to(device)
        return self._gen_bus_matrix

    def _detect_input_type(self, batch: dict) -> str:
        """Detect whether batch is for DNN or GCNN model."""
        if "input" in batch:
            return "dnn"
        elif "e_0_k" in batch:
            return "gcnn"
        else:
            raise ValueError(
                "Cannot detect input type. Batch must contain either "
                "'input' (DNN) or 'e_0_k' (GCNN) key."
            )

    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward pass with automatic input format detection."""
        input_type = self._detect_input_type(batch)

        if input_type == "dnn":
            # DNN expects flat input tensor
            x = batch["input"]
            preds = self.model(x)
        else:
            # GCNN expects full batch dict
            preds = self.model(batch)

        return preds

    def _compute_loss(
        self, batch: dict, preds: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        """
        Compute combined supervised + physics loss.

        Args:
            batch: Input batch with labels and operators
            preds: Model predictions with 'pg', 'vg', 'v_bus'
            prefix: Logging prefix ('train' or 'val')

        Returns:
            Total loss (supervised + kappa * physics)
        """
        # Extract predictions
        pg_pred = preds["pg"]
        vg_pred = preds["vg"]
        v_bus_pred = preds["v_bus"]

        # Extract labels
        pg_label = batch["pg_label"]
        vg_label = batch["vg_label"]

        # Supervised loss: MSE for PG and VG
        loss_sup_pg = F.mse_loss(pg_pred, pg_label)
        loss_sup_vg = F.mse_loss(vg_pred, vg_label)
        loss_supervised = loss_sup_pg + loss_sup_vg

        # Physics loss using AC power flow equations
        # Extract G, B matrices from operators
        operators = batch["operators"]

        # Handle different operator formats
        # Full G/B matrices may be provided directly, or we reconstruct from diag/ndiag
        if "G" in operators:
            G = operators["G"]
            B = operators["B"]
        else:
            # Reconstruct from diagonal and off-diagonal components
            g_ndiag = operators["g_ndiag"]
            b_ndiag = operators["b_ndiag"]
            g_diag = operators["g_diag"]
            b_diag = operators["b_diag"]

            # Handle batched operators (use first sample)
            if g_ndiag.dim() == 3:
                g_ndiag = g_ndiag[0]
                b_ndiag = b_ndiag[0]
                g_diag = g_diag[0]
                b_diag = b_diag[0]

            # Reconstruct full matrices
            G = g_ndiag + torch.diag(g_diag)
            B = b_ndiag + torch.diag(b_diag)

        # Handle batched G, B (use first sample if batched)
        if G.dim() == 3:
            G = G[0]
            B = B[0]

        # Get demands
        pd = batch["pd"]
        qd = batch["qd"]

        # Get generator-bus mapping matrix
        gen_bus_matrix = self._get_gen_bus_matrix(pg_pred.device)

        # Compute physics loss
        phys_result = physics_loss(
            pg=pg_pred,
            v_bus=v_bus_pred,
            pd=pd,
            qd=qd,
            G=G,
            B=B,
            gen_bus_matrix=gen_bus_matrix,
            include_reactive=False,
        )
        loss_physics = phys_result["loss_p"]

        # Total loss
        loss_total = loss_supervised + self.kappa * loss_physics

        # Logging
        self.log(
            f"{prefix}/loss", loss_total, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(f"{prefix}/sup", loss_supervised, on_step=False, on_epoch=True)
        self.log(f"{prefix}/sup_pg", loss_sup_pg, on_step=False, on_epoch=True)
        self.log(f"{prefix}/sup_vg", loss_sup_vg, on_step=False, on_epoch=True)
        self.log(f"{prefix}/phys", loss_physics, on_step=False, on_epoch=True)

        return loss_total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        preds = self._forward(batch)
        loss = self._compute_loss(batch, preds, prefix="train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        preds = self._forward(batch)
        loss = self._compute_loss(batch, preds, prefix="val")
        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer with optional LR scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.lr_scheduler_type is None:
            return optimizer

        if self.lr_scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=self.min_lr,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.lr_scheduler_type == "cosine":
            # CosineAnnealingWarmRestarts: T_0=100 epochs per restart
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=100,
                T_mult=2,
                eta_min=self.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")
