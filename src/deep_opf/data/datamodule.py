"""PyTorch Lightning DataModule for unified OPF data loading.

Supports both 'flat' (DNN) and 'graph' (GCNN) feature types through
a single interface compatible with PyTorch Lightning training loops.
"""

from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import OPFDataset


class OPFDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for OPF datasets.

    Handles dataset creation and dataloader setup for both training
    and validation splits with configurable feature types.

    Example:
        >>> dm = OPFDataModule(
        ...     data_dir="path/to/data",
        ...     batch_size=64,
        ...     feature_type="flat",
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
        >>> val_loader = dm.val_dataloader()
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 64,
        feature_type: Literal["flat", "graph"] = "flat",
        num_workers: int = 0,
        normalize: bool = True,
        feature_params: dict | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing:
                - samples_train.npz
                - samples_eval.npz (or samples_test.npz)
                - topology_operators.npz
                - norm_stats.npz (optional)
            batch_size: Batch size for dataloaders
            feature_type: 'flat' for DNN input, 'graph' for GCNN input
            num_workers: Number of dataloader workers (0 for Windows compatibility)
            normalize: Whether to apply z-score normalization
            feature_params: Additional params passed to OPFDataset
                - For 'graph': {'feature_iterations': int}
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs (requires num_workers > 0)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.feature_type = feature_type
        self.num_workers = num_workers
        self.normalize = normalize
        self.feature_params = feature_params or {}
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # Will be set in setup()
        self.train_dataset: OPFDataset | None = None
        self.val_dataset: OPFDataset | None = None

        # Validate feature_type
        if feature_type not in ("flat", "graph"):
            raise ValueError(
                f"feature_type must be 'flat' or 'graph', got {feature_type}"
            )

    def setup(self, stage: str | None = None) -> None:
        """
        Set up datasets for training and validation.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict' (Lightning convention)
        """
        # Resolve paths
        topo_path = self.data_dir / "topology_operators.npz"
        norm_path = self.data_dir / "norm_stats.npz"

        # Use norm_path only if it exists
        norm_path_str: str | None = str(norm_path) if norm_path.exists() else None

        if stage == "fit" or stage is None:
            # Training dataset
            train_path = self.data_dir / "samples_train.npz"
            if not train_path.exists():
                raise FileNotFoundError(f"Training data not found: {train_path}")

            self.train_dataset = OPFDataset(
                data_path=train_path,
                topo_path=topo_path,
                norm_path=norm_path_str,
                normalize=self.normalize,
                split="train",
                feature_type=self.feature_type,
                feature_params=self.feature_params,
            )

            # Validation dataset (try samples_eval.npz first, then samples_test.npz)
            val_path = self.data_dir / "samples_eval.npz"
            if not val_path.exists():
                val_path = self.data_dir / "samples_test.npz"
            if not val_path.exists():
                raise FileNotFoundError(
                    f"Validation data not found: tried samples_eval.npz and samples_test.npz in {self.data_dir}"
                )

            self.val_dataset = OPFDataset(
                data_path=val_path,
                topo_path=topo_path,
                norm_path=norm_path_str,
                normalize=self.normalize,
                split="eval",
                feature_type=self.feature_type,
                feature_params=self.feature_params,
            )

        if stage == "validate":
            # Only validation dataset
            val_path = self.data_dir / "samples_eval.npz"
            if not val_path.exists():
                val_path = self.data_dir / "samples_test.npz"
            if not val_path.exists():
                raise FileNotFoundError(
                    f"Validation data not found: tried samples_eval.npz and samples_test.npz in {self.data_dir}"
                )

            self.val_dataset = OPFDataset(
                data_path=val_path,
                topo_path=topo_path,
                norm_path=norm_path_str,
                normalize=self.normalize,
                split="eval",
                feature_type=self.feature_type,
                feature_params=self.feature_params,
            )

        if stage == "test":
            # Test dataset (same as validation for now)
            test_path = self.data_dir / "samples_test.npz"
            if not test_path.exists():
                test_path = self.data_dir / "samples_eval.npz"
            if not test_path.exists():
                raise FileNotFoundError(
                    f"Test data not found: tried samples_test.npz and samples_eval.npz in {self.data_dir}"
                )

            self.val_dataset = OPFDataset(
                data_path=test_path,
                topo_path=topo_path,
                norm_path=norm_path_str,
                normalize=self.normalize,
                split="test",
                feature_type=self.feature_type,
                feature_params=self.feature_params,
            )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,  # Avoid batch size issues in training
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader()")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader (uses val_dataset from 'test' stage)."""
        return self.val_dataloader()

    def get_norm_stats(self) -> dict | None:
        """
        Get normalization statistics from training dataset.

        Returns:
            dict with keys: pd_mean, pd_std, qd_mean, qd_std,
                           pg_mean, pg_std, vg_mean, vg_std
            or None if normalization is disabled
        """
        if self.train_dataset is not None:
            return self.train_dataset.get_norm_stats()
        if self.val_dataset is not None:
            return self.val_dataset.get_norm_stats()
        return None

    def get_gen_bus_map(self):
        """Get generator-to-bus mapping from dataset."""
        if self.train_dataset is not None:
            return self.train_dataset.get_gen_bus_map()
        if self.val_dataset is not None:
            return self.val_dataset.get_gen_bus_map()
        raise RuntimeError("setup() must be called first")

    @property
    def n_bus(self) -> int:
        """Number of buses in the power system."""
        ds = self.train_dataset or self.val_dataset
        if ds is None:
            raise RuntimeError("setup() must be called first")
        return ds.pd.shape[1]

    @property
    def n_gen(self) -> int:
        """Number of generators in the power system."""
        ds = self.train_dataset or self.val_dataset
        if ds is None:
            raise RuntimeError("setup() must be called first")
        return ds.pg_labels.shape[1]

    @property
    def input_dim(self) -> int:
        """
        Input dimension for 'flat' feature type.

        Returns: 2*N_BUS + 2*N_BUS^2 (pd + qd + g_flat + b_flat)
        """
        if self.feature_type != "flat":
            raise ValueError("input_dim is only available for 'flat' feature_type")
        n = self.n_bus
        return 2 * n + 2 * n * n

    @property
    def feature_iterations(self) -> int | None:
        """
        Number of feature iterations for 'graph' feature type.

        Returns: k (number of iterations) or None if not set
        """
        if self.feature_type != "graph":
            return None
        ds = self.train_dataset or self.val_dataset
        if ds is None or ds.e_0_k is None:
            return None
        return ds.e_0_k.shape[2]
