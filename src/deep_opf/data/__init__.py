"""Unified data loading utilities for OPF models.

This module provides:
- OPFDataset: PyTorch Dataset supporting 'flat' (DNN) and 'graph' (GCNN) features
- OPFDataModule: PyTorch Lightning DataModule for streamlined data loading

Example usage:
    from deep_opf.data import OPFDataset, OPFDataModule

    # For DNN models (flat features)
    dm = OPFDataModule(
        data_dir="path/to/data",
        batch_size=64,
        feature_type="flat",
    )

    # For GCNN models (graph features)
    dm = OPFDataModule(
        data_dir="path/to/data",
        batch_size=64,
        feature_type="graph",
        feature_params={"feature_iterations": 5},
    )
"""

from .dataset import OPFDataset
from .datamodule import OPFDataModule

__all__ = ["OPFDataset", "OPFDataModule"]
