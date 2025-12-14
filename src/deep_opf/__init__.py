"""deep_opf: Deep Learning for Optimal Power Flow.

A PyTorch library for physics-informed deep learning applied to AC-OPF problems.

Modules:
    data: Data loading and preprocessing (OPFDataModule, OPFDataset)
    models: Neural network architectures (AdmittanceDNN, GCNN)
    loss: Physics-informed loss functions
    optimization: Traditional OPF solvers (AC-OPF with Pyomo)
    task: PyTorch Lightning training module (OPFTask)
"""

from . import data, loss, models, optimization
from .task import OPFTask

__all__ = ["data", "models", "loss", "optimization", "OPFTask"]
