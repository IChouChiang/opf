"""Configuration for DeepOPF-FT Model 03 (128 Neurons)."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture hyperparameters for DeepOPF-FT Baseline."""

    n_bus: int = 6
    n_gen: int = 3

    # Input dimension: 2*N_BUS (Pd, Qd) + 2*N_BUS*N_BUS (G, B flattened)
    # For Case6ww: 12 + 72 = 84
    input_dim: int = 84

    hidden_dim: int = 128  # Reduced from 1000 to match ~46k params
    n_hidden_layers: int = 3  # Number of hidden layers (excluding input/output)
    activation: str = "relu"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 24  # Matched to GCNN optimal
