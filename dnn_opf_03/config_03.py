"""Configuration for DeepOPF-FT Model 03 (Baseline)."""

from dataclasses import dataclass
import json


@dataclass
class ModelConfig:
    """Model architecture hyperparameters for DeepOPF-FT Baseline."""

    n_bus: int = 6
    n_gen: int = 3

    # Input dimension: 2*N_BUS (Pd, Qd) + 2*N_BUS*N_BUS (G, B flattened)
    # For Case6ww: 12 + 72 = 84
    input_dim: int = 84

    # Flexible layer configuration
    # If hidden_layers is provided, it overrides hidden_dim/n_hidden_layers
    hidden_layers: list = None

    # Legacy support
    hidden_dim: int = 1000
    n_hidden_layers: int = 3

    activation: str = "relu"

    @classmethod
    def from_json(cls, json_path):
        """Load configuration from a JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        model_params = config_dict.get("model", {})

        # Filter out unknown keys
        valid_keys = cls.__annotations__.keys()
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}

        return cls(**filtered_params)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 24  # Matched to GCNN optimal
    epochs: int = 50

    @classmethod
    def from_json(cls, json_path):
        """Load configuration from a JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        train_params = config_dict.get("training", {})

        # Filter out unknown keys
        valid_keys = cls.__annotations__.keys()
        filtered_params = {k: v for k, v in train_params.items() if k in valid_keys}

        return cls(**filtered_params)
