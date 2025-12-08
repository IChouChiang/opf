"""Configuration for GCNN OPF Model 01."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""

    n_bus: int = 6  # number of buses in case6ww (Wood & Wollenberg 6-bus)
    n_gen: int = 3  # number of generators in case6ww
    feature_iterations: int = 8  # k feature construction iterations
    channels_gc_in: int = 8  # input channels (should match feature_iterations)
    channels_gc_out: int = 8  # graph conv output channels
    neurons_fc: int = 256  # neurons per FC layer
    n_gc_layers: int = 2  # number of graph conv layers (gc1, gc2)
    n_fc_layers: int = 1  # number of fully connected layers (fc1)
    activation_gs: str = "tanh"  # activation function for GC layers
    dropout: float = 0.0  # dropout probability

    @classmethod
    def from_json(cls, json_path):
        """Load configuration from a JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        model_params = config_dict.get("model", {})

        # Ensure channels_gc_in matches feature_iterations if not explicitly set
        if (
            "feature_iterations" in model_params
            and "channels_gc_in" not in model_params
        ):
            model_params["channels_gc_in"] = model_params["feature_iterations"]

        # Filter out unknown keys to avoid TypeError
        valid_keys = cls.__annotations__.keys()
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}

        return cls(**filtered_params)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    optimizer: str = "adam"
    batch_size: int = 6
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 50
    early_stopping_patience: int = 5
    kappa: float = 0.1
    use_physics_loss: bool = True
    two_stage: bool = False
    phase1_epochs: int = 25
    phase2_epochs: int = 25

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


# Convenience instances for direct import (using defaults)
model = ModelConfig()
training = TrainingConfig()


def load_config(json_path):
    """Load both model and training configs from a JSON file."""
    return ModelConfig.from_json(json_path), TrainingConfig.from_json(json_path)


# Legacy compatibility (optional - remove if you update all imports)
N_BUS = model.n_bus
N_GEN = model.n_gen
CHANNELS_GC_IN = model.channels_gc_in
CHANNELS_GC_OUT = model.channels_gc_out
NEURONS_FC = model.neurons_fc
N_GC_LAYERS = model.n_gc_layers
N_FC_LAYERS = model.n_fc_layers
ACTIVATION = model.activation_gs
OPTIMIZER = training.optimizer
