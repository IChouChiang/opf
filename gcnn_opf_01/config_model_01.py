"""Configuration for GCNN OPF Model 01."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""

    n_bus: int = 6  # number of buses in case6ww (Wood & Wollenberg 6-bus)
    n_gen: int = 3  # number of generators in case6ww
    channels_gc_in: int = 8  # k=8 feature construction iterations (e^0...e^7, f^0...f^7)
    channels_gc_out: int = 8  # graph conv output channels
    neurons_fc: int = 1000  # neurons per FC layer
    n_gc_layers: int = 2  # number of graph conv layers (gc1, gc2)
    n_fc_layers: int = 1  # number of fully connected layers (fc1)
    activation_gs: str = "tanh"  # activation function for GC layers


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    optimizer: str = "adam"
    # Add more as needed: weight_decay, scheduler params, etc.


# Convenience instances for direct import
model = ModelConfig()
training = TrainingConfig()

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
