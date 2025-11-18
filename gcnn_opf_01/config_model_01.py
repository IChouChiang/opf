"""Configuration for GCNN OPF Model 01."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""

    n_bus: int = 39  # number of buses in IEEE 39-bus system
    n_gen: int = 10  # number of generators in IEEE 39-bus system
    channels_gc_in: int = (
        8  # graph conv input channels:e^0, f^0, PD, QD, G_{ndiag}, B_{ndiag}, G_{diag}, B_{diag}
    )
    channels_gc_out: int = 8  # graph conv channels
    neurons_fc: int = 1000  # neurons per FC layer
    n_gc_layers: int = 3  # number of graph conv layers
    n_fc_layers: int = 3  # number of fully connected layers
    activation_gs: str = "tanh"  # activation funtion for GC layers


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
