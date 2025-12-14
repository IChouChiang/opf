"""Deep OPF Models.

This module provides neural network models for Optimal Power Flow prediction.

Available Models:
    - AdmittanceDNN: Fully connected network for flat feature input
    - GCNN: Physics-guided graph convolutional network
    - GraphConv: Physics-guided graph convolution layer (used by GCNN)

All models follow a common interface, returning a dictionary with keys:
    - 'pg': [batch_size, n_gen] - Active power generation
    - 'vg': [batch_size, n_gen] - Generator voltage magnitudes
    - 'v_bus': [batch_size, n_bus, 2] - All bus voltages (e, f components)
"""

from .dnn import AdmittanceDNN
from .gcnn import GCNN, GraphConv

__all__ = [
    "AdmittanceDNN",
    "GCNN",
    "GraphConv",
]
