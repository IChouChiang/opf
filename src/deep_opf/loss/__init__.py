"""Loss functions for OPF deep learning models.

This module provides physics-informed loss functions for training OPF models,
including active power balance constraints derived from AC power flow equations.
"""

from .physics import (
    build_gen_bus_matrix,
    compute_power_from_voltage,
    correlative_loss,
    correlative_loss_pg,  # Legacy alias
    physics_loss,
)

__all__ = [
    "build_gen_bus_matrix",
    "compute_power_from_voltage",
    "correlative_loss",
    "correlative_loss_pg",
    "physics_loss",
]
