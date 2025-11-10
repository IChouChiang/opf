# vm_va_to_e_f.py
# -*- coding: utf-8 -*-
"""
Convert voltage magnitude/angle to Cartesian components.

Given per-bus voltage magnitude Vm (p.u.) and angle Va (degrees),
compute:
    e_i = Vm_i * cos(Va_i)
    f_i = Vm_i * sin(Va_i)
where the trigonometric functions take Va in **radians**.

All inputs are broadcast to 1-D float64 arrays of equal length.
Outputs are 1-D float64 arrays (e, f) suitable for feeding into
a Pyomo AbstractModel.

Example
-------
>>> import numpy as np
>>> from vm_va_to_e_f import vm_va_to_e_f
>>> Vm = np.array([1.0, 0.98, 1.02])
>>> Va = np.array([0.0, -5.0, 3.0])   # degrees
>>> e, f = vm_va_to_e_f(Vm, Va)
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def vm_va_to_e_f(Vm, Va_deg, *, validate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert voltage magnitude (p.u.) and angle (degrees) to (e, f).

    Parameters
    ----------
    Vm : array-like
        Voltage magnitudes in per-unit. Shape (n,) or broadcastable to it.
    Va_deg : array-like
        Voltage angles in **degrees**. Shape (n,) or broadcastable to Vm.
    validate : bool, optional
        If True, checks that inputs are 1-D and of equal length after broadcasting.

    Returns
    -------
    e : np.ndarray (float64, shape (n,))
        e_i = Vm_i * cos(Va_i) with Va_i converted to radians.
    f : np.ndarray (float64, shape (n,))
        f_i = Vm_i * sin(Va_i) with Va_i converted to radians.

    Notes
    -----
    - NaNs in the inputs propagate to the outputs.
    - Angles are interpreted as **degrees** and internally converted to radians.
    - Output dtype is float64 for numerical stability.
    """
    Vm_arr = np.asarray(Vm, dtype=np.float64)
    Va_arr_deg = np.asarray(Va_deg, dtype=np.float64)

    # Broadcast to common shape, then enforce 1-D
    try:
        Vm_b, Va_b = np.broadcast_arrays(Vm_arr, Va_arr_deg)
    except ValueError as exc:
        raise ValueError(f"Vm and Va_deg are not broadcastable: {exc}") from exc

    if validate:
        if Vm_b.ndim != 1:
            # Flatten if the inputs were given as column/row vectors
            if Vm_b.ndim == 2 and 1 in Vm_b.shape:
                Vm_b = Vm_b.reshape(-1)
                Va_b = Va_b.reshape(-1)
            else:
                raise ValueError(
                    f"Expected 1-D inputs after broadcast; got shape {Vm_b.shape}"
                )

        if Vm_b.shape != Va_b.shape:
            raise ValueError(
                f"Shapes must match after broadcast, got {Vm_b.shape} and {Va_b.shape}"
            )

    # Convert angle to radians
    Va_rad = np.deg2rad(Va_b)

    # Cartesian components
    e = Vm_b * np.cos(Va_rad)
    f = Vm_b * np.sin(Va_rad)

    # Ensure float64 1-D outputs
    return e.astype(np.float64, copy=False), f.astype(np.float64, copy=False)


__all__ = ["vm_va_to_e_f"]
