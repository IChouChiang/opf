# ppc_to_g_b.py
# -*- coding: utf-8 -*-
"""
Build bus-admittance matrix from a PYPOWER/MATPOWER case (ppc) and
return its real/imag parts G, B for use in an AbstractModel.

This uses PYPOWER's built-in makeYbus(), which correctly accounts for:
- line series impedance (r, x), charging (b),
- transformer off-nominal tap ratio (ratio) and phase shift (angle),
- branch in/out of service status.

Outputs can be dense NumPy arrays (default) or sparse SciPy matrices.

Example
-------
>>> from pypower.case9 import case9
>>> from ppc_to_g_b import ppc_to_g_b
>>> ppc = case9()
>>> G, B = ppc_to_g_b(ppc)     # dense float64 arrays
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    # PYPOWER API
    from pypower.makeYbus import makeYbus
    from pypower.ext2int import ext2int
except Exception as e:  # pragma: no cover
    raise ImportError("pypower is required. Install with `pip install pypower`.") from e


def ppc_to_g_b(
    ppc: dict,
    *,
    return_sparse: bool = False,
    zero_tol: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a PYPOWER/MATPOWER case dict to (G, B).

    Parameters
    ----------
    ppc : dict
        Case dict with keys "baseMVA", "bus", "branch".
    return_sparse : bool, optional
        If True, return SciPy sparse matrices (CSC) with dtype float64.
        If False (default), return dense NumPy arrays (float64).
    zero_tol : float or None, optional
        If given, values with absolute magnitude < zero_tol are set to 0
        (after conversion to the requested format).

    Returns
    -------
    G : ndarray or sparse matrix (float64)
        Real part of Ybus (conductance matrix).
    B : ndarray or sparse matrix (float64)
        Imag part of Ybus (susceptance matrix).

    Notes
    -----
    - Ybus is built in the row/column order of `ppc["bus"]`.
    - Units: per-unit on `ppc["baseMVA"]` with bus voltage base from the case.
    """
    # Convert to internal bus numbering (0-based consecutive)
    ppc_int = ext2int(ppc)

    baseMVA = float(ppc_int["baseMVA"])
    bus = ppc_int["bus"]
    branch = ppc_int["branch"]

    # Build Ybus using PYPOWER
    Ybus, _, _ = makeYbus(baseMVA, bus, branch)

    if return_sparse:
        # Ensure float64 real/imag sparse matrices (CSC)
        G = Ybus.real.astype(np.float64).tocsc()
        B = Ybus.imag.astype(np.float64).tocsc()
        if zero_tol is not None:
            # Eliminate near-zeros to keep sparsity clean
            G.data[np.abs(G.data) < zero_tol] = 0.0
            G.eliminate_zeros()
            B.data[np.abs(B.data) < zero_tol] = 0.0
            B.eliminate_zeros()
        return G, B
    else:
        # Dense float64 ndarrays
        G = np.asarray(Ybus.real.toarray(), dtype=np.float64)
        B = np.asarray(Ybus.imag.toarray(), dtype=np.float64)
        if zero_tol is not None:
            G[np.abs(G) < zero_tol] = 0.0
            B[np.abs(B) < zero_tol] = 0.0
        return G, B


__all__ = ["ppc_to_g_b"]
