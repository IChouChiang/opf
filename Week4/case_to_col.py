# case_to_col.py
from __future__ import annotations
from typing import Dict
import numpy as np


def _as_f64(a) -> np.ndarray:
    """Ensure 1-D float64 ndarray (copied)."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1).astype(np.float64, copy=False)
    return arr.astype(np.float64, copy=False)


def _col(mat: np.ndarray, idx_1based: int) -> np.ndarray:
    """Get a 1-D float64 column (1-based index) from a 2-D matrix."""
    return _as_f64(mat[:, idx_1based - 1])


def _extract_gencost_csr(gencost_mat: np.ndarray, baseMVA: float):
    """
    Build CSR-style arrays for variable-length gencost rows.

    Returns dict with:
      - gencost_model, gencost_startup, gencost_shutdown, gencost_n
      - gencost_coeff_ptr, gencost_coeff_val     (for model=2 rows)
      - gencost_pw_ptr,    gencost_pw_x, gencost_pw_y  (for model=1 rows)
    All arrays are 1-D float64.
    """
    # Fixed per-row fields
    model = gencost_mat[:, 0].astype(np.float64)
    startup = gencost_mat[:, 1].astype(np.float64)  # currency
    shutdown = gencost_mat[:, 2].astype(np.float64)  # currency
    n = gencost_mat[:, 3].astype(np.float64)  # number of points/coeffs

    coeff_ptr = [0.0]  # float64 for uniformity
    coeff_val = []  # p.u.-scaled coefficients for model=2

    pw_ptr = [0.0]
    pw_x = []  # p.u. P-coordinates for model=1
    pw_y = []  # currency f(P) for model=1

    for i, row in enumerate(gencost_mat):
        mdl = int(row[0])
        N = int(row[3])
        tail = row[4:]

        if mdl == 2:
            # Polynomial: COST = [c_{N-1}, ..., c0] (length N)
            if tail.size != N:
                raise ValueError(
                    f"gencost row {i}: expected {N} coeffs, got {tail.size}"
                )
            # Scale to p.u.: c_k_pu = c_k / baseMVA^k   (k = degree of p^k)
            # Here tail[0] is c_{N-1} (highest degree), degree = N-1-k
            for k, c_native in enumerate(tail):
                degree = (N - 1) - k
                scale = (baseMVA**degree) if degree > 0 else 1.0
                coeff_val.append(float(c_native) / scale)
            coeff_ptr.append(coeff_ptr[-1] + float(N))

        elif mdl == 1:
            # Piecewise linear: COST = [x1, y1, ..., xN, yN] (length 2N)
            if tail.size != 2 * N:
                raise ValueError(
                    f"gencost row {i}: expected {2*N} pw entries, got {tail.size}"
                )
            for j in range(N):
                x_MW = float(tail[2 * j + 0])
                y = float(tail[2 * j + 1])  # currency
                pw_x.append(x_MW / baseMVA)  # p.u.
                pw_y.append(y)
            pw_ptr.append(pw_ptr[-1] + float(N))

        else:
            raise ValueError(f"gencost row {i}: unsupported model {mdl}")

    out = {
        "gencost_model": model,
        "gencost_startup": startup,
        "gencost_shutdown": shutdown,
        "gencost_n": n,
        "gencost_coeff_ptr": np.asarray(coeff_ptr, dtype=np.float64),
        "gencost_coeff_val": np.asarray(coeff_val, dtype=np.float64),
        "gencost_pw_ptr": np.asarray(pw_ptr, dtype=np.float64),
        "gencost_pw_x": np.asarray(pw_x, dtype=np.float64),
        "gencost_pw_y": np.asarray(pw_y, dtype=np.float64),
    }
    return out


def case_to_col(ppc: Dict) -> Dict[str, np.ndarray]:
    """
    Convert a PYPOWER/MATPOWER case dict to a uniform column store.

    Parameters
    ----------
    ppc : dict
        A case dict with keys "baseMVA", "bus", "gen", "branch".

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from standardized field names to 1-D float64 arrays.
    """
    baseMVA = float(ppc["baseMVA"])

    bus = np.asarray(ppc["bus"], dtype=np.float64)
    gen = np.asarray(ppc["gen"], dtype=np.float64)
    branch = np.asarray(ppc["branch"], dtype=np.float64)

    store: Dict[str, np.ndarray] = {}

    # ----------------------
    # BUS table (Table B-1)
    # ----------------------
    # Raw columns
    store["bus_bus_i"] = _col(bus, 1)  # index
    store["bus_type"] = _col(bus, 2)  # enum
    # MW/MVAr -> p.u.
    store["bus_Pd"] = _col(bus, 3) / baseMVA
    store["bus_Qd"] = _col(bus, 4) / baseMVA
    store["bus_Gs"] = _col(bus, 5) / baseMVA
    store["bus_Bs"] = _col(bus, 6) / baseMVA
    # Other metadata / limits (already p.u. or as-specified)
    store["bus_area"] = _col(bus, 7)
    store["bus_Vm"] = _col(bus, 8)  # p.u.
    store["bus_Va"] = _col(bus, 9)  # degrees
    store["bus_baseKV"] = _col(bus, 10)  # kV
    store["bus_zone"] = _col(bus, 11)
    store["bus_Vmax"] = _col(bus, 12)  # p.u.
    store["bus_Vmin"] = _col(bus, 13)  # p.u.

    # -----------------------
    # GEN table (Table B-2)
    # -----------------------
    store["gen_bus"] = _col(gen, 1)
    store["gen_Pg"] = _col(gen, 2) / baseMVA
    store["gen_Qg"] = _col(gen, 3) / baseMVA
    store["gen_Qmax"] = _col(gen, 4) / baseMVA
    store["gen_Qmin"] = _col(gen, 5) / baseMVA
    store["gen_Vg"] = _col(gen, 6)  # p.u.
    store["gen_mBase"] = _col(gen, 7)  # MVA (kept raw)
    store["gen_status"] = _col(gen, 8)  # 1/0
    store["gen_Pmax"] = _col(gen, 9) / baseMVA
    store["gen_Pmin"] = _col(gen, 10) / baseMVA
    store["gen_Pc1"] = _col(gen, 11) / baseMVA
    store["gen_Pc2"] = _col(gen, 12) / baseMVA
    store["gen_Qc1min"] = _col(gen, 13) / baseMVA
    store["gen_Qc1max"] = _col(gen, 14) / baseMVA
    store["gen_Qc2min"] = _col(gen, 15) / baseMVA
    store["gen_Qc2max"] = _col(gen, 16) / baseMVA
    # Ramps: MW/min or MW or MVAr/min -> p.u.(/min)
    store["gen_ramp_agc"] = _col(gen, 17) / baseMVA  # p.u./min
    store["gen_ramp_10"] = _col(gen, 18) / baseMVA  # p.u.
    store["gen_ramp_30"] = _col(gen, 19) / baseMVA  # p.u.
    store["gen_ramp_q"] = _col(gen, 20) / baseMVA  # p.u./min
    store["gen_apf"] = _col(gen, 21)  # unitless

    # ---------------------------
    # BRANCH table (Table B-3)
    # ---------------------------
    store["branch_fbus"] = _col(branch, 1)
    store["branch_tbus"] = _col(branch, 2)
    # r/x/b are already per-unit in MATPOWER case files
    store["branch_r"] = _col(branch, 3)  # p.u.
    store["branch_x"] = _col(branch, 4)  # p.u.
    store["branch_b"] = _col(branch, 5)  # p.u.
    # Rate A/B/C are MVA → convert to p.u. (0 means "no limit")
    rateA = _col(branch, 6)
    rateB = _col(branch, 7)
    rateC = _col(branch, 8)
    with np.errstate(invalid="ignore", divide="ignore"):
        store["branch_rateA"] = np.where(rateA == 0.0, np.inf, rateA / baseMVA)
        store["branch_rateB"] = np.where(rateB == 0.0, np.inf, rateB / baseMVA)
        store["branch_rateC"] = np.where(rateC == 0.0, np.inf, rateC / baseMVA)

    # Tap ratio: 0 → 1.0 (line), otherwise keep (already p.u.)
    ratio = _col(branch, 9)
    store["branch_ratio"] = np.where(ratio == 0.0, 1.0, ratio)

    store["branch_angle"] = _col(branch, 10)  # degrees
    store["branch_status"] = _col(branch, 11)  # 1/0
    store["branch_angmin"] = _col(branch, 12)  # degrees
    store["branch_angmax"] = _col(branch, 13)  # degrees

    # Optionally include baseMVA as a 1-length array to preserve strict uniformity.
    store["baseMVA"] = np.array([baseMVA], dtype=np.float64)

    # -------------------------------
    # GENCOST table (Table B-4)
    # -------------------------------
    gencost = np.asarray(ppc["gencost"], dtype=np.float64)
    gcost = _extract_gencost_csr(gencost, baseMVA)

    # Merge into the main store
    for k, v in gcost.items():
        store[k] = v

    return store
