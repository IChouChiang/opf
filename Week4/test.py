# test.py
# -*- coding: utf-8 -*-
"""
Two-part test for AC-OPF.

Part 1 — Pyomo AbstractModel (our model) + data loaded from helpers:
    - Build model via ac_opf_create()
    - Extract case57 data using case_to_col()
    - Build Ybus -> (G, B) using ppc_to_g_b()
    - Map gencost to polynomial/piecewise Params
    - Solve with Gurobi (NonConvex QCQP / MIQCP)

Part 2 — PYPOWER built-in AC-OPF:
    - Solve case57 with runopf()

All files (ac_opf_create.py, case_to_col.py, ppc_to_g_b.py, vm_va_to_e_f.py)
must be in the same folder as this script.
"""

from __future__ import annotations
import sys
from typing import Dict, Tuple
import numpy as np

import pyomo.environ as pyo

# Helpers from this folder
from ac_opf_create import ac_opf_create
from case_to_col import case_to_col
from ppc_to_g_b import ppc_to_g_b
from vm_va_to_e_f import vm_va_to_e_f

# PYPOWER case + built-in solver
from pypower.case57 import case57
from pypower.runopf import runopf


# ---------- Utilities to build Pyomo instance data ----------


def _build_sets(ppc: dict) -> Tuple[list, list]:
    """BUS uses bus_i numbers; GEN are 0..ng-1 (units)."""
    bus_ids = ppc["bus"][:, 0].astype(int).tolist()
    gen_ids = list(range(len(ppc["gen"])))
    return bus_ids, gen_ids


def _build_gen_bus_map(ppc: dict) -> Dict[int, int]:
    """GEN -> BUS mapping keyed by generator index (0..ng-1)."""
    gen_bus = ppc["gen"][:, 0].astype(int)
    return {g: int(gen_bus[g]) for g in range(len(gen_bus))}


def _build_cost_params_from_gencost(ppc: dict) -> Dict[str, object]:
    """
    Convert MATPOWER gencost rows into:
      - cost_model[g] in {1,2}
      - n_cost[g] >= 1
      - if model==2 (polynomial): cost_coeff[g,k] for k=0..n-1 (PU SCALED)
      - if model==1 (piecewise):  pw_x[g,k], pw_y[g,k] for k=1..n (x in p.u.)
    """
    gc = np.asarray(ppc["gencost"], dtype=float)
    baseMVA = float(ppc["baseMVA"])
    ng = gc.shape[0]

    cost_model = {}
    n_cost = {}
    cost_coeff = {}  # (g,k) -> c_k  (k=0 constant)
    pw_x = {}  # (g,k) -> x_k  (k starts at 1)
    pw_y = {}  # (g,k) -> y_k

    for g in range(ng):
        model = int(gc[g, 0])
        N = int(gc[g, 3])
        tail = gc[g, 4:]

        cost_model[g] = model
        n_cost[g] = N

        if model == 2:
            # Polynomial: tail = [c_{N-1}, ..., c0] in native $/MW^k
            if tail.size != N:
                raise ValueError(
                    f"gencost row {g}: expected {N} coeffs, got {tail.size}"
                )
            # Convert to p.u. coefficients for p = PG_pu:
            # c_k_pu = c_k_native / baseMVA^k, with k=0..N-1
            coeff_high_to_low = tail.tolist()
            for pos, c_native in enumerate(coeff_high_to_low):
                k = (N - 1) - pos  # exponent
                scale = baseMVA**k if k > 0 else 1.0
                cost_coeff[(g, k)] = float(c_native) / scale

        elif model == 1:
            # Piecewise: tail = [x1, y1, ..., xN, yN]
            if tail.size != 2 * N:
                raise ValueError(
                    f"gencost row {g}: expected {2*N} pw entries, got {tail.size}"
                )
            for k in range(1, N + 1):
                x_MW = float(tail[2 * (k - 1) + 0])
                y = float(tail[2 * (k - 1) + 1])
                pw_x[(g, k)] = x_MW / baseMVA  # p.u.
                pw_y[(g, k)] = y  # currency
        else:
            raise ValueError(f"Unsupported cost model {model} at row {g}")

    return dict(
        cost_model=cost_model,
        n_cost=n_cost,
        cost_coeff=cost_coeff,
        pw_x=pw_x,
        pw_y=pw_y,
    )


def _build_instance_data(ppc: dict) -> Dict:
    """Create a Pyomo data dict for ac_opf_create() from a PYPOWER case."""
    store = case_to_col(ppc)  # our uniform columns
    G, B = ppc_to_g_b(ppc, return_sparse=False)
    Vm, Va_deg = store["bus_Vm"], store["bus_Va"]
    e0, f0 = vm_va_to_e_f(Vm, Va_deg)

    BUS, GEN = _build_sets(ppc)
    gen_bus_map = _build_gen_bus_map(ppc)
    cost = _build_cost_params_from_gencost(ppc)

    # Build dicts keyed by IDs
    # Bus params
    PD = {i: store["bus_Pd"][k] for k, i in enumerate(BUS)}
    QD = {i: store["bus_Qd"][k] for k, i in enumerate(BUS)}
    Vmin = {i: store["bus_Vmin"][k] for k, i in enumerate(BUS)}
    Vmax = {i: store["bus_Vmax"][k] for k, i in enumerate(BUS)}

    # Generator limits
    PGmin = {g: store["gen_Pmin"][g] for g in GEN}
    PGmax = {g: store["gen_Pmax"][g] for g in GEN}
    QGmin = {g: store["gen_Qmin"][g] for g in GEN}
    QGmax = {g: store["gen_Qmax"][g] for g in GEN}

    # Admittance matrices mapped to (i,j) by bus ID order
    idx = {i: k for k, i in enumerate(BUS)}
    G_map = {(i, j): float(G[idx[i], idx[j]]) for i in BUS for j in BUS}
    B_map = {(i, j): float(B[idx[i], idx[j]]) for i in BUS for j in BUS}

    # Assemble Pyomo data dict
    data = {
        None: {  # global
            "BUS": {None: BUS},
            "GEN": {None: GEN},
            "GEN_BUS": gen_bus_map,
            "PD": PD,
            "QD": QD,
            "PGmin": PGmin,
            "PGmax": PGmax,
            "QGmin": QGmin,
            "QGmax": QGmax,
            "Vmin": Vmin,
            "Vmax": Vmax,
            "G": G_map,
            "B": B_map,
            # Cost parameters
            "cost_model": cost["cost_model"],
            "n_cost": cost["n_cost"],
            "cost_coeff": cost["cost_coeff"],
            "pw_x": cost["pw_x"],
            "pw_y": cost["pw_y"],
        }
    }

    return data


# ---------- Part 1: our Pyomo model + Gurobi ----------


def solve_pyomo_with_gurobi(ppc: dict):
    print("\n=== Part 1: Pyomo AC-OPF with Gurobi (NonConvex) ===")
    model = ac_opf_create()
    data = _build_instance_data(ppc)
    instance = model.create_instance(data)  # type: ignore

    # Gurobi options: enable nonconvex QCQP/MIQCP with safe limits for overnight run
    opt = pyo.SolverFactory("gurobi")
    if not opt.available():
        raise RuntimeError("Gurobi solver is not available to Pyomo.")
    results = opt.solve(
        instance,
        tee=True,
        options={
            "NonConvex": 2,  # Enable non-convex quadratic optimization
            "MIPGap": 1e-2,  # Relax gap to 1% (faster convergence)
            "Threads": 8,  # Use 8 threads (safe for overnight, leaves cores for system)
            "TimeLimit": 28800,  # 8 hours max (28800 seconds)
            "NodeLimit": 100000,  # Stop after 100k nodes if not converged
            "MIPFocus": 1,  # Focus on finding feasible solutions quickly
        },
    )

    instance.solutions.store_to(results)
    print("\nSolver status:", results.solver.status)
    print("Termination  :", results.solver.termination_condition)

    # Extract a few values
    obj = pyo.value(instance.TotalCost)
    PG = {g: pyo.value(instance.PG[g]) for g in instance.GEN}  # type: ignore
    print(f"Objective value: {obj:.6g}")
    print("First 5 PG (p.u.):", list(PG.items())[:5])

    return instance, results


# ---------- Part 2: built-in PYPOWER AC-OPF ----------


def solve_with_pypower(ppc: dict):
    print("\n=== Part 2: PYPOWER runopf(case57) ===")
    results = runopf(ppc)  # type: ignore  # default AC OPF settings
    success = bool(results["success"])  # type: ignore
    print("Success:", success)
    if success:
        print("Objective (PYPOWER):", float(results["f"]))  # type: ignore
        # Print first few generator outputs (MW)
        print("First 5 PG (MW):", results["gen"][:5, 1].tolist())  # type: ignore
    return results


# ---------- Main ----------

if __name__ == "__main__":
    # Load case
    ppc = case57()

    # Part 1: our model + Gurobi
    try:
        _, _ = solve_pyomo_with_gurobi(ppc)
    except Exception as e:
        print("Pyomo/Gurobi run failed:", e, file=sys.stderr)

    # Part 2: PYPOWER built-in
    try:
        _ = solve_with_pypower(ppc)
    except Exception as e:
        print("PYPOWER runopf failed:", e, file=sys.stderr)
