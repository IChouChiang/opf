# helpers_ac_opf.py - Shared AC-OPF helper functions for Week4
# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false, reportGeneralTypeIssues=false
"""Reusable helper functions for AC Optimal Power Flow experiments.

Extracted from duplicated logic in `test.py` (IEEE 39-bus) and `test2.py` (IEEE 57-bus).

Functions
---------
prepare_ac_opf_data(ppc):
    Convert a PYPOWER case dictionary to a Pyomo data dict for ac_opf_create().
initialize_voltage_from_flatstart(instance, ppc_int):
    Initialize Cartesian voltage variables (e,f) from Vm/Va in internal case.
solve_ac_opf(ppc, verbose=True, time_limit=180, mip_gap=0.03, threads=None):
    Build model, prepare data, create instance, warm start, and solve with Gurobi.

Notes
-----
- Uses PYPOWER's ext2int() to convert to internal numbering before building Ybus.
- Ensures all MW / MVAr quantities are converted to per-unit on baseMVA.
- Cost scaling: a = c2 * baseMVA^2, b = c1 * baseMVA, c = c0 for polynomial costs with PG in p.u.
- Slack bus (type == 3) voltage is fixed to remove rotational symmetry.
"""
from __future__ import annotations
import warnings
import numpy as np
import multiprocessing
import pyomo.environ as pyo

try:
    from ac_opf_create import ac_opf_create
except ImportError:
    from src.ac_opf_create import ac_opf_create

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_ac_opf_data(ppc):
    """Prepare data dict for AC-OPF AbstractModel.create_instance().

    Parameters
    ----------
    ppc : dict
        PYPOWER case dictionary (external numbering)

    Returns
    -------
    data : dict
        Pyomo data dictionary suitable for AbstractModel.create_instance(data=data)
    ppc_int : dict
        Internal-numbered PYPOWER case (ext2int applied)
    """
    from pypower.makeYbus import makeYbus
    from pypower.ext2int import ext2int

    # Convert to internal numbering
    ppc_int = ext2int(ppc)

    baseMVA = float(ppc_int["baseMVA"])
    bus = ppc_int["bus"]
    gen = ppc_int["gen"]
    branch = ppc_int["branch"]
    gencost = ppc_int.get("gencost", None)

    n_bus = bus.shape[0]
    n_gen = gen.shape[0]

    # Build Ybus using PYPOWER
    Ybus, _, _ = makeYbus(baseMVA, bus, branch)
    G = np.asarray(Ybus.real.toarray(), dtype=np.float64)
    B = np.asarray(Ybus.imag.toarray(), dtype=np.float64)

    # Prepare data dict for Pyomo
    data = {None: {}}

    # Sets
    data[None]["BUS"] = {None: list(range(n_bus))}
    data[None]["GEN"] = {None: list(range(n_gen))}

    # GEN_BUS mapping (which bus each generator is on) - internal numbering already 0-based
    gen_bus_map = {int(g): int(gen[g, 0]) for g in range(n_gen)}
    data[None]["GEN_BUS"] = gen_bus_map

    # Bus parameters (demand and voltage limits)
    PD_dict = {}
    QD_dict = {}
    Vmin_dict = {}
    Vmax_dict = {}
    for i in range(n_bus):
        PD_dict[i] = float(bus[i, 2]) / baseMVA  # Pd in p.u.
        QD_dict[i] = float(bus[i, 3]) / baseMVA  # Qd in p.u.
        Vmin_dict[i] = float(bus[i, 12])  # already p.u.
        Vmax_dict[i] = float(bus[i, 11])  # already p.u.
    data[None]["PD"] = PD_dict
    data[None]["QD"] = QD_dict
    data[None]["Vmin"] = Vmin_dict
    data[None]["Vmax"] = Vmax_dict

    # Generator parameters (convert MW/MVAr to p.u.)
    PGmin_dict = {}
    PGmax_dict = {}
    QGmin_dict = {}
    QGmax_dict = {}
    for g in range(n_gen):
        PGmin_dict[g] = float(gen[g, 9]) / baseMVA
        PGmax_dict[g] = float(gen[g, 8]) / baseMVA
        QGmin_dict[g] = float(gen[g, 4]) / baseMVA
        QGmax_dict[g] = float(gen[g, 3]) / baseMVA
    data[None]["PGmin"] = PGmin_dict
    data[None]["PGmax"] = PGmax_dict
    data[None]["QGmin"] = QGmin_dict
    data[None]["QGmax"] = QGmax_dict

    # Admittance matrices (dense Param[BUS,BUS])
    G_dict = {}
    B_dict = {}
    for i in range(n_bus):
        for j in range(n_bus):
            G_dict[(i, j)] = float(G[i, j])
            B_dict[(i, j)] = float(B[i, j])
    data[None]["G"] = G_dict
    data[None]["B"] = B_dict

    # Cost coefficients - support both gencost formats
    cost_model_dict = {}
    n_cost_dict = {}
    cost_coeff_dict = {}
    pw_x_dict = {}
    pw_y_dict = {}
    if gencost is not None and gencost.shape[0] > 0:
        for g in range(n_gen):
            model = int(gencost[g, 0])
            n = int(gencost[g, 3])
            cost_model_dict[g] = model
            n_cost_dict[g] = n
            if model == 2:  # Polynomial
                for k in range(n):
                    c_native = float(gencost[g, 4 + k])
                    degree = (n - 1) - k
                    scale = (baseMVA**degree) if degree > 0 else 1.0
                    cost_coeff_dict[(g, k)] = c_native / scale
            elif model == 1:  # Piecewise linear
                for k in range(1, n + 1):
                    x_MW = float(gencost[g, 2 + 2 * k])
                    y_cost = float(gencost[g, 3 + 2 * k])
                    pw_x_dict[(g, k)] = x_MW / baseMVA
                    pw_y_dict[(g, k)] = y_cost
    else:
        # Fallback default polynomial if no gencost provided
        warnings.warn(
            "No gencost data found in PYPOWER case. Using default quadratic cost "
            "coefficients: c2=0.01, c1=40.0, c0=0.0 (in p.u. scaling).",
            UserWarning,
            stacklevel=2,
        )
        for g in range(n_gen):
            cost_model_dict[g] = 2
            n_cost_dict[g] = 3
            cost_coeff_dict[(g, 0)] = 0.0  # constant
            cost_coeff_dict[(g, 1)] = 40.0  # linear
            cost_coeff_dict[(g, 2)] = 0.01  # quadratic
    data[None]["cost_model"] = cost_model_dict
    data[None]["n_cost"] = n_cost_dict
    data[None]["cost_coeff"] = cost_coeff_dict
    data[None]["pw_x"] = pw_x_dict
    data[None]["pw_y"] = pw_y_dict

    # Legacy a,b,c scaling for fixed quadratic objective
    a_dict = {}
    b_dict = {}
    c_dict = {}
    if gencost is not None and gencost.shape[0] > 0:
        for g in range(n_gen):
            model = int(gencost[g, 0])
            n = int(gencost[g, 3])
            if model == 2:
                coeffs = [float(gencost[g, 4 + k]) for k in range(n)]
                c2 = 0.0
                c1 = 0.0
                c0 = 0.0
                for pos, c_native in enumerate(coeffs):
                    deg = (n - 1) - pos
                    if deg == 2:
                        c2 = c_native
                    elif deg == 1:
                        c1 = c_native
                    elif deg == 0:
                        c0 = c_native
                a_dict[g] = c2 * (baseMVA**2)
                b_dict[g] = c1 * baseMVA
                c_dict[g] = c0
            else:
                a_dict[g] = 0.0
                b_dict[g] = 40.0 * baseMVA
                c_dict[g] = 0.0
    else:
        # Warning already issued above in the cost_coeff section
        for g in range(n_gen):
            a_dict[g] = 0.01 * (baseMVA**2)
            b_dict[g] = 40.0 * baseMVA
            c_dict[g] = 0.0
    data[None]["a"] = a_dict
    data[None]["b"] = b_dict
    data[None]["c"] = c_dict

    return data, ppc_int


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def initialize_voltage_from_flatstart(instance, ppc_int):
    """Initialize voltage variables e,f from Vm/Va values in internal ppc case.

    Also returns (and does not fix) the slack bus index; slack fixing handled by solve_ac_opf.
    """
    bus = ppc_int["bus"]
    n_bus = bus.shape[0]
    for i in range(n_bus):
        Vm = float(bus[i, 7])
        Va_deg = float(bus[i, 8])
        Va_rad = np.deg2rad(Va_deg)
        instance.e[i].value = Vm * np.cos(Va_rad)
        instance.f[i].value = Vm * np.sin(Va_rad)
    slack_indices = [int(i) for i in range(n_bus) if int(bus[i, 1]) == 3]
    return slack_indices[0] if slack_indices else None


# ---------------------------------------------------------------------------
# Solve wrapper
# ---------------------------------------------------------------------------


def solve_ac_opf(ppc, verbose=True, time_limit=180, mip_gap=0.03, threads=None):
    """Solve AC-OPF for given PYPOWER case using Gurobi.

    Parameters
    ----------
    ppc : dict
        PYPOWER external-numbering case dict.
    verbose : bool
        Print solver/model output.
    time_limit : int
        Solver TimeLimit (seconds).
    mip_gap : float
        Relative MIP gap (NonConvex QCQP termination tolerance).
    threads : int | None
        Threads for Gurobi; defaults to half available logical cores.

    Returns
    -------
    instance : ConcreteModel
    result : SolverResults
    """
    if verbose:
        print("Building AC-OPF abstract model...")
    model = ac_opf_create()

    if verbose:
        print("Preparing data from PYPOWER case...")
    data, ppc_int = prepare_ac_opf_data(ppc)

    if verbose:
        print("Creating model instance...")
    instance = model.create_instance(data=data)  # type: ignore[attr-defined]

    # Voltage + generator warm start
    if verbose:
        print("Initializing voltage and generator variables...")
    slack_bus = initialize_voltage_from_flatstart(instance, ppc_int)
    gen = ppc_int["gen"]
    baseMVA = float(ppc_int["baseMVA"])
    for g in instance.GEN:
        # Clip initial values to respect variable bounds
        pg_init = float(gen[g, 1]) / baseMVA
        qg_init = float(gen[g, 2]) / baseMVA

        # Get bounds from the instance
        pg_lb, pg_ub = instance.PG[g].bounds
        qg_lb, qg_ub = instance.QG[g].bounds

        # Clip to bounds if they exist
        if pg_lb is not None:
            pg_init = max(pg_init, pg_lb)
        if pg_ub is not None:
            pg_init = min(pg_init, pg_ub)
        if qg_lb is not None:
            qg_init = max(qg_init, qg_lb)
        if qg_ub is not None:
            qg_init = min(qg_init, qg_ub)

        instance.PG[g].value = pg_init
        instance.QG[g].value = qg_init

    # Fix slack bus voltage (remove rotational symmetry)
    if slack_bus is not None:
        Vm = float(ppc_int["bus"][slack_bus, 7])
        Va_deg = float(ppc_int["bus"][slack_bus, 8])
        Va_rad = np.deg2rad(Va_deg)
        instance.e[slack_bus].fix(Vm * np.cos(Va_rad))
        instance.f[slack_bus].fix(Vm * np.sin(Va_rad))
        if verbose:
            print(f"Fixed slack bus {slack_bus} voltage (Cartesian).")

    # Solver configuration
    if threads is None:
        threads = max(1, multiprocessing.cpu_count() // 2)
    if verbose:
        print(
            f"Configuring Gurobi solver: threads={threads}, time_limit={time_limit}, mip_gap={mip_gap}"
        )
    solver = pyo.SolverFactory("gurobi")
    solver.options["Threads"] = threads
    solver.options["NonConvex"] = 2
    solver.options["TimeLimit"] = int(time_limit)
    solver.options["MIPGap"] = float(mip_gap)

    if verbose:
        print("Solving AC-OPF...")
        print("-" * 60)
    try:
        result = solver.solve(instance, tee=verbose)
    except (
        ValueError,
        Exception,
    ) as e:  # Some environments raise early errors before solve info
        if verbose:
            print(f"Solver raised exception (likely preprocessing failure): {e}")

        # Create a minimal result-like object for error reporting
        class ErrorResult:
            class Solver:
                status = "error"
                termination_condition = "error"

            solver = Solver()

        result = ErrorResult()

    if verbose:
        print("-" * 60)
        print(f"Solver Status: {result.solver.status}")
        print(f"Termination Condition: {result.solver.termination_condition}")
        if result.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.maxTimeLimit,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
        ):
            try:
                obj_val = pyo.value(instance.TotalCost)
                print(f"Objective Value: {obj_val:.2f}")
            except Exception:
                print("Could not retrieve objective value")
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Warning: Solution may not be globally optimal!")
    return instance, result


__all__ = [
    "prepare_ac_opf_data",
    "initialize_voltage_from_flatstart",
    "solve_ac_opf",
]
