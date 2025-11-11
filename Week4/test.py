# test.py - Load case39 and solve AC-OPF
# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false, reportGeneralTypeIssues=false

import numpy as np
import pyomo.environ as pyo
from pypower.api import case39
import multiprocessing

# Import the AC-OPF model builder
from ac_opf_create import ac_opf_create


def prepare_ac_opf_data(ppc):
    """
    Prepare data dict for AC-OPF AbstractModel.create_instance().
    
    Converts PYPOWER case to format expected by ac_opf_create model.
    Uses PYPOWER's makeYbus for admittance matrix.
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
    
    # GEN_BUS mapping (which bus each generator is on)
    gen_bus_map = {}
    for g in range(n_gen):
        bus_id = int(gen[g, 0])  # MATPOWER col 1 (0-indexed)
        gen_bus_map[g] = bus_id
    data[None]["GEN_BUS"] = gen_bus_map
    
    # Bus parameters (demand and voltage limits)
    PD_dict = {}
    QD_dict = {}
    Vmin_dict = {}
    Vmax_dict = {}
    for i in range(n_bus):
        PD_dict[i] = float(bus[i, 2]) / baseMVA  # Pd in p.u.
        QD_dict[i] = float(bus[i, 3]) / baseMVA  # Qd in p.u.
        Vmin_dict[i] = float(bus[i, 12])  # Vmin (already p.u.)
        Vmax_dict[i] = float(bus[i, 11])  # Vmax (already p.u.)
    
    data[None]["PD"] = PD_dict
    data[None]["QD"] = QD_dict
    data[None]["Vmin"] = Vmin_dict
    data[None]["Vmax"] = Vmax_dict
    
    # Generator parameters
    PGmin_dict = {}
    PGmax_dict = {}
    QGmin_dict = {}
    QGmax_dict = {}
    for g in range(n_gen):
        PGmin_dict[g] = float(gen[g, 9]) / baseMVA  # Pmin in p.u.
        PGmax_dict[g] = float(gen[g, 8]) / baseMVA  # Pmax in p.u.
        QGmin_dict[g] = float(gen[g, 4]) / baseMVA  # Qmin in p.u.
        QGmax_dict[g] = float(gen[g, 3]) / baseMVA  # Qmax in p.u.
    
    data[None]["PGmin"] = PGmin_dict
    data[None]["PGmax"] = PGmax_dict
    data[None]["QGmin"] = QGmin_dict
    data[None]["QGmax"] = QGmax_dict
    
    # Admittance matrix G, B (dense format for Pyomo Param[BUS,BUS])
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
                # Coefficients: c_{n-1}, ..., c_1, c_0
                # Scale to p.u.: c_k_pu = c_k / baseMVA^k
                for k in range(n):
                    c_native = float(gencost[g, 4 + k])
                    degree = (n - 1) - k
                    scale = (baseMVA ** degree) if degree > 0 else 1.0
                    cost_coeff_dict[(g, k)] = c_native / scale
                    
            elif model == 1:  # Piecewise linear
                # Points: x1, y1, ..., xn, yn
                for k in range(1, n + 1):
                    x_MW = float(gencost[g, 2 + 2*k])
                    y_cost = float(gencost[g, 3 + 2*k])
                    pw_x_dict[(g, k)] = x_MW / baseMVA  # p.u.
                    pw_y_dict[(g, k)] = y_cost  # currency
    else:
        # No gencost - use default quadratic costs
        for g in range(n_gen):
            cost_model_dict[g] = 2
            n_cost_dict[g] = 3
            # Default: 0.01*P^2 + 40*P + 0
            cost_coeff_dict[(g, 0)] = 0.0  # constant
            cost_coeff_dict[(g, 1)] = 40.0  # linear
            cost_coeff_dict[(g, 2)] = 0.01  # quadratic (p.u. scaled)
    
    data[None]["cost_model"] = cost_model_dict
    data[None]["n_cost"] = n_cost_dict
    data[None]["cost_coeff"] = cost_coeff_dict
    data[None]["pw_x"] = pw_x_dict
    data[None]["pw_y"] = pw_y_dict
    
    # Legacy a, b, c parameters (for backward compatibility, extract from polynomial)
    a_dict = {}
    b_dict = {}
    c_dict = {}
    for g in range(n_gen):
        if cost_model_dict[g] == 2 and n_cost_dict[g] >= 3:
            # Assume quadratic: a*P^2 + b*P + c
            c_dict[g] = cost_coeff_dict.get((g, 0), 0.0)
            b_dict[g] = cost_coeff_dict.get((g, 1), 0.0)
            a_dict[g] = cost_coeff_dict.get((g, 2), 0.0)
        else:
            a_dict[g] = 0.0
            b_dict[g] = 40.0
            c_dict[g] = 0.0
    
    data[None]["a"] = a_dict
    data[None]["b"] = b_dict
    data[None]["c"] = c_dict
    
    return data, ppc_int


def initialize_voltage_from_flatstart(instance, ppc_int):
    """
    Initialize voltage variables e, f from flat start or existing ppc solution.
    """
    bus = ppc_int["bus"]
    n_bus = bus.shape[0]
    
    for i in range(n_bus):
        Vm = float(bus[i, 7])  # Vm (p.u.)
        Va_deg = float(bus[i, 8])  # Va (degrees)
        Va_rad = np.deg2rad(Va_deg)
        
        instance.e[i].value = Vm * np.cos(Va_rad)
        instance.f[i].value = Vm * np.sin(Va_rad)

    # Return index of slack/reference bus (type == 3) to optionally fix angle
    slack_indices = [int(i) for i in range(n_bus) if int(bus[i, 1]) == 3]
    slack_bus = slack_indices[0] if slack_indices else None
    return slack_bus


def solve_ac_opf(ppc, verbose=True):
    """
    Solve AC-OPF for given PYPOWER case using Gurobi with half CPU cores.
    
    Parameters
    ----------
    ppc : dict
        PYPOWER case dictionary
    verbose : bool
        Print solver output
        
    Returns
    -------
    instance : Pyomo ConcreteModel
        Solved model instance
    result : SolverResults
        Solver results object
    """
    # Create abstract model
    if verbose:
        print("Building AC-OPF abstract model...")
    model = ac_opf_create()
    
    # Prepare data
    if verbose:
        print("Preparing data from PYPOWER case...")
    data, ppc_int = prepare_ac_opf_data(ppc)
    
    # Create instance
    if verbose:
        print("Creating model instance...")
    # create_instance with keyword arg (safer for Pyomo version compatibility)
    instance = model.create_instance(data=data)
    
    # Initialize voltages
    if verbose:
        print("Initializing voltage variables...")
    initialize_voltage_from_flatstart(instance, ppc_int)
    # Also initialize PG/QG from case data to help feasibility
    gen = ppc_int["gen"]
    baseMVA = float(ppc_int["baseMVA"])
    for g in instance.GEN:
        pg0 = float(gen[g, 1]) / baseMVA
        qg0 = float(gen[g, 2]) / baseMVA
        instance.PG[g].value = pg0
        instance.QG[g].value = qg0
    # Fix slack bus voltage (angle & magnitude) to reference values to remove rotational symmetry
    slack_bus = None
    bus_int = ppc_int["bus"]
    for i in range(bus_int.shape[0]):
        if int(bus_int[i, 1]) == 3:
            slack_bus = int(i)
            Vm = float(bus_int[i, 7])
            Va_deg = float(bus_int[i, 8])
            Va_rad = np.deg2rad(Va_deg)
            instance.e[slack_bus].fix(Vm * np.cos(Va_rad))
            instance.f[slack_bus].fix(Vm * np.sin(Va_rad))
            break
    
    # Setup Gurobi solver with half CPU cores
    n_threads = max(1, multiprocessing.cpu_count() // 2)
    if verbose:
        print(f"Setting up Gurobi solver with {n_threads} threads (half of available cores)...")
    
    solver = pyo.SolverFactory('gurobi')
    solver.options['Threads'] = n_threads
    solver.options['NonConvex'] = 2  # Required for quadratic constraints
    solver.options['TimeLimit'] = 180  # 3 minutes max
    solver.options['MIPGap'] = 0.03  # 3% optimality gap tolerance
    
    # Solve
    if verbose:
        print("Solving AC-OPF...")
        print("-" * 60)
    
    try:
        result = solver.solve(instance, tee=verbose)
    except ValueError as e:
        if verbose:
            print("Solver raised ValueError (likely no feasible solution yet):", e)
        # Create a dummy results object
        result = pyo.SolverResults()
        result.solver.status = pyo.SolverStatus.aborted
        result.solver.termination_condition = pyo.TerminationCondition.error
    
    if verbose:
        print("-" * 60)
        print(f"Solver Status: {result.solver.status}")
        print(f"Termination Condition: {result.solver.termination_condition}")

        # Check if we have a solution (optimal or feasible)
        has_solution = (
            result.solver.termination_condition == pyo.TerminationCondition.optimal
            or result.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit
            or (hasattr(result.solver, 'message') and 'solution' in str(result.solver.message).lower())
        )

        if has_solution:
            try:
                obj_val = pyo.value(instance.TotalCost)
                print(f"Objective Value: {obj_val:.2f}")
            except Exception:
                print("Could not retrieve objective value")

        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Warning: Solution may not be globally optimal!")
    
    return instance, result


if __name__ == "__main__":
    print("=" * 60)
    print("AC-OPF Test: IEEE 39-Bus System")
    print("=" * 60)
    print()
    
    # Load case39 from PYPOWER
    print("Loading case39 from PYPOWER...")
    ppc = case39()
    
    n_bus = ppc["bus"].shape[0]
    n_gen = ppc["gen"].shape[0]
    n_branch = ppc["branch"].shape[0]
    
    print(f"System size: {n_bus} buses, {n_gen} generators, {n_branch} branches")
    print()
    
    # Solve AC-OPF
    instance, result = solve_ac_opf(ppc, verbose=True)

    # Display results if we have a solution
    has_solution = (
        result.solver.termination_condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.maxTimeLimit,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible
        ) and hasattr(instance, 'PG') and any(instance.PG[g].value is not None for g in instance.GEN)
    )

    if has_solution:
        print()
        print("=" * 60)
        print("Solution Summary")
        print("=" * 60)
        
        # Generator outputs
        print("\nGenerator Active Power Output (p.u.):")
        total_pg = 0.0
        for g in instance.GEN:
            pg_raw = pyo.value(instance.PG[g], exception=False)
            pg = float(pg_raw) if pg_raw is not None else 0.0
            total_pg += pg
            bus_idx = pyo.value(instance.GEN_BUS[g])
            print(f"  Gen {g} (Bus {bus_idx}): {pg:.4f}")
        
        print(f"\nTotal Generation: {total_pg:.4f} p.u.")
        
        # Total demand
        total_pd = 0.0
        for i in instance.BUS:
            val = pyo.value(instance.PD[i])
            if val is not None:
                total_pd += float(val)
        if total_pd is None:
            total_pd = 0.0
        print(f"Total Demand: {total_pd:.4f} p.u.")
        print(f"Losses: {(total_pg - total_pd):.4f} p.u.")
        
        # Voltage magnitudes
        print("\nVoltage Magnitudes (first 10 buses):")
        shown = 0
        for i in instance.BUS:
            if shown >= 10:
                break
            e_val = pyo.value(instance.e[i])
            f_val = pyo.value(instance.f[i])
            if e_val is None or f_val is None:
                continue
            Vm = float(np.sqrt(e_val**2 + f_val**2))
            print(f"  Bus {i}: {Vm:.4f} p.u.")
            shown += 1
    
    print("\nDone!")
