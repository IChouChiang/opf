# test_case57.py - Load case57 and solve AC-OPF
"""
pyright: reportAttributeAccessIssue=false, reportIndexIssue=false, reportGeneralTypeIssues=false
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pyomo.environ as pyo
from pypower.api import case57
from helpers_ac_opf import solve_ac_opf


if __name__ == "__main__":
    print("=" * 60)
    print("AC-OPF Test: IEEE 57-Bus System")
    print("=" * 60)
    print()

    # Load case57 from PYPOWER
    print("Loading case57 from PYPOWER...")
    ppc = case57()

    n_bus = ppc["bus"].shape[0]
    n_gen = ppc["gen"].shape[0]
    n_branch = ppc["branch"].shape[0]

    print(f"System size: {n_bus} buses, {n_gen} generators, {n_branch} branches")
    print()

    # Solve AC-OPF
    instance, result = solve_ac_opf(ppc, verbose=True)

    # Display results if we have a solution
    has_solution = (
        result.solver.termination_condition
        in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.maxTimeLimit,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
        )
        and hasattr(instance, "PG")
        and any(instance.PG[g].value is not None for g in instance.GEN)
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
            bus_idx_val = pyo.value(instance.GEN_BUS[g], exception=False)
            bus_idx_internal = int(bus_idx_val) if bus_idx_val is not None else 0
            # Display external, 1-based bus numbering as in PYPOWER cases
            bus_label = bus_idx_internal + 1
            gen_label = int(g) + 1
            print(f"  Gen {gen_label} (Bus {bus_label}): {pg:.4f}")

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
            # Display 1-based external bus numbering
            print(f"  Bus {int(i)+1}: {Vm:.4f} p.u.")
            shown += 1

    print("\nDone!")
