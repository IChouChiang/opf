"""Debug script for Sample 2 AC-OPF solve with full Gurobi output."""

import sys
from pathlib import Path

# Make gcnn_opf_01 and src importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "gcnn_opf_01"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import multiprocessing
from sample_config_model_01 import load_case39_int, apply_topology
from helpers_ac_opf import solve_ac_opf

# Sample 2 demand data (in p.u., from test_sample_generator.py output)
# These values come from the scenario generator with RNG seed=42, sample index=1
SAMPLE2_PD_PU = np.array(
    [
        0.0,
        0.0,
        3.265612,
        4.939059,
        5.058988,
        5.633891,
        2.225883,
        5.177994,
        4.903695,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.654321,
        3.095893,
        3.148976,
        2.675598,
        0.0,
        2.694491,
        5.924089,
        0.0,
        2.618267,
        -10.864159,
        2.182859,
        3.155063,
        -2.651142,
        2.658954,
        2.775399,
        0.0,
        0.0,
        0.569826,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        9.718376,
    ]
)

SAMPLE2_QD_PU = np.array(
    [
        0.0,
        0.0,
        0.724559,
        1.096463,
        1.123318,
        1.250862,
        0.494191,
        1.149776,
        1.088219,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.589403,
        0.687090,
        0.698970,
        0.593963,
        0.0,
        0.598236,
        1.314464,
        0.0,
        0.581153,
        -2.411812,
        0.484634,
        0.700123,
        -0.588480,
        0.590244,
        0.616199,
        0.0,
        0.0,
        0.126518,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.157301,
    ]
)


def main():
    print("=== Debug: Sample 2 AC-OPF with Full Gurobi Output ===\n")

    # Get CPU count
    n_cpus = multiprocessing.cpu_count()
    print(f"System CPUs: {n_cpus}\n")

    # Load case39
    ppc_int, baseMVA, bus, gen, branch, N_BUS, N_GEN, N_BRANCH = load_case39_int()
    print(f"Loaded case39: {N_BUS} buses, {N_GEN} gens, {N_BRANCH} branches")

    # Build scenario with Sample 2 data
    ppc_scenario = apply_topology(ppc_int, 0)  # base topology

    # Set demands (convert p.u. back to MW/MVAr)
    ppc_scenario["bus"][:, 2] = SAMPLE2_PD_PU * baseMVA  # PD in MW
    ppc_scenario["bus"][:, 3] = SAMPLE2_QD_PU * baseMVA  # QD in MVAr

    print(f"\nSample 2 characteristics:")
    print(
        f"  Total PD: {SAMPLE2_PD_PU.sum():.3f} p.u. ({SAMPLE2_PD_PU.sum()*baseMVA:.2f} MW)"
    )
    print(f"  Min PD: {SAMPLE2_PD_PU.min():.3f} p.u. (bus {SAMPLE2_PD_PU.argmin()})")
    print(f"  Negative PD buses: {(SAMPLE2_PD_PU < 0).sum()}")
    print(f"  Negative PD sum: {SAMPLE2_PD_PU[SAMPLE2_PD_PU < 0].sum():.3f} p.u.")

    print("\n" + "=" * 80)
    print("Starting AC-OPF solve with VERBOSE=True, full Gurobi output...")
    print("=" * 80 + "\n")

    try:
        instance, result = solve_ac_opf(
            ppc_scenario,
            verbose=True,  # Enable full output
            time_limit=300,  # 5 minutes
            mip_gap=0.03,
            threads=n_cpus,
        )

        print("\n" + "=" * 80)
        print("SOLVE COMPLETED")
        print("=" * 80)
        print(f"Solver status: {result.solver.status}")
        print(f"Termination condition: {result.solver.termination_condition}")

        if str(result.solver.termination_condition) == "optimal":
            from pyomo.environ import value

            total_gen = sum(value(instance.PG[g]) for g in instance.GEN)
            total_cost = value(instance.TotalCost)
            print(f"\nObjective: {total_cost:.2f} $/hr")
            print(f"Total generation: {total_gen:.3f} p.u.")
            print(f"Losses: {total_gen - SAMPLE2_PD_PU.sum():.3f} p.u.")
        else:
            print("\nSolution NOT optimal - check Gurobi output above for details")

    except Exception as e:
        print(f"\nEXCEPTION during solve: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
