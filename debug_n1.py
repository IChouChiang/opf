import sys
from pathlib import Path
import numpy as np
import pyomo.environ as pyo

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pypower.api import case6ww, ext2int
from src.helpers_ac_opf import solve_ac_opf


def debug_topology(removed_lines_pairs):
    print(f"\nTesting topology with removed lines: {removed_lines_pairs}")

    # Load base case
    ppc = case6ww()
    ppc = ext2int(ppc)

    # Find branch indices to remove
    branch = ppc["branch"]
    removed_indices = []
    for f, t in removed_lines_pairs:
        f_int, t_int = f - 1, t - 1
        match = np.where(
            ((branch[:, 0] == f_int) & (branch[:, 1] == t_int))
            | ((branch[:, 0] == t_int) & (branch[:, 1] == f_int))
        )[0]
        if len(match) > 0:
            removed_indices.append(match[0])
        else:
            print(f"Warning: Branch {f}-{t} not found")

    if removed_indices:
        print(f"Removing branch indices: {removed_indices}")
        ppc["branch"] = np.delete(ppc["branch"], removed_indices, axis=0)

    # Solve
    try:
        instance, result = solve_ac_opf(ppc, verbose=True)
        print(f"Solver Status: {result.solver.status}")
        print(f"Termination Condition: {result.solver.termination_condition}")

        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("Feasible!")
            # Print some stats
            total_gen = sum(pyo.value(instance.PG[g]) for g in instance.GEN)
            total_load = sum(pyo.value(instance.PD[b]) for b in instance.BUS)
            print(f"Total Gen: {total_gen:.4f}, Total Load: {total_load:.4f}")
        else:
            print("Infeasible or failed.")

    except Exception as e:
        print(f"Exception during solve: {e}")


if __name__ == "__main__":
    # Test the 3 topologies
    topologies = [[(3, 5)], [(1, 5)], [(2, 4)]]

    for topo in topologies:
        debug_topology(topo)
