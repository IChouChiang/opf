import numpy as np
from pypower.api import case6ww, ext2int

# ---------- Topology definition (1-based bus pairs) ----------
# New unseen N-1 contingencies: (3-5), (1-5), (2-4)
# Plus normal operation (empty list) if desired, but user asked for 3 * 400 = 1200 samples
# corresponding to the 3 contingencies.

topology_pairs = [
    [(3, 5)],  # Case 0: Line 3-5 outage
    [(1, 5)],  # Case 1: Line 1-5 outage
    [(2, 4)],  # Case 2: Line 2-4 outage
]


def get_base_case():
    """Load case6ww and convert to internal indexing."""
    ppc = case6ww()
    ppc_int = ext2int(ppc)
    return ppc_int


def find_branch_indices_for_pairs(ppc_int, pairs):
    """Find internal branch indices for a list of (from, to) pairs."""
    branch = ppc_int["branch"]
    indices = []

    for f, t in pairs:
        # Convert 1-based external to 0-based internal indices
        # Note: ext2int might reorder buses, but for case6ww it's usually sequential.
        # We'll assume standard mapping (1->0, 2->1, etc.) for simplicity as case6ww is small.
        # A more robust way would be to map via ppc_int['bus'][:, 0] if we had the original bus numbers preserved there.
        # pypower's ext2int keeps original bus numbers in ppc_int['order']['bus']['e2i'] if needed,
        # but usually for standard cases it's just re-indexing.

        f_int, t_int = f - 1, t - 1

        # Find row where (f, t) or (t, f) matches in the branch matrix (columns 0 and 1)
        match = np.where(
            ((branch[:, 0] == f_int) & (branch[:, 1] == t_int))
            | ((branch[:, 0] == t_int) & (branch[:, 1] == f_int))
        )[0]

        if len(match) > 0:
            indices.append(match[0])
        else:
            print(f"Warning: Branch {f}-{t} not found in case data")

    return indices


def get_topology_config():
    """Returns list of removed branch indices for each topology."""
    ppc_int = get_base_case()

    topology_removed_branches = []
    for pairs in topology_pairs:
        indices = find_branch_indices_for_pairs(ppc_int, pairs)
        topology_removed_branches.append(indices)

    return topology_removed_branches


# ---------- RES Configuration (Same as original) ----------
# Wind at Bus 5, PV at Bus 4, 6
RES_BUSES = {"WIND": [4], "PV": [3, 5]}  # Bus 5 is index 4  # Bus 4 is 3, Bus 6 is 5
