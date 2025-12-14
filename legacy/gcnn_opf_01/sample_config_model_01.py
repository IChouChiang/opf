import numpy as np
import torch

from pypower.api import case6ww, ext2int, makeYbus  # adjust import path if needed

# ---------- Topology definition (1-based bus pairs) ----------

# We treat topo_id = 0 as the base (no outage).
# The tuples are (from_bus, to_bus) in the case6ww bus numbering (1-based external).
TOPOLOGY_BRANCH_PAIRS_1BASED = {
    0: [],  # base case: no line outage
    1: [(5, 2)],
    2: [(1, 2)],
    3: [(2, 3)],
    4: [(5, 6)],
}

# ---------- RES bus configuration ----------
# These are external bus IDs (bus_i) from case6ww.py.
# After ext2int(), map to internal 0-based indices using get_res_bus_indices()
RES_BUS_WIND_EXTERNAL = [5]  # External bus_i (bus 5 has wind)
RES_BUS_PV_EXTERNAL = [4, 6]  # External bus_i (buses 4 and 6 have PV)


def get_res_bus_indices(ppc_int):
    """
    Map external bus IDs to internal 0-based indices after ext2int().

    For case6ww, buses are numbered 1-6 consecutively, so after ext2int
    they map to 0-5 in the same order (bus_i - 1 = internal_index).

    Args:
        ppc_int: internal-indexed case (after ext2int)

    Returns:
        wind_indices: list[int] of 0-based internal indices for wind buses
        pv_indices: list[int] of 0-based internal indices for PV buses
    """
    # For case39, buses 1-39 map directly to indices 0-38 after ext2int
    # because the original bus numbering is already consecutive.
    # The 'order' field in ppc_int contains the e2i mapping if needed.

    if "order" in ppc_int and "bus" in ppc_int["order"]:
        # Use the explicit external-to-internal mapping
        e2i = ppc_int["order"]["bus"]["e2i"]
        wind_indices = [int(e2i[bus_id]) for bus_id in RES_BUS_WIND_EXTERNAL]
        pv_indices = [int(e2i[bus_id]) for bus_id in RES_BUS_PV_EXTERNAL]
    else:
        # Fallback: case6ww has consecutive bus numbering 1-6
        wind_indices = [b - 1 for b in RES_BUS_WIND_EXTERNAL]
        pv_indices = [b - 1 for b in RES_BUS_PV_EXTERNAL]

    return wind_indices, pv_indices


# ---------- Load fluctuation parameters ----------
SIGMA_REL_LOAD = 0.1  # relative std deviation for load fluctuation

# ---------- Wind generation parameters (Weibull distribution + power curve) ----------
LAM_WIND = 5.089  # Weibull scale parameter (m/s)
K_WIND = 2.016  # Weibull shape parameter

# Wind turbine power curve parameters
V_CUT_IN = 4.0  # cut-in wind speed (m/s)
V_RATED = 12.0  # rated wind speed (m/s)
V_CUT_OUT = 25.0  # cut-out wind speed (m/s)

# ---------- PV generation parameters (Beta distribution + power curve) ----------
ALPHA_PV = 2.06  # Beta distribution alpha
BETA_PV = 2.5  # Beta distribution beta
G_STC = 1000.0  # standard test condition irradiance (W/m²)


def load_case6ww_int():
    """
    Load case6ww test system (6-bus Wood & Wollenberg) and convert to internal indexing
    using PYPOWER's ext2int.

    Returns:
        ppc_int : dict
        baseMVA : float
        bus     : np.ndarray [N_BUS, ...]
        gen     : np.ndarray [N_GEN, ...]
        branch  : np.ndarray [N_BRANCH, ...]
        N_BUS, N_GEN, N_BRANCH : ints
    """
    ppc = case6ww()
    ppc_int = ext2int(ppc)

    baseMVA = float(ppc_int["baseMVA"])
    bus = ppc_int["bus"]
    gen = ppc_int["gen"]
    branch = ppc_int["branch"]

    N_BUS = bus.shape[0]
    N_GEN = gen.shape[0]
    N_BRANCH = branch.shape[0]

    return ppc_int, baseMVA, bus, gen, branch, N_BUS, N_GEN, N_BRANCH


def find_branch_indices_for_pairs(ppc_int, pairs_1based):
    """
    Given a list of (from_bus, to_bus) pairs (1-based external bus numbers),
    find branch row indices in ppc_int["branch"] that connect those buses.

    Converts 1-based external bus IDs to 0-based internal indices using
    the e2i mapping from ppc_int["order"]["bus"]["e2i"].

    Returns:
        idx_list: list[int] of branch row indices (0-based)
    """
    branch = ppc_int["branch"]
    fbus = branch[:, 0].astype(int)
    tbus = branch[:, 1].astype(int)

    # Get external-to-internal bus mapping
    if "order" in ppc_int and "bus" in ppc_int["order"]:
        e2i = ppc_int["order"]["bus"]["e2i"]
    else:
        # Fallback: assume consecutive numbering (bus_i -> i-1)
        e2i = {i: i - 1 for i in range(1, len(fbus) + 10)}

    idx_list = []

    for fb_ext, tb_ext in pairs_1based:
        # Convert external 1-based to internal 0-based
        fb_int = int(e2i[fb_ext])
        tb_int = int(e2i[tb_ext])

        # find rows where (fbus==fb_int and tbus==tb_int) or (fbus==tb_int and tbus==fb_int)
        mask = ((fbus == fb_int) & (tbus == tb_int)) | (
            (fbus == tb_int) & (tbus == fb_int)
        )
        rows = np.where(mask)[0]

        # In case6ww, some branch pairs may have parallel lines.
        # We keep all matching rows.
        for r in rows:
            idx_list.append(int(r))

    return idx_list


def apply_topology(ppc_int_base, topo_id):
    """
    Return a *copy* of ppc_int_base with the given topology applied.

    topo_id in {0,1,2,3,4}, where:
        - 0: base case (no outage)
        - 1..4: outages defined in TOPOLOGY_BRANCH_PAIRS_1BASED

    For topo_id > 0, we set branch[branch_idx, BR_STATUS] = 0
    for each affected branch row.
    """
    ppc_int = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in ppc_int_base.items()
    }

    pairs = TOPOLOGY_BRANCH_PAIRS_1BASED.get(topo_id, [])
    if not pairs:
        # base topology: nothing to do
        return ppc_int

    branch_idx_list = find_branch_indices_for_pairs(ppc_int, pairs)

    # MATPOWER/PYPOWER: branch[:, 10] is BR_STATUS (1 = in service, 0 = out)
    BR_STATUS_COL = 10  # confirm this is correct for your PYPOWER version

    for idx in branch_idx_list:
        ppc_int["branch"][idx, BR_STATUS_COL] = 0.0

    return ppc_int


def build_G_B_operators(ppc_int):
    """
    From a given internal-indexed case (with topology already applied),
    build the physics operators:

        G, B:          [N_BUS, N_BUS]
        g_diag, b_diag: [N_BUS]
        g_ndiag, b_ndiag: [N_BUS, N_BUS]

    Returns:
        G, B, g_diag, b_diag, g_ndiag, b_ndiag (all as torch.FloatTensor)
    """
    baseMVA = ppc_int["baseMVA"]
    bus = ppc_int["bus"]
    branch = ppc_int["branch"]

    # makeYbus returns complex Ybus
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    Ybus_np = np.asarray(Ybus.todense())  # Convert matrix to standard array
    G_np = np.real(Ybus_np)
    B_np = np.imag(Ybus_np)

    # Diagonals
    g_diag_np = np.diag(G_np)
    b_diag_np = np.diag(B_np)

    # Off-diagonals = full matrix minus diag
    g_ndiag_np = G_np - np.diag(g_diag_np)
    b_ndiag_np = B_np - np.diag(b_diag_np)

    # Convert to torch tensors (float32)
    G = torch.tensor(G_np, dtype=torch.float32)
    B = torch.tensor(B_np, dtype=torch.float32)
    g_diag = torch.tensor(g_diag_np, dtype=torch.float32)
    b_diag = torch.tensor(b_diag_np, dtype=torch.float32)
    g_ndiag = torch.tensor(g_ndiag_np, dtype=torch.float32)
    b_ndiag = torch.tensor(b_ndiag_np, dtype=torch.float32)

    return G, B, g_diag, b_diag, g_ndiag, b_ndiag


def get_operators_for_topology(topo_id):
    """
    Convenience wrapper:
        1) load case6ww
        2) apply topology topo_id
        3) compute G/B and z-operators

    Returns:
        ppc_int, G, B, g_diag, b_diag, g_ndiag, b_ndiag
    """
    ppc_int_base, baseMVA, bus, gen, branch, N_BUS, N_GEN, N_BRANCH = load_case6ww_int()
    ppc_int_topo = apply_topology(ppc_int_base, topo_id)
    G, B, g_diag, b_diag, g_ndiag, b_ndiag = build_G_B_operators(ppc_int_topo)

    return ppc_int_topo, G, B, g_diag, b_diag, g_ndiag, b_ndiag


def extract_gen_limits(ppc_int):
    """
    Extract generator limits from ppc_int['gen'] for feature construction.

    PYPOWER/MATPOWER gen columns (0-indexed):
        0: GEN_BUS    - bus number (internal)
        3: QMAX       - reactive power max (MVAr)
        4: QMIN       - reactive power min (MVAr)
        8: PMAX       - active power max (MW)
        9: PMIN       - active power min (MW)

    Returns:
        gen_bus_indices : np.ndarray [N_GEN], internal bus indices (0-based)
        PG_min, PG_max  : torch.Tensor [N_GEN], p.u.
        QG_min, QG_max  : torch.Tensor [N_GEN], p.u.
        gen_mask        : torch.Tensor [N_BUS], bool mask (True at generator buses)
    """
    baseMVA = float(ppc_int["baseMVA"])
    gen = ppc_int["gen"]
    N_BUS = ppc_int["bus"].shape[0]
    N_GEN = gen.shape[0]

    # Extract bus indices and limits
    gen_bus_indices = gen[:, 0].astype(int)  # Internal bus indices

    PG_min = torch.tensor(gen[:, 9] / baseMVA, dtype=torch.float32)  # PMIN → p.u.
    PG_max = torch.tensor(gen[:, 8] / baseMVA, dtype=torch.float32)  # PMAX → p.u.
    QG_min = torch.tensor(gen[:, 4] / baseMVA, dtype=torch.float32)  # QMIN → p.u.
    QG_max = torch.tensor(gen[:, 3] / baseMVA, dtype=torch.float32)  # QMAX → p.u.

    # Create generator mask [N_BUS] (True at generator buses)
    gen_mask = torch.zeros(N_BUS, dtype=torch.bool)
    gen_mask[torch.tensor(gen_bus_indices, dtype=torch.long)] = True

    return gen_bus_indices, PG_min, PG_max, QG_min, QG_max, gen_mask
