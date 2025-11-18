import sys
from pathlib import Path

# Make gcnn_opf_01 importable and try robust import of the sample config
REPO_ROOT = Path(__file__).parent.parent
GCNN_DIR = REPO_ROOT / "gcnn_opf_01"
sys.path.insert(0, str(GCNN_DIR))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

try:  # noqa: E402
    import sample_config_model_01 as cfg  # type: ignore  # noqa: E402
except Exception:
    import importlib.util  # noqa: E402

    spec = importlib.util.spec_from_file_location(
        "sample_config_model_01", str(GCNN_DIR / "sample_config_model_01.py")
    )
    assert spec and spec.loader
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)  # type: ignore[attr-defined]

from pypower.api import case6ww  # noqa: E402
from pypower.ext2int import ext2int  # noqa: E402


def main():
    # Build internal-numbered case once (base topology)
    ppc = case6ww()
    ppc_int_base = ext2int(ppc)

    branch = ppc_int_base["branch"]
    fbus = branch[:, 0].astype(int)
    tbus = branch[:, 1].astype(int)
    BR_STATUS_COL = 10

    print("=== Topology outage verification (case6ww) ===")

    for topo_id in [1, 2, 3, 4]:
        pairs = cfg.TOPOLOGY_BRANCH_PAIRS_1BASED.get(topo_id, [])
        idx_expected = cfg.find_branch_indices_for_pairs(ppc_int_base, pairs)

        ppc_applied = cfg.apply_topology(ppc_int_base, topo_id)
        status = ppc_applied["branch"][:, BR_STATUS_COL]
        idx_disabled = np.where(status == 0)[0].astype(int).tolist()

        # Extract (fbus,tbus) for readability
        disabled_pairs = [(int(fbus[i]), int(tbus[i])) for i in idx_disabled]

        print(f"Topo {topo_id}: pairs={pairs}")
        print(f"  Expected indices from pairs: {idx_expected}")
        print(f"  Disabled rows in branch[] : {idx_disabled}")
        print(f"  Disabled (fbus,tbus)      : {disabled_pairs}")

        # Simple check: the disabled rows should match the indices mapped from pairs
        if set(idx_expected) != set(idx_disabled):
            print("  [WARN] Mismatch between expected vs. disabled indices!")
        else:
            print("  [OK] Disabled rows match expected mapping.")
        print("")


if __name__ == "__main__":
    main()
