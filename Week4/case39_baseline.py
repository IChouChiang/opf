# case39_baseline.py
# Run PYPOWER OPF on IEEE 39-bus system and print key metrics.
# After running once, copy selected outputs into the comment block below for reference.

from __future__ import annotations
import numpy as np
from pypower.api import case39, runopf


def summarize_results(res: dict) -> None:
    baseMVA = float(res["baseMVA"])  # typically 100.0
    bus = res["bus"]
    gen = res["gen"]
    branch = res["branch"]

    nbus = bus.shape[0]
    ngen = gen.shape[0]
    nbranch = branch.shape[0]

    # Column indices (MATPOWER):
    BUS_PD, BUS_QD, BUS_VM, BUS_VA, BUS_VMAX, BUS_VMIN = 2, 3, 7, 8, 11, 12
    GEN_PG, GEN_QG = 1, 2
    BR_RATEA, BR_PF, BR_QF, BR_PT, BR_QT = 5, 13, 14, 15, 16  # PF/QF/PT/QT appended after solve

    total_pd = float(bus[:, BUS_PD].sum()) / baseMVA  # p.u.
    total_qd = float(bus[:, BUS_QD].sum()) / baseMVA  # p.u.

    total_pg = float(gen[:, GEN_PG].sum()) / baseMVA
    total_qg = float(gen[:, GEN_QG].sum()) / baseMVA

    losses_pu = total_pg - total_pd

    vm = bus[:, BUS_VM]
    va = bus[:, BUS_VA]

    vm_min = float(vm.min())
    vm_max = float(vm.max())
    va_min = float(va.min())
    va_max = float(va.max())

    # Line MVA loadings (from-end)
    pf = branch[:, BR_PF]
    qf = branch[:, BR_QF]
    rateA = branch[:, BR_RATEA]
    s_from = np.sqrt(pf**2 + qf**2)  # MW/MVAr → MVA on base

    loading = np.zeros_like(s_from)
    mask = rateA > 0
    loading[mask] = 100.0 * s_from[mask] / rateA[mask]
    loading[~mask] = 0.0  # no limit => 0% by convention

    top_idx = np.argsort(loading)[-5:][::-1]

    print("=== PYPOWER OPF: IEEE 39-Bus Baseline ===")
    print(f"Objective cost (res['f']): {res['f']:.6f}")
    print(f"System size: {nbus} buses, {ngen} generators, {nbranch} branches")
    print(f"Total Pd: {total_pd:.4f} p.u., Total Qd: {total_qd:.4f} p.u.")
    print(f"Total Pg: {total_pg:.4f} p.u., Total Qg: {total_qg:.4f} p.u.")
    print(f"Active power losses: {losses_pu:.4f} p.u. ({losses_pu*baseMVA:.2f} MW)")
    print(f"Voltage magnitude range: [{vm_min:.4f}, {vm_max:.4f}] p.u.")
    print(f"Voltage angle range: [{va_min:.2f}, {va_max:.2f}] deg")

    print("\nTop 5 line loadings (from-end):")
    for k in top_idx:
        print(
            f"  line {k:3d}: |S_f|={s_from[k]:6.2f} MVA, rateA={rateA[k]:6.2f} MVA, loading={loading[k]:6.2f}%"
        )

    print("\nGenerator outputs (p.u., on baseMVA):")
    for g in range(ngen):
        print(f"  Gen {g:2d}: Pg={gen[g, GEN_PG]/baseMVA:7.4f}  Qg={gen[g, GEN_QG]/baseMVA:7.4f}")


if __name__ == "__main__":
    # PYPOWER runopf returns only result dict (success flag accessible via
    # res['success']). Older examples sometimes unpack two values.
    res = runopf(case39())
    if not res.get('success', False):
        raise SystemExit("runopf reported failure for case39")
    summarize_results(res)

"""
Snapshot (baseline):
Objective cost: 41864.18 $/hr
Total Pd: 62.5423 p.u.; Total Pg: 62.9783 p.u.; Losses: 0.4360 p.u. (43.60 MW)
Voltage magnitude range: [0.9970, 1.0600] p.u.; angle range: [-16.95, 7.73] deg
Top loaded lines (from-end apparent power / MVA limit %):
    line 13 (6-31): |S_f|=~688.6 MVA, rateA=0 (tx) -> charging, high reactive
    line 19 (10-32): |S_f|=~702.0 MVA
    line 32 (19-33): |S_f|=~650.2 MVA
    line 36 (22-35): |S_f|=~685.5 MVA
    line 45 (29-38): |S_f|=~657.1 MVA
Selected generator Pg (p.u.):
    Bus30: 6.7159, Bus31: 6.4600, Bus32: 6.7116, Bus33: 6.5200, Bus34: 5.0800
    Bus35: 6.6145, Bus36: 5.8000, Bus37: 5.6400, Bus38: 6.5403, Bus39: 6.8959
"""
