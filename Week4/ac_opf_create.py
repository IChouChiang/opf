# ac_opf_create.py
# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false, reportGeneralTypeIssues=false, reportReturnType=false, reportOperatorIssue=false
import pyomo.environ as pyo


def _as_float(val, label):
    v = pyo.value(val, exception=False)
    if v is None:
        raise ValueError(f"{label} is None")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise TypeError(f"{label} is not numeric: {v}")


def ac_opf_create():
    """
    Create an AbstractModel for fixed-topology AC-OPF in Cartesian form.
    - Objective (fixed quadratic):  min Σ_g [ a_g * PG_g^2 + b_g * PG_g + c_g ]
    - Power flow:  uses e_i = V_i cosθ_i, f_i = V_i sinθ_i
      P: (sum_{g at i} PG_g) - PD_i = e_i Σ_j (G_ij e_j - B_ij f_j) + f_i Σ_j (G_ij f_j + B_ij e_j)
      Q: (sum_{g at i} QG_g) - QD_i = f_i Σ_j (G_ij e_j - B_ij f_j) - e_i Σ_j (G_ij f_j + B_ij e_j)
    - Voltage bounds: (Vmin_i)^2 ≤ e_i^2 + f_i^2 ≤ (Vmax_i)^2
    """
    m = pyo.AbstractModel()

    # ===== Sets =====
    m.BUS = pyo.Set(doc="Buses S_B")
    m.GEN = pyo.Set(doc="Generators (units)")

    # Mapping: which bus each generator sits on
    m.GEN_BUS = pyo.Param(
        m.GEN, within=m.BUS, doc="Bus index for each generator (GEN→BUS)"
    )

    # Convenience set: generators located at each bus (computed from GEN_BUS)
    # Use initialize with None and define via BuildAction or constraint rules
    m.GENS_AT_BUS = pyo.Set(m.BUS, within=m.GEN, doc="Generators located at bus i")

    def _init_gens_at_bus(mm):
        # BuildAction to populate GENS_AT_BUS after instance data is loaded
        for i in mm.BUS:
            mm.GENS_AT_BUS[i] = [g for g in mm.GEN if mm.GEN_BUS[g] == i]

    m.InitGensAtBus = pyo.BuildAction(rule=_init_gens_at_bus)

    # ===== Parameters =====
    # Demands (p.u. recommended; ensure consistency with Ybus)
    m.PD = pyo.Param(m.BUS, within=pyo.Reals, default=0.0)
    m.QD = pyo.Param(m.BUS, within=pyo.Reals, default=0.0)

    # Generator cost coefficients (per generator)
    m.a = pyo.Param(m.GEN, within=pyo.NonNegativeReals, default=0.0)
    m.b = pyo.Param(m.GEN, within=pyo.Reals, default=0.0)
    m.c = pyo.Param(m.GEN, within=pyo.Reals, default=0.0)

    # Generator capability limits (per generator)
    m.PGmin = pyo.Param(m.GEN, within=pyo.Reals, default=0.0)
    m.PGmax = pyo.Param(m.GEN, within=pyo.Reals, default=0.0)
    m.QGmin = pyo.Param(m.GEN, within=pyo.Reals, default=0.0)
    m.QGmax = pyo.Param(m.GEN, within=pyo.Reals, default=0.0)

    # Voltage magnitude bounds (per bus)
    m.Vmin = pyo.Param(m.BUS, within=pyo.NonNegativeReals, default=0.95)
    m.Vmax = pyo.Param(m.BUS, within=pyo.NonNegativeReals, default=1.05)

    # Network admittance (dense for simplicity; zeros for non-neighbors)
    m.G = pyo.Param(m.BUS, m.BUS, within=pyo.Reals, default=0.0)
    m.B = pyo.Param(m.BUS, m.BUS, within=pyo.Reals, default=0.0)

    # (Optional/ignored) Flexible-cost placeholders to keep compatibility with
    # any existing data pipelines. These are not used by the objective below.
    m.cost_model = pyo.Param(m.GEN, within=pyo.Integers, default=2, mutable=True)
    m.n_cost = pyo.Param(m.GEN, within=pyo.PositiveIntegers, default=3, mutable=True)
    m.cost_coeff = pyo.Param(m.GEN, pyo.Any, within=pyo.Reals, default=0.0, mutable=True)
    m.pw_x = pyo.Param(m.GEN, pyo.Any, within=pyo.Reals, default=0.0, mutable=True)
    m.pw_y = pyo.Param(m.GEN, pyo.Any, within=pyo.Reals, default=0.0, mutable=True)

    # ===== Variables =====
    def _pg_bounds(mm, g):
        return (mm.PGmin[g], mm.PGmax[g])

    def _qg_bounds(mm, g):
        return (mm.QGmin[g], mm.QGmax[g])

    # Per-generator injections
    m.PG = pyo.Var(m.GEN, bounds=_pg_bounds)
    m.QG = pyo.Var(m.GEN, bounds=_qg_bounds)

    # Cartesian voltage components per bus
    m.e = pyo.Var(m.BUS)  # e_i = V_i cos(theta_i)
    m.f = pyo.Var(m.BUS)  # f_i = V_i sin(theta_i)

    # ===== Objective (fixed quadratic on PG) =====
    # Minimize Σ_g [ a_g * PG_g^2 + b_g * PG_g + c_g ]
    def _obj_rule(mm):
        return pyo.quicksum(mm.a[g] * (mm.PG[g] ** 2) + mm.b[g] * mm.PG[g] + mm.c[g] for g in mm.GEN)

    m.TotalCost = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    # ===== Power balance constraints (per bus) =====
    # Active:  sum_{g at i} PG_g - PD_i = e_i Σ_j (G_ij e_j - B_ij f_j) + f_i Σ_j (G_ij f_j + B_ij e_j)
    def _p_balance_rule(mm, i):
        inj_p = sum(mm.PG[g] for g in mm.GENS_AT_BUS[i]) - mm.PD[i]
        sum_ge = sum(mm.G[i, j] * mm.e[j] - mm.B[i, j] * mm.f[j] for j in mm.BUS)
        sum_gf = sum(mm.G[i, j] * mm.f[j] + mm.B[i, j] * mm.e[j] for j in mm.BUS)
        return inj_p == mm.e[i] * sum_ge + mm.f[i] * sum_gf

    m.PBalance = pyo.Constraint(m.BUS, rule=_p_balance_rule)

    # Reactive: sum_{g at i} QG_g - QD_i = f_i Σ_j (G_ij e_j - B_ij f_j) - e_i Σ_j (G_ij f_j + B_ij e_j)
    def _q_balance_rule(mm, i):
        inj_q = sum(mm.QG[g] for g in mm.GENS_AT_BUS[i]) - mm.QD[i]
        sum_ge = sum(mm.G[i, j] * mm.e[j] - mm.B[i, j] * mm.f[j] for j in mm.BUS)
        sum_gf = sum(mm.G[i, j] * mm.f[j] + mm.B[i, j] * mm.e[j] for j in mm.BUS)
        return inj_q == mm.f[i] * sum_ge - mm.e[i] * sum_gf

    m.QBalance = pyo.Constraint(m.BUS, rule=_q_balance_rule)

    # ===== Voltage magnitude limits =====
    # (Vmin_i)^2 ≤ e_i^2 + f_i^2 ≤ (Vmax_i)^2
    def _vmin_rule(mm, i):
        return (mm.Vmin[i] ** 2) <= (mm.e[i] ** 2 + mm.f[i] ** 2)

    def _vmax_rule(mm, i):
        return (mm.e[i] ** 2 + mm.f[i] ** 2) <= (mm.Vmax[i] ** 2)

    m.VminCon = pyo.Constraint(m.BUS, rule=_vmin_rule)
    m.VmaxCon = pyo.Constraint(m.BUS, rule=_vmax_rule)

    return m
