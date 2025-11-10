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
    - Objective:   sum_{g in GEN} (a_g * PG_g^2 + b_g * PG_g + c_g)
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

    # Cost parameters (for flexible gencost support)
    m.cost_model = pyo.Param(m.GEN, within=pyo.Integers, default=2)
    m.n_cost = pyo.Param(m.GEN, within=pyo.PositiveIntegers, default=3)
    m.cost_coeff = pyo.Param(m.GEN, pyo.Any, within=pyo.Reals, default=0.0)
    m.pw_x = pyo.Param(m.GEN, pyo.Any, within=pyo.Reals, default=0.0)
    m.pw_y = pyo.Param(m.GEN, pyo.Any, within=pyo.Reals, default=0.0)

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

    # ===== Objective (flexible by gencost) =====
    # We support:
    #   - model 2: polynomial  f_g(PG) = sum_{k=0}^{n_g-1} cost_coeff[g,k] * PG^k
    #   - model 1: piecewise linear with points (pw_x[g,k], pw_y[g,k]), k = 1..n_g
    #
    # Required Params on the instance (all p.u.-consistent):
    #   cost_model[g] in {1,2}; n_cost[g] >= 1
    #   If model=2: cost_coeff[g,k] for k in 0..n_cost[g]-1  (c0 is constant term)
    #   If model=1: pw_x[g,k], pw_y[g,k] for k in 1..n_cost[g] with pw_x increasing
    #
    # Implementation detail:
    #   - Introduce FCOST[g] and minimize sum_g FCOST[g]
    #   - For model=2: enforce FCOST[g] == sum_k cost_coeff[g,k] * PG[g]**k
    #   - For model=1: use Pyomo Piecewise to link FCOST[g] and PG[g]

    # Per-generator cost variable
    m.FCOST = pyo.Var(m.GEN)

    def _cost_link_action(model, g):
        # Called at instance time, after Params are populated
        cm_raw = pyo.value(model.cost_model[g])
        cm = 0 if cm_raw is None else int(cm_raw)

        if cm == 1:
            n_raw = pyo.value(model.n_cost[g])
            n = 0 if n_raw is None else int(n_raw)
            if n < 2:
                raise ValueError(
                    f"Piecewise needs at least 2 points for generator {g}, got n={n}."
                )

            bp = [
                _as_float(model.pw_x[g, k], f"pw_x[{g},{k}]") for k in range(1, n + 1)
            ]
            fv = [
                _as_float(model.pw_y[g, k], f"pw_y[{g},{k}]") for k in range(1, n + 1)
            ]

            # strict monotonicity after coercion (now no None can slip through)
            if any(bp[i] >= bp[i + 1] for i in range(len(bp) - 1)):
                raise ValueError(
                    f"Breakpoints must be strictly increasing for generator {g}."
                )

            # now create the Piecewise safely
            setattr(
                model,
                f"PWCost_{g}",
                pyo.Piecewise(
                    model.FCOST[g],
                    model.PG[g],
                    pw_pts=bp,
                    f_rule=fv,
                    pw_constr_type="EQ",
                    pw_repn="SOS2",
                ),
            )

        elif cm == 2:
            n_raw = pyo.value(model.n_cost[g])
            n = 0 if n_raw is None else int(n_raw)
            if n < 1:
                raise ValueError(f"n_cost must be >=1 for generator {g}.")
            expr = pyo.quicksum(
                model.cost_coeff[g, k] * (model.PG[g] ** k) for k in range(n)
            )
            setattr(
                model, f"PolyCostLink_{g}", pyo.Constraint(expr=model.FCOST[g] == expr)
            )

        else:
            raise ValueError(f"Unknown cost_model {cm} for generator {g}.")

    m.CostLinkAction = pyo.BuildAction(m.GEN, rule=_cost_link_action)

    # Final objective: minimize total generation cost (use rule to defer iteration)
    def _obj_rule(mm):
        return pyo.quicksum(mm.FCOST[g] for g in mm.GEN)

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
