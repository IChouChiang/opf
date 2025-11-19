import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# Helper: build generator-to-bus incidence matrix M (A_g2b)
# -------------------------------------------------------------------


def build_A_g2b(N_BUS, GEN_BUS):
    """
    GEN_BUS: 1D iterable of length N_GEN, internal bus indices in [0, N_BUS-1].

    Returns:
        A_g2b: [N_BUS, N_GEN], where
               A_g2b[i, g] = 1 if generator g is at bus i, else 0.
    """
    A = torch.zeros(N_BUS, len(GEN_BUS), dtype=torch.float32)
    for g, bus in enumerate(GEN_BUS):
        A[bus, g] = 1.0
    return A


# -------------------------------------------------------------------
# Physics function f_{PG}(V_out) from Eq. (8) rearranged
# -------------------------------------------------------------------


def f_pg_from_v(v_out, pd, G, B):
    r"""
    Compute f_{PG}(V_out) ∈ ℝ^{N_BUS} using the AC nodal active-power
    equation in Cartesian form.

    Paper Eq. (8) rearranged:

        PG_i - PD_i =
            e_i Σ_j [ z(G)_{ij} e_j - z(B)_{ij} f_j ]
          + f_i Σ_j [ z(G)_{ij} f_j + z(B)_{ij} e_j ]

    ⇒

        PG_i =
            PD_i
          + e_i Σ_j [ z(G)_{ij} e_j - z(B)_{ij} f_j ]
          + f_i Σ_j [ z(G)_{ij} f_j + z(B)_{ij} e_j ]

    Vector form:

        let e, f, PD ∈ ℝ^N, G = z(G), B = z(B)
        Ge = G e, Gf = G f, Be = B e, Bf = B f

        f_PG(V_out) = PD
                      + e ⊙ (Ge - Bf)
                      + f ⊙ (Gf + Be)

    Args:
        v_out: [N_BUS, 2]  -> (e, f) predicted voltages
        pd   : [N_BUS]     -> active demand (including RES as negative load)
        G,B  : [N_BUS, N_BUS] -> z(G), z(B) for current topology

    Returns:
        PG_from_V: [N_BUS] = f_{PG}(V_out)
    """
    e = v_out[:, 0]
    f = v_out[:, 1]

    Ge = G @ e
    Gf = G @ f
    Be = B @ e
    Bf = B @ f

    PG_from_V = pd + e * (Ge - Bf) + f * (Gf + Be)
    return PG_from_V


# -------------------------------------------------------------------
# Correlative loss L = L_supervised + κ L_{Δ,PG}  (Eq. (35))
# -------------------------------------------------------------------


def correlative_loss_pg(
    gen_out,  # [N_GEN, 2] -> (PG_out, VG_out)
    v_out,  # [N_BUS, 2] -> (e, f) = V_out
    PG_label,  # [N_GEN]
    VG_label,  # [N_GEN]  (optional; can be dummy if not used)
    pd,  # [N_BUS]
    G,
    B,  # [N_BUS, N_BUS]
    A_g2b,  # [N_BUS, N_GEN]
    kappa=0.1,
    use_VG_supervised=True,
):
    r"""
    Implements the total loss:

        L = L_supervised + κ L_{Δ,PG}    (Eq. (35))

    where

      L_supervised  = E[(y_out - y_label)^2]        (Eq. (26))
      L_{Δ,PG}      = E[(PG_out - f_{PG}(V_out))^2] (Eq. (27) specialized)

    Here we implement:

      L_supervised = MSE(PG_out, PG_label) [+ MSE(VG_out, VG_label)]
      L_{Δ,PG}     = MSE(PG_out_bus, f_{PG}(V_out))

    with PG_out_bus = A_g2b · PG_out ∈ ℝ^{N_BUS}.

    Args:
        gen_out : [N_GEN, 2], (PG_out, VG_out)
        v_out   : [N_BUS, 2], (e, f)
        PG_label: [N_GEN]
        VG_label: [N_GEN]
        pd      : [N_BUS]
        G, B    : [N_BUS, N_BUS]
        A_g2b   : [N_BUS, N_GEN]
        kappa   : scalar hyperparameter κ (correlation weight)

    Returns:
        loss_total: scalar tensor
        loss_sup  : scalar (supervised)
        loss_pg   : scalar (correlative PG residual)
    """
    PG_out = gen_out[:, 0]  # [N_GEN]
    VG_out = gen_out[:, 1]  # [N_GEN]

    # ----- L_supervised (Eq. (26)) -----
    loss_sup_pg = F.mse_loss(PG_out, PG_label)
    if use_VG_supervised:
        loss_sup_vg = F.mse_loss(VG_out, VG_label)
        loss_sup = loss_sup_pg + loss_sup_vg
    else:
        loss_sup = loss_sup_pg

    # ----- L_{Δ,PG}: PG residual based on f_{PG}(V_out) (Eq. (27)) -----
    # 1) Map generator PG to bus injections: PG_out_bus = A_g2b · PG_out
    PG_out_bus = A_g2b @ PG_out  # [N_BUS]

    # 2) Compute f_{PG}(V_out) using the nodal active-power equation
    PG_from_V = f_pg_from_v(v_out, pd, G, B)  # [N_BUS]

    # 3) MSE over buses
    loss_pg = F.mse_loss(PG_out_bus, PG_from_V)

    # ----- Total loss (Eq. (35)) -----
    loss_total = loss_sup + kappa * loss_pg
    return loss_total, loss_sup, loss_pg


# -------------------------------------------------------------------
# Example training step skeleton (for one sample; extend to batches later)
# -------------------------------------------------------------------


def training_step_single_sample(
    model: nn.Module,
    batch,
    G,
    B,
    A_g2b,
    kappa=0.1,
):
    """
    batch should provide:
        e_0_k, f_0_k : [N_BUS, CHANNELS_GC_IN]
        pd, qd       : [N_BUS]
        PG_label     : [N_GEN]
        VG_label     : [N_GEN]
        g_ndiag, b_ndiag, g_diag, b_diag: physics operators for this topology
    """
    gen_out, v_out = model(
        batch["e_0_k"],
        batch["f_0_k"],
        batch["pd"],
        batch["qd"],
        batch["g_ndiag"],
        batch["b_ndiag"],
        batch["g_diag"],
        batch["b_diag"],
    )

    loss_total, loss_sup, loss_pg = correlative_loss_pg(
        gen_out=gen_out,
        v_out=v_out,
        PG_label=batch["PG_label"],
        VG_label=batch["VG_label"],
        pd=batch["pd"],
        G=G,
        B=B,
        A_g2b=A_g2b,
        kappa=kappa,
        use_VG_supervised=True,
    )

    return loss_total, loss_sup, loss_pg
