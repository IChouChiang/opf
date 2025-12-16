"""Physics-informed loss functions for OPF models.

Implements physics-based loss terms derived from AC power flow equations,
including active power balance (Eq. 8) and reactive power balance constraints.

Ported from legacy/dnn_opf_03/loss_model_03.py and legacy/gcnn_opf_01/loss_model_01.py
with standardized interface for batched inputs.
"""

import torch
import torch.nn.functional as F


def build_gen_bus_matrix(
    n_bus: int,
    gen_bus_indices: torch.Tensor | list[int],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build generator-to-bus incidence matrix A_g2b.

    This matrix maps generator outputs to bus injections:
        P_bus = A_g2b @ P_gen

    Args:
        n_bus: Number of buses in the system
        gen_bus_indices: 1D tensor/list of bus indices for each generator,
                         values in [0, n_bus-1]
        device: Torch device for the output tensor
        dtype: Data type for the output tensor

    Returns:
        A_g2b: [n_bus, n_gen] incidence matrix where
               A_g2b[i, g] = 1 if generator g is at bus i, else 0
    """
    if isinstance(gen_bus_indices, list):
        gen_bus_indices = torch.tensor(gen_bus_indices, dtype=torch.long)

    n_gen = len(gen_bus_indices)
    A = torch.zeros(n_bus, n_gen, dtype=dtype, device=device)

    for g, bus in enumerate(gen_bus_indices):
        A[bus, g] = 1.0

    return A


def compute_power_from_voltage(
    v_bus: torch.Tensor,
    pd: torch.Tensor,
    qd: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute nodal active and reactive power injections from voltage using
    AC power flow equations in Cartesian form.

    Active Power (Eq. 8 from paper):
        P_i = P_D_i + e_i * Σ_j[G_ij*e_j - B_ij*f_j] + f_i * Σ_j[G_ij*f_j + B_ij*e_j]

    Reactive Power:
        Q_i = Q_D_i + f_i * Σ_j[G_ij*e_j - B_ij*f_j] - e_i * Σ_j[G_ij*f_j + B_ij*e_j]

    Args:
        v_bus: Bus voltages [B, n_bus, 2] or [n_bus, 2] where [..., 0]=e, [..., 1]=f
        pd: Active power demand [B, n_bus] or [n_bus]
        qd: Reactive power demand [B, n_bus] or [n_bus]
        G: Conductance matrix [n_bus, n_bus]
        B: Susceptance matrix [n_bus, n_bus]

    Returns:
        P_from_V: Active power at each bus [B, n_bus] or [n_bus]
        Q_from_V: Reactive power at each bus [B, n_bus] or [n_bus]
    """
    # Handle batched vs unbatched inputs
    batched = v_bus.dim() == 3
    if not batched:
        v_bus = v_bus.unsqueeze(0)  # [1, n_bus, 2]
        pd = pd.unsqueeze(0) if pd.dim() == 1 else pd
        qd = qd.unsqueeze(0) if qd.dim() == 1 else qd

    # Extract e and f components: [B, n_bus]
    e = v_bus[..., 0]  # [B, n_bus]
    f = v_bus[..., 1]  # [B, n_bus]

    # Compute matrix-vector products
    # G @ e, G @ f, B @ e, B @ f for each sample in batch
    # Using einsum for batched matmul: [n_bus, n_bus] x [B, n_bus] -> [B, n_bus]
    Ge = torch.einsum("nm,bm->bn", G, e)  # [B, n_bus]
    Gf = torch.einsum("nm,bm->bn", G, f)  # [B, n_bus]
    Be = torch.einsum("nm,bm->bn", B, e)  # [B, n_bus]
    Bf = torch.einsum("nm,bm->bn", B, f)  # [B, n_bus]

    # Active power: P = Pd + e*(Ge - Bf) + f*(Gf + Be)
    P_from_V = pd + e * (Ge - Bf) + f * (Gf + Be)

    # Reactive power: Q = Qd + f*(Ge - Bf) - e*(Gf + Be)
    Q_from_V = qd + f * (Ge - Bf) - e * (Gf + Be)

    if not batched:
        P_from_V = P_from_V.squeeze(0)
        Q_from_V = Q_from_V.squeeze(0)

    return P_from_V, Q_from_V


def physics_loss(
    pg: torch.Tensor,
    v_bus: torch.Tensor,
    pd: torch.Tensor,
    qd: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    gen_bus_matrix: torch.Tensor,
    qg: torch.Tensor | None = None,
    include_reactive: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute physics-based loss terms for active (and optionally reactive) power balance.

    The physics loss measures how well the predicted generator outputs and
    bus voltages satisfy the AC power flow equations.

    Loss_P = MSE(A_g2b @ PG_pred, P_from_V)
    Loss_Q = MSE(A_g2b @ QG_pred, Q_from_V)  [if include_reactive=True]

    Args:
        pg: Predicted active power generation [B, n_gen] or [n_gen]
        v_bus: Predicted bus voltages [B, n_bus, 2] or [n_bus, 2]
               where [..., 0]=e (real), [..., 1]=f (imaginary)
        pd: Active power demand [B, n_bus] or [n_bus]
        qd: Reactive power demand [B, n_bus] or [n_bus]
        G: Conductance matrix [n_bus, n_bus]
        B: Susceptance matrix [n_bus, n_bus]
        gen_bus_matrix: Generator-to-bus incidence [n_bus, n_gen]
        qg: Predicted reactive power generation [B, n_gen] or [n_gen]
            Required if include_reactive=True
        include_reactive: Whether to compute reactive power loss

    Returns:
        Dictionary with:
            - 'loss_p': Active power balance loss (scalar)
            - 'loss_q': Reactive power balance loss (scalar, only if include_reactive)
            - 'loss_physics': Total physics loss (loss_p + loss_q if reactive included)
    """
    # Handle batched vs unbatched
    batched = pg.dim() == 2
    if not batched:
        pg = pg.unsqueeze(0)
        v_bus = v_bus.unsqueeze(0)
        pd = pd.unsqueeze(0) if pd.dim() == 1 else pd
        qd = qd.unsqueeze(0) if qd.dim() == 1 else qd
        if qg is not None:
            qg = qg.unsqueeze(0)

    B_size = pg.shape[0]

    # Compute power from voltage using AC power flow equations
    P_from_V, Q_from_V = compute_power_from_voltage(v_bus, pd, qd, G, B)

    # Map generator power to bus injections: [B, n_bus]
    # PG_bus = gen_bus_matrix @ PG for each sample
    PG_bus = torch.einsum("ng,bg->bn", gen_bus_matrix, pg)  # [B, n_bus]

    # Active power balance loss
    loss_p = F.mse_loss(PG_bus, P_from_V)

    result = {
        "loss_p": loss_p,
        "loss_physics": loss_p,
    }

    # Reactive power balance loss (optional)
    if include_reactive:
        if qg is None:
            raise ValueError("qg must be provided when include_reactive=True")
        QG_bus = torch.einsum("ng,bg->bn", gen_bus_matrix, qg)  # [B, n_bus]
        loss_q = F.mse_loss(QG_bus, Q_from_V)
        result["loss_q"] = loss_q
        result["loss_physics"] = loss_p + loss_q

    return result


def correlative_loss(
    pg: torch.Tensor,
    vg: torch.Tensor,
    v_bus: torch.Tensor,
    pg_label: torch.Tensor,
    vg_label: torch.Tensor,
    pd: torch.Tensor,
    qd: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    gen_bus_matrix: torch.Tensor,
    kappa: float = 0.1,
    include_vg_supervised: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Compute correlative loss combining supervised loss and physics loss.

    Total Loss = L_supervised + κ * L_physics  (Eq. 35 from paper)

    Where:
        L_supervised = MSE(PG_pred, PG_label) + MSE(VG_pred, VG_label)
        L_physics = MSE(A_g2b @ PG_pred, P_from_V(v_bus))

    Args:
        pg: Predicted active power generation [B, n_gen] or [n_gen]
        vg: Predicted generator voltages [B, n_gen] or [n_gen]
        v_bus: Predicted bus voltages [B, n_bus, 2] or [n_bus, 2]
        pg_label: Ground truth PG [B, n_gen] or [n_gen]
        vg_label: Ground truth VG [B, n_gen] or [n_gen]
        pd: Active power demand [B, n_bus] or [n_bus]
        qd: Reactive power demand [B, n_bus] or [n_bus]
        G: Conductance matrix [n_bus, n_bus]
        B: Susceptance matrix [n_bus, n_bus]
        gen_bus_matrix: Generator-to-bus incidence [n_bus, n_gen]
        kappa: Weight for physics loss term (default: 0.1)
        include_vg_supervised: Whether to include VG in supervised loss

    Returns:
        Dictionary with:
            - 'loss_total': Total combined loss
            - 'loss_supervised': Supervised MSE loss
            - 'loss_supervised_pg': PG supervised loss component
            - 'loss_supervised_vg': VG supervised loss component (if included)
            - 'loss_physics': Physics-based loss (active power balance)
    """
    # Supervised loss
    loss_sup_pg = F.mse_loss(pg, pg_label)

    if include_vg_supervised:
        loss_sup_vg = F.mse_loss(vg, vg_label)
        loss_supervised = loss_sup_pg + loss_sup_vg
    else:
        loss_sup_vg = torch.tensor(0.0, device=pg.device)
        loss_supervised = loss_sup_pg

    # Physics loss
    physics_result = physics_loss(
        pg=pg,
        v_bus=v_bus,
        pd=pd,
        qd=qd,
        G=G,
        B=B,
        gen_bus_matrix=gen_bus_matrix,
        include_reactive=False,
    )
    loss_physics = physics_result["loss_p"]

    # Total loss
    loss_total = loss_supervised + kappa * loss_physics

    return {
        "loss_total": loss_total,
        "loss_supervised": loss_supervised,
        "loss_supervised_pg": loss_sup_pg,
        "loss_supervised_vg": loss_sup_vg,
        "loss_physics": loss_physics,
    }


# Backward-compatible alias
def correlative_loss_pg(
    gen_out: torch.Tensor,
    v_out: torch.Tensor,
    PG_label: torch.Tensor,
    VG_label: torch.Tensor,
    pd: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    A_g2b: torch.Tensor,
    kappa: float = 0.1,
    use_VG_supervised: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Legacy-compatible interface for correlative loss.

    This function provides backward compatibility with the legacy interface
    that accepts gen_out as [n_gen, 2] stacked tensor.

    Args:
        gen_out: [n_gen, 2] or [B, n_gen, 2] with (PG, VG) stacked
        v_out: [n_bus, 2] or [B, n_bus, 2] with (e, f)
        PG_label: [n_gen] or [B, n_gen]
        VG_label: [n_gen] or [B, n_gen]
        pd: [n_bus] or [B, n_bus]
        G, B: [n_bus, n_bus]
        A_g2b: [n_bus, n_gen]
        kappa: Physics loss weight
        use_VG_supervised: Include VG in supervised loss

    Returns:
        Tuple of (loss_total, loss_supervised, loss_physics)
    """
    # Unpack gen_out
    if gen_out.dim() == 2:
        # Unbatched: [n_gen, 2]
        pg = gen_out[:, 0]
        vg = gen_out[:, 1]
    else:
        # Batched: [B, n_gen, 2]
        pg = gen_out[..., 0]
        vg = gen_out[..., 1]

    result = correlative_loss(
        pg=pg,
        vg=vg,
        v_bus=v_out,
        pg_label=PG_label,
        vg_label=VG_label,
        pd=pd,
        qd=torch.zeros_like(pd),  # qd not used in active power loss
        G=G,
        B=B,
        gen_bus_matrix=A_g2b,
        kappa=kappa,
        include_vg_supervised=use_VG_supervised,
    )

    return result["loss_total"], result["loss_supervised"], result["loss_physics"]
