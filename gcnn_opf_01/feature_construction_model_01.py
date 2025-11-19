"""
Model-Informed Feature Construction (Section III-C).

Implements the iterative voltage estimation process from Fig. 4:
- Equations (16)–(22): voltage update with physics constraints
- Equations (23)–(24): active/reactive power computation for clamping
- Equation (25): voltage magnitude normalization

Returns e_0_k, f_0_k ∈ R^{N×k} for k iterations (typically k=8).
"""

import torch
import numpy as np


def construct_features(
    pd: torch.Tensor,
    qd: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    g_ndiag: torch.Tensor,
    b_ndiag: torch.Tensor,
    g_diag: torch.Tensor,
    b_diag: torch.Tensor,
    gen_bus_indices: np.ndarray,
    PG_min: torch.Tensor,
    PG_max: torch.Tensor,
    QG_min: torch.Tensor,
    QG_max: torch.Tensor,
    gen_mask: torch.Tensor,
    k: int = 8,
    eps: float = 1e-8,
):
    """
    Construct initial voltage features e_0_k, f_0_k via iterative physics updates.

    Algorithm (from paper Fig. 4):
        1. Initialize: e⁰ = 1, f⁰ = 0
        2. For l = 1 to k:
            a. Compute PG, QG using current e, f (eqs 23, 24)
            b. Clamp generator powers to limits
            c. Compute α, β, δ, λ (eqs 19–22)
            d. Update e^{l+1}, f^{l+1} (eqs 16, 17)
            e. Normalize voltage magnitude (eq 25)
            f. Store e^l, f^l as features
        3. Return stacked features [N, k]

    Args:
        pd, qd         : [N_BUS] active/reactive demand (p.u., RES as negative load)
        G, B           : [N_BUS, N_BUS] full admittance matrices
        g_ndiag, b_ndiag : [N_BUS, N_BUS] off-diagonal operators
        g_diag, b_diag : [N_BUS] diagonal operators
        gen_bus_indices: np.ndarray [N_GEN] internal bus indices for generators
        PG_min, PG_max : [N_GEN] active power limits (p.u.)
        QG_min, QG_max : [N_GEN] reactive power limits (p.u.)
        gen_mask       : [N_BUS] bool mask (True at generator buses)
        k              : number of feature iterations (default 8)
        eps            : small constant to avoid division by zero

    Returns:
        e_0_k : [N_BUS, k] real voltage features
        f_0_k : [N_BUS, k] imaginary voltage features
    """

    N_BUS = pd.shape[0]
    N_GEN = len(gen_bus_indices)

    # Convert gen_bus_indices to tensor for indexing
    gen_bus_indices_tensor = torch.tensor(gen_bus_indices, dtype=torch.long)

    # Ensure pd, qd are [N_BUS] vectors
    if pd.dim() > 1:
        pd = pd.squeeze()
    if qd.dim() > 1:
        qd = qd.squeeze()

    # --- 1) Initialize voltages: e⁰ = 1, f⁰ = 0 ---
    e = torch.ones(N_BUS, dtype=torch.float32)
    f = torch.zeros(N_BUS, dtype=torch.float32)

    # Storage for features
    e_list = []
    f_list = []

    # --- 2) Iterative feature construction ---
    for iteration in range(k):

        # -----------------------------------------------------------------
        # Step 2a: Compute PG, QG at current voltages (eqs 23, 24)
        # -----------------------------------------------------------------
        # Equation (23): PG_i = PD_i + e_i*(Ge)_i - e_i*(Bf)_i + f_i*(Gf)_i + f_i*(Be)_i
        # Equation (24): QG_i = QD_i - f_i*(Ge)_i + f_i*(Bf)_i + e_i*(Gf)_i - e_i*(Be)_i

        Ge = G @ e  # [N_BUS]
        Gf = G @ f  # [N_BUS]
        Be = B @ e  # [N_BUS]
        Bf = B @ f  # [N_BUS]

        # Full-bus PG, QG (including non-gen buses, will clamp only generators)
        PG_bus = pd + e * Ge - e * Bf + f * Gf + f * Be  # [N_BUS]
        QG_bus = qd - f * Ge + f * Bf + e * Gf - e * Be  # [N_BUS]

        # -----------------------------------------------------------------
        # Step 2b: Clamp generator powers to limits (only at gen buses)
        # -----------------------------------------------------------------
        # Extract generator powers using gen_bus_indices
        PG_gen = PG_bus[gen_bus_indices_tensor]  # [N_GEN]
        QG_gen = QG_bus[gen_bus_indices_tensor]  # [N_GEN]

        # Clamp to limits
        PG_gen_clamped = torch.clamp(PG_gen, min=PG_min, max=PG_max)
        QG_gen_clamped = torch.clamp(QG_gen, min=QG_min, max=QG_max)

        # Write back to full-bus vectors (only affects generator buses)
        PG_bus[gen_bus_indices_tensor] = PG_gen_clamped
        QG_bus[gen_bus_indices_tensor] = QG_gen_clamped

        # Update effective demands after clamping
        # PD_eff = PG - (e*Ge - e*Bf + f*Gf + f*Be)
        # QD_eff = QG - (e*Gf - e*Be - f*Ge + f*Bf)
        # But we use the clamped PG/QG to re-compute pd_eff, qd_eff
        pd_eff = PG_bus - (e * Ge - e * Bf + f * Gf + f * Be)
        qd_eff = QG_bus - (e * Gf - e * Be - f * Ge + f * Bf)

        # -----------------------------------------------------------------
        # Step 2c: Compute α, β, δ, λ (eqs 19–22)
        # -----------------------------------------------------------------
        # Equation (19): α = z(G_ndiag) e - z(B_ndiag) f
        alpha = g_ndiag @ e - b_ndiag @ f  # [N_BUS]

        # Equation (20): β = z(G_ndiag) f + z(B_ndiag) e
        beta = g_ndiag @ f + b_ndiag @ e  # [N_BUS]

        # Common term: s = e² + f²
        s = e * e + f * f  # [N_BUS]

        # Equation (21): δ = -PD - (e² + f²) z(G_diag)
        delta = -pd_eff - s * g_diag  # [N_BUS]

        # Equation (22): λ = -QD - (e² + f²) z(B_diag)
        lamb = -qd_eff - s * b_diag  # [N_BUS]

        # -----------------------------------------------------------------
        # Step 2d: Update voltages (eqs 16, 17)
        # -----------------------------------------------------------------
        # Denominator: α² + β²
        denom = alpha * alpha + beta * beta + eps  # avoid division by zero

        # Equation (16): e^{l+1} = (δα - λβ) / (α² + β²)
        e_next = (delta * alpha - lamb * beta) / denom

        # Equation (17): f^{l+1} = (δβ + λα) / (α² + β²)
        f_next = (delta * beta + lamb * alpha) / denom

        # -----------------------------------------------------------------
        # Step 2e: Normalize voltage magnitude (eq 25)
        # -----------------------------------------------------------------
        # v_mag = sqrt(e² + f²) → normalize to unit magnitude
        v_mag = torch.sqrt(e_next * e_next + f_next * f_next + eps)
        e_next = e_next / v_mag
        f_next = f_next / v_mag

        # -----------------------------------------------------------------
        # Step 2f: Store features
        # -----------------------------------------------------------------
        e_list.append(e_next)
        f_list.append(f_next)

        # Update for next iteration
        e = e_next
        f = f_next

    # --- 3) Stack features into [N_BUS, k] tensors ---
    e_0_k = torch.stack(e_list, dim=1)  # [N_BUS, k]
    f_0_k = torch.stack(f_list, dim=1)  # [N_BUS, k]

    return e_0_k, f_0_k


# -------------------------------------------------------------------
# Convenience wrapper for full pipeline
# -------------------------------------------------------------------


def construct_features_from_ppc(
    ppc_int,
    pd: torch.Tensor,
    qd: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    g_ndiag: torch.Tensor,
    b_ndiag: torch.Tensor,
    g_diag: torch.Tensor,
    b_diag: torch.Tensor,
    k: int = 8,
):
    """
    Convenience wrapper that extracts gen limits from ppc_int and calls construct_features.

    Args:
        ppc_int    : internal-numbered PYPOWER case dict
        pd, qd     : [N_BUS] demand vectors (p.u.)
        G, B       : [N_BUS, N_BUS] full admittance matrices
        g_ndiag, b_ndiag, g_diag, b_diag : physics operators
        k          : number of iterations (default 8)

    Returns:
        e_0_k, f_0_k : [N_BUS, k] voltage features
    """
    # Import here to avoid circular dependency
    from sample_config_model_01 import extract_gen_limits

    gen_bus_indices, PG_min, PG_max, QG_min, QG_max, gen_mask = extract_gen_limits(
        ppc_int
    )

    return construct_features(
        pd=pd,
        qd=qd,
        G=G,
        B=B,
        g_ndiag=g_ndiag,
        b_ndiag=b_ndiag,
        g_diag=g_diag,
        b_diag=b_diag,
        gen_bus_indices=gen_bus_indices,
        PG_min=PG_min,
        PG_max=PG_max,
        QG_min=QG_min,
        QG_max=QG_max,
        gen_mask=gen_mask,
        k=k,
    )
