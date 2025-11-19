import torch
import torch.nn as nn
from config_model_01 import (
    N_BUS,
    N_GEN,
    NEURONS_FC,
    CHANNELS_GC_IN,
    CHANNELS_GC_OUT,
)


class GraphConv(nn.Module):
    """
    Physics-guided graph convolution layer implementing Eqs. (16)–(22) + (7)
    from Gao et al., 'A Physics-Guided Graph Convolution Neural Network
    for Optimal Power Flow'.

    Notation:

        N  : number of buses
        Cin: input channels per node
        Cout: output channels per node (e.g. 8)

    Inputs to forward():
        e_0_k : [N, Cin]  -- current e^l (stacked over channels)
        f_0_k : [N, Cin]  -- current f^l (stacked over channels)
        pd    : [N] or [N, 1]  -- P_D
        qd    : [N] or [N, 1]  -- Q_D

    Physics operators stored as buffers:
        g_ndiag : [N, N]  -- z(G_ndiag)
        b_ndiag : [N, N]  -- z(B_ndiag)
        g_diag  : [N]     -- z(G_diag)
        b_diag  : [N]     -- z(B_diag)

    Output:
        e_next : [N, Cout]  -- e^{l+1}
        f_next : [N, Cout]  -- f^{l+1}
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        # Learnable parameters W1, W2, B1, B2 in Eq. (18)
        # We treat e^l, f^l as [N, Cin], so W1,W2: Cin -> Cout
        self.W1 = nn.Linear(in_channels, out_channels, bias=False)
        self.W2 = nn.Linear(in_channels, out_channels, bias=False)
        self.B1 = nn.Parameter(torch.zeros(out_channels))
        self.B2 = nn.Parameter(torch.zeros(out_channels))

        # Activation f(·) in Eq. (7) / (18) is tanh
        self.act = torch.tanh

        # Small epsilon to avoid division by zero
        self.eps = 1e-8

    def forward(
        self,
        e_0_k: torch.Tensor,
        f_0_k: torch.Tensor,
        pd: torch.Tensor,
        qd: torch.Tensor,
        g_ndiag: torch.Tensor,
        b_ndiag: torch.Tensor,
        g_diag: torch.Tensor,
        b_diag: torch.Tensor,
    ):
        """
        Implements:
            - α, β from (19), (20)
            - δ, λ from (21), (22)
            - e^{l+1}, f^{l+1} from (16), (17)
            - then Eq. (7)/(18): Y = f( φ(X,A) W + B )

        """

        e = e_0_k  # [N, Cin], represents e^l, Cin equals CHANNELS_GC_IN
        f = f_0_k  # [N, Cin], represents f^l

        # Ensure pd, qd are [N, 1] for broadcasting
        if pd.dim() == 1:
            pd = pd.view(-1, 1)  # reshapes from [N] to [N, 1]
        if qd.dim() == 1:
            qd = qd.view(-1, 1)

        N, Cin = e.shape

        # ---------------------------------------------------------------------
        # Eq. (19): α = z(G_ndiag) e^l − z(B_ndiag) f^l
        #   g_ndiag, b_ndiag: [N, N]
        #   e, f:             [N, Cin]
        # Result α: [N, Cin]
        # ---------------------------------------------------------------------
        alpha = g_ndiag @ e - b_ndiag @ f

        # ---------------------------------------------------------------------
        # Eq. (20): β = z(G_ndiag) f^l + z(B_ndiag) e^l
        # ---------------------------------------------------------------------
        beta = g_ndiag @ f + b_ndiag @ e

        # ---------------------------------------------------------------------
        # Common term s = e^l ⊙ e^l + f^l ⊙ f^l  (element-wise)
        #   s: [N, Cin]
        # ---------------------------------------------------------------------
        s = e * e + f * f

        # Prepare diagonal operators as [N, 1] to broadcast over channels
        g_diag_col = g_diag.view(N, 1)  # [N, 1]
        b_diag_col = b_diag.view(N, 1)  # [N, 1]

        # ---------------------------------------------------------------------
        # Eq. (21): δ = −P_D − (e^l⊙e^l + f^l⊙f^l) z(G_diag)
        #   pd: [N,1]; s: [N,Cin]; g_diag_col: [N,1]
        #   → broadcast to [N,Cin]
        # ---------------------------------------------------------------------
        delta = -pd - s * g_diag_col  # [N, Cin]

        # ---------------------------------------------------------------------
        # Eq. (22): λ = −Q_D − (e^l⊙e^l + f^l⊙f^l) z(B_diag)
        # ---------------------------------------------------------------------
        lamb = -qd - s * b_diag_col  # [N, Cin]

        # ---------------------------------------------------------------------
        # Denominator α_i^2 + β_i^2  (for each bus & channel) in Eqs. (16)(17)
        # ---------------------------------------------------------------------
        denom = alpha * alpha + beta * beta  # [N, Cin]
        denom = denom + self.eps  # avoid division by zero

        # ---------------------------------------------------------------------
        # Numerators for e^{l+1} and f^{l+1} in Eqs. (16), (17):
        #   e^{l+1}_i = (δ_i α_i − λ_i β_i) / (α_i^2 + β_i^2)
        #   f^{l+1}_i = (δ_i β_i + λ_i α_i) / (α_i^2 + β_i^2)
        # ---------------------------------------------------------------------
        num_e = delta * alpha - lamb * beta  # [N, Cin]
        num_f = delta * beta + lamb * alpha  # [N, Cin]

        e_update = num_e / denom  # [N, Cin], this is φ_e(X,A)
        f_update = num_f / denom  # [N, Cin], this is φ_f(X,A)

        # ---------------------------------------------------------------------
        # Eq. (7) / Eq. (18): Y = f( φ(X,A) W + B )
        # Here φ(X,A) is represented by e_update, f_update.
        # We apply separate affine maps W1,B1 and W2,B2 to e and f,
        # then apply tanh activation.
        #
        #   h1 = e_update W1 + B1
        #   h2 = f_update W2 + B2
        #   e^{l+1} = tanh(h1)
        #   f^{l+1} = tanh(h2)
        # ---------------------------------------------------------------------
        h1 = self.W1(e_update) + self.B1  # [N, Cout]
        h2 = self.W2(f_update) + self.B2  # [N, Cout]

        e_next = self.act(h1)  # [N, Cout], e^{l+1}
        f_next = self.act(h2)  # [N, Cout], f^{l+1}

        return e_next, f_next


class GCNN_OPF_01(nn.Module):
    """
    GCNN model for OPF (case6ww-optimized):
      - 2 GraphConv layers (physics-guided, Eqs. 16–22 with tanh in Eq. 7/18)
      - 1 shared FC trunk
      - 2 heads:
          * gen_head → [N_GEN, 2] (PG, VG)
          * v_head   → [N_BUS, 2] (e, f) for physics/correlative losses
    """

    def __init__(self):
        super().__init__()

        # GraphConv stack (2 layers for case6ww per paper guidance)
        self.gc1 = GraphConv(CHANNELS_GC_IN, CHANNELS_GC_OUT)
        self.gc2 = GraphConv(CHANNELS_GC_OUT, CHANNELS_GC_OUT)

        # Flatten concatenated e,f features
        flat_dim = N_BUS * 2 * CHANNELS_GC_OUT

        self.act_fc = nn.ReLU()
        self.fc1 = nn.Linear(flat_dim, NEURONS_FC)  # shared trunk

        # Heads
        self.fc_gen = nn.Linear(NEURONS_FC, N_GEN * 2)  # (PG, VG)
        self.fc_v = nn.Linear(NEURONS_FC, N_BUS * 2)    # (e, f)

    def forward(
        self,
        e_0_k: torch.Tensor,
        f_0_k: torch.Tensor,
        pd: torch.Tensor,
        qd: torch.Tensor,
        g_ndiag: torch.Tensor,
        b_ndiag: torch.Tensor,
        g_diag: torch.Tensor,
        b_diag: torch.Tensor,
    ):
        """
        Args:
            e_0_k, f_0_k: [N_BUS, CHANNELS_GC_IN]
            pd, qd     : [N_BUS] or [N_BUS, 1]

        Returns:
            gen_out: [N_GEN, 2]  (PG, VG)
            v_out  : [N_BUS, 2]  (e, f)
        """

        # GraphConv block
        e, f = self.gc1(e_0_k, f_0_k, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)
        e, f = self.gc2(e, f, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)

        # Node features → FC trunk
        h = torch.cat([e, f], dim=1)            # [N_BUS, 2*Cout]
        h_flat = h.view(-1)                     # [N_BUS*2*Cout]
        h_fc1 = self.act_fc(self.fc1(h_flat))   # [NEURONS_FC]

        # Heads
        gen_flat = self.fc_gen(h_fc1)           # [N_GEN*2]
        v_flat = self.fc_v(h_fc1)               # [N_BUS*2]

        gen_out = gen_flat.view(N_GEN, 2)
        v_out = v_flat.view(N_BUS, 2)

        return gen_out, v_out
