import torch
import torch.nn as nn
from config_model_01 import ModelConfig


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
        # Handle batched vs unbatched inputs
        if e_0_k.dim() == 3:  # Batched: [B, N, Cin]
            B, N, Cin = e_0_k.shape
            batched = True
        else:  # Unbatched: [N, Cin]
            N, Cin = e_0_k.shape
            batched = False
            # Add batch dimension for uniform processing
            e_0_k = e_0_k.unsqueeze(0)  # [1, N, Cin]
            f_0_k = f_0_k.unsqueeze(0)
            pd = pd.unsqueeze(0) if pd.dim() == 1 else pd
            qd = qd.unsqueeze(0) if qd.dim() == 1 else qd
            B = 1

        e = e_0_k  # [B, N, Cin], represents e^l
        f = f_0_k  # [B, N, Cin], represents f^l

        # Ensure pd, qd are [B, N, 1] for broadcasting
        if pd.dim() == 2:
            pd = pd.unsqueeze(-1)  # [B, N, 1]
        if qd.dim() == 2:
            qd = qd.unsqueeze(-1)

        # Handle batched vs unbatched inputs
        if e_0_k.dim() == 3:  # Batched: [B, N, Cin]
            B, N, Cin = e_0_k.shape
            batched = True
        else:  # Unbatched: [N, Cin]
            N, Cin = e_0_k.shape
            batched = False
            # Add batch dimension for uniform processing
            e_0_k = e_0_k.unsqueeze(0)  # [1, N, Cin]
            f_0_k = f_0_k.unsqueeze(0)
            pd = pd.unsqueeze(0) if pd.dim() == 1 else pd
            qd = qd.unsqueeze(0) if qd.dim() == 1 else qd
            B = 1

        e = e_0_k  # [B, N, Cin]
        f = f_0_k

        # Ensure pd, qd are [B, N, 1] for broadcasting
        if pd.dim() == 2:
            pd = pd.unsqueeze(-1)  # [B, N, 1]
        if qd.dim() == 2:
            qd = qd.unsqueeze(-1)

        # ---------------------------------------------------------------------
        # Eq. (19): α = z(G_ndiag) e^l − z(B_ndiag) f^l
        #   g_ndiag, b_ndiag: [N, N]
        #   e, f:             [B, N, Cin]
        # Result α: [B, N, Cin]
        # Use einsum for batched matrix multiplication
        # ---------------------------------------------------------------------
        alpha = torch.einsum("nm,bmc->bnc", g_ndiag, e) - torch.einsum(
            "nm,bmc->bnc", b_ndiag, f
        )

        # ---------------------------------------------------------------------
        # Eq. (20): β = z(G_ndiag) f^l + z(B_ndiag) e^l
        # ---------------------------------------------------------------------
        beta = torch.einsum("nm,bmc->bnc", g_ndiag, f) + torch.einsum(
            "nm,bmc->bnc", b_ndiag, e
        )

        # ---------------------------------------------------------------------
        # Common term s = e^l ⊙ e^l + f^l ⊙ f^l  (element-wise)
        #   s: [B, N, Cin]
        # ---------------------------------------------------------------------
        s = e * e + f * f

        # Prepare diagonal operators as [N, 1] to broadcast over channels and batch
        if g_diag.dim() == 1:
            g_diag_col = g_diag.view(N, 1)  # [N, 1]
        else:
            g_diag_col = g_diag.view(N, -1)[:, 0:1]  # Take first column if 2D

        if b_diag.dim() == 1:
            b_diag_col = b_diag.view(N, 1)  # [N, 1]
        else:
            b_diag_col = b_diag.view(N, -1)[:, 0:1]

        # ---------------------------------------------------------------------
        # Eq. (21): δ = −P_D − (e^l⊙e^l + f^l⊙f^l) z(G_diag)
        #   pd: [B,N,1]; s: [B,N,Cin]; g_diag_col: [N,1]
        #   → broadcast to [B,N,Cin]
        # ---------------------------------------------------------------------
        delta = -pd - s * g_diag_col.unsqueeze(0)  # [B, N, Cin]

        # ---------------------------------------------------------------------
        # Eq. (22): λ = −Q_D − (e^l⊙e^l + f^l⊙f^l) z(B_diag)
        # ---------------------------------------------------------------------
        lamb = -qd - s * b_diag_col.unsqueeze(0)  # [B, N, Cin]

        # ---------------------------------------------------------------------
        # Denominator α_i^2 + β_i^2  (for each bus & channel) in Eqs. (16)(17)
        # ---------------------------------------------------------------------
        denom = alpha * alpha + beta * beta  # [B, N, Cin]
        denom = denom + self.eps  # avoid division by zero

        # ---------------------------------------------------------------------
        # Numerators for e^{l+1} and f^{l+1} in Eqs. (16), (17):
        #   e^{l+1}_i = (δ_i α_i − λ_i β_i) / (α_i^2 + β_i^2)
        #   f^{l+1}_i = (δ_i β_i + λ_i α_i) / (α_i^2 + β_i^2)
        # ---------------------------------------------------------------------
        num_e = delta * alpha - lamb * beta  # [B, N, Cin]
        num_f = delta * beta + lamb * alpha  # [B, N, Cin]

        e_update = num_e / denom  # [B, N, Cin], this is φ_e(X,A)
        f_update = num_f / denom  # [B, N, Cin], this is φ_f(X,A)

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
        h1 = self.W1(e_update) + self.B1  # [B, N, Cout]
        h2 = self.W2(f_update) + self.B2  # [B, N, Cout]

        e_next = self.act(h1)  # [B, N, Cout], e^{l+1}
        f_next = self.act(h2)  # [B, N, Cout], f^{l+1}

        # Remove batch dimension if input was unbatched
        if not batched:
            e_next = e_next.squeeze(0)  # [N, Cout]
            f_next = f_next.squeeze(0)

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

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # GraphConv stack
        self.gc_layers = nn.ModuleList()
        # First layer: in -> out
        self.gc_layers.append(GraphConv(config.channels_gc_in, config.channels_gc_out))
        # Subsequent layers: out -> out
        for _ in range(config.n_gc_layers - 1):
            self.gc_layers.append(
                GraphConv(config.channels_gc_out, config.channels_gc_out)
            )

        # Flatten concatenated e,f features
        flat_dim = config.n_bus * 2 * config.channels_gc_out

        self.act_fc = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

        # FC trunk
        self.fc_layers = nn.ModuleList()
        input_dim = flat_dim
        for _ in range(config.n_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, config.neurons_fc))
            input_dim = config.neurons_fc

        # Heads
        self.fc_gen = nn.Linear(config.neurons_fc, config.n_gen * 2)  # (PG, VG)
        self.fc_v = nn.Linear(config.neurons_fc, config.n_bus * 2)  # (e, f)

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
            e_0_k, f_0_k: [N_BUS, CHANNELS_GC_IN] or [B, N_BUS, CHANNELS_GC_IN]
            pd, qd      : [N_BUS] or [B, N_BUS]
            g/b_ndiag   : [N_BUS, N_BUS]
            g/b_diag    : [N_BUS] or [N_BUS, 1]

        Returns:
            gen_out: [N_GEN, 2] or [B, N_GEN, 2]  (PG, VG)
            v_out  : [N_BUS, 2] or [B, N_BUS, 2]  (e, f)
        """
        # Determine if batched
        batched = e_0_k.dim() == 3
        if batched:
            B = e_0_k.shape[0]
        else:
            # Add batch dimension for uniform processing
            e_0_k = e_0_k.unsqueeze(0)
            f_0_k = f_0_k.unsqueeze(0)
            pd = pd.unsqueeze(0) if pd.dim() == 1 else pd
            qd = qd.unsqueeze(0) if qd.dim() == 1 else qd
            B = 1

        # GraphConv block (handles batched inputs internally)
        e, f = e_0_k, f_0_k
        for gc in self.gc_layers:
            e, f = gc(e, f, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)

        # Node features → FC trunk
        h = torch.cat([e, f], dim=2)  # [B, N_BUS, 2*Cout]
        h_flat = h.view(B, -1)  # [B, N_BUS*2*Cout]

        # FC layers (process each sample in batch)
        x = h_flat
        for fc in self.fc_layers:
            x = self.act_fc(fc(x))
            x = self.dropout(x)

        # Heads
        gen_flat = self.fc_gen(x)  # [B, N_GEN*2]
        v_flat = self.fc_v(x)  # [B, N_BUS*2]

        gen_out = gen_flat.view(B, self.config.n_gen, 2)  # [B, N_GEN, 2]
        v_out = v_flat.view(B, self.config.n_bus, 2)  # [B, N_BUS, 2]

        # Remove batch dimension if input was unbatched
        if not batched:
            gen_out = gen_out.squeeze(0)  # [N_GEN, 2]
            v_out = v_out.squeeze(0)  # [N_BUS, 2]

        return gen_out, v_out
