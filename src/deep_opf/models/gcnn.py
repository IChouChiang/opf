"""Physics-Guided Graph Convolutional Neural Network for OPF prediction.

Implements the GCNN architecture from Gao et al., "A Physics-Guided Graph
Convolution Neural Network for Optimal Power Flow" (IEEE Trans. Power Systems).

The model uses physics-informed graph convolution layers that incorporate
power flow equations (Eqs. 16-22 from the paper) into the neural network.

Ported from legacy/gcnn_opf_01/model_01.py with standardized interface.
"""

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """
    Physics-guided graph convolution layer implementing Eqs. (16)–(22) + (7)
    from Gao et al., 'A Physics-Guided Graph Convolution Neural Network
    for Optimal Power Flow'.

    The layer uses power system physics (admittance matrices, power balance)
    to guide the message passing between nodes.

    Notation:
        N   : number of buses
        Cin : input channels per node
        Cout: output channels per node

    Physics operators (passed to forward):
        g_ndiag : [N, N] - Off-diagonal conductance matrix z(G_ndiag)
        b_ndiag : [N, N] - Off-diagonal susceptance matrix z(B_ndiag)
        g_diag  : [N]    - Diagonal conductance vector z(G_diag)
        b_diag  : [N]    - Diagonal susceptance vector z(B_diag)

    Args:
        in_channels: Number of input channels per node
        out_channels: Number of output channels per node
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Learnable parameters W1, W2, B1, B2 in Eq. (18)
        # Separate affine maps for e and f voltage components
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
        e: torch.Tensor,
        f: torch.Tensor,
        pd: torch.Tensor,
        qd: torch.Tensor,
        g_ndiag: torch.Tensor,
        b_ndiag: torch.Tensor,
        g_diag: torch.Tensor,
        b_diag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing physics-guided graph convolution.

        Implements:
            - α, β from Eqs. (19), (20)
            - δ, λ from Eqs. (21), (22)
            - e^{l+1}, f^{l+1} from Eqs. (16), (17)
            - Output transform: Y = f(φ(X,A) W + B) from Eq. (7)/(18)

        Args:
            e: Real voltage component [B, N, Cin] or [N, Cin]
            f: Imaginary voltage component [B, N, Cin] or [N, Cin]
            pd: Active power demand [B, N] or [N]
            qd: Reactive power demand [B, N] or [N]
            g_ndiag: Off-diagonal conductance [N, N]
            b_ndiag: Off-diagonal susceptance [N, N]
            g_diag: Diagonal conductance [N]
            b_diag: Diagonal susceptance [N]

        Returns:
            e_next: Updated real voltage component [B, N, Cout] or [N, Cout]
            f_next: Updated imaginary voltage component [B, N, Cout] or [N, Cout]
        """
        # Handle batched vs unbatched inputs
        if e.dim() == 3:  # Batched: [B, N, Cin]
            B, N, Cin = e.shape
            batched = True
        else:  # Unbatched: [N, Cin]
            N, Cin = e.shape
            batched = False
            # Add batch dimension for uniform processing
            e = e.unsqueeze(0)  # [1, N, Cin]
            f = f.unsqueeze(0)
            pd = pd.unsqueeze(0) if pd.dim() == 1 else pd
            qd = qd.unsqueeze(0) if qd.dim() == 1 else qd
            B = 1

        # Ensure pd, qd are [B, N, 1] for broadcasting
        if pd.dim() == 2:
            pd = pd.unsqueeze(-1)  # [B, N, 1]
        if qd.dim() == 2:
            qd = qd.unsqueeze(-1)

        # -----------------------------------------------------------------
        # Eq. (19): α = z(G_ndiag) e^l − z(B_ndiag) f^l
        # -----------------------------------------------------------------
        alpha = torch.einsum("nm,bmc->bnc", g_ndiag, e) - torch.einsum(
            "nm,bmc->bnc", b_ndiag, f
        )

        # -----------------------------------------------------------------
        # Eq. (20): β = z(G_ndiag) f^l + z(B_ndiag) e^l
        # -----------------------------------------------------------------
        beta = torch.einsum("nm,bmc->bnc", g_ndiag, f) + torch.einsum(
            "nm,bmc->bnc", b_ndiag, e
        )

        # -----------------------------------------------------------------
        # Common term s = e^l ⊙ e^l + f^l ⊙ f^l (element-wise)
        # -----------------------------------------------------------------
        s = e * e + f * f  # [B, N, Cin]

        # Prepare diagonal operators as [N, 1] for broadcasting
        if g_diag.dim() == 1:
            g_diag_col = g_diag.view(N, 1)
        else:
            g_diag_col = g_diag.view(N, -1)[:, 0:1]

        if b_diag.dim() == 1:
            b_diag_col = b_diag.view(N, 1)
        else:
            b_diag_col = b_diag.view(N, -1)[:, 0:1]

        # -----------------------------------------------------------------
        # Eq. (21): δ = −P_D − (e^l⊙e^l + f^l⊙f^l) z(G_diag)
        # -----------------------------------------------------------------
        delta = -pd - s * g_diag_col.unsqueeze(0)  # [B, N, Cin]

        # -----------------------------------------------------------------
        # Eq. (22): λ = −Q_D − (e^l⊙e^l + f^l⊙f^l) z(B_diag)
        # -----------------------------------------------------------------
        lamb = -qd - s * b_diag_col.unsqueeze(0)  # [B, N, Cin]

        # -----------------------------------------------------------------
        # Denominator α_i^2 + β_i^2 for Eqs. (16), (17)
        # -----------------------------------------------------------------
        denom = alpha * alpha + beta * beta + self.eps  # [B, N, Cin]

        # -----------------------------------------------------------------
        # Numerators for e^{l+1} and f^{l+1}:
        #   e^{l+1}_i = (δ_i α_i − λ_i β_i) / (α_i^2 + β_i^2)
        #   f^{l+1}_i = (δ_i β_i + λ_i α_i) / (α_i^2 + β_i^2)
        # -----------------------------------------------------------------
        num_e = delta * alpha - lamb * beta
        num_f = delta * beta + lamb * alpha

        e_update = num_e / denom  # [B, N, Cin], φ_e(X,A)
        f_update = num_f / denom  # [B, N, Cin], φ_f(X,A)

        # -----------------------------------------------------------------
        # Eq. (7) / (18): Y = f(φ(X,A) W + B)
        # Apply separate affine maps and tanh activation
        # -----------------------------------------------------------------
        h1 = self.W1(e_update) + self.B1  # [B, N, Cout]
        h2 = self.W2(f_update) + self.B2  # [B, N, Cout]

        e_next = self.act(h1)  # [B, N, Cout]
        f_next = self.act(h2)  # [B, N, Cout]

        # Remove batch dimension if input was unbatched
        if not batched:
            e_next = e_next.squeeze(0)  # [N, Cout]
            f_next = f_next.squeeze(0)

        return e_next, f_next


class GCNN(nn.Module):
    """
    Physics-Guided Graph Convolutional Neural Network for OPF.

    Architecture:
        - Stack of GraphConv layers (physics-guided, Eqs. 16-22 with tanh)
        - MLP trunk with ReLU activation and dropout
        - Three output heads: PG, VG, and V_bus

    Args:
        n_bus: Number of buses in the power system
        n_gen: Number of generators
        in_channels: Input channels per node (default: 8, matches feature iterations)
        hidden_channels: Hidden channels in GraphConv layers (default: 8)
        n_layers: Number of GraphConv layers (default: 2)
        fc_hidden_dim: Hidden dimension for FC trunk (default: 256)
        n_fc_layers: Number of FC layers in trunk (default: 1)
        dropout: Dropout probability (default: 0.0)

    Example:
        >>> model = GCNN(
        ...     n_bus=39,
        ...     n_gen=10,
        ...     in_channels=8,
        ...     hidden_channels=8,
        ...     n_layers=2,
        ...     dropout=0.1,
        ... )
        >>> batch = {
        ...     'e_0_k': torch.randn(32, 39, 8),
        ...     'f_0_k': torch.randn(32, 39, 8),
        ...     'pd': torch.randn(32, 39),
        ...     'qd': torch.randn(32, 39),
        ...     'operators': {
        ...         'g_ndiag': torch.randn(32, 39, 39),
        ...         'b_ndiag': torch.randn(32, 39, 39),
        ...         'g_diag': torch.randn(32, 39),
        ...         'b_diag': torch.randn(32, 39),
        ...     },
        ... }
        >>> out = model(batch)
        >>> out['pg'].shape  # [32, 10]
        >>> out['vg'].shape  # [32, 10]
        >>> out['v_bus'].shape  # [32, 39, 2]
    """

    def __init__(
        self,
        n_bus: int,
        n_gen: int,
        in_channels: int = 8,
        hidden_channels: int = 8,
        n_layers: int = 2,
        fc_hidden_dim: int = 256,
        n_fc_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_bus = n_bus
        self.n_gen = n_gen
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.n_fc_layers = n_fc_layers
        self.dropout_rate = dropout

        # Build GraphConv stack
        self.gc_layers = nn.ModuleList()

        # First layer: in_channels -> hidden_channels
        self.gc_layers.append(GraphConv(in_channels, hidden_channels))

        # Subsequent layers: hidden_channels -> hidden_channels
        for _ in range(n_layers - 1):
            self.gc_layers.append(GraphConv(hidden_channels, hidden_channels))

        # Flatten dimension after GraphConv: [e, f] concatenated
        flat_dim = n_bus * 2 * hidden_channels

        # FC trunk
        self.fc_layers = nn.ModuleList()
        input_dim = flat_dim
        for _ in range(n_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, fc_hidden_dim))
            input_dim = fc_hidden_dim

        self.act_fc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.pg_head = nn.Linear(fc_hidden_dim, n_gen)
        self.vg_head = nn.Linear(fc_hidden_dim, n_gen)
        self.v_bus_head = nn.Linear(fc_hidden_dim, n_bus * 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dictionary containing:
                - 'e_0_k': Real voltage features [B, N, in_channels]
                - 'f_0_k': Imaginary voltage features [B, N, in_channels]
                - 'pd': Active power demand [B, N]
                - 'qd': Reactive power demand [B, N]
                - 'operators': Dict with keys 'g_ndiag', 'b_ndiag', 'g_diag', 'b_diag'
                  Each operator can be [N, N] or [B, N, N] for matrices,
                  [N] or [B, N] for vectors

        Returns:
            Dictionary with keys:
                - 'pg': [B, n_gen] - Active power generation
                - 'vg': [B, n_gen] - Generator voltage magnitudes
                - 'v_bus': [B, n_bus, 2] - All bus voltages (e, f)
        """
        # Extract inputs from batch
        e = batch["e_0_k"]
        f = batch["f_0_k"]
        pd = batch["pd"]
        qd = batch["qd"]

        # Extract operators
        operators = batch["operators"]
        g_ndiag = operators["g_ndiag"]
        b_ndiag = operators["b_ndiag"]
        g_diag = operators["g_diag"]
        b_diag = operators["b_diag"]

        # Handle batched operators: use first sample's operators if batched
        # (assumes same topology within batch)
        if g_ndiag.dim() == 3:
            g_ndiag = g_ndiag[0]
            b_ndiag = b_ndiag[0]
            g_diag = g_diag[0]
            b_diag = b_diag[0]

        # Determine batch size
        if e.dim() == 3:
            B = e.shape[0]
            batched = True
        else:
            B = 1
            batched = False
            e = e.unsqueeze(0)
            f = f.unsqueeze(0)
            pd = pd.unsqueeze(0) if pd.dim() == 1 else pd
            qd = qd.unsqueeze(0) if qd.dim() == 1 else qd

        # Pass through GraphConv layers
        for gc in self.gc_layers:
            e, f = gc(e, f, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)

        # Concatenate e and f features and flatten
        h = torch.cat([e, f], dim=2)  # [B, N, 2*hidden_channels]
        h_flat = h.view(B, -1)  # [B, N*2*hidden_channels]

        # FC trunk
        x = h_flat
        for fc in self.fc_layers:
            x = self.act_fc(fc(x))
            x = self.dropout(x)

        # Output heads
        pg = self.pg_head(x)  # [B, n_gen]
        vg = self.vg_head(x)  # [B, n_gen]
        v_bus_flat = self.v_bus_head(x)  # [B, n_bus*2]
        v_bus = v_bus_flat.view(B, self.n_bus, 2)  # [B, n_bus, 2]

        # Remove batch dimension if input was unbatched
        if not batched:
            pg = pg.squeeze(0)
            vg = vg.squeeze(0)
            v_bus = v_bus.squeeze(0)

        return {
            "pg": pg,
            "vg": vg,
            "v_bus": v_bus,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_bus={self.n_bus}, "
            f"n_gen={self.n_gen}, "
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"n_layers={self.n_layers}, "
            f"fc_hidden_dim={self.fc_hidden_dim}, "
            f"n_fc_layers={self.n_fc_layers}, "
            f"dropout={self.dropout_rate})"
        )
