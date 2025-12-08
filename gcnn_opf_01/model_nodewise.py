import torch
import torch.nn as nn
from config_model_01 import ModelConfig
from model_01 import GraphConv  # Reuse the existing GraphConv layer


class GCNN_OPF_NodeWise(nn.Module):
    """
    True Graph-Invariant GCNN for OPF.

    Difference from GCNN_OPF_01:
    - NO Flattening layer.
    - Uses Node-Wise Readout: The same MLP is applied to every node.
    - Output: [Batch, N_BUS, 2] -> Sliced to get [Batch, N_GEN, 2]

    This architecture allows Inductive Generalization (Train on 30 bus -> Test on 118 bus).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Store generator indices for extraction
        # If not provided in config, assume first n_gen nodes (Case6ww default)
        if hasattr(config, "gen_indices") and config.gen_indices is not None:
            self.gen_indices = torch.tensor(config.gen_indices, dtype=torch.long)
        else:
            self.gen_indices = torch.arange(config.n_gen, dtype=torch.long)

        # GraphConv stack (Same as original)
        self.gc1 = GraphConv(config.channels_gc_in, config.channels_gc_out)
        self.gc2 = GraphConv(config.channels_gc_out, config.channels_gc_out)

        # Node-Wise Readout MLP
        # Input: [Batch, N_BUS, 2 * Channels] (Concatenated e, f)
        # We do NOT multiply by N_BUS here. The Linear layer applies to the last dim.
        input_dim = 2 * config.channels_gc_out

        self.act_fc = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

        # Shared Trunk (Applied to every node independently)
        self.fc1 = nn.Linear(input_dim, config.neurons_fc)

        # Heads (Applied to every node independently)
        # Output is 2 values per node: (P, V) or (e, f) depending on interpretation
        # For gen_head: Predicts (PG, VG) for THAT node
        # For v_head:   Predicts (e, f) for THAT node
        self.fc_gen = nn.Linear(config.neurons_fc, 2)
        self.fc_v = nn.Linear(config.neurons_fc, 2)

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
        # Determine if batched
        batched = e_0_k.dim() == 3
        if batched:
            B = e_0_k.shape[0]
        else:
            e_0_k = e_0_k.unsqueeze(0)
            f_0_k = f_0_k.unsqueeze(0)
            pd = pd.unsqueeze(0) if pd.dim() == 1 else pd
            qd = qd.unsqueeze(0) if qd.dim() == 1 else qd
            B = 1

        # GraphConv block
        e, f = self.gc1(e_0_k, f_0_k, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)
        e, f = self.gc2(e, f, pd, qd, g_ndiag, b_ndiag, g_diag, b_diag)

        # Node features: [B, N_BUS, 2*Cout]
        h = torch.cat([e, f], dim=2)

        # Node-Wise MLP
        # PyTorch Linear applies to the last dimension, so it works on [B, N, C] automatically
        h_fc1 = self.act_fc(self.fc1(h))  # [B, N_BUS, NEURONS_FC]
        h_fc1 = self.dropout(h_fc1)

        # Heads
        # v_out: [B, N_BUS, 2] (e, f for all nodes)
        v_out = self.fc_v(h_fc1)

        # gen_full: [B, N_BUS, 2] (PG, VG predictions for ALL nodes)
        gen_full = self.fc_gen(h_fc1)

        # Extract only the generator nodes for the final output
        # gen_out: [B, N_GEN, 2]
        if self.gen_indices.device != gen_full.device:
            self.gen_indices = self.gen_indices.to(gen_full.device)

        gen_out = torch.index_select(gen_full, 1, self.gen_indices)

        # Remove batch dimension if input was unbatched
        if not batched:
            gen_out = gen_out.squeeze(0)
            v_out = v_out.squeeze(0)

        return gen_out, v_out
