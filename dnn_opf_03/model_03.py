"""DeepOPF-FT Baseline Model (AdmittanceDNN)."""

import torch
import torch.nn as nn
from config_03 import ModelConfig


class AdmittanceDNN(nn.Module):
    """
    Fully Connected Network (MLP) for OPF prediction.

    Input: Concatenated vector [Pd, Qd, vec(G), vec(B)]
    Output: [PG, VG] for all generators
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_bus = config.n_bus
        layers = []
        input_dim = config.input_dim

        # Hidden layers
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = config.hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output heads
        # PG head: [hidden_dim -> n_gen]
        self.pg_head = nn.Linear(config.hidden_dim, config.n_gen)

        # VG head: [hidden_dim -> n_gen]
        self.vg_head = nn.Linear(config.hidden_dim, config.n_gen)

        # V_all head: [hidden_dim -> n_bus * 2] (e, f)
        self.v_all_head = nn.Linear(config.hidden_dim, config.n_bus * 2)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            gen_out: [batch_size, n_gen, 2] (PG, VG)
            v_out: [batch_size, n_bus, 2] (e, f)
        """
        features = self.feature_extractor(x)

        pg = self.pg_head(features)
        vg = self.vg_head(features)

        v_all_flat = self.v_all_head(features)
        v_out = v_all_flat.view(-1, self.n_bus, 2)

        gen_out = torch.stack([pg, vg], dim=-1)

        return gen_out, v_out
