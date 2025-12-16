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

        # Determine layer structure
        if config.hidden_layers is not None and len(config.hidden_layers) > 0:
            layer_sizes = config.hidden_layers
        else:
            layer_sizes = [config.hidden_dim] * config.n_hidden_layers

        # Hidden layers
        for h_dim in layer_sizes:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output heads
        # PG head: [last_hidden_dim -> n_gen]
        self.pg_head = nn.Linear(input_dim, config.n_gen)

        # VG head: [last_hidden_dim -> n_gen]
        self.vg_head = nn.Linear(input_dim, config.n_gen)

        # V_all head: [last_hidden_dim -> n_bus * 2] (e, f)
        self.v_all_head = nn.Linear(input_dim, config.n_bus * 2)

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
