"""AdmittanceDNN Model for OPF prediction.

A fully connected neural network (MLP) that takes flattened grid state
as input and predicts generator setpoints and bus voltages.

Ported from legacy/dnn_opf_03/model_03.py with standardized interface.
"""

import torch
import torch.nn as nn


class AdmittanceDNN(nn.Module):
    """
    Fully Connected Network (MLP) for OPF prediction.

    Input: Concatenated vector [Pd, Qd, vec(G), vec(B)]
    Output: Dictionary with keys {'pg', 'vg', 'v_bus'}

    Args:
        input_dim: Dimension of input features (2*n_bus + 2*n_bus^2 for flat topology)
        hidden_dim: Number of neurons per hidden layer
        num_layers: Number of hidden layers
        n_gen: Number of generators
        n_bus: Number of buses
        dropout: Dropout probability (default: 0.0)

    Example:
        >>> model = AdmittanceDNN(
        ...     input_dim=3120,  # 2*39 + 2*39^2 for case39
        ...     hidden_dim=256,
        ...     num_layers=3,
        ...     n_gen=10,
        ...     n_bus=39,
        ...     dropout=0.1,
        ... )
        >>> x = torch.randn(32, 3120)  # batch of 32
        >>> out = model(x)
        >>> out['pg'].shape  # [32, 10]
        >>> out['vg'].shape  # [32, 10]
        >>> out['v_bus'].shape  # [32, 39, 2]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_gen: int,
        n_bus: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_gen = n_gen
        self.n_bus = n_bus
        self.dropout = dropout

        # Build feature extractor (hidden layers)
        layers: list[nn.Module] = []
        in_features = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output heads
        # PG head: predicts active power generation [n_gen]
        self.pg_head = nn.Linear(hidden_dim, n_gen)

        # VG head: predicts generator voltage magnitude [n_gen]
        self.vg_head = nn.Linear(hidden_dim, n_gen)

        # V_bus head: predicts all bus voltages as (e, f) components [n_bus * 2]
        self.v_bus_head = nn.Linear(hidden_dim, n_bus * 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
               Typically concatenated [Pd, Qd, vec(G), vec(B)]

        Returns:
            Dictionary with keys:
                - 'pg': [batch_size, n_gen] - Active power generation
                - 'vg': [batch_size, n_gen] - Generator voltage magnitudes
                - 'v_bus': [batch_size, n_bus, 2] - All bus voltages (e, f)
        """
        # Extract features through hidden layers
        features = self.feature_extractor(x)

        # Compute outputs from each head
        pg = self.pg_head(features)  # [B, n_gen]
        vg = self.vg_head(features)  # [B, n_gen]

        v_bus_flat = self.v_bus_head(features)  # [B, n_bus * 2]
        v_bus = v_bus_flat.view(-1, self.n_bus, 2)  # [B, n_bus, 2]

        return {
            "pg": pg,
            "vg": vg,
            "v_bus": v_bus,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"n_gen={self.n_gen}, "
            f"n_bus={self.n_bus}, "
            f"dropout={self.dropout})"
        )
