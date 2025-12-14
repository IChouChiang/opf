"""
Hasan DNN (Model M8) for OPF.

Architecture:
- Input: Flattened vector [pd, qd, delta_e, delta_f] (dim = 4 * N_BUS)
- Hidden: 4 layers x 1000 neurons (ReLU)
- Output: 
    - gen_head: [N_GEN, 2] (PG, VG)
    - v_head: [N_BUS, 2] (e, f) for physics loss
"""

import torch
import torch.nn as nn

class HasanDNN(nn.Module):
    def __init__(self, n_bus, n_gen, hidden_dim=1000):
        super(HasanDNN, self).__init__()
        
        self.n_bus = n_bus
        self.n_gen = n_gen
        self.input_dim = 4 * n_bus
        
        # Feature extractor (Shared layers)
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Heads
        # Generator head: Predicts PG and VG for each generator
        self.gen_head = nn.Linear(hidden_dim, n_gen * 2)
        
        # Voltage head: Predicts e and f for each bus (needed for physics loss)
        self.v_head = nn.Linear(hidden_dim, n_bus * 2)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, 4 * N_BUS] flattened input
            
        Returns:
            gen_out: [batch_size, N_GEN, 2] (PG, VG)
            v_out: [batch_size, N_BUS, 2] (e, f)
        """
        # Shared features
        features = self.shared_layers(x)
        
        # Generator output
        gen_out_flat = self.gen_head(features)
        gen_out = gen_out_flat.view(-1, self.n_gen, 2)
        
        # Voltage output
        v_out_flat = self.v_head(features)
        v_out = v_out_flat.view(-1, self.n_bus, 2)
        
        return gen_out, v_out
