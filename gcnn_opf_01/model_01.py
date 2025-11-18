import torch.nn as nn
from config_model_01 import N_BUS, N_GEN, NEURONS_FC, CHANNELS_GC_IN, CHANNELS_GC_OUT


class GCNN_OPF_01(nn.Module):
    def __init__(self):
        super().__init__()
        self.gc1 = GraphConv(CHANNELS_GC_IN, CHANNELS_GC_OUT)
        self.gc2 = GraphConv(CHANNELS_GC_OUT, CHANNELS_GC_OUT)
        self.gc3 = GraphConv(CHANNELS_GC_OUT, CHANNELS_GC_OUT)

        flat_dim = N_BUS * CHANNELS_GC_OUT
        self.act_gc = nn.Tanh()  # Activation for GC layers
        self.act_fc = nn.ReLU()  # Activation for FC layers

        self.fc1 = nn.Linear(flat_dim, NEURONS_FC)
        self.fc2 = nn.Linear(NEURONS_FC, NEURONS_FC)
        self.fc3 = nn.Linear(NEURONS_FC, N_GEN * 2)  # Predict PG and VG

    def forward(self, x):
        # x: [39, CHANNELS_GC]
        h = self.act_gc(self.gc1(x))
        h = self.act_gc(self.gc2(h))
        h = self.act_gc(self.gc3(h))
        h_flat = h.view(-1)  # Flatten
        h_fc1 = self.act_fc(self.fc1(h_flat))
        h_fc2 = self.act_fc(self.fc2(h_fc1))
        out = self.fc3(h_fc2)
        return out  # [N_GEN , 2]
