import sys
from pathlib import Path
import torch
from prettytable import PrettyTable

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dnn_opf_03.model_03 import AdmittanceDNN
from dnn_opf_03.config_03 import ModelConfig


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    config = ModelConfig()
    model = AdmittanceDNN(config)
    print(f"Model Configuration:")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  Hidden Layers: {config.n_hidden_layers}")
    print(f"  Input Dim: {config.input_dim}")

    count_parameters(model)


if __name__ == "__main__":
    main()
