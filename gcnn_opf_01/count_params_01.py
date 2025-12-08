import torch
import sys
import os
from pathlib import Path
import pandas as pd

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_01 import GCNN_OPF_01
from config_model_01 import ModelConfig


def count_parameters(model):
    table = []
    total_params = 0

    print(f"\n{'Layer':<40} | {'Shape':<20} | {'Params':<10}")
    print("-" * 80)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_params = param.numel()
        total_params += num_params
        shape_str = str(list(param.shape))

        print(f"{name:<40} | {shape_str:<20} | {num_params:<10,}")

        # Categorize
        layer_type = "Other"
        if "gc" in name:
            layer_type = "GraphConv"
        elif "fc1" in name:
            layer_type = "FC Trunk"
        elif "fc_gen" in name or "fc_v" in name:
            layer_type = "Heads"

        table.append({"Name": name, "Type": layer_type, "Params": num_params})

    print("-" * 80)
    print(f"{'TOTAL':<63} | {total_params:<10,}")

    # Grouped summary
    df = pd.DataFrame(table)
    if not df.empty:
        print("\nSummary by Layer Type:")
        summary = df.groupby("Type")["Params"].sum().sort_values(ascending=False)
        print(summary)

    return total_params


def main():
    print("Initializing GCNN_OPF_01 with current config...")
    config = ModelConfig()
    print(f"Config: Neurons={config.neurons_fc}, GC_Out={config.channels_gc_out}")

    model = GCNN_OPF_01()
    count_parameters(model)


if __name__ == "__main__":
    main()
