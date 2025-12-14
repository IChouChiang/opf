"""
Count parameters of GCNN OPF model based on configuration.

Usage:
    python gcnn_opf_01/count_params_01.py --config gcnn_opf_01/configs/base.json
"""

import torch
import sys
import os
from pathlib import Path
import pandas as pd

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from model_01 import GCNN_OPF_01
from model_nodewise import GCNN_OPF_NodeWise
from config_model_01 import load_config


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
        elif "fc1" in name or "fc_layers" in name:
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
    parser = argparse.ArgumentParser(description="Count parameters of GCNN OPF model")
    parser.add_argument(
        "--config",
        type=str,
        default="gcnn_opf_01/configs/base.json",
        help="Path to JSON configuration file",
    )
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    model_config, _ = load_config(args.config)

    print(f"Initializing model ({model_config.model_type}) with loaded config...")
    print(
        f"Config: Neurons={model_config.neurons_fc}, GC_Out={model_config.channels_gc_out}, GC_In={model_config.channels_gc_in}"
    )

    if model_config.model_type == "nodewise":
        model = GCNN_OPF_NodeWise(config=model_config)
    else:
        model = GCNN_OPF_01(config=model_config)

    count_parameters(model)


if __name__ == "__main__":
    main()
