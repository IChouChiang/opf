import torch
import sys
import os

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_01 import GCNN_OPF_01


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    model = GCNN_OPF_01()
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")

    # Detailed breakdown
    print("\nBreakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")


if __name__ == "__main__":
    main()
