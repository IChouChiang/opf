"""Calculate model parameter count for architecture comparison.

Usage:
    # Default model (GCNN on case6)
    python scripts/calc_params.py

    # DNN model
    python scripts/calc_params.py model=dnn

    # GCNN with different hidden dimensions
    python scripts/calc_params.py model=gcnn model.architecture.fc_hidden_dim=256

    # DNN with custom architecture
    python scripts/calc_params.py model=dnn model.architecture.hidden_dim=128 model.architecture.num_layers=4

    # Different data configs (affects input dimensions)
    python scripts/calc_params.py model=dnn data=case39
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Reuse instantiate_model from train.py
from train import instantiate_model


def count_parameters(model, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Total parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(count: int) -> str:
    """Format parameter count with K/M suffix."""
    if count >= 1_000_000:
        return f"{count:,} ({count / 1_000_000:.2f}M)"
    elif count >= 1_000:
        return f"{count:,} ({count / 1_000:.2f}K)"
    return f"{count:,}"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Calculate and print model parameter count.

    Args:
        cfg: Hydra configuration object
    """
    # Extract data parameters needed for model instantiation
    n_bus = cfg.data.n_bus
    n_gen = cfg.data.n_gen

    # Instantiate model
    model = instantiate_model(cfg, n_bus, n_gen)

    # Count parameters
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)

    # Print results
    print("\n" + "=" * 50)
    print("Model Parameter Count")
    print("=" * 50)
    print(f"Model: {cfg.model.name.upper()}")
    print(f"Data:  {cfg.data.name} (n_bus={n_bus}, n_gen={n_gen})")
    print("-" * 50)
    print(f"Trainable Parameters: {format_params(trainable)}")
    print(f"Total Parameters:     {format_params(total)}")
    print("=" * 50)

    # Print architecture details
    print("\nArchitecture:")
    print(OmegaConf.to_yaml(cfg.model.architecture))


if __name__ == "__main__":
    main()
