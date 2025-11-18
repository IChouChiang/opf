"""Generate interactive visualization for case6ww."""

import sys
from pathlib import Path

# Add src to path for interactive_viz
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pypower.api import case6ww
from interactive_viz import create_interactive_topology


def main():
    print("=" * 60)
    print("Generating Interactive Visualization: case6ww")
    print("=" * 60)

    # Load case6ww
    ppc = case6ww()

    # Display basic info
    n_bus = ppc["bus"].shape[0]
    n_gen = ppc["gen"].shape[0]
    n_branch = ppc["branch"].shape[0]
    base_mva = ppc["baseMVA"]

    print(f"\nSystem: case6ww")
    print(f"  Buses: {n_bus}")
    print(f"  Generators: {n_gen}")
    print(f"  Branches: {n_branch}")
    print(f"  Base MVA: {base_mva}")

    # Create visualization in viz_topology folder
    output_path = Path(__file__).parent / "case6ww_interactive.html"

    print(f"\nGenerating interactive HTML...")
    print(f"Output: {output_path}")

    create_interactive_topology(
        ppc,
        output_file=str(output_path),
        physics=True,
        layout="barnes_hut",
        height="600px",
    )

    print(f"\n✓ Visualization created: {output_path}")
    print("  Open this file in a browser to interact with the network.")


if __name__ == "__main__":
    main()
