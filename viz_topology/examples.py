# examples.py
"""Example usage of topology visualization for different PYPOWER cases.

NOTE: For SSH/remote sessions, set show_plot=False and figures will be saved to files.
      For local sessions, set show_plot=True to display interactively.
"""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SSH

from pypower.api import case9, case39, case57, case118
from topology_viz import visualize_case, visualize_case_with_branch_labels


# Set this to True if running locally (not over SSH)
SHOW_PLOTS = False


def example_case9():
    """Visualize IEEE 9-bus system."""
    print("\n" + "=" * 60)
    print("Example: IEEE 9-Bus System")
    print("=" * 60)
    ppc = case9()
    visualize_case(
        ppc, 
        show_indices=True, 
        show_generators=True, 
        layout="spring",
        save_path="case9_topology.png",
        show_plot=SHOW_PLOTS
    )


def example_case39():
    """Visualize IEEE 39-bus system."""
    print("\n" + "=" * 60)
    print("Example: IEEE 39-Bus System (New England)")
    print("=" * 60)
    ppc = case39()
    visualize_case(
        ppc, 
        show_indices=True, 
        show_generators=True, 
        layout="kamada_kawai",
        save_path="case39_topology.png",
        show_plot=SHOW_PLOTS
    )


def example_case57():
    """Visualize IEEE 57-bus system."""
    print("\n" + "=" * 60)
    print("Example: IEEE 57-Bus System")
    print("=" * 60)
    ppc = case57()
    visualize_case(
        ppc, 
        show_indices=True, 
        show_generators=True, 
        layout="spring",
        save_path="case57_topology.png",
        show_plot=SHOW_PLOTS
    )


def example_case118():
    """Visualize IEEE 118-bus system."""
    print("\n" + "=" * 60)
    print("Example: IEEE 118-Bus System")
    print("=" * 60)
    ppc = case118()
    # Larger system, use spectral layout
    visualize_case(
        ppc, 
        show_indices=False, 
        show_generators=True, 
        layout="spectral", 
        figsize=(16, 12),
        save_path="case118_topology.png",
        show_plot=SHOW_PLOTS
    )


def example_with_branch_labels():
    """Visualize case9 with branch indices labeled."""
    print("\n" + "=" * 60)
    print("Example: Case9 with Branch Labels")
    print("=" * 60)
    ppc = case9()
    visualize_case_with_branch_labels(
        ppc, 
        layout="spring",
        save_path="case9_branch_labels.png",
        show_plot=SHOW_PLOTS
    )


def run_all_examples():
    """Run all visualization examples."""
    print("\n" + "█" * 60)
    print("PYPOWER Topology Visualization Examples")
    print("█" * 60)
    
    # Small cases
    example_case9()
    
    # Medium cases
    example_case39()
    example_case57()
    
    # Branch labels example
    example_with_branch_labels()
    
    # Uncomment for large case (takes longer)
    # example_case118()
    
    print("\n" + "█" * 60)
    print("All examples completed!")
    print("Check the generated PNG files for visualizations.")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    # Run specific example
    example_case39()
    
    # Or run all examples
    # run_all_examples()
