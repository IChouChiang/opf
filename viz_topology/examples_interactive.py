# examples_interactive.py
"""Interactive visualization examples for various PYPOWER cases."""
from __future__ import annotations
from pypower.api import case9, case39, case57, case118
from interactive_viz import create_interactive_topology


def example_interactive_case9():
    """Interactive IEEE 9-bus system."""
    print("\n" + "=" * 60)
    print("Interactive Example: IEEE 9-Bus System")
    print("=" * 60)
    ppc = case9()
    create_interactive_topology(
        ppc,
        output_file="case9_interactive.html",
        physics=True,
        layout="barnes_hut",
        height="700px",
    )


def example_interactive_case39():
    """Interactive IEEE 39-bus system."""
    print("\n" + "=" * 60)
    print("Interactive Example: IEEE 39-Bus System (New England)")
    print("=" * 60)
    ppc = case39()
    create_interactive_topology(
        ppc,
        output_file="case39_interactive.html",
        physics=True,
        layout="barnes_hut",
        height="850px",
    )


def example_interactive_case57():
    """Interactive IEEE 57-bus system."""
    print("\n" + "=" * 60)
    print("Interactive Example: IEEE 57-Bus System")
    print("=" * 60)
    ppc = case57()
    create_interactive_topology(
        ppc,
        output_file="case57_interactive.html",
        physics=True,
        layout="force_atlas",
        height="900px",
    )


def example_interactive_case118():
    """Interactive IEEE 118-bus system."""
    print("\n" + "=" * 60)
    print("Interactive Example: IEEE 118-Bus System")
    print("=" * 60)
    ppc = case118()
    create_interactive_topology(
        ppc,
        output_file="case118_interactive.html",
        physics=True,
        layout="barnes_hut",
        height="1000px",
    )


def run_all_interactive_examples():
    """Generate all interactive HTML visualizations."""
    print("\n" + "█" * 60)
    print("Generating Interactive PYPOWER Topology Visualizations")
    print("█" * 60)
    
    example_interactive_case9()
    example_interactive_case39()
    example_interactive_case57()
    
    # Large case (optional)
    # example_interactive_case118()
    
    print("\n" + "█" * 60)
    print("All interactive HTML files generated!")
    print("Open the .html files in your browser to interact.")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    # Run specific example
    example_interactive_case39()
    
    # Or run all examples
    # run_all_interactive_examples()
