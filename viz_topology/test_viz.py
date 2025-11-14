# test_viz.py
"""Quick test script for topology visualization.

For SSH/remote sessions, figures are saved to PNG files instead of displayed.
"""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SSH sessions

from pypower.api import case39
from topology_viz import visualize_case, get_network_stats


def test_basic_visualization():
    """Test basic visualization functionality."""
    print("Loading IEEE 39-bus case...")
    ppc = case39()
    
    print("Extracting network statistics...")
    stats = get_network_stats(ppc)
    print(f"Loaded case with {stats['n_bus']} buses, {stats['n_branch']} branches")
    
    print("Visualizing topology (saving to file)...")
    G, pos = visualize_case(
        ppc,
        show_indices=True,
        show_generators=True,
        layout="spring",
        figsize=(14, 10),
        save_path="test_case39_topology.png",
        show_plot=False  # Don't show in SSH session
    )
    
    print(f"\nNetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print("Test completed successfully!")
    print("Check 'test_case39_topology.png' for the visualization.")


if __name__ == "__main__":
    test_basic_visualization()
