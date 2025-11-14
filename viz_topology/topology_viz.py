# topology_viz.py
"""PYPOWER Case Topology Visualization using NetworkX.

Independent module for visualizing power system network topology from PYPOWER cases.
Displays buses, branches, generators with proper indexing and color-coding.

Functions
---------
visualize_case(ppc, show_indices=True, show_generators=True, layout='spring', figsize=(12, 10)):
    Main visualization function for PYPOWER case topology.

get_network_stats(ppc):
    Extract and print network statistics.

Example
-------
>>> from pypower.api import case39
>>> from topology_viz import visualize_case
>>> ppc = case39()
>>> visualize_case(ppc, show_indices=True)
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Literal


def get_network_stats(ppc: dict) -> dict:
    """Extract network statistics from PYPOWER case.
    
    Parameters
    ----------
    ppc : dict
        PYPOWER case dictionary
        
    Returns
    -------
    stats : dict
        Dictionary with network statistics
    """
    bus = ppc["bus"]
    gen = ppc["gen"]
    branch = ppc["branch"]
    
    n_bus = bus.shape[0]
    n_gen = gen.shape[0]
    n_branch = branch.shape[0]
    
    # Bus types: 1=PQ, 2=PV, 3=Slack, 4=isolated
    bus_types = bus[:, 1].astype(int)
    n_pq = np.sum(bus_types == 1)
    n_pv = np.sum(bus_types == 2)
    n_slack = np.sum(bus_types == 3)
    
    # Load and generation
    baseMVA = float(ppc["baseMVA"])
    total_load_mw = np.sum(bus[:, 2])
    total_gen_capacity_mw = np.sum(gen[:, 8])  # Pmax
    
    stats = {
        "n_bus": n_bus,
        "n_gen": n_gen,
        "n_branch": n_branch,
        "n_pq": n_pq,
        "n_pv": n_pv,
        "n_slack": n_slack,
        "baseMVA": baseMVA,
        "total_load_mw": total_load_mw,
        "total_gen_capacity_mw": total_gen_capacity_mw,
    }
    
    return stats


def visualize_case(
    ppc: dict,
    show_indices: bool = True,
    show_generators: bool = True,
    layout: Literal["spring", "kamada_kawai", "circular", "spectral"] = "spring",
    figsize: tuple[int, int] = (14, 10),
    save_path: str | None = None,
    show_plot: bool = False,
) -> tuple[nx.Graph, dict]:
    """Visualize PYPOWER case topology as a network graph.
    
    Parameters
    ----------
    ppc : dict
        PYPOWER case dictionary (external or internal numbering)
    show_indices : bool
        Whether to display bus indices on nodes
    show_generators : bool
        Whether to highlight generator buses with markers
    layout : str
        NetworkX layout algorithm: 'spring', 'kamada_kawai', 'circular', 'spectral'
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        If provided, save figure to this path
    show_plot : bool
        Whether to call plt.show() (set False for SSH/remote sessions)
        
    Returns
    -------
    G : nx.Graph
        NetworkX graph object
    pos : dict
        Node positions
    """
    # Extract data
    bus = ppc["bus"]
    branch = ppc["branch"]
    gen = ppc["gen"]
    baseMVA = float(ppc.get("baseMVA", 100.0))
    
    n_bus = bus.shape[0]
    n_branch = branch.shape[0]
    
    # Build NetworkX graph
    G = nx.Graph()
    
    # Add nodes (buses) - use bus ID from column 0
    bus_ids = bus[:, 0].astype(int)
    for i, bus_id in enumerate(bus_ids):
        bus_type = int(bus[:, 1][i])
        vm = float(bus[:, 7][i])
        va = float(bus[:, 8][i])
        pd = float(bus[:, 2][i])
        qd = float(bus[:, 3][i])
        
        G.add_node(
            bus_id,
            bus_type=bus_type,
            vm=vm,
            va=va,
            pd=pd,
            qd=qd,
            internal_idx=i,
        )
    
    # Add edges (branches)
    for br_idx in range(n_branch):
        f_bus = int(branch[br_idx, 0])
        t_bus = int(branch[br_idx, 1])
        G.add_edge(f_bus, t_bus, branch_idx=br_idx)
    
    # Get generator bus IDs
    gen_buses = set(gen[:, 0].astype(int))
    
    # Compute layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Extract network stats
    stats = get_network_stats(ppc)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate nodes by bus type
    slack_nodes = [n for n in G.nodes() if G.nodes[n]["bus_type"] == 3]
    pv_nodes = [n for n in G.nodes() if G.nodes[n]["bus_type"] == 2]
    pq_nodes = [n for n in G.nodes() if G.nodes[n]["bus_type"] == 1]
    
    # Draw nodes by type with different colors
    nx.draw_networkx_nodes(
        G, pos, nodelist=slack_nodes, node_color="red", 
        node_size=800, label="Slack Bus", alpha=0.9, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=pv_nodes, node_color="lightgreen", 
        node_size=700, label="PV Bus (Generator)", alpha=0.9, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=pq_nodes, node_color="lightblue", 
        node_size=600, label="PQ Bus (Load)", alpha=0.9, ax=ax
    )
    
    # Highlight generator buses with star markers
    if show_generators:
        gen_node_list = [n for n in G.nodes() if n in gen_buses]
        gen_pos_x = [pos[n][0] for n in gen_node_list]
        gen_pos_y = [pos[n][1] for n in gen_node_list]
        ax.scatter(
            gen_pos_x, gen_pos_y, marker="*", s=400, 
            c="gold", edgecolors="black", linewidths=1.5,
            label="Generator", zorder=10
        )
    
    # Draw edges (branches)
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6, edge_color="gray", ax=ax)
    
    # Draw labels (bus indices)
    if show_indices:
        labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels, font_size=10, font_weight="bold", 
            font_color="black", ax=ax
        )
    
    # Add title and stats
    title = (
        f"Power System Topology Visualization\n"
        f"Buses: {stats['n_bus']} | Branches: {stats['n_branch']} | Generators: {stats['n_gen']} | "
        f"Base MVA: {stats['baseMVA']:.0f}"
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    # Legend
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    
    # Clean up axes
    ax.axis("off")
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    
    # Show plot only if requested (for local/non-SSH sessions)
    if show_plot:
        plt.show()
    else:
        # Auto-save if no save_path provided and not showing
        if save_path is None:
            auto_path = f"topology_{stats['n_bus']}bus.png"
            plt.savefig(auto_path, dpi=150, bbox_inches="tight")
            print(f"Figure auto-saved to: {auto_path}")
        plt.close(fig)
    
    # Print network statistics
    print("\n" + "=" * 60)
    print("Network Statistics")
    print("=" * 60)
    print(f"Total buses:           {stats['n_bus']}")
    print(f"  - Slack buses:       {stats['n_slack']}")
    print(f"  - PV buses:          {stats['n_pv']}")
    print(f"  - PQ buses:          {stats['n_pq']}")
    print(f"Total branches:        {stats['n_branch']}")
    print(f"Total generators:      {stats['n_gen']}")
    print(f"Base MVA:              {stats['baseMVA']:.0f}")
    print(f"Total load:            {stats['total_load_mw']:.2f} MW")
    print(f"Total gen capacity:    {stats['total_gen_capacity_mw']:.2f} MW")
    print("=" * 60 + "\n")
    
    return G, pos


def visualize_case_with_branch_labels(
    ppc: dict,
    layout: Literal["spring", "kamada_kawai", "circular"] = "spring",
    figsize: tuple[int, int] = (14, 10),
    save_path: str | None = None,
    show_plot: bool = False,
) -> None:
    """Visualize case with branch indices labeled on edges.
    
    Parameters
    ----------
    ppc : dict
        PYPOWER case dictionary
    layout : str
        Layout algorithm
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show_plot : bool
        Whether to call plt.show()
    """
    G, pos = visualize_case(ppc, show_indices=True, show_generators=True, layout=layout, figsize=figsize, show_plot=False)
    
    # Additional plot with edge labels
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw basic structure
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=600, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
    
    # Draw edge labels (branch indices)
    edge_labels = nx.get_edge_attributes(G, "branch_idx")
    edge_labels_formatted = {edge: f"br{idx}" for edge, idx in edge_labels.items()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels_formatted, font_size=8, font_color="red", ax=ax
    )
    
    ax.set_title("Branch Indices on Edges", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Branch labels figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        if save_path is None:
            auto_path = "topology_branch_labels.png"
            plt.savefig(auto_path, dpi=150, bbox_inches="tight")
            print(f"Branch labels figure auto-saved to: {auto_path}")
        plt.close(fig)


__all__ = ["visualize_case", "visualize_case_with_branch_labels", "get_network_stats"]
