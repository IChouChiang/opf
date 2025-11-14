# interactive_viz.py
"""Interactive PYPOWER Case Topology Visualization using PyVis.

Creates interactive HTML files where users can:
- Click/select nodes to highlight connected edges
- Drag nodes to rearrange layout
- Zoom and pan the network
- See node details on hover
- Physics simulation for dynamic layout

Functions
---------
create_interactive_topology(ppc, output_file='topology.html', physics=True, height='800px'):
    Generate interactive HTML visualization of PYPOWER case.

Example
-------
>>> from pypower.api import case39
>>> from interactive_viz import create_interactive_topology
>>> ppc = case39()
>>> create_interactive_topology(ppc, output_file='case39_interactive.html')
>>> # Open case39_interactive.html in browser
"""
from __future__ import annotations
import numpy as np
from pyvis.network import Network
from typing import Literal


def create_interactive_topology(
    ppc: dict,
    output_file: str = "topology_interactive.html",
    physics: bool = True,
    height: str = "800px",
    width: str = "100%",
    notebook: bool = False,
    layout: Literal["barnes_hut", "force_atlas", "hierarchical", "repulsion"] = "barnes_hut",
) -> Network:
    """Create interactive HTML visualization of PYPOWER case topology.
    
    Parameters
    ----------
    ppc : dict
        PYPOWER case dictionary (external or internal numbering)
    output_file : str
        Output HTML file path
    physics : bool
        Enable physics simulation for node movement
    height : str
        Canvas height (CSS format: '800px', '100vh', etc.)
    width : str
        Canvas width (CSS format: '100%', '1200px', etc.)
    notebook : bool
        Whether running in Jupyter notebook
    layout : str
        Physics layout algorithm: 'barnes_hut', 'force_atlas', 'hierarchical', 'repulsion'
        
    Returns
    -------
    net : pyvis.network.Network
        PyVis Network object
        
    Notes
    -----
    Interaction features:
    - Click node to highlight connected edges
    - Drag nodes to reposition
    - Scroll to zoom
    - Hover for node details
    """
    # Extract data
    bus = ppc["bus"]
    branch = ppc["branch"]
    gen = ppc["gen"]
    baseMVA = float(ppc.get("baseMVA", 100.0))
    
    n_bus = bus.shape[0]
    n_branch = branch.shape[0]
    
    # Create PyVis network
    net = Network(
        height=height,
        width=width,
        notebook=notebook,
        directed=False,
        bgcolor="#ffffff",
        font_color="#000000",
    )
    
    # Configure physics
    if physics:
        if layout == "barnes_hut":
            net.barnes_hut(
                gravity=-8000,
                central_gravity=0.3,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09,
                overlap=0
            )
        elif layout == "force_atlas":
            net.force_atlas_2based(
                gravity=-50,
                central_gravity=0.01,
                spring_length=200,
                spring_strength=0.08,
                damping=0.4
            )
        elif layout == "repulsion":
            net.repulsion(
                node_distance=200,
                central_gravity=0.2,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09
            )
        elif layout == "hierarchical":
            net.set_options("""
            var options = {
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "direction": "UD",
                  "sortMethod": "directed"
                }
              }
            }
            """)
    else:
        net.toggle_physics(False)
    
    # Get generator bus IDs
    gen_buses = set(gen[:, 0].astype(int))
    
    # Bus type colors
    bus_colors = {
        1: "#87CEEB",  # PQ bus - light blue
        2: "#90EE90",  # PV bus - light green
        3: "#FF6B6B",  # Slack bus - red
        4: "#D3D3D3",  # Isolated - gray
    }
    
    bus_type_names = {
        1: "PQ (Load)",
        2: "PV (Generator)",
        3: "Slack (Reference)",
        4: "Isolated",
    }
    
    # Add nodes (buses)
    bus_ids = bus[:, 0].astype(int)
    for i, bus_id in enumerate(bus_ids):
        bus_type = int(bus[i, 1])
        vm = float(bus[i, 7])
        va = float(bus[i, 8])
        pd = float(bus[i, 2])
        qd = float(bus[i, 3])
        vmin = float(bus[i, 12])
        vmax = float(bus[i, 11])
        
        # Node color based on bus type
        color = bus_colors.get(bus_type, "#D3D3D3")
        
        # Generator marker
        is_gen = bus_id in gen_buses
        shape = "star" if is_gen else "dot"
        size = 25 if is_gen else 20
        
        # Title (hover tooltip)
        title = f"""<b>Bus {bus_id}</b><br>
Type: {bus_type_names.get(bus_type, 'Unknown')}<br>
Vm: {vm:.3f} p.u.<br>
Va: {va:.2f}°<br>
Pd: {pd:.2f} MW<br>
Qd: {qd:.2f} MVAr<br>
V limits: [{vmin:.2f}, {vmax:.2f}]"""
        
        if is_gen:
            # Find generator info
            gen_idx = np.where(gen[:, 0] == bus_id)[0]
            if len(gen_idx) > 0:
                g = gen_idx[0]
                pg = float(gen[g, 1])
                qg = float(gen[g, 2])
                pmax = float(gen[g, 8])
                pmin = float(gen[g, 9])
                title += f"""<br><br><b>Generator:</b><br>
Pg: {pg:.2f} MW<br>
Qg: {qg:.2f} MVAr<br>
Pmax: {pmax:.2f} MW<br>
Pmin: {pmin:.2f} MW"""
        
        net.add_node(
            int(bus_id),
            label=str(bus_id),
            title=title,
            color=color,
            shape=shape,
            size=size,
            borderWidth=2,
            borderWidthSelected=4,
        )
    
    # Add edges (branches)
    for br_idx in range(n_branch):
        f_bus = int(branch[br_idx, 0])
        t_bus = int(branch[br_idx, 1])
        r = float(branch[br_idx, 2])
        x = float(branch[br_idx, 3])
        b = float(branch[br_idx, 4])
        rateA = float(branch[br_idx, 5])
        
        # Edge title (hover)
        edge_title = f"""<b>Branch {br_idx}</b><br>
From Bus {f_bus} → To Bus {t_bus}<br>
R: {r:.4f} p.u.<br>
X: {x:.4f} p.u.<br>
B: {b:.4f} p.u.<br>
Rating: {rateA:.1f} MVA"""
        
        # Edge width based on rating (if available)
        if rateA > 0:
            width = 1 + (rateA / 500.0)  # Scale by rating
        else:
            width = 2
        
        net.add_edge(
            f_bus,
            t_bus,
            title=edge_title,
            width=width,
            color="#888888",
            smooth={"type": "continuous"},
            selectionWidth=4,  # Highlighted width when selected
            hoverWidth=0.5,
        )
    
    # Add interaction options
    net.set_options("""
    var options = {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": {
          "enabled": true
        }
      },
      "manipulation": {
        "enabled": false
      },
      "nodes": {
        "font": {
          "size": 14,
          "face": "arial",
          "color": "#000000"
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous"
        },
        "color": {
          "inherit": false,
          "highlight": "#FF6B6B",
          "hover": "#FFD700"
        }
      }
    }
    """)
    
    # Save to HTML
    net.save_graph(output_file)
    
    # Print statistics
    print("=" * 60)
    print(f"Interactive Topology Generated: {output_file}")
    print("=" * 60)
    print(f"Buses: {n_bus}")
    print(f"Branches: {n_branch}")
    print(f"Generators: {len(gen_buses)}")
    print(f"Base MVA: {baseMVA:.0f}")
    print("\nInteraction Guide:")
    print("  • Click node to highlight connected edges")
    print("  • Drag nodes to reposition")
    print("  • Scroll to zoom in/out")
    print("  • Hover over nodes/edges for details")
    print("  • Use navigation buttons (bottom-right)")
    print("=" * 60)
    
    return net


__all__ = ["create_interactive_topology"]
