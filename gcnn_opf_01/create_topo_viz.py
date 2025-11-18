"""Generate case6ww topology visualization with N-1 lines and RES buses highlighted."""

import sys
from pathlib import Path
import re

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pypower.api import case6ww
from interactive_viz import create_interactive_topology


def highlight_html(html_path, n1_branches, wind_buses, pv_buses):
    """Modify HTML to highlight N-1 lines (red) and RES buses (green/yellow)."""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Highlight edges (N-1 lines in red, width=4)
    for from_bus, to_bus in n1_branches:
        # Pattern with spaces: "from": X, ... "to": Y
        pattern1 = f'"from": {from_bus},'
        pattern2 = f'"from": {to_bus},'

        # Search for edges with this from value
        for from_val, to_val in [(from_bus, to_bus), (to_bus, from_bus)]:
            from_pattern = f'"from": {from_val},'
            to_pattern = f'"to": {to_val},'

            start_pos = 0
            while True:
                idx = html.find(from_pattern, start_pos)
                if idx == -1:
                    break

                # Check if this edge also has the matching to value
                # Look ahead up to the next edge object
                next_edge = html.find('{"color"', idx + 1)
                if next_edge == -1:
                    next_edge = len(html)

                edge_segment = html[idx : min(idx + 500, next_edge)]
                if to_pattern in edge_segment:
                    # Found the matching edge! Now find the full object
                    obj_start = html.rfind("{", 0, idx)
                    obj_end = html.find("}, {", idx)
                    if obj_end == -1:
                        obj_end = html.find("}]", idx)

                    if obj_start != -1 and obj_end != -1:
                        edge_obj = html[obj_start : obj_end + 1]
                        # Replace color and add width
                        new_edge_obj = re.sub(
                            r'"color": "#[0-9A-Fa-f]{6}"',
                            '"color": "#FF0000"',
                            edge_obj,
                        )
                        new_edge_obj = re.sub(
                            r'"width": [\d.]+', '"width": 4', new_edge_obj
                        )
                        html = html.replace(edge_obj, new_edge_obj, 1)
                        print(f"  Highlighted edge {from_val}-{to_val} in red")
                        break

                start_pos = idx + 1

    # Highlight wind buses (green)
    for bus_id in wind_buses:
        pattern = f'"id": {bus_id},'
        idx = html.find(pattern)
        if idx != -1:
            obj_start = html.rfind("{", 0, idx)
            obj_end = html.find("}, {", idx)
            if obj_end == -1:
                obj_end = html.find("}]", idx)
            if obj_start != -1 and obj_end != -1:
                node_obj = html[obj_start : obj_end + 1]
                # Change color to green
                new_node_obj = re.sub(
                    r'"color": "#[0-9A-Fa-f]{6}"', '"color": "#00FF00"', node_obj
                )
                html = html.replace(node_obj, new_node_obj, 1)
                print(f"  Highlighted bus {bus_id} in green (wind)")
        else:
            print(f"  Warning: Bus {bus_id} not found")

    # Highlight PV buses (yellow)
    for bus_id in pv_buses:
        pattern = f'"id": {bus_id},'
        idx = html.find(pattern)
        if idx != -1:
            obj_start = html.rfind("{", 0, idx)
            obj_end = html.find("}, {", idx)
            if obj_end == -1:
                obj_end = html.find("}]", idx)
            if obj_start != -1 and obj_end != -1:
                node_obj = html[obj_start : obj_end + 1]
                # Change color to yellow
                new_node_obj = re.sub(
                    r'"color": "#[0-9A-Fa-f]{6}"', '"color": "#FFFF00"', node_obj
                )
                html = html.replace(node_obj, new_node_obj, 1)
                print(f"  Highlighted bus {bus_id} in yellow (PV)")
        else:
            print(f"  Warning: Bus {bus_id} not found")

    # Write modified HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("=" * 60)
    print("Generating case6ww Topology with Highlights")
    print("=" * 60)

    ppc = case6ww()

    # Define N-1 branch pairs (1-based external bus numbers)
    n1_branches = [
        (5, 2),  # Topo 1
        (1, 2),  # Topo 2
        (2, 3),  # Topo 3
        (5, 6),  # Topo 4
    ]

    # Define RES buses (1-based external)
    wind_buses = [5]
    pv_buses = [4, 6]

    # Output path
    output_path = REPO_ROOT / "gcnn_opf_01" / "topo_N-1_model_01.html"

    print(f"\nN-1 Contingency Lines (red):")
    for fb, tb in n1_branches:
        print(f"  ({fb}, {tb})")

    print(f"\nWind Buses (green): {wind_buses}")
    print(f"PV Buses (yellow): {pv_buses}")

    print(f"\nGenerating base visualization...")
    print(f"Output: {output_path}")

    # Create visualization
    create_interactive_topology(
        ppc,
        output_file=str(output_path),
        physics=True,
        layout="barnes_hut",
        height="600px",
    )

    print(f"\nApplying highlights...")
    highlight_html(output_path, n1_branches, wind_buses, pv_buses)

    print(f"\n✓ Topology visualization created!")
    print(f"  - Red edges: N-1 contingency lines")
    print(f"  - Green nodes: Wind generation buses")
    print(f"  - Yellow nodes: PV generation buses")


if __name__ == "__main__":
    main()
