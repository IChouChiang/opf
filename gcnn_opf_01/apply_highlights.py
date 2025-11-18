"""Apply highlights to existing topo_N-1_model_01.html file."""

import re
from pathlib import Path


def highlight_html(html_path, n1_branches, wind_buses, pv_buses):
    """Modify HTML to highlight N-1 lines (red) and RES buses (green/yellow)."""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Highlight edges (N-1 lines in red, width=4)
    for from_bus, to_bus in n1_branches:
        # Pattern with spaces: "from": X, ... "to": Y
        for from_val, to_val in [(from_bus, to_bus), (to_bus, from_bus)]:
            from_pattern = f'"from": {from_val},'
            to_pattern = f'"to": {to_val},'

            start_pos = 0
            while True:
                idx = html.find(from_pattern, start_pos)
                if idx == -1:
                    break

                # Check if this edge also has the matching to value
                next_edge = html.find('{"color"', idx + 1)
                if next_edge == -1:
                    next_edge = len(html)

                edge_segment = html[idx : min(idx + 500, next_edge)]
                if to_pattern in edge_segment:
                    # Found the matching edge!
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

    # Write modified HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# Main
html_path = Path(__file__).parent / "topo_N-1_model_01.html"

n1_branches = [(5, 2), (1, 2), (2, 3), (5, 6)]
wind_buses = [5]
pv_buses = [4, 6]

print("Applying highlights to topo_N-1_model_01.html...")
highlight_html(html_path, n1_branches, wind_buses, pv_buses)
print("Done!")
