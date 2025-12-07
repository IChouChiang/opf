import json
import sys
import os
import numpy as np
from pathlib import Path

# Add src to path to import pypower if needed, though pypower is installed in env
# But we need to import the specific case files if they are local, or use pypower.api
from pypower.api import case6ww, case9, case39, case57, case14

def export_case_to_json(ppc, name, output_dir):
    """
    Export PYPOWER case dict to a JSON file for visualization.
    """
    
    # Extract data
    bus = ppc['bus']
    branch = ppc['branch']
    gen = ppc['gen']
    baseMVA = ppc['baseMVA']
    
    # Prepare Nodes (Buses)
    nodes = []
    for i in range(bus.shape[0]):
        bus_id = int(bus[i, 0])
        bus_type = int(bus[i, 1])
        pd = float(bus[i, 2])
        qd = float(bus[i, 3])
        
        # Determine type label
        type_label = "PQ"
        if bus_type == 2:
            type_label = "PV"
        elif bus_type == 3:
            type_label = "Slack"
            
        nodes.append({
            "id": bus_id,
            "label": f"Bus {bus_id}",
            "type": type_label,
            "pd": pd,
            "qd": qd,
            # Add generation info if present (will be summed later if multiple gens)
            "pg": 0.0,
            "qg": 0.0
        })
        
    # Map bus ID to index for easier lookup if needed, but ID is usually fine for graph libs
    
    # Add Generator Info
    for i in range(gen.shape[0]):
        gen_bus = int(gen[i, 0])
        pg = float(gen[i, 1])
        qg = float(gen[i, 2])
        
        # Find the node
        for node in nodes:
            if node["id"] == gen_bus:
                node["pg"] += pg
                node["qg"] += qg
                if node["type"] == "PQ": # Should not happen for valid cases but good for safety
                    node["type"] = "PV" 
                break

    # Prepare Edges (Branches)
    edges = []
    for i in range(branch.shape[0]):
        f_bus = int(branch[i, 0])
        t_bus = int(branch[i, 1])
        r = float(branch[i, 2])
        x = float(branch[i, 3])
        b = float(branch[i, 4])
        rateA = float(branch[i, 5])
        
        edges.append({
            "id": i, # Branch index as ID
            "source": f_bus,
            "target": t_bus,
            "r": r,
            "x": x,
            "b": b,
            "rateA": rateA,
            "active": True # Default to active
        })

    data = {
        "name": name,
        "baseMVA": baseMVA,
        "nodes": nodes,
        "edges": edges
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{name}.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {name} to {output_path}")

def main():
    output_dir = Path("viz_nextjs/public/data")
    
    # Export cases
    export_case_to_json(case6ww(), "case6ww", output_dir)
    export_case_to_json(case9(), "case9", output_dir)
    export_case_to_json(case14(), "case14", output_dir)
    # export_case_to_json(case39(), "case39", output_dir) # Optional
    
if __name__ == "__main__":
    main()
