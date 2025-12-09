# extract_case_data.py
# Extract PYPOWER case data and export to JSON for web visualization

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Any
from pypower.api import case39


def numpy_to_python(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj


def extract_case_data(ppc: dict) -> dict:
    """
    Extract and structure PYPOWER case data for visualization.
    
    Bus Types:
        1: PQ bus (load bus)
        2: PV bus (generator bus, voltage controlled)
        3: Reference/Slack bus
        4: Isolated bus
    """
    # Column indices based on MATPOWER format
    BUS_I, BUS_TYPE, PD, QD = 0, 1, 2, 3
    BUS_GS, BUS_BS, BUS_AREA = 4, 5, 6
    VM, VA, BASE_KV, ZONE, VMAX, VMIN = 7, 8, 9, 10, 11, 12
    
    F_BUS, T_BUS, BR_R, BR_X, BR_B = 0, 1, 2, 3, 4
    RATE_A, RATE_B, RATE_C = 5, 6, 7
    TAP, SHIFT, BR_STATUS = 8, 9, 10
    
    GEN_BUS, PG, QG, QMAX, QMIN = 0, 1, 2, 3, 4
    VG, MBASE, GEN_STATUS = 5, 6, 7
    PMAX, PMIN = 8, 9
    
    bus_data = ppc['bus']
    branch_data = ppc['branch']
    gen_data = ppc['gen']
    base_mva = float(ppc['baseMVA'])
    
    # Extract bus information
    buses = []
    for i, bus_row in enumerate(bus_data):
        bus_info = {
            'id': int(bus_row[BUS_I]),
            'index': i,
            'type': int(bus_row[BUS_TYPE]),
            'type_name': {1: 'PQ', 2: 'PV', 3: 'Slack', 4: 'Isolated'}.get(int(bus_row[BUS_TYPE]), 'Unknown'),
            'pd': float(bus_row[PD]),  # MW
            'qd': float(bus_row[QD]),  # MVAr
            'vm': float(bus_row[VM]),  # p.u.
            'va': float(bus_row[VA]),  # degrees
            'baseKV': float(bus_row[BASE_KV]),
            'vmax': float(bus_row[VMAX]),
            'vmin': float(bus_row[VMIN]),
            'area': int(bus_row[BUS_AREA]),
        }
        buses.append(bus_info)
    
    # Extract branch information
    branches = []
    for i, branch_row in enumerate(branch_data):
        branch_info = {
            'id': i,
            'from_bus': int(branch_row[F_BUS]),
            'to_bus': int(branch_row[T_BUS]),
            'resistance': float(branch_row[BR_R]),  # p.u.
            'reactance': float(branch_row[BR_X]),  # p.u.
            'charging': float(branch_row[BR_B]),  # p.u.
            'rateA': float(branch_row[RATE_A]),  # MVA
            'rateB': float(branch_row[RATE_B]),  # MVA
            'rateC': float(branch_row[RATE_C]),  # MVA
            'tap': float(branch_row[TAP]),
            'shift': float(branch_row[SHIFT]),  # degrees
            'status': int(branch_row[BR_STATUS]),  # 1=in-service, 0=out
        }
        branches.append(branch_info)
    
    # Extract generator information
    generators = []
    for i, gen_row in enumerate(gen_data):
        gen_info = {
            'id': i,
            'bus': int(gen_row[GEN_BUS]),
            'pg': float(gen_row[PG]),  # MW
            'qg': float(gen_row[QG]),  # MVAr
            'qmax': float(gen_row[QMAX]),  # MVAr
            'qmin': float(gen_row[QMIN]),  # MVAr
            'vg': float(gen_row[VG]),  # p.u.
            'pmax': float(gen_row[PMAX]),  # MW
            'pmin': float(gen_row[PMIN]),  # MW
            'status': int(gen_row[GEN_STATUS]),
        }
        generators.append(gen_info)
    
    return {
        'baseMVA': base_mva,
        'name': 'IEEE 39-Bus System',
        'buses': buses,
        'branches': branches,
        'generators': generators,
    }


if __name__ == '__main__':
    # Load case39
    print("Loading case39 from PYPOWER...")
    ppc = case39()
    
    # Extract data
    print("Extracting data...")
    case_data = extract_case_data(ppc)
    
    # Create visualization folder
    viz_dir = Path(__file__).parent / 'visualization'
    viz_dir.mkdir(exist_ok=True)
    
    # Save to JSON
    output_file = viz_dir / 'case39_data.json'
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(case_data, f, indent=2)
    
    print(f"✓ Data extracted successfully!")
    print(f"  - {len(case_data['buses'])} buses")
    print(f"  - {len(case_data['branches'])} branches")
    print(f"  - {len(case_data['generators'])} generators")
    print(f"  - Base MVA: {case_data['baseMVA']}")
