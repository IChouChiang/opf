# Power System Topology Visualization

Independent tool for visualizing PYPOWER/MATPOWER case topologies with NetworkX.

## Purpose

Load any PYPOWER baseline case (case9, case39, case57, etc.) and visualize:
- Bus connectivity (nodes)
- Branch/transmission lines (edges)
- Generator locations
- Bus indices and labels
- Network topology layout

## Features

- **Interactive visualization** using matplotlib
- **Node coloring** by bus type (slack, PV, PQ)
- **Generator markers** at generation buses
- **Edge labels** showing branch indices
- **Bus indices** clearly visible
- **Network statistics** summary

## Dependencies

Requires `opf311` conda environment with:
- `pypower` — baseline cases
- `networkx` — graph construction
- `matplotlib` — visualization
- `numpy` — data processing

## Usage

**For SSH/Remote Sessions (default):**
```python
from pypower.api import case39
from topology_viz import visualize_case

# Load and visualize (saves to PNG file)
ppc = case39()
G, pos = visualize_case(
    ppc, 
    show_indices=True, 
    show_generators=True,
    save_path="my_topology.png",  # Specify output file
    show_plot=False  # Don't attempt to display (SSH session)
)
# Check 'my_topology.png' for the result
```

**For Local Sessions (with display):**
```python
from pypower.api import case39
from topology_viz import visualize_case

# Load and visualize (shows interactive window)
ppc = case39()
visualize_case(ppc, show_indices=True, show_generators=True, show_plot=True)
```

**Quick test:**
```bash
conda activate opf311
cd viz_topology
python test_viz.py          # Generates test_case39_topology.png
python examples.py          # Generates case39_topology.png
```

## File Structure

```
viz_topology/
├── README.md              # This file
├── topology_viz.py        # Main visualization module
├── examples.py            # Usage examples for different cases
└── test_viz.py            # Test script
```

## Quick Start

```bash
conda activate opf311
cd viz_topology

# Generate visualizations (saved as PNG files)
python test_viz.py          # Test with case39
python examples.py          # Multiple case examples

# Or run specific examples
python -c "from examples import example_case9; example_case9()"
python -c "from examples import example_case57; example_case57()"
```

Generated files will be saved as PNG images in the current directory.

## SSH/Remote Usage Note

When working over SSH (like VS Code Remote), the code is configured to:
- Use `matplotlib.use('Agg')` backend (no display required)
- Automatically save figures to PNG files
- Set `show_plot=False` by default

To view results:
1. Run the visualization script
2. Download/view the generated PNG file from the viz_topology directory
3. Or use VS Code's image preview feature

## Output

- Displays network graph with labeled buses and branches
- Color-coded nodes: 
  - 🔴 Red: Slack bus (reference)
  - 🟢 Green: PV bus (generator)
  - 🔵 Blue: PQ bus (load)
- Generator symbols at generation buses
- Branch indices on edges

---

**Note:** This is an independent visualization tool, separate from the AC-OPF optimization workflow in Week4.
