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

### Static Visualization (PNG)
- **Interactive visualization** using matplotlib
- **Node coloring** by bus type (slack, PV, PQ)
- **Generator markers** at generation buses
- **Edge labels** showing branch indices
- **Bus indices** clearly visible
- **Network statistics** summary

### Interactive Visualization (HTML)
- **Click nodes** to highlight connected edges
- **Drag nodes** to reposition dynamically
- **Hover tooltips** with detailed bus/branch information
- **Zoom and pan** controls
- **Physics simulation** with multiple layout algorithms
- **Navigation buttons** for easy control
- **Works in any browser** (no installation needed)

## Dependencies

Requires `opf311` conda environment with:
- `pypower` — baseline cases
- `networkx` — graph construction (static viz)
- `matplotlib` — visualization (static viz)
- `pyvis` — interactive HTML visualization
- `numpy` — data processing

## Usage

### Static Visualization (PNG files)

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

### Interactive Visualization (HTML files)

**Generate interactive HTML (works everywhere - local or SSH):**
```python
from pypower.api import case39
from interactive_viz import create_interactive_topology

# Generate interactive HTML file
ppc = case39()
create_interactive_topology(
    ppc,
    output_file="case39_interactive.html",
    physics=True,  # Enable physics simulation
    layout="barnes_hut",  # Layout algorithm
    height="800px"
)
# Open case39_interactive.html in your browser
```

**Quick test:**
```bash
conda activate opf311
cd viz_topology

# Static PNG visualization
python test_viz.py          # Generates test_case39_topology.png
python examples.py          # Generates case39_topology.png

# Interactive HTML visualization
python test_interactive.py  # Generates case39_interactive.html
python examples_interactive.py  # Generates case39_interactive.html
```

## File Structure

```
viz_topology/
├── README.md                    # This file
├── topology_viz.py              # Static visualization (PNG/matplotlib)
├── interactive_viz.py           # Interactive visualization (HTML/PyVis)
├── examples.py                  # Static visualization examples
├── examples_interactive.py      # Interactive visualization examples
├── test_viz.py                  # Static test script
├── test_interactive.py          # Interactive test script
└── .gitignore                   # Excludes generated files
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

### Static PNG
- Displays network graph with labeled buses and branches
- Color-coded nodes: 
  - 🔴 Red: Slack bus (reference)
  - 🟢 Green: PV bus (generator)
  - 🔵 Blue: PQ bus (load)
- Generator symbols at generation buses
- Branch indices on edges

### Interactive HTML
- Fully interactive network in browser
- Click any bus to highlight all connected branches
- Drag buses to rearrange layout
- Hover over buses/branches for detailed tooltips:
  - Bus info: type, voltage, angle, load, limits
  - Generator info: Pg, Qg, Pmax, Pmin
  - Branch info: impedance, rating, from/to buses
- Physics-based layout with real-time simulation
- Navigation controls (zoom, pan, fit view)
- Works offline - no internet connection needed

---

**Note:** This is an independent visualization tool, separate from the AC-OPF optimization workflow in Week4.
