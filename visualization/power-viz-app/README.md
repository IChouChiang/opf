# PYPOWER Visualization Tool

Interactive web-based visualization for PYPOWER case files with N-1 contingency analysis.

## Features

- 🗺️ **Interactive Network Topology** - Force-directed graph visualization
- 🎨 **Bus Type Differentiation** - Distinct icons for PQ, PV, and Slack buses
- ⚡ **N-1 Contingency Analysis** - Toggle branches to test network connectivity
- 🌐 **Island Detection** - Real-time identification of disconnected components
- 💎 **Modern UI** - Dark theme with glassmorphism effects

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.11 with PYPOWER installed (for data extraction)

### Installation

1. **Extract case data** (already done for case39):
   ```bash
   cd e:\DOCUMENT\Learn_Py\opf
   conda activate opf311
   python visualization\extract_case_data.py
   ```

2. **Install dependencies**:
   ```bash
   cd visualization\power-viz-app
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open browser**: http://localhost:3000

## Usage

### Viewing the Network

- The network topology displays automatically with all branches active
- **Blue circles** = PQ buses (load buses)
- **Green diamonds** = PV buses (generator buses)
- **Red stars** = Slack bus (reference bus)
- **Yellow dots** = Indicates bus has a generator

### N-1 Contingency Analysis

1. Use the **Branch Controller** (left panel) to toggle branches on/off
2. Watch the **Connectivity Indicator** (right panel) update in real-time
3. When network splits, disconnected islands are colored differently
4. Use **"Show All"** / **"Hide All"** for bulk operations

### Interactive Controls

- **Drag** nodes to rearrange the layout
- **Scroll** to zoom in/out
- **Pan** by dragging the background
- **Hover** over buses to see details

## Project Structure

```
visualization/
├── extract_case_data.py          # Python data extraction script
├── case39_data.json              # Extracted case data
└── power-viz-app/                # Next.js application
    ├── app/
    │   ├── components/           # React components
    │   ├── types/                # TypeScript types
    │   ├── utils/                # Utility functions
    │   └── page.tsx              # Main page
    └── public/
        └── case39_data.json      # Case data for client
```

## Technology Stack

- **Frontend**: Next.js 16, React, TypeScript
- **Styling**: Tailwind CSS with custom dark theme
- **Visualization**: D3.js v7 force-directed graph
- **Analysis**: BFS algorithm for connectivity detection

## Data Format

The tool works with PYPOWER case files. Currently configured for IEEE 39-bus system:
- 39 buses (29 PQ, 9 PV, 1 Slack)
- 46 branches (transmission lines)
- 10 generators

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## License

This project is part of the OPF research project.

## Credits

Built with Next.js, D3.js, and Tailwind CSS. Data from PYPOWER library.
