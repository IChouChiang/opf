# OPF Topology Visualization (Next.js)

A modern interactive tool for N-k contingency analysis and topology visualization.

## Features

- **Interactive Graph**: Visualize power system topology (Buses & Branches).
- **Contingency Simulation**: Click on transmission lines to "cut" them (simulate N-1, N-k outages).
- **Real-time Connectivity Check**: Automatically detects if the system splits into islands.
- **Status Dashboard**: Track number of cuts and system health.

## Setup

1.  **Install Dependencies**:
    ```bash
    npm install
    ```

2.  **Generate Data**:
    The app loads data from `public/data/`. Use the Python script to export your PYPOWER cases:
    ```bash
    # From the root of the repo
    python viz_topology/export_to_json.py
    ```

3.  **Run Development Server**:
    ```bash
    npm run dev
    ```
    Open [http://localhost:3000](http://localhost:3000) in your browser.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Visualization**: react-force-graph-2d
- **Icons**: Lucide React
