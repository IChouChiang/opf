"""
Helper script to launch the Experiment Dashboard.

Usage:
    python app/run_dashboard.py [--port PORT]
"""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent / "experiment_dashboard.py"

    # Parse simple args
    port = "8501"  # default
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 1 < len(sys.argv) - 1:
            port = sys.argv[i + 2]

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        port,
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"🚀 Launching dashboard on http://localhost:{port}")
    print(f"   Command: {' '.join(cmd[:5])} ...")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
