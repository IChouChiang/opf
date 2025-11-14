# test_interactive.py
"""Test script for interactive PyVis topology visualization."""
from __future__ import annotations
from pypower.api import case9, case39
from interactive_viz import create_interactive_topology


def test_case9_interactive():
    """Generate interactive HTML for IEEE 9-bus system."""
    print("\nGenerating interactive visualization for IEEE 9-bus...")
    ppc = case9()
    net = create_interactive_topology(
        ppc,
        output_file="case9_interactive.html",
        physics=True,
        layout="barnes_hut",
        height="700px",
    )
    print("\n✓ Open 'case9_interactive.html' in your browser!\n")


def test_case39_interactive():
    """Generate interactive HTML for IEEE 39-bus system."""
    print("\nGenerating interactive visualization for IEEE 39-bus...")
    ppc = case39()
    net = create_interactive_topology(
        ppc,
        output_file="case39_interactive.html",
        physics=True,
        layout="barnes_hut",
        height="800px",
    )
    print("\n✓ Open 'case39_interactive.html' in your browser!\n")


def test_multiple_layouts():
    """Compare different physics layouts for case9."""
    print("\nGenerating multiple layouts for case9...")
    ppc = case9()
    
    layouts = ["barnes_hut", "force_atlas", "repulsion"]
    
    for layout in layouts:
        filename = f"case9_{layout}.html"
        print(f"\n  Creating {filename}...")
        create_interactive_topology(
            ppc,
            output_file=filename,
            physics=True,
            layout=layout,
            height="700px",
        )
    
    print("\n✓ Generated multiple layouts - compare in browser!\n")


if __name__ == "__main__":
    # Run basic test
    test_case39_interactive()
    
    # Uncomment to test other cases
    # test_case9_interactive()
    # test_multiple_layouts()
