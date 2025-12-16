"""
Quick test of dataset generation with 10 samples.
Verifies the pipeline works before running the full 12k generation.
"""

import sys
from pathlib import Path

# Modify constants in generate_dataset.py for testing
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "gcnn_opf_01"))

# Import and patch configuration
import generate_dataset as gen_module

# Override configuration for quick test
gen_module.N_TRAIN = 10
gen_module.N_TEST = 3
gen_module.CHECKPOINT_INTERVAL = 5
gen_module.OUTPUT_DIR = Path(__file__).parent / "data_test"

print("=" * 70)
print("QUICK TEST: Generating 10 train + 3 test samples")
print("=" * 70)

# Run the main generation
gen_module.main()

print("\n" + "=" * 70)
print("✓ TEST PASSED - Ready for full generation")
print("=" * 70)
