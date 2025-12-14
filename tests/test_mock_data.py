"""Quick test of mock data creation."""

import sys
import tempfile
from pathlib import Path
import numpy as np

# Import the function from verify_pipeline_case6.py
sys.path.insert(0, str(Path(__file__).parent))
from verify_pipeline_case6 import create_mock_data


def main():
    """Test mock data creation."""
    print("Testing mock data creation...")

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="test_mock_"))
    print(f"Temp dir: {temp_dir}")

    try:
        # Create mock data
        create_mock_data(temp_dir)

        # Check if files were created
        files = list(temp_dir.glob("*"))
        print(f"\nCreated files:")
        for f in files:
            print(f"  {f.name} ({f.stat().st_size} bytes)")

            # Try to load and check shape
            if f.suffix == ".npz":
                data = np.load(f)
                print(f"    Keys: {list(data.keys())}")
                for key in data.keys():
                    print(
                        f"      {key}: shape={data[key].shape}, dtype={data[key].dtype}"
                    )

        print(f"\n[OK] Mock data creation successful!")

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)
        print(f"\nCleaned up {temp_dir}")


if __name__ == "__main__":
    main()
