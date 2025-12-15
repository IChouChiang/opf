#!/usr/bin/env python
"""
Verification script for logging and progress infrastructure.
- Runs a short SSH-mode training.
- Verifies status file and CSV experiment log.
"""
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_FILE = ROOT / "current_status.txt"
CSV_LOG = ROOT / "experiments_log.csv"


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command and raise if it fails."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def assert_status_file() -> None:
    """Assert current_status.txt exists and contains an Epoch line."""
    if not STATUS_FILE.exists():
        raise AssertionError(f"Missing status file: {STATUS_FILE}")
    content = STATUS_FILE.read_text(encoding="utf-8")
    has_epoch = any(line.startswith("Epoch") for line in content.splitlines())
    if not has_epoch:
        raise AssertionError(
            "current_status.txt does not contain a line starting with 'Epoch'"
        )
    print("✓ current_status.txt contains epoch status")


def assert_csv_log() -> None:
    """Assert experiments_log.csv has a row for dnn on case6ww."""
    if not CSV_LOG.exists():
        raise AssertionError(f"Missing CSV log: {CSV_LOG}")
    with CSV_LOG.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") == "dnn" and row.get("dataset") == "case6ww":
                print("✓ experiments_log.csv contains dnn on case6ww")
                return
    raise AssertionError(
        "experiments_log.csv is missing the expected dnn/case6ww entry"
    )


def cleanup() -> None:
    """Remove transient status file; keep CSV log for history."""
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()
        print("Cleaned up current_status.txt")


def main() -> None:
    # Pre-clean status file to avoid stale content
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "data=case6",
        "model=dnn",
        "train.max_epochs=2",
        "train.mode=ssh",
        "+train.fast_dev_run=False",
        "train.accelerator=cpu",
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
    ]

    run_command(cmd)
    assert_status_file()
    assert_csv_log()
    cleanup()
    print("\n✅ MLOps Infrastructure Verified")


if __name__ == "__main__":
    main()
