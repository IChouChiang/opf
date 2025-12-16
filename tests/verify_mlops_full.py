#!/usr/bin/env python
"""
Full MLOps logging verification:
- Mock evaluation logging via deep_opf.utils.logger.log_evaluation_to_csv
- Verify experiments_log.csv updated with Evaluation row
- Run a fast training job and verify training row contains log_dir
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_LOG = ROOT / "experiments_log.csv"


def read_csv_rows() -> list[dict]:
    if not CSV_LOG.exists():
        return []
    with CSV_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_lines() -> list[str]:
    if not CSV_LOG.exists():
        return []
    return CSV_LOG.read_text(encoding="utf-8").splitlines()


def run_eval_logging() -> None:
    """Call log_evaluation_to_csv with dummy metrics using subprocess."""
    code = (
        "from pathlib import Path\n"
        "from deep_opf.utils.logger import log_evaluation_to_csv\n"
        "csv_path = Path('experiments_log.csv')\n"
        "metrics = {'R2_PG': 0.99, 'Pacc_PG': 98.5}\n"
        "log_evaluation_to_csv('dnn', 'case6ww', metrics, csv_path=csv_path)\n"
    )
    # Run in project root to write experiments_log.csv there
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Eval logging failed. STDERR:\n" + result.stderr)
        print("STDOUT:\n" + result.stdout)
        raise RuntimeError("log_evaluation_to_csv execution failed")


def assert_evaluation_row(before_rows: int) -> None:
    rows = read_csv_rows()
    lines = read_csv_lines()
    if len(rows) <= before_rows and not any("Evaluation" in l for l in lines):
        raise AssertionError("CSV did not gain a new row after evaluation logging")
    # Find any row flagged as Evaluation containing the expected metric
    for r in rows[before_rows:]:
        phase = r.get("phase") or r.get("stage") or ""
        if "Evaluation" in phase:
            pacc = r.get("Pacc_PG") or r.get("Pacc_VG") or ""
            if "98.5" in str(pacc) or any("98.5" in str(v) for v in r.values()):
                print("✓ Evaluation row logged with Pacc=98.5")
                return
    # Fallback to raw line search for mixed-header files
    for line in lines:
        if "Evaluation" in line and "98.50" in line:
            print("✓ Evaluation row logged with Pacc=98.5 (raw line)")
            return
    raise AssertionError("No Evaluation row with Pacc=98.5 found")


def run_training() -> None:
    """Run a minimal training to produce a training log row in CSV."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "data=case6",
        "model=dnn",
        "train.max_epochs=1",
        "train.mode=ssh",
        "+train.fast_dev_run=False",
        "train.accelerator=cpu",
        "hydra.job.chdir=False",
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print("Training failed. STDERR:\n" + result.stderr)
        print("STDOUT:\n" + result.stdout)
        raise RuntimeError("Training run failed")


def assert_training_row_has_log_dir() -> None:
    # Mixed CSV headers: search raw lines for a Training row that ends with a path
    lines = read_csv_lines()
    for line in lines:
        if "Training" in line and "," in line:
            tokens = line.split(",")
            candidate = tokens[-1].strip()
            if candidate:
                p = Path(candidate)
                if not p.is_absolute():
                    p = (ROOT / candidate).resolve()
                if p.exists():
                    print("✓ Training row contains log_dir path")
                    return
    raise AssertionError("No Training row with a valid path found in CSV")


def main() -> None:
    before = len(read_csv_rows())
    run_eval_logging()
    assert_evaluation_row(before)
    run_training()
    assert_training_row_has_log_dir()
    print("\n✅ Full MLOps (Train + Eval logging) Verified")


if __name__ == "__main__":
    main()
