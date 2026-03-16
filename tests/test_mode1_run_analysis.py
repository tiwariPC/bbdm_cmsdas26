"""
Step 2: Processor run (mode 1) and pkl structure.
Requires: coffea, and 2017 data at PATH_2017.
Run from project root: pytest tests/test_mode1_run_analysis.py -v
"""
import os
import pickle
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(ROOT, "output_2017.pkl")


def _has_coffea():
    try:
        import coffea  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_coffea(), reason="coffea not installed")
def test_run_analysis_mode1():
    """Run run_analysis.py (no --full) and check exit code and output file."""
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    result = subprocess.run(
        [sys.executable, "run_analysis.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"run_analysis.py failed: {result.stderr}"
    assert "One-file mode" in result.stdout or "one file per dataset" in result.stdout.lower()
    assert os.path.isfile(PKL), f"Expected {PKL} to exist"


@pytest.mark.skipif(not _has_coffea(), reason="coffea not installed")
def test_pkl_structure():
    """Load output_2017.pkl and assert structure (met_sr, cutflow)."""
    if not os.path.isfile(PKL):
        pytest.skip("output_2017.pkl not found; run test_run_analysis_mode1 first")
    with open(PKL, "rb") as f:
        results = pickle.load(f)
    assert isinstance(results, dict), "results should be a dict"
    assert results, "results should be non-empty"
    for name, acc in results.items():
        assert "met_sr" in acc, f"Dataset {name} missing met_sr"
        assert "cutflow" in acc, f"Dataset {name} missing cutflow"
        # met_sr is a hist with axes
        assert hasattr(acc["met_sr"], "axes") or hasattr(acc["met_sr"], "axes"), (
            f"met_sr for {name} should be a hist"
        )
