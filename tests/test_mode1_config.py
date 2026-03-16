"""
Step 1: Config and file discovery (mode 1).
Run from project root: pytest tests/test_mode1_config.py -v
"""
import os
import sys

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_path_2017_exists():
    """PATH_2017 must exist for EOS data."""
    from config.datasets_2017 import PATH_2017
    assert os.path.isdir(PATH_2017), f"PATH_2017 not found: {PATH_2017}"


def test_get_filesets_mode1():
    """get_filesets(full=False) returns one file per dataset."""
    from config.datasets_2017 import get_filesets
    filesets = get_filesets(full=False)
    assert filesets, "get_filesets(full=False) returned empty"
    for name, paths in filesets.items():
        assert len(paths) == 1, f"Mode 1 should have 1 file per dataset, got {len(paths)} for {name}"


def test_get_one_file_per_group():
    """get_one_file_per_group() returns one data and one background file."""
    from config.datasets_2017 import get_one_file_per_group
    one_per = get_one_file_per_group()
    assert one_per, "get_one_file_per_group() returned empty"
    assert "data" in one_per and one_per["data"], "No data file"
    assert "background" in one_per and one_per["background"], "No background file"
    assert len(one_per["data"]) == 1 and len(one_per["background"]) == 1
