"""
Dataset config and file discovery for 2017.

All 2017 files (data, background, and later signal) live under:
  BASE_PATH / 2017 / <dataset_dir> / *.root

Dataset dirs are discovered by listing subdirs of PATH_2017.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

BASE_PATH = "/eos/cms/store/group/phys_susy/sus-23-008/cmsdas2026/2017"
YEAR = 2017
PATH_2017 = os.path.join(BASE_PATH, "2017")

# Optional: group dataset dirs into categories for plotting.
# Keys are short names (e.g. "data", "ttbar"); values are list of subdir name prefixes or full names.
# If empty, all subdirs are treated as separate datasets (key = subdir name).
DATA_PREFIXES = ("MET-", "SingleElectron-", "SingleMuon-")  # subdirs starting with these → "data"
# Backgrounds: all other subdirs. Signal: add later to SIGNAL_PREFIXES or SIGNAL_DIRS.
SIGNAL_PREFIXES = ()  # placeholder for when signal is added


def _list_dataset_dirs():
    """List subdirs of PATH_2017 that contain .root files (one level only)."""
    if not os.path.isdir(PATH_2017):
        return []
    out = []
    for name in sorted(os.listdir(PATH_2017)):
        subdir = os.path.join(PATH_2017, name)
        if os.path.isdir(subdir):
            # Check that it has at least one .root file
            for _ in Path(subdir).glob("*.root"):
                out.append(name)
                break
    return out


def _files_for_subdir(subdir_name, limit=None):
    """Return list of full paths to .root files in PATH_2017/subdir_name. Optionally limit count."""
    subdir = os.path.join(PATH_2017, subdir_name)
    files = sorted(Path(subdir).glob("*.root"))
    files = [os.path.join(subdir, f.name) for f in files]
    if limit is not None and limit > 0:
        files = files[:limit]
    return files


def get_filesets(full=False, max_files_per_dataset=None):
    """
    Build fileset dict: dataset_name -> list of file paths.

    - full=False: one file per dataset (for quick runs and plots).
    - full=True: all files per dataset (for full analysis).
    - max_files_per_dataset: if set, cap each dataset to this many files (useful for testing).
    """
    dataset_dirs = _list_dataset_dirs()
    if not dataset_dirs:
        return {}

    limit = 1 if not full else None
    if max_files_per_dataset is not None:
        if limit is None:
            limit = max_files_per_dataset
        else:
            limit = min(limit, max_files_per_dataset)

    filesets = {}
    for subdir_name in dataset_dirs:
        files = _files_for_subdir(subdir_name, limit=limit)
        if files:
            filesets[subdir_name] = files
    return filesets


def get_filesets_grouped(full=False, max_files_per_dataset=None):
    """
    Like get_filesets, but group into "data" and background groups.

    Returns dict: "data" -> [files from MET-*, SingleElectron-*, SingleMuon-*],
                  "ttbar" -> [files from TT*],
                  "Zvv" -> [files from ZJetsToNuNu*],
                  "Wjets" -> [files from WJets*],
                  ... etc. Other MC subdirs are grouped by first token (e.g. QCD_*, DY* -> DYJets).
    """
    raw = get_filesets(full=full, max_files_per_dataset=max_files_per_dataset)
    if not raw:
        return {}

    grouped = {}
    for subdir_name, files in raw.items():
        if any(subdir_name.startswith(p) for p in DATA_PREFIXES):
            key = "data"
        else:
            # Short name: e.g. TTToSemiLeptonic_... -> ttbar, ZJetsToNuNu_... -> Zvv, WJetsToLNu_... -> Wjets
            if subdir_name.startswith("TT"):
                key = "ttbar"
            elif subdir_name.startswith("ZJetsToNuNu"):
                key = "Zvv"
            elif subdir_name.startswith("WJetsToLNu"):
                key = "Wjets"
            elif subdir_name.startswith("DYJets"):
                key = "DY"
            elif subdir_name.startswith("QCD"):
                key = "QCD"
            elif subdir_name.startswith("ST_"):
                key = "single_top"
            else:
                key = subdir_name.split("-")[0].split("_")[0] if "_" in subdir_name else subdir_name[:20]
        if key not in grouped:
            grouped[key] = []
        grouped[key].extend(files)
    return grouped


def get_one_file_per_group(full=False):
    """
    For one-file / mode 1: return one file chosen at random from data and one from background.
    Returns {"data": [path], "background": [path]} for notebooks.
    """
    grouped = get_filesets_grouped(full=False, max_files_per_dataset=1)
    if not grouped:
        return {}
    out = {}
    if "data" in grouped and grouped["data"]:
        out["data"] = [random.choice(grouped["data"])]
    # One background file chosen at random from any background dataset
    bg_files = [f for k in grouped if k != "data" for f in grouped[k]]
    if bg_files:
        out["background"] = [random.choice(bg_files)]
    return out


# YAML-based configuration ----------------------------------------------------

SHORT_YAML = os.path.join(os.path.dirname(__file__), "datasets_2017_short.yaml")
FULL_YAML = os.path.join(os.path.dirname(__file__), "datasets_2017_full.yaml")


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_live_datasets() -> Dict[str, List[str]]:
    """
    Load one-file-per-dataset config for the live exercise from YAML.

    Returns dict with keys "data", "backgrounds", "signal",
    each mapping to a list of file paths.
    """
    cfg = _load_yaml(SHORT_YAML)
    out: Dict[str, List[str]] = {}
    for group in ("data", "backgrounds", "signal"):
        entries = cfg.get(group, []) or []
        files: List[str] = []
        for entry in entries:
            path = entry.get("file")
            if path:
                files.append(path)
        if files:
            out[group] = files
    return out


def load_full_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Load full-analysis dataset config from YAML.

    Returns dict keyed by dataset name with fields:
      - files: list of path or glob patterns
      - xsec, sumw, year, isData, label
    """
    cfg = _load_yaml(FULL_YAML)
    out: Dict[str, Dict[str, Any]] = {}
    for group in ("data", "backgrounds", "signal"):
        entries = cfg.get(group, []) or []
        for entry in entries:
            name = entry.get("name")
            if not name:
                continue
            out[name] = {
                "group": group,
                "files": entry.get("files", []),
                "xsec": entry.get("xsec"),
                "sumw": entry.get("sumw"),
                "year": entry.get("year"),
                "isData": entry.get("isData", False),
                "label": entry.get("label", name),
            }
    return out


def get_one_file_per_group_from_yaml() -> Dict[str, List[str]]:
    """
    Convenience wrapper for notebooks: use live YAML if present,
    otherwise fall back to dynamic discovery.
    """
    live = load_live_datasets()
    if live:
        out: Dict[str, List[str]] = {}
        if "data" in live and live["data"]:
            out["data"] = [live["data"][0]]
        # background: pick first background file from any background entry
        bg_files = live.get("backgrounds", [])
        if bg_files:
            out["background"] = [bg_files[0]]
        if out:
            return out
    return get_one_file_per_group()


def get_full_filesets_from_yaml() -> Dict[str, List[str]]:
    """
    Build fileset dict dataset_name -> list of file paths from full YAML.
    If YAML is missing or empty, fall back to dynamic discovery.
    """
    import glob

    full = load_full_datasets()
    if not full:
        return get_filesets(full=True)
    filesets: Dict[str, List[str]] = {}
    for name, meta in full.items():
        files: List[str] = []
        for pattern in meta.get("files", []):
            files.extend(sorted(glob.glob(pattern)))
        if files:
            filesets[name] = files
    if not filesets:
        return get_filesets(full=True)
    return filesets
