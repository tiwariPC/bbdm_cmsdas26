"""
Dataset config and file discovery for 2017.

All 2017 files (data, background, and later signal) live under:
  BASE_PATH / <year> / <dataset_dir> / *.root
  e.g. .../cmsdas2026/2017/MET-Run2017B-.../

Dataset dirs are discovered by listing subdirs of PATH_2017 (dynamic fallback
when full YAML globs match no files — e.g. use lxplus/eos access).
"""

import os
import re
import random
import glob
from pathlib import Path
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# EOS area root (do *not* include the year twice — PATH_2017 is BASE_PATH/year)
BASE_PATH = "/eos/cms/store/group/phys_susy/sus-23-008/cmsdas2026"
YEAR = 2017
PATH_2017 = os.path.join(BASE_PATH, str(YEAR))

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
                  ... etc. Other MC subdirs are grouped by first token (e.g. DY* -> DYJets).
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

# Fallback trigger list if YAML does not define triggers (e.g. older configs)
DEFAULT_TRIGGERS = [
    "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
    "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
    "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
    "HLT_Ele27_WPTight_Gsf",
    "HLT_Ele32_WPTight_Gsf_L1DoubleEG",
    "HLT_Ele32_WPTight_Gsf",
    "HLT_Ele35_WPTight_Gsf",
    "HLT_IsoMu24",
    "HLT_IsoMu27",
    "HLT_IsoTkMu27",
    "HLT_IsoTkMu24",
    "HLT_Photon200",
]


def get_trigger_list(prefer_full: bool = True) -> List[str]:
    """
    Return the analysis HLT path list from YAML config.

    - prefer_full=True: read from datasets_2017_full.yaml first, else short.
    - If neither YAML has a 'triggers' key, return DEFAULT_TRIGGERS.
    """
    for path in ([FULL_YAML, SHORT_YAML] if prefer_full else [SHORT_YAML, FULL_YAML]):
        cfg = _load_yaml(path)
        triggers = cfg.get("triggers")
        if triggers and isinstance(triggers, list):
            return [str(t) for t in triggers]
    return list(DEFAULT_TRIGGERS)


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _expand_short_entry_files(entry: Dict[str, Any], default_nfiles: int = 1) -> List[str]:
    """
    Expand a short-YAML entry into one or more concrete ROOT files.

    Supports (in priority order):
      - files: [path_or_glob, ...]
      - file: path_or_glob
    and optional:
      - nfiles: int (max number of concrete files to keep after expansion)
    """
    patterns: List[str] = []
    if isinstance(entry.get("files"), list):
        patterns.extend([str(p) for p in entry.get("files", []) if p])
    elif entry.get("file"):
        patterns.append(str(entry.get("file")))

    nfiles = int(entry.get("nfiles", default_nfiles))
    out: List[str] = []
    for p in patterns:
        if any(ch in p for ch in ["*", "?", "["]):
            out.extend(sorted(glob.glob(p, recursive=True)))
        else:
            out.append(p)

    # De-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    if nfiles > 0:
        uniq = uniq[:nfiles]
    return uniq


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
            files.extend(_expand_short_entry_files(entry, default_nfiles=1))
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
    data_entries = cfg.get("data", []) or []
    has_singleelectron = any(str(e.get("name", "")).startswith("SingleElectron_Run2017") for e in data_entries)
    met_entries = [e for e in data_entries if str(e.get("name", "")).startswith("MET_Run2017")]
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
                "masspoint_xsec_pb": entry.get("masspoint_xsec_pb", {}),
                "sumw": entry.get("sumw"),
                "year": entry.get("year"),
                "isData": entry.get("isData", False),
                "label": entry.get("label", name),
                # Optional: merged pickle group (see merge_groups / merge_prefix_rules in YAML)
                "merge_as": entry.get("merge_as"),
            }
    # Backward-compatible convenience: if only MET data was configured, also expose
    # SingleElectron data with mirrored Run2017 file globs (works for combined and per-era names).
    if (not has_singleelectron) and met_entries:
        for met_entry in met_entries:
            met_name = str(met_entry.get("name", ""))
            suffix = met_name.replace("MET_Run2017", "")
            se_name = f"SingleElectron_Run2017{suffix}"
            met_files = met_entry.get("files", []) or []
            se_files = [f.replace("MET-Run2017", "SingleElectron-Run2017") for f in met_files]
            out[se_name] = {
                "group": "data",
                "files": se_files,
                "xsec": None,
                "masspoint_xsec_pb": {},
                "sumw": None,
                "year": met_entry.get("year", 2017),
                "isData": True,
                "label": f"Data SingleElectron 2017{suffix}",
                "merge_as": met_entry.get("merge_as"),
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
            out["data"] = [random.choice(live["data"])]
        # background: pick one random background file from any background entry
        bg_files = live.get("backgrounds", [])
        if bg_files:
            out["background"] = [random.choice(bg_files)]
        if out:
            return out
    return get_one_file_per_group()


def get_short_fileset_and_labels() -> tuple:
    """
    Load one-file-per-dataset from short YAML for Session 3.
    Returns (fileset, labels):
      - fileset: dict name -> [path] for "data" and each background (and optionally signal).
      - labels: dict name -> display label for plots/legends (from YAML or built-in).
    """
    cfg = _load_yaml(SHORT_YAML)
    default_labels = {
        "DYJets": "Z(ll)+jets ",
        "ZJets": "Z(#nu#nu)+jets ",
        "WJets": "W(l#nu)+jets ",
        "DIBOSON": "WW/WZ/ZZ ",
        "STop": "Single t ",
        "Top": "t#bar{t} ",
        "SMH": "SMH ",
    }
    fileset: Dict[str, List[str]] = {}
    labels: Dict[str, str] = {}
    for group in ("data", "backgrounds", "signal"):
        entries = cfg.get(group, []) or []
        for entry in entries:
            name = entry.get("name")
            files = _expand_short_entry_files(entry, default_nfiles=1)
            if not name or not files:
                continue
            fileset[name] = files
            labels[name] = entry.get("label") or default_labels.get(name, name)
    return fileset, labels


def get_short_datasets_meta() -> Dict[str, Dict[str, Any]]:
    """
    Session 3 helper: load short YAML and return per-dataset metadata.

    Returns dict: name -> {files: [path], label: str, xsec: float|None, isData: bool}
    """
    cfg = _load_yaml(SHORT_YAML)
    default_labels = {
        "DYJets": "Z(ll)+jets ",
        "ZJets": "Z(#nu#nu)+jets ",
        "WJets": "W(l#nu)+jets ",
        "DIBOSON": "WW/WZ/ZZ ",
        "STop": "Single t ",
        "Top": "t#bar{t} ",
        "SMH": "SMH ",
    }
    out: Dict[str, Dict[str, Any]] = {}
    for group in ("data", "backgrounds", "signal"):
        entries = cfg.get(group, []) or []
        for entry in entries:
            name = entry.get("name")
            files = _expand_short_entry_files(entry, default_nfiles=1)
            if not name or not files:
                continue
            out[name] = {
                "files": files,
                "label": entry.get("label") or default_labels.get(name, name),
                "xsec": entry.get("xsec"),
                "isData": bool(entry.get("isData", False)),
            }
    return out


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
            files.extend(sorted(glob.glob(pattern, recursive=True)))
        if files:
            filesets[name] = files
    if not filesets:
        return get_filesets(full=True)
    return filesets


MASSPOINT_BRANCH_RE = re.compile(r"^GenModel_MH3_(\d+)_MH4_(\d+)_Mchi_(\d+)$")


def parse_signal_masspoint_branch(branch_name: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse branch like ``GenModel_MH3_600_MH4_350_Mchi_1`` -> (600, 350, 1).
    Returns None for non-matching names.
    """
    m = MASSPOINT_BRANCH_RE.match(str(branch_name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def signal_masspoint_key(mh3: int, mh4: int, mchi: int) -> str:
    """Canonical output key for per-masspoint signal accumulators."""
    return f"signal_mA{int(mh3)}_ma{int(mh4)}_mchi{int(mchi)}"


def is_signal_key(name: str) -> bool:
    """True for merged signal keys and per-masspoint signal keys."""
    n = str(name)
    return n == "signal" or n.startswith("signal_") or n.startswith("bbDM")


def discover_signal_masspoint_branches(
    files: Iterable[str],
    treename: str = "Events",
    target_mh3: int = 600,
    target_mchi: int = 1,
    max_files_to_scan: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Discover available signal masspoint branches from signal files.

    Returns
    -------
    list[tuple[str, str]]
        Sorted list of (output_key, branch_name), e.g.
        ``[("signal_mA600_ma200_mchi1", "GenModel_MH3_600_MH4_200_Mchi_1"), ...]``
    """
    try:
        import uproot
    except ImportError as exc:
        raise RuntimeError("uproot is required to discover signal masspoint branches") from exc

    discovered: Dict[int, str] = {}
    scanned = 0
    for path in files:
        if max_files_to_scan is not None and scanned >= max_files_to_scan:
            break
        scanned += 1
        try:
            with uproot.open(f"{path}:{treename}") as tree:
                keys = tree.keys()
        except Exception:
            continue
        for b in keys:
            parsed = parse_signal_masspoint_branch(str(b))
            if not parsed:
                continue
            mh3, mh4, mchi = parsed
            if mh3 == int(target_mh3) and mchi == int(target_mchi):
                discovered[mh4] = str(b)

    out: List[Tuple[str, str]] = []
    for mh4 in sorted(discovered):
        out.append((signal_masspoint_key(target_mh3, mh4, target_mchi), discovered[mh4]))
    return out


# --- Merged processor output (run_analysis.py) --------------------------------
#
# Fine-tune which YAML/EOS dataset names merge into which pickle key via YAML:
#   - merge_groups: { "DYJetsToLL_M50_HT100to200": "DYJets", ... }  (exact names)
#   - merge_prefix_rules: [ { prefix: "DYJetsToLL_M50_HT", group: "DYJets" }, ... ]
#     Longer prefixes are matched first.
#   - Per entry under data/backgrounds/signal: merge_as: DYJets
# See datasets_2017_full.yaml and datasets_2017_short.yaml for examples.

# Order for pretty-printing / stable pickle iteration (not required for correctness)
MERGED_GROUP_ORDER = (
    "data",
    "DYJets",
    "ZJets",
    "WJets",
    "Top",
    "STop",
    "DIBOSON",
    "SMH",
    "signal",
)


def _merged_sample_key_heuristic(dataset_name: str) -> str:
    """
    Map a full per-sample name (from YAML / EOS) to a short physics group when no YAML rule matches.

    Groups: data, DYJets, ZJets, WJets, Top, STop, DIBOSON, SMH, signal (bbDM), or unchanged fallback.
    """
    n = dataset_name
    if n.startswith("MET_") or "Run2017" in n or n.startswith("SingleElectron") or n.startswith("SingleMuon"):
        return "data"
    if n.startswith("DYJetsToLL") or n.startswith("DYJets"):
        return "DYJets"
    if n.startswith("ZJetsToNuNu"):
        return "ZJets"
    if n.startswith("WJetsToLNu"):
        return "WJets"
    if n.startswith("TT"):
        return "Top"
    if n.startswith("ST_"):
        return "STop"
    if n in ("WW", "WZ", "ZZ"):
        return "DIBOSON"
    # Keep split signal keys (signal_*) as-is; collapse only legacy bbDM dataset names.
    if n == "signal" or n.startswith("bbDM"):
        return "signal"
    if n.startswith(("ttH", "ggZH", "ZH_", "WminusH", "WplusH")):
        return "SMH"
    return n


@functools.lru_cache(maxsize=1)
def load_merge_config() -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Load merge mapping from datasets_2017_short.yaml then datasets_2017_full.yaml.

    Returns
    -------
    merge_map :
        Exact dataset name -> merged group (from ``merge_groups`` and per-entry ``merge_as``).
    prefix_rules :
        List of (prefix, group), sorted by descending prefix length (longest match wins).
    """
    merge_map: Dict[str, str] = {}
    # Last YAML file wins for the same prefix (SHORT then FULL → full overrides).
    prefix_by_prefix: Dict[str, str] = {}

    def _ingest_yaml(path: str) -> None:
        cfg = _load_yaml(path)
        if not cfg:
            return
        for k, v in (cfg.get("merge_groups") or {}).items():
            merge_map[str(k)] = str(v)
        for group in ("data", "backgrounds", "signal"):
            for entry in cfg.get(group, []) or []:
                name = entry.get("name")
                ma = entry.get("merge_as")
                if name and ma:
                    merge_map[str(name)] = str(ma)
        for rule in cfg.get("merge_prefix_rules") or []:
            p = rule.get("prefix")
            g = rule.get("group")
            if p and g:
                prefix_by_prefix[str(p)] = str(g)

    _ingest_yaml(SHORT_YAML)
    _ingest_yaml(FULL_YAML)
    prefix_rules = sorted(prefix_by_prefix.items(), key=lambda x: -len(x[0]))
    return merge_map, prefix_rules


def resolve_merge_key(
    dataset_name: str,
    merge_map: Optional[Dict[str, str]] = None,
    prefix_rules: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Resolve merged pickle key: exact map, then longest prefix rule, then heuristics.
    """
    mm = merge_map if merge_map is not None else {}
    pr = prefix_rules if prefix_rules is not None else []
    n = str(dataset_name)
    if n in mm:
        return mm[n]
    pr_sorted = sorted(pr, key=lambda x: -len(x[0]))
    for prefix, group in pr_sorted:
        if n.startswith(prefix):
            return group
    return _merged_sample_key_heuristic(n)


def merged_sample_key(dataset_name: str) -> str:
    """Same as resolve_merge_key with YAML config from load_merge_config()."""
    mm, pr = load_merge_config()
    return resolve_merge_key(dataset_name, mm, pr)


def merge_processor_results_by_group(
    results: Dict[str, Any],
    merge_map: Optional[Dict[str, str]] = None,
    prefix_rules: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Sum Coffea dict_accumulators from each per-dataset run into one entry per merged group.

    Parameters
    ----------
    results :
        Output of run_analysis loop: {full_dataset_name: dict_accumulator}.
    merge_map, prefix_rules :
        If both are None, values are loaded from YAML via load_merge_config().

    Returns
    -------
    dict
        {group_name: merged accumulator}, e.g. ``{"data": ..., "DYJets": ..., "ZJets": ...}``.
    """
    from collections import defaultdict

    if not results:
        return {}

    if merge_map is None and prefix_rules is None:
        merge_map, prefix_rules = load_merge_config()
    else:
        if merge_map is None:
            merge_map = {}
        if prefix_rules is None:
            prefix_rules = []

    buckets: Dict[str, List[Any]] = defaultdict(list)
    for name, acc in results.items():
        buckets[resolve_merge_key(str(name), merge_map, prefix_rules)].append(acc)

    merged: Dict[str, Any] = {}
    for key, acc_list in buckets.items():
        total = acc_list[0]
        for a in acc_list[1:]:
            total = total + a
        merged[key] = total

    # Stable key order: known groups first, then any other merged keys
    ordered: Dict[str, Any] = {}
    for k in MERGED_GROUP_ORDER:
        if k in merged:
            ordered[k] = merged[k]
    for k in sorted(merged.keys()):
        if k not in ordered:
            ordered[k] = merged[k]
    return ordered


def is_merged_results_format(results: Dict[str, Any]) -> bool:
    """True if pickle looks like merge_processor_results_by_group output (short group keys)."""
    if not results:
        return False
    keys = set(results.keys())
    # Legacy pickles use long YAML names like DYJetsToLL_M50_HT100to200
    if any(k.startswith("DYJetsToLL_") or k.startswith("ZJetsToNuNu_") for k in keys):
        return False
    return bool(keys & set(MERGED_GROUP_ORDER))


def data_and_bkg_keys(results: Dict[str, Any]) -> tuple:
    """
    Return (data_key_list, bkg_key_list) for loading merged or legacy processor pickles.

    Merged format: data -> ``[\"data\"]``, backgrounds exclude ``signal`` unless you add it by hand.
    Legacy: discover MET/Run2017/Single* as data, rest as backgrounds.
    """
    if isinstance(results, dict) and isinstance(results.get("samples"), dict):
        results = results["samples"]

    keys = list(results.keys())
    if is_merged_results_format(results):
        data_keys = [k for k in ("data", "data_MET", "data_SingleElectron", "MET", "SingleElectron") if k in results]
        if not data_keys:
            data_keys = ["data"]
        bkg_keys = [k for k in keys if (k not in data_keys) and (not is_signal_key(k))]
        return data_keys, bkg_keys
    data_keys = [
        d
        for d in keys
        if ("Run2017" in d) or ("Single" in d) or d.startswith("MET_") or d.startswith("data_")
    ]
    bkg_keys = [d for d in keys if (d not in data_keys) and (not is_signal_key(d))]
    return data_keys, bkg_keys
