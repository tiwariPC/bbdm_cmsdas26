#!/usr/bin/env python3
"""
Helper utilities to inspect and plot all histogram variables stored in a bbDM output pickle.

Notebook one-liner:
    import config.plot_pkl_variables as pp; pp.plot_all_variables_grid("output/output_2017_full.pkl")

Note:
    Processor outputs from ``run_analysis.py`` are expected to be already normalized
    (MC scaled to luminosity). This plotting helper does not apply any extra scaling.
"""

import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from hist import Hist
try:
    import mplhep as hep
except Exception:
    hep = None

PROCESS_COLORS = {
    "DYJets": "#3BAF2A",
    "ZJets": "#2A64AD",
    "WJets": "#8B4FC9",
    "DIBOSON": "#1F3FB3",
    "STop": "#F39C34",
    "Top": "#D97A00",
    "SMH": "#C62828",
    "QCD": "#6E6E6E",
}
DEFAULT_COLOR = "#9A9A9A"
TARGET_SIGNAL_KEYS = ("signal_mA600_ma250_mchi1", "signal_mA600_ma500_mchi1")
SR_HIST_KEYS = {"recoil", "cos_theta_star"}
# Relative normalization uncertainty used for the "systematics" band in plots.
SYSTEMATIC_REL_UNC = 0.10
DATA_KEY_CANDIDATES = (
    "data",
    "data_MET",
    "data_SingleElectron",
    "MET",
    "SingleElectron",
    "SingleMuon",
    "MET_Run2017",
    "SingleElectron_Run2017",
    "SingleMuon_Run2017",
)
REGION_ORDER = ("sr", "zecr", "zmucr", "tecr", "tmucr")


def _title_inside_axes(ax, text: str, loc: str = "upper left") -> None:
    """
    Show a title inside the plot area (axes coordinates) instead of matplotlib's axis title.
    Keeps labels clear: default is upper-left (inside panel).
    """
    if loc == "upper left":
        xy, ha, va = (0.02, 0.98), "left", "top"
    elif loc == "upper right":
        xy, ha, va = (0.98, 0.98), "right", "top"
    elif loc == "lower left":
        xy, ha, va = (0.02, 0.02), "left", "bottom"
    else:
        xy, ha, va = (0.5, 0.98), "center", "top"
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.92,
            linewidth=0.8,
        ),
        zorder=25,
    )


def _apply_plot_style() -> None:
    if hep is not None:
        hep.style.use("CMS")
    plt.rcParams.update(
        {
            "axes.linewidth": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "legend.frameon": False,
        }
    )


def _resolve_plot_dir(save_dir: Optional[str]) -> Path:
    """Resolve and create the output plot directory."""
    if save_dir:
        out = Path(save_dir).expanduser()
    else:
        out = Path("output/plots")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_plot_name(name: str) -> str:
    s = str(name).strip().replace(" ", "_")
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


def load_results(pkl_path: str) -> Dict:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return _flatten_schema_samples(raw)


def _flatten_schema_samples(raw: Dict) -> Dict:
    """
    Convert structured bundle format
      {"metadata": {...}, "samples": {...}}
    into legacy plotting shape:
      {sample_name: {"cutflow": ..., "<hist_key>": HistAccumulator, ...}, ...}
    """
    if not isinstance(raw, dict):
        return raw
    samples = raw.get("samples")
    if not isinstance(samples, dict):
        return raw

    flat: Dict = {}
    for name, sample in samples.items():
        if not isinstance(sample, dict):
            continue
        acc = {}
        cutflow = sample.get("cutflow")
        if isinstance(cutflow, dict):
            acc["cutflow"] = cutflow

        hbr = sample.get("hists_by_region", {})
        if isinstance(hbr, dict):
            for key, value in hbr.items():
                acc[f"{key}_by_region"] = value

        hists = sample.get("hists", {})
        if isinstance(hists, dict):
            for key, value in hists.items():
                acc[key] = value

        # Accept mixed payloads where hist keys may still be top-level.
        for key, value in sample.items():
            if key in ("cutflow", "hists_by_region", "hists", "metadata"):
                continue
            if _unwrap_hist(value) is not None and key not in acc:
                acc[key] = value
        flat[name] = acc
    return flat


def _unwrap_hist(obj):
    if isinstance(obj, Hist):
        return obj
    if hasattr(obj, "_hist"):
        return obj._hist
    return None


def list_histogram_keys(results: Dict, key_hint: Optional[str] = None):
    if not results:
        return []
    probe_key = key_hint if key_hint in results else next(iter(results.keys()))
    acc = results[probe_key]
    keys = [k for k, v in acc.items() if _unwrap_hist(v) is not None]
    # Prefer region-aware histograms if available.
    region_keys = [k[:-10] for k in keys if k.endswith("_by_region")]
    if region_keys:
        return sorted(set(region_keys))
    return keys


def _is_signal_key(name: str) -> bool:
    n = str(name)
    return n == "signal" or n.startswith("signal_") or n.startswith("bbDM")


def _is_data_key(name: str) -> bool:
    n = str(name)
    if n in DATA_KEY_CANDIDATES:
        return True
    if n.startswith("data_"):
        return True
    if n.startswith("MET"):
        return True
    if n.startswith("SingleElectron"):
        return True
    if n.startswith("SingleMuon"):
        return True
    return False


def _pick_names(results: Dict, datasets: Optional[Iterable[str]]) -> list:
    names = list(datasets) if datasets is not None else list(results.keys())
    return [n for n in names if n in results]


def _background_sort_key(name: str) -> tuple:
    order = {
        "DYJets": 0,
        "ZJets": 1,
        "WJets": 2,
        "Top": 3,
        "STop": 4,
        "DIBOSON": 5,
        "QCD": 6,
        "SMH": 7,
    }
    return (order.get(name, 99), str(name))


def _classify_datasets(results: Dict, names: Iterable[str]) -> tuple:
    name_list = list(names)
    data_keys = [n for n in name_list if _is_data_key(n)]
    signal_keys = [n for n in TARGET_SIGNAL_KEYS if n in name_list]
    bkg_keys = [n for n in name_list if (n not in data_keys) and (n not in signal_keys) and (not _is_signal_key(n)) and (not _is_data_key(n))]
    bkg_keys = sorted(bkg_keys, key=_background_sort_key)
    return data_keys, bkg_keys, signal_keys


def _data_keys_for_region(data_keys: Iterable[str], region: Optional[str]) -> list:
    keys = list(data_keys)
    def _first(prefixes):
        for p in prefixes:
            for k in keys:
                if str(k).startswith(p):
                    return k
        return None
    # Region-to-data mapping:
    # - sr, zmucr, tmucr -> MET data
    # - zecr, tecr -> SingleElectron data
    # Fall back to generic "data" if dedicated stream is unavailable.
    if region in ("sr", "zmucr", "tmucr"):
        k = _first(("data_MET", "MET_Run2017", "MET"))
        if k is not None:
            return [k]
        if "data" in keys:
            return ["data"]
        return []
    if region in ("zecr", "tecr"):
        k = _first(("data_SingleElectron", "SingleElectron_Run2017", "SingleElectron"))
        if k is not None:
            return [k]
        if "data" in keys:
            return ["data"]
        return []
    # Non-region/legacy: prefer generic data key if present, else MET.
    if "data" in keys:
        return ["data"]
    k = _first(("MET_Run2017", "MET"))
    if k is not None:
        return [k]
    return keys[:1]


def _extract_hist(results: Dict, dataset_name: str, hist_key: str):
    if dataset_name not in results:
        return None
    acc = results[dataset_name]
    region_key = f"{hist_key}_by_region"
    if region_key in acc:
        return _unwrap_hist(acc.get(region_key))
    if hist_key in acc:
        return _unwrap_hist(acc.get(hist_key))
    return None


def _region_axis_name(hist_obj: Hist) -> Optional[str]:
    for ax in hist_obj.axes:
        if getattr(ax, "name", None) == "region":
            return "region"
    return None


def _available_regions(results: Dict, names: Iterable[str], hist_key: str) -> list:
    regions = []
    for n in names:
        h = _extract_hist(results, n, hist_key)
        if h is None:
            continue
        if _region_axis_name(h) is None:
            return []
        try:
            cats = [str(x) for x in h.axes["region"]]
        except Exception:
            cats = []
        for r in cats:
            if r not in regions:
                regions.append(r)
    ordered = [r for r in REGION_ORDER if r in regions]
    extra = [r for r in regions if r not in ordered]
    return ordered + extra


def _check_edges_match(ref_edges: np.ndarray, test_edges: np.ndarray) -> bool:
    return len(ref_edges) == len(test_edges) and np.allclose(ref_edges, test_edges)


def _step_extend(y: np.ndarray) -> np.ndarray:
    """Convert per-bin values to post-step values with len(edges)=len(y)+1."""
    if len(y) == 0:
        return np.array([0.0], dtype=float)
    return np.r_[y, y[-1]]


def _draw_stacked_panel(
    ax,
    rax,
    results: Dict,
    hist_key: str,
    names: Iterable[str],
    region: Optional[str] = None,
    panel_title: Optional[str] = None,
) -> None:
    data_keys, bkg_keys, signal_keys = _classify_datasets(results, names)
    is_sr = (region == "sr") if region is not None else (hist_key in SR_HIST_KEYS)
    active_data_keys = [] if is_sr else _data_keys_for_region(data_keys, region)

    bkg_hists = []
    for name in bkg_keys:
        if _is_data_key(name):
            continue
        h = _extract_hist(results, name, hist_key)
        if h is not None:
            if region is not None:
                if _region_axis_name(h) is None:
                    # For region panels, never fall back to non-region histograms.
                    continue
                try:
                    h = h[{"region": region}]
                except Exception:
                    continue
            bkg_hists.append((name, h))

    if not bkg_hists:
        ax.text(0.5, 0.5, f"Missing: {hist_key}", ha="center", va="center")
        ax.set_axis_off()
        rax.set_axis_off()
        return

    ref_edges = bkg_hists[0][1].axes[0].edges
    centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])
    widths = np.diff(ref_edges)
    bottom = np.zeros_like(centers, dtype=float)
    pred_var = np.zeros_like(centers, dtype=float)

    for name, h in bkg_hists:
        edges = h.axes[0].edges
        if not _check_edges_match(ref_edges, edges):
            continue
        vals = np.asarray(h.values(), dtype=float)
        vars_h = h.variances()
        if vars_h is not None:
            pred_var = pred_var + np.clip(np.asarray(vars_h, dtype=float), 0.0, None)
        else:
            pred_var = pred_var + np.clip(vals, 0.0, None)
        ax.bar(
            centers,
            vals,
            width=widths,
            bottom=bottom,
            align="center",
            color=PROCESS_COLORS.get(name, DEFAULT_COLOR),
            edgecolor="none",
            linewidth=0.0,
            alpha=0.9,
            label=name,
        )
        bottom = bottom + vals

    pred = np.asarray(bottom, dtype=float)
    stat_unc = np.sqrt(np.clip(pred_var, 0.0, None))
    syst_unc = SYSTEMATIC_REL_UNC * np.clip(pred, 0.0, None)
    total_unc = np.sqrt(stat_unc**2 + syst_unc**2)

    # Main-pad uncertainty bands.
    tot_lo = np.clip(pred - total_unc, 0.0, None)
    tot_hi = pred + total_unc
    ax.fill_between(
        ref_edges,
        _step_extend(tot_lo),
        _step_extend(tot_hi),
        step="post",
        facecolor="none",
        edgecolor="#b22222",
        hatch="xx",
        linewidth=0.0,
        alpha=0.7,
        label="_nolegend_",
        zorder=6,
    )

    # Data overlay:
    # - CRs: use configured data stream (MET or SingleElectron)
    # - SR: use background sum as pseudo-data for a consistent ratio panel.
    data_sum = None
    if active_data_keys:
        data_sum = None
        for name in active_data_keys:
            h = _extract_hist(results, name, hist_key)
            if h is None:
                continue
            if region is not None:
                if _region_axis_name(h) is None:
                    continue
                try:
                    h = h[{"region": region}]
                except Exception:
                    continue
            if not _check_edges_match(ref_edges, h.axes[0].edges):
                continue
            data_sum = h if data_sum is None else (data_sum + h)
    if data_sum is not None:
        dvals = np.asarray(data_sum.values(), dtype=float)
        dvars = data_sum.variances()
        derr = (
            np.sqrt(np.clip(np.asarray(dvars, dtype=float), 0.0, None))
            if dvars is not None
            else np.sqrt(np.clip(dvals, 0.0, None))
        )
        dlabel = "Data"
    else:
        dvals = np.asarray(bottom, dtype=float).copy()
        derr = np.sqrt(np.clip(dvals, 0.0, None))
        dlabel = "Pseudo-data (MC sum)" if is_sr else "Data"
    ax.errorbar(
        centers,
        dvals,
        yerr=derr,
        fmt="o",
        color="black",
        markersize=3.5,
        linewidth=1.0,
        label=dlabel,
        zorder=10,
    )

    # Signal is shown only in SR and only for the requested two signal keys.
    if is_sr:
        sig_styles = {
            "signal_mA600_ma250_mchi1": dict(color="#E41A1C", linestyle="--"),
            "signal_mA600_ma500_mchi1": dict(color="#000000", linestyle="-"),
        }
        for skey in signal_keys:
            h = _extract_hist(results, skey, hist_key)
            if h is None:
                continue
            if region is not None:
                if _region_axis_name(h) is None:
                    continue
                try:
                    h = h[{"region": region}]
                except Exception:
                    continue
            if not _check_edges_match(ref_edges, h.axes[0].edges):
                continue
            svals = np.asarray(h.values(), dtype=float)
            style = sig_styles.get(skey, dict(color="#E41A1C", linestyle="--"))
            ax.step(ref_edges[:-1], svals, where="post", linewidth=2.2, label=skey, **style)

    ttl = (
        panel_title
        if panel_title is not None
        else (f"{hist_key} [{region}]" if region is not None else hist_key)
    )
    _title_inside_axes(ax, ttl, loc="upper left")
    if hep is not None:
        # Treat only real CR overlays as data; SR pseudo-data stays simulation-like.
        has_real_data = data_sum is not None
        hep.cms.label(" ", data=has_real_data, lumi=41.5, year=2017, ax=ax)
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(1e-3, ymin), max(1.0, ymax) * 100.0)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", ncol=2, fontsize=10.5, frameon=False, columnspacing=1.0, handlelength=1.6)

    # Ratio panel: Data/Pred.
    ratio = np.divide(dvals, pred, out=np.full_like(dvals, np.nan, dtype=float), where=pred > 0.0)
    rerr = np.divide(derr, pred, out=np.full_like(derr, np.nan, dtype=float), where=pred > 0.0)

    total_ratio = np.divide(total_unc, pred, out=np.full_like(total_unc, np.nan, dtype=float), where=pred > 0.0)
    rax.fill_between(
        ref_edges,
        _step_extend(1.0 - total_ratio),
        _step_extend(1.0 + total_ratio),
        step="post",
        facecolor="none",
        edgecolor="#b22222",
        hatch="xx",
        linewidth=0.0,
        alpha=0.7,
        label="Stat. + Syst. unc.",
        zorder=2,
    )
    rax.errorbar(centers, ratio, yerr=rerr, fmt="o", color="black", markersize=3.5, linewidth=1.0)
    rax.axhline(1.0, color="black", linewidth=1.0, alpha=0.8)
    rax.set_ylim(0.4, 1.6)
    rax.set_ylabel("Data/Pred.")
    rax.set_xlabel(hist_key)
    rax.grid(alpha=0.18, axis="y")
    rax.legend(loc="upper right", fontsize=9.5, frameon=False, handlelength=1.6)


def sum_histogram(results: Dict, hist_key: str, datasets: Optional[Iterable[str]] = None):
    names = list(datasets) if datasets is not None else list(results.keys())
    total = None
    for name in names:
        if name not in results:
            continue
        h = _unwrap_hist(results[name].get(hist_key))
        if h is None:
            continue
        total = h.copy() if total is None else (total + h)
    return total


def get_cutflow(results: Dict, datasets: Optional[Iterable[str]] = None):
    names = list(datasets) if datasets is not None else list(results.keys())
    merged = {}
    for name in names:
        if name not in results:
            continue
        cf = results[name].get("cutflow", {})
        for cut, value in cf.items():
            merged[cut] = merged.get(cut, 0) + int(value)
    return merged


def print_cutflow(results: Dict, datasets: Optional[Iterable[str]] = None):
    cf = get_cutflow(results, datasets=datasets)
    preferred_order = [
        "all_events",
        "masspoint",
        "trigger",
        "njet_ge1",
        "nbjet_ge2",
        "lepton_veto",
        "min_dphi",
        "met_pf_calo",
    ]
    preferred_order += sorted([k for k in cf if k.startswith("recoil_gt_")])
    preferred_order += ["signal_region", "presel"]
    ordered = [k for k in preferred_order if k in cf]
    ordered += sorted([k for k in cf if k not in ordered])

    print("Cutflow")
    print("-" * 42)
    for cut in ordered:
        print(f"{cut:20s} : {cf[cut]:10d}")


def print_cutflow_by_region(results: Dict, datasets: Optional[Iterable[str]] = None):
    """
    Print one cutflow block per analysis region.

    Notes
    -----
    - SR has full step-by-step counters.
    - CR blocks use available counters from processor cutflow (final region yields).
    """
    cf = get_cutflow(results, datasets=datasets)

    sr_steps = [
        "all_events",
        "masspoint",
        "trigger",
        "njet_ge1",
        "nbjet_ge2",
        "lepton_veto",
        "min_dphi",
        "met_pf_calo",
    ]
    sr_steps += sorted([k for k in cf if k.startswith("recoil_gt_")])
    sr_steps += ["signal_region"]

    region_blocks = {
        "sr": [k for k in sr_steps if k in cf],
        "zecr": [k for k in ("trigger", "zecr") if k in cf],
        "zmucr": [k for k in ("trigger", "zmucr") if k in cf],
        "tecr": [k for k in ("trigger", "tecr") if k in cf],
        "tmucr": [k for k in ("trigger", "tmucr") if k in cf],
    }

    print("Cutflow by region")
    print("-" * 42)
    for reg in REGION_ORDER:
        keys = region_blocks.get(reg, [])
        if not keys:
            continue
        print(f"[{reg}]")
        for k in keys:
            print(f"  {k:18s} : {cf[k]:10d}")
        print("")


def _cutflow_steps_for_region(cf: Dict[str, int], region: str) -> list:
    recoil_keys = sorted([k for k in cf if k.startswith("recoil_gt_")])
    recoil_key = recoil_keys[0] if recoil_keys else None
    if region == "sr":
        steps = [
            ("trigger", "trigger"),
            ("nJet(2-3)", "njet_2to3" if "njet_2to3" in cf else "njet_ge1"),
            ("nBJet(=2)", "nbjet_eq2" if "nbjet_eq2" in cf else "nbjet_ge2"),
            ("nLep=0", "lepton_veto"),
            ("min_dPhi", "min_dphi"),
            ("MET(PF-calo)", "met_pf_calo"),
            ("recoil", recoil_key) if recoil_key else None,
            ("sr", "signal_region"),
        ]
        return [s for s in steps if s is not None and s[1] in cf]
    if region == "zecr":
        order = [("trigger", "trigger"), ("presel", "zecr_presel"), ("2ee", "zecr_twolep"), ("leadLepPt", "zecr_leadlep"), ("mll", "zecr_mll"), ("zecr", "zecr")]
        return [s for s in order if s[1] in cf]
    if region == "zmucr":
        order = [("trigger", "trigger"), ("presel", "zmucr_presel"), ("2mumu", "zmucr_twolep"), ("leadLepPt", "zmucr_leadlep"), ("mll", "zmucr_mll"), ("zmucr", "zmucr")]
        return [s for s in order if s[1] in cf]
    if region == "tecr":
        order = [("trigger", "trigger"), ("presel", "tecr_presel"), ("1e", "tecr_onelep"), ("leadLepPt", "tecr_leppt"), ("mT", "tecr_mt"), ("nBJet=2", "tecr_nbjet"), ("nNonB>=2", "tecr_nnonb"), ("tecr", "tecr")]
        return [s for s in order if s[1] in cf]
    if region == "tmucr":
        order = [("trigger", "trigger"), ("presel", "tmucr_presel"), ("1mu", "tmucr_onelep"), ("leadLepPt", "tmucr_leppt"), ("mT", "tmucr_mt"), ("nBJet=2", "tmucr_nbjet"), ("nNonB>=2", "tmucr_nnonb"), ("tmucr", "tmucr")]
        return [s for s in order if s[1] in cf]
    return []


def plot_cutflow_by_region(
    pkl_path: str,
    datasets: Optional[Iterable[str]] = None,
    save_plots: bool = False,
    save_dir: Optional[str] = None,
):
    """
    Plot one stacked cutflow panel per region.
    """
    _apply_plot_style()
    results = load_results(pkl_path)
    names = _pick_names(results, datasets)
    data_keys, bkg_keys, _ = _classify_datasets(results, names)
    cf_all = get_cutflow(results, datasets=datasets)

    figs = {}
    out_dir = _resolve_plot_dir(save_dir) if save_plots else None
    for region in REGION_ORDER:
        steps = _cutflow_steps_for_region(cf_all, region)
        if not steps:
            continue

        xlabels = [lbl for lbl, _ in steps]
        x = np.arange(len(xlabels), dtype=float)
        bottom = np.zeros_like(x, dtype=float)

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for b in bkg_keys:
            cf_b = results.get(b, {}).get("cutflow", {})
            vals = np.array([float(cf_b.get(k, 0.0)) for _, k in steps], dtype=float)
            if np.all(vals <= 0):
                continue
            ax.bar(
                x,
                vals,
                bottom=bottom,
                width=0.8,
                color=PROCESS_COLORS.get(b, DEFAULT_COLOR),
                edgecolor="none",
                linewidth=0.0,
                alpha=0.9,
                label=b,
            )
            bottom += vals

        dkeys = _data_keys_for_region(data_keys, region)
        if region != "sr" and dkeys:
            dvals = np.zeros_like(x, dtype=float)
            for dk in dkeys:
                cf_d = results.get(dk, {}).get("cutflow", {})
                dvals += np.array([float(cf_d.get(k, 0.0)) for _, k in steps], dtype=float)
            derr = np.sqrt(np.clip(dvals, 0.0, None))
            ax.errorbar(x, dvals, yerr=derr, fmt="o", color="black", markersize=5, linewidth=1.0, label="Data")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=0)
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(1e-1, ymin), max(1.0, ymax) * 10.0)
        ax.set_ylabel("Events")
        _title_inside_axes(ax, f"Cutflow [{region}]", loc="upper left")
        ax.grid(alpha=0.18, axis="y")
        ax.legend(loc="upper right", ncol=2, fontsize=10.5, frameon=False, columnspacing=1.0, handlelength=1.6)
        fig.tight_layout()
        if out_dir is not None:
            fig.savefig(out_dir / f"cutflow_{region}.png", dpi=160)
        figs[region] = fig
    return figs


def plot_all_variables_grid(
    pkl_path: str,
    datasets: Optional[Iterable[str]] = None,
    ncols: int = 3,
    figsize_per_panel=(6.6, 5.6),
    show_cutflow: bool = True,
    show_cutflow_plots: bool = False,
    save_plots: bool = False,
    save_dir: Optional[str] = None,
    save_individual: bool = True,
    save_grid: bool = False,
):
    """
    Plot every histogram variable found in the pickle on a matplotlib grid.
    No additional normalization is applied here.
    """
    _apply_plot_style()
    results = load_results(pkl_path)
    selected_names = _pick_names(results, datasets)
    hist_keys = list_histogram_keys(results)
    if not hist_keys:
        raise RuntimeError("No histogram variables found in the pickle.")

    panel_specs = []
    for key in hist_keys:
        regions = _available_regions(results, selected_names, key)
        if regions:
            for r in regions:
                panel_specs.append((key, r))
        # In the new output schema we only plot region-aware histograms.
        # Legacy non-region hist keys are intentionally skipped here.
    if not panel_specs:
        raise RuntimeError(
            "No region-aware histograms found. "
            "Use a new-schema pickle from run_analysis.py output bundle."
        )

    nvars = len(panel_specs)
    nrows = math.ceil(nvars / ncols)
    fig = plt.figure(
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows * 1.35),
        constrained_layout=True,
    )
    gs = GridSpec(
        nrows * 2,
        ncols,
        figure=fig,
        height_ratios=sum(([3.0, 1.0] for _ in range(nrows)), []),
    )
    axes = np.empty((nrows * 2, ncols), dtype=object)

    for i, (key, reg) in enumerate(panel_specs):
        prow = i // ncols
        pcol = i % ncols
        ax = fig.add_subplot(gs[2 * prow, pcol])
        rax = fig.add_subplot(gs[2 * prow + 1, pcol], sharex=ax)
        axes[2 * prow, pcol] = ax
        axes[2 * prow + 1, pcol] = rax
        _draw_stacked_panel(ax, rax, results, key, selected_names, region=reg)
        plt.setp(ax.get_xticklabels(), visible=False)

        if save_plots and save_individual:
            out_dir = _resolve_plot_dir(save_dir)
            f1 = plt.figure(figsize=(6.4, 5.2), constrained_layout=True)
            gs1 = GridSpec(2, 1, figure=f1, height_ratios=[3.0, 1.0])
            ax1 = f1.add_subplot(gs1[0, 0])
            rax1 = f1.add_subplot(gs1[1, 0], sharex=ax1)
            _draw_stacked_panel(ax1, rax1, results, key, selected_names, region=reg)
            plt.setp(ax1.get_xticklabels(), visible=False)
            tag = f"{_safe_plot_name(key)}_{_safe_plot_name(reg)}" if reg is not None else _safe_plot_name(key)
            f1.savefig(out_dir / f"{tag}.png", dpi=160)
            plt.close(f1)

    total_slots = nrows * ncols
    for j in range(nvars, total_slots):
        prow = j // ncols
        pcol = j % ncols
        ax = fig.add_subplot(gs[2 * prow, pcol])
        rax = fig.add_subplot(gs[2 * prow + 1, pcol], sharex=ax)
        ax.set_axis_off()
        rax.set_axis_off()
        axes[2 * prow, pcol] = ax
        axes[2 * prow + 1, pcol] = rax

    if show_cutflow:
        print_cutflow_by_region(results, datasets=datasets)
    if show_cutflow_plots:
        plot_cutflow_by_region(
            pkl_path,
            datasets=datasets,
            save_plots=save_plots,
            save_dir=save_dir,
        )
    if save_plots and save_grid:
        out_dir = _resolve_plot_dir(save_dir)
        fig.savefig(out_dir / "all_variables_grid.png", dpi=160)

    return fig, axes


def plot_variable_stacked(
    pkl_path: str,
    hist_key: str,
    datasets: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
    save_plots: bool = False,
    save_dir: Optional[str] = None,
):
    """
    Plot one histogram key as stacked processes with project aesthetics.
    """
    _apply_plot_style()
    results = load_results(pkl_path)
    names = _pick_names(results, datasets)
    regions = _available_regions(results, names, hist_key)
    if regions:
        ncols = min(3, len(regions))
        nrows = math.ceil(len(regions) / ncols)
        fig = plt.figure(figsize=(6.4 * ncols, 5.2 * nrows), constrained_layout=True)
        gs = GridSpec(
            nrows * 2,
            ncols,
            figure=fig,
            height_ratios=sum(([3.0, 1.0] for _ in range(nrows)), []),
        )
        axes = np.empty((nrows * 2, ncols), dtype=object)
        for i, reg in enumerate(regions):
            prow = i // ncols
            pcol = i % ncols
            ax = fig.add_subplot(gs[2 * prow, pcol])
            rax = fig.add_subplot(gs[2 * prow + 1, pcol], sharex=ax)
            axes[2 * prow, pcol] = ax
            axes[2 * prow + 1, pcol] = rax
            _draw_stacked_panel(
                ax,
                rax,
                results,
                hist_key,
                names,
                region=reg,
                panel_title=(f"{title} [{reg}]" if title else None),
            )
            plt.setp(ax.get_xticklabels(), visible=False)
        total_slots = nrows * ncols
        for j in range(len(regions), total_slots):
            prow = j // ncols
            pcol = j % ncols
            ax = fig.add_subplot(gs[2 * prow, pcol])
            rax = fig.add_subplot(gs[2 * prow + 1, pcol], sharex=ax)
            ax.set_axis_off()
            rax.set_axis_off()
            axes[2 * prow, pcol] = ax
            axes[2 * prow + 1, pcol] = rax
        if save_plots:
            out_dir = _resolve_plot_dir(save_dir)
            fig.savefig(out_dir / f"{hist_key}_stacked.png", dpi=160)
        return fig, axes

    fig = plt.figure(figsize=(6.4, 5.2), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    rax = fig.add_subplot(gs[1, 0], sharex=ax)
    _draw_stacked_panel(ax, rax, results, hist_key, names, panel_title=title or hist_key)
    plt.setp(ax.get_xticklabels(), visible=False)
    if save_plots:
        out_dir = _resolve_plot_dir(save_dir)
        fig.savefig(out_dir / f"{hist_key}_stacked.png", dpi=160)
    return fig, (ax, rax)
