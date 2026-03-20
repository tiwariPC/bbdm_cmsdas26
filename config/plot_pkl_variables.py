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
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from hist import Hist

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


def _apply_plot_style() -> None:
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


def load_results(pkl_path: str) -> Dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


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
    return [k for k, v in acc.items() if _unwrap_hist(v) is not None]


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


def plot_all_variables_grid(
    pkl_path: str,
    datasets: Optional[Iterable[str]] = None,
    ncols: int = 3,
    figsize_per_panel=(4.6, 3.6),
    show_cutflow: bool = True,
):
    """
    Plot every histogram variable found in the pickle on a matplotlib grid.
    No additional normalization is applied here.
    """
    _apply_plot_style()
    results = load_results(pkl_path)
    hist_keys = list_histogram_keys(results)
    if not hist_keys:
        raise RuntimeError("No histogram variables found in the pickle.")

    nvars = len(hist_keys)
    nrows = math.ceil(nvars / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, key in enumerate(hist_keys):
        h = sum_histogram(results, key, datasets=datasets)
        ax = axes_flat[i]
        if h is None:
            ax.text(0.5, 0.5, f"Missing: {key}", ha="center", va="center")
            ax.set_axis_off()
            continue
        h.plot(ax=ax, linewidth=1.8)
        ax.set_title(key)
        ax.grid(alpha=0.18)

    for j in range(nvars, len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.tight_layout()

    if show_cutflow:
        print_cutflow(results, datasets=datasets)

    return fig, axes


def plot_variable_stacked(
    pkl_path: str,
    hist_key: str,
    datasets: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
):
    """
    Plot one histogram key as stacked processes with project aesthetics.
    """
    _apply_plot_style()
    results = load_results(pkl_path)
    names = list(datasets) if datasets is not None else list(results.keys())
    names = [n for n in names if n in results]

    arrays = []
    labels = []
    colors = []
    for name in names:
        h = _unwrap_hist(results[name].get(hist_key))
        if h is None:
            continue
        values = h.values()
        edges = h.axes[0].edges
        arrays.append((values, edges))
        labels.append(name)
        colors.append(PROCESS_COLORS.get(name, DEFAULT_COLOR))

    if not arrays:
        raise RuntimeError(f"Histogram '{hist_key}' not available for selected datasets.")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # Draw stacked bars with common binning.
    first_edges = arrays[0][1]
    centers = 0.5 * (first_edges[:-1] + first_edges[1:])
    widths = np.diff(first_edges)
    bottom = np.zeros_like(centers, dtype=float)
    for (vals, edges), label, color in zip(arrays, labels, colors):
        if len(edges) != len(first_edges) or np.any(edges != first_edges):
            raise ValueError("All datasets must share the same binning for stacked plotting.")
        ax.bar(
            centers,
            vals,
            width=widths,
            bottom=bottom,
            align="center",
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=label,
        )
        bottom = bottom + vals
    ax.set_title(title or hist_key)
    ax.legend(loc="upper right", ncol=2, fontsize=9, frameon=False, columnspacing=1.0, handlelength=1.6)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig, ax

