#!/usr/bin/env python3
"""
Collect AsymptoticLimits outputs from Combine and make a simple limit-vs-masspoint plot.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
import uproot


def _parse_masspoint_from_cardstem(stem: str) -> Tuple[Optional[int], Optional[int], str]:
    # Expected examples:
    #   datacard_signal_mA600_ma250_mchi1
    #   higgsCombine.datacard_signal_mA600_ma250_mchi1.AsymptoticLimits.mH120.root
    ma = re.search(r"ma(\d+)", stem)
    mA = re.search(r"mA(\d+)", stem)
    ma_v = int(ma.group(1)) if ma else None
    mA_v = int(mA.group(1)) if mA else None
    label = f"mA={mA_v}, ma={ma_v}" if (mA_v is not None and ma_v is not None) else stem
    return mA_v, ma_v, label


def _read_limit_file(path: Path):
    with uproot.open(path) as f:
        t = f["limit"]
        q = np.asarray(t["quantileExpected"].array(library="np"), dtype=float)
        lim = np.asarray(t["limit"].array(library="np"), dtype=float)

    out = {
        "exp2dn": np.nan,
        "exp1dn": np.nan,
        "exp": np.nan,
        "exp1up": np.nan,
        "exp2up": np.nan,
        "obs": np.nan,
    }
    for qq, ll in zip(q, lim):
        if qq < 0:
            out["obs"] = float(ll)
        elif abs(qq - 0.025) < 1e-4:
            out["exp2dn"] = float(ll)
        elif abs(qq - 0.16) < 1e-4:
            out["exp1dn"] = float(ll)
        elif abs(qq - 0.50) < 1e-4:
            out["exp"] = float(ll)
        elif abs(qq - 0.84) < 1e-4:
            out["exp1up"] = float(ll)
        elif abs(qq - 0.975) < 1e-4:
            out["exp2up"] = float(ll)
    return out


def _sort_key(item):
    mA, ma, _label, _vals = item
    return (mA if mA is not None else 10**9, ma if ma is not None else 10**9)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Combine asymptotic limits for all mass points.")
    ap.add_argument(
        "--combine-dir",
        default="output/combine_cards",
        help="Directory containing higgsCombine*.AsymptoticLimits*.root files",
    )
    ap.add_argument("--output", default="output/combine_plots/limits_all_masspoints.png")
    ap.add_argument("--show-observed", action="store_true", help="Draw observed limits if available.")
    args = ap.parse_args()

    cdir = Path(args.combine_dir)
    files = sorted(cdir.glob("higgsCombine*.AsymptoticLimits*.root"))
    if not files:
        raise RuntimeError(f"No AsymptoticLimits ROOT files found in: {cdir}")

    rows: List[Tuple[Optional[int], Optional[int], str, dict]] = []
    for p in files:
        mA, ma, label = _parse_masspoint_from_cardstem(p.stem)
        vals = _read_limit_file(p)
        rows.append((mA, ma, label, vals))

    rows = sorted(rows, key=_sort_key)
    ma_values = np.array([r[1] for r in rows], dtype=float)
    exp = np.array([r[3]["exp"] for r in rows], dtype=float)
    e1dn = np.array([r[3]["exp1dn"] for r in rows], dtype=float)
    e1up = np.array([r[3]["exp1up"] for r in rows], dtype=float)
    e2dn = np.array([r[3]["exp2dn"] for r in rows], dtype=float)
    e2up = np.array([r[3]["exp2up"] for r in rows], dtype=float)
    obs = np.array([r[3]["obs"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 8))
    hep.cms.label("", lumi=41.5, loc=0, llabel="", fontsize=20, ax=ax)

    fil_2sigma = ax.fill_between(
        ma_values, e2dn, e2up, alpha=0.9, color="#F5BB54", label=r"95$\%$ expected"
    )
    fil_1sigma = ax.fill_between(
        ma_values, e1dn, e1up, alpha=0.9, color="#607641", label=r"68$\%$ expected"
    )
    line_med = ax.plot(
        ma_values, exp, linestyle="dashed", color="black", linewidth=1.8, label="Median expected"
    )

    line_obs = None
    if args.show_observed and np.any(np.isfinite(obs)):
        line_obs = ax.plot(ma_values, obs, linestyle="solid", color="black", linewidth=2, label="Observed")

    ax.axhline(y=1.0, color="red", linestyle="-.", alpha=0.7, linewidth=1.8)

    ax.set_xlim(50, 500)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 1000)
    ticks_x = [50, 100, 200, 300, 400, 500]
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticks_x, fontsize=20)
    ax.tick_params(axis="y", labelsize=24)

    ticks_y = [0.1, 1, 10, 20, 30, 100, 200, 1000]
    tick_labels = [
        r"$10^{-1}$",
        r"$1$",
        r"$10$",
        r"$20$",
        r"$30$",
        r"$10^{2}$",
        r"$2 \times 10^{2}$",
        r"$10^{3}$",
    ]
    ax.yaxis.set_major_locator(plt.FixedLocator(ticks_y))
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel(r"$\mathit{m}_{\mathsf{a}}$ (GeV)", usetex=True)
    ax.set_ylabel(
        r"$\mathsf{95\% CL\hspace{0.3cm}\sigma(pp\rightarrow b\bar{b}\chi\bar{\chi})}$(fb)",
        labelpad=0.2,
        usetex=True,
    )

    heading = "2HDM+a"
    body = "\n".join(
        (
            r"",
            r"$\mathsf{b\bar{b}+p_T^{miss}}$",
            r"$\mathit{m}_{\mathsf{A}}$ = 600 GeV",
            r"$\mathit{m}_{\chi}$ = 1 GeV",
            r"tan$\beta$ = 35",
            r"sin$\theta$ = 0.7",
        )
    )
    ax.text(0.030, 0.92, heading, transform=ax.transAxes, fontsize=22)
    ax.text(0.045, 0.67, body, transform=ax.transAxes, fontsize=20, usetex=True)

    legend_handles = []
    if line_obs is not None:
        legend_handles.append(line_obs[0])
    legend_handles.append(line_med[0])
    legend_handles.extend([fil_1sigma, fil_2sigma])
    ax.legend(
        handles=legend_handles, frameon=False, fancybox=True, fontsize=20, loc="upper right"
    )
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved limit plot: {out}")


if __name__ == "__main__":
    main()
