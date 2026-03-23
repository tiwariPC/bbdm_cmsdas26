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
    x = np.arange(len(rows), dtype=float)
    labels = [r[2] for r in rows]
    exp = np.array([r[3]["exp"] for r in rows], dtype=float)
    e1dn = np.array([r[3]["exp1dn"] for r in rows], dtype=float)
    e1up = np.array([r[3]["exp1up"] for r in rows], dtype=float)
    e2dn = np.array([r[3]["exp2dn"] for r in rows], dtype=float)
    e2up = np.array([r[3]["exp2up"] for r in rows], dtype=float)
    obs = np.array([r[3]["obs"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(rows)), 5.8))
    ax.fill_between(x, e2dn, e2up, color="#ffeb3b", alpha=0.9, label="Expected ±2σ")
    ax.fill_between(x, e1dn, e1up, color="#4caf50", alpha=0.9, label="Expected ±1σ")
    ax.plot(x, exp, "--", color="black", linewidth=1.8, label="Expected")
    if args.show_observed and np.any(np.isfinite(obs)):
        ax.plot(x, obs, "-", color="black", linewidth=1.6, label="Observed")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("95% CL upper limit on signal strength μ")
    ax.set_xlabel("Signal mass point")
    ax.set_yscale("log")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved limit plot: {out}")


if __name__ == "__main__":
    main()
