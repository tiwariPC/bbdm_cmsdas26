#!/usr/bin/env python3
"""
Build binned datacards for CMS Combine from a bbDM output pickle.

This uses per-bin SR yields of the chosen observable (default: cos_theta_star)
and writes one datacard per signal mass point found in the pickle.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def _ensure_repo_on_syspath() -> None:
    """
    Ensure the project root is importable before unpickling.

    The output pickle may contain custom classes (e.g. processor.HistAccumulator)
    that require ``processor`` and other repo modules on ``sys.path``.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _unwrap_hist(obj):
    if hasattr(obj, "_hist"):
        return obj._hist
    return obj


def _is_data(name: str) -> bool:
    n = str(name)
    return n.startswith("data_") or n.startswith("MET") or n.startswith("SingleElectron") or n.startswith("SingleMuon")


def _is_signal(name: str) -> bool:
    n = str(name)
    return n.startswith("signal_") or n.startswith("bbDM")


def _as_flat_samples(raw: Dict) -> Dict:
    if not isinstance(raw, dict):
        raise RuntimeError("Pickle payload is not a dict.")
    samples = raw.get("samples")
    if not isinstance(samples, dict):
        # Legacy shape already: {sample: accumulator}
        return raw
    out = {}
    for name, sample in samples.items():
        if not isinstance(sample, dict):
            continue
        acc = {}
        if isinstance(sample.get("cutflow"), dict):
            acc["cutflow"] = sample["cutflow"]
        hbr = sample.get("hists_by_region", {})
        if isinstance(hbr, dict):
            for k, v in hbr.items():
                acc[f"{k}_by_region"] = v
        hs = sample.get("hists", {})
        if isinstance(hs, dict):
            acc.update(hs)
        for k, v in sample.items():
            if k in ("cutflow", "hists_by_region", "hists", "metadata"):
                continue
            if k not in acc:
                acc[k] = v
        out[name] = acc
    return out


def _extract_hist(samples: Dict, sample_name: str, hist_key: str):
    acc = samples.get(sample_name, {})
    for key in (f"{hist_key}_by_region", hist_key):
        if key in acc:
            return _unwrap_hist(acc[key])
    return None


def _yield_from_hist(hist_obj, region: str = "sr") -> float:
    if hist_obj is None:
        return 0.0
    h = hist_obj
    try:
        if any(getattr(ax, "name", None) == "region" for ax in h.axes):
            h = h[{"region": region}]
    except Exception:
        pass
    vals = h.values()
    return float(vals.sum()) if hasattr(vals, "sum") else float(sum(vals))


def _yields_by_bin_from_hist(hist_obj, region: str = "sr") -> list[float]:
    if hist_obj is None:
        return []
    h = hist_obj
    try:
        if any(getattr(ax, "name", None) == "region" for ax in h.axes):
            h = h[{"region": region}]
    except Exception:
        pass
    vals = h.values()
    arr = vals.tolist() if hasattr(vals, "tolist") else list(vals)
    return [float(max(0.0, v)) for v in arr]


def _pick_data_observation(samples: Dict, hist_key: str, region: str) -> Optional[list[float]]:
    # Prefer data_MET for SR, then any data-like key.
    preferred = ["data_MET", "MET_Run2017", "MET"]
    keys = list(samples.keys())
    for k in preferred:
        if k in keys:
            return _yields_by_bin_from_hist(_extract_hist(samples, k, hist_key), region=region)
    for k in keys:
        if _is_data(k):
            return _yields_by_bin_from_hist(_extract_hist(samples, k, hist_key), region=region)
    return None


def _build_datacard_text(
    signal_name: str,
    signal_rates: list[float],
    bkg_rates: Dict[str, list[float]],
    observation: list[float],
    lumi_rel_unc: float,
    sig_norm_rel_unc: float,
    bkg_norm_rel_unc_by_proc: Dict[str, float],
) -> str:
    nbins = len(signal_rates)
    if nbins == 0:
        raise RuntimeError("Signal histogram has zero bins after region projection.")
    if len(observation) != nbins:
        raise RuntimeError("Data observation bin count does not match signal bin count.")
    for bkg, vals in bkg_rates.items():
        if len(vals) != nbins:
            raise RuntimeError(f"Background '{bkg}' bin count does not match signal bin count.")

    proc_names = [signal_name] + list(bkg_rates.keys())
    proc_ids = [0] + list(range(1, len(proc_names)))
    bin_names = [f"sr_bin{i+1}" for i in range(nbins)]

    card_bins = []
    card_proc_names = []
    card_proc_ids = []
    card_rates = []
    for ib in range(nbins):
        for ip, pname in enumerate(proc_names):
            card_bins.append(bin_names[ib])
            card_proc_names.append(pname)
            card_proc_ids.append(str(proc_ids[ip]))
            if ip == 0:
                card_rates.append(f"{max(0.0, signal_rates[ib]):.8g}")
            else:
                card_rates.append(f"{max(0.0, bkg_rates[pname][ib]):.8g}")

    obs_ints = [str(int(round(max(0.0, v)))) for v in observation]

    lines = []
    lines.append(f"imax {nbins}")
    lines.append("jmax *")
    lines.append("kmax *")
    lines.append("------------")
    lines.append("bin " + " ".join(bin_names))
    lines.append("observation " + " ".join(obs_ints))
    lines.append("------------")
    lines.append("bin " + " ".join(card_bins))
    lines.append("process " + " ".join(card_proc_names))
    lines.append("process " + " ".join(card_proc_ids))
    lines.append("rate " + " ".join(card_rates))
    lines.append("------------")

    # Common lumi uncertainty on all MC components (signal + bkgs).
    lumi_factor = 1.0 + float(lumi_rel_unc)
    lines.append("lumi lnN " + " ".join([f"{lumi_factor:.6g}"] * len(card_rates)))

    # Signal normalization nuisance.
    sig_factor = 1.0 + float(sig_norm_rel_unc)
    sig_row = []
    for _ in range(nbins):
        sig_row.append(f"{sig_factor:.6g}")
        sig_row.extend(["-"] * (len(proc_names) - 1))
    lines.append("sig_norm lnN " + " ".join(sig_row))

    # Per-background normalization nuisances.
    for ibkg, bkg in enumerate(bkg_rates.keys(), start=1):
        rel = float(bkg_norm_rel_unc_by_proc.get(bkg, 0.10))
        fac = 1.0 + rel
        row = []
        for _ in range(nbins):
            per_bin_row = ["-"] * len(proc_names)
            per_bin_row[ibkg] = f"{fac:.6g}"
            row.extend(per_bin_row)
        lines.append(f"norm_{bkg} lnN " + " ".join(row))

    return "\n".join(lines) + "\n"


def _safe_name(name: str) -> str:
    out = str(name).replace("/", "_").replace(" ", "_")
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in out)


def build_cards(
    input_pkl: Path,
    output_dir: Path,
    hist_key: str,
    region: str,
) -> Tuple[int, Path]:
    _ensure_repo_on_syspath()
    raw = pickle.load(open(input_pkl, "rb"))
    samples = _as_flat_samples(raw)
    meta = raw.get("metadata", {}) if isinstance(raw, dict) else {}

    # Nuisance defaults from bundle metadata if present.
    syst = meta.get("systematics", {}) if isinstance(meta, dict) else {}
    lumi_rel_unc = float(syst.get("lumi_rel_unc", 0.025))
    sig_norm_rel_unc = float(syst.get("default_signal_norm_rel_unc", 0.10))
    bkg_norm_rel_unc_by_proc = dict(syst.get("bkg_norm_rel_unc_by_process", {}))

    names = list(samples.keys())
    signal_names = [n for n in names if _is_signal(n)]
    bkg_names = [n for n in names if (not _is_signal(n)) and (not _is_data(n))]
    bkg_names = sorted(bkg_names)

    if not signal_names:
        raise RuntimeError("No signal samples found in pickle.")
    if not bkg_names:
        raise RuntimeError("No background samples found in pickle.")

    bkg_rates = {b: _yields_by_bin_from_hist(_extract_hist(samples, b, hist_key), region=region) for b in bkg_names}
    observation = _pick_data_observation(samples, hist_key=hist_key, region=region)
    if observation is None:
        if bkg_rates:
            nbins = len(next(iter(bkg_rates.values())))
            observation = [0.0] * nbins
            for vals in bkg_rates.values():
                for i, v in enumerate(vals):
                    observation[i] += v
        else:
            observation = []

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for s in sorted(signal_names):
        srates = _yields_by_bin_from_hist(_extract_hist(samples, s, hist_key), region=region)
        if len(srates) == 0:
            raise RuntimeError(
                f"Signal '{s}' has no histogram bins for hist_key='{hist_key}' in region='{region}'."
            )
        if sum(srates) <= 0.0:
            raise RuntimeError(
                f"Signal '{s}' has zero total yield for hist_key='{hist_key}' in region='{region}'. "
                "Check that this observable is filled for signal in the input pickle."
            )
        text = _build_datacard_text(
            signal_name=s,
            signal_rates=srates,
            bkg_rates=bkg_rates,
            observation=observation,
            lumi_rel_unc=lumi_rel_unc,
            sig_norm_rel_unc=sig_norm_rel_unc,
            bkg_norm_rel_unc_by_proc=bkg_norm_rel_unc_by_proc,
        )
        card_path = output_dir / f"datacard_{_safe_name(s)}.txt"
        card_path.write_text(text)
        count += 1

    # Save helper shell commands.
    run_sh = output_dir / "run_combine_all.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run from CMSSW with Combine available.",
        f"CARDS_DIR=\"{output_dir}\"",
        "OUT_DIR=\"${CARDS_DIR}\"",
        "",
        "for card in \"${CARDS_DIR}\"/datacard_*.txt; do",
        "  base=$(basename \"${card}\" .txt)",
        "  # Goodness-of-fit (saturated) + toy p-value",
        "  combine -M GoodnessOfFit -d \"${card}\" --algo=saturated -n .\"${base}\"",
        "  combine -M GoodnessOfFit -d \"${card}\" --algo=saturated -t 200 --toysFrequentist -n .\"${base}\"",
        "  # Asymptotic 95% CL limit",
        "  combine -M AsymptoticLimits -d \"${card}\" -n .\"${base}\"",
        "done",
        "",
        "mv higgsCombine*.root \"${OUT_DIR}\"/ 2>/dev/null || true",
        "echo \"Combine outputs in: ${OUT_DIR}\"",
    ]
    run_sh.write_text("\n".join(lines) + "\n")
    os.chmod(run_sh, 0o755)
    return count, run_sh


def main() -> None:
    ap = argparse.ArgumentParser(description="Build simple Combine datacards from bbDM pickle output.")
    ap.add_argument("--input", required=True, help="Path to output pickle (e.g. output/output_2017_full.pkl)")
    ap.add_argument("--output-dir", default="output/combine_cards", help="Directory to write datacards and combine outputs")
    ap.add_argument(
        "--hist-key",
        default="cos_theta_star",
        help="Histogram key to use for rates and data observation (default: cos_theta_star)",
    )
    ap.add_argument("--region", default="sr", help="Region category to project (default: sr)")
    args = ap.parse_args()

    count, run_sh = build_cards(
        input_pkl=Path(args.input),
        output_dir=Path(args.output_dir),
        hist_key=args.hist_key,
        region=args.region,
    )
    print(f"Wrote {count} datacards.")
    print(f"Combine runner script: {run_sh}")


if __name__ == "__main__":
    main()
