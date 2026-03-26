#!/usr/bin/env python3
"""
Build simple counting datacards for CMS Combine from a bbDM output pickle.

This uses SR recoil yields (sum of bins) and writes one datacard per signal
mass point found in the pickle.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


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


def _pick_data_observation(samples: Dict, region: str) -> Optional[float]:
    # Prefer data_MET for SR, then any data-like key.
    preferred = ["data_MET", "MET_Run2017", "MET"]
    keys = list(samples.keys())
    for k in preferred:
        if k in keys:
            return _yield_from_hist(_extract_hist(samples, k, "recoil"), region=region)
    for k in keys:
        if _is_data(k):
            return _yield_from_hist(_extract_hist(samples, k, "recoil"), region=region)
    return None


def _build_datacard_text(
    signal_name: str,
    signal_rate: float,
    bkg_rates: Dict[str, float],
    observation: float,
    lumi_rel_unc: float,
    sig_norm_rel_unc: float,
    bkg_norm_rel_unc_by_proc: Dict[str, float],
) -> str:
    processes = [signal_name] + list(bkg_rates.keys())
    rates = [signal_rate] + [bkg_rates[p] for p in bkg_rates]

    proc_ids = [0] + list(range(1, len(processes)))
    obs_int = int(round(max(0.0, observation)))

    lines = []
    lines.append("imax 1")
    lines.append("jmax *")
    lines.append("kmax *")
    lines.append("------------")
    lines.append("bin sr")
    lines.append(f"observation {obs_int}")
    lines.append("------------")
    lines.append("bin " + " ".join(["sr"] * len(processes)))
    lines.append("process " + " ".join(processes))
    lines.append("process " + " ".join(str(x) for x in proc_ids))
    lines.append("rate " + " ".join(f"{max(0.0, r):.8g}" for r in rates))
    lines.append("------------")

    # Common lumi uncertainty on all MC components (signal + bkgs).
    lumi_factor = 1.0 + float(lumi_rel_unc)
    lines.append("lumi lnN " + " ".join([f"{lumi_factor:.6g}"] * len(processes)))

    # Signal normalization nuisance.
    sig_factor = 1.0 + float(sig_norm_rel_unc)
    sig_row = [f"{sig_factor:.6g}"] + ["-"] * (len(processes) - 1)
    lines.append("sig_norm lnN " + " ".join(sig_row))

    # Per-background normalization nuisances.
    for ib, bkg in enumerate(bkg_rates.keys(), start=1):
        rel = float(bkg_norm_rel_unc_by_proc.get(bkg, 0.10))
        fac = 1.0 + rel
        row = ["-"] * len(processes)
        row[ib] = f"{fac:.6g}"
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

    bkg_rates = {
        b: max(0.0, _yield_from_hist(_extract_hist(samples, b, hist_key), region=region))
        for b in bkg_names
    }
    observation = _pick_data_observation(samples, region=region)
    if observation is None:
        observation = sum(bkg_rates.values())

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for s in sorted(signal_names):
        srate = max(0.0, _yield_from_hist(_extract_hist(samples, s, hist_key), region=region))
        text = _build_datacard_text(
            signal_name=s,
            signal_rate=srate,
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
    ap.add_argument("--hist-key", default="recoil", help="Histogram key to use for rates (default: recoil)")
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
