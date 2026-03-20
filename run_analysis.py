#!/usr/bin/env python3
"""
Run bbDM processor: one-file mode (quick) or full analysis over all files.

Usage:
  python run_analysis.py                    # one file per dataset
  python run_analysis.py --full              # all files (full analysis)
  python run_analysis.py --full --year 2017   # full analysis, 2017
  python run_analysis.py --full --dataset DYJetsToLL_M50_HT400to600
  python run_analysis.py --full -o output_2017.coffea

Output is saved as a pickle in the output/ directory: by default a dict of merged physics groups
(data, DYJets, ZJets, WJets, Top, STop, DIBOSON, SMH, plus signal_mA600_ma*_mchi1 keys) -> accumulator.
Use --per-dataset for one entry per YAML dataset (legacy). Load in a notebook from
output/output_2017.pkl or output/output_2017_full.pkl.
"""

import argparse
import os
import sys

# Project root and output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "output"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _scale_histograms_in_accumulator(acc, factor: float) -> None:
    """
    Scale all hist-like entries in a Coffea dict_accumulator in place.

    Notes
    -----
    - Hist entries are wrapped by HistAccumulator and expose ``_hist``.
    - Cutflow/int entries are intentionally left unchanged.
    """
    for value in acc.values():
        if hasattr(value, "_hist"):
            value._hist = value._hist * float(factor)


def _load_signal_masspoint_xsecs() -> dict:
    """
    Build mapping:
      signal_mA{mA}_ma{ma}_mchi1 -> xsec_pb
    from ``masspoint_xsec_pb`` in full YAML signal entries.
    """
    from config.datasets_2017 import load_full_datasets, signal_masspoint_key

    out = {}
    full_meta = load_full_datasets()
    for meta in full_meta.values():
        if meta.get("group") != "signal":
            continue
        mp_map = meta.get("masspoint_xsec_pb") or {}
        for raw_key, xsec in mp_map.items():
            parts = str(raw_key).split("_")
            # Expected format: ma_<ma>_mA_<mA>
            if len(parts) != 4 or parts[0] != "ma" or parts[2] != "mA":
                continue
            try:
                ma = int(parts[1])
                mA = int(parts[3])
                out[signal_masspoint_key(mA, ma, 1)] = float(xsec)
            except Exception:
                continue
    return out


def _load_full_dataset_meta() -> dict:
    """
    Mapping: dataset_name -> {"xsec": float|None, "isData": bool}
    from full YAML.
    """
    from config.datasets_2017 import load_full_datasets

    out = {}
    for name, meta in load_full_datasets().items():
        out[str(name)] = {
            "xsec": meta.get("xsec"),
            "isData": bool(meta.get("isData", False)),
        }
    return out

def main():
    parser = argparse.ArgumentParser(description="Run bbDM processor (one file or full)")
    parser.add_argument("--full", action="store_true", help="Run on all files (full analysis)")
    parser.add_argument("--year", type=int, default=2017, help="Year (default 2017)")
    parser.add_argument(
        "--lumi-fb",
        type=float,
        default=41.5,
        help="Integrated luminosity in /fb for MC normalization (default: 41.5)",
    )
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (default: output_2017.pkl or output_2017_full.pkl)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max files per dataset (for testing full run)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run only one dataset key from fileset (useful for Condor split jobs).",
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        help="Save one accumulator per YAML dataset (legacy). Default: merge into physics groups (data, DYJets, …).",
    )
    args = parser.parse_args()

    # Config and file discovery
    from config.datasets_2017 import (
        get_filesets,
        get_full_filesets_from_yaml,
        PATH_2017,
        discover_signal_masspoint_branches,
    )
    signal_xsec_by_key = _load_signal_masspoint_xsecs()
    full_dataset_meta = _load_full_dataset_meta()
    lumi_pb = float(args.lumi_fb) * 1000.0

    # Prefer YAML if present for full analysis; fall back to dynamic discovery.
    if args.full:
        filesets = get_full_filesets_from_yaml()
    else:
        filesets = get_filesets(full=False, max_files_per_dataset=args.max_files)
    if not filesets:
        print("No files found under", PATH_2017)
        sys.exit(1)

    if args.dataset:
        if args.dataset not in filesets:
            print(f"Dataset '{args.dataset}' not found. Available keys: {len(filesets)}")
            for k in sorted(filesets.keys()):
                print(" -", k)
            sys.exit(1)
        filesets = {args.dataset: filesets[args.dataset]}

    total_files = sum(len(f) for f in filesets.values())
    print(f"Datasets: {len(filesets)}, total files: {total_files}")
    if not args.full:
        print("(One-file mode: one file per dataset)")

    # Processor
    from processor.bbdm_processor import bbDMProcessor

    try:
        from coffea import processor
        from coffea.nanoevents.schemas import NanoAODSchema
    except ImportError:
        print("coffea not found. Install with: pip install coffea")
        sys.exit(1)

    treename = "Events"  # NanoAOD tree name (often "Events")
    runner = processor.Runner(
        executor=processor.IterativeExecutor(),
        schema=NanoAODSchema,
    )

    # Run per dataset (Option A: separate histograms per dataset)
    results = {}
    for dataset_name, file_list in filesets.items():
        if not file_list:
            continue
        if args.max_files:
            file_list = file_list[: args.max_files]
        fileset = {dataset_name: file_list}
        print(f"Processing {dataset_name} ({len(file_list)} files)...")
        try:
            # Split randomized bbDM signal by all available MH3=600, Mchi=1 masspoint branches.
            if dataset_name.startswith("bbDM"):
                masspoints = discover_signal_masspoint_branches(
                    file_list, treename=treename, target_mh3=600, target_mchi=1
                )
                if masspoints:
                    print(f"  Found {len(masspoints)} signal masspoints (MH3=600):")
                    for key, branch in masspoints:
                        print(f"    - {key} <- {branch}")
                        proc = bbDMProcessor(signal_mass_branch=branch)
                        out = runner(fileset, treename, proc)
                        # Normalize split signal histograms to lumi using per-masspoint xsec.
                        # N_gen is taken from cutflow["masspoint"] filled before trigger cuts.
                        n_gen = float(out["cutflow"].get("masspoint", 0))
                        xsec_pb = signal_xsec_by_key.get(key)
                        if xsec_pb is not None and n_gen > 0.0:
                            sf = (float(xsec_pb) * lumi_pb) / n_gen
                            _scale_histograms_in_accumulator(out, sf)
                            print(
                                f"      normalized: xsec={xsec_pb:.6g} pb, "
                                f"Ngen={int(n_gen)}, lumi={args.lumi_fb:.3g}/fb, sf={sf:.6g}"
                            )
                        else:
                            print(
                                f"      warning: no masspoint normalization for {key} "
                                f"(xsec={xsec_pb}, Ngen={int(n_gen)})"
                            )
                        results[key] = out
                else:
                    print("  No MH3=600 signal masspoint branches found; keeping unsplit signal dataset.")
                    proc = bbDMProcessor()
                    out = runner(fileset, treename, proc)
                    results[dataset_name] = out
            else:
                proc = bbDMProcessor()
                out = runner(fileset, treename, proc)
                # Normalize non-data MC datasets to lumi using YAML xsec and preselection event count.
                ds_meta = full_dataset_meta.get(dataset_name, {})
                is_data = bool(ds_meta.get("isData", False))
                xsec_pb = ds_meta.get("xsec")
                n_gen = float(out["cutflow"].get("all_events", 0))
                if (not is_data) and (xsec_pb is not None) and (n_gen > 0.0):
                    sf = (float(xsec_pb) * lumi_pb) / n_gen
                    _scale_histograms_in_accumulator(out, sf)
                    print(
                        f"  normalized: xsec={float(xsec_pb):.6g} pb, "
                        f"Ngen={int(n_gen)}, lumi={args.lumi_fb:.3g}/fb, sf={sf:.6g}"
                    )
                elif not is_data:
                    print(
                        f"  warning: no MC normalization for {dataset_name} "
                        f"(xsec={xsec_pb}, Ngen={int(n_gen)})"
                    )
                results[dataset_name] = out
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not results:
        print("No results.")
        sys.exit(1)

    if not args.per_dataset:
        from config.datasets_2017 import merge_processor_results_by_group

        results = merge_processor_results_by_group(results)
        print("Merged results keys:", list(results.keys()))

    # Save to output directory
    import pickle
    os.makedirs(os.path.join(SCRIPT_DIR, OUTPUT_DIR), exist_ok=True)
    outfile = args.output
    if not outfile:
        outfile = f"output_{args.year}_full.pkl" if args.full else f"output_{args.year}.pkl"
    if not os.path.isabs(outfile):
        # Keep optional subdirectories under output/ (e.g. shards/DYJets.pkl).
        outfile = os.path.join(SCRIPT_DIR, OUTPUT_DIR, outfile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {outfile}")
    print(f"Load in a notebook: results = pickle.load(open('{outfile}','rb'))")


if __name__ == "__main__":
    main()
