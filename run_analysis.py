#!/usr/bin/env python3
"""
Run bbDM processor: one-file mode (quick) or full analysis over all files.

Usage:
  python run_analysis.py                    # one file per dataset
  python run_analysis.py --full              # all files (full analysis)
  python run_analysis.py --full --year 2017   # full analysis, 2017
  python run_analysis.py --full -o output_2017.coffea

Output is saved as a pickle in the output/ directory: dict of dataset_name -> accumulator
(histograms, cutflow). Load in a notebook from output/output_2017.pkl or output/output_2017_full.pkl.
"""

import argparse
import os
import sys

# Project root and output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "output"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

def main():
    parser = argparse.ArgumentParser(description="Run bbDM processor (one file or full)")
    parser.add_argument("--full", action="store_true", help="Run on all files (full analysis)")
    parser.add_argument("--year", type=int, default=2017, help="Year (default 2017)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (default: output_2017.pkl or output_2017_full.pkl)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max files per dataset (for testing full run)")
    args = parser.parse_args()

    # Config and file discovery
    from config.datasets_2017 import (
        get_filesets,
        get_full_filesets_from_yaml,
        PATH_2017,
    )

    # Prefer YAML if present for full analysis; fall back to dynamic discovery.
    if args.full:
        filesets = get_full_filesets_from_yaml()
    else:
        filesets = get_filesets(full=False, max_files_per_dataset=args.max_files)
    if not filesets:
        print("No files found under", PATH_2017)
        sys.exit(1)

    total_files = sum(len(f) for f in filesets.values())
    print(f"Datasets: {len(filesets)}, total files: {total_files}")
    if not args.full:
        print("(One-file mode: one file per dataset)")

    # Processor
    from processor.bbdm_processor import bbDMProcessor

    try:
        from coffea.processor import run_uproot_job, IterativeExecutor
        use_runner = False
    except ImportError:
        try:
            from coffea import processor
            use_runner = True
        except ImportError:
            print("coffea not found. Install with: pip install coffea")
            sys.exit(1)

    treename = "Events"  # NanoAOD tree name (often "Events")
    proc = bbDMProcessor()

    # Run per dataset (Option A: separate histograms per dataset)
    results = {}
    for dataset_name, file_list in filesets.items():
        if not file_list:
            continue
        fileset = {dataset_name: file_list}
        print(f"Processing {dataset_name} ({len(file_list)} files)...")
        try:
            if use_runner:
                runner = processor.Runner(executor=processor.IterativeExecutor())
                out = runner(fileset, treename=treename, processor_instance=proc)
            else:
                out = run_uproot_job(
                    fileset,
                    treename,
                    proc,
                    executor=IterativeExecutor(),
                )
            results[dataset_name] = out
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not results:
        print("No results.")
        sys.exit(1)

    # Save to output directory
    import pickle
    os.makedirs(os.path.join(SCRIPT_DIR, OUTPUT_DIR), exist_ok=True)
    outfile = args.output
    if not outfile:
        outfile = f"output_{args.year}_full.pkl" if args.full else f"output_{args.year}.pkl"
    if not os.path.isabs(outfile):
        outfile = os.path.join(SCRIPT_DIR, OUTPUT_DIR, os.path.basename(outfile))

    with open(outfile, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {outfile}")
    print(f"Load in a notebook: results = pickle.load(open('{outfile}','rb'))")


if __name__ == "__main__":
    main()
