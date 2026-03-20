#!/usr/bin/env python3
"""
Merge per-dataset Condor shard outputs into one pickle.

Expected shard format:
  output/shards/<DATASET>.pkl
where each shard stores {dataset_name: dict_accumulator}.
"""

import argparse
import glob
import os
import pickle
import sys
from typing import Any, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.datasets_2017 import merge_processor_results_by_group


def merge_shards(shard_glob: str) -> Dict[str, Any]:
    files = sorted(glob.glob(shard_glob))
    # Fallback for older/new Condor transfer behaviors that place shards at repo root.
    if not files and shard_glob == "output/shards/*.pkl":
        files = sorted(glob.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No shard files matched: {shard_glob}")

    merged_per_dataset: Dict[str, Any] = {}
    for path in files:
        with open(path, "rb") as f:
            shard = pickle.load(f)
        if not isinstance(shard, dict):
            raise TypeError(f"Shard is not a dict: {path}")
        for dataset_name, acc in shard.items():
            if dataset_name in merged_per_dataset:
                merged_per_dataset[dataset_name] = merged_per_dataset[dataset_name] + acc
            else:
                merged_per_dataset[dataset_name] = acc
    return merge_processor_results_by_group(merged_per_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Condor per-dataset shard pickles")
    parser.add_argument(
        "--shards",
        default="output/shards/*.pkl",
        help="Glob for shard pickle files (default: output/shards/*.pkl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/output_2017_full.pkl",
        help="Merged output pickle path (default: output/output_2017_full.pkl)",
    )
    args = parser.parse_args()

    results = merge_shards(args.shards)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Merged {len(results)} groups into {args.output}")
    print("Keys:", list(results.keys()))


if __name__ == "__main__":
    main()
