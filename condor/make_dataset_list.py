#!/usr/bin/env python3
"""Generate condor/datasets_2017_full.txt from datasets_2017_full.yaml."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.datasets_2017 import load_full_datasets


def main() -> None:
    names = list(load_full_datasets().keys())
    out = os.path.join(SCRIPT_DIR, "datasets_2017_full.txt")
    with open(out, "w") as f:
        for name in names:
            f.write(f"{name}\n")
    print(f"Wrote {len(names)} dataset names to {out}")


if __name__ == "__main__":
    main()
