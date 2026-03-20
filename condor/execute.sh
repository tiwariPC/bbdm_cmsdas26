#!/bin/bash

set -euo pipefail

# Source the LCG environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh

# (optional) print env for debugging
echo "Python path: $(which python3)"
python3 --version

DATASET="${1:-}"
if [ -z "${DATASET}" ]; then
  echo "ERROR: missing dataset argument"
  exit 2
fi

echo "Running dataset: ${DATASET}"
# One shard output per dataset; keep per-dataset format for robust post-merge.
python3 run_analysis.py --full --year 2017 --dataset "${DATASET}" --per-dataset -o "shards/${DATASET}.pkl"
