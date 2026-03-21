#!/bin/bash

set -euo pipefail

# Source the LCG environment
# Disable nounset only for sourcing, then restore strict mode.
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh
set -u


DATASET="${1:-}"
if [ -z "${DATASET}" ]; then
  echo "ERROR: missing dataset argument"
  exit 2
fi

cd /afs/cern.ch/work/p/ptiwari/cmsdas2026/bbdm_cmsdas26/
echo "Running dataset: ${DATASET}"
# One shard output per dataset; keep per-dataset format for robust post-merge.
python3 run_analysis.py --full --year 2017 --dataset "${DATASET}" --per-dataset -o "shards/${DATASET}.pkl"
