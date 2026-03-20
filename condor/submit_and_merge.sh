#!/bin/bash

set -euo pipefail

SUBMIT_FILE="condor/submit.sub"
MERGE_SCRIPT="condor/merge_condor_outputs.py"
mkdir -p output/shards

echo "Submitting dataset-split Condor jobs..."
SUBMIT_OUT="$(condor_submit "${SUBMIT_FILE}")"
echo "${SUBMIT_OUT}"

CLUSTER_ID="$(echo "${SUBMIT_OUT}" | sed -n 's/.*cluster \([0-9]\+\)\..*/\1/p' | head -n 1)"
if [ -z "${CLUSTER_ID}" ]; then
  echo "ERROR: Could not parse cluster id from condor_submit output."
  exit 2
fi

LOG_FILE="condor_${CLUSTER_ID}.log"
echo "Waiting for all jobs in cluster ${CLUSTER_ID} to finish..."
condor_wait "${LOG_FILE}"

echo "Merging shard outputs..."
python3 "${MERGE_SCRIPT}"
echo "Done: output/output_2017_full.pkl"
