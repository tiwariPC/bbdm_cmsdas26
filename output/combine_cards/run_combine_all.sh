#!/usr/bin/env bash
set -euo pipefail

# Run from CMSSW with Combine available.
CARDS_DIR="/eos/user/p/ptiwari/SWAN_projects/bbdm_cmsdas26/output/combine_cards"
OUT_DIR="${CARDS_DIR}"

for card in "${CARDS_DIR}"/datacard_*.txt; do
  base=$(basename "${card}" .txt)
  # Goodness-of-fit (saturated) + toy p-value
  combine -M GoodnessOfFit -d "${card}" --algo=saturated -n ."${base}"
  combine -M GoodnessOfFit -d "${card}" --algo=saturated -t 200 --toysFrequentist -n ."${base}"
  # Asymptotic 95% CL limit
  combine -M AsymptoticLimits -d "${card}" -n ."${base}"
done

mv higgsCombine*.root "${OUT_DIR}"/ 2>/dev/null || true
echo "Combine outputs in: ${OUT_DIR}"
