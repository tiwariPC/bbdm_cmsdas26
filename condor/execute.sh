#!/bin/bash

# Source the LCG environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh

# (optional) print env for debugging
echo "Python path: $(which python3)"
python3 --version

# Run your analysis
python3 run_analysis.py --full --year 2017 -o output_2017_full.pkl