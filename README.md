# bbDM DAS Long Exercise

**Search for Dark Matter with b-jets and Missing Transverse Energy (bbDM) using CMS Run-2 data**

A complete teaching package for a long exercise similar to those used in the CMS Data Analysis School (DAS). The exercise introduces a simplified dark matter search at the LHC using the CMS experiment.

## Overview

- **Topic**: pp → b b̄ + χ χ̄ (invisible dark matter)
- **Signature**: two or more b-jets, large MET, no isolated leptons
- **Backgrounds**: tt̄, Z→νν+jets, W+jets
- **Audience**: PhD students or early-stage researchers (basic Python, no Coffea/NanoAOD experience)

## Folder Structure

```
bbDM_DAS_LongExercise/
├── session1_intro_and_datasets.ipynb    # Session 1: Intro, CMS data, Coffea
├── session2_object_selection.ipynb      # Session 2: Object selection, b-tagging
├── session3_signal_region_analysis.ipynb # Session 3: Signal region, yields
├── session4_systematics_fitting_limits.ipynb # Session 4: Weights, systematics, fit, limits
├── config/
│   └── datasets_2017.py                # 2017 paths and file discovery
├── condor/
│   └── submit_condor.sub               # Condor submit for full analysis
├── processor/
│   └── bbdm_processor.py               # Reusable Coffea processor
├── run_analysis.py                      # One-file or full run (--full)
├── requirements.txt                     # Python dependencies
├── scripts/
│   ├── setup.sh                         # Source LCG, install deps into .local, optional --jupyter
│   └── run_mode1_tests.py               # Mode 1 smoke test (config, run_analysis, S1–S4)
├── tests/
│   ├── test_mode1_config.py             # Config and file discovery
│   ├── test_mode1_run_analysis.py       # Processor run and pkl structure
│   └── test_mode1_notebooks.py          # Session 1–4 notebook-equivalent logic
├── SWAN.md                              # Running on CERN SWAN
├── datasets/
│   └── dataset_guide.md                 # How to get NanoAOD samples
├── solutions/
│   ├── solutions_session1.ipynb
│   ├── solutions_session2.ipynb
│   ├── solutions_session3.ipynb
│   └── solutions_session4.ipynb
├── figures/                             # Optional diagrams
├── README.md                            # This file
└── InstructorGuide.md                   # Teaching notes and expected results
```

### On Lxplus `LCG_105_swan`
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh
```


### On CERN SWAN with `LCG_105a`


1. Choose `LCG_105a` when starting the SWAN session


2. Select 4 CPUs and 16GB Memory


3. Click on `Start new Session`
