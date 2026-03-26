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
│   ├── datasets_2017.py                # 2017 paths, merge rules for pickle output
│   ├── datasets_2017_full.yaml         # merge_groups / merge_prefix_rules (fine-tune merged pickle keys)
│   ├── datasets_2017_short.yaml        # same merge_* kept in sync for short / one-file workflows
│   └── Cert_*_JSON*.txt                # optional: 2017 golden JSON for data lumi filtering
├── condor/
│   └── submit_condor.sub               # Condor submit for full analysis
├── processor/
│   └── bbdm_processor.py               # Reusable Coffea processor
├── run_analysis.py                      # One-file or full run (--full)
├── datasets/
│   └── dataset_guide.md                 # How to get NanoAOD samples
├── solutions/                           # Hints for the excercises
├── figures/                             # Optional diagrams
├── README.md                            # This file
```

Clone this repository in your `/eos/user/<first letter of your username>/<username>/SWAN_projects/`
```bash
git clone https://github.com/tiwariPC/bbdm_cmsdas26.git
```

### On Lxplus `LCG_105_swan`
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh
```
Then you can convert the full notebook and run it or you take code cell individually and run them. Best way is to just open SWAN session after you have cloned the repo in SWAN_projects directory in your eos area.


### On CERN SWAN with `LCG_105a`

1. CLick on [https://swan.cern.ch/hub/home](https://swan.cern.ch/hub/home)

2.  Choose `LCG_105a` when starting the SWAN session by clicking Software stack option

3. Select 4 CPUs and 16GB Memory

4. Click on `Start new Session`

You will see bbdm_cmsdas26 repo there

### Certified lumisections (golden JSON) for data

`run_analysis.py` filters **only collision-data** filesets (never MC or signal) to CMS-certified `(run, luminosityBlock)` pairs using a golden JSON **before** the trigger step. By default it looks for:

- `config/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt`

If that file is missing, data are processed without lumi masking (a warning is printed). Override the path with the environment variable `BBDM_GOLDEN_JSON` or `python run_analysis.py --golden-json /path/to/cert.json`. Use `--no-golden-json` only for debugging.

Official files: [CMS luminosity / certification](https://twiki.cern.ch/twiki/bin/view/CMSPublic/LuminosityCMS).
