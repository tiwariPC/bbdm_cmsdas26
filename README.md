# bbDM DAS Long Exercise

**Search for Dark Matter with b-jets and Missing Transverse Energy (bbDM) using CMS Run-2 data**

A complete teaching package for a long exercise similar to those used in the CMS Data Analysis School (DAS). The exercise introduces a simplified dark matter search at the LHC using the CMS experiment.

## Overview

- **Topic**: pp → b b̄ + χ χ̄ (invisible dark matter)
- **Signature**: two or more b-jets, large MET, no isolated leptons
- **Backgrounds**: tt̄, Z→νν+jets, W+jets
- **Audience**: PhD students or early-stage researchers (basic Python, no Coffea/NanoAOD experience)
- **Duration**: 12 hours total (4 sessions × 3 hours)

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
│   ├── setup_venv.sh                    # Create .venv and install deps
│   └── start.sh                         # Activate venv, optional --jupyter
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

## Prerequisites

- Python 3.8+
- Jupyter (or JupyterLab)
- Install dependencies:

```bash
pip install coffea matplotlib hist uproot
```

For full Coffea + NanoAOD (optional, for real data access):

```bash
pip install coffea[dask] xrootd
```

Or use the project’s **requirements.txt** and setup script (see below).

### Using the setup script

From the project root, create a virtual environment and install dependencies:

```bash
bash scripts/setup_venv.sh
```

Then activate and optionally start Jupyter:

```bash
source scripts/start.sh          # activate only
bash scripts/start.sh --jupyter  # activate and start Jupyter
```

**On CERN SWAN:** see [SWAN.md](SWAN.md) for session configuration (CPU, memory), setup steps, and kernel selection.

## Running the Exercise

1. Start Jupyter: `jupyter notebook` or `jupyter lab`
2. Work through the notebooks in order: Session 1 → 2 → 3 → 4
3. Use small example NanoAOD files (see `datasets/dataset_guide.md`) or the processor on a teaching cluster

### Two run modes (2017 samples)

Input files for 2017 are under `/eos/cms/store/group/phys_susy/sus-23-008/cmsdas2026/2017`. The config in `config/datasets_2017.py` discovers datasets and builds file lists.

- **One-file mode:** In Session 1, use the optional cell that loads one file from data and one from background via the config. You can also run: `python run_analysis.py` (one file per dataset) and inspect the output.
- **Full analysis:** Run on all files and save merged histograms: `python run_analysis.py --full -o output_2017_full.pkl`. To run on Condor from project root: `condor_submit condor/submit_condor.sub`. See `condor/README.md`.

## Sessions

| Session | Notebook | Focus |
|---------|----------|--------|
| 1 | `session1_intro_and_datasets.ipynb` | Dark matter motivation, CMS data formats, NanoAOD, Coffea basics, loading data, basic plots |
| 2 | `session2_object_selection.ipynb` | Jet selection, b-tagging (DeepJet), lepton veto, event cleaning |
| 3 | `session3_signal_region_analysis.ipynb` | Signal region definition, MET and yields, control regions, background composition |
| 4 | `session4_systematics_fitting_limits.ipynb` | Weights and systematics, binned fit, goodness of fit, limit calculation |

## License and Contact

This material is intended for educational use at CMS DAS-style schools. For questions, refer to `InstructorGuide.md` or your local organisers.
