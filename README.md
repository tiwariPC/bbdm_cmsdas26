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

## Prerequisites

- Python 3.8+
- Jupyter (or JupyterLab)
- Install dependencies:

### Local or non-LCG environment

```bash
pip install coffea matplotlib hist uproot
```

For full Coffea + NanoAOD (optional, for real data access):

```bash
pip install coffea[dask] xrootd
```

Or use the project’s **requirements.txt** and setup script (see below).

### On CERN SWAN with `LCG_109_swan`

When running on SWAN with the `LCG_109_swan` view, the system stack already provides recent versions of `coffea`, `awkward`, `numpy`, etc., but its `uproot` is slightly older than what Coffea expects. To work around this without touching the LCG installation:

1. Source the LCG view in your SWAN terminal:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_swan/x86_64-el9-gcc13-opt/setup.sh
cd /eos/user/<USER>/SWAN_projects/bbdm_cmsdas26
```

2. Install a newer `uproot` **into the repository-local `.local-uproot` directory** (once per user):

```bash
python -m pip install --target ./.local-uproot --no-deps 'uproot>=5.7.0'
```

3. When running Python scripts or notebooks for this exercise from the shell, prepend the repo-local `uproot` to `PYTHONPATH` so that it is found before the LCG one, while still using NumPy from the LCG stack:

```bash
PYTHONPATH="./.local-uproot:${PYTHONPATH}" python session1_intro_and_datasets.py
```

In SWAN notebooks, you can achieve the same effect by running this once in a cell at the top of the notebook:

```python
import os, sys
repo_root = "/eos/user/<USER>/SWAN_projects/bbdm_cmsdas26"
sys.path.insert(0, os.path.join(repo_root, ".local-uproot"))
```

After this, `import uproot` will use the repo-local version (with the `RNTuple` API Coffea expects), while `numpy` and `numba` continue to come from the `LCG_109_swan` environment.

### Using the setup script

From the project root, source the setup script (LCG + .local, no .venv):

```bash
source scripts/setup.sh
```

To skip sourcing the LCG view (e.g. if SWAN already provides the stack):

```bash
SKIP_LCG=1 source scripts/setup.sh
```

To start Jupyter directly from the same environment:

```bash
bash scripts/setup.sh --jupyter
```

**On CERN SWAN:** see [SWAN.md](SWAN.md) for session configuration (CPU, memory) and SWAN-specific setup steps using `scripts/setup.sh`.

## Running the Exercise

1. Start Jupyter: `jupyter notebook` or `jupyter lab`
2. Work through the notebooks in order: Session 1 → 2 → 3 → 4
3. Use small example NanoAOD files (see `datasets/dataset_guide.md`) or the processor on a teaching cluster

### Two run modes (2017 samples)

Input files for 2017 are under `/eos/cms/store/group/phys_susy/sus-23-008/cmsdas2026/2017`. The config in `config/datasets_2017.py` discovers datasets and builds file lists.

- **One-file mode:** In Session 1, use the optional cell that loads one file from data and one from background via the config. You can also run: `python run_analysis.py` (one file per dataset) and inspect the output.
- **Full analysis:** Run on all files and save merged histograms: `python run_analysis.py --full -o output_2017_full.pkl`. To run on Condor from project root: `condor_submit condor/submit_condor.sub`. See `condor/README.md`.

### Testing (Mode 1)

To smoke-test that **single-file mode** and the **Session 1–4 workflow** are working end-to-end (requires 2017 data at the configured path and the environment from `scripts/setup.sh`):

```bash
source scripts/setup.sh
python scripts/run_mode1_tests.py
```

Use `--skip-run-analysis` to reuse an existing `output_2017.pkl` and only run config + notebook-equivalent steps:

```bash
python scripts/run_mode1_tests.py --skip-run-analysis
```

## Sessions

| Session | Notebook | Focus |
|---------|----------|--------|
| 1 | `session1_intro_and_datasets.ipynb` | Dark matter motivation, CMS data formats, NanoAOD, Coffea basics, loading data, basic plots |
| 2 | `session2_object_selection.ipynb` | Jet selection, b-tagging (DeepJet), lepton veto, event cleaning |
| 3 | `session3_signal_region_analysis.ipynb` | Signal region definition, MET and yields, control regions, background composition |
| 4 | `session4_systematics_fitting_limits.ipynb` | Weights and systematics, binned fit, goodness of fit, limit calculation |

## License and Contact

This material is intended for educational use at CMS DAS-style schools. For questions, refer to `InstructorGuide.md` or your local organisers.
