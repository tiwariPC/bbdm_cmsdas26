# Dataset Guide — bbDM Long Exercise

This guide explains how to find and use CMS NanoAOD samples for the bbDM long exercise. For the school, small example files or a limited set of files on a teaching cluster are recommended.

---

## 1. Querying CMS datasets with DAS

The **Data Aggregation System (DAS)** is used to discover CMS datasets and file locations.

### Basic DAS commands

From lxplus or a machine with CMS environment:

```bash
# List datasets matching a pattern (e.g. NanoAOD for 2017)
dasgoclient -query="dataset dataset=/TT*/*RunIISummer*17*NanoAOD*"

# List files for a given dataset
dasgoclient -query="file dataset=/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM" limit=5

# Get a single file (XRootD path) for testing
dasgoclient -query="file dataset=/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM" limit=1
```

### Finding child datasets (e.g. NanoAOD from MiniAOD)

If you have a **parent** dataset (e.g. MiniAOD), use the **child** query to list datasets that were produced from it (e.g. NanoAOD):

```bash
dasgoclient -query="child dataset=/DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM"
```

This returns the child dataset(s)—often one or more NanoAOD (or other derived) datasets. You can then use the returned dataset name in a **file** query to get XRootD paths. To find NanoAOD from any MiniAOD:

```bash
# Replace with your MiniAOD dataset name
dasgoclient -query="child dataset=/YourProcess/YourCampaign-MiniAOD.../MINIAODSIM"
```

### Web interface

- DAS web: <https://cmsweb.cern.ch/das/>
- Example query: `dataset=/TT*/*NanoAOD*`

---

## 2. Example NanoAOD samples (Run-2)

Use **UL (UltraLegacy)** NanoAOD when available. Example dataset names (format may vary by production):

### Backgrounds

| Process        | Example dataset name (pattern) |
|----------------|----------------------------------|
| **tt̄ (semi-leptonic)** | `TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-.../NANOAODSIM` |
| **tt̄ (hadronic)**     | `TTToHadronic_TuneCP5_13TeV-powheg-pythia8/...` |
| **Z → νν + jets**     | `ZJetsToNuNu_HT-*_TuneCP5_13TeV-madgraphMLM-pythia8/...` or `DYJetsToNuNu_...` |
| **W + jets**          | `WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/...` or `WJetsToLNu_HT-*_...` |
| **QCD (multijet)**    | `QCD_HT*_TuneCP5_13TeV-madgraph-pythia8/...` |

### Signal (if available)

- bbDM or 2HDM+a style signals may appear under names like `*bbDM*` or `*2HDMa*` in NanoAOD; check your analysis group or MC production.

---

## 3. XRootD paths

DAS returns file paths. Use them with **XRootD** (prefix `root://`) when reading from the grid:

```text
root://xrootd-cms.infn.it//store/mc/RunIISummer20UL17NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/...
```

In Python (Coffea / uproot), you can pass such a path directly:

```python
files = [
    "root://xrootd-cms.infn.it//store/mc/.../file1.root",
    "root://xrootd-cms.infn.it//store/mc/.../file2.root",
]
events = NanoEventsFactory.from_root(files, schemaclass=NanoAODSchema).events()
```

You need **xrootd** and **auth** (e.g. grid certificate or VOMS proxy) when reading from CERN/INFN storage.

---

## 4. Small example files for the school

- **Option A**: Use 1–2 small NanoAOD files per process (e.g. one file of TTToSemiLeptonic, one of ZJetsToNuNu). Limit to a few thousand events per file if you pre-skim.
- **Option B**: Pre-download a few files to a shared teaching directory and point students to local paths, e.g. `/data/bbDM_DAS/TTToSemiLeptonic_1.root`.
- **Option C**: Use the **CMS Open Data** NanoAOD-style releases if the exercise is adapted for Open Data; check <http://opendata.cern.ch> and the corresponding NanoAOD documentation.

Always document in the notebook or README:
- The exact dataset name or file list used
- The NanoAOD version (e.g. v9) and campaign (e.g. RunIISummer20UL17) so students know which branch names and IDs to use.

---

## 5. NanoAOD branches used in this exercise

Relevant collections (names as in standard NanoAOD):

- **Jets**: `Jet_pt`, `Jet_eta`, `Jet_phi`, `Jet_mass`, `Jet_jetId`, `Jet_btagDeepFlavB`
- **MET**: `MET_pt`, `MET_phi`
- **Electrons**: `Electron_pt`, `Electron_eta`, `Electron_cutBased`, etc.
- **Muons**: `Muon_pt`, `Muon_eta`, `Muon_tightId`, `Muon_pfRelIso04_all`, etc.
- **Gen (MC)**: `genWeight`

With **NanoEventsFactory** and **NanoAODSchema**, these are accessed as `events.Jet.pt`, `events.MET.pt`, etc.

---

## 6. Summary

1. Use **DAS** to find datasets and file lists.
2. Prefer **UL NanoAOD** (e.g. RunIISummer20UL17NanoAODv9) for Run-2.
3. Use **XRootD** paths when reading from the grid; use local paths for pre-downloaded files.
4. For the school, use a **small set of files** (e.g. 1–2 per process) and document dataset names and NanoAOD version in the exercise.
