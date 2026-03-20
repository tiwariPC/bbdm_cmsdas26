# Condor full analysis

This setup submits **one Condor job per dataset** (from `condor/datasets_2017_full.txt`), which avoids a single long job getting restarted.

## Run

1. From the project root (`bbdm_cmsdas26`), submit all dataset jobs:
   ```bash
   python3 condor/make_dataset_list.py
   condor_submit condor/submit.sub
   ```

2. After jobs finish, merge shards into one output file:
   ```bash
   python3 condor/merge_condor_outputs.py
   ```

Optional one-command flow (submit, wait, merge):
```bash
condor/submit_and_merge.sh
```

This writes:
- Shards: `output/shards/<DATASET>.pkl`
- Final merged output: `output/output_2017_full.pkl`

## Notebook load

```python
import pickle
results = pickle.load(open("output/output_2017_full.pkl", "rb"))
```

Ensure your environment has `coffea`, `uproot`, `awkward` (e.g. lxplus or conda env). Input files are read from EOS (`config.datasets_2017.PATH_2017`).
