# Condor full analysis

To run the full analysis (all files) on Condor:

1. From the **project root** (`bbdm_cmsdas26`):
   ```bash
   condor_submit condor/submit.sub
   ```

2. The job runs `run_analysis.py --full --year 2017` and writes `output_2017_full.pkl` to the job directory. Condor transfers it back (see `transfer_output_files`).

3. Load the output in a notebook to make comparison plots:
   ```python
   import pickle
   results = pickle.load(open("output_2017_full.pkl", "rb"))
   # results[dataset_name]["recoil"].plot()  etc.
   ```

Ensure your environment has `coffea`, `uproot`, `awkward` (e.g. on lxplus or with a conda env). Input files are read from EOS (`config.datasets_2017.PATH_2017`).
