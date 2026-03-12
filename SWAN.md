# Running the bbDM exercise on CERN SWAN

This project runs on [SWAN](https://swan.cern.ch) (CERN's Jupyter service). Your project lives on EOS, so it is visible in SWAN and persists across sessions.

## 1. Configure your session (before starting)

On the SWAN start page (**Configure your session**), set:

**Session resources**

| Option   | Recommended | Notes |
|----------|-------------|--------|
| **CPU**  | 2 cores     | Enough for notebooks and one-file runs. Use 4 for heavier local tests. |
| **Memory** | 8 GB      | Sufficient for one NanoAOD file and plots. Use **16 GB** if you load multiple samples or see memory errors. |
| **GPU**  | None        | Not needed for this analysis. |

**Software**

- **Source:** LCG  
- **Software stack:** latest (e.g. 109)  
- **Platform:** AlmaLinux 9 (gcc13)  
- **Environment script:** leave empty (unless you use a CERNBox setup script)  
- **Use Python packages installed on CERNBox:** optional  

Then click **Start my Session**.

## 2. Open the project in SWAN

Open your project (e.g. `SWAN_projects/bbdm_cmsdas26` in your EOS space). You should see the notebooks, `config/`, `processor/`, `scripts/`, and `requirements.txt`.

## 3. Open a Terminal in SWAN

In the SWAN interface, open a **Terminal** (same environment and filesystem as your notebooks).

## 4. Run the setup script

In the terminal:

```bash
cd /eos/user/<your-username>/SWAN_projects/bbdm_cmsdas26
bash scripts/setup_venv.sh
```

Use the path SWAN shows for your project if different. **Use `bash scripts/setup_venv.sh`** (not `source`). This creates a `.venv`, installs dependencies from `requirements.txt`, and registers the Jupyter kernel **Python (bbDM CMS DAS)**.

## 5. Use the kernel in notebooks

In any notebook, go to **Kernel → Change kernel** and choose **Python (bbDM CMS DAS)**. Then run the notebook as usual.

## 6. Optional: Conda instead of venv

If the SWAN terminal does not have `python3` or the script fails, SWAN also offers **Conda**. You can create a Conda env in the project, install from `requirements.txt`, then run:

```bash
python -m ipykernel install --user --name=bbdm-cmsdas26 --display-name="Python (bbDM CMS DAS)"
```

and select that kernel in notebooks.

## 7. Data path

2017 inputs are under `/eos/cms/store/group/phys_susy/sus-23-008/cmsdas2026/2017` and are readable from SWAN (EOS is mounted). The config in `config/datasets_2017.py` discovers datasets there; use the optional cell in Session 1 to load one file from data and one from background.
