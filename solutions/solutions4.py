# %% [markdown]
# # Session 4 — Solutions
#
# Worked solutions for weights, systematics, fitting, goodness of fit, and limit calculation. Requires processor output (output_2017.pkl or output_2017_full.pkl) from run_analysis.py.

# %% [markdown]
# ## Load results and build data / background histograms

# %%
# Ensure project root is on sys.path (for SWAN / any kernel)
import sys, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir:
    sys.path.insert(0, os.path.join(_script_dir, ".."))
# Avoid recursion limit during deep pickle load (accumulator + hist)
try:
    sys.setrecursionlimit(8000)
except Exception:
    pass

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson, chi2

import os
def _load_results():
    candidates = [
        "output/output_2017.pkl", "output/output_2017_full.pkl",
        "output_2017.pkl", "output_2017_full.pkl",
        "../output/output_2017.pkl", "../output/output_2017_full.pkl",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return pickle.load(open(path, "rb"))
    raise FileNotFoundError("Run run_analysis.py first. Outputs go to output/.")
results = _load_results()

all_datasets = list(results.keys())
data_datasets = [d for d in all_datasets if "Run2017" in d or "Single" in d or "MET" in d]
bkg_datasets = [d for d in all_datasets if d not in data_datasets]

# Main observable: cos(theta*) when available, else MET SR
def get_main_hist(h):
    return h.get("cos_theta_star_sr") or h["met_sr"]

def sum_hists(results, names):
    out = None
    for n in names:
        if n not in results:
            continue
        h = get_main_hist(results[n])
        if out is None:
            out = h.copy()
        else:
            out = out + h
    return out

data_hist = sum_hists(results, data_datasets)
bkg_hist = sum_hists(results, bkg_datasets)
if data_hist is not None:
    obs = np.asarray(data_hist.values()).flatten()
    obs = np.maximum(obs, 0)
else:
    obs = None
if bkg_hist is not None:
    bkg = np.asarray(bkg_hist.values()).flatten()
    bkg = np.maximum(bkg, 1e-6)
    if obs is None:
        obs = np.zeros(len(bkg))
else:
    bkg = np.ones(25) * 10
    if obs is None:
        obs = np.zeros(25)
print("Data sum:", obs.sum(), "Bkg sum:", bkg.sum(), "(main observable: cos_theta_star_sr if present else met_sr)")

# %% [markdown]
# ## Binned fit (one scale factor for background)

# %%
def nll(scale):
    lam = scale * bkg
    return -np.sum(poisson.logpmf(obs.astype(int), np.maximum(lam, 1e-10)))

fit = minimize(nll, x0=[1.0], bounds=[(0.01, 10)])
scale_best = fit.x[0]
expected = scale_best * bkg
print("Best-fit background scale:", scale_best)
print("Total observed:", obs.sum(), "Total expected (after fit):", expected.sum())

# %% [markdown]
# ## Goodness of fit

# %%
chi2_val = np.sum((obs - expected)**2 / np.where(expected > 0, expected, 1))
ndof = len(obs) - 1
pvalue = 1 - chi2.cdf(chi2_val, ndof)
print(f"chi2 = {chi2_val:.2f}, ndof = {ndof}, p-value = {pvalue:.4f}")
pulls = (obs - expected) / np.sqrt(np.where(expected > 0, expected, 1))
plt.figure(figsize=(8, 3))
plt.bar(range(len(pulls)), pulls, edgecolor="black", alpha=0.7)
plt.xlabel("Bin")
plt.ylabel("Pull")
plt.title("Pull distribution")
plt.axhline(0, color="gray", ls="--")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Simple 95% CL upper limit (single bin: total SR yield)

# %%
# One-bin: observed = obs.sum(), background = expected.sum(); upper limit on signal s at 95% CL
n_obs = int(obs.sum())
b = expected.sum()
# Simple approximate: s_95 such that P(n <= n_obs | b + s_95) = 0.05 (one-sided)
from scipy.stats import poisson
def p_exceed(s):
    return 1 - poisson.cdf(n_obs, b + s)
s_vals = np.linspace(0, 50, 200)
p_vals = [p_exceed(s) for s in s_vals]
idx = np.argmin(np.abs(np.array(p_vals) - 0.05))
s_95 = s_vals[idx]
print(f"Approximate 95% CL upper limit on signal events (single bin): s_95 ≈ {s_95:.1f}")
