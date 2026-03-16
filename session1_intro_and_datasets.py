# Ensure project root is on sys.path (for SWAN / run as script)
import sys
import os
if "config" not in sys.modules:
    _root = os.path.abspath(os.getcwd())
    while _root != os.path.dirname(_root):
        if os.path.isdir(os.path.join(_root, "config")) and os.path.isfile(os.path.join(_root, "requirements.txt")):
            sys.path.insert(0, _root)
            break
        _root = os.path.dirname(_root)

# %% [markdown]
# # Session 1 — Introduction, CMS Data, and Coffea Basics
#
# **Learning objectives:**
# - Understand dark matter collider searches
# - Understand CMS data formats and NanoAOD
# - Load data using Coffea and explore event content
# - Produce basic kinematic plots
#
# **Duration:** ~3 hours

# %% [markdown]
# ---
# ## 1. Introduction to Dark Matter
#
# About 85% of the matter in the universe does not emit light: we call it **dark matter (DM)**. Evidence comes from galaxy rotation curves, gravitational lensing, and the cosmic microwave background.
#
# At colliders like the LHC we search for DM by producing it in proton–proton collisions. If DM is stable and weakly interacting, it escapes the detector **without leaving a trace** — we infer its presence from **missing transverse energy (MET)**.

# %% [markdown]
# ### Why colliders?
#
# - We can produce DM in the lab if it couples to quarks, gluons, or electroweak particles.
# - The signature is **MET** (imbalance in transverse momentum) when DM particles leave the detector.
# - Heavy-flavor jets (b-jets) are important in many models (e.g. 2HDM+a, bbDM) where DM couples to the Higgs sector and production goes through b quarks.

# %% [markdown]
# ### Signal process (simplified)
#
# We consider a simplified model:
#
# $$pp \to b\bar{b} + \chi\bar{\chi}$$
#
# where \(\chi\) is the dark matter particle (invisible). **Experimental signature:**
# - Two or more **b-jets**
# - Large **MET**
# - **No isolated leptons** (to reduce W/Z + lepton backgrounds)

# %% [markdown]
# ---
# ## 2. Dark Matter Searches at Colliders
#
# | Observable | Role |
# |-----------|------|
# | **MET** | Indicates invisible particles carrying away momentum |
# | **b-jets** | Signal model favours production via heavy flavour |
# | **Lepton veto** | Reduces W+jets and semileptonic tt̄ |
#
# **Main backgrounds:** tt̄, Z→νν+jets, W+jets. We will define a **signal region** and compare data (or MC) to expectations.

# %% [markdown]
# ---
# ## 3. Overview of the CMS Detector
#
# CMS is a general-purpose detector at the LHC. Key components:
# - **Tracker** — charged particle momentum
# - **Calorimeters** — energy of electrons, photons, jets
# - **Muon system** — muon identification
#
# Neutrinos (and DM) do not leave signals; their total transverse momentum shows up as **missing transverse energy**.
#
# *Discussion:* Why can’t we "see" dark matter in the detector?

# %% [markdown]
# ---
# ## 4. CMS Data Formats
#
# CMS stores collision data and simulation in **ROOT** files. Processing levels:
#
# - **RAW** — raw detector data
# - **AOD** — analysis object data (reconstructed objects)
# - **NanoAOD** — reduced format with only the most used branches, designed for analysis
#
# We will use **NanoAOD** because it is small and fast to process.

# %% [markdown]
# ---
# ## 5. What is NanoAOD?
#
# NanoAOD contains:
# - **Jets** (pt, η, φ, mass, b-tag discriminants, ID)
# - **MET** (pt, φ)
# - **Electrons, Muons** (kinematics, ID, isolation)
# - **Triggers, generator info** (for MC)
#
# Branch names follow the pattern `Object_variable`, e.g. `Jet_pt`, `MET_pt`. In Coffea we access them as `events.Jet.pt`, `events.MET.pt`.

# %% [markdown]
# ---
# ## 6. Introduction to Coffea
#
# **Coffea** (Columnar Object Framework for Effective Analysis) is a Python library for HEP analysis. It:
# - Reads ROOT/NanoAOD with **uproot**
# - Uses **Awkward Array** for jagged (variable-length) data
# - Supports **columnar** operations (efficient over many events)
#
# Install (if needed):
# ```bash
# pip install coffea matplotlib hist uproot
# ```

# %%
# Check that Coffea and related packages are available
try:
    import coffea
    import awkward as ak
    import numpy as np
    import uproot as uproot
    print("coffea:", coffea.__version__)
    print("awkward:", ak.__version__)
    print("numpy:", np.__version__)
    print("uproot:", uproot.__version__)
except ImportError as e:
    print("Missing package:", e)

# %% [markdown]
# ---
# ## 7. Loading NanoAOD Files
#
# We use `NanoEventsFactory` from Coffea to load NanoAOD. You need a path to a NanoAOD file (local or XRootD).
#
# **For the school:** Your instructor will provide a small sample file path. Replace the path below with yours.

# %%
# # LCG 109 SWAN: uproot may lack RNTuple; Coffea's _is_interpretable then raises. Patch so TTrees still work.
# try:
#     import uproot.behaviors
#     if not hasattr(uproot.behaviors, "RNTuple"):
#         class _RNTupleHasFields:
#             pass
#         uproot.behaviors.RNTuple = type("RNTuple", (), {"HasFields": _RNTupleHasFields})
# except Exception:
#     pass


from coffea.nanoevents import NanoEventsFactory, BaseSchema

# Option 1: BaseSchema works with standard NanoAOD branch names
# Option 2: For full NanoAOD naming use: from coffea.nanoevents.schemas import NanoAODSchema

def load_events(filepath):
    """Load one NanoAOD file and return events."""
    return NanoEventsFactory.from_root({filepath: "Events"}, schemaclass=BaseSchema).events()

# Example: local file (replace with your file path)
filepath = "/eos/cms/store/group/phys_susy/sus-23-008/cmsdas2026/2017/MET-Run2017B-02Apr2020-v1/FA2D6DDA-28C1-8A4E-B170-B7217A9ED411.root"
events = load_events(filepath)
print("Number of events:", len(events))

print("To run this cell, set filepath to a NanoAOD file and uncomment the lines above.")

# %% [markdown]
# ### Exercise 1.1
# Load a NanoAOD file provided by your instructor. Print the total number of events. If you don’t have a file yet, you can skip this cell or use a dummy path and catch the error.

# %% [markdown]
# ### Option: load one file from data and one from background (2017)
#
# If the 2017 samples are available under the configured path, you can load one data file and one background file like this. The config is in `config/datasets_2017.py`.


# One-file mode: one file from data, one from background (uses config/datasets_2017.py)
try:
    from config.datasets_2017 import get_one_file_per_group
    filesets = get_one_file_per_group()
    if filesets:
        if "data" in filesets and filesets["data"]:
            events_data = load_events(filesets["data"][0])
            print("Data: number of events =", len(events_data))
            events = events_data
        if "background" in filesets and filesets["background"]:
            events_bkg = load_events(filesets["background"][0])
            print("Background: number of events =", len(events_bkg))
            if "events" not in dir():
                events = events_bkg
    else:
        print("No files found from config. Set filepath manually and use load_events(filepath).")
except Exception as e:
    print("Config not available or error:", e)
    print("Set filepath manually and use load_events(filepath).")

# %%
# Your code here:
# filepath = "..."  # set this
# events = load_events(filepath)
# len(events)

# %% [markdown]
# ---
# ## 8. Inspecting Event Content
#
# Once events are loaded, inspect the available collections. In Coffea, `events` is an Awkward array of events; each attribute (e.g. `Jet`, `MET`) is a jagged structure.

# %%
# Uncomment after loading events
# print("Collections:", [k for k in dir(events) if not k.startswith("_")])
# print("\nJets: ", events.Jet)
# print("MET: ", events.MET.pt[:5], "... (first 5 events)")
# print("Number of jets per event (first 10):", ak.num(events.Jet)[:10])

# %% [markdown]
# ### Key branches to explore
#
# | Collection | Useful branches |
# |------------|------------------|
# | `events.Jet` | `pt`, `eta`, `phi`, `mass`, `jetId`, `btagDeepFlavB` |
# | `events.MET` | `pt`, `phi` |
# | `events.Electron` | `pt`, `eta`, `cutBased` |
# | `events.Muon` | `pt`, `eta`, `tightId`, `pfRelIso04_all` |

# %%
# Exercise 1.2: Print the list of Jet branch names (field names)
# Hint: events.Jet.fields or dir(events.Jet)
# Your code:

# %% [markdown]
# ---
# ## 9. Basic Plotting
#
# We use **matplotlib** (or **hist**) to plot distributions. With Awkward Arrays, we often **flatten** per-object quantities (e.g. jet pT) to get one entry per object, or use per-event quantities (e.g. MET) directly.

# %%
import matplotlib.pyplot as plt

# Example: plot MET for the first 10k events (if available)
# met = events.MET.pt
# plt.figure(figsize=(6,4))
# plt.hist(ak.to_numpy(met), bins=50, range=(0, 400), edgecolor="black", alpha=0.7)
# plt.xlabel("MET [GeV]")
# plt.ylabel("Events")
# plt.title("Missing transverse energy")
# plt.show()

print("Uncomment and run after loading events.")

# %% [markdown]
# ### Exercise 1.3 — Jet pT
# Fill a histogram of **jet pT** for all jets in your sample. Use `events.Jet.pt`, flatten with `ak.flatten(events.Jet.pt)`, and plot with matplotlib. Use bins from 0 to 500 GeV.

# %% [markdown]
# ### Optional: Explore electrons and muons
# In the same way you explored jets, print the first few entries of `events.Electron.pt` and `events.Muon.pt`. How many events have at least one electron or one muon?

# %%
# Optional: uncomment when events are loaded
# print("Electron pT (first 5 events):", events.Electron.pt[:5])
# print("Muon pT (first 5 events):", events.Muon.pt[:5])
# n_ele = ak.count(events.Electron.pt, axis=1)
# n_mu = ak.count(events.Muon.pt, axis=1)
# print("Events with >=1 electron:", ak.sum(n_ele >= 1))
# print("Events with >=1 muon:", ak.sum(n_mu >= 1))

# %%
# Your code: flatten Jet pt and plot
# jpt = ak.flatten(events.Jet.pt)
# plt.hist(ak.to_numpy(jpt), bins=50, range=(0, 500), ...)

# %% [markdown]
# ### Exercise 1.4 — Jet multiplicity
# Plot the **number of jets per event** (use `ak.num(events.Jet)`). Use bins from 0 to 15. Label axes and add a title.

# %%
# Your code: jet multiplicity
# njets = ak.num(events.Jet)
# plt.hist(ak.to_numpy(njets), bins=15, range=(0, 15), ...)

# %% [markdown]
# ### Exercise 1.5 — MET distribution
# Produce a **MET** distribution (same as in the example above). Use range 0–400 GeV and about 50 bins. Add axis labels and a title.

# %%
# Your code: MET distribution
# met = events.MET.pt
# plt.hist(ak.to_numpy(met), bins=50, range=(0, 400), ...)

# %% [markdown]
# ---
# ## 10. Summary — Session 1
#
# - **Dark matter** at colliders is searched for via MET and (in our case) b-jets.
# - **NanoAOD** is a reduced ROOT format; we access it with **Coffea** and **Awkward Arrays**.
# - **Loading:** `NanoEventsFactory.from_root({filepath: "Events"}, schemaclass=NanoAODSchema).events()`
# - **Inspecting:** `events.Jet`, `events.MET`, etc.; use `ak.flatten` for per-object plots and `ak.num` for multiplicities.
# - **Plotting:** Use matplotlib (or hist) with `ak.to_numpy()` for histogram inputs.
#
# **Next session:** We will apply jet quality cuts, b-tagging, and lepton veto to define a clean object selection.
