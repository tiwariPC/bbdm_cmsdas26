# %% [markdown]
# # Session 1 — Solutions
#
# This notebook contains worked solutions for Session 1.

# %% [markdown]
# ## Loading events and counting (Ex 1.1)
#

# %%
# from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# NanoAODSchema.warn_missing_crossrefs = False


# def load_events(filepath):
#     return NanoEventsFactory.from_root(
#         {filepath: "Events"},
#         schemaclass=NanoAODSchema,
#         metadata={"dataset": "nanoaod"},
#         mode="eager",
#     ).events()

import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema


def load_events(filepath):
    """Load one NanoAOD file and return events."""
    # return NanoEventsFactory.from_root({filepath: "Events"}, schemaclass=NanoAODSchema).events()
    return NanoEventsFactory.from_root(filepath, schemaclass=NanoAODSchema).events()
# Example solution for Exercise 1.1:
# The student chooses a NanoAOD file path themselves.
# filepath = "/Users/ptiwari/Development/cmsdas2026/samples/MET-Run2017B-02Apr2020-v1/B70D2647-A99A-074D-B064-D862B2B36894.root"  # replace with a real file
# # For the school, this should be one of the provided NanoAOD files.

# # Load events and print basic information
# # (in a live solution, this cell would be run with a real filepath)
# try:
#     events = load_events(filepath)
#     print("Total number of events:", len(events))
#     print("run (first 5):", events.run[:5])
#     print("luminosityBlock (first 5):", events.luminosityBlock[:5])
#     print("event (first 5):", events.event[:5])
# except Exception as e:
#     print("Could not load events — set a valid filepath.")
#     print(e)

import sys
sys.path.append("..")
from config.datasets_2017 import get_one_file_per_group_from_yaml
files = get_one_file_per_group_from_yaml()
data_file = files["data"][0]
bkg_file = files["background"][0]
print(data_file, bkg_file)
events_data = load_events(data_file)
events_bkg = load_events(bkg_file)
events= load_events(bkg_file)
print(events.fields)

# %%
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep

hep.style.use("CMS")

# %% [markdown]
# ## Inspect branches (Ex 1.2)
#

# %%
print("Jet fields:", events.Jet.fields)

# %% [markdown]
# ## Basic plotting (no specific exercise)
#
# As a first example of plotting with matplotlib, we can make a simple MET distribution using the loaded `events` object.

# %%
# Example: simple MET histogram
met = events.MET.pt
plt.figure()
plt.hist(ak.to_numpy(met), bins=50, range=(0, 400), edgecolor="black", alpha=0.7)
plt.xlabel("MET [GeV]")
plt.ylabel("Events")
plt.title("Missing transverse energy")
plt.show()

# %% [markdown]
# ## Jet pT plot (Ex 1.3)
#
# Fill a histogram of jet transverse momentum for all jets in the sample.

# %%
# Solution: jet pT histogram
jpt = ak.flatten(events.Jet.pt)
plt.figure()
plt.hist(ak.to_numpy(jpt), bins=50, range=(0, 500), edgecolor="black", alpha=0.7)
plt.xlabel("Jet p$_T$ [GeV]")
plt.ylabel("Jets")
plt.title("Jet transverse momentum")
plt.show()

# %% [markdown]
# ## Jet multiplicity (Ex 1.4)
#
# Plot the distribution of the number of jets per event.

# %%
# Solution: jet multiplicity histogram
njets = ak.num(events.Jet)
plt.figure()
plt.hist(ak.to_numpy(njets), bins=15, range=(0, 15), edgecolor="black", alpha=0.7)
plt.xlabel("Jet multiplicity")
plt.ylabel("Events")
plt.title("Number of jets per event")
plt.show()

# %% [markdown]
# ## Lepton content (Ex 1.5)
#
# Inspect basic kinematics for electrons and muons using the loaded `events` object:
# - Print the first few values of **pt, eta, and phi** for `events.Electron` and `events.Muon`.
# - Count how many events have at least one electron or at least one muon.
#

# %%

# Solution: lepton content
print("Electron pt (first 5 events):", events.Electron.pt[:5])
print("Electron eta (first 5 events):", events.Electron.eta[:5])
print("Electron phi (first 5 events):", events.Electron.phi[:5])

print("Muon pt (first 5 events):", events.Muon.pt[:5])
print("Muon eta (first 5 events):", events.Muon.eta[:5])
print("Muon phi (first 5 events):", events.Muon.phi[:5])

n_ele = ak.count(events.Electron.pt, axis=1)
n_mu = ak.count(events.Muon.pt, axis=1)
print("Events with >=1 electron:", ak.sum(n_ele >= 1))
print("Events with >=1 muon:", ak.sum(n_mu >= 1))

# %% [markdown]
# ## Lepton kinematic plots (Ex 1.6)
#
# Make simple histograms of lepton kinematics:
# - Plot pt for electrons and muons (e.g. 0–200 GeV),
# - Plot eta for electrons and muons,
# - Plot phi for electrons and muons.
#

# %%

# Example solution: lepton kinematic histograms
# Flatten per-lepton quantities
el_pt = ak.flatten(events.Electron.pt)
el_eta = ak.flatten(events.Electron.eta)
el_phi = ak.flatten(events.Electron.phi)

mu_pt = ak.flatten(events.Muon.pt)
mu_eta = ak.flatten(events.Muon.eta)
mu_phi = ak.flatten(events.Muon.phi)

# Electron pt
plt.figure()
plt.hist(ak.to_numpy(el_pt), bins=50, range=(0, 200), edgecolor="black", alpha=0.7)
plt.xlabel("Electron p$_T$ [GeV]")
plt.ylabel("Electrons")
plt.title("Electron transverse momentum")
plt.show()

# Muon pt
plt.figure()
plt.hist(ak.to_numpy(mu_pt), bins=50, range=(0, 200), edgecolor="black", alpha=0.7)
plt.xlabel("Muon p$_T$ [GeV]")
plt.ylabel("Muons")
plt.title("Muon transverse momentum")
plt.show()

# (Eta/phi plots would be analogous.)
