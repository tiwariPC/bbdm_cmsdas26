"""
bbDM Coffea Processor for CMS NanoAOD

Reads NanoAOD, applies object and event selection for a simplified
bb + MET dark matter search, and fills histograms.

Signal region: MET > 200 GeV, >= 2 b-jets, 0 isolated leptons.
"""

import os
import warnings
from typing import Optional

import numpy as np
import awkward as ak
from coffea import processor
from coffea.lumi_tools import LumiMask
from coffea.processor import dict_accumulator, defaultdict_accumulator
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from hist import Hist, axis

from config.datasets_2017 import get_trigger_list

NanoAODSchema.warn_missing_crossrefs = False


def get_nanoevents(filepath, schemaclass=NanoAODSchema, max_entries=None, treename="Events"):
    """
    Load one NanoAOD file and return events.

    Parameters
    ----------
    filepath : str
        Input ROOT file path.
    schemaclass : coffea schema, optional
        NanoAOD schema class.
    max_entries : int or None, optional
        If provided and >0, read at most this many entries from the file.
    """
    kwargs = {"schemaclass": schemaclass}
    if max_entries is not None and int(max_entries) > 0:
        kwargs["entry_stop"] = int(max_entries)
    # Uproot/Coffea combos can behave differently for remote ROOT files.
    # Try specifying the tree explicitly to avoid ReadOnlyDirectory issues,
    # but keep a safe fallback for older/newer coffea behaviors.
    try:
        return NanoEventsFactory.from_root(filepath, treepath=treename, **kwargs).events()
    except Exception as e1:
        # Only retry with dict form for the specific ReadOnlyDirectory/typenames-style failure.
        msg = str(e1)
        if "ReadOnlyDirectory" in msg or "typenames" in msg:
            return NanoEventsFactory.from_root({filepath: treename}, **kwargs).events()
        raise


class HistAccumulator(processor.AccumulatorABC):
    """
    Wrap a `hist.Hist` so it can live inside `dict_accumulator`.

    `coffea.processor.dict_accumulator` expects values that implement the
    `AccumulatorABC` interface (`identity()` + in-place `add()`).
    """

    def __init__(self, hist_obj: Hist):
        self._hist = hist_obj

    def identity(self):
        h = self._hist.copy()
        # Zero all bins (including flow); view can be (value, variance) in hist/boost_histogram
        v = h.view(flow=True)
        if hasattr(v.dtype, "names") and v.dtype.names:
            for name in v.dtype.names:
                v[name][...] = 0
        else:
            v[...] = 0
        return HistAccumulator(h)

    def add(self, other):
        if not isinstance(other, HistAccumulator):
            raise ValueError(f"Cannot add HistAccumulator with {type(other)}")
        self._hist += other._hist

    def __add__(self, other):
        """Support Hist + HistAccumulator and HistAccumulator + Hist for Session 4 sum_hists."""
        from hist import Hist
        if isinstance(other, HistAccumulator):
            return self._hist + other._hist
        if isinstance(other, Hist):
            return self._hist + other
        return NotImplemented

    def __radd__(self, other):
        from hist import Hist
        if isinstance(other, Hist):
            return other + self._hist
        return NotImplemented

    def __getattr__(self, name):
        if name == "_hist":
            return object.__getattribute__(self, "_hist")
        return getattr(self._hist, name)

    def __getstate__(self):
        return {"_hist": self._hist}

    def __setstate__(self, state):
        if isinstance(state, dict) and "_hist" in state:
            self._hist = state["_hist"]
        elif isinstance(state, (tuple, list)) and len(state) == 2 and isinstance(state[1], dict):
            # State is from the wrapped hist.Hist (pickle stored Hist's state for our slot)
            from hist import Hist
            self._hist = Hist.__new__(Hist)
            self._hist.__setstate__(state)
        else:
            self._hist = state[0] if (isinstance(state, (tuple, list)) and len(state) > 0) else state

    def __repr__(self):
        return f"HistAccumulator({self._hist!r})"


# -----------------------------------------------------------------------------
# Selection constants (Run-2 typical values; adjust for your analysis)
# -----------------------------------------------------------------------------
JET_PT_MIN = 30.0   # GeV
JET_ETA_MAX = 2.4
BTAG_WP_MEDIUM = 0.2783   # DeepFlavB medium working point (2017)
RECOIL_MIN = 250.0  # GeV, SR threshold on MET (named recoil for consistent naming with CR)
MET_SR_MIN = RECOIL_MIN  # backward-compatible alias
LEP_PT_MIN = 10.0   # GeV
LEP_ETA_MAX = 2.5
# Preselection:
#   SR: | pTmiss(PF)/pTmiss(calo) - 1 |
#   CR: | pTmiss(PF)/U(calo) - 1 |
MET_PF_CALO_DELTA_MAX = 0.5
REGION_NAMES = ("sr", "zecr", "zmucr", "tecr", "tmucr")
LEAD_JET_PT_MIN_CR = 100.0
Z_CR_MLL_LO, Z_CR_MLL_HI = 70.0, 110.0
LEAD_LEP_PT_CR = 30.0
TOP_CR_MT_MAX = 160.0


def met_pf_calo_delta_sr(events):
    """
    SR definition: | pTmiss(PF)/pTmiss(calo) - 1 |.

    Uses ``MET`` and ``CaloMET`` collections when ``CaloMET`` exists in the file.
    Returns ``None`` if ``CaloMET`` is missing (e.g. custom skims) — callers skip the cut.
    """
    if not hasattr(events, "CaloMET"):
        return None
    calo = events.CaloMET
    if not hasattr(calo, "pt") or not hasattr(calo, "phi"):
        return None
    met_pt = events.MET.pt
    calo_pt = ak.where(calo.pt > 0, calo.pt, 1.0)
    return np.abs((met_pt / calo_pt) - 1.0)


def met_pf_calo_delta_cr(events, sum_lep_px, sum_lep_py):
    """
    CR definition: | pTmiss(PF)/U(calo) - 1 | with
      U(calo) = | -(pTmiss(calo) + sum pT(leptons)) |.
    """
    if not hasattr(events, "CaloMET"):
        return None
    calo = events.CaloMET
    if not hasattr(calo, "pt") or not hasattr(calo, "phi"):
        return None
    c_pt = calo.pt
    c_phi = calo.phi
    c_x = c_pt * np.cos(c_phi)
    c_y = c_pt * np.sin(c_phi)
    u_x = -(c_x + sum_lep_px)
    u_y = -(c_y + sum_lep_py)
    u_calo = np.hypot(u_x, u_y)
    u_calo = ak.where(u_calo > 0, u_calo, 1.0)
    met_pt = events.MET.pt
    return np.abs((met_pt / u_calo) - 1.0)


def met_pf_calo_mask(
    events,
    max_delta=MET_PF_CALO_DELTA_MAX,
    mode="sr",
    sum_lep_px=None,
    sum_lep_py=None,
):
    """
    ``True`` if the PF/calo consistency variable is < max_delta.

    - mode="sr": | pTmiss(PF)/pTmiss(calo) - 1 |
    - mode="cr": | pTmiss(PF)/U(calo) - 1 |

    If CaloMET is unavailable, all events pass.
    """
    if mode == "cr":
        if sum_lep_px is None or sum_lep_py is None:
            raise ValueError("mode='cr' requires sum_lep_px and sum_lep_py")
        d = met_pf_calo_delta_cr(events, sum_lep_px, sum_lep_py)
    else:
        d = met_pf_calo_delta_sr(events)
    if d is None:
        return ak.ones_like(events.event, dtype=bool)
    return d < max_delta


def met_pf_calo_consistency_mask(*args, **kwargs):
    """Backward-compatible alias; prefer ``met_pf_calo_mask``."""
    return met_pf_calo_mask(*args, **kwargs)


def select_good_jets(events):
    """
    Select jets with pt > JET_PT_MIN, |eta| < JET_ETA_MAX.
    Assumes NanoAOD branch names: Jet.pt, Jet.eta, Jet.jetId.
    """
    jets = events.Jet
    mask = (
        (jets.pt > JET_PT_MIN) &
        (np.abs(jets.eta) < JET_ETA_MAX) &
        (jets.jetId >= 2)   # tight jet ID
    )
    return jets[mask]


def min_dphi_jets_recoil(jets, recoil_phi):
    """
    Minimum Δφ between any selected jet and recoil direction (per event).
    Returns array of shape (nEvents,).
    """
    dphi = np.abs(jets.phi - recoil_phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return ak.min(dphi, axis=1)


def min_dphi_jets_met(jets, met_phi):
    """Backward-compatible alias; prefer ``min_dphi_jets_recoil``."""
    return min_dphi_jets_recoil(jets, met_phi)


def count_bjets(jets, wp=BTAG_WP_MEDIUM):
    """Count jets passing b-tag working point (DeepFlavB)."""
    return ak.sum(jets.btagDeepFlavB > wp, axis=1)


def select_tight_electrons(events):
    """Tight electron selection for control regions."""
    ele = events.Electron
    mask = (
        (ele.pt > LEP_PT_MIN) &
        (np.abs(ele.eta) < LEP_ETA_MAX) &
        (ele.cutBased >= 2)   # tight cut-based ID
    )
    return ele[mask]


def select_tight_muons(events):
    """Tight muon selection for control regions."""
    mu = events.Muon
    mask = (
        (mu.pt > LEP_PT_MIN) &
        (np.abs(mu.eta) < LEP_ETA_MAX) &
        (mu.tightId) &
        (mu.pfRelIso04_all < 0.15)
    )
    return mu[mask]


def n_tight_leptons(events):
    """Number of tight electrons + tight muons per event."""
    nele = ak.count(select_tight_electrons(events).pt, axis=1)
    nmu = ak.count(select_tight_muons(events).pt, axis=1)
    return nele + nmu


def select_electrons(events):
    """Electron selection for SR lepton veto."""
    ele = events.Electron
    mask = (
        (ele.pt > LEP_PT_MIN) &
        (np.abs(ele.eta) < LEP_ETA_MAX) &
        (ele.cutBased >= 1)   # veto/loose WP
    )
    return ele[mask]


def select_muons(events):
    """Muon selection for SR lepton veto."""
    mu = events.Muon
    mask = (
        (mu.pt > LEP_PT_MIN) &
        (np.abs(mu.eta) < LEP_ETA_MAX) &
        (mu.softId) &
        (mu.pfRelIso04_all < 0.25)
    )
    return mu[mask]


def n_leptons(events):
    """Number of veto electrons + veto muons per event (SR veto)."""
    nele = ak.count(select_electrons(events).pt, axis=1)
    nmu = ak.count(select_muons(events).pt, axis=1)
    return nele + nmu


def get_recoil(events, use_tight=False):
    """
    Event recoil, computed before SR/CR masks.

    By default it uses veto leptons (SR-style). Set ``use_tight=True`` for tight leptons.
    """
    met_pt = events.MET.pt
    met_phi = events.MET.phi
    if use_tight:
        ele = select_tight_electrons(events)
        mu = select_tight_muons(events)
    else:
        ele = select_electrons(events)
        mu = select_muons(events)
    sum_lep_px = ak.sum(ele.pt * np.cos(ele.phi), axis=1) + ak.sum(mu.pt * np.cos(mu.phi), axis=1)
    sum_lep_py = ak.sum(ele.pt * np.sin(ele.phi), axis=1) + ak.sum(mu.pt * np.sin(mu.phi), axis=1)
    return recoil_pt(met_pt, met_phi, sum_lep_px, sum_lep_py)


def recoil_pt(met_pt, met_phi, sum_lep_px, sum_lep_py):
    """
    Recoil magnitude (control-region variable): U = -(pTmiss + sum(lep)).
    Returns |U| in GeV (same shape as inputs). All inputs per event.
    """
    met_x = ak.to_numpy(met_pt) * np.cos(ak.to_numpy(met_phi))
    met_y = ak.to_numpy(met_pt) * np.sin(ak.to_numpy(met_phi))
    px = ak.to_numpy(sum_lep_px)
    py = ak.to_numpy(sum_lep_py)
    u_x = -(met_x + px)
    u_y = -(met_y + py)
    return ak.Array(np.sqrt(u_x**2 + u_y**2))


def cos_theta_star(jet0, jet1):
    """
    cos(theta*) from the two leading jets: |tanh(Δη/2)| with Δη = η_jet0 - η_jet1.
    Definition: ctsValue = abs(tanh(dEtaJet12 / 2)).
    jet0, jet1: awkward arrays with .eta (one entry per event).
    Returns array of shape (nEvents,) in [0, 1]; use only for events with valid jet pair.
    """
    eta0 = ak.to_numpy(jet0.eta)
    eta1 = ak.to_numpy(jet1.eta)
    d_eta = eta0 - eta1
    cts = np.abs(np.tanh(d_eta / 2.0))
    return ak.Array(np.clip(cts, 0.0, 1.0))


class bbDMProcessor(processor.ProcessorABC):
    """
    Coffea processor for bbDM search: object selection, event selection, histograms.

    **Golden JSON (certified lumi)** is applied **only** when ``is_data=True`` and a valid
    ``golden_json_path`` is given. MC and signal are never lumi-masked.
    """

    def __init__(
        self,
        jet_pt_min=JET_PT_MIN,
        jet_eta_max=JET_ETA_MAX,
        btag_wp=BTAG_WP_MEDIUM,
        recoil_min=RECOIL_MIN,
        signal_mass_branch: str = None,
        is_data: bool = False,
        golden_json_path: Optional[str] = None,
    ):
        self.jet_pt_min = jet_pt_min
        self.jet_eta_max = jet_eta_max
        self.btag_wp = btag_wp
        self.recoil_min = recoil_min
        self.signal_mass_branch = signal_mass_branch
        self.is_data = bool(is_data)
        self._lumi_mask = None
        if golden_json_path and not self.is_data:
            warnings.warn(
                "golden_json_path ignored: lumi mask is for collision data only (is_data=False).",
                stacklevel=2,
            )
        if self.is_data and golden_json_path:
            gjp = str(golden_json_path)
            if os.path.isfile(gjp):
                self._lumi_mask = LumiMask(gjp)
            else:
                warnings.warn(
                    f"Golden JSON not found; data will not be lumi-filtered: {gjp}",
                    stacklevel=2,
                )

        # Histogram definitions
        self._make_histograms()

    def _make_histograms(self):
        """Define all histograms (same structure for each dataset)."""
        reg_ax = axis.StrCategory(REGION_NAMES, name="region")
        self._hist_jet_pt = Hist(
            axis.Regular(50, 0, 500, name="jet_pt", label="Jet p$_T$ [GeV]"),
            storage="weight",
        )
        self._hist_jet_pt_by_region = Hist(
            reg_ax,
            axis.Regular(50, 0, 500, name="jet_pt", label="Jet p$_T$ [GeV]"),
            storage="weight",
        )
        self._hist_jet_mult = Hist(
            axis.Regular(15, 0, 15, name="njet", label="Jet multiplicity"),
            storage="weight",
        )
        self._hist_jet_mult_by_region = Hist(
            reg_ax,
            axis.Regular(15, 0, 15, name="njet", label="Jet multiplicity"),
            storage="weight",
        )
        self._hist_bjet_mult = Hist(
            axis.Regular(6, 0, 6, name="nbjet", label="b-jet multiplicity"),
            storage="weight",
        )
        self._hist_bjet_mult_by_region = Hist(
            reg_ax,
            axis.Regular(6, 0, 6, name="nbjet", label="b-jet multiplicity"),
            storage="weight",
        )
        self._hist_met = Hist(
            axis.Regular(60, 0, 600, name="met", label="MET [GeV]"),
            storage="weight",
        )
        self._hist_met_by_region = Hist(
            reg_ax,
            axis.Regular(60, 0, 600, name="met", label="MET [GeV]"),
            storage="weight",
        )
        self._hist_recoil = Hist(
            axis.Variable([250, 300, 400, 550, 1000], name="recoil", label="Recoil [GeV]"),
            storage="weight",
        )
        self._hist_recoil_by_region = Hist(
            reg_ax,
            axis.Variable([250, 300, 400, 550, 1000], name="recoil", label="Recoil [GeV]"),
            storage="weight",
        )
        self._hist_cos_theta_star = Hist(
            axis.Regular(4, 0, 1, name="cos_theta_star", label="cos #theta*"),
            storage="weight",
        )
        self._hist_cos_theta_star_by_region = Hist(
            reg_ax,
            axis.Regular(4, 0, 1, name="cos_theta_star", label="cos #theta*"),
            storage="weight",
        )
        self._hist_lead_jet_pt = Hist(
            axis.Regular(50, 0, 500, name="lead_jet_pt", label="Leading jet p$_T$ [GeV]"),
            storage="weight",
        )
        self._hist_lead_jet_pt_by_region = Hist(
            reg_ax,
            axis.Regular(50, 0, 500, name="lead_jet_pt", label="Leading jet p$_T$ [GeV]"),
            storage="weight",
        )
        self._hist_nlep = Hist(
            axis.Regular(6, 0, 6, name="nlep", label="Lepton multiplicity"),
            storage="weight",
        )
        self._hist_nlep_by_region = Hist(
            reg_ax,
            axis.Regular(6, 0, 6, name="nlep", label="Lepton multiplicity"),
            storage="weight",
        )
        self._hist_min_dphi_jets_recoil = Hist(
            axis.Regular(32, 0, 3.2, name="min_dphi_jets_recoil", label="min #Delta#phi(jet, recoil)"),
            storage="weight",
        )
        self._hist_min_dphi_jets_recoil_by_region = Hist(
            reg_ax,
            axis.Regular(32, 0, 3.2, name="min_dphi_jets_recoil", label="min #Delta#phi(jet, recoil)"),
            storage="weight",
        )
        self._hist_met_pf_calo_delta = Hist(
            axis.Regular(40, 0, 2.0, name="met_pf_calo_delta", label="|pTmiss(PF)/pTmiss(calo) - 1|"),
            storage="weight",
        )
        self._hist_met_pf_calo_delta_by_region = Hist(
            reg_ax,
            axis.Regular(40, 0, 2.0, name="met_pf_calo_delta", label="|pTmiss(PF)/pTmiss(calo) - 1|"),
            storage="weight",
        )
        self._hist_recoil_all = Hist(
            axis.Regular(80, 0, 1000, name="recoil_all", label="Recoil [GeV]"),
            storage="weight",
        )
        self._hist_recoil_all_by_region = Hist(
            reg_ax,
            axis.Regular(80, 0, 1000, name="recoil_all", label="Recoil [GeV]"),
            storage="weight",
        )

    @property
    def accumulator(self):
        """Initial accumulator: all histograms empty; merge-safe for chunked processing."""
        return dict_accumulator({
            "jet_pt": HistAccumulator(self._hist_jet_pt).identity(),
            "jet_pt_by_region": HistAccumulator(self._hist_jet_pt_by_region).identity(),
            "jet_mult": HistAccumulator(self._hist_jet_mult).identity(),
            "jet_mult_by_region": HistAccumulator(self._hist_jet_mult_by_region).identity(),
            "bjet_mult": HistAccumulator(self._hist_bjet_mult).identity(),
            "bjet_mult_by_region": HistAccumulator(self._hist_bjet_mult_by_region).identity(),
            "met": HistAccumulator(self._hist_met).identity(),
            "met_by_region": HistAccumulator(self._hist_met_by_region).identity(),
            "recoil": HistAccumulator(self._hist_recoil).identity(),
            "recoil_by_region": HistAccumulator(self._hist_recoil_by_region).identity(),
            "cos_theta_star": HistAccumulator(self._hist_cos_theta_star).identity(),
            "cos_theta_star_by_region": HistAccumulator(self._hist_cos_theta_star_by_region).identity(),
            "lead_jet_pt": HistAccumulator(self._hist_lead_jet_pt).identity(),
            "lead_jet_pt_by_region": HistAccumulator(self._hist_lead_jet_pt_by_region).identity(),
            "nlep": HistAccumulator(self._hist_nlep).identity(),
            "nlep_by_region": HistAccumulator(self._hist_nlep_by_region).identity(),
            "min_dphi_jets_recoil": HistAccumulator(self._hist_min_dphi_jets_recoil).identity(),
            "min_dphi_jets_recoil_by_region": HistAccumulator(self._hist_min_dphi_jets_recoil_by_region).identity(),
            "met_pf_calo_delta": HistAccumulator(self._hist_met_pf_calo_delta).identity(),
            "met_pf_calo_delta_by_region": HistAccumulator(self._hist_met_pf_calo_delta_by_region).identity(),
            "recoil_all": HistAccumulator(self._hist_recoil_all).identity(),
            "recoil_all_by_region": HistAccumulator(self._hist_recoil_all_by_region).identity(),
            "cutflow": defaultdict_accumulator(int),
        })

    def process(self, events):
        """
        Process one chunk of events: apply selections and fill histograms.
        """
        dataset = events.metadata.get("dataset", "unknown")
        out = self.accumulator

        out["cutflow"]["all_events"] += int(len(events))

        # Optional signal masspoint filter for randomized-signal files.
        # If the requested branch is absent in this chunk/file, return empty output safely.
        if self.signal_mass_branch:
            if self.signal_mass_branch not in events.fields:
                return out
            mass_mask = events[self.signal_mass_branch] > 0
            n_mass = int(ak.sum(mass_mask))
            out["cutflow"]["masspoint"] += n_mass
            if n_mass == 0:
                return out
            events = events[mass_mask]
        else:
            out["cutflow"]["masspoint"] += int(len(events))

        # ----- Data only: certified lumisections (golden JSON), before trigger -----
        if self._lumi_mask is not None:
            if "run" not in events.fields or "luminosityBlock" not in events.fields:
                warnings.warn(
                    "Data sample missing run/luminosityBlock; skipping golden JSON filter.",
                    stacklevel=2,
                )
            else:
                good_lumi = self._lumi_mask(events.run, events.luminosityBlock)
                out["cutflow"]["golden_json"] += int(ak.sum(good_lumi))
                events = events[good_lumi]

        # ----- Trigger: OR of analysis HLT paths (applied to data and MC) -----
        trigger_list = get_trigger_list()
        hlt_fields = set(events.HLT.fields) if hasattr(events, "HLT") and hasattr(events.HLT, "fields") else set()
        trigger_mask = ak.zeros_like(events.event, dtype=bool)
        if hlt_fields:
            for tname in trigger_list:
                if tname in hlt_fields:
                    trigger_mask = trigger_mask | events.HLT[tname]
            # Some signal files can have sparse/non-matching trigger bits.
            # For signal-only processing, keep all events in that case.
            if int(ak.sum(trigger_mask)) == 0 and str(dataset).startswith(("bbDM", "signal_")):
                trigger_mask = ak.ones_like(events.event, dtype=bool)
        else:
            # If no HLT information exists in this schema, do not drop events.
            trigger_mask = ak.ones_like(events.event, dtype=bool)
        out["cutflow"]["trigger"] += int(ak.sum(trigger_mask))
        events = events[trigger_mask]

        # Weights: use genWeight if present, else 1 (keep as awkward for indexing)
        weight = ak.ones_like(events.MET.pt, dtype=np.float64)
        if hasattr(events, "genWeight"):
            weight = ak.values_astype(np.sign(events.genWeight), np.float64)
        weight = ak.fill_none(weight, 1.0)

        # ----- Object selection -----
        good_jets = select_good_jets(events)
        njets = ak.num(good_jets)
        nbjets = count_bjets(good_jets, self.btag_wp)
        ele = select_electrons(events)
        mu = select_muons(events)
        nlep = ak.count(ele.pt, axis=1) + ak.count(mu.pt, axis=1)
        met = events.MET.pt
        met_phi = events.MET.phi
        recoil_all = get_recoil(events)
        min_dphi = min_dphi_jets_recoil(good_jets, met_phi)
        # PF vs Calo MET consistency: after Δphi(jet,MET), before recoil threshold (skipped if CaloMET missing)
        met_pf_calo_ok = met_pf_calo_mask(events)

        # ----- Cumulative cutflow -----
        # SR jet phase space: exactly 2-3 jets and exactly 2 b-jets.
        cut_njet = (njets >= 2) & (njets <= 3)
        cut_nbjet = cut_njet & (nbjets == 2)
        cut_lepveto = cut_nbjet & (nlep == 0)
        cut_min_dphi = cut_lepveto & (min_dphi > 0.5)
        cut_met_pf_calo = cut_min_dphi & met_pf_calo_ok
        cut_recoil = cut_met_pf_calo & (recoil_all > self.recoil_min)

        # Keep legacy keys for backward compatibility and add explicit SR keys.
        out["cutflow"]["njet_ge1"] += int(ak.sum(cut_njet))
        out["cutflow"]["nbjet_ge2"] += int(ak.sum(cut_nbjet))
        out["cutflow"]["njet_2to3"] += int(ak.sum(cut_njet))
        out["cutflow"]["nbjet_eq2"] += int(ak.sum(cut_nbjet))
        out["cutflow"]["lepton_veto"] += int(ak.sum(cut_lepveto))
        out["cutflow"]["min_dphi"] += int(ak.sum(cut_min_dphi))
        out["cutflow"]["met_pf_calo"] += int(ak.sum(cut_met_pf_calo))
        out["cutflow"][f"recoil_gt_{int(self.recoil_min)}"] += int(ak.sum(cut_recoil))

        # ----- Pre-selection: at least one jet (for plots) -----
        presel = cut_njet
        w = weight[presel]
        out["cutflow"]["presel"] += int(ak.sum(presel))

        # Per-jet weight (broadcast event weight to jets)
        w_jet = ak.flatten(ak.broadcast_arrays(w, good_jets[presel].pt)[0])
        out["jet_pt"].fill(jet_pt=ak.to_numpy(ak.flatten(good_jets[presel].pt)), weight=ak.to_numpy(ak.fill_none(w_jet, 1.0)))
        out["jet_mult"].fill(njet=ak.to_numpy(njets[presel]), weight=ak.to_numpy(ak.fill_none(w, 1.0)))
        out["bjet_mult"].fill(nbjet=ak.to_numpy(nbjets[presel]), weight=ak.to_numpy(ak.fill_none(w, 1.0)))
        out["met"].fill(met=ak.to_numpy(met[presel]), weight=ak.to_numpy(ak.fill_none(w, 1.0)))
        lead_pt = ak.fill_none(ak.firsts(good_jets[presel].pt), 0.0)
        out["lead_jet_pt"].fill(lead_jet_pt=ak.to_numpy(lead_pt), weight=ak.to_numpy(ak.fill_none(w, 1.0)))
        out["nlep"].fill(nlep=ak.to_numpy(nlep[presel]), weight=ak.to_numpy(ak.fill_none(w, 1.0)))
        out["min_dphi_jets_recoil"].fill(
            min_dphi_jets_recoil=ak.to_numpy(min_dphi[presel]),
            weight=ak.to_numpy(ak.fill_none(w, 1.0)),
        )
        out["recoil_all"].fill(
            recoil_all=ak.to_numpy(recoil_all[presel]),
            weight=ak.to_numpy(ak.fill_none(w, 1.0)),
        )
        met_pf_calo_delta = met_pf_calo_delta_sr(events)
        if met_pf_calo_delta is not None:
            out["met_pf_calo_delta"].fill(
                met_pf_calo_delta=ak.to_numpy(met_pf_calo_delta[presel]),
                weight=ak.to_numpy(ak.fill_none(w, 1.0)),
            )

        # ----- Signal region -----
        # Define SR explicitly ONCE and reuse everywhere (SR histograms + region-wise filling).
        sr_mask = (
            (njets >= 2)
            & (njets <= 3)
            & (nbjets == 2)
            & (nlep == 0)
            & (min_dphi > 0.5)
            & met_pf_calo_ok
            & (recoil_all > self.recoil_min)
        )
        sr = sr_mask
        w_sr = weight[sr]
        out["cutflow"]["signal_region"] += int(ak.sum(sr))
        out["recoil"].fill(recoil=ak.to_numpy(recoil_all[sr]), weight=ak.to_numpy(ak.fill_none(w_sr, 1.0)))
        # cos(theta*) from two leading jets in SR
        good_jets_sr = good_jets[sr]
        jets_pad = ak.pad_none(good_jets_sr, 2)
        jet0 = jets_pad[:, 0]
        jet1 = jets_pad[:, 1]
        has_two = ak.num(good_jets_sr) >= 2
        mask = has_two & ~ak.is_none(jet1)
        if ak.sum(mask) > 0:
            cts = cos_theta_star(jet0[mask], jet1[mask])
            out["cos_theta_star"].fill(
                cos_theta_star=ak.to_numpy(cts),
                weight=ak.to_numpy(ak.fill_none(w_sr[mask], 1.0)),
            )

        # ----- Control regions (match Session 3 definitions) -----
        lead_jet_pt = ak.fill_none(ak.firsts(good_jets.pt), 0.0)

        # Z->ll control regions (tight opposite-sign same-flavor dileptons)
        tight_ele = select_tight_electrons(events)
        tight_mu = select_tight_muons(events)
        nele_t = ak.count(tight_ele.pt, axis=1)
        nmu_t = ak.count(tight_mu.pt, axis=1)
        two_ee = (nele_t == 2) & (nmu_t == 0) & (ak.sum(tight_ele.charge, axis=1) == 0)
        two_mumu = (nele_t == 0) & (nmu_t == 2) & (ak.sum(tight_mu.charge, axis=1) == 0)
        tight_ele_pad = ak.pad_none(tight_ele, 2)
        tight_mu_pad = ak.pad_none(tight_mu, 2)
        pair_ee = tight_ele_pad[:, 0] + tight_ele_pad[:, 1]
        pair_mumu = tight_mu_pad[:, 0] + tight_mu_pad[:, 1]
        sum_lep_px_z = ak.where(
            two_ee,
            ak.fill_none(pair_ee.pt, 0.0) * np.cos(ak.fill_none(pair_ee.phi, 0.0)),
            ak.where(
                two_mumu,
                ak.fill_none(pair_mumu.pt, 0.0) * np.cos(ak.fill_none(pair_mumu.phi, 0.0)),
                ak.full_like(met, 0.0),
            ),
        )
        sum_lep_py_z = ak.where(
            two_ee,
            ak.fill_none(pair_ee.pt, 0.0) * np.sin(ak.fill_none(pair_ee.phi, 0.0)),
            ak.where(
                two_mumu,
                ak.fill_none(pair_mumu.pt, 0.0) * np.sin(ak.fill_none(pair_mumu.phi, 0.0)),
                ak.full_like(met, 0.0),
            ),
        )
        recoil_z = recoil_pt(met, met_phi, sum_lep_px_z, sum_lep_py_z)
        met_pf_calo_ok_z = met_pf_calo_mask(events, mode="cr", sum_lep_px=sum_lep_px_z, sum_lep_py=sum_lep_py_z)
        presel_z = (njets >= 1) & (lead_jet_pt > LEAD_JET_PT_MIN_CR) & (min_dphi > 0.5) & met_pf_calo_ok_z & (recoil_z > RECOIL_MIN)
        mll_ee = ak.where(two_ee, ak.fill_none(pair_ee.mass, -1.0), ak.full_like(met, -1.0))
        mll_mumu = ak.where(two_mumu, ak.fill_none(pair_mumu.mass, -1.0), ak.full_like(met, -1.0))
        lead_lep_pt_z = ak.where(
            two_ee,
            ak.max(tight_ele.pt, axis=1),
            ak.where(two_mumu, ak.max(tight_mu.pt, axis=1), ak.full_like(met, 0.0)),
        )
        zecr = presel_z & two_ee & (lead_lep_pt_z > LEAD_LEP_PT_CR) & (mll_ee > Z_CR_MLL_LO) & (mll_ee < Z_CR_MLL_HI)
        zmucr = presel_z & two_mumu & (lead_lep_pt_z > LEAD_LEP_PT_CR) & (mll_mumu > Z_CR_MLL_LO) & (mll_mumu < Z_CR_MLL_HI)
        zecr_twolep = presel_z & two_ee
        zecr_leadlep = zecr_twolep & (lead_lep_pt_z > LEAD_LEP_PT_CR)
        zecr_mll = zecr_leadlep & (mll_ee > Z_CR_MLL_LO) & (mll_ee < Z_CR_MLL_HI)
        zmucr_twolep = presel_z & two_mumu
        zmucr_leadlep = zmucr_twolep & (lead_lep_pt_z > LEAD_LEP_PT_CR)
        zmucr_mll = zmucr_leadlep & (mll_mumu > Z_CR_MLL_LO) & (mll_mumu < Z_CR_MLL_HI)

        # Top control regions (single tight lepton)
        one_ele = (nele_t == 1) & (nmu_t == 0)
        one_mu = (nele_t == 0) & (nmu_t == 1)
        lep_pt_t = ak.fill_none(
            ak.where(one_ele, ak.firsts(tight_ele.pt), ak.where(one_mu, ak.firsts(tight_mu.pt), ak.full_like(met, 0.0))),
            0.0,
        )
        lep_phi_t = ak.fill_none(
            ak.where(one_ele, ak.firsts(tight_ele.phi), ak.where(one_mu, ak.firsts(tight_mu.phi), ak.full_like(met, 0.0))),
            0.0,
        )
        sum_lep_px_t = lep_pt_t * np.cos(lep_phi_t)
        sum_lep_py_t = lep_pt_t * np.sin(lep_phi_t)
        recoil_t = recoil_pt(met, met_phi, sum_lep_px_t, sum_lep_py_t)
        met_pf_calo_ok_t = met_pf_calo_mask(events, mode="cr", sum_lep_px=sum_lep_px_t, sum_lep_py=sum_lep_py_t)
        presel_t = (njets >= 1) & (lead_jet_pt > LEAD_JET_PT_MIN_CR) & (min_dphi > 0.5) & met_pf_calo_ok_t & (recoil_t > RECOIL_MIN)
        dphi_lep_met = np.abs(ak.to_numpy(met_phi) - ak.to_numpy(lep_phi_t))
        dphi_lep_met = np.where(dphi_lep_met > np.pi, 2 * np.pi - dphi_lep_met, dphi_lep_met)
        mt = ak.Array(np.sqrt(2.0 * ak.to_numpy(met) * ak.to_numpy(lep_pt_t) * (1.0 - np.cos(dphi_lep_met))))
        n_non_b = njets - nbjets
        common_t = presel_t & (lep_pt_t > LEAD_LEP_PT_CR) & (mt < TOP_CR_MT_MAX) & (nbjets == 2) & (n_non_b >= 2)
        tecr = common_t & one_ele
        tmucr = common_t & one_mu
        tecr_onelep = presel_t & one_ele
        tecr_leppt = tecr_onelep & (lep_pt_t > LEAD_LEP_PT_CR)
        tecr_mt = tecr_leppt & (mt < TOP_CR_MT_MAX)
        tecr_nb = tecr_mt & (nbjets == 2)
        tecr_nnonb = tecr_nb & (n_non_b >= 2)
        tmucr_onelep = presel_t & one_mu
        tmucr_leppt = tmucr_onelep & (lep_pt_t > LEAD_LEP_PT_CR)
        tmucr_mt = tmucr_leppt & (mt < TOP_CR_MT_MAX)
        tmucr_nb = tmucr_mt & (nbjets == 2)
        tmucr_nnonb = tmucr_nb & (n_non_b >= 2)

        out["cutflow"]["zecr"] += int(ak.sum(zecr))
        out["cutflow"]["zmucr"] += int(ak.sum(zmucr))
        out["cutflow"]["tecr"] += int(ak.sum(tecr))
        out["cutflow"]["tmucr"] += int(ak.sum(tmucr))
        out["cutflow"]["zecr_presel"] += int(ak.sum(presel_z))
        out["cutflow"]["zecr_twolep"] += int(ak.sum(zecr_twolep))
        out["cutflow"]["zecr_leadlep"] += int(ak.sum(zecr_leadlep))
        out["cutflow"]["zecr_mll"] += int(ak.sum(zecr_mll))
        out["cutflow"]["zmucr_presel"] += int(ak.sum(presel_z))
        out["cutflow"]["zmucr_twolep"] += int(ak.sum(zmucr_twolep))
        out["cutflow"]["zmucr_leadlep"] += int(ak.sum(zmucr_leadlep))
        out["cutflow"]["zmucr_mll"] += int(ak.sum(zmucr_mll))
        out["cutflow"]["tecr_presel"] += int(ak.sum(presel_t))
        out["cutflow"]["tecr_onelep"] += int(ak.sum(tecr_onelep))
        out["cutflow"]["tecr_leppt"] += int(ak.sum(tecr_leppt))
        out["cutflow"]["tecr_mt"] += int(ak.sum(tecr_mt))
        out["cutflow"]["tecr_nbjet"] += int(ak.sum(tecr_nb))
        out["cutflow"]["tecr_nnonb"] += int(ak.sum(tecr_nnonb))
        out["cutflow"]["tmucr_presel"] += int(ak.sum(presel_t))
        out["cutflow"]["tmucr_onelep"] += int(ak.sum(tmucr_onelep))
        out["cutflow"]["tmucr_leppt"] += int(ak.sum(tmucr_leppt))
        out["cutflow"]["tmucr_mt"] += int(ak.sum(tmucr_mt))
        out["cutflow"]["tmucr_nbjet"] += int(ak.sum(tmucr_nb))
        out["cutflow"]["tmucr_nnonb"] += int(ak.sum(tmucr_nnonb))

        # ---------------------------------------------------------------------
        # Region masks used for region-aware histograms
        #
        # IMPORTANT: define each region explicitly here, so region-wise histograms
        # cannot be contaminated by events failing that region's full selection.
        # ---------------------------------------------------------------------
        zecr_mask = (
            (njets >= 1)
            & (lead_jet_pt > LEAD_JET_PT_MIN_CR)
            & (min_dphi > 0.5)
            & met_pf_calo_ok_z
            & (recoil_z > RECOIL_MIN)
            & two_ee
            & (lead_lep_pt_z > LEAD_LEP_PT_CR)
            & (mll_ee > Z_CR_MLL_LO)
            & (mll_ee < Z_CR_MLL_HI)
        )
        zmucr_mask = (
            (njets >= 1)
            & (lead_jet_pt > LEAD_JET_PT_MIN_CR)
            & (min_dphi > 0.5)
            & met_pf_calo_ok_z
            & (recoil_z > RECOIL_MIN)
            & two_mumu
            & (lead_lep_pt_z > LEAD_LEP_PT_CR)
            & (mll_mumu > Z_CR_MLL_LO)
            & (mll_mumu < Z_CR_MLL_HI)
        )

        tecr_mask = (
            (njets >= 1)
            & (lead_jet_pt > LEAD_JET_PT_MIN_CR)
            & (min_dphi > 0.5)
            & met_pf_calo_ok_t
            & (recoil_t > RECOIL_MIN)
            & one_ele
            & (lep_pt_t > LEAD_LEP_PT_CR)
            & (mt < TOP_CR_MT_MAX)
            & (nbjets == 2)
            & (n_non_b >= 2)
        )
        tmucr_mask = (
            (njets >= 1)
            & (lead_jet_pt > LEAD_JET_PT_MIN_CR)
            & (min_dphi > 0.5)
            & met_pf_calo_ok_t
            & (recoil_t > RECOIL_MIN)
            & one_mu
            & (lep_pt_t > LEAD_LEP_PT_CR)
            & (mt < TOP_CR_MT_MAX)
            & (nbjets == 2)
            & (n_non_b >= 2)
        )

        region_masks = {
            "sr": sr_mask,
            "zecr": zecr_mask,
            "zmucr": zmucr_mask,
            "tecr": tecr_mask,
            "tmucr": tmucr_mask,
        }
        region_recoil = {
            "sr": recoil_all,
            "zecr": recoil_z,
            "zmucr": recoil_z,
            "tecr": recoil_t,
            "tmucr": recoil_t,
        }
        region_delta = {
            "sr": met_pf_calo_delta_sr(events),
            "zecr": met_pf_calo_delta_cr(events, sum_lep_px_z, sum_lep_py_z),
            "zmucr": met_pf_calo_delta_cr(events, sum_lep_px_z, sum_lep_py_z),
            "tecr": met_pf_calo_delta_cr(events, sum_lep_px_t, sum_lep_py_t),
            "tmucr": met_pf_calo_delta_cr(events, sum_lep_px_t, sum_lep_py_t),
        }

        for reg, rmask in region_masks.items():
            if int(ak.sum(rmask)) == 0:
                continue
            wr = ak.fill_none(weight[rmask], 1.0)
            gj = good_jets[rmask]
            nj = njets[rmask]
            nb = nbjets[rmask]
            met_r = met[rmask]
            recoil_r = region_recoil[reg][rmask]
            lead_pt_r = ak.fill_none(ak.firsts(gj.pt), 0.0)
            nlep_r = nlep[rmask]
            mindphi_r = min_dphi[rmask]
            recoil_all_r = recoil_all[rmask]

            out["jet_mult_by_region"].fill(region=reg, njet=ak.to_numpy(nj), weight=ak.to_numpy(wr))
            out["bjet_mult_by_region"].fill(region=reg, nbjet=ak.to_numpy(nb), weight=ak.to_numpy(wr))
            out["met_by_region"].fill(region=reg, met=ak.to_numpy(met_r), weight=ak.to_numpy(wr))
            out["recoil_by_region"].fill(region=reg, recoil=ak.to_numpy(recoil_r), weight=ak.to_numpy(wr))
            out["lead_jet_pt_by_region"].fill(region=reg, lead_jet_pt=ak.to_numpy(lead_pt_r), weight=ak.to_numpy(wr))
            out["nlep_by_region"].fill(region=reg, nlep=ak.to_numpy(nlep_r), weight=ak.to_numpy(wr))
            out["min_dphi_jets_recoil_by_region"].fill(
                region=reg, min_dphi_jets_recoil=ak.to_numpy(mindphi_r), weight=ak.to_numpy(wr)
            )
            out["recoil_all_by_region"].fill(region=reg, recoil_all=ak.to_numpy(recoil_all_r), weight=ak.to_numpy(wr))

            dreg = region_delta.get(reg)
            if dreg is not None:
                out["met_pf_calo_delta_by_region"].fill(
                    region=reg,
                    met_pf_calo_delta=ak.to_numpy(dreg[rmask]),
                    weight=ak.to_numpy(wr),
                )

            if int(ak.sum(nj > 0)) > 0:
                wj = ak.flatten(ak.broadcast_arrays(wr, gj.pt)[0])
                out["jet_pt_by_region"].fill(
                    region=reg,
                    jet_pt=ak.to_numpy(ak.flatten(gj.pt)),
                    weight=ak.to_numpy(ak.fill_none(wj, 1.0)),
                )

            jets_pad_r = ak.pad_none(gj, 2)
            j0_r = jets_pad_r[:, 0]
            j1_r = jets_pad_r[:, 1]
            has_two_r = ak.num(gj) >= 2
            m2 = has_two_r & ~ak.is_none(j1_r)
            if ak.sum(m2) > 0:
                cts_r = cos_theta_star(j0_r[m2], j1_r[m2])
                out["cos_theta_star_by_region"].fill(
                    region=reg,
                    cos_theta_star=ak.to_numpy(cts_r),
                    weight=ak.to_numpy(ak.fill_none(wr[m2], 1.0)),
                )

        return out

    def postprocess(self, accumulator):
        """Optional: merge cutflows or scale histograms. For now, return as is."""
        return accumulator
