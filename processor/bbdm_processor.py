"""
bbDM Coffea Processor for CMS NanoAOD

Reads NanoAOD, applies object and event selection for a simplified
bb + MET dark matter search, and fills histograms.

Signal region: MET > 200 GeV, >= 2 b-jets, 0 isolated leptons.
"""

import numpy as np
import awkward as ak
from coffea import processor
from coffea.processor import dict_accumulator, defaultdict_accumulator
from coffea.nanoevents import NanoEventsFactory, BaseSchema
import hist
from hist import Hist, axis


def get_nanoevents(filepath, schemaclass=BaseSchema):
    """Load NanoAOD file into Coffea NanoEvents."""
    return NanoEventsFactory.from_root(filepath, schemaclass=schemaclass).events()


# -----------------------------------------------------------------------------
# Selection constants (Run-2 typical values; adjust for your analysis)
# -----------------------------------------------------------------------------
JET_PT_MIN = 30.0   # GeV
JET_ETA_MAX = 2.4
BTAG_WP_MEDIUM = 0.2783   # DeepFlavB medium working point (2017)
MET_SR_MIN = 200.0  # GeV, signal region
LEP_PT_MIN = 10.0   # GeV
LEP_ETA_MAX = 2.5


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


def count_bjets(jets, wp=BTAG_WP_MEDIUM):
    """Count jets passing b-tag working point (DeepFlavB)."""
    return ak.sum(jets.btagDeepFlavB > wp, axis=1)


def select_tight_electrons(events):
    """Tight electron selection for veto (pt, eta, minimal ID)."""
    ele = events.Electron
    mask = (
        (ele.pt > LEP_PT_MIN) &
        (np.abs(ele.eta) < LEP_ETA_MAX) &
        (ele.cutBased >= 2)   # tight cut-based ID
    )
    return ele[mask]


def select_tight_muons(events):
    """Tight muon selection for veto."""
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


class bbDMProcessor(processor.ProcessorABC):
    """
    Coffea processor for bbDM search: object selection, event selection, histograms.
    """

    def __init__(
        self,
        jet_pt_min=JET_PT_MIN,
        jet_eta_max=JET_ETA_MAX,
        btag_wp=BTAG_WP_MEDIUM,
        met_sr_min=MET_SR_MIN,
    ):
        self.jet_pt_min = jet_pt_min
        self.jet_eta_max = jet_eta_max
        self.btag_wp = btag_wp
        self.met_sr_min = met_sr_min

        # Histogram definitions
        self._make_histograms()

    def _make_histograms(self):
        """Define all histograms (same structure for each dataset)."""
        self._hist_jet_pt = Hist(
            axis.Regular(50, 0, 500, name="jet_pt", label="Jet p$_T$ [GeV]"),
            storage="weight",
        )
        self._hist_jet_mult = Hist(
            axis.Regular(15, 0, 15, name="njet", label="Jet multiplicity"),
            storage="weight",
        )
        self._hist_bjet_mult = Hist(
            axis.Regular(6, 0, 6, name="nbjet", label="b-jet multiplicity"),
            storage="weight",
        )
        self._hist_met = Hist(
            axis.Regular(60, 0, 600, name="met", label="MET [GeV]"),
            storage="weight",
        )
        self._hist_met_sr = Hist(
            axis.Regular(50, 200, 700, name="met_sr", label="MET (SR) [GeV]"),
            storage="weight",
        )
        self._hist_lead_jet_pt = Hist(
            axis.Regular(50, 0, 500, name="lead_jet_pt", label="Leading jet p$_T$ [GeV]"),
            storage="weight",
        )

    @property
    def accumulator(self):
        """Initial accumulator: all histograms empty; merge-safe for chunked processing."""
        return dict_accumulator({
            "jet_pt": self._hist_jet_pt.copy(),
            "jet_mult": self._hist_jet_mult.copy(),
            "bjet_mult": self._hist_bjet_mult.copy(),
            "met": self._hist_met.copy(),
            "met_sr": self._hist_met_sr.copy(),
            "lead_jet_pt": self._hist_lead_jet_pt.copy(),
            "cutflow": defaultdict_accumulator(int),
            "datasets": [],
        })

    def process(self, events):
        """
        Process one chunk of events: apply selections and fill histograms.
        """
        dataset = events.metadata.get("dataset", "unknown")
        out = self.accumulator
        out["datasets"].append(dataset)

        # Weights: use genWeight if present, else 1 (keep as awkward for indexing)
        weight = ak.ones_like(events.MET.pt)
        if hasattr(events, "genWeight"):
            weight = np.sign(events.genWeight)

        # ----- Object selection -----
        good_jets = select_good_jets(events)
        njets = ak.num(good_jets)
        nbjets = count_bjets(good_jets, self.btag_wp)
        nlep = n_tight_leptons(events)
        met = events.MET.pt

        # ----- Pre-selection: at least one jet (for plots) -----
        presel = njets >= 1
        w = weight[presel]
        out["cutflow"]["presel"] += int(ak.sum(presel))

        # Per-jet weight (broadcast event weight to jets)
        w_jet = ak.flatten(ak.broadcast_arrays(w, good_jets[presel].pt)[0])
        out["jet_pt"].fill(jet_pt=ak.flatten(good_jets[presel].pt), weight=ak.to_numpy(w_jet))
        out["jet_mult"].fill(njet=njets[presel], weight=ak.to_numpy(w))
        out["bjet_mult"].fill(nbjet=nbjets[presel], weight=ak.to_numpy(w))
        out["met"].fill(met=met[presel], weight=ak.to_numpy(w))
        lead_pt = ak.fill_none(ak.firsts(good_jets[presel].pt), 0.0)
        out["lead_jet_pt"].fill(lead_jet_pt=ak.to_numpy(lead_pt), weight=ak.to_numpy(w))

        # ----- Signal region -----
        sr = (
            (njets >= 1) &
            (nbjets >= 2) &
            (nlep == 0) &
            (met > self.met_sr_min)
        )
        w_sr = weight[sr]
        out["cutflow"]["signal_region"] += int(ak.sum(sr))
        out["met_sr"].fill(met_sr=met[sr], weight=ak.to_numpy(w_sr))

        return out

    def postprocess(self, accumulator):
        """Optional: merge cutflows or scale histograms. For now, return as is."""
        return accumulator
