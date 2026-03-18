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
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from hist import Hist, axis

NanoAODSchema.warn_missing_crossrefs = False


def get_nanoevents(filepath, schemaclass=NanoAODSchema):
    """Load one NanoAOD file and return events (same pattern as Session 1 load_events)."""
    return NanoEventsFactory.from_root(filepath, schemaclass=schemaclass).events()


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
        # Delegate common hist API (e.g. .fill, .plot, .axes, .values)
        return getattr(self._hist, name)

    def __repr__(self):
        return f"HistAccumulator({self._hist!r})"


# -----------------------------------------------------------------------------
# Selection constants (Run-2 typical values; adjust for your analysis)
# -----------------------------------------------------------------------------
JET_PT_MIN = 30.0   # GeV
JET_ETA_MAX = 2.4
BTAG_WP_MEDIUM = 0.2783   # DeepFlavB medium working point (2017)
MET_SR_MIN = 250.0  # GeV, signal region
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


def min_dphi_jets_met(jets, met_phi):
    """
    Minimum Δφ between any selected jet and MET direction (per event).
    Returns array of shape (nEvents,).
    """
    dphi = np.abs(jets.phi - met_phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return ak.min(dphi, axis=1)


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
        self._hist_cos_theta_star_sr = Hist(
            axis.Regular(25, 0, 1, name="cos_theta_star_sr", label="cos #theta* (SR)"),
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
            "jet_pt": HistAccumulator(self._hist_jet_pt).identity(),
            "jet_mult": HistAccumulator(self._hist_jet_mult).identity(),
            "bjet_mult": HistAccumulator(self._hist_bjet_mult).identity(),
            "met": HistAccumulator(self._hist_met).identity(),
            "met_sr": HistAccumulator(self._hist_met_sr).identity(),
            "cos_theta_star_sr": HistAccumulator(self._hist_cos_theta_star_sr).identity(),
            "lead_jet_pt": HistAccumulator(self._hist_lead_jet_pt).identity(),
            "cutflow": defaultdict_accumulator(int),
        })

    def process(self, events):
        """
        Process one chunk of events: apply selections and fill histograms.
        """
        dataset = events.metadata.get("dataset", "unknown")
        out = self.accumulator

        # Weights: use genWeight if present, else 1 (keep as awkward for indexing)
        weight = ak.ones_like(events.MET.pt, dtype=np.float64)
        if hasattr(events, "genWeight"):
            weight = ak.values_astype(np.sign(events.genWeight), np.float64)
        weight = ak.fill_none(weight, 1.0)

        # ----- Object selection -----
        good_jets = select_good_jets(events)
        njets = ak.num(good_jets)
        nbjets = count_bjets(good_jets, self.btag_wp)
        nlep = n_tight_leptons(events)
        met = events.MET.pt
        met_phi = events.MET.phi
        min_dphi = min_dphi_jets_met(good_jets, met_phi)

        # ----- Pre-selection: at least one jet (for plots) -----
        presel = njets >= 1
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

        # ----- Signal region -----
        sr = (
            (njets >= 1) &
            (nbjets >= 2) &
            (nlep == 0) &
            (met > self.met_sr_min) &
            (min_dphi > 0.5)
        )
        w_sr = weight[sr]
        out["cutflow"]["signal_region"] += int(ak.sum(sr))
        out["met_sr"].fill(met_sr=ak.to_numpy(met[sr]), weight=ak.to_numpy(ak.fill_none(w_sr, 1.0)))
        # cos(theta*) from two leading jets in SR
        good_jets_sr = good_jets[sr]
        jets_pad = ak.pad_none(good_jets_sr, 2)
        jet0 = jets_pad[:, 0]
        jet1 = jets_pad[:, 1]
        has_two = ak.num(good_jets_sr) >= 2
        mask = has_two & ~ak.is_none(jet1)
        if ak.sum(mask) > 0:
            cts = cos_theta_star(jet0[mask], jet1[mask])
            out["cos_theta_star_sr"].fill(
                cos_theta_star_sr=ak.to_numpy(cts),
                weight=ak.to_numpy(ak.fill_none(w_sr[mask], 1.0)),
            )

        return out

    def postprocess(self, accumulator):
        """Optional: merge cutflows or scale histograms. For now, return as is."""
        return accumulator
