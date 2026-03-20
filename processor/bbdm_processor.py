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

from config.datasets_2017 import get_trigger_list

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
    """

    def __init__(
        self,
        jet_pt_min=JET_PT_MIN,
        jet_eta_max=JET_ETA_MAX,
        btag_wp=BTAG_WP_MEDIUM,
        recoil_min=RECOIL_MIN,
        signal_mass_branch: str = None,
    ):
        self.jet_pt_min = jet_pt_min
        self.jet_eta_max = jet_eta_max
        self.btag_wp = btag_wp
        self.recoil_min = recoil_min
        self.signal_mass_branch = signal_mass_branch

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
        self._hist_recoil = Hist(
            axis.Variable([250, 300, 400, 550, 1000], name="recoil", label="Recoil [GeV]"),
            storage="weight",
        )
        self._hist_cos_theta_star = Hist(
            axis.Regular(4, 0, 1, name="cos_theta_star", label="cos #theta*"),
            storage="weight",
        )
        self._hist_lead_jet_pt = Hist(
            axis.Regular(50, 0, 500, name="lead_jet_pt", label="Leading jet p$_T$ [GeV]"),
            storage="weight",
        )
        self._hist_nlep = Hist(
            axis.Regular(6, 0, 6, name="nlep", label="Lepton multiplicity"),
            storage="weight",
        )
        self._hist_min_dphi_jets_met = Hist(
            axis.Regular(32, 0, 3.2, name="min_dphi_jets_met", label="min #Delta#phi(jet, MET)"),
            storage="weight",
        )
        self._hist_met_pf_calo_delta = Hist(
            axis.Regular(40, 0, 2.0, name="met_pf_calo_delta", label="|pTmiss(PF)/pTmiss(calo) - 1|"),
            storage="weight",
        )
        self._hist_recoil_all = Hist(
            axis.Regular(80, 0, 1000, name="recoil_all", label="Recoil [GeV]"),
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
            "recoil": HistAccumulator(self._hist_recoil).identity(),
            "cos_theta_star": HistAccumulator(self._hist_cos_theta_star).identity(),
            "lead_jet_pt": HistAccumulator(self._hist_lead_jet_pt).identity(),
            "nlep": HistAccumulator(self._hist_nlep).identity(),
            "min_dphi_jets_met": HistAccumulator(self._hist_min_dphi_jets_met).identity(),
            "met_pf_calo_delta": HistAccumulator(self._hist_met_pf_calo_delta).identity(),
            "recoil_all": HistAccumulator(self._hist_recoil_all).identity(),
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

        # ----- Trigger: OR of analysis HLT paths (first cut, applied to data and MC) -----
        trigger_list = get_trigger_list()
        hlt_fields = set(events.HLT.fields) if hasattr(events, "HLT") and hasattr(events.HLT, "fields") else set()
        trigger_mask = ak.ones_like(events.event, dtype=bool)
        if hlt_fields:
            for tname in trigger_list:
                if tname in hlt_fields:
                    trigger_mask = trigger_mask | events.HLT[tname]
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
        min_dphi = min_dphi_jets_met(good_jets, met_phi)
        # PF vs Calo MET consistency: after Δphi(jet,MET), before recoil threshold (skipped if CaloMET missing)
        met_pf_calo_ok = met_pf_calo_mask(events)

        # ----- Cumulative cutflow -----
        cut_njet = njets >= 1
        cut_nbjet = cut_njet & (nbjets >= 2)
        cut_lepveto = cut_nbjet & (nlep == 0)
        cut_min_dphi = cut_lepveto & (min_dphi > 0.5)
        cut_met_pf_calo = cut_min_dphi & met_pf_calo_ok
        cut_recoil = cut_met_pf_calo & (recoil_all > self.recoil_min)

        out["cutflow"]["njet_ge1"] += int(ak.sum(cut_njet))
        out["cutflow"]["nbjet_ge2"] += int(ak.sum(cut_nbjet))
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
        out["min_dphi_jets_met"].fill(
            min_dphi_jets_met=ak.to_numpy(min_dphi[presel]),
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
        sr = cut_recoil
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

        return out

    def postprocess(self, accumulator):
        """Optional: merge cutflows or scale histograms. For now, return as is."""
        return accumulator
