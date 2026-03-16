"""
Steps 3–6: Session 1–4 notebook logic (mode 1).
Requires: coffea, 2017 data at PATH_2017; Session 4 requires output_2017.pkl.
Run from project root: pytest tests/test_mode1_notebooks.py -v
"""
import os
import pickle
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PKL = os.path.join(ROOT, "output_2017.pkl")


def _has_coffea():
    try:
        import coffea  # noqa: F401
        return True
    except ImportError:
        return False


# --- Session 1: load events ---
@pytest.mark.skipif(not _has_coffea(), reason="coffea not installed")
def test_session1_load_events():
    """Session 1: get_one_file_per_group + load_events for data and background."""
    from config.datasets_2017 import get_one_file_per_group
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    NanoAODSchema.warn_missing_crossrefs = False

    def load_events(filepath):
        return NanoEventsFactory.from_root(
            {filepath: "Events"},
            schemaclass=NanoAODSchema,
            metadata={"dataset": "nanoaod"},
            mode="eager",
        ).events()

    filesets = get_one_file_per_group()
    assert filesets, "get_one_file_per_group() returned empty"
    if "data" in filesets and filesets["data"]:
        events_data = load_events(filesets["data"][0])
        assert len(events_data) > 0
        assert hasattr(events_data, "Jet") and hasattr(events_data, "MET")
    if "background" in filesets and filesets["background"]:
        events_bkg = load_events(filesets["background"][0])
        assert len(events_bkg) > 0
        assert hasattr(events_bkg, "Jet") and hasattr(events_bkg, "MET")


# --- Session 2: object selection ---
@pytest.mark.skipif(not _has_coffea(), reason="coffea not installed")
def test_session2_object_selection():
    """Session 2: select_good_jets, count_bjets, n_tight_leptons on background events."""
    from config.datasets_2017 import get_one_file_per_group
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    from processor.bbdm_processor import (
        select_good_jets,
        count_bjets,
        n_tight_leptons,
    )

    NanoAODSchema.warn_missing_crossrefs = False

    def load_events(filepath):
        return NanoEventsFactory.from_root(
            {filepath: "Events"},
            schemaclass=NanoAODSchema,
            metadata={"dataset": "nanoaod"},
            mode="eager",
        ).events()

    filesets = get_one_file_per_group()
    assert "background" in filesets and filesets["background"]
    events = load_events(filesets["background"][0])
    good_jets = select_good_jets(events)
    nbjets = count_bjets(good_jets)
    nlep = n_tight_leptons(events)
    assert len(good_jets) == len(events)  # jagged, same number of events
    assert len(nbjets) == len(events)
    assert len(nlep) == len(events)


# --- Session 3: cutflow and MET SR ---
@pytest.mark.skipif(not _has_coffea(), reason="coffea not installed")
def test_session3_cutflow_and_met_sr():
    """Session 3: cutflow (≥1 jet, ≥2 b-jets, 0 leptons, MET>200) and MET SR shape."""
    import awkward as ak

    from config.datasets_2017 import get_one_file_per_group
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    from processor.bbdm_processor import (
        select_good_jets,
        count_bjets,
        n_tight_leptons,
    )

    NanoAODSchema.warn_missing_crossrefs = False
    MET_SR_MIN = 200.0

    def load_events(filepath):
        return NanoEventsFactory.from_root(
            {filepath: "Events"},
            schemaclass=NanoAODSchema,
            metadata={"dataset": "nanoaod"},
            mode="eager",
        ).events()

    filesets = get_one_file_per_group()
    assert "background" in filesets and filesets["background"]
    events = load_events(filesets["background"][0])
    good_jets = select_good_jets(events)
    njets = ak.num(good_jets)
    nbjets = count_bjets(good_jets)
    nlep = n_tight_leptons(events)
    met = events.MET.pt

    N0 = len(events)
    N1 = int(ak.sum(njets >= 1))
    N2 = int(ak.sum((njets >= 1) & (nbjets >= 2)))
    N3 = int(ak.sum((njets >= 1) & (nbjets >= 2) & (nlep == 0)))
    N4 = int(ak.sum((njets >= 1) & (nbjets >= 2) & (nlep == 0) & (met > MET_SR_MIN)))

    assert N1 <= N0 and N2 <= N1 and N3 <= N2 and N4 <= N3
    sr_mask = (njets >= 1) & (nbjets >= 2) & (nlep == 0) & (met > MET_SR_MIN)
    met_sr = met[sr_mask]
    assert len(met_sr) == N4


# --- Session 4: load pkl, data/bkg split, get_met_sr_hist ---
def test_session4_load_pkl():
    """Session 4: load output_2017.pkl and check structure."""
    if not os.path.isfile(PKL):
        pytest.skip("output_2017.pkl not found; run run_analysis.py first")
    with open(PKL, "rb") as f:
        results = pickle.load(f)
    assert results
    example = list(results.values())[0]
    assert "met_sr" in example
    assert "cutflow" in example
    all_datasets = list(results.keys())
    data_datasets = [d for d in all_datasets if "Run2017" in d or "Single" in d or "MET" in d]
    bkg_datasets = [d for d in all_datasets if d not in data_datasets]
    # At least we have dataset names
    assert len(all_datasets) > 0


def test_session4_get_met_sr_hist():
    """Session 4: get_met_sr_hist returns hist for a dataset."""
    if not os.path.isfile(PKL):
        pytest.skip("output_2017.pkl not found")
    with open(PKL, "rb") as f:
        results = pickle.load(f)
    name = list(results.keys())[0]
    hist_obj = results[name]["met_sr"]
    assert hist_obj is not None
    # hist has .axes or .values()
    assert hasattr(hist_obj, "axes") or hasattr(hist_obj, "values")
