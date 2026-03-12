# Instructor Guide — bbDM DAS Long Exercise

This guide helps instructors run the four-session bbDM long exercise. It covers expected plots, typical yields, common mistakes, and discussion questions.

---

## 1. Expected Plots and Results

### Session 1
- **Jet pT**: Peaks at low pT (~30–80 GeV), tail to high pT. Raw multiplicity before selection: typically 5–15 jets per event in tt̄.
- **Jet multiplicity**: Roughly Poisson-like; mean depends on process (e.g. tt̄ ~6–8).
- **MET**: Rising at low MET, then falling; tt̄ and W+jets contribute at low MET, Z→νν at high MET.
- **Electron/Muon pT**: Leptonic samples show clear peaks; hadronic samples show mostly low pT or empty.

### Session 2
- **Jet multiplicity (after jet selection)**: Shifted to lower values; shape similar.
- **b-jet multiplicity**: Most events have 0–2 b-jets; tt̄ enhances 1–2 b-jets.
- **Leading jet pT**: Harder spectrum after selection; peak around 50–100 GeV.

### Session 3
- **MET in signal region (MET > 200 GeV, ≥2 b-jets, 0 leptons)**: Should show clear excess at high MET if signal is present; otherwise background-dominated with Z→νν important.
- **Yield table**: Example order of magnitude (depends on luminosity and samples):
  - Before SR: tt̄ ≫ W+jets, Z→νν, QCD.
  - After SR: Z→νν and tt̄ dominant; W+jets reduced by lepton veto.

### Session 4
- **Weights and systematics**: Students use processor output (output_2017.pkl); nominal histograms are filled with genWeight (sign). Simple normalisation systematics (e.g. ±20% per process) can be applied by scaling histograms.
- **Binned fit**: Poisson likelihood fit of signal + backgrounds to data in MET SR; best-fit signal strength μ (or background scale factors). Plot data vs fitted model.
- **Goodness of fit**: χ², number of degrees of freedom, p-value; optionally pulls per bin. Typical pitfalls: bins with zero expected yield (add a small constant or merge bins).
- **Limit**: 95% CL upper limit on signal strength (or on number of signal events). Asymptotic formula or pyhf. Expect approximate limit of order few × background uncertainty if no signal.

---

## 2. Typical Event Yields (Illustrative)

Use small samples for the school. Example scale (adjust to your samples):

| Selection step        | tt̄ (approx) | Z→νν | W+jets |
|-----------------------|-------------|------|--------|
| Preselection (jets)   | 100%        | 100% | 100%   |
| ≥2 b-jets             | ~50%        | ~5%  | ~10%   |
| nlep = 0              | ~35%        | ~100%| ~70%   |
| MET > 200 GeV         | ~5%         | ~20% | ~3%    |

Final signal region: Z→νν and tt̄ are the main backgrounds; W+jets is suppressed by the lepton veto.

---

## 3. Common Student Mistakes

1. **Wrong branch names**: NanoAOD uses `Jet_pt`, `MET_pt`, etc. Check spelling and use the exact names from the TTree.
2. **Axis/array confusion**: Coffea/awkward uses jagged arrays. Use `event.Jet.pt` etc., not scalar indexing without `ak.flatten` or `ak.num`.
3. **b-tag threshold**: Use the stated working point (e.g. DeepFlavB > 0.2783 for medium). Different WPs give very different yields.
4. **Lepton veto**: Require *no* tight leptons; forgetting the veto or applying it to the wrong collection inflates W+jets and tt̄.
5. **MET sign**: Use MET magnitude (`MET_pt`), not components, for the cut.
6. **Unit consistency**: NanoAOD is in GeV; ensure plot labels say "GeV" and cuts are in GeV.

---

## 4. Discussion Questions for Students

- **Why does tt̄ dominate after b-jet selection?**  
  Top quarks decay to b quarks; tt̄ events typically have two b-jets. So requiring ≥2 b-jets strongly favours tt̄.

- **Why does MET suppress W+jets background?**  
  In W+jets, MET comes mainly from the neutrino from W decay. The W is often low-pT, so MET is modest. Requiring MET > 200 GeV keeps mostly events with mismeasurement or rare high-pT W; the bulk of W+jets is removed.

- **What physics process produces large MET in Z→νν?**  
  Z→νν gives two neutrinos carrying away momentum; the MET is the vector sum of their transverse momenta. High-pT Z production naturally gives large MET, so Z→νν is a key background in the signal region.

- **Why use b-jets in a dark matter search?**  
  In models like 2HDM+a or bbDM, DM couples to the Higgs sector; production often involves b quarks (e.g. pp→φ→bb̄+DM). Selecting b-jets both enhances signal and reduces light-jet QCD background.

- **What is the role of control regions?**  
  Control regions (e.g. single-lepton for tt̄) are used to validate background modelling and normalisation with data before applying the signal-region selection.

---

## 5. Pacing and Tips

- **Session 1**: Leave time for Coffea install and first run. Have a small NanoAOD file (or pre-cached NanoAOD) so everyone can load events quickly.
- **Session 2**: Emphasise that object selection (jet ID, b-tag, lepton ID) is the same as in real analyses; only thresholds are simplified.
- **Session 3**: If time is short, focus on one or two backgrounds and the signal region definition; control regions can be introduced conceptually.
- **Session 4**: Ensure output_2017.pkl (or output_2017_full.pkl) exists. If students only have one-file runs, the histograms may be sparse; synthetic or pre-made histograms can be provided for the fit/limit part. For limits, a single-bin (total SR yield) exercise is often enough in the time available.

Encourage students to **change cuts** (e.g. MET threshold, b-tag WP) and observe how yields and shapes change. This reinforces the connection between selection and physics.

---

## 6. Solutions

Full worked solutions are in `solutions/` (solutions_session1.ipynb through solutions_session4.ipynb). Use them to verify results and to help students who get stuck.
