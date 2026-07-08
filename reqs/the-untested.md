# the-untested: predict bioactivity of the never-tested natural world

Working name. Predict the medicinal (bioactivity) properties of natural products
that were **never assayed**, from their molecular structure, aggregate the
predictions up to the source plants/fungi, and render a **plant × property
certainty map**. Demo foregrounds antimicrobial resistance (AMR).

The thesis in one line: train QSAR on the labelled ChEMBL compounds (176k carry a
panel label), score the 227k LOTUS naturals that nobody ever tested, and show
which untested plants likely carry activity, with a first-class confidence over
where the model is blind.

## Honesty rules (non-negotiable, on screen)

These are part of the spec. Without them, breadth produces confident wrong answers.

1. **Scaffold split, never random.** Random splits leak analogs across
   train/test and inflate QSAR scores. Split by Bemis-Murcko scaffold so the
   model is scored on structurally novel molecules, the same job it does live.
2. **Beat the priors, or you earned nothing.** Headline metric is lift over the
   **taxonomic prior** (same plant family as a known-active plant, from the LOTUS
   taxonomy columns), computed in the batch map next to every organism verdict.
   The **folk prior** from Dr. Duke's ethnobotany is a second baseline in the
   design, not built in v1.
3. **Applicability domain is first-class.** Every prediction carries a
   familiarity = max Tanimoto to a capped sample of the training set in
   fingerprint space. A weird alkaloid the model never saw reads as a long shot,
   not a confident wrong answer.
4. **Coverage per plant.** A plant prediction is only as trustworthy as the
   fraction of its chemistry we actually know. Show "40 of 60 known compounds
   characterized" next to every plant verdict.
5. **Binding-active is not a cure.** `pchembl` activity in an assay ≠ medicinal
   in a human. Permanent, loud, on every prediction. This is a research triage
   tool, not medical advice.
6. **Population-split caveat.** Natural products that reached ChEMBL are a
   biased sample. State it; read the lift over baseline, not the absolute AUC.

## AI-system card

- **Prediction problem:** multi-task binary classification. Molecule → activity
  vector across a target panel (active/inactive at a `pchembl` threshold per
  target). Multi-task so thin targets borrow strength from fat ones.
- **KPI:** count of well-founded, novel bioactivity leads (untested plant ×
  property) that survive the priors and the applicability-domain filter.
- **ML proxy metric:** per-target ROC-AUC / PR-AUC / precision@k on scaffold-
  held-out molecules, headline = **lift over the 1-NN Tanimoto baseline** and the
  taxonomic prior; plus applicability-domain coverage.
- **Data sources:** LOTUS (plant↔molecule + SMILES + taxonomy + chem-class),
  ChEMBL (molecule→target→pchembl label), Dr. Duke's (folk-use, the parked folk prior).
- **ML-system type:** hybrid. Batch inference precomputes the full map
  (227k molecules × panel → activity + confidence, aggregated to 37k organisms)
  into feature groups; a real-time KServe deployment scores any molecule/plant
  on demand, including brand-new SMILES not in LOTUS.
- **Consumed:** UI, the certainty map, plant/molecule search, an AMR attention
  rail (untested plants ranked by predicted antimicrobial activity × confidence
  × coverage), per-plant dossier showing the honest baselines.
- **Monitored:** applicability-domain coverage of live queries, per-target
  calibration drift, label drift as ChEMBL releases update.

## Data

| source | gives | access | scale |
|---|---|---|---|
| LOTUS frozen (Zenodo concept DOI 5794106) | plant ↔ molecule links | bulk `.csv.gz` | 544k links, 227k molecules, 37k organisms |
| LOTUS metadata (same record) | SMILES, formula, mass, xlogp, NPClassifier + ClassyFire class, full organism taxonomy (domain→species) | bulk `.csv.gz` | 227k molecules |
| ChEMBL 37 (EBI FTP bulk) | molecule → target → `pchembl` activity | SQLite/H5 dump (REST too flaky for bulk) | 2.9M compounds; training pool = the labelled compounds on a panel target, ~42k overlap with LOTUS is the validation slice |
| Dr. Duke's phytochem (USDA) | plant → chemical → ethnobotanical use | bulk / site | folk-prior baseline (parked) |

Join key across all: **InChIKey** (exact and first-block skeleton both ~42k, so
the strict exact join is enough; no fuzzy matching).

Target panel v1 (AMR-led, chosen in F2 by measured label density, `panel.py` is
the single source of truth): 11 AMR heads, *Plasmodium falciparum* (malaria),
*Mycobacterium tuberculosis*, *Staphylococcus aureus*, *Escherichia coli*,
*Pseudomonas aeruginosa*, *Enterococcus faecium*, *Candida albicans*, the
kinetoplastid parasites *Trypanosoma cruzi* / *Leishmania donovani* /
*Trypanosoma brucei*, and Beta-lactamase, plus 3 human cytotoxicity heads (A549,
HepG2, HeLa) that fatten the multi-task representation and power a selectivity
read. The label-poor ESKAPE members (*Klebsiella*, *Acinetobacter*,
*Enterobacter*) did not clear the density cut and are not in v1.

---

## Pipelines (FTI, ordered by blocked-by)

### F: Feature pipelines (model-independent transformations, MITs)

**F1 · ingest-lotus**: download LOTUS frozen + metadata, normalize, write two
offline FGs: `natural_product` (InChIKey PK; SMILES, formula, exact_mass, xlogp,
npclassifier_pathway/superclass/class, classyfire) and `organism_compound`
(organism ↔ InChIKey with full taxonomy columns for the taxonomic prior).
Backfill once; refresh only on LOTUS releases (rare).
→ skills: **hops-data-sources**, **hops-fg**. Blocked-by: none.

**F2 · ingest-chembl**: pull the ChEMBL 37 bulk dump (SQLite/H5, never the REST
API for bulk), extract activities with a non-null `pchembl_value` for the target
panel, binarize active/inactive per target at a fixed `pchembl` threshold, write
offline FG `compound_activity` (InChIKey ↔ target ↔ label). **Step one of this
pipeline is the overlap measurement**: count labeled molecules per target after
joining ChEMBL InChIKeys to LOTUS, and fix the final target panel from the
counts. → skills: **hops-data-sources**, **hops-fg**. Blocked-by: none.

**F2b · pivot-labels**: `compound_activity` is keyed by (InChIKey, target), so a
feature view cannot join it on InChIKey alone. Pivot it into the wide FG
`compound_labels`: one row per molecule, one column per panel target (1 active /
0 inactive / null unmeasured), keyed by InChIKey, so the FV joins it 1:1 to the
features. → skills: **hops-fg**. Blocked-by: F2.

**F3 · featurize-molecules**: shared skew-free extractor `chem_features.py`
(imported by this pipeline and by serving): SMILES → 2048-bit Morgan fingerprint
(base64-packed) + Bemis-Murcko scaffold + 10 physchem descriptors. Featurize the
LOTUS naturals and every ChEMBL compound on a panel target, write offline FG
`molecule_features` (the model input). RDKit needs a cloned env.
→ skills: **hops-features**, **hops-fg**, **hops-environments**. Blocked-by: F1, F2.

**F3b · smiles-map**: the featurizer packs the fingerprint and drops SMILES, but
the stage-2 GNN eats the molecule graph, built from SMILES. Persist canonical
SMILES for exactly the molecules in `molecule_features` into the sidecar FG
`molecule_smiles`, keyed by InChIKey, 1:1 with the features. → skills:
**hops-fg**. Blocked-by: F3.

**F4 · ingest-ethnobotany** (parked): Dr. Duke's folk-use → FG `folk_use`, the
folk-prior baseline. Not built in v1. → skills: **hops-data-sources**,
**hops-fg**. Blocked-by: none.

### T: Training pipeline (model-dependent transformations, MDTs, + model)

**T1 · EDA**: profile the LOTUS↔ChEMBL join, confirm the measured per-target
label density, leakage audit. Enforce **scaffold split** (Bemis-Murcko), verify
no InChIKey/scaffold crosses train/test, check class balance per target. Confirm
the taxonomic-prior baseline computes.
→ skills: **hops-eda**, **hops-eda-checklist**. Blocked-by: F1, F2, F3.

**T2 · train-qsar**: FV `qsar_fv` = `compound_labels` (root/spine) JOIN
`molecule_features` on InChIKey. The label FG is the root so features are fetched
as-of the label event_time and a later-written label still matches. Per-target
label masking (a molecule unlabeled for target *k* contributes no loss there).
- **Stage 1 (the bar):** per-target HistGradientBoosting on the fingerprint + 10
  descriptors. Registered as `amr_qsar`, the served champion.
- **Stage 2:** multi-task Chemprop message-passing GNN on the molecule graph,
  served the same labels through `qsar_gnn_fv` (`compound_labels` JOIN
  `molecule_smiles`), scored on the same scaffold fold-0 test as the bar.
  Registered as `amr_qsar_gnn`.
- **Applicability-domain layer:** `ad_score` = max Tanimoto to a capped sample of
  the training fingerprints, stored per prediction. Out-of-domain reads as a long
  shot, not a confident answer.
- **Evaluate:** per-target ROC-AUC / PR-AUC / precision@k on scaffold-held-out
  molecules, applicability-domain coverage, and **lift over the 1-NN Tanimoto
  baseline**. Register model + eval images. **Promote the GNN only if it clears
  the FP-GBM bar.** Register every run (lineage), pick champion by advertised
  metric at serve time.
→ skills: **hops-train**, **hops-transformations**, **hops-fv**,
**hops-environments**. Blocked-by: T1.

### I: Inference pipelines

**I1 · map-batch-inference**: score all 227k LOTUS molecules × target panel with
the champion → activity prob + AD confidence; aggregate to organism level (plant
× property, weighted by chemistry coverage). Write FGs `molecule_prediction` and
`plant_property_map`. Scheduled on model refresh. → skills:
**hops-batch-inference**, **hops-fg**. Blocked-by: T2, F3.

**I2 · qsar-serving**: KServe deployment scoring any molecule (SMILES) or plant
(name → its LOTUS compounds → aggregate) on demand, including brand-new SMILES
absent from LOTUS. Uses the shared `chem_features.py` (no skew). Returns activity
+ confidence + the driving molecule + nearest known drug (Tanimoto) + AD flag.
→ skills: **hops-online-inference**, **hops-environments**. Blocked-by: T2.

**I3 · app**: the UI. A chemical-space certainty map (t-SNE of the fingerprints)
recoloured per disease, an attention rail of the untested organisms ranked by
predicted activity, a per-organism dossier that draws the molecule, links the
source organism to Wikipedia, and shows the applicability-domain familiarity, a
live SMILES scorer, and a broad-spectrum view that recolours the map by how many
diseases a molecule hits at once and flags low-familiarity frequent-hitters. The
"binding-active is not a cure" warning is permanent on every screen. Logs inputs
+ predictions for monitoring. → skills: **hops-app**, **hops-monitoring**.
Blocked-by: I1, I2.

## Engine note (taste sibling, later)

The pipeline is item → molecules → predict-a-property → certainty map. The
food/flavor sibling (ingredient → flavor molecules → predict recipe co-occurrence)
is the same engine with a different label source. Keep `chem_features.py` and the
FG/FV shapes label-agnostic so the taste build drops in as a second target set,
not a rewrite. Not built now.

## Build order

F1, F2 in parallel (no deps) → F2b (needs F2) → F3 (needs F1, F2) → F3b (needs
F3) → T1 → T2 → I1, I2 → I3. F4 parked.
