# the-untested: predict bioactivity of the never-tested natural world

Working name. Predict the medicinal (bioactivity) properties of natural products
that were **never assayed**, from their molecular structure, aggregate the
predictions up to the source plants/fungi, and render a **plant × property
certainty map**. Demo foregrounds antimicrobial resistance (AMR).

The thesis in one line: train QSAR on the ~42k natural products that ChEMBL has
measured, score the ~185k that nobody ever tested, and show which untested
plants likely carry activity, with calibrated confidence over where we are blind.

## Honesty rules (non-negotiable, on screen)

These are the spec, not decoration. Breadth without them is confident lies.

1. **Scaffold split, never random.** Random splits leak analogs across
   train/test and inflate QSAR scores. Split by Bemis-Murcko scaffold so the
   model is scored on structurally novel molecules, the same job it does live.
2. **Beat the priors, or you earned nothing.** Headline metric is lift over two
   baselines already sitting in the data: the **taxonomic prior** (same plant
   family as a known-active plant, from the LOTUS taxonomy columns) and the
   **folk prior** (already used in ethnobotany, from Dr. Duke's). On screen,
   measured, same spirit as the where-on-earth zero-shot baseline.
3. **Applicability domain is first-class.** Every prediction carries a
   calibrated confidence = conformal / distance-to-training in chemical space.
   A weird alkaloid the model never saw returns "outside what I know," not a
   confident wrong answer. This layer IS the SOTA part.
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
- **ML proxy metric:** per-target PR-AUC / AUC on scaffold-held-out molecules,
  headline = **lift over taxonomic + folk prior**; plus calibration (the
  confidence must mean what it says).
- **Data sources:** LOTUS (plant↔molecule + SMILES + taxonomy + chem-class),
  ChEMBL (molecule→target→pchembl label), Dr. Duke's (folk-use, for the prior).
- **ML-system type:** hybrid. Batch inference precomputes the full map
  (227k molecules × panel → activity + confidence, aggregated to 37k organisms)
  into feature groups; a real-time KServe deployment scores any molecule/plant
  on demand, including brand-new SMILES not in LOTUS.
- **Consumed:** UI, the certainty map, plant/molecule search, an AMR attention
  rail (untested plants ranked by predicted antimicrobial activity × confidence
  × coverage), per-plant dossier showing the honest baselines.
- **Monitored:** applicability-domain coverage of live queries, per-target
  calibration drift, label drift as ChEMBL releases update.

## Data (proven feasible, pulled from the pod)

| source | gives | access | scale |
|---|---|---|---|
| LOTUS frozen (Zenodo 19360665) | plant ↔ molecule links | bulk `.csv.gz` | 674k links, 227k molecules, 37k organisms |
| LOTUS metadata (same record) | SMILES, formula, mass, xlogp, NPClassifier + ClassyFire class, full organism taxonomy (domain→species) | bulk `.csv.gz` | 227k molecules |
| ChEMBL 37 (EBI FTP bulk) | molecule → target → `pchembl` activity | SQLite/H5 dump (REST too flaky for bulk) | 2.9M compounds; ~42k overlap with LOTUS = the labeled pool |
| Dr. Duke's phytochem (USDA) | plant → chemical → ethnobotanical use | bulk / site | folk-prior baseline |

Join key across all: **InChIKey** (exact and first-block skeleton both ~42k, so
the strict exact join is enough; no fuzzy matching).

Target panel v1 (AMR-led): *Plasmodium falciparum* had huge label sets but the
demo leads AMR: *Mycobacterium tuberculosis*, *Staphylococcus aureus*, and the
ESKAPE pathogens (*Klebsiella*, *Acinetobacter*, *Pseudomonas*, *Enterobacter*),
plus a handful of high-density human protein targets to fatten the multi-task
head. Final panel chosen in F2 by measured label density.

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

**F3 · featurize-molecules**: shared skew-free extractor `chem_features.py`
(imported by this pipeline and by serving): SMILES → Morgan fingerprint (+ a few
physchem descriptors). Featurize all 227k LOTUS molecules, write offline FG
`molecule_features` (the model input). RDKit needs a cloned env.
→ skills: **hops-features**, **hops-fg**, **hops-environments**. Blocked-by: F1.

**F4 · ingest-ethnobotany** (v1.5): Dr. Duke's folk-use → FG `folk_use`, the
folk-prior baseline. → skills: **hops-data-sources**, **hops-fg**. Blocked-by: none.

### T: Training pipeline (model-dependent transformations, MDTs, + model)

**T1 · EDA**: profile the LOTUS↔ChEMBL join, confirm the measured per-target
label density, leakage audit. Enforce **scaffold split** (Bemis-Murcko), verify
no InChIKey/scaffold crosses train/test, check class balance per target. Confirm
the taxonomic-prior and folk-prior baselines compute.
→ skills: **hops-eda**, **hops-eda-checklist**. Blocked-by: F1, F2, F3.

**T2 · train-qsar**: FV `qsar_fv` = `molecule_features` JOIN `compound_activity`
on InChIKey, with scaffold-group training data. MDTs on the FV: per-target label
masking (a molecule unlabeled for target *k* contributes no loss there), per-
target class weighting.
- **Stage 1 (the bar):** multi-task FP + gradient boosting / small MLP head.
- **Stage 2 (SOTA):** Chemprop-style message-passing GNN on molecule graphs.
- **Applicability-domain layer:** conformal prediction / distance-to-training in
  fingerprint space → calibrated per-prediction confidence.
- **Evaluate:** per-target ROC/PR-AUC on scaffold-held-out molecules,
  calibration curves, applicability-domain coverage, and **lift over taxonomic +
  folk priors**. Register model + images (per-target ROC/PR, calibration,
  coverage, lift-vs-baseline bars). **Ship the GNN only if it beats the
  FP-GBM bar live.** Register every run (lineage), pick champion by advertised
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

**I3 · app**: the UI. Certainty map, plant/molecule search, AMR attention rail
(untested plants ranked by predicted antimicrobial activity × confidence ×
coverage), per-plant dossier with the priors shown side by side, permanent
"binding-active ≠ cure" warning, external links to the source records (LOTUS,
ChEMBL, PubChem). Logs inputs + predictions for monitoring.
→ skills: **hops-app**, **hops-monitoring**. Blocked-by: I1, I2.

## Engine note (taste sibling, later)

The pipeline is item → molecules → predict-a-property → certainty map. The
food/flavor sibling (ingredient → flavor molecules → predict recipe co-occurrence)
is the same engine with a different label source. Keep `chem_features.py` and the
FG/FV shapes label-agnostic so the taste build drops in as a second target set,
not a rewrite. Not built now.

## Build order

F1, F2, F4 in parallel (no deps) → F3 (needs F1) → T1 → T2 → I1, I2 → I3.
