"""I1 batch map: score every untested natural product across the AMR panel and
aggregate the predictions up to the source plants and fungi.

Loads the amr_qsar champion (per-target models + applicability-domain reference),
predicts each of the 227k LOTUS naturals on all panel heads with a calibrated
in-domain confidence, then aggregates to organisms: for each (organism, target)
the best in-domain molecule is the plant's score, with coverage and the driving
compound. The taxonomic prior (family active-rate among known-active naturals)
is merged in as the honest baseline the model must beat.

Writes molecule_prediction (per molecule) and plant_property_map (per organism,
target). Runs rdkit-free: fingerprints are already stored, unpack is pure numpy.
Blocked-by T2 (amr_qsar) + F1 (natural_product, organism_compound) + F3.
"""
import base64
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_here] + sorted(glob.glob("/hopsfs/Users/*/the-untested")):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)
from panel import PANEL, ycol, amr_targets  # noqa: E402

DESCRIPTORS = ["mol_wt", "logp", "tpsa", "hbd", "hba", "rot_bonds",
               "aromatic_rings", "ring_count", "frac_csp3", "heavy_atoms"]
AD_MIN = 0.30          # max-Tanimoto below this = out of domain, low confidence
MODEL_NAME = "amr_qsar"


def unpack(s):
    return np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8)).astype(np.float32)


def tanimoto_max(q_fp, ref_fp, ref_pop):
    q_pop = q_fp.sum(1)
    out = np.zeros(len(q_fp), np.float32)
    for s in range(0, len(q_fp), 2000):
        e = min(s + 2000, len(q_fp))
        inter = q_fp[s:e] @ ref_fp.T
        denom = q_pop[s:e, None] + ref_pop[None, :] - inter
        out[s:e] = np.max(inter / np.maximum(denom, 1), axis=1)
    return out


def load_champion(mr):
    try:
        model = mr.get_best_model(MODEL_NAME, "mean_amr_auc", "max")
    except Exception:
        model = mr.get_model(MODEL_NAME, version=1)
    d = model.download()
    models = {tid: joblib.load(f"{d}/models/{tid}.joblib")
              for tid in PANEL if os.path.exists(f"{d}/models/{tid}.joblib")}
    ad = np.load(f"{d}/ad_reference.npz")
    print(f"champion {MODEL_NAME} v{model.version}, {len(models)} heads", flush=True)
    return model, models, ad["fp"].astype(np.float32), ad["pop"].astype(np.float32)


def main():
    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    mr = proj.get_model_registry()
    model, models, ad_fp, ad_pop = load_champion(mr)

    print("reading naturals + features ...", flush=True)
    nat_keys = set(fs.get_feature_group("natural_product", 1).read()["inchikey"])
    mf = fs.get_feature_group("molecule_features", 1).read()
    mf = mf[mf["inchikey"].isin(nat_keys)].drop_duplicates("inchikey").reset_index(drop=True)
    print(f"scoring {len(mf):,} natural products", flush=True)

    fp = np.vstack([unpack(b) for b in mf["fp_b64"].values])
    desc = np.nan_to_num(mf[DESCRIPTORS].to_numpy(np.float32))
    X = np.hstack([fp, desc]).astype(np.float32)
    ad_score = tanimoto_max(fp, ad_fp, ad_pop)

    preds = pd.DataFrame({"inchikey": mf["inchikey"].values,
                          "ad_score": np.round(ad_score, 4)})
    for tid in PANEL:
        if tid in models:
            preds[f"prob_{tid.lower()}"] = np.round(
                models[tid].predict_proba(X)[:, 1], 4)
    preds["as_of"] = pd.Timestamp.utcnow()

    mp_fg = fs.get_or_create_feature_group(
        name="molecule_prediction", version=1,
        description="Per-natural-product predicted activity across the AMR panel "
                    "(prob_<target>) plus applicability-domain confidence "
                    "(ad_score = max Tanimoto to the training set). The untested "
                    "map at the molecule level.",
        primary_key=["inchikey"], event_time="as_of",
        online_enabled=False, statistics_config=False)
    mp_fg.insert(preds, write_options={"start_offline_materialization": True})
    print(f"inserted {len(preds):,} molecule predictions", flush=True)

    _plant_map(fs, preds, ad_score)


def _plant_map(fs, preds, ad_score):
    """Aggregate molecule predictions to (organism, target): best in-domain
    molecule = plant score, plus coverage and the taxonomic prior baseline."""
    oc = fs.get_feature_group("organism_compound", 1).read()
    oc = oc[["organism", "inchikey", "tax_family"]]
    pm = oc.merge(preds, on="inchikey", how="inner")

    rows = []
    for tid in amr_targets():
        col = f"prob_{tid.lower()}"
        if col not in pm.columns:
            continue
        sub = pm[["organism", "inchikey", "tax_family", "ad_score", col]].dropna(subset=[col])
        indom = sub[sub["ad_score"] >= AD_MIN]
        # family prior: active-rate proxy = mean predicted prob per family (the
        # "your cousins score high" baseline the per-molecule model must beat)
        fam_prior = indom.groupby("tax_family")[col].mean().rename("family_prior")
        for org, g in sub.groupby("organism"):
            gd = g[g["ad_score"] >= AD_MIN]
            pool = gd if len(gd) else g
            i = pool[col].idxmax()
            fam = pool.loc[i, "tax_family"]
            rows.append({
                "organism": org, "target_chembl_id": tid,
                "score": round(float(pool.loc[i, col]), 4),
                "driver_inchikey": pool.loc[i, "inchikey"],
                "driver_ad": round(float(pool.loc[i, "ad_score"]), 4),
                "n_compounds": int(len(g)), "n_in_domain": int(len(gd)),
                "family": fam,
                "family_prior": round(float(fam_prior.get(fam, np.nan)), 4)
                if pd.notna(fam) else None,
            })
    pmap = pd.DataFrame(rows)
    pmap["coverage"] = (pmap["n_in_domain"] / pmap["n_compounds"].clip(lower=1)).round(3)
    pmap["as_of"] = pd.Timestamp.utcnow()
    print(f"plant_property_map: {len(pmap):,} (organism, target) rows", flush=True)

    pp_fg = fs.get_or_create_feature_group(
        name="plant_property_map", version=1,
        description="Per (organism, AMR target) predicted activity: best "
                    "in-domain molecule score + its driving compound, coverage "
                    "(fraction of the plant's chemistry the model is confident "
                    "about), and the family taxonomic-prior baseline.",
        primary_key=["organism", "target_chembl_id"], event_time="as_of",
        online_enabled=False, statistics_config=False)
    pp_fg.insert(pmap, write_options={"start_offline_materialization": True})
    print(f"inserted {len(pmap):,} plant-property rows", flush=True)


if __name__ == "__main__":
    main()
