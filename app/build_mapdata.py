"""Build the chemical-space map data the app canvas renders.

Samples the scored naturals (all the top hits per AMR pathogen + a random
background), projects their fingerprints to 2D with t-SNE, attaches per-pathogen
probabilities, applicability-domain confidence, SMILES, and the plants that
contain each molecule. Writes one JSON the FastAPI server serves live.

Runs on untested-train-env (sklearn + numpy; fingerprints already stored).
"""
import base64
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_here] + sorted(glob.glob("/hopsfs/Users/*/the-untested")):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)
from panel import PANEL, amr_targets  # noqa: E402

TOP_PER_TARGET = 250
BACKGROUND = 5000
# in a job, __file__ is the uploaded copy under Resources; write to the FUSE repo
_repo = next(iter(glob.glob("/hopsfs/Users/*/the-untested")), _here)
OUT = f"{_repo}/app/static/mapdata.json"


def unpack(s):
    return np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8)).astype(np.float32)


def main():
    import hopsworks
    from sklearn.manifold import TSNE
    fs = hopsworks.login().get_feature_store()

    preds = fs.get_feature_group("molecule_prediction", 1).read()
    prob_cols = [f"prob_{t.lower()}" for t in amr_targets()
                 if f"prob_{t.lower()}" in preds.columns]

    # sample: the strongest molecule per AMR head + a random background
    keep = set()
    for c in prob_cols:
        keep |= set(preds.nlargest(TOP_PER_TARGET, c)["inchikey"])
    bg = preds.sample(min(BACKGROUND, len(preds)), random_state=0)["inchikey"]
    keep |= set(bg)
    sample = preds[preds["inchikey"].isin(keep)].reset_index(drop=True)
    print(f"sample: {len(sample):,} molecules", flush=True)

    mf = fs.get_feature_group("molecule_features", 1).read()
    mf = mf[mf["inchikey"].isin(keep)][["inchikey", "fp_b64"]]
    sample = sample.merge(mf, on="inchikey", how="inner")

    fp = np.vstack([unpack(b) for b in sample["fp_b64"].values])
    print("running t-SNE ...", flush=True)
    xy = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=0).fit_transform(fp)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))   # normalize to [0,1]

    # plants that contain each sampled molecule (a few each)
    oc = fs.get_feature_group("organism_compound", 1).read()
    oc = oc[oc["inchikey"].isin(keep)]
    orgs = oc.groupby("inchikey")["organism"].apply(
        lambda s: sorted(set(s))[:6]).to_dict()

    np_fg = fs.get_feature_group("natural_product", 1).read()[["inchikey", "smiles"]]
    smi = dict(zip(np_fg["inchikey"], np_fg["smiles"]))

    points = []
    for i, row in sample.iterrows():
        ik = row["inchikey"]
        points.append({
            "ik": ik, "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
            "ad": round(float(row["ad_score"]), 3),
            "smiles": smi.get(ik, ""), "orgs": orgs.get(ik, []),
            "p": {t: round(float(row[f"prob_{t.lower()}"]), 3)
                  for t in amr_targets() if f"prob_{t.lower()}" in sample.columns},
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"panel": {t: PANEL[t][0] for t in amr_targets()},
               "points": points}, open(OUT, "w"))
    print(f"wrote {len(points):,} points -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
