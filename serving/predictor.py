"""I2 online inference: on-demand QSAR for any molecule.

Given a SMILES (a brand-new molecule, not necessarily in LOTUS), featurizes it
with the SAME shared chem_features.py bundled at training time (no skew), scores
every AMR-panel head, and returns each probability plus an applicability-domain
confidence (max Tanimoto to the training set) so an out-of-domain molecule reads
as "unknown", not a confident guess.

Deployed as a custom PYTHON predictor on untested-serve-env (pandas-inference-
pipeline + RDKit). The model bundle (per-target models + ad_reference +
chem_features.py + panel.py) is the amr_qsar champion.
"""
import base64
import glob
import os
import sys

import joblib
import numpy as np

AD_MIN = 0.30


def load_model_file(name):
    for root in (os.environ.get("MODEL_FILES_PATH"),
                 os.environ.get("ARTIFACT_FILES_PATH"),
                 "/mnt/models", "/mnt/artifacts"):
        if root:
            hits = glob.glob(f"{root}/**/{name}", recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found under the model/artifact mounts")


def _tanimoto_max(q_fp, ref_fp, ref_pop):
    inter = q_fp @ ref_fp.T
    denom = q_fp.sum(1)[:, None] + ref_pop[None, :] - inter
    return np.max(inter / np.maximum(denom, 1), axis=1)


class Predict:
    def __init__(self):
        bundle = os.path.dirname(load_model_file("chem_features.py"))
        sys.path.insert(0, bundle)
        import chem_features as cf
        from panel import PANEL
        self.cf = cf
        self.panel = PANEL
        self.models = {}
        for tid in PANEL:
            try:
                self.models[tid] = joblib.load(load_model_file(f"{tid}.joblib"))
            except FileNotFoundError:
                pass
        ad = np.load(load_model_file("ad_reference.npz"))
        self.ad_fp = ad["fp"].astype(np.float32)
        self.ad_pop = ad["pop"].astype(np.float32)
        print(f"loaded {len(self.models)} heads, AD ref {self.ad_fp.shape}", flush=True)

    def _rows(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs.get("instances", inputs.get("inputs", [inputs]))
        for item in inputs:
            yield item.get("smiles") if isinstance(item, dict) else item

    def predict(self, inputs):
        out = []
        for smiles in self._rows(inputs):
            fp, desc = self.cf.featurize(smiles)
            if fp is None:
                out.append({"smiles": smiles, "error": "unparseable SMILES"})
                continue
            X = np.hstack([fp, desc]).reshape(1, -1).astype(np.float32)
            ad = float(_tanimoto_max(fp.reshape(1, -1).astype(np.float32),
                                     self.ad_fp, self.ad_pop)[0])
            probs = {tid: {"label": self.panel[tid][0], "kind": self.panel[tid][1],
                           "prob": round(float(m.predict_proba(X)[0, 1]), 4)}
                     for tid, m in self.models.items()}
            out.append({"smiles": smiles, "ad_score": round(ad, 4),
                        "in_domain": ad >= AD_MIN, "predictions": probs})
        return {"predictions": out}
