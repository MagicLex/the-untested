"""Autoresearch experiment script (Karpathy autoresearch, Hopsworks recipe).

ONE file the loop edits per experiment. Reads the cached qsar_fv parquet, trains
a per-target QSAR over the AMR panel, and prints the three lines the loop greps:

    val_metric: <mean AMR ROC-AUC, scaffold-held-out>
    peak_memory_gb: <float>
    training_seconds: <float>

Self-contained: no RDKit (fingerprint bits are already in the cache), no feature
store read (uses the parquet), so it runs fast on pandas-training-pipeline.

The optimization target is FIXED: maximize mean AMR AUC. Do not change the metric,
the panel, or the scaffold split. Everything else (model class, hyperparameters,
descriptor use, feature engineering on the fingerprint) is fair game. Edit only
the CONFIG block and build_model().
"""
import base64
import glob
import hashlib
import json
import os
import resource
import shutil
import time

import numpy as np
import pandas as pd

# ============================ CONFIG (edit me) ============================
USE_DESC = True          # include the 10 physchem descriptors alongside the fingerprint


def build_model():
    """Return a fresh untrained sklearn-style classifier for one target."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)
# =========================================================================

PANEL = {
    "CHEMBL364": ("P. falciparum (malaria)", "amr"),
    "CHEMBL2026": ("Beta-lactamase", "amr"),
    "CHEMBL368": ("T. cruzi (Chagas)", "amr"),
    "CHEMBL367": ("L. donovani", "amr"),
    "CHEMBL612849": ("T. brucei", "amr"),
    "CHEMBL352": ("S. aureus", "amr"),
    "CHEMBL354": ("E. coli", "amr"),
    "CHEMBL360": ("M. tuberculosis", "amr"),
    "CHEMBL348": ("P. aeruginosa", "amr"),
    "CHEMBL366": ("C. albicans", "amr"),
    "CHEMBL357": ("E. faecium", "amr"),
    "CHEMBL392": ("A549 (lung)", "cytotox"),
    "CHEMBL395": ("HepG2 (liver)", "cytotox"),
    "CHEMBL399": ("HeLa", "cytotox"),
}
DESCRIPTORS = ["mol_wt", "logp", "tpsa", "hbd", "hba", "rot_bonds",
               "aromatic_rings", "ring_count", "frac_csp3", "heavy_atoms"]
N_FOLDS = 5


def ycol(tid):
    return "y_" + tid.lower()


def unpack(s):
    return np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8)).astype(np.float32)


def scaffold_fold(scaffold, inchikey):
    key = scaffold if scaffold else inchikey
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % N_FOLDS


def find_cache():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [f"{here}/../data/qsar_cache.parquet",
              *glob.glob("/hopsfs/Users/*/the-untested/data/qsar_cache.parquet")]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("qsar_cache.parquet not found; run prepare.py first")


def main():
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    df = pd.read_parquet(find_cache())
    fp = np.vstack([unpack(b) for b in df["fp_b64"].values])
    if USE_DESC:
        desc = np.nan_to_num(df[DESCRIPTORS].to_numpy(np.float32))
        X = np.hstack([fp, desc]).astype(np.float32)
    else:
        X = fp
    folds = np.array([scaffold_fold(s, k) for s, k in
                      zip(df["scaffold"].fillna("").values, df["inchikey"].values)])

    out = "model"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(f"{out}/models")
    results = {}
    for tid, (label, kind) in PANEL.items():
        yt = df[ycol(tid)].to_numpy(np.float32)
        ridx = np.where(~np.isnan(yt))[0]
        y = yt[ridx].astype(int)
        te = folds[ridx] == 0
        tr = ~te
        m = build_model()
        m.fit(X[ridx][tr], y[tr])
        import joblib
        joblib.dump(m, f"{out}/models/{tid}.joblib")
        rec = {"label": label, "kind": kind, "n": int(len(ridx))}
        if te.sum() >= 20 and len(np.unique(y[te])) == 2:
            p = m.predict_proba(X[ridx][te])[:, 1]
            rec["roc_auc"] = round(float(roc_auc_score(y[te], p)), 4)
        results[tid] = rec
        print(f"{label:26s} auc={rec.get('roc_auc','n/a')}", flush=True)

    amr_aucs = [r["roc_auc"] for r in results.values()
                if r["kind"] == "amr" and "roc_auc" in r]
    val = float(np.mean(amr_aucs))
    json.dump({"results": results, "mean_amr_auc": round(val, 4),
               "use_desc": USE_DESC}, open(f"{out}/metrics.json", "w"), indent=2)

    # run-progression chart across kept experiments (recipe: images on the card)
    _progress_plot(out)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"val_metric: {val:.4f}")
    print(f"peak_memory_gb: {peak_gb:.2f}")
    print(f"training_seconds: {time.time() - t0:.1f}")

    # register this experiment as the next version of the search model line.
    # Runs in the job pod (has SDK access); keep/discard is a git decision, but
    # every successful experiment registers so the registry shows the full run.
    import sys
    desc = sys.argv[1] if len(sys.argv) > 1 else "experiment"
    try:
        import hopsworks
        mr = hopsworks.login().get_model_registry()
        model = mr.python.create_model(
            "autoresearch_amr", metrics={"mean_amr_auc": round(val, 4)},
            description=f"{desc[:200]} | mean_amr_auc={val:.4f} use_desc={USE_DESC}")
        model.save(out)
        print(f"registered autoresearch_amr v{model.version}", flush=True)
    except Exception as e:
        print(f"register failed (metrics still valid): {str(e)[:150]}", flush=True)


def _progress_plot(out):
    """Per-target AUC bar for the model card."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = json.load(open(f"{out}/metrics.json")) if os.path.exists(f"{out}/metrics.json") else None
    rows = [(r["label"], r["roc_auc"]) for r in json.load(open(f"{out}/metrics.json"))["results"].values()
            if "roc_auc" in r] if m else []
    if not rows:
        return
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(rows))))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#34d399")
    ax.axvline(0.5, color="#ef4444", lw=0.8, ls="--")
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("ROC-AUC (scaffold-held-out)")
    fig.tight_layout(); fig.savefig(f"{out}/per_target_auc.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
