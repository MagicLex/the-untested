"""T2 train-qsar (stage 1, the bar): per-target gradient-boosted QSAR on the
AMR panel, scaffold-split, scored against the 1-NN Tanimoto similarity-search
baseline every real QSAR must beat, with an applicability-domain layer.

For each panel target it trains a HistGradientBoosting classifier on the 2048-bit
Morgan fingerprint + 10 descriptors, evaluates on scaffold-held-out molecules
(ROC-AUC, PR-AUC), and compares to nearest-neighbour Tanimoto search. The model
bundle (per-target models + an AD reference set + the shared chem_features.py)
is registered with per-target eval images.

Stage 2 (the message-passing GNN) is a separate run that must beat this bar to
ship. Runs on untested-train-env (pandas-training-pipeline + RDKit).
"""
import glob
import json
import os
import sys
import tempfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_here] + sorted(glob.glob("/hopsfs/Users/*/the-untested")):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)
import chem_features as cf  # noqa: E402
from panel import PANEL, ycol  # noqa: E402

N_FOLDS = 5           # scaffold hash folds; fold 0 is the held-out test set
AD_REF_CAP = 8000     # cap the applicability-domain reference for Tanimoto
BASE_REF_CAP = 4000   # cap the per-target similarity-search reference
MODEL_NAME = "amr_qsar"
FV_NAME = "qsar_fv"


def scaffold_fold(scaffold, inchikey):
    """Deterministic hash fold. Same scaffold -> same fold (no leak). Acyclic
    molecules (empty scaffold) fall back to their InChIKey so they do not all
    clump into one fold."""
    import hashlib
    key = scaffold if scaffold else inchikey
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % N_FOLDS


def feature_view(fs):
    """qsar_fv: molecule_features (fingerprint + scaffold + descriptors) joined
    1:1 to compound_labels (the wide multi-task label matrix) on InChIKey. This
    is the training/serving contract; serving selects features through the same
    view, and the label columns are the panel heads."""
    try:
        fv = fs.get_feature_view(FV_NAME, version=1)
        if fv is not None:
            return fv
    except Exception:
        pass
    mf = fs.get_feature_group("molecule_features", 1)
    cl = fs.get_feature_group("compound_labels", 1)
    q = mf.select_all().join(cl.select_all(), on=["inchikey"], join_type="inner")
    fv = fs.create_feature_view(
        name=FV_NAME, version=1, query=q, labels=[ycol(t) for t in PANEL],
        description="AMR-panel QSAR: molecule fingerprint + descriptors + "
                    "scaffold joined 1:1 to the wide multi-task label matrix. "
                    "Labels are the per-target active heads (pchembl>=6).")
    print(f"created {FV_NAME} v1", flush=True)
    return fv


def load(fs):
    fv = feature_view(fs)
    X, y = fv.training_data()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    print(f"qsar_fv rows={len(X):,}  label cols={y.shape[1]}", flush=True)
    return X, y


def build_matrix(X):
    """Unpack fingerprints + descriptors into a float32 matrix, one row per
    molecule (the FV join is 1:1, so rows align with the label matrix)."""
    fp = np.vstack([cf.unpack(b) for b in X["fp_b64"].values])
    desc = np.nan_to_num(X[cf.DESCRIPTORS].to_numpy(np.float32))
    Xmat = np.hstack([fp, desc]).astype(np.float32)
    ikey = (X["inchikey"].values if "inchikey" in X.columns
            else X.index.astype(str).values)   # PK may be dropped by training_data()
    folds = np.array([scaffold_fold(s, ik) for s, ik in
                      zip(X["scaffold"].fillna("").values, ikey)])
    return Xmat, fp, folds


def tanimoto_max(q_fp, ref_fp, ref_pop):
    """Max Tanimoto of each query row to the reference set. Chunked BLAS."""
    q_pop = q_fp.sum(1)
    out = np.zeros(len(q_fp), np.float32)
    for s in range(0, len(q_fp), 2000):
        e = min(s + 2000, len(q_fp))
        inter = q_fp[s:e] @ ref_fp.T
        denom = q_pop[s:e, None] + ref_pop[None, :] - inter
        out[s:e] = np.max(inter / np.maximum(denom, 1), axis=1)
    return out


def train_target(tid, ylabels, Xmat, fp, folds):
    yt = ylabels[ycol(tid)].to_numpy(np.float32)
    ridx = np.where(~np.isnan(yt))[0]      # rows measured for this target
    y = yt[ridx].astype(int)
    f = folds[ridx]
    te = f == 0
    tr = ~te
    Xtr, Xte, ytr, yte = Xmat[ridx][tr], Xmat[ridx][te], y[tr], y[te]
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)
    m.fit(Xtr, ytr)
    metrics = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
               "active_rate": float(y.mean())}
    if te.sum() >= 20 and len(np.unique(yte)) == 2:
        p = m.predict_proba(Xte)[:, 1]
        metrics["roc_auc"] = round(float(roc_auc_score(yte, p)), 3)
        metrics["pr_auc"] = round(float(average_precision_score(yte, p)), 3)
        # 1-NN Tanimoto similarity-search baseline: score = max sim to a train active
        act = fp[ridx][tr][ytr == 1]
        if len(act) > BASE_REF_CAP:
            act = act[np.random.RandomState(0).choice(len(act), BASE_REF_CAP, False)]
        base = tanimoto_max(fp[ridx][te], act, act.sum(1))
        metrics["baseline_roc_auc"] = round(float(roc_auc_score(yte, base)), 3)
        metrics["lift"] = round(metrics["roc_auc"] - metrics["baseline_roc_auc"], 3)
    return m, metrics


def plots(results, out_dir):
    amr = [(PANEL[t][0], r) for t, r in results.items()
           if PANEL[t][1] == "amr" and "roc_auc" in r]
    amr.sort(key=lambda x: x[1]["roc_auc"])
    labels = [a[0] for a in amr]
    model_auc = [a[1]["roc_auc"] for a in amr]
    base_auc = [a[1]["baseline_roc_auc"] for a in amr]
    yy = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(labels))))
    ax.barh(yy + 0.2, model_auc, 0.4, label="QSAR model", color="#34d399")
    ax.barh(yy - 0.2, base_auc, 0.4, label="1-NN Tanimoto", color="#64748b")
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.5, color="#ef4444", lw=0.8, ls="--")
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("ROC-AUC (scaffold-held-out)")
    ax.set_title("AMR QSAR vs similarity-search baseline"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_dir}/auc_vs_baseline.png", dpi=130)
    plt.close(fig)


def main():
    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    Xdf, y = load(fs)
    Xmat, fp, folds = build_matrix(Xdf)

    results, models = {}, {}
    for tid, (label, kind) in PANEL.items():
        m, metrics = train_target(tid, y, Xmat, fp, folds)
        models[tid] = m
        results[tid] = {"label": label, "kind": kind, **metrics}
        print(f"{label:26s} n={metrics['n_train']+metrics['n_test']:6d} "
              f"auc={metrics.get('roc_auc','n/a')} "
              f"base={metrics.get('baseline_roc_auc','n/a')} "
              f"lift={metrics.get('lift','n/a')}", flush=True)

    amr_lifts = [r["lift"] for r in results.values()
                 if r["kind"] == "amr" and "lift" in r]
    amr_aucs = [r["roc_auc"] for r in results.values()
                if r["kind"] == "amr" and "roc_auc" in r]
    summary = {"mean_amr_auc": round(float(np.mean(amr_aucs)), 3),
               "mean_amr_lift": round(float(np.mean(amr_lifts)), 3),
               "n_amr_heads": len(amr_aucs)}
    print(f"\nSUMMARY {summary}", flush=True)

    # applicability-domain reference: diverse sample of all training fingerprints
    ref = fp if len(fp) <= AD_REF_CAP else fp[
        np.random.RandomState(0).choice(len(fp), AD_REF_CAP, False)]

    out = tempfile.mkdtemp()
    os.makedirs(f"{out}/models")
    for tid, m in models.items():
        joblib.dump(m, f"{out}/models/{tid}.joblib")
    np.savez_compressed(f"{out}/ad_reference.npz",
                        fp=ref.astype(np.uint8), pop=ref.sum(1).astype(np.int16))
    json.dump({"panel": {t: PANEL[t] for t in PANEL}, "results": results,
               "summary": summary}, open(f"{out}/metrics.json", "w"), indent=2)
    import shutil
    shutil.copy(f"{_here}/chem_features.py", f"{out}/chem_features.py")
    shutil.copy(f"{_here}/panel.py", f"{out}/panel.py")
    plots(results, out)

    mr = proj.get_model_registry()
    model = mr.python.create_model(
        MODEL_NAME,
        metrics={"mean_amr_auc": summary["mean_amr_auc"],
                 "mean_amr_lift": summary["mean_amr_lift"]},
        description="Per-target gradient-boosted QSAR on the AMR panel (stage-1 "
                    "bar). Scaffold-split ROC-AUC vs 1-NN Tanimoto baseline, with "
                    "an applicability-domain reference. Beats similarity search "
                    f"by {summary['mean_amr_lift']} mean AUC on {summary['n_amr_heads']} AMR heads.")
    model.save(out)
    print(f"registered {MODEL_NAME} v{model.version}", flush=True)


if __name__ == "__main__":
    main()
