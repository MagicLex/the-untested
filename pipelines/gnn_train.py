"""T3 train-gnn (stage 2): a multi-task message-passing GNN (Chemprop D-MPNN) on
the AMR panel, evaluated on the SAME scaffold fold-0 held-out molecules as the
stage-1 GBM so the two are compared head-to-head, not on different test sets.

One shared graph encoder over the molecule graph, 14 per-target binary heads,
NaN-masked multitask BCE (a molecule contributes only to the targets it was
measured on). Fold assignment reuses the GBM's exact scaffold hash: fold 0 = test
(the bar's test set), fold 1 = validation, folds 2-4 = train.

No GPU on the cluster, so this trains on CPU. `--sample N` subsamples the TRAIN
pool only (test stays the full fold 0) for a fast feasibility run before the
full-data run; the subsample size is logged, never silent. Reports per-head
ROC-AUC / PR-AUC / precision@k and the mean AMR AUC, prints them beside the
stage-1 bar, and (with --register) registers the model every run. Promote to
amr_qsar only if it clears the bar. Runs on untested-gnn-env.

    python3 pipelines/gnn_train.py [--sample N] [--epochs E] [--batch B]
                                   [--workers W] [--register]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_here] + sorted(glob.glob("/hopsfs/Users/*/the-untested")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from panel import PANEL, ycol  # noqa: E402

N_FOLDS = 5
FV_NAME = "qsar_gnn_fv"
MODEL_NAME = "amr_qsar_gnn"
BAR_MODEL = "amr_qsar"
SMILES_COL = "molecule_smiles_smiles"
PANEL_TIDS = list(PANEL)


def scaffold_fold(scaffold, inchikey):
    """Identical to the stage-1 GBM's fold hash, so fold 0 here is the exact same
    held-out molecules the bar was scored on. Acyclic (empty scaffold) falls back
    to the InChIKey."""
    key = scaffold if scaffold else inchikey
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % N_FOLDS


def feature_view(fs):
    try:
        fv = fs.get_feature_view(FV_NAME, version=1)
        if fv is not None:
            return fv
    except Exception:
        pass
    # compound_labels v1 offline table is corrupt (AFS read + Hudi write both
    # fail); v2 is the clean re-materialize. The GBM's qsar_fv still roots on v1
    # until that is fixed deliberately, awake.
    cl = fs.get_feature_group("compound_labels", 2)
    ms = fs.get_feature_group("molecule_smiles", 1)
    q = cl.select_all().join(ms.select_all(), on=["inchikey"], join_type="inner")
    return fs.create_feature_view(
        name=FV_NAME, version=1, query=q, labels=[ycol(t) for t in PANEL],
        description="Graph-input contract for the stage-2 GNN: compound_labels "
                    "joined 1:1 to molecule_smiles on InChIKey.")


def load(fs):
    import time
    fv = feature_view(fs)
    # The Arrow Flight query service flakes under cluster load on free tier
    # ("Could not read data using Hopsworks Query Service"). Bounded retry so a
    # transient read miss self-heals in-job instead of failing the whole run.
    for attempt in range(6):
        try:
            X, y = fv.training_data()
            break
        except Exception as e:
            if attempt == 5:
                raise
            print(f"training_data read failed ({str(e)[:80]}); retry {attempt+1}/5 in 60s",
                  flush=True)
            time.sleep(60)
    ik = (X["inchikey"] if "inchikey" in X.columns
          else X["molecule_smiles_inchikey"]).astype(str).values
    smi = X[SMILES_COL].astype(str).values
    # scaffold from molecule_features -> byte-identical folds to the GBM bar
    mf = fs.get_feature_group("molecule_features", 1).select(["inchikey", "scaffold"]).read()
    scaf_map = dict(zip(mf["inchikey"].astype(str), mf["scaffold"].fillna("")))
    scaf = np.array([scaf_map.get(k, "") for k in ik])
    folds = np.array([scaffold_fold(s, k) for s, k in zip(scaf, ik)])
    Y = y[[ycol(t) for t in PANEL]].to_numpy(np.float32)
    print(f"qsar_gnn_fv rows={len(smi):,}  labels={Y.shape[1]}  "
          f"fold sizes={np.bincount(folds, minlength=N_FOLDS).tolist()}", flush=True)
    return smi, Y, folds


def make_dataset(smi, Y, idx):
    from chemprop import data
    dps = []
    for i in idx:
        try:
            dps.append(data.MoleculeDatapoint.from_smi(smi[i], Y[i]))
        except Exception:
            pass
    return data.MoleculeDataset(dps)


def head_metrics(ytrue, p, ks=(50, 100)):
    m = {}
    mask = ~np.isnan(ytrue)
    yt = ytrue[mask].astype(int)
    pp = p[mask]
    if len(yt) >= 20 and len(np.unique(yt)) == 2:
        m["roc_auc"] = round(float(roc_auc_score(yt, pp)), 3)
        m["pr_auc"] = round(float(average_precision_score(yt, pp)), 3)
        order = np.argsort(-pp)
        for k in ks:
            m[f"p@{k}"] = round(float(yt[order[:min(k, len(yt))]].mean()), 3)
        m["n_test"] = int(len(yt))
        m["active_rate"] = round(float(yt.mean()), 3)
    return m


def bar_per_head(mr):
    """Stage-1 GBM per-head ROC-AUC, read from the champion's bundle, for a
    side-by-side. Best-effort: a missing bar just prints as n/a."""
    try:
        m = mr.get_model(BAR_MODEL, version=1)
        d = m.download()
        res = json.load(open(os.path.join(d, "metrics.json")))["results"]
        return {t: r.get("roc_auc") for t, r in res.items()}
    except Exception as e:
        print(f"(could not load stage-1 bar: {e})", flush=True)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args()

    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    smi, Y, folds = load(fs)

    rng = np.random.RandomState(0)
    te = np.where(folds == 0)[0]
    va = np.where(folds == 1)[0]
    tr = np.where(folds >= 2)[0]
    if args.sample and args.sample < len(tr):
        tr = rng.choice(tr, args.sample, replace=False)
        print(f"SUBSAMPLE train -> {len(tr):,} rows (test/val full)", flush=True)
    print(f"train={len(tr):,} val={len(va):,} test={len(te):,}", flush=True)

    from chemprop import data, models, nn
    import lightning.pytorch as pl

    train_dset, val_dset, test_dset = (make_dataset(smi, Y, ix) for ix in (tr, va, te))
    train_loader = data.build_dataloader(train_dset, batch_size=args.batch,
                                         num_workers=args.workers)
    val_loader = data.build_dataloader(val_dset, batch_size=args.batch,
                                       num_workers=args.workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, batch_size=args.batch,
                                        num_workers=args.workers, shuffle=False)

    T = Y.shape[1]
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(n_tasks=T)
    model = models.MPNN(mp, agg, ffn, batch_norm=True,
                        metrics=[nn.metrics.BinaryAUROC()])

    # cap val cost on CPU: the fold-1 val set is large and only feeds the epoch
    # monitoring metric, so a slice is enough. Test (fold 0) is scored in full.
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="cpu", devices=1,
                         enable_checkpointing=False, logger=False,
                         enable_progress_bar=False, num_sanity_val_steps=0,
                         limit_val_batches=20)
    trainer.fit(model, train_loader, val_loader)

    preds = trainer.predict(model, test_loader)
    P = np.concatenate([np.asarray(b) for b in preds], axis=0).reshape(len(test_dset), T)

    bar = bar_per_head(proj.get_model_registry())
    results, amr_aucs = {}, []
    Yte = Y[te]
    print(f"\n{'target':26s} {'gnn_auc':>8s} {'bar_auc':>8s} {'pr_auc':>7s} "
          f"{'p@50':>6s} {'p@100':>6s}", flush=True)
    for j, tid in enumerate(PANEL_TIDS):
        hm = head_metrics(Yte[:, j], P[:, j])
        label, kind = PANEL[tid]
        results[tid] = {"label": label, "kind": kind, **hm}
        b = bar.get(tid)
        if kind == "amr" and "roc_auc" in hm:
            amr_aucs.append(hm["roc_auc"])
        print(f"{label:26s} {hm.get('roc_auc','n/a'):>8} {str(b) if b else 'n/a':>8} "
              f"{hm.get('pr_auc','n/a'):>7} {hm.get('p@50','n/a'):>6} "
              f"{hm.get('p@100','n/a'):>6}", flush=True)

    summary = {"mean_amr_auc": round(float(np.mean(amr_aucs)), 3) if amr_aucs else None,
               "n_amr_heads": len(amr_aucs), "n_train": int(len(tr)),
               "sample": args.sample, "epochs": args.epochs}
    print(f"\nSUMMARY {summary}", flush=True)

    if not args.register:
        print("(smoke run, not registered)", flush=True)
        return

    out = tempfile.mkdtemp()
    trainer.save_checkpoint(os.path.join(out, "model.ckpt"))
    json.dump({"panel": {t: PANEL[t] for t in PANEL}, "results": results,
               "summary": summary}, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    import shutil
    # the job relocates this script to /hopsfs/Resources/jobs/...; derive the
    # real repo from an imported repo module, not __file__ (the logged scar).
    repo = os.path.dirname(os.path.abspath(sys.modules["panel"].__file__))
    shutil.copy(os.path.join(repo, "panel.py"), os.path.join(out, "panel.py"))

    mr = proj.get_model_registry()
    mdl = mr.python.create_model(
        MODEL_NAME, metrics={"mean_amr_auc": summary["mean_amr_auc"]},
        description=f"Stage-2 multi-task Chemprop D-MPNN on the AMR panel, same "
                    f"scaffold fold-0 test as {BAR_MODEL}. Mean AMR AUC "
                    f"{summary['mean_amr_auc']} on {summary['n_amr_heads']} heads, "
                    f"trained on {summary['n_train']:,} molecules.")
    mdl.save(out)
    print(f"registered {MODEL_NAME} v{mdl.version}", flush=True)


if __name__ == "__main__":
    main()
