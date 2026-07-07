"""F3 featurize-molecules: SMILES -> model input, stored in molecule_features.

Featurizes every molecule the model needs input for:
  - all LOTUS naturals (the prediction set), SMILES from natural_product;
  - every ChEMBL compound on a target with enough labels to be a training task
    (>= MIN_TARGET_COMPOUNDS), SMILES from the ChEMBL chemreps bulk file.

The fingerprint + descriptors come from the shared chem_features.py so training,
the batch map, and serving all featurize identically (no skew). The 2048-bit
fingerprint is stored base64-packed in one column; descriptors as named floats.

Runs on untested-chem-env (RDKit). Blocked-by F1 (natural_product) and F2
(compound_activity). Featurization is CPU-bound, so it fans out over the job's
cores with a process pool.
"""
import glob
import gzip
import os
import sys
import tempfile
from multiprocessing import Pool

import pandas as pd
import requests

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_here] + sorted(glob.glob("/hopsfs/Users/*/the-untested")):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import chem_features as cf  # noqa: E402

MIN_TARGET_COMPOUNDS = 50
CHEMREPS = ("https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
            "chembl_37_chemreps.txt.gz")


def _worker(item):
    ik, smiles = item
    packed, desc, scaf = cf.featurize_full(smiles)
    if packed is None:
        return None
    return (ik, packed, scaf, *desc)


def _chembl_smiles(needed):
    """Stream the ChEMBL chemreps bulk file, return {inchikey: smiles} for the
    InChIKeys in `needed`."""
    tmp = os.path.join(tempfile.gettempdir(), "chemreps.txt.gz")
    print(f"downloading chemreps for {len(needed):,} ChEMBL compounds", flush=True)
    with requests.get(CHEMREPS, stream=True, timeout=1200) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    out = {}
    with gzip.open(tmp, "rt") as fh:
        next(fh)  # header: chembl_id, canonical_smiles, standard_inchi, standard_inchi_key
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            ik = parts[3]
            if ik in needed:
                out[ik] = parts[1]
    os.remove(tmp)
    print(f"resolved SMILES for {len(out):,} ChEMBL compounds", flush=True)
    return out


def main():
    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()

    print("reading compound_activity for training targets ...", flush=True)
    ca = fs.get_feature_group("compound_activity", 1).read()
    tcounts = ca.groupby("target_chembl_id")["inchikey"].nunique()
    keep_targets = set(tcounts[tcounts >= MIN_TARGET_COMPOUNDS].index)
    train_keys = set(ca[ca["target_chembl_id"].isin(keep_targets)]["inchikey"])
    print(f"{len(keep_targets):,} training targets (>= {MIN_TARGET_COMPOUNDS} "
          f"compounds) -> {len(train_keys):,} training compounds", flush=True)

    print("reading natural_product ...", flush=True)
    nat = fs.get_feature_group("natural_product", 1).read()[["inchikey", "smiles"]]
    nat = nat.dropna(subset=["smiles"]).drop_duplicates("inchikey")
    smiles = dict(zip(nat["inchikey"], nat["smiles"]))
    print(f"{len(smiles):,} natural products with SMILES", flush=True)

    chembl_only = train_keys - set(smiles)
    smiles.update(_chembl_smiles(chembl_only))

    items = [(ik, s) for ik, s in smiles.items() if isinstance(s, str) and s]
    print(f"featurizing {len(items):,} molecules ...", flush=True)
    cols = ["inchikey", "fp_b64", "scaffold", *cf.DESCRIPTORS]
    with Pool(processes=max(1, os.cpu_count())) as pool:
        rows = [r for r in pool.imap_unordered(_worker, items, chunksize=2000)
                if r is not None]
    print(f"featurized {len(rows):,} (dropped {len(items) - len(rows):,} bad SMILES)",
          flush=True)

    df = pd.DataFrame(rows, columns=cols)
    df["as_of"] = pd.Timestamp.utcnow()

    fg = fs.get_or_create_feature_group(
        name="molecule_features", version=1,
        description="Skew-free molecule input: 2048-bit Morgan fingerprint "
                    "(base64-packed) + Bemis-Murcko scaffold + 10 physchem "
                    "descriptors, from chem_features.py. LOTUS naturals + ChEMBL "
                    "training compounds. Keyed by InChIKey.",
        primary_key=["inchikey"], event_time="as_of",
        online_enabled=False, statistics_config=False)
    fg.insert(df, write_options={"start_offline_materialization": True})
    print(f"inserted {len(df):,} rows into molecule_features", flush=True)


if __name__ == "__main__":
    main()
