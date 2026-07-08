"""F3b smiles-map: persist canonical SMILES for exactly the molecules already in
molecule_features, so the graph-input GNN (stage 2) can rebuild molecule graphs.

The fingerprint featurizer (features_pipeline) packs the Morgan bits and then
drops SMILES; a message-passing GNN needs the SMILES back. This writes a thin
sidecar FG keyed by InChIKey, 1:1 with molecule_features, and reuses that
pipeline's ChEMBL chemreps streamer so both resolve SMILES identically (no skew).
The GBM contract (molecule_features / qsar_fv) is untouched.

Runs on the feature env: no RDKit, just a text join + one chemreps stream. Heavy
FG read, so give the job 4 cores / 16 GB. Blocked-by F3 (molecule_features).
"""
import glob
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_here] + sorted(glob.glob("/hopsfs/Users/*/the-untested")):
    for _d in (_p, os.path.join(_p, "pipelines")):
        if os.path.isdir(_d) and _d not in sys.path:
            sys.path.insert(0, _d)

import pandas as pd  # noqa: E402

from features_pipeline import _chembl_smiles  # noqa: E402  reuse the exact stream


def main():
    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()

    mf = fs.get_feature_group("molecule_features", 1)
    need = set(mf.select(["inchikey"]).read()["inchikey"].tolist())
    print(f"{len(need):,} molecules in molecule_features need SMILES", flush=True)

    nat_fg = fs.get_feature_group("natural_product", 1)
    nat = nat_fg.read()[["inchikey", "smiles"]].dropna(subset=["smiles"]) \
        .drop_duplicates("inchikey")
    smiles = {ik: s for ik, s in zip(nat["inchikey"], nat["smiles"]) if ik in need}
    print(f"{len(smiles):,} resolved from natural_product", flush=True)

    smiles.update(_chembl_smiles(need - set(smiles)))
    smiles = {ik: s for ik, s in smiles.items() if isinstance(s, str) and s}
    print(f"{len(smiles):,} molecules with SMILES "
          f"({len(need) - len(smiles):,} unresolved)", flush=True)

    df = pd.DataFrame({"inchikey": list(smiles), "smiles": list(smiles.values())})
    df["as_of"] = pd.Timestamp.utcnow()

    fg = fs.get_or_create_feature_group(
        name="molecule_smiles", version=1,
        description="Canonical SMILES per molecule in molecule_features (LOTUS "
                    "naturals + ChEMBL training compounds), keyed by InChIKey. "
                    "Graph input for the message-passing GNN; the fingerprint FG "
                    "drops SMILES after packing the Morgan bits.",
        primary_key=["inchikey"], event_time="as_of", parents=[mf, nat_fg],
        online_enabled=False, statistics_config=False)
    fg.insert(df, write_options={"start_offline_materialization": True})
    print(f"inserted {len(df):,} rows into molecule_smiles", flush=True)


if __name__ == "__main__":
    main()
