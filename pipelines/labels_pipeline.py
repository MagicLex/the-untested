"""F2b pivot-labels: reshape the panel's activity into the wide multi-task
label group the feature view joins 1:1 to molecule_features.

compound_activity is keyed by (InChIKey, target), so a feature view cannot join
it on InChIKey alone (Hopsworks requires the full join key). The multi-task
label is therefore stored as compound_labels: one row per molecule, one column
per panel target, value in {1 active, 0 inactive, NaN unmeasured}. A molecule
with no panel label is absent.

Rebuild whenever the panel changes. Blocked-by F2 (compound_activity).
"""
import glob
import os
import sys

import pandas as pd

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_here] + sorted(glob.glob("/hopsfs/Users/*/the-untested")):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)
from panel import PANEL, ycol  # noqa: E402


def main():
    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()

    print("reading compound_activity ...", flush=True)
    ca = fs.get_feature_group("compound_activity", 1).read()
    ca = ca[ca["target_chembl_id"].isin(PANEL)][["inchikey", "target_chembl_id", "active"]]

    wide = ca.pivot_table(index="inchikey", columns="target_chembl_id",
                          values="active", aggfunc="max")
    wide = wide.reindex(columns=list(PANEL))          # every panel target a column
    wide.columns = [ycol(t) for t in wide.columns]
    wide = wide.astype("float64").reset_index()       # NaN = unmeasured
    wide["as_of"] = pd.Timestamp.utcnow()

    n_lab = int(wide.drop(columns=["inchikey", "as_of"]).notna().sum().sum())
    print(f"{len(wide):,} molecules with a panel label, {n_lab:,} labels total",
          flush=True)

    fg = fs.get_or_create_feature_group(
        name="compound_labels", version=1,
        description="Wide multi-task AMR labels: one row per molecule, one "
                    "column per panel target (1 active / 0 inactive / null "
                    "unmeasured). Joined 1:1 to molecule_features in qsar_fv.",
        primary_key=["inchikey"], event_time="as_of",
        online_enabled=False, statistics_config=False)
    fg.insert(wide, write_options={"start_offline_materialization": True})
    print(f"inserted {len(wide):,} rows into compound_labels", flush=True)


if __name__ == "__main__":
    main()
