"""Cache qsar_fv training data to a local parquet so every autoresearch
experiment re-trains without re-reading the feature store (recipe quirk #1).
Run once in the terminal (a read + write, no heavy fit, no RDKit).

    python3 autoresearch/prepare.py
"""
import os

import hopsworks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/data/qsar_cache.parquet"


def main():
    fs = hopsworks.login().get_feature_store()
    fv = fs.get_feature_view("qsar_fv", version=1)
    X, y = fv.training_data()
    X = X.drop(columns=[c for c in ("molecule_features_inchikey",
                                    "molecule_features_as_of", "as_of")
                        if c in X.columns], errors="ignore")
    X.columns = [c.replace("molecule_features_", "") for c in X.columns]
    df = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    for c in y.columns:
        df[c] = y[c].values
    os.makedirs(f"{REPO}/data", exist_ok=True)
    df.to_parquet(OUT)
    print(f"cached {len(df):,} rows x {df.shape[1]} cols -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
