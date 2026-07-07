"""F2 ingest-chembl: the bioactivity label, from the ChEMBL bulk SQLite dump.

The ChEMBL REST API dies on bulk pulls, so this uses the canonical SQLite
release. It downloads the dump (~5.8 GB), extracts the single .db file to local
scratch, runs ONE aggregating query over every activity that carries a
`pchembl_value` (the normalized potency), collapses repeat measurements to one
row per (molecule, target), and writes the `compound_activity` feature group.

No target panel is applied here on purpose: we store every pchembl-labelled
(InChIKey, target) pair across all of ChEMBL. The AMR target panel is chosen
downstream (T1/T2) from measured LOTUS-overlap density, not hardcoded. Training
uses all labelled compounds; the natural-product overlap is the validation slice.

`active` is pchembl_mean >= 6.0 (<=1 uM), the standard QSAR actives cutoff;
pchembl_mean is kept so training can rethreshold per target.

ChEMBL releases quarterly, so this is scheduled loosely (monthly). The release
version is resolved from the FTP "latest" directory, nothing is pinned.
"""
import os
import re
import sqlite3
import tarfile
import tempfile

import pandas as pd
import requests

FTP_LATEST = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
ACTIVE_PCHEMBL = 6.0

QUERY = """
SELECT cs.standard_inchi_key      AS inchikey,
       td.chembl_id               AS target_chembl_id,
       td.pref_name               AS target_pref_name,
       td.organism                AS target_organism,
       td.target_type             AS target_type,
       AVG(act.pchembl_value)     AS pchembl_mean,
       COUNT(*)                   AS n_meas
FROM activities act
JOIN assays a               ON act.assay_id = a.assay_id
JOIN target_dictionary td   ON a.tid = td.tid
JOIN compound_structures cs ON act.molregno = cs.molregno
WHERE act.pchembl_value IS NOT NULL
  AND cs.standard_inchi_key IS NOT NULL
GROUP BY cs.standard_inchi_key, td.chembl_id
"""


def _resolve_sqlite_url():
    idx = requests.get(FTP_LATEST, timeout=60).text
    m = re.search(r"chembl_(\d+)_sqlite\.tar\.gz", idx)
    if not m:
        raise RuntimeError("no chembl_*_sqlite.tar.gz in latest FTP listing")
    return m.group(0), int(m.group(1)), FTP_LATEST + m.group(0)


def _download_and_extract(url, fname):
    scratch = tempfile.gettempdir()
    tar_path = os.path.join(scratch, fname)
    print(f"downloading {url}", flush=True)
    with requests.get(url, stream=True, timeout=3600) as r:
        r.raise_for_status()
        with open(tar_path, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    print(f"extracting .db from {tar_path}", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".db"))
        tar.extract(member, path=scratch)
        db_path = os.path.join(scratch, member.name)
    os.remove(tar_path)
    print(f"db at {db_path} ({os.path.getsize(db_path) / 1e9:.1f} GB)", flush=True)
    return db_path


def _query(db_path):
    con = sqlite3.connect(db_path)
    print("running aggregating activity query (this takes a few minutes)", flush=True)
    df = pd.read_sql_query(QUERY, con)
    con.close()
    df["pchembl_mean"] = pd.to_numeric(df["pchembl_mean"], errors="coerce")
    df["active"] = (df["pchembl_mean"] >= ACTIVE_PCHEMBL).astype(int)
    print(f"{len(df):,} (molecule, target) label rows  |  "
          f"{df['inchikey'].nunique():,} compounds  |  "
          f"{df['target_chembl_id'].nunique():,} targets  |  "
          f"{df['active'].mean():.1%} active", flush=True)
    return df


def main():
    import hopsworks
    fname, version, url = _resolve_sqlite_url()
    print(f"ChEMBL release {version}", flush=True)
    db_path = _download_and_extract(url, fname)
    try:
        df = _query(db_path)
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
    df["chembl_release"] = version
    df["as_of"] = pd.Timestamp.utcnow()

    proj = hopsworks.login()
    fs = proj.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="compound_activity", version=1,
        description="ChEMBL bioactivity labels: one row per (molecule InChIKey, "
                    "target), mean pchembl over repeat measurements plus an "
                    "active flag (pchembl>=6.0, <=1uM). Every pchembl-labelled "
                    "pair across all of ChEMBL; the target panel is selected "
                    "downstream from LOTUS-overlap density.",
        primary_key=["inchikey", "target_chembl_id"], event_time="as_of",
        online_enabled=False, statistics_config=False)
    fg.insert(df, write_options={"start_offline_materialization": True})
    print(f"inserted {len(df):,} activity labels into compound_activity", flush=True)


if __name__ == "__main__":
    main()
