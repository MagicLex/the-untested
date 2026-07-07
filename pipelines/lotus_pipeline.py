"""F1 ingest-lotus: the plant<->molecule map + molecule structures + taxonomy.

Downloads the LOTUS frozen metadata (one bulk CSV on Zenodo, structure-organism
pairs with SMILES, chemical class, and full organism taxonomy) and writes two
offline feature groups:

  natural_product     one row per molecule (InChIKey): SMILES, formula, mass,
                      xlogp, NPClassifier + ClassyFire chemical class. The
                      prediction set (the untested naturals) lives here.
  organism_compound   one row per (organism, molecule): the map edge plus the
                      organism's full taxonomy (domain..species). The taxonomic
                      prior baseline reads from these columns.

Backfill once; LOTUS releases are rare, so this is scheduled loosely (monthly).
Nothing is hardcoded to a snapshot: the versioned Zenodo record is resolved from
the concept DOI at run time.
"""
import gzip
import io
import os
import tempfile

import pandas as pd
import requests

# LOTUS concept DOI (always resolves to the latest frozen release).
ZENODO_CONCEPT = "https://zenodo.org/api/records/5794106"
META_FILE = "260413_frozen_metadata.csv.gz"

# metadata columns we keep (by name; the file has 39)
MOL_COLS = {
    "structure_inchikey": "inchikey",
    "structure_smiles": "smiles",
    "structure_molecular_formula": "formula",
    "structure_exact_mass": "exact_mass",
    "structure_xlogp": "xlogp",
    "structure_taxonomy_npclassifier_01pathway": "npc_pathway",
    "structure_taxonomy_npclassifier_02superclass": "npc_superclass",
    "structure_taxonomy_npclassifier_03class": "npc_class",
    "structure_taxonomy_classyfire_01kingdom": "cf_kingdom",
    "structure_taxonomy_classyfire_02superclass": "cf_superclass",
    "structure_taxonomy_classyfire_03class": "cf_class",
}
ORG_COLS = {
    "structure_inchikey": "inchikey",
    "organism_name": "organism",
    "organism_taxonomy_01domain": "tax_domain",
    "organism_taxonomy_02kingdom": "tax_kingdom",
    "organism_taxonomy_03phylum": "tax_phylum",
    "organism_taxonomy_04class": "tax_class",
    "organism_taxonomy_05order": "tax_order",
    "organism_taxonomy_06family": "tax_family",
    "organism_taxonomy_08genus": "tax_genus",
    "organism_taxonomy_09species": "tax_species",
}


def _resolve_meta_url():
    rec = requests.get(ZENODO_CONCEPT, timeout=60).json()
    for f in rec["files"]:
        if f["key"] == META_FILE:
            return f["links"]["self"]
    raise RuntimeError(f"{META_FILE} not found in LOTUS Zenodo record")


def _load_metadata():
    url = _resolve_meta_url()
    print(f"streaming LOTUS metadata {url}", flush=True)
    tmp = os.path.join(tempfile.gettempdir(), META_FILE)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    keep = sorted(set(MOL_COLS) | set(ORG_COLS))
    with gzip.open(tmp, "rt") as fh:
        df = pd.read_csv(fh, usecols=keep, dtype=str)
    os.remove(tmp)
    print(f"loaded {len(df):,} structure-organism rows", flush=True)
    return df


def _molecules(df):
    m = df[list(MOL_COLS)].rename(columns=MOL_COLS)
    m = m.dropna(subset=["inchikey", "smiles"]).drop_duplicates("inchikey")
    for c in ("exact_mass", "xlogp"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    print(f"distinct molecules: {len(m):,}", flush=True)
    return m.reset_index(drop=True)


def _links(df):
    o = df[list(ORG_COLS)].rename(columns=ORG_COLS)
    o = o.dropna(subset=["inchikey", "organism"]).drop_duplicates(["organism", "inchikey"])
    print(f"distinct organism-molecule links: {len(o):,}  "
          f"({o['organism'].nunique():,} organisms)", flush=True)
    return o.reset_index(drop=True)


def main():
    import hopsworks
    df = _load_metadata()
    mol = _molecules(df)
    lnk = _links(df)
    now = pd.Timestamp.utcnow()
    mol["as_of"] = now
    lnk["as_of"] = now

    proj = hopsworks.login()
    fs = proj.get_feature_store()

    np_fg = fs.get_or_create_feature_group(
        name="natural_product", version=1,
        description="LOTUS natural products, one row per molecule (InChIKey): "
                    "SMILES, formula, exact mass, xlogp, NPClassifier + "
                    "ClassyFire chemical class. The prediction set for the "
                    "untested-bioactivity map.",
        primary_key=["inchikey"], event_time="as_of",
        online_enabled=False, statistics_config=False)
    np_fg.insert(mol, write_options={"start_offline_materialization": True})
    print(f"inserted {len(mol):,} molecules into natural_product", flush=True)

    oc_fg = fs.get_or_create_feature_group(
        name="organism_compound", version=1,
        description="LOTUS organism<->molecule edges with full organism "
                    "taxonomy (domain..species). The plant<->molecule map and "
                    "the source of the taxonomic-prior baseline.",
        primary_key=["organism", "inchikey"], event_time="as_of",
        online_enabled=False, statistics_config=False)
    oc_fg.insert(lnk, write_options={"start_offline_materialization": True})
    print(f"inserted {len(lnk):,} links into organism_compound", flush=True)


if __name__ == "__main__":
    main()
