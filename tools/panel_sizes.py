"""Training-pool sizing: for the AMR panel candidates, how many TOTAL labelled
compounds (synthetic + natural) exist per target, next to the natural-only count.
The total is the training pool; the natural count is the validation/prediction
slice. Run after measure_overlap.py.

    python3 tools/panel_sizes.py
"""
import hopsworks
import pandas as pd

CANDIDATES = {
    "CHEMBL364": "P. falciparum (malaria)",
    "CHEMBL368": "T. cruzi (Chagas)",
    "CHEMBL367": "L. donovani",
    "CHEMBL612849": "T. brucei",
    "CHEMBL352": "S. aureus",
    "CHEMBL354": "E. coli",
    "CHEMBL360": "M. tuberculosis",
    "CHEMBL348": "P. aeruginosa",
    "CHEMBL357": "E. faecium",
    "CHEMBL366": "C. albicans",
    "CHEMBL2026": "Beta-lactamase (E. coli)",
    "CHEMBL1857": "FabI enoyl-reductase (E. coli)",
}


def main():
    fs = hopsworks.login().get_feature_store()
    nat = set(fs.get_feature_group("natural_product", 1).read()["inchikey"])
    ca = fs.get_feature_group("compound_activity", 1).read()
    sub = ca[ca["target_chembl_id"].isin(CANDIDATES)]
    rows = []
    for tid, label in CANDIDATES.items():
        t = sub[sub["target_chembl_id"] == tid]
        tnat = t[t["inchikey"].isin(nat)]
        rows.append({
            "target": label,
            "total_compounds": t["inchikey"].nunique(),
            "total_active_rate": round(t["active"].mean(), 3) if len(t) else 0,
            "nat_compounds": tnat["inchikey"].nunique(),
            "nat_active_rate": round(tnat["active"].mean(), 3) if len(tnat) else 0,
        })
    out = pd.DataFrame(rows).sort_values("total_compounds", ascending=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
