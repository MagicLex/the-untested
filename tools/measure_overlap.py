"""Overlap measurement: how many LOTUS natural products carry a ChEMBL label,
per target. This is the number that sizes the whole system and fixes the AMR
target panel. Run once F1 (natural_product) and F2 (compound_activity) have
materialized.

    python3 tools/measure_overlap.py

Prints: total natural-product label overlap, the top targets by labelled-natural
count, and the AMR-organism targets specifically (the demo panel candidates).
"""
import hopsworks
import pandas as pd

AMR = [
    "Mycobacterium tuberculosis", "Staphylococcus aureus", "Escherichia coli",
    "Klebsiella pneumoniae", "Acinetobacter baumannii", "Pseudomonas aeruginosa",
    "Enterococcus faecium", "Enterobacter", "Candida albicans",
    "Plasmodium falciparum", "Trypanosoma brucei", "Trypanosoma cruzi",
    "Leishmania",
]


def _summarize(ca_nat):
    g = ca_nat.groupby(["target_chembl_id", "target_pref_name", "target_organism"])
    out = g.agg(n_nat_labeled=("inchikey", "nunique"),
                n_nat_active=("active", "sum")).reset_index()
    out["active_rate"] = (out["n_nat_active"] / out["n_nat_labeled"]).round(3)
    return out.sort_values("n_nat_labeled", ascending=False)


def main():
    fs = hopsworks.login().get_feature_store()
    print("reading natural_product ...", flush=True)
    nat = fs.get_feature_group("natural_product", 1).read()
    nat_keys = set(nat["inchikey"])
    print(f"  {len(nat_keys):,} natural-product InChIKeys", flush=True)

    print("reading compound_activity ...", flush=True)
    ca = fs.get_feature_group("compound_activity", 1).read()
    print(f"  {len(ca):,} activity rows | {ca['inchikey'].nunique():,} compounds "
          f"| {ca['target_chembl_id'].nunique():,} targets", flush=True)

    ca_nat = ca[ca["inchikey"].isin(nat_keys)].copy()
    print(f"\n=== natural products with a ChEMBL label: "
          f"{ca_nat['inchikey'].nunique():,} of {len(nat_keys):,} "
          f"({ca_nat['inchikey'].nunique()/len(nat_keys):.1%}) ===", flush=True)

    summ = _summarize(ca_nat)
    pd.set_option("display.width", 160, "display.max_colwidth", 40)
    print("\n=== top 25 targets by labelled-natural count ===")
    print(summ.head(25).to_string(index=False))

    amr = summ[summ["target_organism"].fillna("").str.contains(
        "|".join(AMR), case=False, regex=True)]
    print("\n=== AMR-organism targets (panel candidates) ===")
    print(amr.head(40).to_string(index=False))

    summ.to_csv("data/overlap_by_target.csv", index=False)
    print(f"\nfull table -> data/overlap_by_target.csv ({len(summ):,} targets)")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()
