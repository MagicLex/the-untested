"""The AMR target panel: single source of truth, imported by the label pivot
pipeline, training, the batch map, and serving.

AMR heads are the product (which untested plant might hit a drug-resistant
pathogen). Cytotox heads are auxiliary: they power the selectivity overlay
(active on the pathogen, not on human cells) and fatten the shared chemistry.

target_chembl_id -> (label, kind). Hopsworks lowercases feature names, so the
wide label column for a target is ycol(tid) = "y_<tid lowercased>".
"""
PANEL = {
    "CHEMBL364": ("P. falciparum (malaria)", "amr"),
    "CHEMBL2026": ("Beta-lactamase", "amr"),
    "CHEMBL368": ("T. cruzi (Chagas)", "amr"),
    "CHEMBL367": ("L. donovani", "amr"),
    "CHEMBL612849": ("T. brucei", "amr"),
    "CHEMBL352": ("S. aureus", "amr"),
    "CHEMBL354": ("E. coli", "amr"),
    "CHEMBL360": ("M. tuberculosis", "amr"),
    "CHEMBL348": ("P. aeruginosa", "amr"),
    "CHEMBL366": ("C. albicans", "amr"),
    "CHEMBL357": ("E. faecium", "amr"),
    "CHEMBL392": ("A549 (lung)", "cytotox"),
    "CHEMBL395": ("HepG2 (liver)", "cytotox"),
    "CHEMBL399": ("HeLa", "cytotox"),
}


def ycol(tid):
    return "y_" + tid.lower()


def amr_targets():
    return [t for t, (_, k) in PANEL.items() if k == "amr"]
