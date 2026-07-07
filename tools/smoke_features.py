"""Smoke test: validate chem_features on the RDKit env before the big F3 job."""
import glob
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_here] + sorted(glob.glob("/hopsfs/Users/*/the-untested")):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import chem_features as cf

TESTS = {
    "quinine": "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "bad": "not_a_smiles",
}

for name, smi in TESTS.items():
    packed, desc = cf.featurize_packed(smi)
    if packed is None:
        print(f"{name}: rejected (as expected for bad)" if name == "bad"
              else f"{name}: UNEXPECTED None", flush=True)
        continue
    fp = cf.unpack(packed)
    print(f"{name}: fp_bits_on={int(fp.sum())} b64_len={len(packed)} "
          f"desc={[round(d, 1) for d in desc]} scaffold={cf.scaffold(smi)}", flush=True)
