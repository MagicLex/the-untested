"""Shared, skew-free molecule featurizer. One source of truth for turning a
SMILES into model input, imported by the featurize pipeline (F3), the training
scaffold split (T1/T2), the batch map (I1), and the serving predictor (I2).

Keep this the ONLY place SMILES becomes features. Fingerprint radius/size and
the descriptor list are frozen here so training and serving never diverge.

Needs RDKit (jobs/serving run on an env clone that pins it).
"""
import base64

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

FP_RADIUS = 2
FP_BITS = 2048
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

# frozen descriptor order (the model input beyond the fingerprint)
DESCRIPTORS = [
    "mol_wt", "logp", "tpsa", "hbd", "hba", "rot_bonds",
    "aromatic_rings", "ring_count", "frac_csp3", "heavy_atoms",
]


def mol(smiles):
    """Parse a SMILES to an RDKit mol, or None if it does not parse."""
    if not smiles or not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def fingerprint(m):
    """2048-bit Morgan (ECFP4) as a float32 numpy vector."""
    fp = _GEN.GetFingerprint(m)
    arr = np.zeros(FP_BITS, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def descriptors(m):
    """Frozen physchem descriptor vector, same order as DESCRIPTORS."""
    return np.array([
        Descriptors.MolWt(m),
        Crippen.MolLogP(m),
        rdMolDescriptors.CalcTPSA(m),
        rdMolDescriptors.CalcNumHBD(m),
        rdMolDescriptors.CalcNumHBA(m),
        rdMolDescriptors.CalcNumRotatableBonds(m),
        rdMolDescriptors.CalcNumAromaticRings(m),
        rdMolDescriptors.CalcNumRings(m),
        rdMolDescriptors.CalcFractionCSP3(m),
        m.GetNumHeavyAtoms(),
    ], dtype=np.float32)


def scaffold(smiles):
    """Bemis-Murcko scaffold SMILES, the grouping key for a leak-free split.
    Empty string for acyclic molecules (no ring system)."""
    m = mol(smiles)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def featurize(smiles):
    """SMILES -> (fingerprint[2048], descriptors[10]) as float32 arrays, or
    (None, None) if the SMILES does not parse. Callers concatenate as needed."""
    m = mol(smiles)
    if m is None:
        return None, None
    return fingerprint(m), descriptors(m)


def pack(fp):
    """2048-bit fingerprint (0/1 float array) -> base64 string (256 bytes
    packed) for compact single-column feature-group storage. Lossless."""
    return base64.b64encode(np.packbits(fp.astype(np.uint8)).tobytes()).decode("ascii")


def unpack(s):
    """base64 packed fingerprint -> 2048 float32 array. Inverse of pack()."""
    return np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8)).astype(np.float32)


def featurize_packed(smiles):
    """SMILES -> (packed_fp base64 str, descriptors[10 floats]) for building a
    molecule_features row, or (None, None) on a bad SMILES."""
    fp, desc = featurize(smiles)
    if fp is None:
        return None, None
    return pack(fp), desc.tolist()
