"""Cheminformatics helpers: standardization, hashing, featurization, scaffolds.

Most functions accept either a SMILES string or an RDKit ``Mol`` and return
``None`` for unparseable inputs, so they can be applied to whole columns
without try/except.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

# Silence RDKit warnings that flood notebooks (we'll surface failures via None).
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Parsing + standardization
# ---------------------------------------------------------------------------

_CHARGE_NEUTRALIZER = rdMolStandardize.Uncharger()
_FRAGMENT_PARENT = rdMolStandardize.LargestFragmentChooser()


def to_mol(smi: str | Chem.Mol | None) -> Chem.Mol | None:
    """Parse a SMILES (or pass through a Mol). Returns None on failure."""
    if smi is None:
        return None
    if isinstance(smi, Chem.Mol):
        return smi
    if not isinstance(smi, str) or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi)


def standardize(smi: str | Chem.Mol | None) -> Chem.Mol | None:
    """Sanitize, take the largest fragment (drop counterions), neutralize charges.

    Stereo is preserved. Tautomers are NOT canonicalized — that's a stronger
    transformation often best decided per-task.
    """
    mol = to_mol(smi)
    if mol is None:
        return None
    try:
        mol = _FRAGMENT_PARENT.choose(mol)
        mol = _CHARGE_NEUTRALIZER.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def standardize_smiles(smi: str | None) -> str | None:
    mol = standardize(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def to_inchikey(smi: str | Chem.Mol | None) -> str | None:
    mol = to_mol(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scaffolds
# ---------------------------------------------------------------------------

def bemis_murcko(smi: str | Chem.Mol | None, generic: bool = False) -> str | None:
    """Bemis-Murcko scaffold. With generic=True, atoms become carbons / bonds single."""
    mol = to_mol(smi)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def morgan_fp(smi: str | Chem.Mol | None, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Morgan / ECFP bit vector as uint8 numpy array."""
    mol = to_mol(smi)
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bv = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def morgan_fp_batch(
    smiles: Sequence[str | None], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """Stack Morgan FPs into an (N, n_bits) uint8 matrix; failed parses become zero rows."""
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for i, s in enumerate(smiles):
        mol = to_mol(s)
        if mol is None:
            continue
        bv = gen.GetFingerprint(mol)
        Chem.DataStructs.ConvertToNumpyArray(bv, out[i])
    return out


def tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Tanimoto on uint8 bit-vectors."""
    inter = np.bitwise_and(fp_a, fp_b).sum()
    union = np.bitwise_or(fp_a, fp_b).sum()
    return float(inter) / float(union) if union else 0.0


# ---------------------------------------------------------------------------
# Physchem properties
# ---------------------------------------------------------------------------

def compute_physchem(smi: str | Chem.Mol | None) -> dict[str, float] | None:
    """Common property set: MW, logP, TPSA, HBD, HBA, rotbonds, fsp3, ring counts."""
    mol = to_mol(smi)
    if mol is None:
        return None
    try:
        return {
            "mw": Descriptors.MolWt(mol),
            "heavy_atoms": Descriptors.HeavyAtomCount(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rotbonds": Lipinski.NumRotatableBonds(mol),
            "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "n_rings": rdMolDescriptors.CalcNumRings(mol),
            "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "formal_charge": Chem.GetFormalCharge(mol),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DataFrame conveniences
# ---------------------------------------------------------------------------

def add_standard_columns(
    df: pd.DataFrame, smi_col: str = "smiles", drop_failed: bool = False
) -> pd.DataFrame:
    """Add std_smiles, inchikey, scaffold, and physchem columns. Idempotent."""
    out = df.copy()
    mols = out[smi_col].map(standardize)
    out["std_smiles"] = mols.map(lambda m: Chem.MolToSmiles(m) if m is not None else None)
    out["inchikey"] = mols.map(lambda m: Chem.MolToInchiKey(m) if m is not None else None)
    out["scaffold"] = mols.map(lambda m: bemis_murcko(m))
    physchem = mols.map(lambda m: compute_physchem(m) if m is not None else None)
    physchem_df = pd.json_normalize(physchem.where(physchem.notna(), {})).set_index(out.index)
    out = pd.concat([out, physchem_df], axis=1)
    if drop_failed:
        out = out.dropna(subset=["std_smiles"]).reset_index(drop=True)
    return out
