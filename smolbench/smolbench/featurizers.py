"""Featurizer registry — drop-in molecular fingerprints/descriptors from SMILES.

Every featurizer is a callable ``list[str] -> np.ndarray (n, d)``. All are RDKit-based
(no network, no GPU) except the optional pretrained ones, which degrade gracefully.

Register your own with ``@register_featurizer("name")``.
"""
from __future__ import annotations
import warnings
import numpy as np
from functools import lru_cache

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit.Chem import rdReducedGraphs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

FEATURIZERS: dict[str, callable] = {}


def register_featurizer(name):
    def deco(fn):
        FEATURIZERS[name] = fn
        return fn
    return deco


def _mols(smiles):
    return [Chem.MolFromSmiles(str(s)) for s in smiles]


def _stack(vectors, dim):
    out = np.zeros((len(vectors), dim), dtype=np.float32)
    for i, v in enumerate(vectors):
        if v is not None:
            DataStructs.ConvertToNumpyArray(v, out[i])
    return out


@register_featurizer("morgan")
def morgan(smiles, radius=2, n_bits=2048):
    """ECFP-like circular fingerprint (radius 2 = ECFP4)."""
    vecs = [AllChem.GetMorganFingerprintAsBitVect(m, radius, n_bits) if m else None for m in _mols(smiles)]
    return _stack(vecs, n_bits)


@register_featurizer("morgan_counts")
def morgan_counts(smiles, radius=2, n_bits=2048):
    mols = _mols(smiles)
    out = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        fp = AllChem.GetHashedMorganFingerprint(m, radius, n_bits)
        for idx, c in fp.GetNonzeroElements().items():
            out[i, idx] = c
    return out


@register_featurizer("maccs")
def maccs(smiles):
    """166-bit MACCS structural keys."""
    vecs = [MACCSkeys.GenMACCSKeys(m) if m else None for m in _mols(smiles)]
    return _stack(vecs, 167)


@register_featurizer("avalon")
def avalon(smiles, n_bits=1024):
    try:
        from rdkit.Avalon import pyAvalonTools
    except Exception:
        warnings.warn("Avalon unavailable; skipping"); return np.zeros((len(smiles), n_bits), np.float32)
    vecs = [pyAvalonTools.GetAvalonFP(m, nBits=n_bits) if m else None for m in _mols(smiles)]
    return _stack(vecs, n_bits)


@register_featurizer("atompair")
def atompair(smiles, n_bits=2048):
    vecs = [rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=n_bits) if m else None for m in _mols(smiles)]
    return _stack(vecs, n_bits)


@register_featurizer("torsion")
def torsion(smiles, n_bits=2048):
    vecs = [rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=n_bits) if m else None for m in _mols(smiles)]
    return _stack(vecs, n_bits)


@register_featurizer("erg")
def erg(smiles):
    """Extended reduced graph — pharmacophore-flavored, low-dim, often activity-aligned."""
    out = []
    for m in _mols(smiles):
        try:
            out.append(np.asarray(rdReducedGraphs.GetErGFingerprint(m)) if m else None)
        except Exception:
            out.append(None)
    dim = next((len(v) for v in out if v is not None), 315)
    X = np.zeros((len(out), dim), np.float32)
    for i, v in enumerate(out):
        if v is not None and len(v) == dim:
            X[i] = v
    return X


_DESC = [n for n, _ in Descriptors._descList]


@register_featurizer("rdkit_desc")
def rdkit_desc(smiles):
    """~210 physicochemical/topological descriptors (interpretable)."""
    from rdkit.ML.Descriptors import MoleculeDescriptors
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC)
    out = np.full((len(smiles), len(_DESC)), np.nan, np.float32)
    for i, m in enumerate(_mols(smiles)):
        if m is None:
            continue
        try:
            out[i] = calc.CalcDescriptors(m)
        except Exception:
            pass
    return out


def featurize(smiles, name, **kwargs):
    """Featurize a SMILES list by registered name. Returns (n, d) float array."""
    if name not in FEATURIZERS:
        raise KeyError(f"Unknown featurizer '{name}'. Available: {sorted(FEATURIZERS)}")
    X = FEATURIZERS[name](list(smiles), **kwargs)
    # impute non-finite (descriptor blowups) with column medians
    if not np.isfinite(X).all():
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        bad = ~np.isfinite(X)
        X[bad] = np.take(med, np.where(bad)[1])
    return X


def available_featurizers():
    return sorted(FEATURIZERS)
