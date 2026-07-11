"""Cross-validation split strategies for molecular data.

Scaffold and cluster splits give HONEST estimates on analog-expansion / novel-chemistry
test sets — random CV is typically ~0.1 RAE optimistic. Scaffold is the default.
"""
from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def _scaffold(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return ""


def scaffold_folds(smiles, n_folds=5, seed=0):
    """Bemis-Murcko scaffold K-fold: no scaffold spans train+val."""
    scaf = np.array([_scaffold(s) for s in smiles])
    uniq = list(dict.fromkeys(scaf))
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    buckets = {s: i % n_folds for i, s in enumerate(uniq)}
    fold_id = np.array([buckets[s] for s in scaf])
    return [(np.where(fold_id != k)[0], np.where(fold_id == k)[0]) for k in range(n_folds)]


def random_folds(n, n_folds=5, seed=0):
    idx = np.arange(n); np.random.default_rng(seed).shuffle(idx)
    parts = np.array_split(idx, n_folds)
    return [(np.setdiff1d(idx, p), p) for p in parts]


def butina_folds(smiles, n_folds=5, cutoff=0.6, seed=0):
    """Cluster by Butina (Tanimoto) then assign whole clusters to folds."""
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(str(s)), 2, 2048) for s in smiles]
    n = len(fps); dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    order = list(range(len(clusters))); np.random.default_rng(seed).shuffle(order)
    fold_id = np.zeros(n, int)
    for rank, ci in enumerate(order):
        for idx in clusters[ci]:
            fold_id[idx] = rank % n_folds
    return [(np.where(fold_id != k)[0], np.where(fold_id == k)[0]) for k in range(n_folds)]


def make_folds(smiles, strategy="scaffold", n_folds=5, seed=0):
    if strategy == "scaffold":
        return scaffold_folds(smiles, n_folds, seed)
    if strategy == "random":
        return random_folds(len(smiles), n_folds, seed)
    if strategy in ("butina", "cluster"):
        return butina_folds(smiles, n_folds, seed=seed)
    raise ValueError(f"Unknown CV strategy '{strategy}' (scaffold|random|butina)")
