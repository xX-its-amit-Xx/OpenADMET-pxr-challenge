"""Adversarial review: 253 unblind vs 260 still-blind."""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

RAW = "d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/raw"
SUB = "d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/submissions"

unblind = pd.read_csv(f"{RAW}/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
test = pd.read_csv(f"{RAW}/pxr-challenge_TEST_BLINDED.csv")
train = pd.read_csv(f"{RAW}/pxr-challenge_TRAIN.csv")
nb320 = pd.read_csv(f"{SUB}/nb320_phase2_top50_slsqp.csv")

print(f"unblind rows: {len(unblind)}")
print(f"test rows: {len(test)}")
print(f"nb320 rows: {len(nb320)}")

unblind_ids = set(unblind["Molecule Name"])
test_ids = set(test["Molecule Name"])
still_blind_ids = test_ids - unblind_ids
print(f"still_blind: {len(still_blind_ids)}")

unblind_smi = dict(zip(unblind["Molecule Name"], unblind["SMILES"]))
test_smi = dict(zip(test["Molecule Name"], test["SMILES"]))


def mol(s):
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def scaffold(s):
    m = mol(s)
    if m is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc)
    except Exception:
        return None


def fp(s):
    m = mol(s)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def feats(s):
    m = mol(s)
    if m is None:
        return None, None
    return Descriptors.MolWt(m), Descriptors.MolLogP(m)


# Train fingerprints
train_fps = [fp(s) for s in train["SMILES"]]
train_fps = [f for f in train_fps if f is not None]


def top1(s):
    f = fp(s)
    if f is None:
        return np.nan
    sims = DataStructs.BulkTanimotoSimilarity(f, train_fps)
    return max(sims)


# Build per-group stats
groups = {
    "unblind_253": [(mid, unblind_smi[mid]) for mid in unblind_ids],
    "still_blind_260": [(mid, test_smi[mid]) for mid in still_blind_ids],
}

for name, items in groups.items():
    smis = [s for _, s in items]
    mws, logps = zip(*[feats(s) for s in smis])
    mws = [m for m in mws if m is not None]
    logps = [l for l in logps if l is not None]
    scs = set(scaffold(s) for s in smis) - {None}
    top1s = [top1(s) for s in smis]
    print(f"\n--- {name} (n={len(smis)}) ---")
    print(f"  unique scaffolds: {len(scs)}")
    print(f"  MW mean/std: {np.mean(mws):.1f} / {np.std(mws):.1f}")
    print(f"  logP mean/std: {np.mean(logps):.2f} / {np.std(logps):.2f}")
    print(f"  top1 Tanimoto-to-train median: {np.nanmedian(top1s):.3f}")
    print(f"  top1 Tanimoto mean: {np.nanmean(top1s):.3f}")
    print(f"  top1 < 0.35 frac: {np.mean(np.array(top1s) < 0.35):.3f}")

# Scaffold overlap
sc_un = set(scaffold(unblind_smi[m]) for m in unblind_ids) - {None}
sc_sb = set(scaffold(test_smi[m]) for m in still_blind_ids) - {None}
print(f"\nScaffold overlap unblind vs still_blind: {len(sc_un & sc_sb)} / {len(sc_sb)} still-blind scaffolds also in unblind")
print(f"Scaffolds unique to still_blind: {len(sc_sb - sc_un)}")

# nb320 prediction ranges
nb320_map = dict(zip(nb320["Molecule Name"], nb320["pEC50"]))
preds_unblind = np.array([nb320_map[m] for m in unblind_ids if m in nb320_map])
preds_blind = np.array([nb320_map[m] for m in still_blind_ids if m in nb320_map])

print(f"\nnb320 preds on unblind (n={len(preds_unblind)}): min={preds_unblind.min():.3f} max={preds_unblind.max():.3f}")
print(f"nb320 preds on still_blind (n={len(preds_blind)}): min={preds_blind.min():.3f} max={preds_blind.max():.3f}")

# fraction of still-blind preds outside the [min, max] of unblind preds
lo, hi = preds_unblind.min(), preds_unblind.max()
out_frac = np.mean((preds_blind < lo) | (preds_blind > hi))
print(f"Still-blind preds OUTSIDE unblind [min,max]: {out_frac:.4f} ({int(out_frac*len(preds_blind))} of {len(preds_blind)})")

# more lenient: outside [p1, p99] of unblind preds
p1, p99 = np.percentile(preds_unblind, [1, 99])
out_frac_99 = np.mean((preds_blind < p1) | (preds_blind > p99))
print(f"Still-blind preds OUTSIDE unblind [p1,p99]: {out_frac_99:.4f}")
p5, p95 = np.percentile(preds_unblind, [5, 95])
out_frac_95 = np.mean((preds_blind < p5) | (preds_blind > p95))
print(f"Still-blind preds OUTSIDE unblind [p5,p95]: {out_frac_95:.4f}")

# pEC50 label distribution on unblind to check sampling bias
print(f"\nunblind pEC50 dist: mean={unblind['pEC50'].mean():.3f} std={unblind['pEC50'].std():.3f} median={unblind['pEC50'].median():.3f}")
print(f"train pEC50 dist: mean={train['pEC50'].mean():.3f} std={train['pEC50'].std():.3f}")
print(f"unblind pEC50 ranges: min={unblind['pEC50'].min():.3f} max={unblind['pEC50'].max():.3f}")
print(f"unblind hit rate (pEC50>=6): {(unblind['pEC50']>=6).mean():.4f} vs train {(train['pEC50']>=6).mean():.4f}")
