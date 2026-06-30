"""
nb2098 — F2 cohort stratified validation across 253 unblind
============================================================
Slice the 253 by scaffold-novelty tier and report which candidate wins per tier:
  TIER A: scaf_train_freq >= 5   (well-supported)
  TIER B: 1 <= scaf_train_freq < 5 (sparse)
  TIER C: scaf_train_freq == 0   (novel, F2 cohort — the -0.11 RAE prize zone)

Candidates evaluated:
  - nb1191 (LB-SAFE deep-verified)
  - nb2060 (HOLD_DEEP_FAIL)
  - nb1162 (POST-unblind risk)
  - chemprop_aux (PRE-unblind anchor)

Output: data/processed/nb2098_summary.json
"""
import numpy as np, json, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

def rae(y, p):
    if len(y) < 3:
        return None
    return float(np.mean(np.abs(y - p)) / np.mean(np.abs(y - np.mean(y))))

def scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return ""

unb_idx = np.load(PROC / "_audit_unblind_idx.npy")
y_unb = np.load(PROC / "_audit_unblind_y.npy")

# Load train + test SMILES
train = pd.read_csv(ROOT / "data" / "raw" / "pxr-challenge_TRAIN.csv")
test = pd.read_csv(ROOT / "data" / "raw" / "pxr-challenge_TEST_BLINDED.csv")

train_scaf = train["SMILES"].apply(scaffold)
test_scaf = test["SMILES"].apply(scaffold).reset_index(drop=True)
scaf_freq = train_scaf.value_counts().to_dict()

unb_scaf = test_scaf.iloc[unb_idx].values
unb_freq = np.array([scaf_freq.get(s, 0) for s in unb_scaf])

tiers = {
    "A_well_supported": unb_freq >= 5,
    "B_sparse": (unb_freq >= 1) & (unb_freq < 5),
    "C_novel_F2": unb_freq == 0,
}
print("Tier sizes:")
for t, mask in tiers.items():
    print(f"  {t}: n={mask.sum()}")

candidates = {
    "nb1191": np.load(PROC / "te_nb1191.npy")[unb_idx],
    "nb2060": np.load(PROC / "te_nb2060.npy")[unb_idx],
    "nb1162": np.load(PROC / "te_nb1162.npy")[unb_idx],
    "chemprop_aux": np.load(PROC / "te_chemprop_aux.npy")[unb_idx],
}

# Per-tier RAE table
results = {}
for t, mask in tiers.items():
    if mask.sum() < 3:
        continue
    row = {"n": int(mask.sum())}
    for k, p in candidates.items():
        r = rae(y_unb[mask], p[mask])
        row[k] = round(r, 4) if r is not None else None
    # winner = min RAE
    winner = min((k for k in candidates), key=lambda k: row[k] if row[k] is not None else 1e9)
    row["winner"] = winner
    row["winner_rae"] = row[winner]
    results[t] = row

# Overall
overall = {"n": 253}
for k, p in candidates.items():
    overall[k] = round(rae(y_unb, p), 4)
overall["winner"] = min((k for k in candidates), key=lambda k: overall[k])

print("\nPer-tier RAE (in-sample on 253):")
for t, row in results.items():
    print(f"  {t}: n={row['n']:>3}  winner={row['winner']:14}  rae={row['winner_rae']:.4f}")
print(f"  overall: n={overall['n']}  winner={overall['winner']}  rae={overall[overall['winner']]:.4f}")

# F2-prescription: which candidate is best on novel-scaffold tier?
f2_winner = results.get("C_novel_F2", {}).get("winner")
f2_rae = results.get("C_novel_F2", {}).get("winner_rae")
print(f"\nF2 PRESCRIPTION: novel-scaffold winner = {f2_winner} (RAE {f2_rae})")

summary = {
    "method": "nb2098_f2_cohort_stratified_validation",
    "tier_sizes": {t: int(m.sum()) for t, m in tiers.items()},
    "per_tier_rae": results,
    "overall": overall,
    "f2_prescription": {
        "winner_on_novel_scaffold": f2_winner,
        "winner_rae": f2_rae,
        "note": "Use this candidate as F2 router specialist; blend with overall winner on well-supported"
    },
}
with open(PROC / "nb2098_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\n" + json.dumps(summary, indent=2, default=str))
