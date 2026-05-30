"""nb212 -- Blend nb211 (QReg div15+Chemprop) with nb94 fine-tuned ChemBERTa.

Requires:
  data/processed/oof_nb94_chemberta_mtr.npy   (from Kaggle nb94 run)
  data/processed/te_nb94_chemberta_mtr.npy
  data/processed/oof_nb211_div15_chemprop_blend.npy
  data/processed/te_nb211_div15_chemprop_blend.npy

Analysis:
  - Compute OOF RAE for ChemBERTa alone
  - Compute error correlation with nb211
  - Grid search over blend weight w: blend = (1-w)*nb211 + w*ChemBERTa
  - Save best if OOF RAE < 0.296886 AND test std ratio >= 0.58
"""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

PREV_BEST = 0.296886
COLLAPSE_THRESH = 0.58

tr = load_train()
te = load_test()
y = tr["pec50"].values.astype(np.float64)

# Load nb211 (current best)
oof211 = np.load(DATA_PROCESSED / "oof_nb211_div15_chemprop_blend.npy").flatten()
te211  = np.load(DATA_PROCESSED / "te_nb211_div15_chemprop_blend.npy").flatten()
print(f"nb211: OOF RAE={rae(y, oof211):.6f}  std={oof211.std():.4f}  te_std={te211.std():.4f}")

# Load ChemBERTa (nb94)
bert_oof_path = DATA_PROCESSED / "oof_nb94_chemberta_mtr.npy"
bert_te_path  = DATA_PROCESSED / "te_nb94_chemberta_mtr.npy"

if not bert_oof_path.exists():
    print(f"ERROR: {bert_oof_path} not found. Run Kaggle nb94 first.", flush=True)
    sys.exit(1)

oof94 = np.load(bert_oof_path).flatten().astype(np.float64)
te94  = np.load(bert_te_path).flatten().astype(np.float64)

oof94_rae = rae(y, oof94)
err211 = oof211 - y
err94  = oof94 - y
err_corr = np.corrcoef(err211, err94)[0, 1]

print(f"nb94 ChemBERTa: OOF RAE={oof94_rae:.6f}  std={oof94.std():.4f}  te_std={te94.std():.4f}")
print(f"Error correlation nb211 vs nb94: {err_corr:.4f}")
print(f"Needed: OOF RAE < 0.65, err_corr < 0.53 for blend to help")
print()

if oof94_rae > 0.80:
    print(f"ChemBERTa OOF RAE {oof94_rae:.4f} too high — blend won't help")
    sys.exit(0)

# Grid search blend weight
print("--- Blend grid: w=ChemBERTa weight ---")
best_rae = PREV_BEST
best_w = None
best_oof = None
best_te = None

for w in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050,
          0.075, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend_oof = (1 - w) * oof211 + w * oof94
    blend_te  = (1 - w) * te211  + w * te94
    r = rae(y, blend_oof)
    ratio = blend_te.std() / blend_oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    note = ""
    if r < best_rae:
        note = f" *NEW BEST {r:.6f}*"
        best_rae = r
        best_w = w
        best_oof = blend_oof.copy()
        best_te = blend_te.copy()
    print(f"  w={w:.3f}  OOF RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{note}")

print()
if best_oof is not None:
    ratio = best_te.std() / best_oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"Best: w={best_w:.3f}  OOF RAE={best_rae:.6f}  ratio={ratio:.4f}  [{flag}]")

    if ratio >= COLLAPSE_THRESH:
        out_stem = "nb212_chemberta_blend"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", best_oof)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy", best_te)
        sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        print(f"Saved {SUBMISSIONS/f'{out_stem}.csv'}")
        print(f"  OOF RAE improvement: {PREV_BEST:.6f} → {best_rae:.6f} (delta={PREV_BEST-best_rae:.6f})")
    else:
        print(f"  ratio={ratio:.4f} < {COLLAPSE_THRESH} — test predictions collapsed, skipping")
else:
    print(f"No blend beats PREV_BEST={PREV_BEST:.6f}")
