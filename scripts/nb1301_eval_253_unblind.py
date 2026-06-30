"""nb1301 — Evaluate current best model on 253 unblinded test compounds.

Now that Phase-1 unblind labels are public, we can measure TRUE blinded performance.
Establishes the honest gate for all subsequent experiments:
  - Current model RAE on 253 unblinded → real blinded error estimate
  - Per-compound analysis: which compounds are we worst on? (cliffs? low-activity?)
  - Saves unblind_eval.json + unblind_preds.csv for downstream use

Strategy: 253 unblinded stay OUT of training until we finalize the model.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.chem import morgan_fp_batch

SD   = "C:/pxr_work/search"
UBD  = "C:/pxr_work/phase1_unblind"
OUT  = "C:/pxr_work/phase1_unblind"
BEST = f"{SD}/best_ensemble.json"

# ── Load 253 unblinded labels from raw HF CSV (avoid the std_smiles Mol-object bug)
raw = pd.read_csv(f"{UBD}/phase1_unblinded_raw.csv")
print(f"Raw unblind: {len(raw)} rows, cols={list(raw.columns)}")

# Column names from HF dataset vary; find name and pec50 columns
name_col = [c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower()][0]
pec_col  = [c for c in raw.columns if "pec50" in c.lower() or "activity" in c.lower()][0]
print(f"  Using name_col='{name_col}', pec_col='{pec_col}'")
raw = raw[[name_col, pec_col]].dropna()
raw.columns = ["name", "pec50_true"]
print(f"  {len(raw)} valid unblinded labels, pEC50 range [{raw.pec50_true.min():.2f}, {raw.pec50_true.max():.2f}]")

# ── Load test set (513 compounds), find the 253 unblinded indices
te = load_test().reset_index(drop=True)
te["idx"] = te.index
unblind_mask = te["name"].isin(set(raw["name"]))
unblind_idx  = te.index[unblind_mask].tolist()
blind_idx    = te.index[~unblind_mask].tolist()
print(f"Matched {unblind_mask.sum()} / 253 unblinded in test set ({len(blind_idx)} remain blinded)")

# Merge true labels onto test set order
te_ub = te[unblind_mask].merge(raw, on="name", how="left")
y_true = te_ub["pec50_true"].to_numpy()

# ── Load cached predictions for all 513 test compounds
def load_te(name):
    path = f"{SD}/{name}_te.npy"
    if not os.path.exists(path): return None
    v = np.load(path, allow_pickle=True)
    return v.ravel() if v.ndim > 1 else v

# Each *_te.npy has 513 predictions (one per test compound)
members = {
    "gnn":          load_te("gnn"),
    "tabnet":       load_te("tabnet"),
    "chemeleon_lgbm": load_te("chemeleon_lgbm"),
    "tabpfn":       load_te("tabpfn"),
    "tabicl":       load_te("tabicl"),
}
members = {k: v for k, v in members.items() if v is not None and len(v) == 513}
print(f"Loaded {len(members)} prediction arrays: {list(members.keys())}")

# Load the GBM ensemble predictions — need to reconstruct them
# Best ensemble from best_ensemble.json is a GBM stack; reconstruct from OOF + te
# For a quick eval, use the saved submission CSV (nb1299)
best_cfg = json.load(open(BEST))
sub_path = f"D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/{best_cfg['submission']}"
sub = pd.read_csv(sub_path)
print(f"Loaded submission: {sub_path} ({len(sub)} rows)")

te2 = load_test().reset_index(drop=True)
sub_merged = te2[["name"]].merge(sub, left_on="name", right_on="Molecule Name", how="left")
pred_all = sub_merged["pEC50"].to_numpy()

# Extract predictions for the 253 unblinded
pred_ub = pred_all[unblind_idx]
y_mean  = float(np.median(np.concatenate([
    load_train().dropna(subset=["pec50"])["pec50"].to_numpy(), y_true
])))

def rae(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d > 0 else float("nan")

rae_ub  = rae(y_true, pred_ub)
mae_ub  = float(np.abs(y_true - pred_ub).mean())
corr_ub = float(np.corrcoef(y_true, pred_ub)[0, 1])
bias_ub = float((pred_ub - y_true).mean())

print(f"\n=== 253-UNBLINDED HONEST EVALUATION ===")
print(f"RAE  = {rae_ub:.4f}  (internal best claimed {best_cfg['rae']:.4f})")
print(f"MAE  = {mae_ub:.4f}")
print(f"corr = {corr_ub:.4f}")
print(f"bias = {bias_ub:+.4f}")

# Per-compound analysis — worst 20
residuals = pred_ub - y_true
te_ub = te_ub.copy()
te_ub["pred"]  = pred_ub
te_ub["resid"] = residuals
te_ub["abs_e"] = np.abs(residuals)

print(f"\nWorst 10 unblinded compounds:")
worst = te_ub.nlargest(10, "abs_e")[["name", "pec50_true", "pred", "resid"]]
print(worst.to_string(index=False))

# Coverage: similarity of 253 unblinded to training set (Tanimoto)
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
ub_smi  = te_ub["smiles"].tolist()
tr_smi  = tr["smiles"].tolist()
try:
    fp_ub = morgan_fp_batch(ub_smi).astype(np.float32)
    fp_tr = morgan_fp_batch(tr_smi[:2000]).astype(np.float32)
    inter = fp_ub @ fp_tr.T
    union = fp_ub.sum(1)[:, None] + fp_tr.sum(1)[None, :] - inter
    tani  = (inter / np.maximum(union, 1)).max(1)
    med_tani = float(np.median(tani))
    frac_low = float((tani < 0.4).mean())
    print(f"\nCoverage (253 unblinded vs train): median Tanimoto {med_tani:.3f}, "
          f"{frac_low*100:.1f}% < 0.4 (low coverage)")
except Exception as e:
    med_tani = frac_low = None
    print(f"Coverage calc failed: {e}")

# Save results
result = {
    "n_unblinded": len(y_true),
    "n_blind": len(blind_idx),
    "rae_on_253":  round(rae_ub, 4),
    "mae_on_253":  round(mae_ub, 4),
    "corr_on_253": round(corr_ub, 4),
    "bias_on_253": round(bias_ub, 4),
    "internal_best_rae": best_cfg["rae"],
    "med_tanimoto_to_train": round(med_tani, 3) if med_tani else None,
    "frac_low_coverage": round(frac_low, 3) if frac_low else None,
    "submission": best_cfg["submission"],
}
json.dump(result, open(f"{OUT}/unblind_eval.json", "w"), indent=2)

# Save per-compound predictions for downstream gating
out_df = te_ub[["name", "smiles", "pec50_true", "pred", "resid", "abs_e"]].copy()
out_df.to_csv(f"{OUT}/unblind_preds.csv", index=False)

# Save unblinded test indices for other scripts
np.save(f"{UBD}/unblind_te_idx.npy", np.array(unblind_idx, dtype=np.int64))
np.save(f"{UBD}/blind_te_idx.npy",   np.array(blind_idx, dtype=np.int64))
np.save(f"{UBD}/unblind_y_true.npy", y_true.astype(np.float64))

print(f"\nSaved → {OUT}/unblind_eval.json")
print(json.dumps(result, indent=2))
