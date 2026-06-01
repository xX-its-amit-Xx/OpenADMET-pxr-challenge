"""nb422: Per-compound chemprop-family disagreement / uncertainty.

Loads 4 chemprop-family test predictors:
  - nb93_chemprop_large_gpu
  - nb130_external_pxr
  - nb264_chemprop_mt
  - chemprop_aux

For each of the 513 test compounds, compute std and range across the 4 predictors
(chemprop-family disagreement). Compare distribution on 253 unblind vs 260 still-blind.
Check correlation between disagreement and |y - mean_4_chemprop| on the unblind set.
Identify the 30%-tail threshold (top 30% disagreement = HIGH uncertainty target).
Save data/processed/nb422_compound_uncertainty.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROC = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/processed")
RAW = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/raw")

# 1) Load test ordering (513 compounds, deterministic order matching te_*.npy arrays)
test = pd.read_csv(RAW / "pxr-challenge_TEST_BLINDED.csv")
assert len(test) == 513, len(test)
names = test["Molecule Name"].to_numpy()

# 2) Load 4 chemprop-family predictors
preds = {
    "nb93_chemprop_large_gpu": np.load(PROC / "te_nb93_chemprop_large_gpu.npy"),
    "nb130_external_pxr":     np.load(PROC / "te_nb130_external_pxr.npy"),
    "nb264_chemprop_mt":      np.load(PROC / "te_nb264_chemprop_mt.npy"),
    "chemprop_aux":           np.load(PROC / "te_chemprop_aux.npy"),
}
for k, v in preds.items():
    assert v.shape == (513,), (k, v.shape)

P = np.column_stack([preds[k] for k in preds])  # (513, 4)
disagreement_std = P.std(axis=1, ddof=0)
disagreement_range = P.max(axis=1) - P.min(axis=1)
mean4 = P.mean(axis=1)

# 3) Phase-1 unblind labels
unblind = pd.read_csv(RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
unblind_lookup = dict(zip(unblind["Molecule Name"], unblind["pEC50"]))
y_true = np.array([unblind_lookup.get(n, np.nan) for n in names], dtype=float)
in_unblind = ~np.isnan(y_true)
n_unblind = int(in_unblind.sum())
n_blind = int((~in_unblind).sum())
print(f"unblind={n_unblind}  still_blind={n_blind}")

# 4) Histograms / summary
def summary(arr, mask, label):
    a = arr[mask]
    print(f"  {label:14s} n={len(a):3d}  median={np.median(a):.4f}  p90={np.quantile(a, 0.90):.4f}  mean={a.mean():.4f}  max={a.max():.4f}")

print("\n--- chemprop-family disagreement (std across 4 predictors) ---")
summary(disagreement_std, in_unblind, "unblind(253)")
summary(disagreement_std, ~in_unblind, "still_blind(260)")
summary(disagreement_std, np.ones(513, dtype=bool), "all(513)")

print("\n--- chemprop-family disagreement (range across 4 predictors) ---")
summary(disagreement_range, in_unblind, "unblind(253)")
summary(disagreement_range, ~in_unblind, "still_blind(260)")

# 5) Correlation: disagreement vs |y - mean_4_chemprop| on unblind
err_unblind = np.abs(y_true[in_unblind] - mean4[in_unblind])
dis_unblind = disagreement_std[in_unblind]
pearson = float(np.corrcoef(dis_unblind, err_unblind)[0, 1])
# spearman (rank-based)
from scipy.stats import spearmanr
rho, p = spearmanr(dis_unblind, err_unblind)
print(f"\nCorrelation disagreement_std vs |y - mean_4_chemprop| (n={n_unblind}):")
print(f"  pearson  r = {pearson:.4f}")
print(f"  spearman r = {float(rho):.4f}  (p={float(p):.3g})")

# 6) Identify top-30% disagreement threshold (compute on ALL 513 so blind compounds get tagged consistently)
tau_hi = float(np.quantile(disagreement_std, 0.70))   # 30% above this is HIGH
tau_lo = float(np.quantile(disagreement_std, 0.30))   # bottom 30% is LOW (for symmetry / future use)
is_high = disagreement_std >= tau_hi
print(f"\nThreshold (top 30% of disagreement_std across all 513): tau_hi={tau_hi:.4f}")
print(f"                                                  tau_lo={tau_lo:.4f}")
print(f"  n_high (all)         = {int(is_high.sum())}  ({is_high.mean()*100:.1f}%)")
print(f"  n_high in unblind    = {int((is_high & in_unblind).sum())}")
print(f"  n_high in still_blind= {int((is_high & ~in_unblind).sum())}")

# error on high vs low-disagreement unblind (sanity)
hi_mask_u = is_high & in_unblind
lo_mask_u = (~is_high) & in_unblind
err_hi = np.abs(y_true[hi_mask_u] - mean4[hi_mask_u]).mean() if hi_mask_u.any() else np.nan
err_lo = np.abs(y_true[lo_mask_u] - mean4[lo_mask_u]).mean() if lo_mask_u.any() else np.nan
print(f"  unblind |err| HIGH-dis  = {err_hi:.4f} (n={int(hi_mask_u.sum())})")
print(f"  unblind |err| not-HIGH  = {err_lo:.4f} (n={int(lo_mask_u.sum())})")

# 7) Save CSV
out = pd.DataFrame({
    "compound_name": names,
    "smiles": test["SMILES"].to_numpy(),
    "pred_nb93_chemprop_large_gpu": preds["nb93_chemprop_large_gpu"],
    "pred_nb130_external_pxr":     preds["nb130_external_pxr"],
    "pred_nb264_chemprop_mt":      preds["nb264_chemprop_mt"],
    "pred_chemprop_aux":           preds["chemprop_aux"],
    "mean_4_chemprop":             mean4,
    "chemprop_disagreement_std":   disagreement_std,
    "chemprop_disagreement_range": disagreement_range,
    "is_high_uncertainty":         is_high.astype(int),
    "in_unblind_or_blind":         np.where(in_unblind, "unblind", "blind"),
    "y_unblind":                   y_true,  # NaN for still-blind
    "abs_err_vs_mean4":            np.where(in_unblind, np.abs(y_true - mean4), np.nan),
})
out_path = PROC / "nb422_compound_uncertainty.csv"
out.to_csv(out_path, index=False)
print(f"\nWrote {out_path}  shape={out.shape}")

# 8) Compact JSON-ish summary for the parent agent
import json
summary_obj = {
    "n_unblind": n_unblind,
    "n_blind": n_blind,
    "median_disagreement_unblind": float(np.median(disagreement_std[in_unblind])),
    "p90_disagreement_unblind":    float(np.quantile(disagreement_std[in_unblind], 0.9)),
    "median_disagreement_blind":   float(np.median(disagreement_std[~in_unblind])),
    "p90_disagreement_blind":      float(np.quantile(disagreement_std[~in_unblind], 0.9)),
    "pearson_disagreement_vs_error":  pearson,
    "spearman_disagreement_vs_error": float(rho),
    "tau_hi_top30":   tau_hi,
    "tau_lo_bot30":   tau_lo,
    "n_high_total":           int(is_high.sum()),
    "n_high_in_unblind":      int((is_high & in_unblind).sum()),
    "n_high_in_still_blind":  int((is_high & ~in_unblind).sum()),
    "mean_abs_err_high_unblind": float(err_hi),
    "mean_abs_err_notHigh_unblind": float(err_lo),
}
print("\nSUMMARY_JSON:\n" + json.dumps(summary_obj, indent=2))
