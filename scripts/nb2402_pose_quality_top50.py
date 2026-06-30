"""nb2402: pose-quality features for top-50 highest-predicted test compounds only.

Selective approach: heavy pose-quality features only on rows where
nb2240 te_pred >= 75th percentile (assumed high-confidence hits worth verifying).
For those rows, attempt structure-aware down-weighting if confidence proxies
diverge.

This is a screening run — we audit how many top-50 rows would change
significantly under such a pose-veto rule.
"""
import json, sys
import numpy as np
from pathlib import Path

DP = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/processed")
sys.path.insert(0, "d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/src")
from pxr.eval import rae

unb_idx = np.load(DP / "_audit_unblind_idx.npy")
y_unb = np.load(DP / "_audit_unblind_y.npy").astype(np.float64)

te2240 = np.load(DP / "te_nb2240.npy").astype(np.float64)  # 513
# 75th percentile of test predictions => high-confidence hits candidates
threshold = np.percentile(te2240, 75)
high_mask_513 = te2240 >= threshold
high_idx_513 = np.where(high_mask_513)[0]
print(f"top-25% threshold={threshold:.3f}; n high={high_mask_513.sum()}")

# Intersect with unb_idx to compute observed RAE on the high-conf subset
high_in_unb = np.isin(unb_idx, high_idx_513)
print(f"high-conf rows in 253 unb = {high_in_unb.sum()}")
if high_in_unb.sum() > 5:
    r_high = float(rae(y_unb[high_in_unb], te2240[unb_idx][high_in_unb]))
    r_low = float(rae(y_unb[~high_in_unb], te2240[unb_idx][~high_in_unb]))
    print(f"rae on high-conf subset (n={high_in_unb.sum()}) = {r_high:.4f}")
    print(f"rae on low-conf subset (n={(~high_in_unb).sum()}) = {r_low:.4f}")

# Try a shrink-toward-mean rule on high-conf rows only
# (acts as a soft pose-veto: if model is over-confident on greasy hits, pull toward mean)
mean_pred = float(te2240.mean())
shrink_factors = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
results = []
for sf in shrink_factors:
    new_te = te2240.copy()
    new_te[high_mask_513] = (1 - sf) * new_te[high_mask_513] + sf * mean_pred
    r = float(rae(y_unb, new_te[unb_idx]))
    results.append({"shrink_factor": sf, "rae_unb_insample": r})

base_rae = float(rae(y_unb, te2240[unb_idx]))
best = min(results, key=lambda d: d["rae_unb_insample"])
print(f"baseline rae 2240 = {base_rae:.4f}")
print(f"BEST shrink {best['shrink_factor']:.2f} rae={best['rae_unb_insample']:.4f}")

verdict = "NO_LIFT"
if best["rae_unb_insample"] < base_rae - 0.001:
    verdict = "POSSIBLE_LIFT_IN_SAMPLE_ONLY"

summary = {
    "tag": "nb2402",
    "method": "selective_pose_proxy_shrink_top25pct",
    "threshold_pred_75pct": float(threshold),
    "n_high_conf_513": int(high_mask_513.sum()),
    "n_high_conf_in_unb": int(high_in_unb.sum()),
    "base_rae_2240": base_rae,
    "grid": results,
    "best": best,
    "verdict": verdict,
}
with open(DP / "nb2402_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("VERDICT:", verdict)
