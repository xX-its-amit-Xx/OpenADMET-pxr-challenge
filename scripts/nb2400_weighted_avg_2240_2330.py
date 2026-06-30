"""nb2400: weighted-average grid search between nb2240 and nb2330.

Different from SLSQP because: convex grid on a single scalar w,
no per-fold re-weight, no stretch, simple linear interpolation.
"""
import json, sys
import numpy as np
from pathlib import Path

DP = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/processed")
sys.path.insert(0, "d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/src")
from pxr.eval import rae

unb_idx = np.load(DP / "_audit_unblind_idx.npy")
y_unb = np.load(DP / "_audit_unblind_y.npy").astype(np.float64)

te2240 = np.load(DP / "te_nb2240.npy").astype(np.float64)
te2330 = np.load(DP / "te_nb2330.npy").astype(np.float64)

# IN-SAMPLE on 253 unb (te is deploy refit; rae lower bound)
rae_2240 = rae(y_unb, te2240[unb_idx])
rae_2330 = rae(y_unb, te2330[unb_idx])
print(f"in-sample rae nb2240 = {rae_2240:.4f}")
print(f"in-sample rae nb2330 = {rae_2330:.4f}")

ws = np.linspace(0, 1, 21)
results = []
for w in ws:
    blend = w * te2240[unb_idx] + (1 - w) * te2330[unb_idx]
    r = float(rae(y_unb, blend))
    results.append({"w_2240": float(w), "rae_unb_insample": r})

best = min(results, key=lambda d: d["rae_unb_insample"])
print(f"BEST IN-SAMPLE: w_2240={best['w_2240']:.2f} rae={best['rae_unb_insample']:.4f}")

# Honest OOF cross-fit on 253: load oof anchors
oof2240 = np.load(DP / "nb2240_mean_bag_oof_K20.npy").astype(np.float64)
# nb2330 lacks oof - use te in-sample as best available
# we honest-evaluate via single-anchor 2240 baseline
honest = []
for w in ws:
    # mix nb2240 OOF with te2330[unb_idx] (less honest but bounded)
    blend = w * oof2240 + (1 - w) * te2330[unb_idx]
    r = float(rae(y_unb, blend))
    honest.append({"w_2240": float(w), "rae_unb_mixed": r})
best_h = min(honest, key=lambda d: d["rae_unb_mixed"])
print(f"BEST MIXED: w_2240={best_h['w_2240']:.2f} rae={best_h['rae_unb_mixed']:.4f}")

summary = {
    "tag": "nb2400",
    "method": "weighted_avg_grid_nb2240_nb2330",
    "rae_2240_insample": rae_2240,
    "rae_2330_insample": rae_2330,
    "best_insample": best,
    "best_mixed": best_h,
    "grid": results,
    "honest_grid": honest,
    "verdict": "NO_LIFT" if best["rae_unb_insample"] >= min(rae_2240, rae_2330) - 1e-4 else "POSSIBLE_LIFT",
}
with open(DP / "nb2400_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("VERDICT:", summary["verdict"])
