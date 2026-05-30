"""nb178 -- Distribution surgery on test predictions.

User suggested: 'going into the model weights and just changing them to make
sure that you get the kind of distribution that you want or getting a really
low relative absolute error'.

We can do this AT THE PREDICTION LEVEL (post-process):
1. **Quantile match**: map test predictions to train pEC50 quantile distribution.
2. **Variance expansion**: scale test predictions so std matches train std (currently
   te_std=0.60 vs train_std=1.03).
3. **Mean alignment**: shift test mean to match train mean.
4. **Isotonic surgery via per-bin calibration**: split test into quantile bins,
   adjust each bin's mean to match the corresponding train quantile's mean.

All variants saved as candidate submissions for testing. We'll see which (if any)
move LB.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pxr.data import load_train, load_test
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb178: Distribution surgery on test predictions ===\n")
    tr = load_train()
    te_df = load_test()
    y_tr = tr["pEC50"].values.astype(np.float64) if "pEC50" in tr.columns else tr["pec50"].values.astype(np.float64)

    te_pred = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")
    te_names = te_df["Molecule Name"].tolist() if "Molecule Name" in te_df.columns else te_df["name"].tolist()

    print(f"Base te_pred: mean={te_pred.mean():.3f}, std={te_pred.std():.3f}, min={te_pred.min():.3f}, max={te_pred.max():.3f}")
    print(f"Train labels: mean={y_tr.mean():.3f}, std={y_tr.std():.3f}, min={y_tr.min():.3f}, max={y_tr.max():.3f}")

    candidates = []

    # 1) Quantile match (most aggressive — replaces values with quantile-matched train values)
    sorted_train = np.sort(y_tr)
    rank_te = np.argsort(np.argsort(te_pred))  # ranks 0..512
    # Map rank to quantile in train distribution
    quantile_te = (rank_te + 0.5) / len(te_pred)
    qmatch_pred = np.array([np.quantile(y_tr, q) for q in quantile_te])
    print(f"\nQuantile-match: mean={qmatch_pred.mean():.3f}, std={qmatch_pred.std():.3f}")
    candidates.append(("240_qmatch_train", qmatch_pred))

    # 2) Variance scaling to match train std
    target_std = y_tr.std()
    scaling = target_std / te_pred.std()
    var_scaled = te_pred.mean() + (te_pred - te_pred.mean()) * scaling
    print(f"\nVar-scaled (std->{target_std:.3f}): mean={var_scaled.mean():.3f}, std={var_scaled.std():.3f}")
    candidates.append(("241_var_scaled_train_std", var_scaled))

    # 3) Half-variance scaling (less aggressive)
    half_target = (te_pred.std() + target_std) / 2
    half_scaling = half_target / te_pred.std()
    half_scaled = te_pred.mean() + (te_pred - te_pred.mean()) * half_scaling
    print(f"Half-var-scaled (std->{half_target:.3f}): std={half_scaled.std():.3f}")
    candidates.append(("242_var_scaled_half", half_scaled))

    # 4) Mean alignment to train mean
    mean_aligned = te_pred + (y_tr.mean() - te_pred.mean())
    print(f"\nMean-aligned (->{y_tr.mean():.3f}): mean={mean_aligned.mean():.3f}, std={mean_aligned.std():.3f}")
    candidates.append(("243_mean_aligned_train", mean_aligned))

    # 5) Quantile transform via piecewise linear interpolation
    # Less harsh than full quantile match — preserves order, adjusts spread
    quantile_te_q = np.argsort(np.argsort(te_pred)) / (len(te_pred) - 1)
    qt = np.quantile(y_tr, quantile_te_q)
    # Blend qt with original
    for w in [0.25, 0.50]:
        b = (1 - w) * te_pred + w * qt
        print(f"qt-blend w={w}: mean={b.mean():.3f}, std={b.std():.3f}")
        candidates.append((f"244_qt_blend_w{int(w*100):02d}", b))

    # 6) Rank-preserving variance expansion - amplify deviations from median
    median = np.median(te_pred)
    for amp in [1.3, 1.5, 1.8]:
        amplified = median + (te_pred - median) * amp
        print(f"amp x{amp}: mean={amplified.mean():.3f}, std={amplified.std():.3f}")
        candidates.append((f"245_amp_{int(amp*10):02d}", amplified))

    # 7) Clip-and-expand: clip then rescale
    p5, p95 = np.percentile(te_pred, [5, 95])
    clip_target = y_tr.min(), y_tr.max()
    clipped = np.clip(te_pred, p5, p95)
    expand_scaling = (clip_target[1] - clip_target[0]) / (p95 - p5)
    expanded = clip_target[0] + (clipped - p5) * expand_scaling
    print(f"\nClip-expand to train range: mean={expanded.mean():.3f}, std={expanded.std():.3f}")
    candidates.append(("246_clip_expand", expanded))

    # Save all candidates
    print("\nSaving candidate submissions:")
    for name, pred in candidates:
        path = SUBMISSIONS / f"{name}.csv"
        pred_clip = np.clip(pred, 0.5, 8.5)  # safety clip
        sub = pd.DataFrame({"Molecule Name": te_names, "pEC50": pred_clip})
        sub.to_csv(path, index=False)
        print(f"  {name}.csv  mean={pred_clip.mean():.3f}  std={pred_clip.std():.3f}")

    print("\n*** Distribution surgery candidates ready. ***")
    print("Recommend submitting in this priority order:")
    print("  1. 244_qt_blend_w25 (mildest, blends in train quantile shape at 25%)")
    print("  2. 244_qt_blend_w50 (stronger blend)")
    print("  3. 246_clip_expand (clips outliers + expands)")
    print("  4. 240_qmatch_train (most aggressive — full quantile replacement)")


if __name__ == "__main__":
    main()
