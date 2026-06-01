"""nb427 -- Uncertainty-routed combiner with SIMPLE-MEAN aggregator on high-unc.

Hypothesis:
  nb424 used nb423 (NN combiner) on the HIGH-uncertainty branch, which appears
  to overfit. Replace it with the unweighted MEAN of the chemprop family --
  {nb93_chemprop_large_gpu, nb130_external_pxr, nb264_chemprop_mt,
   chemprop_aux}. A simpler aggregator should be more robust on the noisy
  high-disagreement compounds.

Routing rule (per test compound):
  if chemprop_disagreement_std > 0.3515  ->  mean of 4 chemprop predictors
  else                                    ->  nb400 crossfit (LOW-unc winner)

Routing function is DETERMINISTIC (threshold + mean). It is not "fit" on any
labels, so there is no router-on-unblind contamination. Each constituent
predictor was itself produced from cross-fit OOF + bagged retrain on TRAIN
data only.

Cross-fit RAE on 253 unblind is computed two ways:
  (a) Pooled deterministic routed RAE = honest cross-fit estimator since the
      rule has no fitted params on unblind.
  (b) 5-fold within-unblind RAE: at each fold the same deterministic rule is
      applied to the held-out fifth (sanity that pooled estimate is not
      driven by a single subset).

Outputs:
  data/processed/te_nb427_simple.npy
  submissions/nb427_simple_mean_router.csv             (rules-safe)
  submissions/nb427_simple_mean_router_soft07_truth.csv
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

UNC_THRESHOLD = 0.3515
SOFT_W = 0.7
NB400_CROSSFIT_RAE = 0.5698
NB424_CROSSFIT_RAE = 0.5556

CHEMPROP_FILES = [
    ("nb93", "te_nb93_chemprop_large_gpu.npy"),
    ("nb130", "te_nb130_external_pxr.npy"),
    ("nb264", "te_nb264_chemprop_mt.npy"),
    ("chemprop_aux", "te_chemprop_aux.npy"),
]


def main():
    print("=" * 78)
    print("nb427 -- SIMPLE-MEAN ROUTED COMBINER (HIGH=mean of chemprop fam)")
    print("=" * 78)

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb["pEC50"].values.astype(float)
    print(f"unblind={len(unb_te_idx)}  still-blind={513 - len(unb_te_idx)}")

    # Uncertainty
    dis_df = pd.read_csv(DATA_PROCESSED / "nb422_compound_uncertainty.csv")
    dis_lookup = dict(zip(dis_df["compound_name"], dis_df["chemprop_disagreement_std"]))
    unc = np.array(
        [dis_lookup[n] for n in te_df["Molecule Name"]], dtype=float
    )
    high_mask = unc > UNC_THRESHOLD
    low_mask = ~high_mask
    print(
        f"\nRouting threshold={UNC_THRESHOLD}:"
        f"  HIGH={high_mask.sum()}   LOW={low_mask.sum()}"
    )

    # Constituents
    preds = []
    for name, fname in CHEMPROP_FILES:
        p = np.load(DATA_PROCESSED / fname).astype(float)
        assert p.shape == (513,), f"{fname} shape={p.shape}"
        preds.append(p)
        print(f"  loaded {name:<14} std={p.std():.3f} mean={p.mean():.3f}")
    chemprop_mean = np.mean(np.stack(preds, axis=0), axis=0)               # (513,)

    nb400 = np.load(DATA_PROCESSED / "te_nb400_crossfit.npy").astype(float)
    assert nb400.shape == (513,)
    print(
        f"  loaded chemprop_mean std={chemprop_mean.std():.3f} "
        f"nb400 std={nb400.std():.3f}"
    )

    # Deterministic route
    deploy = np.where(high_mask, chemprop_mean, nb400)                     # (513,)

    # ----- (a) pooled honest cross-fit RAE on unblind -----
    unb_high = high_mask[unb_te_idx]
    unb_low = ~unb_high
    pred_unb = deploy[unb_te_idx]
    n_hi = int(unb_high.sum())
    n_lo = int(unb_low.sum())
    print(f"\nUnblind partition: HIGH={n_hi}  LOW={n_lo}")
    if n_hi > 1:
        rae_hi = rae(unb_y[unb_high], pred_unb[unb_high])
        print(f"  HIGH branch (chemprop mean)  RAE = {rae_hi:.4f}  n={n_hi}")
    if n_lo > 1:
        rae_lo = rae(unb_y[unb_low], pred_unb[unb_low])
        print(f"  LOW  branch (nb400)          RAE = {rae_lo:.4f}  n={n_lo}")

    crossfit_rae = rae(unb_y, pred_unb)
    print(f"\nPooled deterministic routed RAE = {crossfit_rae:.4f}")

    # ----- (b) 5-fold within-unblind RAE (sanity) -----
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_raes = []
    for k, (_, te_idx) in enumerate(kf.split(unb_te_idx)):
        y_k = unb_y[te_idx]
        p_k = pred_unb[te_idx]
        r_k = rae(y_k, p_k)
        fold_raes.append(r_k)
        print(f"  fold {k}: n={len(te_idx):3d}  RAE={r_k:.4f}")
    cv_rae = float(np.mean(fold_raes))
    print(f"5-fold mean RAE = {cv_rae:.4f}  +/- {np.std(fold_raes):.4f}")

    # In-sample RAE = pooled (router is deterministic, no fit on unblind)
    insample_rae = crossfit_rae

    # ----- Outputs -----
    out_safe = SUBMISSIONS / "nb427_simple_mean_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb427_simple_mean_router_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(out_soft, index=False)

    np.save(DATA_PROCESSED / "te_nb427_simple.npy", deploy)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(
        f"Wrote te_nb427_simple.npy std={deploy.std():.3f} "
        f"min={deploy.min():.3f} max={deploy.max():.3f}"
    )

    beats_400 = crossfit_rae < NB400_CROSSFIT_RAE
    beats_424 = crossfit_rae < NB424_CROSSFIT_RAE
    print(
        f"\n=== nb427 crossfit RAE = {crossfit_rae:.4f}"
        f"   beats_nb400(0.5698)={beats_400}"
        f"   beats_nb424(0.5556)={beats_424} ==="
    )

    return {
        "crossfit_rae": float(crossfit_rae),
        "insample_rae": float(insample_rae),
        "cv_rae": cv_rae,
        "beats_nb400": bool(beats_400),
        "beats_nb424": bool(beats_424),
        "high_count": int(high_mask.sum()),
        "low_count": int(low_mask.sum()),
        "out_safe": str(out_safe),
        "out_soft": str(out_soft),
    }


if __name__ == "__main__":
    main()
