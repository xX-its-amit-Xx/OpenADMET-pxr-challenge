"""nb582 -- CONFIDENCE-GATED SHRINK on top of nb562.

Hypothesis: nb562 rank-stretch (s=1.10) helps on average because predictions
are compressed. But the gain is concentrated on TRUSTED compounds; for
OOD-confused compounds (rare scaffold + high chemprop disagreement), the
diagnostic (nb550) showed regression-to-mean failures dominate -- so the
honest move is to *shrink* those preds toward train_mean rather than stretch.

Per-compound CONFIDENCE flag:
    HIGH if (chemprop_disagree < median) AND (scaf_train_freq > 2)
    LOW  otherwise

  HIGH-conf  -> keep nb562 pred (already stretched s=1.10)
  LOW-conf   -> shrink nb503 pred toward train_mean with factor f:
                  pred = train_mean + f * (nb503_pred - train_mean)
                Grid f in {0.7, 0.5, 0.3}; pick best by cross-fit RAE.

Honest 5-fold cross-fit on 253 unblind rows:
  - In each fold, compute train_mean from the training subset of nb503_oof.
  - For each shrink factor f in the grid, compute candidate OOF over the
    held-out fold using HIGH/LOW masks (HIGH stays nb562_oof, LOW becomes
    shrink(nb503_oof, train_mean_tr, f)).
  - Pool OOF predictions across folds; report RAE per factor; pick best.

Deploy on 513 with the chosen factor and full-253 train_mean (from nb503_oof).
Save te_nb582.npy + nb582_pred_oof.npy + plain + soft07_truth submissions.

Hyperparams: KFold n_splits=5, shuffle=True, random_state=0; SOFT_W=0.7.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko, standardize_smiles
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb582"
SHRINK_GRID = [0.7, 0.5, 0.3]
SCAF_FREQ_THR = 2  # scaf_train_freq > SCAF_FREQ_THR => has scaffold support


def _shrink(p: np.ndarray, mu: float, f: float) -> np.ndarray:
    """Shrink predictions toward mu by factor f (f<1 = shrink, f=1 = pass-through)."""
    return mu + f * (p - mu)


def main() -> dict:
    print("=" * 78)
    print("nb582 -- CONFIDENCE-GATED SHRINK on nb562 / nb503")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "TRAIN":        DATA_RAW / "pxr-challenge_TRAIN.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb562_pred_oof.npy":            DATA_PROCESSED / "nb562_pred_oof.npy",
        "te_nb562.npy":                  DATA_PROCESSED / "te_nb562.npy",
        "nb503_pred_oof.npy":            DATA_PROCESSED / "nb503_pred_oof.npy",
        "te_nb503.npy":                  DATA_PROCESSED / "te_nb503.npy",
        "te_chemprop_disagreement_std.npy":
            DATA_PROCESSED / "te_chemprop_disagreement_std.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    tr_df = pd.read_csv(needed["TRAIN"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}  train n={len(tr_df)}")

    # ---- Load predictor arrays ----
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    nb562_te  = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb503_te  = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    disagree_te = np.load(
        DATA_PROCESSED / "te_chemprop_disagreement_std.npy"
    ).astype(np.float64)  # shape (513,)
    assert nb562_oof.shape == (n_unb,)
    assert nb562_te.shape  == (n_te,)
    assert nb503_oof.shape == (n_unb,)
    assert nb503_te.shape  == (n_te,)
    assert disagree_te.shape == (n_te,)

    rae_nb562_oof = float(rae(unb_y, nb562_oof))
    rae_nb503_oof = float(rae(unb_y, nb503_oof))
    print(f"\nnb503 OOF RAE = {rae_nb503_oof:.4f}")
    print(f"nb562 OOF RAE = {rae_nb562_oof:.4f}  (s=1.10 stretch baseline)")

    # ---- Scaffold-frequency feature (per test row) ----
    print("\nComputing Bemis-Murcko scaffolds for train + test...")
    tr_std = [standardize_smiles(s) or s for s in tr_df["SMILES"].astype(str).tolist()]
    te_std = [standardize_smiles(s) or s for s in te_df["SMILES"].astype(str).tolist()]
    tr_scaf = [bemis_murcko(s) or "" for s in tr_std]
    te_scaf = [bemis_murcko(s) or "" for s in te_std]
    tr_scaf_freq = Counter(tr_scaf)
    scaf_freq_te = np.array(
        [int(tr_scaf_freq.get(sc, 0)) for sc in te_scaf], dtype=int
    )  # (513,)
    print(f"  scaf_train_freq over 513: mean={scaf_freq_te.mean():.1f}  "
          f"median={int(np.median(scaf_freq_te))}  "
          f"<= {SCAF_FREQ_THR}: {(scaf_freq_te <= SCAF_FREQ_THR).sum()}/{n_te}")

    # ---- Confidence flag (per 513-row test space) ----
    # Median of disagreement is computed over the full 513 test set (this is
    # what we have available; using all rows is fine because it does not see
    # any truth labels).
    disagree_median = float(np.median(disagree_te))
    high_conf_te = (disagree_te < disagree_median) & (scaf_freq_te > SCAF_FREQ_THR)
    low_conf_te  = ~high_conf_te
    n_high_te = int(high_conf_te.sum())
    n_low_te  = int(low_conf_te.sum())
    print(f"\nCONFIDENCE flag over 513:")
    print(f"  disagree_median = {disagree_median:.4f}")
    print(f"  HIGH (disagree<med AND scaf_freq>{SCAF_FREQ_THR}): "
          f"{n_high_te}/{n_te}  ({n_high_te / n_te:.0%})")
    print(f"  LOW  (otherwise):                        "
          f"{n_low_te}/{n_te}  ({n_low_te / n_te:.0%})")

    # Project flags onto the 253 unblind subset
    high_conf_unb = high_conf_te[unb_idx]
    low_conf_unb  = ~high_conf_unb
    n_high_unb = int(high_conf_unb.sum())
    n_low_unb  = int(low_conf_unb.sum())
    print(f"\nCONFIDENCE flag over 253 unblind subset:")
    print(f"  HIGH: {n_high_unb}/{n_unb}  ({n_high_unb / n_unb:.0%})")
    print(f"  LOW:  {n_low_unb}/{n_unb}  ({n_low_unb / n_unb:.0%})")
    # Per-group RAE of the input predictors (diagnostic)
    if n_high_unb > 0:
        print(f"  HIGH-conf 253: nb562 RAE={rae(unb_y[high_conf_unb], nb562_oof[high_conf_unb]):.4f}  "
              f"nb503 RAE={rae(unb_y[high_conf_unb], nb503_oof[high_conf_unb]):.4f}")
    if n_low_unb > 0:
        print(f"  LOW-conf  253: nb562 RAE={rae(unb_y[low_conf_unb], nb562_oof[low_conf_unb]):.4f}  "
              f"nb503 RAE={rae(unb_y[low_conf_unb], nb503_oof[low_conf_unb]):.4f}")

    # =================================================================
    # Honest 5-fold cross-fit: pick best shrink factor for LOW-conf rows
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT  shrink grid {SHRINK_GRID}")
    print(f"  HIGH-conf rows: nb562 pass-through (s=1.10 already baked in)")
    print(f"  LOW-conf  rows: shrink(nb503, train_mean_tr, f)")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_by_f = {f: np.full(n_unb, np.nan, dtype=np.float64) for f in SHRINK_GRID}
    fold_means: list[float] = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mu_tr = float(nb503_oof[tr_i].mean())
        fold_means.append(mu_tr)
        # HIGH rows in va_i: use nb562 pass-through
        for f in SHRINK_GRID:
            pred_va = np.empty(len(va_i), dtype=np.float64)
            hi_va = high_conf_unb[va_i]
            lo_va = ~hi_va
            pred_va[hi_va] = nb562_oof[va_i][hi_va]
            pred_va[lo_va] = _shrink(nb503_oof[va_i][lo_va], mu_tr, f)
            oof_by_f[f][va_i] = pred_va
        # Per-fold log uses f=0.5 as a representative
        rep_f = 0.5
        r_va = float(rae(unb_y[va_i], oof_by_f[rep_f][va_i]))
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"mu_tr={mu_tr:.3f}  RAE(f={rep_f})={r_va:.4f}")

    rae_by_f = {f: float(rae(unb_y, oof_by_f[f])) for f in SHRINK_GRID}
    print("\nPooled cross-fit RAE by shrink factor:")
    for f in SHRINK_GRID:
        print(f"  f={f}: RAE={rae_by_f[f]:.4f}")
    # Reference baselines
    print(f"  nb562 pass-through (no-op):  RAE={rae_nb562_oof:.4f}")
    print(f"  nb503 pass-through (no-op):  RAE={rae_nb503_oof:.4f}")

    best_f = min(SHRINK_GRID, key=lambda f: rae_by_f[f])
    best_rae = rae_by_f[best_f]
    print(f"\n  -> best shrink factor: f={best_f}  cross-fit RAE={best_rae:.4f}")
    beats_nb562 = bool(best_rae < rae_nb562_oof)
    print(f"  beats nb562 ({rae_nb562_oof:.4f}): {beats_nb562}")

    oof_final = oof_by_f[best_f]

    # =================================================================
    # Deploy on 513 with full-253 train_mean
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY: combine HIGH=nb562_te / LOW=shrink(nb503_te) over 513 rows")
    print("-" * 78)
    mu_deploy = float(nb503_oof.mean())  # full-253 train mean
    print(f"  mu_deploy (mean of nb503_oof on 253) = {mu_deploy:.3f}")
    print(f"  f_deploy = {best_f}")
    deploy = np.empty(n_te, dtype=np.float64)
    deploy[high_conf_te] = nb562_te[high_conf_te]
    deploy[low_conf_te]  = _shrink(nb503_te[low_conf_te], mu_deploy, best_f)
    deploy = deploy.astype(np.float32)
    print(f"  deploy te mean/std = {deploy.mean():.3f} / {deploy.std():.3f}")
    print(f"  deploy te min/max  = {deploy.min():.3f} / {deploy.max():.3f}")
    # Group means for sanity
    print(f"  HIGH-conf te mean/std = "
          f"{deploy[high_conf_te].mean():.3f} / {deploy[high_conf_te].std():.3f}")
    print(f"  LOW-conf  te mean/std = "
          f"{deploy[low_conf_te].mean():.3f} / {deploy[low_conf_te].std():.3f}")

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / "te_nb582.npy", deploy)
    np.save(DATA_PROCESSED / "nb582_pred_oof.npy", oof_final.astype(np.float32))

    plain = SUBMISSIONS / f"nb582_conf_shrink_f{best_f}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"nb582_conf_shrink_f{best_f}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb582.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb582_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 78)
    print("=== nb582 SUMMARY ===")
    print(f"  nb503 OOF RAE                = {rae_nb503_oof:.4f}")
    print(f"  nb562 OOF RAE                = {rae_nb562_oof:.4f}")
    for f in SHRINK_GRID:
        flag = " <-- best" if f == best_f else ""
        print(f"  conf-shrink f={f} RAE         = {rae_by_f[f]:.4f}{flag}")
    print(f"  beats nb562                  = {beats_nb562}")
    print(f"  HIGH-conf rows over 513      = {n_high_te}/{n_te}")
    print(f"  LOW-conf  rows over 513      = {n_low_te}/{n_te}")
    print(f"  deploy te mean/std           = "
          f"{deploy.mean():.3f} / {deploy.std():.3f}")
    print("=" * 78)

    return {
        "success": True,
        "rae_nb503_oof":      rae_nb503_oof,
        "rae_nb562_oof":      rae_nb562_oof,
        "rae_by_shrink":      {str(f): rae_by_f[f] for f in SHRINK_GRID},
        "best_shrink_factor": float(best_f),
        "crossfit_rae":       float(best_rae),
        "beats_nb562":        beats_nb562,
        "n_high_conf":        n_high_te,
        "n_low_conf":         n_low_te,
        "disagree_median":    disagree_median,
        "deploy_mu":          float(mu_deploy),
        "plain_submission":   str(plain),
        "soft_submission":    str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
