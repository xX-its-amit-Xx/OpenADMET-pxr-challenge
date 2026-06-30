"""nb1223 -- Recursive residual on nb1211 anchor (MACCS-166 features).

Hypothesis:
    nb1211 is the blend of two BoBs (mean_mean variant, pooled RAE 0.5451).
    Its residual against truth has SHRUNK variance vs nb1070-only residual,
    but may still carry MACCS-correlated structure that a SHALLOW residual on
    top can extract. Capacity is capped aggressively because residual
    variance is smaller than at the nb1192 stage.

Protocol:
  1. Anchor = nb1211_mean_oof (pooled RAE 0.5451 on 253 unblind).
  2. residual = y_unb - nb1211_mean_oof
  3. Features: MACCS-166 from data/processed/te_maccs.npy (513,167) sliced
     to unb 253. Zero-variance bits drop out for LGBM with no penalty.
  4. Per seed s in {0, 1, 7, 42, 137}:
       KFold(n=5, shuffle=True, random_state=s) on 253 unblind.
       VERY SHALLOW LGBM Huber:
           max_depth=2, num_leaves=3, n_estimators=40, lr=0.03,
           min_child_samples=30, objective=huber(alpha=1.0).
       Cross-fit residual_oof_s; pred_corr_s = anchor_oof + residual_oof_s.
  5. mean_bag_oof = mean over seeds of pred_corr_s; pooled RAE.
  6. Verdict at 0.003 margin vs nb1211 (0.5451).

Outputs:
  data/processed/nb1223_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1223_mean_bag_oof.npy            (253,)   float32
  data/processed/nb1223_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1223"
ANCHOR = "nb1211"
ANCHOR_FILE = "nb1211_mean_oof.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771   # for residual-std-ratio comparison
NB1211_MEAN_REF = 0.5451     # the anchor we must beat
MARGIN = 0.003

MACCS_CACHE = DATA_PROCESSED / "te_maccs.npy"   # (513, 167) uint8


def _lgbm_params(seed: int) -> dict:
    """VERY SHALLOW LGBM Huber -- nb1211 residual variance is even more
    shrunk than nb1192 (BoB-of-BoBs), so cap capacity aggressively."""
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.03,
        n_estimators=40,
        max_depth=2,
        num_leaves=3,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RECURSIVE residual LGBM on nb1211 anchor (MACCS-166)")
    print(f"          seeds  = {RESID_SEEDS}")
    print(f"          target = y_unb - nb1211_mean_oof")
    print(f"          feats  = MACCS-166 (te_maccs.npy = 167 bits)")
    print(f"          LGBM   : max_depth=2, num_leaves=3, n_est=40, lr=0.03,")
    print(f"                   min_child_samples=30, obj=huber(alpha=1.0)")
    print("=" * 78)

    # --- load unblind alignment + labels ---
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}  unb_idx[:5]={unb_idx[:5].tolist()}")

    # --- anchor ---
    anchor_path = DATA_PROCESSED / ANCHOR_FILE
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF (run nb1211 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: "
            f"{anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR_FILE} shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref {NB1211_MEAN_REF:.4f})")

    # --- nb1070 reference for residual-std ratio ---
    nb1070_path = DATA_PROCESSED / "nb1070_pred_oof.npy"
    nb1070_resid_std = None
    if nb1070_path.exists():
        nb1070_oof = np.load(nb1070_path).astype(np.float64)
        if nb1070_oof.shape[0] == n_unb:
            r1070 = y_unb - nb1070_oof
            nb1070_resid_std = float(r1070.std())
            print(f"[load] nb1070 reference residual std = {nb1070_resid_std:.4f}")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")
    ratio_nb1211_over_nb1070 = None
    if nb1070_resid_std is not None:
        ratio_nb1211_over_nb1070 = float(residual.std() / nb1070_resid_std)
        print(f"[resid] std_ratio (nb1211 / nb1070) = "
              f"{ratio_nb1211_over_nb1070:.4f}")

    # --- MACCS features (513, 167) -> slice to (253, 167) ---
    if not MACCS_CACHE.exists():
        raise FileNotFoundError(
            f"{MACCS_CACHE} not found; required MACCS cache."
        )
    X_te_maccs = np.load(MACCS_CACHE).astype(np.float32)
    print(f"[feat] te_maccs full shape = {X_te_maccs.shape}")
    if X_te_maccs.shape[0] < (int(unb_idx.max()) + 1):
        raise ValueError(
            f"te_maccs.npy too short for unb_idx: "
            f"{X_te_maccs.shape[0]} vs idx_max={int(unb_idx.max())}"
        )
    X_unb = X_te_maccs[unb_idx].astype(np.float32)
    n_zero_var = int((X_unb.std(axis=0) == 0).sum())
    print(f"[feat] MACCS unblind shape = {X_unb.shape}  "
          f"zero-variance bits = {n_zero_var}")

    print("\n" + "-" * 78)
    print("PER-SEED RESIDUAL CROSS-FIT (VERY SHALLOW depth=2, MACCS-166)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1211": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1211 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATION")
    print("-" * 78)
    print(f"   per-seed RAE list    = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean        = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median      = {rae_per_seed_median:.4f}")
    print(f"   per-seed std         = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max     = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag) = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1211 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   nb1211 anchor        = {NB1211_MEAN_REF:.4f}")

    beats_nb1211 = rae_mean_bag < NB1211_MEAN_REF - MARGIN

    if beats_nb1211:
        verdict = "RECURSIVE_RESIDUAL_NB1211_MACCS_BEATS_ANCHOR_BY_MARGIN"
    elif abs(rae_mean_bag - NB1211_MEAN_REF) < MARGIN:
        verdict = "RECURSIVE_RESIDUAL_NB1211_MACCS_TIES_ANCHOR_NO_SIGNAL_LEFT"
    else:
        verdict = "RECURSIVE_RESIDUAL_NB1211_MACCS_HURTS_ANCHOR_OVERFITS_NOISE"
    print(f"   verdict              = {verdict}")

    # --- save artifacts ---
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_file": ANCHOR_FILE,
        "feature_source": "maccs_166_from_te_maccs_npy",
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "feature_zero_variance_bits": n_zero_var,
        "lgbm_max_depth": 2,
        "lgbm_num_leaves": 3,
        "lgbm_n_estimators": 40,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 30,
        "lgbm_alpha_huber": 1.0,
        "rae_anchor_nb1211": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_std_nb1070": nb1070_resid_std,
        "residual_std_ratio_nb1211_over_nb1070": ratio_nb1211_over_nb1070,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "delta_mean_bag_vs_nb1211": rae_mean_bag - rae_anchor,
        "margin": MARGIN,
        "beats_nb1211": bool(beats_nb1211),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1211_mean_ref": NB1211_MEAN_REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("rae_anchor_nb1211", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag",
              "delta_mean_bag_vs_nb1211",
              "residual_std", "residual_std_nb1070",
              "residual_std_ratio_nb1211_over_nb1070",
              "beats_nb1211",
              "verdict"):
        print(f"  {k}: {res.get(k)}")
