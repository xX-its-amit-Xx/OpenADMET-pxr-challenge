"""nb1233 -- Novelty-weighted residual-LGBM bag on nb1070 anchor, MACCS-167.

Hypothesis:
    Per the phase-1 post-mortem, novel-scaffold rows are systematically
    over-predicted (F2 failure mode, +1.23 mean residual on greasy-novel
    inactives).  A residual learner that UPWEIGHTS rare-scaffold samples
    during fit may extract structure for the failure tail without disturbing
    the common-scaffold predictions (which are already near-calibrated).

    Twin of nb1183 (MACCS residual bag on nb1070) -- same anchor, same
    features, same LGBM capacity / seeds / folds -- the ONLY change is that
    each unblind sample carries weight w_i = 1 / sqrt(max(1, freq_i)) so
    rare-scaffold rows tilt the residual fit toward themselves.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = nb1070 pred_oof (constant across seeds).
  2. residual = y_unb - nb1070_oof
  3. weight_i = 1 / sqrt(max(1, scaf_train_freq_i))
  4. KFold(n=5, shuffle=True, random_state=s) on 253 unblind rows.
  5. Shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80, lr=0.05,
     min_child_samples=20, alpha=1.0), sample_weight=w[train_loc] each fold.
  6. pred_corrected_s = nb1070_oof + residual_oof_s; pooled RAE.

Mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds of pred_corr_s).
Verdict vs nb1183 mean-bag (0.5513) at 0.003 margin.

Scaffold-frequency lookup:
    Cached at data/processed/postmortem/pm_test_chem_all513.parquet, column
    `scaf_train_freq` (already pre-computed via Murcko scaffold).  Sliced via
    `_audit_unblind_idx.npy` to align to the 253 unblind rows.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1233_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1233_mean_bag_oof.npy            (253,)   float32
  data/processed/nb1233_median_bag_oof.npy          (253,)   float32
  data/processed/nb1233_summary.json
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
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1233"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"   # (4139, 167) uint8
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"   # (513, 167)  uint8

PM_CHEM_PATH = DATA_PROCESSED / "postmortem" / "pm_test_chem_all513.parquet"

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1183_MEAN_BAG_REF = 0.5513   # MACCS residual bag on nb1070 (un-weighted twin)
DECISION_MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1183 (un-weighted twin)."""
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, weights: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(
            X[tr_loc],
            residual[tr_loc],
            sample_weight=weights[tr_loc],
        )
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_maccs_unblind(n_test_expected: int, unb_idx: np.ndarray) -> np.ndarray:
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X_te = np.load(MACCS_TE_PATH)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"MACCS test cache shape mismatch: {X_te.shape} "
            f"vs n_test={n_test_expected}"
        )
    if X_te.shape[1] not in (166, 167):
        raise ValueError(
            f"MACCS test cache unexpected width: {X_te.shape[1]} "
            f"(expected 166 or 167)"
        )
    X_unb = X_te[unb_idx].astype(np.float32)
    return X_unb


def _load_scaf_freq_unblind(unb_idx: np.ndarray) -> np.ndarray:
    """Pull `scaf_train_freq` for the 253 unblind rows from cached postmortem
    parquet (pre-computed Murcko-scaffold lookup against the 4,139 train set).
    Falls back to inline computation if the parquet is absent.
    """
    if PM_CHEM_PATH.exists():
        df = pd.read_parquet(PM_CHEM_PATH)
        if "scaf_train_freq" not in df.columns:
            raise ValueError(
                f"{PM_CHEM_PATH} missing column scaf_train_freq"
            )
        if len(df) != 513:
            raise ValueError(
                f"{PM_CHEM_PATH} has {len(df)} rows (expected 513)"
            )
        sub = df.iloc[unb_idx]
        if "is_unblind" in df.columns and not sub["is_unblind"].all():
            raise ValueError(
                "unb_idx selects rows that are not is_unblind=True in "
                f"{PM_CHEM_PATH}"
            )
        freq = sub["scaf_train_freq"].astype(np.int64).to_numpy()
        print(f"[freq] loaded scaf_train_freq from cached {PM_CHEM_PATH.name}")
        return freq

    # ---- Fallback: compute inline via RDKit Murcko scaffold ----
    print(f"[freq] {PM_CHEM_PATH} missing -- computing inline via RDKit Murcko")
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from pxr.data import load_train

    def _scaffold(smi: str) -> str:
        if not isinstance(smi, str) or not smi:
            return ""
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return ""
            sc = MurckoScaffold.GetScaffoldForMol(m)
            return Chem.MolToSmiles(sc) if sc is not None else ""
        except Exception:
            return ""

    tr = load_train()
    smi_col = "smiles" if "smiles" in tr.columns else (
        "SMILES" if "SMILES" in tr.columns else None
    )
    if smi_col is None:
        raise ValueError(f"load_train columns missing SMILES: {tr.columns.tolist()}")
    tr_scaffolds = [_scaffold(s) for s in tr[smi_col].tolist()]
    from collections import Counter
    sc_count = Counter(tr_scaffolds)

    te = load_test()
    te_smi_col = "smiles" if "smiles" in te.columns else (
        "SMILES" if "SMILES" in te.columns else None
    )
    te_scaffolds = [_scaffold(s) for s in te[te_smi_col].tolist()]
    freq_all = np.array([sc_count.get(s, 0) for s in te_scaffolds], dtype=np.int64)
    return freq_all[unb_idx]


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- NOVELTY-WEIGHTED residual-LGBM bag on top of nb1070, "
          f"MACCS-167 features, {len(RESID_SEEDS)} KFold seeds")
    print(f"          seeds   = {RESID_SEEDS}")
    print(f"          weight  = 1 / sqrt(max(1, scaf_train_freq))")
    print(f"          target  = y_unb - nb1070_pred_oof")
    print(f"          LGBM    : max_depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF (run nb1070 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: {anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    print(f"[feat] loading cached MACCS test matrix, slicing to "
          f"{n_unb} unblind rows ...")
    X_unb = _load_maccs_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat] X_unb shape = {X_unb.shape}")

    # ---- Novelty weights ----
    print("\n" + "-" * 78)
    print("NOVELTY WEIGHTS (1 / sqrt(max(1, scaf_train_freq)))")
    print("-" * 78)
    freq_unb = _load_scaf_freq_unblind(unb_idx)
    if freq_unb.shape[0] != n_unb:
        raise ValueError(
            f"scaf_train_freq slice shape {freq_unb.shape} != n_unb={n_unb}"
        )
    freq_clip = np.maximum(1, freq_unb).astype(np.float64)
    weights = 1.0 / np.sqrt(freq_clip)
    w_min, w_med, w_max = float(weights.min()), float(np.median(weights)), float(weights.max())
    w_std = float(weights.std())
    w_mean = float(weights.mean())
    n_w1 = int((weights >= 1.0 - 1e-12).sum())
    n_w_lt1 = int((weights < 1.0 - 1e-12).sum())
    print(f"   freq raw   : min={int(freq_unb.min())}  median={int(np.median(freq_unb))}  "
          f"max={int(freq_unb.max())}  n_zero={int((freq_unb==0).sum())}  "
          f"n_one={int((freq_unb==1).sum())}")
    print(f"   weight     : min={w_min:.4f}  median={w_med:.4f}  max={w_max:.4f}  "
          f"std={w_std:.4f}  mean={w_mean:.4f}")
    print(f"   weight==1.0 (rare/novel, freq<=1): n={n_w1}/{n_unb}")
    print(f"   weight<1.0  (common, freq>=2)   : n={n_w_lt1}/{n_unb}")

    # ---- Per-seed residual cross-fit (weighted) ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (weighted, depth=3, MACCS {X_unb.shape[1]})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, weights, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1183 mean_bag ref    = {NB1183_MEAN_BAG_REF:.4f}  "
          f"(un-weighted MACCS twin)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_MEAN_BAG_REF - DECISION_MARGIN

    if beats_nb1183:
        verdict = "NOVELTY_WEIGHTING_BEATS_UNWEIGHTED_NB1183"
    elif abs(rae_mean_bag - NB1183_MEAN_BAG_REF) < DECISION_MARGIN:
        verdict = "NOVELTY_WEIGHTING_TIES_NB1183_NO_NEW_SIGNAL"
    elif beats_nb1070:
        verdict = "NOVELTY_WEIGHTING_HELPS_NB1070_BUT_LOSES_TO_NB1183"
    else:
        verdict = "NOVELTY_WEIGHTING_HURTS_VS_NB1070_AND_NB1183"
    print(f"   verdict                = {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "maccs_cached_167",
        "weight_scheme": "1/sqrt(max(1,scaf_train_freq))",
        "scaf_freq_source": (
            str(PM_CHEM_PATH) if PM_CHEM_PATH.exists() else "inline_rdkit_murcko"
        ),
        "maccs_cache_train": str(MACCS_TR_PATH),
        "maccs_cache_test": str(MACCS_TE_PATH),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "freq_min": int(freq_unb.min()),
        "freq_median": float(np.median(freq_unb)),
        "freq_max": int(freq_unb.max()),
        "freq_n_zero": int((freq_unb == 0).sum()),
        "freq_n_one": int((freq_unb == 1).sum()),
        "weight_min": w_min,
        "weight_median": w_med,
        "weight_max": w_max,
        "weight_std": w_std,
        "weight_mean": w_mean,
        "n_weight_eq_one": n_w1,
        "n_weight_lt_one": n_w_lt1,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_nb1070": rae_median_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_MEAN_BAG_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
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
    for k in (
        "rae_anchor_nb1070",
        "weight_min", "weight_median", "weight_max", "weight_std",
        "n_weight_eq_one", "n_weight_lt_one",
        "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_median", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "beats_nb1070", "beats_nb1183",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
