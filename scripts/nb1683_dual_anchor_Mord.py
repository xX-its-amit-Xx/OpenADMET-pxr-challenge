"""nb1683 -- Dual-anchor (chemprop_aux + nb1070) residual bag, K=20 Mordred.

HYPOTHESIS:
    Single-anchor (chemprop_aux) residual learning has been thoroughly mined
    (nb1554 mean_bag 0.5163 is best PRE-unblind candidate so far).  Extending
    to a 2-anchor approach where the residual target is centered on the MEAN
    of two complementary anchors gives the residual learner two informative
    inputs simultaneously:
        anchor_1 = chemprop_aux te[unb_idx]      (in_RAE 0.6216, PRE-unblind)
        anchor_2 = nb1070_pred_oof               (in_RAE 0.5771, honest 5f cf)
    The mean of the two anchors should lower the residual scale (because the
    two anchors disagree biologically and their disagreement averages out)
    and feeding BOTH anchors as input features lets the LGBM Huber-residual
    learn a per-row reweighting between them.

PROTOCOL:
    1. anchor_1 = chemprop_aux te[unb_idx]
       anchor_2 = nb1070_pred_oof (already shape (253,) honest CV)
       mean_anchor = (anchor_1 + anchor_2) / 2
       residual = y_unb - mean_anchor.
    2. Features (24 cols):
       - top-20 Mordred  (nb1523 best_K=20 SHAP ranking)
       - pred_chembl_pec50 (cached PRE-unblind 513 ChEMBL kNN-5)
       - mean_sim (cached PRE-unblind 513 ChEMBL top-5 mean Tanimoto)
       - anchor_1 (chemprop_aux te[unb_idx])
       - anchor_2 (nb1070_pred_oof)
    3. 5-seed bag shallow LGBM Huber, 5-fold cross-fit per seed.
       Per-seed corrected = mean_anchor + residual_oof.
       Pool mean_bag and median_bag.
    4. Verdict at 0.003 margin vs nb1554 mean_bag (0.5163).

Outputs:
    scripts/nb1683_dual_anchor_Mord.py
    data/processed/nb1683_summary.json
    data/processed/nb1683_mean_bag_oof.npy        (253,) float32
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
import lightgbm as lgb

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1683"
ANCHOR_1 = "chemprop_aux"
ANCHOR_2 = "nb1070"
ANCHOR_1_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ANCHOR_2_OOF_PATH = DATA_PROCESSED / "nb1070_pred_oof.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"

# Cached PRE-unblind ChEMBL kNN arrays (513,)
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
NB1070_REF = 0.5771   # honest 5f cross-fit bag-median on 253
NB1554_REF = 0.5163   # current PRE-unblind PRIMARY (5-way K-tuned CatBoost)
DECISION_MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow Huber LGBM tuned for residual learning at n=253."""
    return dict(
        objective="huber",
        alpha=0.9,
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.05,
        min_child_samples=8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=3,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DUAL-ANCHOR residual LGBM-Huber  "
          f"(anchors: {ANCHOR_1} + {ANCHOR_2})")
    print(f"        seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"        refs:  chemprop_aux ({CHEMPROP_AUX_REF:.4f})  "
          f"nb1070 ({NB1070_REF:.4f})  nb1554 ({NB1554_REF:.4f})  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth + indices ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Load anchors ----
    if not ANCHOR_1_TE_PATH.exists():
        raise FileNotFoundError(f"missing {ANCHOR_1_TE_PATH}")
    te_anchor1_513 = np.load(ANCHOR_1_TE_PATH).astype(np.float64)
    if te_anchor1_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor1_513.shape}"
        )
    anchor_1 = te_anchor1_513[unb_idx]
    rae_anchor_1 = float(rae(y_unb, anchor_1))
    print(f"[load] {ANCHOR_1} te[unb_idx]  in_RAE = {rae_anchor_1:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    if not ANCHOR_2_OOF_PATH.exists():
        raise FileNotFoundError(f"missing {ANCHOR_2_OOF_PATH}")
    anchor_2 = np.load(ANCHOR_2_OOF_PATH).astype(np.float64)
    if anchor_2.shape[0] != n_unb:
        raise ValueError(
            f"nb1070 pred_oof shape mismatch: {anchor_2.shape} vs n_unb={n_unb}"
        )
    rae_anchor_2 = float(rae(y_unb, anchor_2))
    print(f"[load] {ANCHOR_2} pred_oof    in_RAE = {rae_anchor_2:.4f}  "
          f"(ref {NB1070_REF:.4f})")

    # ---- Mean anchor + residual ----
    mean_anchor = 0.5 * (anchor_1 + anchor_2)
    rae_mean_anchor = float(rae(y_unb, mean_anchor))
    median_anchor = np.median(np.stack([anchor_1, anchor_2], axis=0), axis=0)
    rae_median_anchor = float(rae(y_unb, median_anchor))
    print(f"[anchor] MEAN(anchor1, anchor2)   in_RAE = {rae_mean_anchor:.4f}")
    print(f"[anchor] MEDIAN(anchor1, anchor2) in_RAE = {rae_median_anchor:.4f}  "
          f"(== mean for 2-element)")

    pearson_anchors = float(np.corrcoef(anchor_1, anchor_2)[0, 1])
    print(f"[anchor] pearson(anchor_1, anchor_2)     = {pearson_anchors:.4f}")

    residual = y_unb - mean_anchor
    print(f"[resid] residual mean={residual.mean():+.4f}  "
          f"std={residual.std():.4f}")

    # ---- Load Mordred top-K=20 ----
    if not NB1523_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB1523_SUMMARY}")
    with open(NB1523_SUMMARY) as f:
        sum_1523 = json.load(f)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    assert K_Mord_best == int(sum_1523["best_K"])
    print(f"[reuse] top-{K_Mord_best} Mordred cols (nb1523 best_K)")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top  = {X_mord_unb_top.shape}")

    # ---- ChEMBL kNN cached arrays ----
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != n_test:
        raise ValueError(
            f"pred_chembl_513 shape mismatch: {pred_chembl_513.shape}"
        )
    pred_chembl_unb = pred_chembl_513[unb_idx]
    mean_sim_unb = sim_chembl_513[unb_idx]
    print(f"[feat] pred_chembl_unb  = {pred_chembl_unb.shape}  "
          f"mean_sim_unb = {mean_sim_unb.shape}")

    # ---- Build 24-col feature matrix ----
    X_unb = np.concatenate(
        [
            X_mord_unb_top,                       # 20
            pred_chembl_unb.reshape(-1, 1),       # 1
            mean_sim_unb.reshape(-1, 1),          # 1
            anchor_1.astype(np.float32).reshape(-1, 1),   # 1
            anchor_2.astype(np.float32).reshape(-1, 1),   # 1
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = K_Mord_best + 4
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   DUAL-ANCHOR MORD-20 matrix: {X_unb.shape}  "
          f"(top-{K_Mord_best} Mord + pred_chembl + sim + "
          f"anchor_1 + anchor_2)")

    # ---- Per-seed LGBM residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED LGBM-HUBER RESIDUAL CROSS-FIT (dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = mean_anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_mean_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_mean_anchor": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_meanA = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}  "
              f"wall = {time.time() - ts:.1f}s")

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

    # ---- Pearson vs prior PRE-unblind candidates ----
    def _pearson_vs(path: Path):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(mean_bag_oof, oof)[0, 1])

    pearson_vs_anchor1 = float(np.corrcoef(mean_bag_oof, anchor_1)[0, 1])
    pearson_vs_anchor2 = float(np.corrcoef(mean_bag_oof, anchor_2)[0, 1])
    pearson_vs_mean_anchor = float(
        np.corrcoef(mean_bag_oof, mean_anchor)[0, 1])
    pearson_vs_nb1554 = _pearson_vs(DATA_PROCESSED / "nb1554_mean_bag_oof.npy")
    pearson_vs_nb1543 = _pearson_vs(DATA_PROCESSED / "nb1543_mean_bag_oof.npy")
    pearson_vs_nb1501 = _pearson_vs(DATA_PROCESSED / "nb1501_mean_bag_oof.npy")

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
          f"(d_vs_meanA = {rae_mean_bag - rae_mean_anchor:+.4f}  "
          f"d_vs_a1 = {rae_mean_bag - rae_anchor_1:+.4f}  "
          f"d_vs_a2 = {rae_mean_bag - rae_anchor_2:+.4f}  "
          f"d_vs_nb1554 = {rae_mean_bag - NB1554_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1554 = {rae_median_bag - NB1554_REF:+.4f})")
    print(f"   Pearson(mean_bag, anchor_1)    = {pearson_vs_anchor1:.4f}")
    print(f"   Pearson(mean_bag, anchor_2)    = {pearson_vs_anchor2:.4f}")
    print(f"   Pearson(mean_bag, mean_anchor) = {pearson_vs_mean_anchor:.4f}")
    if pearson_vs_nb1554 is not None:
        print(f"   Pearson(mean_bag, nb1554)      = {pearson_vs_nb1554:.4f}")
    if pearson_vs_nb1543 is not None:
        print(f"   Pearson(mean_bag, nb1543)      = {pearson_vs_nb1543:.4f}")
    if pearson_vs_nb1501 is not None:
        print(f"   Pearson(mean_bag, nb1501)      = {pearson_vs_nb1501:.4f}")

    # ---- Verdict ----
    beats_anchor_1 = rae_mean_bag < rae_anchor_1 - DECISION_MARGIN
    beats_anchor_2 = rae_mean_bag < rae_anchor_2 - DECISION_MARGIN
    beats_mean_anchor = rae_mean_bag < rae_mean_anchor - DECISION_MARGIN
    beats_nb1554 = rae_mean_bag < NB1554_REF - DECISION_MARGIN
    flat_vs_nb1554 = abs(rae_mean_bag - NB1554_REF) < DECISION_MARGIN

    if beats_nb1554:
        verdict = "DUAL_ANCHOR_MORD20_BEATS_NB1554_NEW_PRE_UNBLIND_PRIMARY"
    elif flat_vs_nb1554:
        verdict = "DUAL_ANCHOR_MORD20_FLAT_VS_NB1554"
    elif beats_mean_anchor:
        verdict = "DUAL_ANCHOR_MORD20_BEATS_MEAN_ANCHOR_BUT_WORSE_THAN_NB1554"
    elif beats_anchor_2:
        verdict = "DUAL_ANCHOR_MORD20_BEATS_NB1070_BUT_WORSE_THAN_MEAN_ANCHOR"
    elif beats_anchor_1:
        verdict = "DUAL_ANCHOR_MORD20_BEATS_CHEMPROP_AUX_BUT_WORSE_THAN_NB1070"
    else:
        verdict = "DUAL_ANCHOR_MORD20_HURTS_ANCHORS"

    pre_unblind_clean = False
    # Note: anchor_2 (nb1070) is POST-unblind in the strict sense (5f CV on
    # the 253 unb_idx target).  This is honest cross-fit RAE but it is not
    # a PRE-unblind te slice -- flag accordingly for ladder protocol.

    print(f"   verdict                = {verdict}")
    print(f"   pre_unblind_clean      = {pre_unblind_clean}  "
          f"(anchor_2 = nb1070 5-fold OOF on 253; honest CV but not "
          f"PRE-unblind te slice)")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor_1": ANCHOR_1,
        "anchor_2": ANCHOR_2,
        "anchor_1_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_2_kind": "honest_5fold_cross_fit_oof_on_253",
        "anchor_1_path": str(ANCHOR_1_TE_PATH),
        "anchor_2_path": str(ANCHOR_2_OOF_PATH),
        "anchor_blend": "MEAN",
        "data_source": ("Mordred_nb1030_cache + cached_chembl_pec50_513 + "
                        "cached_sim_chembl_513 + chemprop_aux_te + "
                        "nb1070_pred_oof"),
        "model_family": "LightGBM_Huber",
        "lgbm_objective": "huber",
        "lgbm_alpha": 0.9,
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 8,
        "K_Mord_best": K_Mord_best,
        "K_source": {"Mordred": "nb1523 best_K"},
        "n_unb": n_unb,
        "n_top_mordred": int(K_Mord_best),
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "mordred": int(K_Mord_best),
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "anchor_1_chemprop_aux": 1,
            "anchor_2_nb1070": 1,
            "total": int(feat_dim),
        },
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_1_chemprop_aux": rae_anchor_1,
        "rae_anchor_2_nb1070": rae_anchor_2,
        "rae_mean_anchor": rae_mean_anchor,
        "rae_median_anchor": rae_median_anchor,
        "pearson_anchor_1_vs_anchor_2": pearson_anchors,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_mean_anchor": rae_mean_bag - rae_mean_anchor,
        "delta_mean_bag_vs_anchor_1": rae_mean_bag - rae_anchor_1,
        "delta_mean_bag_vs_anchor_2": rae_mean_bag - rae_anchor_2,
        "delta_mean_bag_vs_nb1554": rae_mean_bag - NB1554_REF,
        "delta_median_bag_vs_nb1554": rae_median_bag - NB1554_REF,
        "beats_anchor_1": bool(beats_anchor_1),
        "beats_anchor_2": bool(beats_anchor_2),
        "beats_mean_anchor": bool(beats_mean_anchor),
        "beats_nb1554": bool(beats_nb1554),
        "flat_vs_nb1554": bool(flat_vs_nb1554),
        "pearson_vs_anchor_1": pearson_vs_anchor1,
        "pearson_vs_anchor_2": pearson_vs_anchor2,
        "pearson_vs_mean_anchor": pearson_vs_mean_anchor,
        "pearson_vs_nb1554": pearson_vs_nb1554,
        "pearson_vs_nb1543": pearson_vs_nb1543,
        "pearson_vs_nb1501": pearson_vs_nb1501,
        "verdict": verdict,
        "pre_unblind_clean": pre_unblind_clean,
        "pre_unblind_clean_note": (
            "anchor_2 (nb1070) is an honest 5-fold cross-fit on the 253 "
            "unblind labels (rank-stretch grid + bag-median over 5 seeds); "
            "it is NOT a PRE-unblind te slice.  This dual-anchor candidate "
            "is therefore LB-faithful at the cross-fit level for the "
            "residual-stack contribution but must be re-validated before "
            "promotion to the LB ladder."
        ),
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1070_ref": NB1070_REF,
        "nb1554_ref": NB1554_REF,
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
        "K_Mord_best", "feat_dim", "feat_breakdown",
        "rae_anchor_1_chemprop_aux", "rae_anchor_2_nb1070",
        "rae_mean_anchor", "pearson_anchor_1_vs_anchor_2",
        "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_mean_anchor",
        "delta_mean_bag_vs_nb1554",
        "delta_median_bag_vs_nb1554",
        "beats_anchor_1", "beats_anchor_2",
        "beats_mean_anchor",
        "beats_nb1554", "flat_vs_nb1554",
        "pearson_vs_mean_anchor",
        "pearson_vs_nb1554",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
