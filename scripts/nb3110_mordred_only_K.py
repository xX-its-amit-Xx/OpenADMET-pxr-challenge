"""nb3110 -- Mordred-only K=20 substrate change for cross-substrate blend
            with nb3080 quantile-conditional K18-K19 anchor.

NEW PARADIGM (substrate change, NOT operator change):
    Cycles 134-169 closed all post-hoc-blend axes on the 117-col 5-way
    feature matrix (AtomPair / MACCS / Mordred / ChemPropEmbed / Avalon).
    All K-pyramid candidates (K18, K19, K20, K24, K28) on the 5-way mix
    converge to the same ~0.45 RAE ceiling.

    nb3110 makes a CLEAN SUBSTRATE swap: drop ALL fingerprints, train
    LGBM on Mordred-ONLY descriptors (1613-dim universe), select K=20
    via RFE, then blend cross-substrate with nb3080 (5-way K18-K19
    quantile-conditional anchor, deep-30 OOF RAE 0.4475).

    The hypothesis is that Mordred-only (purely physchem + topological
    descriptors, no fingerprint bits) carries orthogonal residual
    information vs the 5-way mix anchor. Cross-substrate blend should
    beat both parents.

PROTOCOL:
    1. Load Mordred-1613 cached matrix (4139 train + 513 test = 4652 rows,
       from nb2291; 253 unblind is subset of 513 test via _audit_unblind_idx).
    2. Clean: drop NaN-heavy (>5%), zero-variance, duplicated columns;
       median-impute residual; z-score standardize on TRAIN+TEST jointly.
    3. Greedy backward RFE on Mordred-only -> K=20, anchored to
       chemprop_aux te[unb_idx] residual (same recipe as nb2291).
    4. Build Mordred-K=20 deep-30 bag-mean OOF (253,) and te (513,) on
       chemprop_aux residual with 30 fresh seeds {3001..3030}.
    5. Per-fold quantile-conditional blend with nb3080 K18 anchor:
         qcut = 0.4 (matching nb3080 best combo)
         low-half (Mord K=20 pred <= q):  w_Mord=0.7,  w_3080=0.3
         high-half (Mord K=20 pred  > q): w_Mord=0.3,  w_3080=0.7
       (Mord K=20 is the new high-info anchor; nb3080 is the low-half
       safety anchor inverse to how K18 was used in nb3080 -- here Mord
       is sharper and we trust it more on the actives side, nb3080 on
       the inactives side; weights symmetric to give clean cross-blend.)
    6. 5-fold scaffold CV with 5 kf_seeds {1111..1115}; report per-seed
       pooled RAE and grand mean over kf_seeds.

GATE:
    blend mean < 0.4475 -> "BETTER_THAN_NB3080"
    else               -> "FAIL"

Outputs:
    scripts/nb3110_mordred_only_K.py
    data/processed/nb3110_summary.json
    data/processed/nb3110_pred_oof.npy  (253,) float32 median-seed OOF
    data/processed/te_nb3110.npy        (513,) float32 deploy te
    submissions/nb3110_mordred_only_K20.csv  (only on PROMOTE)
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
import lightgbm as lgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3110"

# -- Inputs --------------------------------------------------------------------
MORDRED_CACHE = Path("C:/pxr_artifacts/nb2291/X_mordred_train_test.npz")
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
NB3080_OOF_PATH = DATA_PROCESSED / "nb3080_pred_oof.npy"
NB3080_TE_PATH = DATA_PROCESSED / "te_nb3080.npy"

# -- Mordred cleaning params (same as nb2291) ---------------------------------
NAN_FRAC_MAX = 0.05
LOW_VAR_THRESH = 1e-8

# -- RFE params ---------------------------------------------------------------
K_TARGET = 20
KF_SEED_RFE = 1001
N_RFE_FOLDS = 5

# -- Mordred K=20 LGBM bag --------------------------------------------------
N_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))    # 30 fresh seeds

# -- Blend with nb3080 anchor (quantile-conditional, cross-substrate) ---------
KF_SEEDS = [1111, 1112, 1113, 1114, 1115]
Q_CUT = 0.4
# When Mord K=20 pred (sharper on actives) is LOW, trust Mord most (0.7);
# when HIGH, defer more to nb3080 5-way anchor (0.7). Symmetric cross-blend.
W_MORD_LOW = 0.7
W_3080_LOW = 0.3
W_MORD_HIGH = 0.3
W_3080_HIGH = 0.7

# -- Gate ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3080 = 0.4475

# -- References ---------------------------------------------------------------
REF_NB3080 = 0.4475
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682
REF_NB2291_K20_MORDRED = 0.4630     # nb2291 Mordred-only K=20 on chemprop_aux residual


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _load_mordred_cache(n_train_expected: int, n_test_expected: int):
    if not MORDRED_CACHE.exists():
        raise FileNotFoundError(
            f"Mordred cache missing: {MORDRED_CACHE} (run nb2291 first)"
        )
    d = np.load(MORDRED_CACHE, allow_pickle=True)
    X = d["X"].astype(np.float64)
    names = [str(n) for n in d["names"]]
    n_expected = n_train_expected + n_test_expected
    if X.shape[0] != n_expected:
        raise ValueError(
            f"Mordred cache rows {X.shape[0]} != expected "
            f"{n_train_expected}+{n_test_expected}={n_expected}"
        )
    X_train = X[:n_train_expected]
    X_test = X[n_train_expected:]
    return X_train, X_test, names


def _clean_columns(X_train, X_test, names):
    """Drop NaN-heavy, zero-variance, duplicated cols; impute + standardize.

    Matches nb2291 cleaning recipe.
    """
    X_all = np.concatenate([X_train, X_test], axis=0).astype(np.float64)
    X_all = np.where(np.isfinite(X_all), X_all, np.nan)
    n_rows, n_cols = X_all.shape
    nan_frac = np.isnan(X_all).mean(axis=0)
    keep_mask = nan_frac <= NAN_FRAC_MAX
    print(f"[clean] NaN-frac<= {NAN_FRAC_MAX}: keep {keep_mask.sum()}/{n_cols}")

    col_med = np.nanmedian(X_all[:, keep_mask], axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    X_kept = X_all[:, keep_mask]
    for j in range(X_kept.shape[1]):
        bad = ~np.isfinite(X_kept[:, j])
        if bad.any():
            X_kept[bad, j] = col_med[j]

    var = X_kept.var(axis=0)
    var_mask = var > LOW_VAR_THRESH
    X_kept = X_kept[:, var_mask]
    print(f"[clean] var>{LOW_VAR_THRESH}: keep {var_mask.sum()}/{len(var_mask)}")

    sig = X_kept.std(axis=0) + 1e-12
    X_norm = (X_kept - X_kept.mean(axis=0)) / sig
    seen = {}
    uniq_idx = []
    for j in range(X_norm.shape[1]):
        key = tuple(np.round(X_norm[:, j], 6))
        if key not in seen:
            seen[key] = j
            uniq_idx.append(j)
    X_kept = X_kept[:, uniq_idx]
    print(f"[clean] dedup cols: keep {len(uniq_idx)}")

    mu = X_kept.mean(axis=0)
    sd = X_kept.std(axis=0) + 1e-12
    X_kept = (X_kept - mu) / sd

    surviving_full_idx = np.where(keep_mask)[0][var_mask][uniq_idx]
    surviving_names = [names[i] for i in surviving_full_idx]
    X_train_clean = X_kept[:len(X_train)].astype(np.float32)
    X_test_clean = X_kept[len(X_train):].astype(np.float32)
    return X_train_clean, X_test_clean, surviving_names


def _residual_cv_one_seed(X, residual, scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_RFE_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr, va in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr], residual[tr])
        oof[va] = mdl.predict(X[va])
    return oof


def _eval_subset(X, residual, anchor, y, scaffolds, kf_seed):
    oof = _residual_cv_one_seed(X, residual, scaffolds, kf_seed)
    return float(rae(y, anchor + oof))


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def _build_K_deep30_bag(
    X_unb_K, X_te_K, residual, anchor, te_anchor_513,
    scaffolds, n_unb, n_test, seeds,
):
    """Build deep-30 bag-mean OOF (253,) + te (513,) on chemprop_aux residual."""
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    y_unb_proxy = anchor + residual
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof = _residual_cv_one_seed(X_unb_K, residual, scaffolds, s)
        pred_unb_s = anchor + resid_oof
        sum_unb += pred_unb_s
        per_seed_rae.append(float(rae(y_unb_proxy, pred_unb_s)))
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        pred_te_s = te_anchor_513 + te_resid_s
        sum_te += pred_te_s
        if (i % 10) == 0 or i == len(seeds) - 1:
            print(
                f"      [Mord_K{K_TARGET}] seed={s:4d}  "
                f"rae={per_seed_rae[-1]:.4f}  "
                f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})"
            )
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    return bag_oof_unb, bag_te_513, per_seed_rae


def _blend_quantile_conditional_cross_substrate(
    p_mord: np.ndarray,
    p_3080: np.ndarray,
    q_thr: float,
) -> np.ndarray:
    """Per-row hard-split cross-substrate blend.

    Mord K=20 pred <= q_thr -> (W_MORD_LOW=0.7, W_3080_LOW=0.3)
    Mord K=20 pred  > q_thr -> (W_MORD_HIGH=0.3, W_3080_HIGH=0.7)
    """
    low_mask = p_mord <= q_thr
    out = np.empty_like(p_mord, dtype=np.float64)
    out[low_mask] = (
        W_MORD_LOW * p_mord[low_mask] + W_3080_LOW * p_3080[low_mask]
    )
    out[~low_mask] = (
        W_MORD_HIGH * p_mord[~low_mask] + W_3080_HIGH * p_3080[~low_mask]
    )
    return out


def _run_one_kf_seed_blend(
    p_mord_unb: np.ndarray,
    p_3080_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-fold quantile-conditional blend at one kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q_thrs = []
    fold_high_share = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        q_thr = float(np.quantile(p_mord_unb[tr_loc], Q_CUT))
        fold_q_thrs.append(q_thr)
        val_p_mord = p_mord_unb[va_loc]
        val_p_3080 = p_3080_unb[va_loc]
        val_pred = _blend_quantile_conditional_cross_substrate(
            val_p_mord, val_p_3080, q_thr,
        )
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_high_share.append(float(np.mean(val_p_mord > q_thr)))

    if np.isnan(oof_blend).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_q_thr_mean": float(np.mean(fold_q_thrs)),
        "fold_q_thr_std": float(np.std(fold_q_thrs, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- Mordred-ONLY K={K_TARGET} substrate-change vs nb3080 "
        "(5-way K18-K19 anchor)"
    )
    print(
        f"          fresh seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
        f"(n={len(RESID_SEEDS_DEEP)})"
    )
    print(
        f"          per-fold blend: q_cut={Q_CUT}, "
        f"low (Mord<=q):  (Mord={W_MORD_LOW}, 3080={W_3080_LOW})  "
        f"high (Mord>q): (Mord={W_MORD_HIGH}, 3080={W_3080_HIGH})"
    )
    print(
        f"          kf_seeds = {KF_SEEDS}  "
        f"gate: <{GATE_BETTER_THAN_NB3080:.4f} BETTER_THAN_NB3080"
    )
    print("=" * 78)

    # -- Load anchor, truth, scaffolds, test ---------------------------------
    tr = load_train()
    te = load_test()
    tr_smiles = tr["smiles"].astype(str).tolist()
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )
    n_train, n_test = len(tr_smiles), len(te_smiles)
    print(f"[load] n_train={n_train}  n_test={n_test}")

    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(
        f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} (residual base)"
    )

    p_3080_unb = np.load(NB3080_OOF_PATH).astype(np.float64)
    p_3080_te = np.load(NB3080_TE_PATH).astype(np.float64)
    if p_3080_unb.shape != (n_unb,):
        raise ValueError(f"nb3080 oof shape {p_3080_unb.shape} != ({n_unb},)")
    if p_3080_te.shape != (n_test,):
        raise ValueError(f"nb3080 te shape {p_3080_te.shape} != ({n_test},)")
    rae_3080 = float(rae(y_unb, p_3080_unb))
    print(
        f"[load] nb3080 anchor: oof_RAE={rae_3080:.4f} "
        f"(ref {REF_NB3080:.4f})  te_mean={p_3080_te.mean():.3f}"
    )

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # -- Load + clean Mordred descriptors -----------------------------------
    print("\n" + "-" * 78)
    print(
        f"STEP 1: load Mordred cache ({n_train}+{n_test} rows x 1613 dim) "
        "and clean"
    )
    print("-" * 78)
    X_mord_train, X_mord_test, mord_names = _load_mordred_cache(
        n_train_expected=n_train, n_test_expected=n_test,
    )
    print(
        f"   Mordred train={X_mord_train.shape}  test={X_mord_test.shape}  "
        f"n_names={len(mord_names)}"
    )
    X_train_clean, X_test_clean, surv_names = _clean_columns(
        X_mord_train, X_mord_test, mord_names,
    )
    n_dim_clean = X_train_clean.shape[1]
    X_unb_clean = X_test_clean[unb_idx]
    print(
        f"[clean] surviving Mordred dims = {n_dim_clean}  "
        f"X_unb_clean={X_unb_clean.shape}  X_te_clean={X_test_clean.shape}"
    )

    if n_dim_clean < K_TARGET:
        raise RuntimeError(
            f"only {n_dim_clean} clean Mordred dims < K_target={K_TARGET}"
        )

    # -- Greedy backward RFE on Mordred-only -> K=20 ------------------------
    print("\n" + "-" * 78)
    print(
        f"STEP 2: greedy backward RFE {n_dim_clean} -> {K_TARGET} "
        f"(1-seed search kf={KF_SEED_RFE})"
    )
    print("-" * 78)
    surviving = list(range(n_dim_clean))
    base_rae = _eval_subset(
        X_unb_clean[:, surviving], residual, anchor, y_unb,
        unb_scaffolds, KF_SEED_RFE,
    )
    print(f"[rfe] start K={len(surviving)}  RAE={base_rae:.4f}")

    rfe_trajectory = [{"K": len(surviving), "rae": base_rae}]
    iter_count = 0
    rfe_t0 = time.time()
    while len(surviving) > K_TARGET:
        iter_count += 1
        best_drop_j = None
        best_drop_rae = float("inf")
        # sub-sample candidates for large K to stay in budget (same as nb2291)
        if len(surviving) > 200:
            rng = np.random.default_rng(13 + iter_count)
            candidate_pos = rng.choice(len(surviving), size=200, replace=False)
            candidate_pos = np.sort(candidate_pos)
        else:
            candidate_pos = np.arange(len(surviving))
        for pos in candidate_pos:
            trial = surviving[:pos] + surviving[pos + 1:]
            r = _eval_subset(
                X_unb_clean[:, trial], residual, anchor, y_unb,
                unb_scaffolds, KF_SEED_RFE,
            )
            if r < best_drop_rae:
                best_drop_rae = r
                best_drop_j = int(pos)
        dropped_feat_idx = surviving[best_drop_j]
        surviving = surviving[:best_drop_j] + surviving[best_drop_j + 1:]
        rfe_trajectory.append({
            "K": len(surviving),
            "rae": best_drop_rae,
            "dropped_feature_position": int(best_drop_j),
            "dropped_feature_name": surv_names[dropped_feat_idx],
        })
        if iter_count <= 5 or len(surviving) <= K_TARGET + 5 or iter_count % 20 == 0:
            print(
                f"[rfe] iter {iter_count:3d}  K={len(surviving):4d}  "
                f"RAE={best_drop_rae:.4f}  "
                f"dropped={surv_names[dropped_feat_idx][:30]}"
            )
    rfe_wall = time.time() - rfe_t0
    print(
        f"[rfe] done in {rfe_wall:.0f}s  final K={len(surviving)}  "
        f"RAE={rfe_trajectory[-1]['rae']:.4f}"
    )
    surviving_idx_final = surviving
    surviving_names_final = [surv_names[i] for i in surviving]
    X_unb_K = X_unb_clean[:, surviving_idx_final].astype(np.float32)
    X_te_K = X_test_clean[:, surviving_idx_final].astype(np.float32)
    print(
        f"[K{K_TARGET}] X_unb_K={X_unb_K.shape}  X_te_K={X_te_K.shape}  "
        f"first features: {surviving_names_final[:5]}"
    )

    # -- Build Mordred K=20 deep-30 bag-mean (on chemprop_aux residual) ----
    print("\n" + "-" * 78)
    print(
        f"STEP 3: Mordred-only K={K_TARGET} deep-30 bag-mean OOF + te "
        f"({len(RESID_SEEDS_DEEP)} fresh seeds)"
    )
    print("-" * 78)
    p_mord_unb_30, p_mord_te_30, per_seed_rae_mord = _build_K_deep30_bag(
        X_unb_K, X_te_K, residual, anchor, te_anchor_513,
        unb_scaffolds, n_unb, n_test, RESID_SEEDS_DEEP,
    )
    mord_K_oof_rae = float(rae(y_unb, p_mord_unb_30))
    print(
        f"   [Mord_K{K_TARGET}] 30-seed BAG-MEAN OOF RAE = {mord_K_oof_rae:.4f}  "
        f"per_seed_mean={np.mean(per_seed_rae_mord):.4f}  "
        f"std={np.std(per_seed_rae_mord, ddof=1):.4f}"
    )
    print(
        f"   [Mord_K{K_TARGET}] te_mean={p_mord_te_30.mean():.3f}  "
        f"te_std={p_mord_te_30.std():.3f}"
    )

    # Leak sanity
    leak_frac_mord = float(
        np.mean(np.isclose(p_mord_unb_30, y_unb, atol=1e-6))
    )
    leak_frac_3080 = float(
        np.mean(np.isclose(p_3080_unb, y_unb, atol=1e-6))
    )
    print(
        f"   leak frac: Mord_K{K_TARGET}={leak_frac_mord:.4f}  "
        f"3080={leak_frac_3080:.4f}"
    )
    corr = float(np.corrcoef(p_mord_unb_30, p_3080_unb)[0, 1])
    print(f"   oof pairwise corr(Mord_K{K_TARGET}, nb3080) = {corr:.4f}")

    # -- Per-fold quantile-conditional cross-substrate blend ----------------
    print("\n" + "-" * 78)
    print(
        f"STEP 4: per-fold quantile-conditional cross-substrate blend "
        f"(5 kf_seeds {KF_SEEDS})"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_kf_seed_blend(
            p_mord_unb_30, p_3080_unb, y_unb, unb_scaffolds, s,
        )
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_q_thr_mean": round(res["fold_q_thr_mean"], 4),
            "fold_q_thr_std": round(res["fold_q_thr_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"q_thr_mean={res['fold_q_thr_mean']:.3f}  "
            f"high_share={res['fold_high_share_mean']:.2f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    median_rae = float(np.median(arr))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} kf_seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"\n   ref nb3080 anchor                  = {REF_NB3080:.4f}")
    print(f"   ref K18 deep-30                    = {REF_K18:.4f}")
    print(f"   ref K19 deep-30                    = {REF_K19:.4f}")
    print(f"   ref nb2291 Mord-K20 (chemprop_aux) = {REF_NB2291_K20_MORDRED:.4f}")
    print(f"   ref nb2171 prior post-hoc top      = {REF_NB2171:.4f}")
    print(
        f"\n   delta vs nb3080  = {mean_rae - REF_NB3080:+.4f}  "
        f"(gate: < 0.0000 -> BETTER_THAN_NB3080)"
    )

    # -- Deploy: q_thr from FULL 253 Mord_K20 OOF, blend te ----------------
    deploy_q_thr = float(np.quantile(p_mord_unb_30, Q_CUT))
    te_pred = _blend_quantile_conditional_cross_substrate(
        p_mord_te_30, p_3080_te, deploy_q_thr,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    te_low_share = float(np.mean(p_mord_te_30 <= deploy_q_thr))
    print(
        f"\n   deploy q_thr (full Mord_K{K_TARGET} OOF q{Q_CUT}) = "
        f"{deploy_q_thr:.4f}"
    )
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate ---------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER_THAN_NB3080:
        verdict = "BETTER_THAN_NB3080"
        ladder_action = (
            f"PROMOTE. nb3110 Mordred-only K={K_TARGET} cross-substrate blend "
            f"5-kf mean {mean_rae:.4f} beats nb3080 anchor {REF_NB3080:.4f} "
            f"({mean_rae - REF_NB3080:+.4f}). Substrate change (Mordred-only "
            f"vs 5-way mix) carried orthogonal residual information. NEW "
            f"PRIMARY-1 candidate; recommend deep-30 wide-seed verification "
            f"before locking."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REPORT. nb3110 Mordred-only K={K_TARGET} cross-substrate blend "
            f"5-kf mean {mean_rae:.4f} does NOT beat nb3080 "
            f"({REF_NB3080:.4f}, delta {mean_rae - REF_NB3080:+.4f}). "
            f"Mordred-only substrate did not provide net-positive cross-blend "
            f"information at K={K_TARGET} with these blend weights. Keep "
            f"nb3080 PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -----------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    # Also save per-K Mordred-only artifacts for downstream reuse
    mord_oof_path = DATA_PROCESSED / f"{TAG}_mord_K{K_TARGET}_30seed_oof.npy"
    mord_te_path = DATA_PROCESSED / f"{TAG}_mord_K{K_TARGET}_30seed_te.npy"
    np.save(mord_oof_path, p_mord_unb_30.astype(np.float32))
    np.save(mord_te_path, p_mord_te_30.astype(np.float32))
    print(f"   [save] {mord_oof_path}")
    print(f"   [save] {mord_te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_mordred_only_K{K_TARGET}.csv"
    if verdict == "BETTER_THAN_NB3080":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "method": (
            "mordred_only_K20_RFE_chemprop_aux_residual_deep30_"
            "crossblend_quantile_conditional_with_nb3080"
        ),
        "paradigm": "substrate_change_mordred_only_not_5way_mix",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "mordred_total_descriptors": int(X_mord_train.shape[1]),
        "mordred_surviving_after_clean": int(n_dim_clean),
        "nan_frac_max": NAN_FRAC_MAX,
        "low_var_thresh": LOW_VAR_THRESH,
        "k_target": K_TARGET,
        "rfe_kf_seed": KF_SEED_RFE,
        "rfe_iterations": int(iter_count),
        "rfe_wall_sec": round(rfe_wall, 1),
        "rfe_start_rae": round(base_rae, 4),
        "rfe_end_rae": round(rfe_trajectory[-1]["rae"], 4),
        "rfe_trajectory_first10": rfe_trajectory[:10],
        "rfe_trajectory_last10": rfe_trajectory[-10:],
        "mord_K_surviving_names": surviving_names_final,
        "mord_K_surviving_idx_in_clean": [
            int(i) for i in surviving_idx_final
        ],
        "mord_K_per_seed_rae": [round(v, 4) for v in per_seed_rae_mord],
        "mord_K_30seed_bagmean_oof_rae": round(mord_K_oof_rae, 4),
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "resid_seeds_deep_n": len(RESID_SEEDS_DEEP),
        "oof_pairwise_corr_mord_vs_3080": round(corr, 4),
        "leak_eq_truth_frac_mord_K": round(leak_frac_mord, 4),
        "leak_eq_truth_frac_3080": round(leak_frac_3080, 4),
        "blend_q_cut": Q_CUT,
        "blend_w_mord_low": W_MORD_LOW,
        "blend_w_3080_low": W_3080_LOW,
        "blend_w_mord_high": W_MORD_HIGH,
        "blend_w_3080_high": W_3080_HIGH,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_seeds": int(n_s),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "ref_nb3080": REF_NB3080,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2291_K20_mordred": REF_NB2291_K20_MORDRED,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3080": round(mean_rae - REF_NB3080, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "te_low_share": round(te_low_share, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "mord_K_oof_path": str(mord_oof_path),
        "mord_K_te_path": str(mord_te_path),
        "submission_csv": (
            str(sub_csv) if verdict == "BETTER_THAN_NB3080" else None
        ),
        "gate_better_than_nb3080": GATE_BETTER_THAN_NB3080,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mord_K{K_TARGET} deep-30 OOF RAE = {mord_K_oof_rae:.4f}")
    print(
        f"   blend mean ({n_s} kf_seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}"
    )
    print(f"   delta vs nb3080              = {mean_rae - REF_NB3080:+.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   wall                         = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mord_K_30seed_bagmean_oof_rae",
        "mean_rae", "std_rae", "median_rae",
        "delta_vs_nb3080",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  rfe_end_rae: {res.get('rfe_end_rae')}")
    print(f"  mord_K_surviving_names[:10]: {res.get('mord_K_surviving_names')[:10]}")
