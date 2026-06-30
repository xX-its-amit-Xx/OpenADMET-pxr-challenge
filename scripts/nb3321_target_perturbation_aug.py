"""nb3321 -- Train-time TARGET PERTURBATION augmentation on K=18 residual-LGBM.

NEW PARADIGM (label-noise data augmentation via target jitter):
    Every prior K=18 deep-30 build fits the residual-LGBM on the EXACT residual
    (y_unb - anchor) for all 30 bag-seeds; the only inter-seed variation is the
    LGBM random_state (feature/row subsampling). This collapses the 30-seed bag
    to a low-variance point estimate but does nothing to regularize toward
    label-noise robustness.

    nb3321 adds small per-seed Gaussian jitter to the TRAINING target before
    fitting each bag member:

        for bag-seed s in {3001..3030}:
            eps_s   ~ N(0, JITTER_SIGMA^2)  over the 253 unblind rows
                      (RNG seeded by s -> reproducible, DIFFERENT per seed)
            y_jit_s = y_unb + eps_s
            resid_s = y_jit_s - anchor          # anchor = chemprop_aux[unb]
            fit residual-LGBM on resid_s (5-fold cross-fit for OOF;
                full-fit for te); pred_unb_s = anchor + resid_oof_s.

    The bag-mean over 30 seeds averages out the zero-mean jitter, but each tree
    ensemble has been pushed to NOT over-fit the exact label values -- a soft
    Tikhonov-on-targets regularizer that may improve generalization to the
    novel-scaffold OOD tail (the dominant PXR failure mode).

    Jitter is applied to TRAINING targets ONLY. OOF predictions are always
    scored against the TRUE y_unb, and within each seed the cross-fit never
    lets a fold-val row's (jittered) target reach its own prediction -> clean,
    no leak. JITTER_SIGMA = 0.10 ~ half the 0.24 log-unit assay noise floor.

    Feature path + LGBM recipe are byte-identical to nb2960 K=18 (chemprop_aux
    anchor + residual-LGBM on the 18-col SHAP/RFE slice of the 117-col 5-way
    feature matrix). Only the training target is perturbed.

POST-HOC: nb3173-style LEARNED per-fold clip on the 30-seed bag-mean OOF:
    Per outer fold, inner grid-search (q_low, q_high) minimizing fold-TRAIN RAE,
    then clip fold-VAL to those learned (lo, hi). 15 fresh kf_seeds {1216..1230}.

GATE (on 15-seed PER-FOLD-MEAN of the post-clip val RAE):
    per_fold_mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

References:
    nb3173 learned-clip on nb3080 wide-bag   = 0.4422 (per-fold-mean ~0.4423)
    nb3170 fixed q05/q95 on nb3080           = 0.4437
    nb3080 wide-seed mean                    = 0.4475
    nb2960 K=18 deep-30 OOF (no jitter)      = 0.4536
    nb2171 5-anchor pyramid ceiling          = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json            (k18_idx_in_117col)
    data/processed/nb2960_K18_30seed_oof.npy      (no-jitter parity check)
    + all 117-col feature caches (see nb2960.build_117col_feature_matrix)

Outputs:
    data/processed/nb3321_summary.json
    data/processed/nb3321_pred_oof.npy   (253,) float32 -- median-seed post-clip OOF
    data/processed/te_nb3321.npy         (513,) float32 -- deploy te (clipped)
    submissions/nb3321_target_perturbation_aug.csv  (only on BETTER verdict)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import RDLogger
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# -- Import nb2960's feature builder + LGBM recipe verbatim --------------------
_NB2960_PATH = Path(__file__).resolve().parent / "nb2960_fresh_K_rebuild_for_nb2943.py"
_spec = importlib.util.spec_from_file_location("nb2960_mod", _NB2960_PATH)
nb2960 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb2960)  # noqa: E402

TAG = "nb3321"
PARENT_TAG = "nb2960_K18"

# -- Anchor + residual params (IDENTICAL to nb2960) ---------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"
CACHED_K18_NOJITTER_OOF = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))   # 30 fresh seeds {3001..3030}

# -- Target perturbation -------------------------------------------------------
JITTER_SIGMA = 0.10   # std of per-seed Gaussian label noise (log-units)

# -- Learned per-fold clip grid (mirror nb3173) -------------------------------
Q_LOW_GRID = [0.01, 0.02, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- CV protocol (post-clip evaluation) ---------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 FRESH seeds {1216..1230}

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423   # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_NB3173 = 0.4422
REF_NB3170_FIXED = 0.4437
REF_NB3080 = 0.4475
REF_K18_DEEP30 = 0.4536
REF_NB2171 = 0.4682


def _residual_cross_fit_one_seed_jit(X, residual_jit, seed):
    """5-fold cross-fit residual-LGBM on a JITTERED residual.

    Fold partition + LGBM random_state both keyed on `seed` (matches nb2960).
    Returns OOF predictions of the jittered residual (253,).
    """
    n = len(residual_jit)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**nb2960._lgbm_params(seed))
        mdl.fit(X[tr_loc], residual_jit[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_full_then_predict_te_jit(X_unb, residual_jit, X_te, seed):
    mdl = lgb.LGBMRegressor(**nb2960._lgbm_params(seed))
    mdl.fit(X_unb, residual_jit)
    return mdl.predict(X_te).astype(np.float32)


def build_K18_jitter_bag(X_unb_K, X_te_K, anchor, anchor_te_513, y_unb,
                         seeds, sigma, n_unb, n_test):
    """Deep bag with per-seed target jitter.

    For each bag-seed s:
        eps_s   ~ N(0, sigma^2)  (RNG seeded by s)
        y_jit   = y_unb + eps_s
        resid   = y_jit - anchor
        resid_oof = 5-fold cross-fit(resid)         -> pred_unb_s = anchor + resid_oof
        resid_te  = full-fit(resid) -> predict X_te  -> pred_te_s  = anchor_te + resid_te

    Returns:
        bag_oof_unb : (n_unb,) float64   -- mean over seeds of pred_unb_s
        bag_te_513  : (n_test,) float64  -- mean over seeds of pred_te_s
        per_seed_rae: list[float]        -- per-seed pred_unb_s RAE vs TRUE y_unb
        eps_stats   : dict               -- realized jitter diagnostics
    """
    sum_unb = np.zeros(n_unb, dtype=np.float64)
    sum_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    realized_sigmas = []
    for i, s in enumerate(seeds):
        ts = time.time()
        rng = np.random.default_rng(s)
        eps = rng.normal(0.0, sigma, size=n_unb)
        realized_sigmas.append(float(eps.std()))
        y_jit = y_unb + eps
        resid_jit = y_jit - anchor

        resid_oof = _residual_cross_fit_one_seed_jit(X_unb_K, resid_jit, s)
        pred_unb_s = anchor + resid_oof          # evaluate vs TRUE y_unb
        sum_unb += pred_unb_s
        per_seed_rae.append(float(rae(y_unb, pred_unb_s)))

        resid_te = _train_full_then_predict_te_jit(X_unb_K, resid_jit, X_te_K, s)
        pred_te_s = anchor_te_513 + resid_te
        sum_te += pred_te_s

        if (i % 10) == 0 or i == len(seeds) - 1:
            print(f"      [K18+jit] seed={s:4d}  rae={per_seed_rae[-1]:.4f}  "
                  f"eps_std={realized_sigmas[-1]:.3f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(seeds)})")
    bag_oof_unb = sum_unb / len(seeds)
    bag_te_513 = sum_te / len(seeds)
    eps_stats = {
        "target_sigma": sigma,
        "realized_sigma_mean": float(np.mean(realized_sigmas)),
        "realized_sigma_min": float(np.min(realized_sigmas)),
        "realized_sigma_max": float(np.max(realized_sigmas)),
    }
    return bag_oof_unb, bag_te_513, per_seed_rae, eps_stats


def _pick_best_clip(y_tr, pred_tr):
    """Inner grid search: (q_low*, q_high*) minimizing fold-train RAE (nb3173)."""
    best_rae = np.inf
    best_ql, best_qh = Q_LOW_GRID[0], Q_HIGH_GRID[-1]
    best_lo = float(np.quantile(y_tr, best_ql))
    best_hi = float(np.quantile(y_tr, best_qh))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(y_tr, ql))
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(y_tr, qh))
            if hi <= lo:
                continue
            clipped = np.clip(pred_tr, lo, hi)
            r = float(rae(y_tr, clipped))
            if r < best_rae:
                best_rae = r
                best_ql, best_qh, best_lo, best_hi = ql, qh, lo, hi
    return best_ql, best_qh, best_lo, best_hi


def _run_clip_one_seed(pred_base, y_unb, unb_scaffolds, kf_seed):
    """nb3173 learned-clip at one kf_seed. Returns pooled + per-fold-mean RAE."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_ql, fold_qh = [], []
    n_clip_lo = n_clip_hi = 0
    for tr_loc, va_loc in splits:
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], pred_base[tr_loc])
        fold_ql.append(ql)
        fold_qh.append(qh)
        val_pred = pred_base[va_loc]
        n_clip_lo += int(np.sum(val_pred < lo))
        n_clip_hi += int(np.sum(val_pred > hi))
        clipped = np.clip(val_pred, lo, hi)
        oof_clip[va_loc] = clipped
        fold_val_raes.append(float(rae(y_unb[va_loc], clipped)))
    if np.isnan(oof_clip).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits incomplete")
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": float(rae(y_unb, oof_clip)),
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "n_clipped_lo": n_clip_lo,
        "n_clipped_hi": n_clip_hi,
        "oof": oof_clip,
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TARGET PERTURBATION aug on K=18 residual-LGBM "
          f"(jitter sigma={JITTER_SIGMA})")
    print(f"          bag seeds = {len(RESID_SEEDS_DEEP)} "
          f"{{{RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]}}}")
    print(f"          post-hoc  = nb3173 learned per-fold clip")
    print(f"          clip kf_seeds = {len(KF_SEEDS)} "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          GATE: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
    print("=" * 78)

    # -- Load truth, anchor, scaffolds ---------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[anchor] chemprop_aux te[unb] RAE = {rae_anchor:.4f}")

    # -- Build 117-col matrix, slice K=18 ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: rebuild 117-col matrix + slice K=18 (nb2960 recipe)")
    print("-" * 78)
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"   K=18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    X_te_full, chembl_pool_size = nb2960.build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   X_unb_K = {X_unb_K.shape}   X_te_K = {X_te_K.shape}")

    # -- Parity check: no-jitter rebuild should match nb2960 cached K18 -------
    print("\n" + "-" * 78)
    print("STEP 2: parity check -- rebuild K=18 WITHOUT jitter (sigma=0)")
    print("-" * 78)
    pc_oof, _pc_te, pc_seed_rae, _ = build_K18_jitter_bag(
        X_unb_K, X_te_K, anchor, te_anchor_513, y_unb,
        RESID_SEEDS_DEEP, 0.0, n_unb, n_test,
    )
    pc_rae = float(rae(y_unb, pc_oof))
    cached = np.load(CACHED_K18_NOJITTER_OOF).astype(np.float64)
    cached_rae = float(rae(y_unb, cached))
    parity_max_abs = float(np.max(np.abs(pc_oof - cached)))
    print(f"   no-jitter rebuild RAE = {pc_rae:.4f}  "
          f"(nb2960 cached = {cached_rae:.4f})")
    print(f"   max|rebuild - cached| = {parity_max_abs:.4f}  "
          f"(expect ~0; confirms faithful K=18 reconstruction)")

    # -- Build jittered deep-30 bag ------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: jittered deep-30 bag (sigma={JITTER_SIGMA}, "
          f"per-seed N(0,sigma^2) on targets)")
    print("-" * 78)
    bag_oof, bag_te, per_seed_rae, eps_stats = build_K18_jitter_bag(
        X_unb_K, X_te_K, anchor, te_anchor_513, y_unb,
        RESID_SEEDS_DEEP, JITTER_SIGMA, n_unb, n_test,
    )
    bag_rae = float(rae(y_unb, bag_oof))
    print(f"   jittered 30-seed bag-mean OOF RAE = {bag_rae:.4f}")
    print(f"   per-seed RAE mean={np.mean(per_seed_rae):.4f}  "
          f"std={np.std(per_seed_rae, ddof=1):.4f}  "
          f"[{min(per_seed_rae):.4f}, {max(per_seed_rae):.4f}]")
    print(f"   realized eps_std mean={eps_stats['realized_sigma_mean']:.3f}")
    print(f"   delta vs no-jitter bag = {bag_rae - pc_rae:+.4f}")
    print(f"   delta vs K18 deep-30 ref ({REF_K18_DEEP30}) = "
          f"{bag_rae - REF_K18_DEEP30:+.4f}")

    leak_eq = float(np.mean(np.isclose(bag_oof, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN: {leak_eq:.1%} bag rows == truth -- possible leak")

    # -- Learned per-fold clip over 15 fresh kf_seeds ------------------------
    print("\n" + "-" * 78)
    print(f"STEP 4: nb3173 learned per-fold clip, {len(KF_SEEDS)} fresh "
          f"kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    all_ql, all_qh = [], []
    for s in KF_SEEDS:
        res = _run_clip_one_seed(bag_oof, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_ql.extend(res["fold_ql"])
        all_qh.extend(res["fold_qh"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
              f"per_fold_mean={res['per_fold_val_rae_mean']:.4f}  "
              f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})")

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    pfm_arr = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(pfm_arr)

    pfm_mean = float(pfm_arr.mean())
    pfm_std = float(pfm_arr.std(ddof=1)) if n_s > 1 else 0.0
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = pfm_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14
    ci_low = pfm_mean - t_mult * sem
    ci_high = pfm_mean + t_mult * sem

    ql_counter = Counter(all_ql)
    qh_counter = Counter(all_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} kf_seeds)")
    print("-" * 78)
    print(f"   per-fold-mean   = {pfm_mean:.4f} +/- {pfm_std:.4f}  "
          f"(GATE metric)")
    print(f"   95% CI          = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled-mean     = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   per-fold min/max= [{pfm_arr.min():.4f}, {pfm_arr.max():.4f}]")
    print(f"   ql_mode={ql_mode}  qh_mode={qh_mode}  "
          f"ql_dist={dict(ql_counter)}  qh_dist={dict(qh_counter)}")
    print(f"\n   delta vs nb3173 ({REF_NB3173})       = "
          f"{pfm_mean - REF_NB3173:+.4f}")
    print(f"   delta vs nb3170 fixed ({REF_NB3170_FIXED}) = "
          f"{pfm_mean - REF_NB3170_FIXED:+.4f}")

    # -- Deploy te: pick clip on FULL 253, apply to bag_te -------------------
    print("\n" + "-" * 78)
    print("STEP 5: deploy te -- learned clip on full 253, apply to bag te")
    print("-" * 78)
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, bag_oof)
    te_pred = np.clip(bag_te, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(bag_te < deploy_lo))
    n_te_hi = int(np.sum(bag_te > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy clip = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
          f"({deploy_lo:.3f}, {deploy_hi:.3f})")
    print(f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513")
    print(f"   te(513): mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f}  max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF for save (ranked by per-fold-mean)
    med_idx = int(np.argsort(pfm_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_idx]
    oof_for_save = oof_stack[med_idx].astype(np.float32)
    print(f"   median seed = {median_seed} "
          f"(per_fold_mean={pfm_arr[med_idx]:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if pfm_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3321 target-perturbation aug (sigma="
            f"{JITTER_SIGMA}) per-fold-mean {pfm_mean:.4f} clears BETTER gate "
            f"{GATE_BETTER:.4f}. Label-noise augmentation on the K=18 "
            f"residual-LGBM + learned clip beats the nb3173 post-hoc ceiling "
            f"({pfm_mean - REF_NB3173:+.4f} vs {REF_NB3173}). Jitter "
            f"regularizes toward novel-scaffold OOD robustness. Re-verify "
            f"deep-30 / wider sigma sweep before PRIMARY-1 swap."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3321 target-perturbation aug (sigma={JITTER_SIGMA}) "
            f"per-fold-mean {pfm_mean:.4f} fails BETTER gate {GATE_BETTER:.4f} "
            f"({pfm_mean - GATE_BETTER:+.4f}). Adding N(0,{JITTER_SIGMA}^2) "
            f"label noise to K=18 training targets does not improve "
            f"generalization on the 253-unblind cross-fit; the 30-seed bag "
            f"already averages out the zero-mean jitter, leaving the post-hoc "
            f"ceiling at the nb3173 value {REF_NB3173}. Confirms cycle-169 "
            f"finding that post-hoc-blend gains require substrate change, not "
            f"target/operator perturbation. Keep current PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_target_perturbation_aug.csv"
    if verdict == "BETTER":
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
        "parent_tag": PARENT_TAG,
        "method": "target_perturbation_label_noise_aug_K18_then_nb3173_learned_clip",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": round(rae_anchor, 4),
        "anchor_pre_unblind": True,
        "jitter_sigma": JITTER_SIGMA,
        "jitter_per_seed": True,
        "realized_eps_stats": eps_stats,
        "K18_idx_in_117col": K18_idx.tolist(),
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "resid_folds": RESID_FOLDS,
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        # parity
        "nojitter_rebuild_rae": round(pc_rae, 4),
        "nb2960_cached_K18_rae": round(cached_rae, 4),
        "parity_max_abs_diff": round(parity_max_abs, 6),
        "nojitter_per_seed_rae_mean": round(float(np.mean(pc_seed_rae)), 4),
        # jittered bag
        "jitter_bag_oof_rae": round(bag_rae, 4),
        "jitter_per_seed_rae_mean": round(float(np.mean(per_seed_rae)), 4),
        "jitter_per_seed_rae_std": round(float(np.std(per_seed_rae, ddof=1)), 4),
        "delta_jitter_minus_nojitter_bag": round(bag_rae - pc_rae, 4),
        "delta_jitter_bag_vs_K18_deep30": round(bag_rae - REF_K18_DEEP30, 4),
        "bag_leak_eq_truth_frac": round(leak_eq, 4),
        # post-clip aggregate
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "per_fold_mean": round(pfm_mean, 4),
        "per_fold_std": round(pfm_std, 4),
        "per_fold_sem": round(sem, 4),
        "per_fold_ci95_low": round(ci_low, 4),
        "per_fold_ci95_high": round(ci_high, 4),
        "per_fold_min": round(float(pfm_arr.min()), 4),
        "per_fold_max": round(float(pfm_arr.max()), 4),
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        # deploy
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        # refs + gate
        "ref_nb3173": REF_NB3173,
        "delta_vs_nb3173": round(pfm_mean - REF_NB3173, 4),
        "ref_nb3170_fixed": REF_NB3170_FIXED,
        "delta_vs_nb3170_fixed": round(pfm_mean - REF_NB3170_FIXED, 4),
        "ref_nb3080": REF_NB3080,
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_nb2171": REF_NB2171,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   jitter sigma          = {JITTER_SIGMA}")
    print(f"   jitter bag OOF RAE    = {bag_rae:.4f} "
          f"(no-jitter {pc_rae:.4f}, delta {bag_rae - pc_rae:+.4f})")
    print(f"   per-fold-mean ({n_s} kf) = {pfm_mean:.4f} +/- {pfm_std:.4f}  "
          f"[GATE {GATE_BETTER:.4f}]")
    print(f"   delta vs nb3173       = {pfm_mean - REF_NB3173:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "jitter_sigma", "nojitter_rebuild_rae", "nb2960_cached_K18_rae",
        "parity_max_abs_diff", "jitter_bag_oof_rae",
        "delta_jitter_minus_nojitter_bag",
        "per_fold_mean", "per_fold_std", "per_fold_ci95_low", "per_fold_ci95_high",
        "delta_vs_nb3173", "ql_mode", "qh_mode",
        "deploy_lo", "deploy_hi", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
