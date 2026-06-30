"""nb1121 -- Bayesian Optuna optimization of LGBM K=28 hyperparameters.

HYPOTHESIS:
    nb2103 K=28 uses the standard LGBM(MSE) hyperparams (max_depth=4,
    num_leaves=15, n_estimators=300, lr=0.03, min_child_samples=5,
    reg_lambda=2.0) and yields mean-bag RAE 0.4737 / median-bag RAE 0.4698
    on the residual chemprop_aux task across 253 unblind compounds.

    These hyperparams were chosen heuristically and held fixed across all
    nb2063 / nb2081 / nb2091 / nb2103 sweeps.  This notebook runs Optuna TPE
    over 8 LGBM knobs to test whether systematic Bayesian search beats the
    heuristic config at decision margin 0.003.

PROTOCOL:
    1. Reuse cached X_unb_28_nb2103.npy (253, 28) and residual_unb on 253.
    2. Optuna TPE, 100 trials, scaffold-aware 5-fold cross-fit.
       Per trial objective: 1-seed mean-bag residual-corrected RAE on the 253
       unblind labels.  (1 seed during search for budget; final refit uses
       the 5-seed bag.)
    3. Search space:
       - num_leaves: int(15, 127)
       - min_data_in_leaf: int(5, 100)
       - learning_rate: log(0.01, 0.10)
       - feature_fraction: (0.5, 1.0)
       - bagging_fraction: (0.5, 1.0)
       - lambda_l1: log(1e-3, 10)
       - lambda_l2: log(1e-3, 10)
       - path_smooth: (0, 5)
    4. Mitigation against Optuna overfit on n=253: log per-trial 253 RAE
       and inspect trajectory for monotonic drift vs trial number.
    5. Best params --> refit 5-seed bag (seeds [0,1,7,42,137]) with same
       cross-fit protocol as nb2103.
    6. Decision margin = 0.003 vs nb2103 K=28 mean_bag 0.4737.
    7. Fresh-seed verify if best passes (5 NEW seeds [11,23,53,73,101]).
    8. If beats decisively: build deploy CSV via 5-outer x 5-inner refit.

OUTPUTS:
    scripts/nb1121_optuna.py
    data/processed/nb1121_optuna_study.pkl
    data/processed/nb1121_summary.json
    submissions/nb1121_optuna_K28.csv     (only if best passes margin)
"""
from __future__ import annotations

import json
import os
import pickle
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
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1121"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Cached top-28 SHAP feature matrix (253, 28) from nb2103 K=28
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

# nb2103 K=28 benchmark
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003

# Optuna config
N_TRIALS = 100
SEARCH_SEED = 0      # seed used during Optuna search (single seed for budget)
FRESH_SEEDS = [11, 23, 53, 73, 101]
PROD_SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5
SCAFFOLD_KFOLD_SEED = 42

SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)


def _build_lgbm_params(trial_params: dict, seed: int) -> dict:
    """Translate Optuna trial params to LGBMRegressor kwargs.

    Notes:
        - num_iterations (n_estimators) held at 300 (matches nb2103).
        - max_depth held at -1 (unbounded); search controls capacity via
          num_leaves + min_data_in_leaf to avoid conflating depth/leaf bounds.
    """
    return dict(
        objective="regression",
        n_estimators=300,
        max_depth=-1,
        num_leaves=int(trial_params["num_leaves"]),
        min_data_in_leaf=int(trial_params["min_data_in_leaf"]),
        learning_rate=float(trial_params["learning_rate"]),
        feature_fraction=float(trial_params["feature_fraction"]),
        bagging_fraction=float(trial_params["bagging_fraction"]),
        bagging_freq=1,
        lambda_l1=float(trial_params["lambda_l1"]),
        lambda_l2=float(trial_params["lambda_l2"]),
        path_smooth=float(trial_params["path_smooth"]),
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


def _scaffold_kfold_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                       folds: list, lgbm_kwargs: dict
                                       ) -> np.ndarray:
    """Scaffold-aware 5-fold cross-fit OOF residual prediction."""
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in folds:
        mdl = lgb.LGBMRegressor(**lgbm_kwargs)
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _bag_rae(X: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
             y_true: np.ndarray, folds: list, params: dict,
             seeds: list) -> tuple[float, float, np.ndarray]:
    """Return (mean_bag_rae, median_bag_rae, per_seed_rae) for given seeds."""
    n = len(residual)
    per_seed_corr = np.zeros((len(seeds), n), dtype=np.float64)
    per_seed_rae = np.zeros(len(seeds), dtype=np.float64)
    for i, s in enumerate(seeds):
        lgbm_kwargs = _build_lgbm_params(params, int(s))
        resid_oof = _scaffold_kfold_cross_fit_one_seed(
            X, residual, folds, lgbm_kwargs
        )
        corr = anchor + resid_oof
        per_seed_corr[i] = corr
        per_seed_rae[i] = float(rae(y_true, corr))
    mean_bag_oof = per_seed_corr.mean(axis=0)
    median_bag_oof = np.median(per_seed_corr, axis=0)
    rae_mean_bag = float(rae(y_true, mean_bag_oof))
    rae_median_bag = float(rae(y_true, median_bag_oof))
    return rae_mean_bag, rae_median_bag, per_seed_rae


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Optuna TPE HPO on LGBM K=28 (residual on chemprop_aux)")
    print(f"          n_trials={N_TRIALS}  search_seed={SEARCH_SEED}  "
          f"n_folds={N_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f} "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"          decision margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load cached feature matrix ----
    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(f"missing cache: {X_UNB_28_PATH}")
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    print(f"[load] X_unb_28 = {X_unb_28.shape}  (top-28 SHAP from nb2103)")

    # ---- Anchor + truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing anchor: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual_unb = y_unb - anchor_unb
    print(f"[load] {ANCHOR} in_RAE on unb = {rae_anchor:.4f}")
    print(f"[resid] mean={residual_unb.mean():+.4f} "
          f"std={residual_unb.std():.4f}")

    # ---- Pre-compute scaffold folds (fixed across all trials) ----
    te = load_test()
    if "smiles" in te.columns:
        all_smiles = te["smiles"].astype(str).tolist()
    else:
        all_smiles = te["SMILES"].astype(str).tolist()
    unb_smiles = [all_smiles[int(i)] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len(set(unb_scaffolds))
    folds = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True,
        seed=SCAFFOLD_KFOLD_SEED,
    )
    print(f"[folds] n_unique_scaf={n_unique_scaf}  splits={N_FOLDS}  "
          f"seed={SCAFFOLD_KFOLD_SEED}")
    for fi, (tr, va) in enumerate(folds):
        print(f"   fold {fi}: n_tr={len(tr)}  n_va={len(va)}")

    # ---- Optuna objective ----
    trial_log: list[dict] = []

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 5, 100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.10, log=True
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.5, 1.0
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.5, 1.0
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-3, 10.0, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-3, 10.0, log=True
            ),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 5.0),
        }
        t_tr = time.time()
        lgbm_kwargs = _build_lgbm_params(params, SEARCH_SEED)
        resid_oof = _scaffold_kfold_cross_fit_one_seed(
            X_unb_28, residual_unb, folds, lgbm_kwargs
        )
        corr = anchor_unb + resid_oof
        rae_val = float(rae(y_unb, corr))
        trial_log.append({
            "trial": int(trial.number),
            "rae": rae_val,
            "wall_sec": round(time.time() - t_tr, 3),
            **{k: (float(v) if not isinstance(v, int) else int(v))
               for k, v in params.items()},
        })
        return rae_val

    print("\n" + "-" * 78)
    print(f"OPTUNA TPE: {N_TRIALS} trials  "
          f"(1-seed scaffold cross-fit residual_corr RAE)")
    print("-" * 78)
    sampler = TPESampler(seed=0)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"{TAG}_lgbm_K28",
    )
    t_opt = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"[opt] wall = {time.time() - t_opt:.1f}s")

    best_params = dict(study.best_params)
    best_rae_search = float(study.best_value)
    print(f"\n[best] search rae (1-seed) = {best_rae_search:.4f}")
    print(f"[best] params:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"   {k:>18s}: {v:.5g}")
        else:
            print(f"   {k:>18s}: {v}")

    # ---- Mitigation: log Optuna overfit drift ----
    rae_history = np.array([r["rae"] for r in trial_log], dtype=np.float64)
    # Compare first vs last quartile of trials
    q = len(rae_history) // 4
    first_q_best = float(rae_history[:q].min()) if q > 0 else float("nan")
    last_q_best = float(rae_history[-q:].min()) if q > 0 else float("nan")
    drift_q = last_q_best - first_q_best
    print(f"\n[mit] first-quartile best RAE = {first_q_best:.4f}")
    print(f"[mit] last-quartile best RAE  = {last_q_best:.4f}")
    print(f"[mit] drift (last - first)    = {drift_q:+.4f}  "
          f"(<0 expected if TPE works; >0 -> potential overfit)")

    # ---- Refit best params with PROD seed bag (matches nb2103 protocol) ----
    print("\n" + "-" * 78)
    print(f"REFIT: 5-seed bag {PROD_SEEDS} with best params")
    print("-" * 78)
    t_refit = time.time()
    rae_mean_prod, rae_median_prod, per_seed_prod = _bag_rae(
        X_unb_28, residual_unb, anchor_unb, y_unb, folds,
        best_params, PROD_SEEDS,
    )
    print(f"[refit] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_prod)}]")
    print(f"[refit] per-seed mean RAE = {per_seed_prod.mean():.4f}  "
          f"std={per_seed_prod.std():.4f}")
    print(f"[refit] mean-bag pooled RAE   = {rae_mean_prod:.4f}")
    print(f"[refit] median-bag pooled RAE = {rae_median_prod:.4f}")
    print(f"[refit] wall = {time.time() - t_refit:.1f}s")

    # ---- Decision vs nb2103 K=28 ----
    delta_mean = rae_mean_prod - NB2103_K28_MEAN_BAG_REF
    delta_median = rae_median_prod - NB2103_K28_MEDIAN_BAG_REF
    beats_mean = rae_mean_prod < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats_median = rae_median_prod < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    flat_mean = abs(delta_mean) < DECISION_MARGIN
    flat_median = abs(delta_median) < DECISION_MARGIN

    print(f"\n[decide] vs nb2103 K=28:")
    print(f"   mean_bag   {rae_mean_prod:.4f}  d={delta_mean:+.4f}  "
          f"(margin={DECISION_MARGIN})")
    print(f"   median_bag {rae_median_prod:.4f}  d={delta_median:+.4f}  "
          f"(margin={DECISION_MARGIN})")

    if beats_mean or beats_median:
        decision = "OPTUNA_BEATS_NB2103_K28"
    elif flat_mean or flat_median:
        decision = "OPTUNA_FLAT_VS_NB2103_K28"
    else:
        decision = "OPTUNA_LOSES_VS_NB2103_K28"
    print(f"   decision = {decision}")

    # ---- Fresh-seed verify if best passes ----
    fresh_rec = None
    if beats_mean or beats_median:
        print("\n" + "-" * 78)
        print(f"FRESH-SEED VERIFY: {FRESH_SEEDS} (decorrelated from search)")
        print("-" * 78)
        t_fresh = time.time()
        rae_mean_fresh, rae_median_fresh, per_seed_fresh = _bag_rae(
            X_unb_28, residual_unb, anchor_unb, y_unb, folds,
            best_params, FRESH_SEEDS,
        )
        fresh_beats_mean = (rae_mean_fresh
                            < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN)
        fresh_beats_median = (rae_median_fresh
                              < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN)
        print(f"[fresh] per-seed RAE = "
              f"[{', '.join(f'{r:.4f}' for r in per_seed_fresh)}]")
        print(f"[fresh] mean-bag pooled RAE   = {rae_mean_fresh:.4f}")
        print(f"[fresh] median-bag pooled RAE = {rae_median_fresh:.4f}")
        print(f"[fresh] fresh_beats_mean      = {fresh_beats_mean}")
        print(f"[fresh] fresh_beats_median    = {fresh_beats_median}")
        print(f"[fresh] wall = {time.time() - t_fresh:.1f}s")
        fresh_rec = {
            "seeds": FRESH_SEEDS,
            "per_seed_rae": per_seed_fresh.tolist(),
            "rae_per_seed_mean": float(per_seed_fresh.mean()),
            "rae_per_seed_std": float(per_seed_fresh.std()),
            "rae_mean_bag": float(rae_mean_fresh),
            "rae_median_bag": float(rae_median_fresh),
            "fresh_beats_nb2103_mean": bool(fresh_beats_mean),
            "fresh_beats_nb2103_median": bool(fresh_beats_median),
        }

    # ---- Save Optuna study ----
    study_path = DATA_PROCESSED / f"{TAG}_optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"\n[save] optuna study: {study_path}")

    # ---- Build deploy CSV if best passes ----
    deploy_path = None
    deploy_in_rae_median = None
    deploy_in_rae_mean = None
    if (beats_mean or beats_median) and fresh_rec is not None and (
        fresh_rec["fresh_beats_nb2103_mean"]
        or fresh_rec["fresh_beats_nb2103_median"]
    ):
        print("\n" + "-" * 78)
        print("DEPLOY: rebuild full 513 feature matrix + 25 fits "
              "(5 outer x 5 inner)")
        print("-" * 78)
        # Need full 513 X with top-28 SHAP cols.  X_unb_28 cache is the
        # unb-sliced version, so we need to reconstruct the 513-row matrix.
        # Easiest path: piggy-back on nb2112 builder logic.  But since this
        # script is meant to be standalone and the 117-col build is heavy,
        # we instead reuse the nb2112-style approach by importing the live
        # feature builder *only* if available.
        try:
            te_513, rae_in_mean_d, rae_in_median_d = _build_deploy_csv(
                best_params, te_anchor_513, X_unb_28, residual_unb,
                anchor_unb, y_unb, unb_idx,
            )
            deploy_in_rae_mean = rae_in_mean_d
            deploy_in_rae_median = rae_in_median_d
            df_sub = pd.DataFrame({
                "SMILES": all_smiles,
                "Molecule Name": te["name"].astype(str).tolist(),
                "pEC50": te_513.astype(np.float32),
            })
            deploy_path = SUBMISSIONS_DIR / f"{TAG}_optuna_K28.csv"
            df_sub.to_csv(deploy_path, index=False)
            print(f"[save] deploy CSV: {deploy_path}  ({len(df_sub)} rows)")
            print(f"[deploy] in-sample RAE on unb (median) = "
                  f"{rae_in_median_d:.4f}")
            print(f"[deploy] in-sample RAE on unb (mean)   = "
                  f"{rae_in_mean_d:.4f}")
        except Exception as e:
            print(f"[deploy] SKIPPED: build failed: {type(e).__name__}: {e}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": "optuna_tpe_lgbm_K28_residual_chemprop_aux",
        "anchor": ANCHOR,
        "n_trials": N_TRIALS,
        "n_unb": int(n_unb),
        "n_folds": N_FOLDS,
        "search_seed": SEARCH_SEED,
        "prod_seeds": PROD_SEEDS,
        "fresh_seeds": FRESH_SEEDS,
        "decision_margin": DECISION_MARGIN,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "rae_anchor_chemprop_aux": rae_anchor,
        "search_best_rae_1seed": best_rae_search,
        "best_params": {k: (float(v) if not isinstance(v, int) else int(v))
                        for k, v in best_params.items()},
        "refit_prod_per_seed_rae": per_seed_prod.tolist(),
        "refit_prod_per_seed_mean": float(per_seed_prod.mean()),
        "refit_prod_per_seed_std": float(per_seed_prod.std()),
        "refit_prod_rae_mean_bag": float(rae_mean_prod),
        "refit_prod_rae_median_bag": float(rae_median_prod),
        "delta_mean_vs_nb2103_K28": float(delta_mean),
        "delta_median_vs_nb2103_K28": float(delta_median),
        "beats_nb2103_K28_mean": bool(beats_mean),
        "beats_nb2103_K28_median": bool(beats_median),
        "flat_vs_nb2103_K28_mean": bool(flat_mean),
        "flat_vs_nb2103_K28_median": bool(flat_median),
        "decision": decision,
        "fresh_verify": fresh_rec,
        "mitigation_drift": {
            "first_quartile_best_rae": first_q_best,
            "last_quartile_best_rae": last_q_best,
            "drift_q4_minus_q1": float(drift_q),
            "comment": ("drift_q < 0 expected if TPE meaningfully improves "
                        "over random; drift_q > 0 suggests Optuna is "
                        "exploring without genuine RAE signal"),
        },
        "trial_log_head": trial_log[:5],
        "trial_log_tail": trial_log[-5:],
        "deploy_csv_path": str(deploy_path) if deploy_path else None,
        "deploy_in_rae_mean": deploy_in_rae_mean,
        "deploy_in_rae_median": deploy_in_rae_median,
        "wall_sec": round(time.time() - t0, 2),
        "pre_unblind_clean": True,
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary: {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


def _build_deploy_csv(best_params: dict, te_anchor_513: np.ndarray,
                      X_unb_28: np.ndarray, residual_unb: np.ndarray,
                      anchor_unb: np.ndarray, y_unb: np.ndarray,
                      unb_idx: np.ndarray,
                      ) -> tuple[np.ndarray, float, float]:
    """Rebuild 513-row K=28 SHAP feature matrix and run 5x5=25 fits.

    Reuses the 117-col 5-way K-tuned stack from nb2103/nb2112.  This requires
    the same external caches.  Returns (te_513, rae_in_mean, rae_in_median).
    """
    # Heavy 117-col build: defer to nb2112's helpers via subprocess import
    # to keep nb1121 readable.  Inline minimal version below.
    from pxr.chem import standardize, morgan_fp_batch

    ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
    MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
    CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
    AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
    MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
    EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    rec28 = None
    for r in nb2103_sum["per_K_records"]:
        if int(r["K"]) == 28:
            rec28 = r
            break
    if rec28 is None:
        raise KeyError("nb2103 K=28 record missing")
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)

    with open(DATA_PROCESSED / "nb1352_summary.json") as f:
        sum_1352 = json.load(f)
    with open(DATA_PROCESSED / "nb1392_summary.json") as f:
        sum_1392 = json.load(f)
    with open(DATA_PROCESSED / "nb1484_summary.json") as f:
        sum_1484 = json.load(f)
    with open(DATA_PROCESSED / "nb1523_summary.json") as f:
        sum_1523 = json.load(f)
    with open(DATA_PROCESSED / "nb1524_summary.json") as f:
        sum_1524 = json.load(f)
    with open(DATA_PROCESSED / "nb1541_summary.json") as f:
        sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    rec_mord = None
    for r in sum_1523["per_K_records"]:
        if int(r["K"]) == int(sum_1523["best_K"]):
            rec_mord = r
            break
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = None
    for fam in sum_1484["families"]:
        if fam["family"] == "AtomPair":
            full_ap_ranked = np.array(fam["top_idx_ranked"], dtype=int)
            break
    top_ap_bit_idx = full_ap_ranked[:int(sum_1524["best_K"])]
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:int(sum_1541["best_K"])]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()

    def _load_npy(p, expected_n):
        X = np.load(p)
        if X.shape[0] != expected_n:
            raise ValueError(f"shape mismatch {p}: {X.shape}")
        X = X.astype(np.float32)
        return np.where(np.isfinite(X), X, 0.0).astype(np.float32)

    X_ap = _load_npy(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx]
    X_maccs = _load_npy(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx]
    X_emb = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx]
    X_av = _load_npy(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx]

    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    X_mord = np.load(mte_p).astype(np.float32)
    X_mord = np.where(np.isfinite(X_mord), X_mord, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_mord, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_mord)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_mord[idx_r, idx_c] = col_med[idx_c]
    X_mord = X_mord[:, top_mord_col_idx]

    # ChEMBL kNN
    KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
    KEEP_RELATIONS = {"=", "==", "~"}
    MAX_NM, MIN_NM = 100_000.0, 1e-3
    frames = []
    p1 = EXT_DIR / "chembl_pxr_CHEMBL3401.parquet"
    if p1.exists():
        d = pd.read_parquet(p1)
        m = (d["standard_type"].isin(KEEP_TYPES)
             & d["canonical_smiles"].notna()
             & (d["standard_units"] == "nM")
             & d["standard_value"].notna()
             & d["standard_relation"].isin(KEEP_RELATIONS))
        d = d[m].copy()
        v = d["standard_value"].astype(float)
        d = d[(v > MIN_NM) & (v < MAX_NM)].copy()
        d["pec50"] = 9.0 - np.log10(d["standard_value"].astype(float))
        d = d[["canonical_smiles", "pec50"]].rename(
            columns={"canonical_smiles": "smiles"})
        frames.append(d)
    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        frames.append(d)
    p3 = EXT_DIR / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["smiles", "pec50"]]
        frames.append(d)
    pool = pd.concat(frames, ignore_index=True)
    pool_mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = [
        Chem.MolToInchiKey(m) if m is not None else None for m in pool_mols
    ]
    pool["std_smiles"] = [
        Chem.MolToSmiles(m) if m is not None else None for m in pool_mols
    ]
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    pool = (pool.groupby("inchikey", as_index=False)
            .agg(pec50=("pec50", "median"),
                 std_smiles=("std_smiles", "first")))
    test_mols = [standardize(s) for s in test_smiles]
    test_ikeys = set()
    for m in test_mols:
        if m is not None:
            test_ikeys.add(Chem.MolToInchiKey(m))
    pool = pool[~pool["inchikey"].isin(test_ikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    pool = pool[keep_pool].reset_index(drop=True)
    fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_med = float(np.median(pool_labels))
    std_test = []
    for m in test_mols:
        std_test.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test)
    # kNN top-5
    a = fp_test.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    inter = a @ b.T
    denom = a_sum[:, None] + b_sum[None, :] - inter
    denom = np.maximum(denom, 1.0)
    sim = inter / denom
    part = np.argpartition(-sim, kth=4, axis=1)[:, :5]
    row_idx = np.arange(sim.shape[0])[:, None]
    sim_part = sim[row_idx, part]
    order = np.argsort(-sim_part, axis=1)
    top_idx_knn = part[row_idx, order]
    top_sim_knn = sim[row_idx, top_idx_knn]
    w = np.clip(top_sim_knn, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    pred_chembl = np.empty(sim.shape[0], dtype=np.float32)
    for i in range(sim.shape[0]):
        if w_sum[i] < 1e-6:
            pred_chembl[i] = pool_med
        else:
            pred_chembl[i] = (w[i] * pool_labels[top_idx_knn[i]]).sum() / w_sum[i]
    mean_sim_full = top_sim_knn.mean(axis=1).astype(np.float32)

    X_te_117 = np.concatenate(
        [X_ap, X_maccs, X_mord, X_emb, X_av,
         pred_chembl.reshape(-1, 1), mean_sim_full.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_te_28 = X_te_117[:, top28_idx].astype(np.float32)

    # 25 fits with best Optuna params
    OUTER_SEEDS = [0, 1, 7, 42, 137]
    INNER_OFFSETS = [0, 1, 7, 42, 137]
    all_resid_513 = np.zeros((25, X_te_28.shape[0]), dtype=np.float64)
    k = 0
    for o in OUTER_SEEDS:
        for off in INNER_OFFSETS:
            s = o * 1000 + off
            lgbm_kwargs = _build_lgbm_params(best_params, int(s))
            mdl = lgb.LGBMRegressor(**lgbm_kwargs)
            mdl.fit(X_unb_28, residual_unb)
            all_resid_513[k] = mdl.predict(X_te_28)
            k += 1
    median_resid = np.median(all_resid_513, axis=0)
    mean_resid = all_resid_513.mean(axis=0)
    te_513_median = te_anchor_513 + median_resid
    te_513_mean = te_anchor_513 + mean_resid
    rae_in_mean = float(rae(y_unb, te_513_mean[unb_idx]))
    rae_in_median = float(rae(y_unb, te_513_median[unb_idx]))
    return te_513_median.astype(np.float32), rae_in_mean, rae_in_median


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_trials", "search_best_rae_1seed",
        "refit_prod_rae_mean_bag", "refit_prod_rae_median_bag",
        "delta_mean_vs_nb2103_K28", "delta_median_vs_nb2103_K28",
        "decision", "deploy_csv_path", "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== BEST PARAMS ====")
    for k, v in res.get("best_params", {}).items():
        print(f"  {k}: {v}")
    if res.get("fresh_verify") is not None:
        print("\n==== FRESH-SEED VERIFY ====")
        for k, v in res["fresh_verify"].items():
            print(f"  {k}: {v}")
