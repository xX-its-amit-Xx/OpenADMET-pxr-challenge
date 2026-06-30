"""nb1115 -- ITERATIVE (MICE-style) COUNTER-ASSAY IMPUTATION.

HYPOTHESIS:
    The PXR-null counter-assay is observed on 2,858 of 4,139 train compounds
    (1,281 missing). Median-imputing pec50_null kills its signal as a feature.
    A MICE-style loop -- predict null using main features + pec50, then refit
    main pec50 LGBM with the imputed null as a feature, iterate -- should
    extract a real selectivity feature for the 1,281 unlabelled rows.

PROTOCOL:
    1. Load train (4,139) and counter (2,858) -- align by InChIKey.
    2. Round 0: median-impute pec50_null on missing rows.
    3. Round 1..R: scaffold 5-fold CV
         - main LGBM (target=pec50) predicts pec50 OOF on train using
           [combined features + imputed_null].
         - null LGBM (target=pec50_null) predicts null OOF on train using
           [combined features + pec50_oof_from_main].
         - Replace imputed_null on missing rows with null OOF predictions.
       Stop when mean |delta| on missing rows < 1e-3 (or after R=5 rounds).
    4. Quality gate: held-out R^2 on real-null subset (split the 2,858 80/20,
       use the same MICE loop trained on the 80%, predict on 20%). R^2 >= 0.30
       required to proceed.
    5. 5-fold scaffold cross-fit on the 253 unblind: train main LGBM on
       4,139 with [combined + final imputed null] -> predict 253.
    6. Train-only-feature trap check: test_std(pec50_null_imputed_on_513) must
       be >= 0.80 vs train_std ~ 1.03.
    7. Compare honest cross-fit RAE vs nb2103 K=28 (0.4737 / 0.4698).
    8. If beats by margin >= 0.003, build deploy CSV; otherwise REJECT.

CONFIG:
    - LGBM: n_estimators=500, num_leaves=64, max_depth=-1, lr=0.05
            (matches nb730 main / nb06 conventions)
    - MICE rounds: 5 max, converge if max |delta| < 1e-3
    - Quality gate R^2 floor: 0.30 (memory rule)
    - Beat-margin vs nb2103 K=28: 0.003 RAE (project convention)

OUTPUT:
    scripts/nb1115_mice_imputation.py
    data/processed/nb1115_summary.json
    data/processed/nb1115_pred_oof_253.npy  (if quality gate passes)
    data/processed/te_nb1115.npy            (if quality gate + beat target)
    data/processed/nb1115_null_imputed_train.npy  (length 4139)
    data/processed/nb1115_null_imputed_test.npy   (length 513)
    submissions/nb1115_mice_deploy.csv      (if beats nb2103 K=28)
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
from rdkit import RDLogger
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.chem import bemis_murcko, to_inchikey
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

TAG = "nb1115"
SEED = 0
N_FOLDS = 5
MAX_ROUNDS = 5
CONVERGE_TOL = 1e-3
R2_FLOOR = 0.30
HELDOUT_FRAC = 0.20
DECISION_MARGIN = 0.003
TEST_STD_MIN = 0.80
TRAIN_STD_REF = 1.03

# References
NB2103_K28_MEAN_BAG = 0.4737   # nb2103 K=28 mean-bag RAE on 253 unblind
NB2103_K28_MEDIAN_BAG = 0.4698 # nb2103 K=28 median-bag RAE on 253 unblind

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=1.0,
    objective="regression",
    verbose=-1,
    n_jobs=2,
    random_state=SEED,
)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _scaffold_oof_lgbm(
    X: np.ndarray, y: np.ndarray, splits: list, params: dict, tag: str
) -> np.ndarray:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = mdl.predict(X[va_idx])
    r = float(rae(y, oof))
    print(f"  [{tag}] scaffold CV RAE = {r:.4f}")
    return oof


def _mice_loop(
    X_train: np.ndarray,
    pec50: np.ndarray,
    null_init: np.ndarray,
    null_observed_mask: np.ndarray,
    splits: list,
    max_rounds: int,
    tag_prefix: str = "",
) -> tuple[np.ndarray, list[dict]]:
    """Run MICE-style imputation on `null_init`; returns final imputed null + trace."""
    null_imp = null_init.copy()
    trace: list[dict] = []
    missing_mask = ~null_observed_mask
    n_missing = int(missing_mask.sum())
    print(f"  [{tag_prefix}MICE] starting with {n_missing} missing rows")

    pec50_oof_prev = pec50.copy()  # bootstrap: use real labels in round 1
    for r in range(1, max_rounds + 1):
        t_round = time.time()
        # Main head: target=pec50, features = [X_train, null_imp]
        X_main = np.hstack([X_train, null_imp.reshape(-1, 1)]).astype(np.float32)
        pec50_oof = _scaffold_oof_lgbm(
            X_main, pec50, splits, LGBM_PARAMS, f"{tag_prefix}R{r}-main"
        )

        # Null head: target=pec50_null (only on observed rows for training,
        # predict on missing). Train fold-wise but only use observed rows in
        # training portion. To keep it scaffold-honest, drop missing rows from
        # training portion of each fold; predict on the full fold val set.
        X_null = np.hstack([X_train, pec50_oof.reshape(-1, 1)]).astype(np.float32)
        null_oof = np.full(len(pec50), np.nan, dtype=np.float64)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            tr_obs = tr_idx[null_observed_mask[tr_idx]]
            if len(tr_obs) < 100:
                # fold too small to fit a null model; fall back to round prev
                null_oof[va_idx] = null_imp[va_idx]
                continue
            mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
            mdl.fit(X_null[tr_obs], null_init[tr_obs])
            null_oof[va_idx] = mdl.predict(X_null[va_idx])

        # Replace imputed null on missing rows; leave observed rows pinned to truth
        new_null = null_imp.copy()
        new_null[missing_mask] = null_oof[missing_mask]
        if missing_mask.any():
            delta = float(np.mean(np.abs(new_null[missing_mask] - null_imp[missing_mask])))
            max_delta = float(np.max(np.abs(new_null[missing_mask] - null_imp[missing_mask])))
        else:
            delta = 0.0
            max_delta = 0.0
        rae_round = float(rae(pec50, pec50_oof))
        # R^2 of null OOF on observed rows
        r2_obs = _r2(null_init[null_observed_mask], null_oof[null_observed_mask])
        wall = time.time() - t_round
        trace.append({
            "round": r,
            "pec50_oof_rae": rae_round,
            "null_oof_r2_on_observed": r2_obs,
            "mean_abs_delta_on_missing": delta,
            "max_abs_delta_on_missing": max_delta,
            "wall_sec": round(wall, 2),
        })
        print(f"  [{tag_prefix}R{r}] pec50 RAE={rae_round:.4f}  "
              f"null R^2(obs)={r2_obs:.4f}  "
              f"mean|d|={delta:.4f}  max|d|={max_delta:.4f}  "
              f"wall={wall:.1f}s")
        null_imp = new_null
        pec50_oof_prev = pec50_oof
        if delta < CONVERGE_TOL:
            print(f"  [{tag_prefix}MICE] converged at round {r} (mean|d|<{CONVERGE_TOL})")
            break

    return null_imp, trace


def _impute_null_for_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    pec50_train: np.ndarray,
    null_imputed_train: np.ndarray,
    null_observed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit null head on observed rows of train, predict on 513 test.

    Bootstrap: first predict pec50 on 513 from a model trained on X_train ONLY
    (no null feature), then use that pec50_hat as the null head's pec50 column.
    """
    # bootstrap pec50_hat for test: train pec50 model WITHOUT null feature
    mdl_pec50_boot = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_pec50_boot.fit(X_train, pec50_train)
    pec50_hat_test = mdl_pec50_boot.predict(X_test).astype(np.float64)

    # null head: train on [X_train_obs, true pec50_obs] -> predict on 513 with
    # [X_test, pec50_hat_test]
    X_null_tr_obs = np.hstack(
        [X_train[null_observed_mask], pec50_train[null_observed_mask].reshape(-1, 1)]
    ).astype(np.float32)
    mdl_null = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_null.fit(X_null_tr_obs, null_imputed_train[null_observed_mask])
    X_null_te = np.hstack(
        [X_test, pec50_hat_test.reshape(-1, 1)]
    ).astype(np.float32)
    null_hat_test = mdl_null.predict(X_null_te).astype(np.float64)
    return null_hat_test, pec50_hat_test


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MICE-style counter-assay imputation")
    print(f"   beat target: nb2103 K=28 mean_bag {NB2103_K28_MEAN_BAG:.4f} / "
          f"median_bag {NB2103_K28_MEDIAN_BAG:.4f}  margin={DECISION_MARGIN}")
    print("=" * 78)

    needed = {
        "TRAIN":     DATA_RAW / "pxr-challenge_TRAIN.csv",
        "COUNTER":   DATA_RAW / "pxr-challenge_counter-assay_TRAIN.csv",
        "TEST":      DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    train_df = pd.read_csv(needed["TRAIN"])
    counter_df = pd.read_csv(needed["COUNTER"])
    test_df = pd.read_csv(needed["TEST"])
    unb_df = pd.read_csv(needed["UNBLINDED"])

    n_tr = len(train_df)
    n_te = len(test_df)

    # ----- Step 1: dedup train rows to one-per-InChIKey (median pEC50 over reps)
    print("\n--- Step 1: align train + counter by InChIKey ---")
    train_df["ik"] = train_df["SMILES"].apply(to_inchikey)
    counter_df["ik"] = counter_df["SMILES"].apply(to_inchikey)

    # Keep first SMILES per InChIKey for stability; aggregate pec50 by median
    train_dedup = (
        train_df.dropna(subset=["ik", "pEC50"])
        .groupby("ik", as_index=False)
        .agg(
            SMILES=("SMILES", "first"),
            pec50_pxr=("pEC50", "median"),
            mol_name=("Molecule Name", "first"),
            n_reps=("pEC50", "count"),
        )
    )
    counter_dedup = (
        counter_df.dropna(subset=["ik", "pEC50"])
        .groupby("ik", as_index=False)
        .agg(pec50_null=("pEC50", "median"))
    )
    print(f"  train rows: {n_tr}  unique InChIKey w/ pEC50: {len(train_dedup)}")
    print(f"  counter rows: {len(counter_df)}  unique InChIKey: {len(counter_dedup)}")

    merged = pd.merge(
        train_dedup, counter_dedup, on="ik", how="left"
    ).reset_index(drop=True)
    n_unique = len(merged)
    null_observed_mask = merged["pec50_null"].notna().values
    n_observed = int(null_observed_mask.sum())
    n_missing = n_unique - n_observed
    print(f"  merged unique compounds: {n_unique}")
    print(f"    null OBSERVED: {n_observed}  MISSING: {n_missing}")

    # ----- Test set DataFrame -----
    test_smiles = test_df["SMILES"].astype(str).tolist()

    # Build aligned unblind on 513-row test set index
    name_to_idx = {n: i for i, n in enumerate(test_df["Molecule Name"])}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int)
    y_unb = unb_df["pEC50"].astype(np.float64).values
    n_unb = len(unb_df)
    print(f"  test n={n_te}  unblind n={n_unb}")

    # ----- Step 2: features once (combined Morgan+RDKit) on merged + test
    print("\n--- Step 2: featurize merged (unique) + test (combined) ---")
    all_smi = merged["SMILES"].tolist() + test_smiles
    X_all = combined(all_smi)
    X_all = impute(X_all)
    X_train = X_all[:n_unique].astype(np.float32)
    X_test = X_all[n_unique:].astype(np.float32)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}  "
          f"~{X_all.nbytes/1e6:.1f} MB")

    # ----- Step 3: scaffold splits on merged
    print("\n--- Step 3: scaffold-CV splits on merged ---")
    scafs = [bemis_murcko(s) for s in merged["SMILES"]]
    splits = scaffold_kfold_indices(scafs, n_splits=N_FOLDS, seed=SEED)

    # ----- Step 4: Round 0 median impute
    median_null = float(np.nanmedian(merged["pec50_null"].values))
    null_init = merged["pec50_null"].astype(np.float64).values.copy()
    null_init[~null_observed_mask] = median_null
    pec50 = merged["pec50_pxr"].astype(np.float64).values
    train_null_std_obs = float(np.nanstd(merged.loc[null_observed_mask, "pec50_null"].values))
    print(f"  median null = {median_null:.3f}  obs std = {train_null_std_obs:.3f}")

    # ----- Step 5: MICE loop (FULL data)
    print("\n--- Step 5: MICE loop on full merged data ---")
    null_imputed_full, mice_trace_full = _mice_loop(
        X_train, pec50, null_init, null_observed_mask, splits,
        max_rounds=MAX_ROUNDS, tag_prefix="full-"
    )

    # Imputed null on missing rows
    null_imp_on_missing = null_imputed_full[~null_observed_mask]
    imp_mean = float(null_imp_on_missing.mean())
    imp_std = float(null_imp_on_missing.std())
    print(f"\n  imputed null on missing rows: mean={imp_mean:.3f}  "
          f"std={imp_std:.3f}  (n={len(null_imp_on_missing)})")

    # ----- Step 6: quality gate via 80/20 held-out on observed rows -----
    print("\n--- Step 6: quality gate (80/20 held-out R^2 on real-null subset) ---")
    rng = np.random.default_rng(SEED)
    obs_idx = np.where(null_observed_mask)[0]
    perm = rng.permutation(obs_idx)
    n_held = int(len(perm) * HELDOUT_FRAC)
    held_idx = perm[:n_held]
    keep_idx = perm[n_held:]
    miss_idx_global = np.where(~null_observed_mask)[0]

    # Hidden mask: keep_idx + miss_idx are "available" observed for training the
    # null head; held_idx is hidden (treated as missing for imputation purposes,
    # but we know the true null value).
    null_init_q = null_init.copy()
    null_init_q[held_idx] = median_null  # treat as missing
    obs_mask_q = null_observed_mask.copy()
    obs_mask_q[held_idx] = False

    # Same MICE loop, but with held_idx hidden
    null_imputed_q, mice_trace_q = _mice_loop(
        X_train, pec50, null_init_q, obs_mask_q, splits,
        max_rounds=MAX_ROUNDS, tag_prefix="heldout-"
    )
    null_pred_held = null_imputed_q[held_idx]
    null_true_held = merged["pec50_null"].values[held_idx]
    held_r2 = _r2(null_true_held, null_pred_held)
    held_corr = float(np.corrcoef(null_true_held, null_pred_held)[0, 1])
    print(f"\n  held-out R^2 = {held_r2:.4f}  held-out corr = {held_corr:.4f}  "
          f"n_held = {len(held_idx)}")
    print(f"  R^2 gate (>= {R2_FLOOR}): "
          f"{'PASS' if held_r2 >= R2_FLOOR else 'FAIL'}")

    quality_pass = bool(held_r2 >= R2_FLOOR)

    # ----- Step 7: impute null on 513 test set -----
    print("\n--- Step 7: impute null on 513 test compounds ---")
    null_test_hat, pec50_test_hat = _impute_null_for_test(
        X_train, X_test, pec50, null_imputed_full, null_observed_mask
    )
    null_test_std = float(null_test_hat.std())
    null_test_mean = float(null_test_hat.mean())
    print(f"  null_test  mean={null_test_mean:.3f}  std={null_test_std:.3f}")
    print(f"  train_null_std(obs)={train_null_std_obs:.3f}  "
          f"ratio={null_test_std/max(train_null_std_obs, 1e-6):.3f}")
    ratio_vs_ref = null_test_std / TRAIN_STD_REF
    print(f"  test_std / TRAIN_STD_REF({TRAIN_STD_REF}) = {ratio_vs_ref:.3f}  "
          f"floor={TEST_STD_MIN}")
    trap_pass = bool(ratio_vs_ref >= TEST_STD_MIN)
    print(f"  TRAIN-ONLY-FEATURE TRAP CHECK: "
          f"{'PASS' if trap_pass else 'FAIL (collapses on test)'}")

    # ----- Step 8: 5-fold scaffold cross-fit on the 253 unblind -----
    # Build the augmented main feature matrix for 253:
    #   train: X_train + null_imputed_full (for full merged set), target pec50
    #   test: X_test + null_test_hat
    # Honest cross-fit on 253 means: KFold over 253, but model is trained on
    # 4139 + held-out-fold-of-253 features available is irrelevant -- the 253
    # are predicted using a model trained ONLY on the 4139 (deploy refit).
    # That is the standard "deploy-on-test" eval; for honest unblind 5-fold,
    # we use the deploy model and slice te[unb_idx]. Optional: 5-fold KFold
    # on the 253 by REFITTING per-fold WITH 4139+held-in subset of 253? That
    # would be leak. Honest path: trust the deploy model trained on 4139 only.
    print("\n--- Step 8: deploy refit on full merged (target=pec50) ---")
    X_main_tr = np.hstack(
        [X_train, null_imputed_full.reshape(-1, 1)]
    ).astype(np.float32)
    X_main_te = np.hstack(
        [X_test, null_test_hat.reshape(-1, 1)]
    ).astype(np.float32)
    # Honest cross-fit on the merged set (4139 unique) to get an OOF pec50 RAE
    # plus the deploy refit on test.
    print("  -- scaffold-5-fold OOF on merged (sanity check, NOT LB metric) --")
    pec50_oof_main_with_null = _scaffold_oof_lgbm(
        X_main_tr, pec50, splits, LGBM_PARAMS, "main_with_null"
    )
    print("  -- scaffold-5-fold OOF on merged WITHOUT null (control) --")
    pec50_oof_main_no_null = _scaffold_oof_lgbm(
        X_train, pec50, splits, LGBM_PARAMS, "main_no_null"
    )
    delta_oof = float(rae(pec50, pec50_oof_main_with_null)
                      - rae(pec50, pec50_oof_main_no_null))
    print(f"  delta OOF RAE (with - without null) = {delta_oof:+.4f}  "
          f"(negative = null helps)")

    print("\n  -- deploy refit (train on 4139, predict 513) --")
    mdl_deploy = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_deploy.fit(X_main_tr, pec50)
    pec50_deploy_513 = mdl_deploy.predict(X_main_te).astype(np.float64)
    deploy_test_std = float(pec50_deploy_513.std())
    print(f"  deploy 513 pred  mean={pec50_deploy_513.mean():.3f}  "
          f"std={deploy_test_std:.3f}")

    # Honest cross-fit on the 253 unblind: split 253 into 5 scaffold folds,
    # but DO NOT add labels into training. The 4139-trained deploy model is
    # static; the "cross-fit" is just an in-sample rae computation. We instead
    # report rae(y_unb, pec50_deploy_513[unb_idx]) as the LB-faithful in_RAE.
    pred_oof_253 = pec50_deploy_513[unb_idx]
    in_rae = float(rae(y_unb, pred_oof_253))
    print(f"  IN-SAMPLE-ON-UNBLIND RAE (te[unb_idx]) = {in_rae:.4f}  "
          f"(LB-faithful estimate)")

    # ----- Step 9: comparison vs nb2103 K=28 -----
    delta_vs_nb2103 = in_rae - NB2103_K28_MEAN_BAG
    beats_nb2103 = bool(in_rae < NB2103_K28_MEAN_BAG - DECISION_MARGIN)
    print(f"\n  delta vs nb2103 K=28 mean_bag = {delta_vs_nb2103:+.4f}  "
          f"(margin={DECISION_MARGIN})")
    print(f"  beats nb2103 K=28: {beats_nb2103}")

    # ----- Verdict -----
    if not quality_pass:
        verdict = f"FAIL_QUALITY_GATE_R2_{held_r2:.3f}_LT_{R2_FLOOR}"
        deploy_csv = None
    elif not trap_pass:
        verdict = (
            f"FAIL_TRAIN_ONLY_FEATURE_TRAP_test_std_{null_test_std:.3f}"
            f"_ratio_{ratio_vs_ref:.3f}_LT_{TEST_STD_MIN}"
        )
        deploy_csv = None
    elif beats_nb2103:
        verdict = f"BEATS_NB2103_K28_at_RAE_{in_rae:.4f}"
        deploy_csv = SUBMISSIONS / f"{TAG}_mice_deploy.csv"
    elif abs(delta_vs_nb2103) < DECISION_MARGIN:
        verdict = f"FLAT_VS_NB2103_K28_at_RAE_{in_rae:.4f}"
        deploy_csv = None
    else:
        verdict = f"HURTS_VS_NB2103_K28_delta_{delta_vs_nb2103:+.4f}"
        deploy_csv = None
    print(f"\n  VERDICT: {verdict}")

    # ----- Save artefacts -----
    np.save(DATA_PROCESSED / f"{TAG}_null_imputed_train.npy",
            null_imputed_full.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_null_imputed_test.npy",
            null_test_hat.astype(np.float32))
    if quality_pass:
        np.save(DATA_PROCESSED / f"{TAG}_pred_oof_253.npy",
                pred_oof_253.astype(np.float32))
        np.save(DATA_PROCESSED / f"te_{TAG}.npy",
                pec50_deploy_513.astype(np.float32))
    if deploy_csv is not None:
        pd.DataFrame({
            "SMILES": test_df["SMILES"],
            "Molecule Name": test_df["Molecule Name"],
            "pEC50": pec50_deploy_513.astype(np.float32),
        }).to_csv(deploy_csv, index=False)
        print(f"  WROTE deploy CSV: {deploy_csv}")

    summary = {
        "tag": TAG,
        "method": "MICE_iterative_counter_assay_imputation",
        "lgbm_params": {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"},
        "n_unique_train": int(n_unique),
        "n_null_observed": int(n_observed),
        "n_null_missing": int(n_missing),
        "median_null_seed": median_null,
        "train_null_std_observed": train_null_std_obs,
        "max_rounds": MAX_ROUNDS,
        "converge_tol": CONVERGE_TOL,
        "mice_trace_full": mice_trace_full,
        "mice_trace_heldout": mice_trace_q,
        "n_rounds_full_used": len(mice_trace_full),
        "imputed_null_on_missing_mean": imp_mean,
        "imputed_null_on_missing_std": imp_std,
        "heldout_n": int(n_held),
        "heldout_r2": float(held_r2),
        "heldout_corr": float(held_corr),
        "heldout_r2_floor": R2_FLOOR,
        "quality_gate_pass": quality_pass,
        "null_test_mean": null_test_mean,
        "null_test_std": null_test_std,
        "null_test_to_train_ratio": null_test_std / max(train_null_std_obs, 1e-6),
        "null_test_to_ref_ratio": ratio_vs_ref,
        "test_std_floor": TEST_STD_MIN,
        "train_std_ref": TRAIN_STD_REF,
        "train_only_trap_pass": trap_pass,
        "main_oof_rae_with_null": float(rae(pec50, pec50_oof_main_with_null)),
        "main_oof_rae_no_null": float(rae(pec50, pec50_oof_main_no_null)),
        "delta_oof_with_minus_no_null": delta_oof,
        "deploy_513_pred_mean": float(pec50_deploy_513.mean()),
        "deploy_513_pred_std": deploy_test_std,
        "in_sample_unblind_rae": in_rae,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG,
        "delta_vs_nb2103_K28_mean_bag": delta_vs_nb2103,
        "decision_margin": DECISION_MARGIN,
        "beats_nb2103_K28": beats_nb2103,
        "verdict": verdict,
        "deploy_csv": str(deploy_csv) if deploy_csv is not None else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_unique_train", "n_null_observed", "n_null_missing",
        "n_rounds_full_used",
        "imputed_null_on_missing_mean", "imputed_null_on_missing_std",
        "heldout_r2", "heldout_corr", "quality_gate_pass",
        "null_test_mean", "null_test_std",
        "null_test_to_train_ratio", "null_test_to_ref_ratio",
        "train_only_trap_pass",
        "main_oof_rae_with_null", "main_oof_rae_no_null",
        "delta_oof_with_minus_no_null",
        "in_sample_unblind_rae",
        "nb2103_K28_mean_bag_ref",
        "delta_vs_nb2103_K28_mean_bag",
        "beats_nb2103_K28", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
