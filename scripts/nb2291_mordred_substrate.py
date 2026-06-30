"""nb2291 -- Fresh Mordred 1500+ descriptor substrate as alternative to 117-col K-tuned matrix.

HYPOTHESIS:
    nb2231 RFE on the 117-col 5-way K-tuned slice retained only 4 of 11 Mordred
    columns in K=20.  But the 117-col matrix exposes only the nb1523-best-K
    Mordred subset (a SHAP-pruned slice on ~50 cols).  Build a FRESH full
    Mordred 1613-descriptor universe (mordred-community, ignore_3D=True),
    clean NaN/inf, drop low-variance + duplicated cols, then run greedy
    backward RFE to K=20 directly on this Mordred-only substrate.  Compare
    vs nb2240 K=20 anchor (mean_bag RAE 0.4630 on 253 unblind chemprop_aux
    residual).

PROTOCOL:
    1. Compute Mordred descriptors via mordred-community on 4139+513 (1613 cols).
       (We need 4392 rows total; we only score on 253 unb_idx rows but
       compute the train pool for scaling / NaN filtering breadth.)
    2. Drop NaN/inf cols (>5% missing), drop zero-variance cols, drop
       duplicated cols.  Median-impute residual NaNs.  Standardize.
    3. Greedy backward RFE from full surviving set to K=20.  Anchor =
       chemprop_aux te[unb_idx]; learner = LGBM(MSE) on residual,
       scaffold 5-fold CV, 1 seed for the RFE search.
    4. Final K=20: 5-seed mean-bag scaffold-CV RAE.
    5. Compare vs nb2240 K=20 anchor (0.4630).  Decision margin 0.003.

If mordred is not installed (no `from mordred import Calculator`), SKIP all
compute, write a documented summary with status=skipped, and exit 0.

Outputs:
    scripts/nb2291_mordred_substrate.py
    data/processed/nb2291_summary.json
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
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2291"
SUMMARY_PATH = DATA_PROCESSED / f"{TAG}_summary.json"

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

NB2240_K20_REF_OOF = 0.4630
DECISION_MARGIN = 0.003

N_FOLDS = 5
KF_SEEDS_FULL = [1001, 1002, 1003, 1004, 1005]
KF_SEED_RFE = 1001
RESID_SEEDS = [0, 1, 7, 42, 137]

K_TARGET = 20
NAN_FRAC_MAX = 0.05
LOW_VAR_THRESH = 1e-8

CACHE_DIR = Path("C:/pxr_artifacts/nb2291")


def _write_skipped(reason: str) -> int:
    summary = {
        "tag": TAG,
        "status": "skipped",
        "reason": reason,
        "k_target": K_TARGET,
        "anchor": "chemprop_aux",
        "reference_K20_anchor_rae": NB2240_K20_REF_OOF,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SKIP] {reason}")
    print(f"[SKIP] wrote {SUMMARY_PATH}")
    return 0


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


def _compute_mordred(smiles_list, cache_path: Path):
    """Returns (X, feature_names) float32 with cache."""
    from mordred import Calculator, descriptors
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return d["X"].astype(np.float32), list(d["names"])
    calc = Calculator(descriptors, ignore_3D=True)
    n_desc = len(calc.descriptors)
    print(f"[mordred] computing {n_desc} descriptors on {len(smiles_list)} mols...")
    rows = []
    t0 = time.time()
    LOG_EVERY = max(50, len(smiles_list) // 20)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            rows.append([np.nan] * n_desc)
            continue
        try:
            res = calc(mol).fill_missing(np.nan)
            vals = []
            for v in res:
                try:
                    vv = float(v)
                except Exception:
                    vv = np.nan
                vals.append(vv)
            rows.append(vals)
        except Exception:
            rows.append([np.nan] * n_desc)
        if (i + 1) % LOG_EVERY == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(smiles_list) - i - 1) / rate
            print(f"   {i+1}/{len(smiles_list)}  rate={rate:.1f}/s  eta={eta:.0f}s")
    X = np.asarray(rows, dtype=np.float64)
    names = [str(d) for d in calc.descriptors]
    print(f"[mordred] done X={X.shape}  dt={time.time()-t0:.0f}s")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X.astype(np.float32),
                        names=np.array(names, dtype=object))
    return X.astype(np.float32), names


def _clean_columns(X_train: np.ndarray, X_test: np.ndarray, names):
    """Drop NaN/inf-heavy cols, zero-variance cols, duplicates; impute + standardize."""
    X_all = np.concatenate([X_train, X_test], axis=0).astype(np.float64)
    X_all = np.where(np.isfinite(X_all), X_all, np.nan)
    n_rows, n_cols = X_all.shape
    nan_frac = np.isnan(X_all).mean(axis=0)
    keep_mask = nan_frac <= NAN_FRAC_MAX
    print(f"[clean] NaN-frac<= {NAN_FRAC_MAX}: keep {keep_mask.sum()}/{n_cols}")

    # median-impute residual NaN
    col_med = np.nanmedian(X_all[:, keep_mask], axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    X_kept = X_all[:, keep_mask]
    for j in range(X_kept.shape[1]):
        bad = ~np.isfinite(X_kept[:, j])
        if bad.any():
            X_kept[bad, j] = col_med[j]

    # zero-variance drop
    var = X_kept.var(axis=0)
    var_mask = var > LOW_VAR_THRESH
    X_kept = X_kept[:, var_mask]
    print(f"[clean] var>{LOW_VAR_THRESH}: keep {var_mask.sum()}/{len(var_mask)}")

    # duplicate-column drop (signature)
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

    # standardize (z-score)
    mu = X_kept.mean(axis=0)
    sd = X_kept.std(axis=0) + 1e-12
    X_kept = (X_kept - mu) / sd

    surviving_idx_of_kept = np.where(keep_mask)[0][var_mask][uniq_idx]
    surviving_names = [names[i] for i in surviving_idx_of_kept]
    X_train_clean = X_kept[:len(X_train)].astype(np.float32)
    X_test_clean = X_kept[len(X_train):].astype(np.float32)
    return X_train_clean, X_test_clean, surviving_names


def _residual_cv_one_seed(X, residual, scaffolds, kf_seed):
    import lightgbm as lgb
    splits = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS,
                                    shuffle=True, seed=kf_seed)
    n = len(residual)
    oof = np.full(n, np.nan)
    for tr, va in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr], residual[tr])
        oof[va] = mdl.predict(X[va])
    return oof


def _eval_subset(X, residual, anchor, y, scaffolds, kf_seed):
    oof = _residual_cv_one_seed(X, residual, scaffolds, kf_seed)
    return float(rae(y, anchor + oof))


def _eval_subset_meanbag(X, residual, anchor, y, scaffolds, seeds):
    per_seed_rae = []
    oofs = []
    for s in seeds:
        oof = _residual_cv_one_seed(X, residual, scaffolds, s)
        per_seed_rae.append(float(rae(y, anchor + oof)))
        oofs.append(anchor + oof)
    mean_bag = np.mean(np.stack(oofs, axis=0), axis=0)
    return float(rae(y, mean_bag)), float(np.mean(per_seed_rae)), per_seed_rae


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- fresh Mordred 1613-dim substrate for K=20 residual learner")
    print("=" * 78)

    try:
        import mordred  # noqa: F401
        from mordred import Calculator, descriptors  # noqa: F401
    except Exception as e:
        return _write_skipped(f"mordred import failed: {e!r}")

    try:
        import lightgbm  # noqa: F401
    except Exception as e:
        return _write_skipped(f"lightgbm import failed: {e!r}")

    if not ANCHOR_TE_PATH.exists():
        return _write_skipped(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    if not UNB_IDX_PATH.exists() or not UNB_Y_PATH.exists():
        return _write_skipped("unblind audit files missing")

    # -- Load data --
    tr = load_train()
    te = load_test()
    tr_smiles = tr["smiles"].astype(str).tolist()
    te_smiles = te["smiles"].astype(str).tolist()
    n_train, n_test = len(tr_smiles), len(te_smiles)
    print(f"[load] n_train={n_train}  n_test={n_test}")

    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_uniq_scaf}")

    # -- Compute Mordred on train + test (cached) --
    cache_path = CACHE_DIR / "X_mordred_train_test.npz"
    X_mord, mord_names = _compute_mordred(tr_smiles + te_smiles, cache_path)
    X_mord_train = X_mord[:n_train]
    X_mord_test = X_mord[n_train:]
    print(f"[mordred] train={X_mord_train.shape}  test={X_mord_test.shape}")

    # -- Clean (filter+impute+dedup+standardize) on TRAIN+TEST jointly --
    X_train_clean, X_test_clean, surv_names = _clean_columns(
        X_mord_train, X_mord_test, mord_names,
    )
    n_dim = X_train_clean.shape[1]
    X_unb_clean = X_test_clean[unb_idx]
    print(f"[clean] surviving dims = {n_dim}")
    print(f"[feat] X_unb_clean = {X_unb_clean.shape}  X_te_clean = {X_test_clean.shape}")

    if n_dim < K_TARGET:
        return _write_skipped(
            f"only {n_dim} surviving dims < K_target={K_TARGET}"
        )

    # -- Greedy backward RFE: drop one-at-a-time until K_TARGET --
    print("\n" + "-" * 78)
    print(f"GREEDY BACKWARD RFE from {n_dim} -> {K_TARGET}  (1-seed search)")
    print("-" * 78)
    surviving = list(range(n_dim))
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
        # for very large K, sub-sample candidates each pass to keep wall<budget
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
            print(f"[rfe] iter {iter_count:3d}  K={len(surviving):4d}  "
                  f"RAE={best_drop_rae:.4f}  dropped={surv_names[dropped_feat_idx][:30]}")

    rfe_wall = time.time() - rfe_t0
    print(f"[rfe] done in {rfe_wall:.0f}s  final K={len(surviving)}")

    surviving_idx_final = surviving
    surviving_names_final = [surv_names[i] for i in surviving]
    X_unb_K20 = X_unb_clean[:, surviving_idx_final]
    X_te_K20 = X_test_clean[:, surviving_idx_final]

    # -- Final 5-seed mean-bag CV at K=20 --
    print("\n" + "-" * 78)
    print(f"FINAL K={K_TARGET} 5-SEED MEAN-BAG SCAFFOLD-CV  seeds={KF_SEEDS_FULL}")
    print("-" * 78)
    per_seed_rae_kf = []
    oofs_kf = []
    for s in KF_SEEDS_FULL:
        oof = _residual_cv_one_seed(
            X_unb_K20, residual, unb_scaffolds, s,
        )
        per_seed_rae_kf.append(float(rae(y_unb, anchor + oof)))
        oofs_kf.append(anchor + oof)
        print(f"   kf_seed={s}  RAE={per_seed_rae_kf[-1]:.4f}")
    mean_bag_kf = np.mean(np.stack(oofs_kf, axis=0), axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_kf))
    rae_per_seed_mean = float(np.mean(per_seed_rae_kf))
    print(f"[final] mean_bag RAE = {rae_mean_bag:.4f}  per_seed_mean = "
          f"{rae_per_seed_mean:.4f}")

    delta_vs_nb2240 = rae_mean_bag - NB2240_K20_REF_OOF
    if rae_mean_bag <= NB2240_K20_REF_OOF - DECISION_MARGIN:
        verdict = "BEATS_NB2240"
    elif rae_mean_bag >= NB2240_K20_REF_OOF + DECISION_MARGIN:
        verdict = "LOSES_TO_NB2240"
    else:
        verdict = "FLAT_VS_NB2240"
    print(f"[gate] vs nb2240 K=20 (ref {NB2240_K20_REF_OOF:.4f}): "
          f"delta={delta_vs_nb2240:+.4f}  -> {verdict}")

    # -- Persist substrate artifacts for downstream use --
    artifacts_dir = CACHE_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "X_unb_mordred_clean.npy", X_unb_clean.astype(np.float32))
    np.save(artifacts_dir / "X_te_mordred_clean.npy", X_test_clean.astype(np.float32))
    np.save(artifacts_dir / "X_unb_mordred_K20.npy", X_unb_K20.astype(np.float32))
    np.save(artifacts_dir / "X_te_mordred_K20.npy", X_te_K20.astype(np.float32))

    summary = {
        "tag": TAG,
        "status": "completed",
        "method": "fresh_mordred_1613_clean_RFE_K20_chemprop_aux_residual",
        "anchor": "chemprop_aux",
        "rae_anchor_chemprop_aux": rae_anchor,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_scaffolds": int(n_uniq_scaf),
        "mordred_total_descriptors": int(X_mord.shape[1]),
        "mordred_surviving_after_clean": int(n_dim),
        "nan_frac_max": NAN_FRAC_MAX,
        "low_var_thresh": LOW_VAR_THRESH,
        "k_target": K_TARGET,
        "k_final": int(len(surviving)),
        "kf_seeds": KF_SEEDS_FULL,
        "n_folds": N_FOLDS,
        "rfe_iterations": int(iter_count),
        "rfe_wall_sec": float(rfe_wall),
        "rfe_trajectory_first10": rfe_trajectory[:10],
        "rfe_trajectory_last10": rfe_trajectory[-10:],
        "surviving_K20_names": surviving_names_final,
        "surviving_K20_idx_in_clean": [int(i) for i in surviving_idx_final],
        "rae_K20_per_seed": per_seed_rae_kf,
        "rae_K20_per_seed_mean": rae_per_seed_mean,
        "rae_K20_mean_bag": rae_mean_bag,
        "delta_vs_nb2240_K20": delta_vs_nb2240,
        "nb2240_K20_reference_rae": NB2240_K20_REF_OOF,
        "decision_margin": DECISION_MARGIN,
        "verdict_vs_nb2240": verdict,
        "X_unb_mordred_clean_path": str(artifacts_dir / "X_unb_mordred_clean.npy"),
        "X_te_mordred_clean_path": str(artifacts_dir / "X_te_mordred_clean.npy"),
        "X_unb_mordred_K20_path": str(artifacts_dir / "X_unb_mordred_K20.npy"),
        "X_te_mordred_K20_path": str(artifacts_dir / "X_te_mordred_K20.npy"),
        "wall_sec": float(time.time() - t0),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] wrote {SUMMARY_PATH}")
    print(f"[done] wall = {time.time()-t0:.0f}s")
    print(f"[done] verdict = {verdict}  rae_mean_bag = {rae_mean_bag:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
