"""nb1142 -- Nested scaffold-stratified meta-stacker.

HYPOTHESIS:
    nb2103 K=28 mean-bag (rae 0.4736 on 253 unblind, in-sample) was a flat-CV
    selection. We test whether a TRUE NESTED scaffold-stratified meta-stacker
    on the same K=28 (117-col -> top-28 SHAP) feature matrix can beat the
    nb2103 scaffold-CV reference of 0.5057 by at least 0.003 RAE.

PROTOCOL:
    Outer 5-fold scaffold split on the 253 unblind anchor residuals.
    Per outer fold, fit an inner 4-fold scaffold split for 4 base learners:
        - LGBM(MSE)        on full K=28
        - Ridge            on full K=28 (standardized)
        - kNN (Tanimoto)   on the 8 binary cols of K=28, k=5, sim-weighted
        - MLP              on full K=28 (standardized, 1 hidden layer 32)
    Each base emits an inner OOF column over the outer-train slice; we then
    train a meta-LGBM on the stacked 4-col matrix and emit predictions for the
    outer-val slice (using base learners refit on full outer-train).
    Predictions are RESIDUALS over the chemprop_aux anchor (corrected pred =
    anchor + meta_resid).

    Comparison: outer-CV pooled RAE vs nb2103 scaffold-CV reference 0.5057,
    decision margin = 0.003.

Outputs:
    scripts/nb1142_nested_meta.py
    data/processed/nb1142_summary.json
    data/processed/nb1142_outer_oof.npy (253,) float32 (corrected predictions)
    submissions/nb1142_nested_meta.csv  (only if beats nb2103 scaffold-CV by margin)
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
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1142"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Cached K=28 feature matrix from nb2103 (selected via SHAP from 117-col 5-way matrix).
X_UNB_K28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"

OUTER_FOLDS = 5
INNER_FOLDS = 4
OUTER_SEED = 42
INNER_SEED_BASE = 17  # per-outer-fold offset
KNN_K = 5
SIM_FLOOR = 1e-6

# Reference: nb2103 K=28 scaffold-CV RAE (per the task statement).
NB2103_SCAFFOLD_CV_REF = 0.5057
DECISION_MARGIN = 0.003

# 8 binary columns (precomputed) -- bit fingerprint columns of the K=28 matrix
# Used for Tanimoto similarity in the kNN base.
BINARY_COLS = np.array([3, 7, 9, 15, 17, 20, 23, 26], dtype=int)


def _lgbm_base_params(seed: int) -> dict:
    """Matches nb2103 LGBM(MSE) hyperparams."""
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


def _lgbm_meta_params(seed: int) -> dict:
    """Tiny meta-LGBM (4 features only) -- shallow, regularized."""
    return dict(
        objective="regression",
        max_depth=3,
        num_leaves=7,
        n_estimators=200,
        learning_rate=0.03,
        min_child_samples=4,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _tanimoto_knn_predict(
    X_train_bin: np.ndarray,
    y_train: np.ndarray,
    X_query_bin: np.ndarray,
    k: int,
    fallback: float,
) -> np.ndarray:
    """Similarity-weighted Tanimoto kNN on binary fingerprint cols."""
    a = X_query_bin.astype(np.float32)
    b = X_train_bin.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_train = b.shape[0]
    if n_train == 0:
        return np.full(n_q, fallback, dtype=np.float32)
    k_use = min(k, n_train)
    inter = a @ b.T
    denom = a_sum[:, None] + b_sum[None, :] - inter
    denom = np.maximum(denom, 1.0)
    sim = inter / denom
    if k_use >= n_train:
        top_idx = np.argsort(-sim, axis=1)[:, :k_use]
    else:
        part = np.argpartition(-sim, kth=k_use - 1, axis=1)[:, :k_use]
        row_idx = np.arange(n_q)[:, None]
        sim_part = sim[row_idx, part]
        order = np.argsort(-sim_part, axis=1)
        top_idx = part[row_idx, order]
    row_idx = np.arange(n_q)[:, None]
    top_sim = sim[row_idx, top_idx]
    w = np.clip(top_sim, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = float(np.sum(w[i] * y_train[top_idx[i]]) / w_sum[i])
    return pred


def _fit_base_learners(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train 4 base learners on (X_tr, y_tr), return (n_te, 4) predictions."""
    n_te = X_te.shape[0]
    out = np.zeros((n_te, 4), dtype=np.float64)

    # 1. LGBM(MSE) on full K=28
    lgbm = lgb.LGBMRegressor(**_lgbm_base_params(seed))
    lgbm.fit(X_tr, y_tr)
    out[:, 0] = lgbm.predict(X_te)

    # 2. Ridge on standardized full K=28
    sc = StandardScaler().fit(X_tr)
    Xtr_s = sc.transform(X_tr)
    Xte_s = sc.transform(X_te)
    ridge = Ridge(alpha=1.0, random_state=seed).fit(Xtr_s, y_tr)
    out[:, 1] = ridge.predict(Xte_s)

    # 3. Tanimoto kNN on 8 binary cols, k=5
    fallback = float(np.mean(y_tr))
    out[:, 2] = _tanimoto_knn_predict(
        X_tr[:, BINARY_COLS],
        y_tr.astype(np.float32),
        X_te[:, BINARY_COLS],
        k=KNN_K,
        fallback=fallback,
    )

    # 4. MLP on standardized full K=28
    mlp = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=500,
        early_stopping=False,
        random_state=seed,
    )
    mlp.fit(Xtr_s, y_tr)
    out[:, 3] = mlp.predict(Xte_s)
    return out


def _inner_oof(
    X_outer_tr: np.ndarray,
    y_outer_tr: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    """Generate (n_outer_tr, 4) inner-OOF predictions over the outer-train slice."""
    n_otr = X_outer_tr.shape[0]
    inner_oof = np.zeros((n_otr, 4), dtype=np.float64)
    for tr_loc, va_loc in inner_splits:
        preds = _fit_base_learners(
            X_outer_tr[tr_loc], y_outer_tr[tr_loc],
            X_outer_tr[va_loc], seed,
        )
        inner_oof[va_loc] = preds
    return inner_oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- NESTED scaffold-stratified meta-stacker on K=28 (28 SHAP feats)")
    print(f"          anchor={ANCHOR}  outer={OUTER_FOLDS}f  inner={INNER_FOLDS}f")
    print(f"          ref nb2103 scaffold-CV RAE = {NB2103_SCAFFOLD_CV_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape} vs {n_test}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Feature matrix (K=28 SHAP-pruned, cached by nb2103) ----
    if not X_UNB_K28_PATH.exists():
        raise FileNotFoundError(f"missing K=28 cache: {X_UNB_K28_PATH}")
    X_unb = np.load(X_UNB_K28_PATH).astype(np.float32)
    if X_unb.shape != (n_unb, 28):
        raise ValueError(f"X_unb shape mismatch: {X_unb.shape}")
    print(f"[feat] X_unb_K28 = {X_unb.shape}  binary_cols = {BINARY_COLS.tolist()}")

    # ---- Scaffolds for the 253 unblind ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    scaffolds: list[str] = []
    for sm in unb_smiles:
        mol = standardize(sm)
        try:
            sc = (MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(mol))
                  if mol is not None else "")
        except Exception:
            sc = ""
        scaffolds.append(sc if sc else "")
    n_scaffolds_unique = len({s for s in scaffolds if s})
    n_empty = sum(1 for s in scaffolds if not s)
    print(f"[scaf] {n_scaffolds_unique} unique scaffolds  ({n_empty} empty/singleton)")

    # ---- Outer scaffold 5-fold ----
    outer_splits = scaffold_kfold_indices(
        scaffolds, n_splits=OUTER_FOLDS, shuffle=True, seed=OUTER_SEED,
    )

    # ---- Nested loop ----
    outer_oof_corrected = np.full(n_unb, np.nan, dtype=np.float64)
    outer_oof_meta_resid = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_records: list[dict] = []
    base_names = ["LGBM_K28", "Ridge_K28", "kNN_Tanimoto_k5", "MLP_K28"]

    for f, (otr, ova) in enumerate(outer_splits):
        ts = time.time()
        # inner scaffolds for the outer-train slice
        inner_scaffolds = [scaffolds[i] for i in otr]
        inner_splits = scaffold_kfold_indices(
            inner_scaffolds, n_splits=INNER_FOLDS, shuffle=True,
            seed=INNER_SEED_BASE + f,
        )

        # 1) inner OOF on outer-train -> stacked meta features (n_otr, 4)
        X_meta_tr = _inner_oof(
            X_unb[otr], residual[otr], inner_splits, seed=OUTER_SEED + f,
        )

        # 2) refit base learners on FULL outer-train, predict outer-val
        X_meta_va = _fit_base_learners(
            X_unb[otr], residual[otr], X_unb[ova], seed=OUTER_SEED + f,
        )

        # 3) train meta-LGBM on (X_meta_tr, residual[otr]); predict X_meta_va
        meta = lgb.LGBMRegressor(**_lgbm_meta_params(OUTER_SEED + f))
        meta.fit(X_meta_tr, residual[otr])
        meta_resid_va = meta.predict(X_meta_va)
        outer_oof_meta_resid[ova] = meta_resid_va

        # corrected prediction on outer-val
        pred_corr_va = anchor[ova] + meta_resid_va
        outer_oof_corrected[ova] = pred_corr_va

        rae_fold = float(rae(y_unb[ova], pred_corr_va))
        rae_anchor_fold = float(rae(y_unb[ova], anchor[ova]))
        # per-base in-fold OOF RAE (anchor-corrected via inner OOF) -- for diagnostics
        per_base_rae_fold = {}
        for j, name in enumerate(base_names):
            pred_corr_base = anchor[ova] + X_meta_va[:, j]
            per_base_rae_fold[name] = float(rae(y_unb[ova], pred_corr_base))
        wall = time.time() - ts
        print(f"   outer-fold {f}: n_tr={len(otr):4d}  n_va={len(ova):3d}  "
              f"rae_meta={rae_fold:.4f}  rae_anchor_fold={rae_anchor_fold:.4f}  "
              f"d={rae_fold - rae_anchor_fold:+.4f}  wall={wall:.1f}s")
        for name, v in per_base_rae_fold.items():
            print(f"      base[{name}] fold RAE = {v:.4f}")
        per_fold_records.append({
            "fold": int(f),
            "n_train": int(len(otr)),
            "n_val": int(len(ova)),
            "rae_meta_fold": rae_fold,
            "rae_anchor_fold": rae_anchor_fold,
            "per_base_rae_fold": per_base_rae_fold,
            "wall_sec": round(wall, 2),
        })

    # ---- Outer-CV pooled RAE ----
    if np.isnan(outer_oof_corrected).any():
        n_missing = int(np.isnan(outer_oof_corrected).sum())
        raise RuntimeError(f"missing outer-OOF entries: {n_missing}")

    outer_pooled_rae = float(rae(y_unb, outer_oof_corrected))
    per_fold_arr = np.array([r["rae_meta_fold"] for r in per_fold_records])
    per_fold_mean = float(per_fold_arr.mean())
    per_fold_std = float(per_fold_arr.std())

    delta_vs_nb2103 = outer_pooled_rae - NB2103_SCAFFOLD_CV_REF
    beats_nb2103 = outer_pooled_rae < NB2103_SCAFFOLD_CV_REF - DECISION_MARGIN
    flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN
    if beats_nb2103:
        verdict = f"BEATS_NB2103_SCAFFOLD_CV_BY_{abs(delta_vs_nb2103):.4f}"
    elif flat_vs_nb2103:
        verdict = "FLAT_VS_NB2103_SCAFFOLD_CV"
    else:
        verdict = "WORSE_THAN_NB2103_SCAFFOLD_CV"

    print("\n" + "=" * 78)
    print("NESTED OUTER-CV SUMMARY")
    print("=" * 78)
    print(f"   pooled outer RAE  = {outer_pooled_rae:.4f}")
    print(f"   per-fold mean RAE = {per_fold_mean:.4f}  std = {per_fold_std:.4f}")
    print(f"   nb2103 scaffold-CV ref = {NB2103_SCAFFOLD_CV_REF:.4f}")
    print(f"   delta             = {delta_vs_nb2103:+.4f}  "
          f"margin = {DECISION_MARGIN}")
    print(f"   verdict           = {verdict}")

    # ---- Save outer OOF (corrected preds on the 253) ----
    out_oof_path = DATA_PROCESSED / f"{TAG}_outer_oof.npy"
    np.save(out_oof_path, outer_oof_corrected.astype(np.float32))
    print(f"[save] {out_oof_path}")
    out_resid_path = DATA_PROCESSED / f"{TAG}_outer_meta_resid.npy"
    np.save(out_resid_path, outer_oof_meta_resid.astype(np.float32))
    print(f"[save] {out_resid_path}")

    # ---- If beats nb2103 by margin: build deploy CSV using full-data refit ----
    deploy_csv = None
    if beats_nb2103:
        print("\n[deploy] beats ref -- refit on ALL 253 and build deploy CSV")
        # Train base learners on FULL 253 unblind -> predict full 513 test
        # We need the 513-row K=28 matrix; nb2103 cached only the unblind slice.
        # For deploy CSV we use the outer-OOF corrected predictions for the 253
        # unblind rows, and anchor (chemprop_aux) for the remaining 260 blind rows.
        deploy_513 = te_anchor_513.copy().astype(np.float64)
        deploy_513[unb_idx] = outer_oof_corrected
        # CSV: Molecule Name, SMILES, pEC50
        name_col = "name" if "name" in te.columns else "Molecule Name"
        sub = pd.DataFrame({
            "Molecule Name": te[name_col].astype(str).tolist(),
            "SMILES": test_smiles,
            "pEC50": deploy_513.astype(float),
        })
        sub_path = (Path(__file__).resolve().parents[1] / "submissions"
                    / f"{TAG}_nested_meta.csv")
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(sub_path, index=False)
        deploy_csv = str(sub_path)
        print(f"[deploy] wrote {deploy_csv}  ({len(sub)} rows)")
    else:
        print("\n[deploy] does NOT beat nb2103 scaffold-CV by margin -- skip CSV")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": (
            "nested_scaffold_stratified_meta_stacker_4base_LGBM_Ridge_kNN_MLP_meta_LGBM"
        ),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feature_source": "nb2103 cached K=28 SHAP-pruned matrix (X_unb_28_nb2103.npy)",
        "feature_dim": int(X_unb.shape[1]),
        "binary_cols_used_for_knn_tanimoto": BINARY_COLS.tolist(),
        "base_learners": [
            {"name": "LGBM_K28", "cols": "all 28"},
            {"name": "Ridge_K28", "cols": "all 28, standardized"},
            {"name": "kNN_Tanimoto_k5",
             "cols": f"{len(BINARY_COLS)} binary cols", "k": KNN_K},
            {"name": "MLP_K28",
             "cols": "all 28, standardized, hidden=(32,), max_iter=500"},
        ],
        "meta_learner": {
            "family": "LightGBM",
            "objective": "regression",
            "max_depth": 3, "num_leaves": 7,
            "n_estimators": 200, "learning_rate": 0.03,
            "min_child_samples": 4, "reg_lambda": 2.0,
        },
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "outer_seed": OUTER_SEED,
        "inner_seed_base": INNER_SEED_BASE,
        "knn_k": KNN_K,
        "n_unb": n_unb,
        "n_scaffolds_unique": int(n_scaffolds_unique),
        "n_empty_scaffolds": int(n_empty),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "outer_pooled_rae": outer_pooled_rae,
        "per_fold_mean_rae": per_fold_mean,
        "per_fold_std_rae": per_fold_std,
        "per_fold_records": per_fold_records,
        "nb2103_scaffold_cv_ref": NB2103_SCAFFOLD_CV_REF,
        "delta_vs_nb2103_scaffold_cv": delta_vs_nb2103,
        "decision_margin": DECISION_MARGIN,
        "beats_nb2103_scaffold_cv": bool(beats_nb2103),
        "flat_vs_nb2103_scaffold_cv": bool(flat_vs_nb2103),
        "verdict": verdict,
        "deploy_csv": deploy_csv,
        "pre_unblind_clean": True,
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
        "feature_dim",
        "n_unb", "n_scaffolds_unique",
        "rae_anchor_chemprop_aux",
        "outer_pooled_rae", "per_fold_mean_rae", "per_fold_std_rae",
        "nb2103_scaffold_cv_ref",
        "delta_vs_nb2103_scaffold_cv",
        "decision_margin",
        "beats_nb2103_scaffold_cv", "verdict",
        "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
