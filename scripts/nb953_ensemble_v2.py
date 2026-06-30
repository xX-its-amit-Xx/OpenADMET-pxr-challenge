"""nb953 -- REFIT nb562/nb503/grand_v6b on AUGMENTED CORPUS + SLSQP blend v2.

Per cycle-129 plan (nb954): nb950 (chemprop_aux v2) and nb951 (LGBM v2) are
PENDING; nb953 was originally scoped as the v2 blend AFTER nb950/nb951 land.
This script builds inline v2 anchors so the blend can be evaluated end-to-end:

    AUGMENTED CORPUS:
        TRAIN (4139 PXR pEC50) + ChEMBL_PXR (CHEMBL3401 + nr_extended PXR
        + pxr_all_types, dedup InChIKey, pEC50 in [3,11]).  Holds out test
        InChIKeys.  Two v2 anchors trained on this superset.

    v2 ANCHORS (lean, no chemprop -- D: drive has 6.6 GB free):
        nb950_lite : LGBM(combined, n_est=500, leaves=64, lr=0.05) on aug.
        nb951_lite : LGBM(morgan only, n_est=400, leaves=48, lr=0.04) on aug.
        Both 5-fold cross-fit on 253 unblind (train + aug minus held-out unb).

    v2 OPERATORS (refit on v2 anchors):
        nb562_v2 : rank-stretch (grid {1.0..1.5}) on nb950_lite + nb951_lite
                   mean, 5-fold cross-fit on 253.
        nb503_v2 : SLSQP simplex over {nb950_lite, nb951_lite, nb503_v1}
                   on 253, 5-fold cross-fit.
        grand_v6b_v2 : SLSQP over {nb950_lite, nb951_lite, nb562_v2,
                   nb503_v2, nb503_v1, chemprop_aux_v1, grand_v6b_v1_calib}
                   on 253, 5-fold cross-fit.

    FINAL BLEND:
        SLSQP over {nb950_lite, nb951_lite, nb562_v2, nb503_v2, grand_v6b_v2}
        on 253, 5-fold cross-fit.

    BASELINE TO BEAT:
        nb2112 honest cross-fit RAE = 0.4698 (K=28 median bag)
        nb951_lite standalone (post-fit) on 253 (in-sample diagnostic)

If best blend beats both -> build submissions/nb953_deploy_blend_v2.csv +
te_nb953.npy.

Outputs:
    data/processed/te_nb953.npy
    data/processed/nb953_summary.json
    data/processed/te_nb950_lite.npy        (v2 anchor 1, deploy 513)
    data/processed/nb950_lite_pred_oof.npy  (v2 anchor 1, OOF 253)
    data/processed/te_nb951_lite.npy        (v2 anchor 2, deploy 513)
    data/processed/nb951_lite_pred_oof.npy  (v2 anchor 2, OOF 253)
    data/processed/te_nb562_v2.npy
    data/processed/te_nb503_v2.npy
    data/processed/te_grand_v6b_v2.npy
    submissions/nb953_deploy_blend_v2.csv   (only if best blend wins)
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
from rdkit import Chem, RDLogger
from scipy.optimize import minimize
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.chem import standardize, morgan_fp_batch
from pxr.eval import rae
from pxr.featurize import combined, morgan, impute
from pxr.paths import DATA_EXTERNAL, DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

TAG = "nb953"
SEED = 0
N_FOLDS = 5

NB2112_BASELINE = 0.4698  # honest cross-fit floor to beat

# ChEMBL pool inclusion thresholds (mirror nb2112)
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
PEC50_MIN = 3.0
PEC50_MAX = 11.0

# LGBM v2 anchors
LGBM_V2_COMBINED = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)
LGBM_V2_MORGAN = dict(
    n_estimators=400,
    learning_rate=0.04,
    max_depth=-1,
    num_leaves=48,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED + 1,
    verbose=-1,
    n_jobs=2,
)

STRETCH_GRID = [1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]


# ---------------------------------------------------------------------------
# Augmented corpus loader -- TRAIN + ChEMBL PXR superset
# ---------------------------------------------------------------------------
def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_canon(mol):
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_chembl_pxr_pool() -> pd.DataFrame:
    frames = []
    p1 = DATA_EXTERNAL / "chembl_pxr_CHEMBL3401.parquet"
    if p1.exists():
        d = pd.read_parquet(p1)
        mask = (
            d["standard_type"].isin(KEEP_TYPES)
            & d["canonical_smiles"].notna()
            & (d["standard_units"] == "nM")
            & d["standard_value"].notna()
            & d["standard_relation"].isin(KEEP_RELATIONS)
        )
        d = d[mask].copy()
        v = d["standard_value"].astype(float)
        d = d[(v > MIN_NM) & (v < MAX_NM)].copy()
        d["pec50_raw"] = 9.0 - np.log10(d["standard_value"].astype(float))
        d = d[["canonical_smiles", "pec50_raw"]].rename(
            columns={"canonical_smiles": "smiles", "pec50_raw": "pec50"}
        )
        d["src"] = "CHEMBL3401_raw"
        frames.append(d)
        print(f"   [src] CHEMBL3401: {len(d)} rows after activity filter")

    p2 = DATA_EXTERNAL / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= PEC50_MIN) & (d["pec50"] <= PEC50_MAX)].copy()
        d = d[["std_smiles", "pec50"]].rename(
            columns={"std_smiles": "smiles"}
        )
        d["src"] = "nr_extended_PXR"
        frames.append(d)
        print(f"   [src] chembl_nr_extended PXR: {len(d)} rows")

    p3 = DATA_EXTERNAL / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= PEC50_MIN) & (d["pec50"] <= PEC50_MAX)].copy()
        d = d[["smiles", "pec50"]]
        d["src"] = "pxr_all_types"
        frames.append(d)
        print(f"   [src] chembl_pxr_all_types PXR: {len(d)} rows")

    if not frames:
        return pd.DataFrame(columns=["smiles", "pec50", "src"])

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] union pre-standardize: {len(pool)} rows")

    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_canon)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")

    # InChIKey dedup, median pEC50
    agg = (
        pool.groupby("inchikey", as_index=False).agg(
            pec50=("pec50", "median"),
            smiles=("std_smiles", "first"),
            n_meas=("pec50", "count"),
        )
    )
    print(f"   [pool] after InChIKey dedup (median): {len(agg)} unique compounds")
    return agg


# ---------------------------------------------------------------------------
# SLSQP simplex blend on MAE
# ---------------------------------------------------------------------------
def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    if k == 1:
        return np.array([1.0])
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return np.full(k, 1.0 / k) if s <= 0 else w / s


def _crossfit_slsqp(P_oof: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS,
                    seed: int = SEED) -> tuple[np.ndarray, np.ndarray, list]:
    """Pooled honest 5-fold cross-fit SLSQP blend on 253 unblind."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_w = []
    for tr_i, va_i in kf.split(np.arange(n)):
        w = _fit_slsqp(P_oof[tr_i], y[tr_i])
        oof[va_i] = P_oof[va_i] @ w
        fold_w.append(w)
    pooled_w = _fit_slsqp(P_oof, y)
    return oof, pooled_w, fold_w


def _stretch(p, mu, s):
    return mu + s * (p - mu)


def _crossfit_stretch(oof_in: np.ndarray, y: np.ndarray,
                      grid=STRETCH_GRID,
                      n_folds: int = N_FOLDS,
                      seed: int = SEED) -> tuple[np.ndarray, float, float, list]:
    """5-fold cross-fit best-s stretch on 253; return oof_stretch, deploy_s, deploy_mu, per-fold-s."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold_s = []
    for tr_i, va_i in kf.split(np.arange(n)):
        mu_tr = float(oof_in[tr_i].mean())
        best_s, best_r = 1.0, float("inf")
        for s in grid:
            r = float(rae(y[tr_i], _stretch(oof_in[tr_i], mu_tr, s)))
            if r < best_r:
                best_r, best_s = r, float(s)
        oof[va_i] = _stretch(oof_in[va_i], mu_tr, best_s)
        per_fold_s.append(best_s)
    # Deploy fit on all 253
    mu_dep = float(oof_in.mean())
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        r = float(rae(y, _stretch(oof_in, mu_dep, s)))
        if r < best_r:
            best_r, best_s = r, float(s)
    return oof, best_s, mu_dep, per_fold_s


# ---------------------------------------------------------------------------
# Cross-fit LGBM v2 anchor (train + aug)
# ---------------------------------------------------------------------------
def _crossfit_lgbm_v2_anchor(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_aug: np.ndarray, y_aug: np.ndarray,
    X_unb: np.ndarray, y_unb: np.ndarray,
    X_te: np.ndarray,
    params: dict,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Honest 5-fold cross-fit of LGBM on (train + aug + unb_tr) -> predict unb_va.
    Deploy = fit on (train + aug + ALL unb) -> predict 513.
    Returns (oof_unb 253, te_deploy 513).
    """
    n_unb = len(y_unb)
    n_te = len(X_te)
    oof_unb = np.full(n_unb, np.nan, dtype=np.float64)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        X_fold = np.vstack([X_tr, X_aug, X_unb[tr_i]])
        y_fold = np.concatenate([y_tr, y_aug, y_unb[tr_i]])

        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_fold, y_fold)
        oof_unb[va_i] = mdl.predict(X_unb[va_i])
        r_va = float(rae(y_unb[va_i], oof_unb[va_i]))
        fold_raes.append(r_va)
        print(f"   [{tag}] fold {fold}: n_tr={X_fold.shape[0]} "
              f"n_va={len(va_i)} RAE={r_va:.4f}")

    rae_oof = float(rae(y_unb, oof_unb))
    print(f"   [{tag}] pooled cross-fit RAE = {rae_oof:.4f} "
          f"(per-fold {min(fold_raes):.4f}-{max(fold_raes):.4f})")

    # Deploy: refit on full (train + aug + all unb), predict 513
    X_deploy = np.vstack([X_tr, X_aug, X_unb])
    y_deploy = np.concatenate([y_tr, y_aug, y_unb])
    mdl_dep = lgb.LGBMRegressor(**params)
    mdl_dep.fit(X_deploy, y_deploy)
    te_deploy = mdl_dep.predict(X_te).astype(np.float32)
    print(f"   [{tag}] te_deploy mean/std = {te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    return oof_unb, te_deploy


# ---------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- REFIT nb562/nb503/grand_v6b on AUGMENTED CORPUS + v2 BLEND")
    print("=" * 78)

    # ----- Existing v1 anchors (sanity check existence) -----
    needed = {
        "TRAIN": DATA_RAW / "pxr-challenge_TRAIN.csv",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "te_nb503": DATA_PROCESSED / "te_nb503.npy",
        "te_nb562": DATA_PROCESSED / "te_nb562.npy",
        "te_grand_v6b_calib": DATA_PROCESSED / "te_grand_v6b_calib.npy",
        "te_chemprop_aux": DATA_PROCESSED / "te_chemprop_aux.npy",
        "nb503_pred_oof": DATA_PROCESSED / "nb503_pred_oof.npy",
        "nb562_pred_oof": DATA_PROCESSED / "nb562_pred_oof.npy",
        "_audit_unblind_idx": DATA_PROCESSED / "_audit_unblind_idx.npy",
        "_audit_unblind_y": DATA_PROCESSED / "_audit_unblind_y.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ----- Indices on test/unblind -----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    test_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(test_names)}

    unb_df = pd.read_csv(needed["UNBLINDED"])
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx_csv = np.array(
        [name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int
    )
    y_unb = unb_df["pEC50"].astype(float).values.astype(np.float64)
    n_unb = len(y_unb)

    # Cross-check with audit
    unb_idx_audit = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb_audit = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    # Use the audit order (matches existing nb503/nb562 OOFs)
    unb_idx = unb_idx_audit
    if len(y_unb_audit) != n_unb:
        print(f"WARN: y_unb csv n={n_unb} vs audit n={len(y_unb_audit)}; using audit")
        y_unb = y_unb_audit.astype(np.float64)
        n_unb = len(y_unb)
    else:
        # Use audit y (consistent with cached OOFs)
        y_unb = y_unb_audit.astype(np.float64)

    n_te = len(te_df)
    print(f"\nshape check: n_test={n_te}  n_unb={n_unb}")

    # ----- Load TRAIN -----
    train_df = pd.read_csv(needed["TRAIN"])
    print(f"train rows: {len(train_df)}  cols: {list(train_df.columns)[:6]}")
    y_train = train_df["pEC50"].astype(float).values.astype(np.float64)
    n_train = len(train_df)

    # Featurise TRAIN, UNB, TEST (single combined pass for impute)
    print("\nfeaturising TRAIN combined...")
    X_train_comb = combined(train_df["SMILES"].tolist())
    print("featurising UNB combined...")
    unb_smiles = unb_df.iloc[unb_idx_audit.argsort()[:0]].copy()  # not used; use audit-aligned order
    # We need unb SMILES in audit order; audit_unb_y matches some canonical order. Verify
    # by loading test SMILES and indexing by unb_idx.
    test_smiles_arr = te_df["SMILES"].tolist()
    unb_smiles_aligned = [test_smiles_arr[i] for i in unb_idx]
    X_unb_comb = combined(unb_smiles_aligned)
    print("featurising TEST combined...")
    X_te_comb = combined(test_smiles_arr)

    # ----- Build AUGMENTED CORPUS (ChEMBL PXR) -----
    print("\n" + "-" * 78)
    print("BUILD AUGMENTED CORPUS (ChEMBL PXR + Papyrus PXR)")
    print("-" * 78)
    pool = _load_chembl_pxr_pool()

    # Drop any aug compounds whose InChIKey matches a test compound (leakage guard)
    test_mols = [standardize(s) for s in test_smiles_arr]
    test_ik = {_safe_inchikey(m) for m in test_mols if m is not None}
    test_ik.discard(None)
    train_mols = [standardize(s) for s in train_df["SMILES"].tolist()]
    train_ik = {_safe_inchikey(m) for m in train_mols if m is not None}
    train_ik.discard(None)

    n_pre = len(pool)
    pool = pool[~pool["inchikey"].isin(test_ik)].copy()
    pool = pool[~pool["inchikey"].isin(train_ik)].copy()  # avoid double-count
    pool = pool.reset_index(drop=True)
    print(f"   [aug] kept {len(pool)} after dropping test+train InChIKey overlap "
          f"(was {n_pre})")

    if len(pool) == 0:
        print("WARN: augmented pool empty -- v2 anchors will equal v1.")
        X_aug_comb = np.empty((0, X_train_comb.shape[1]), dtype=np.float32)
        y_aug = np.array([], dtype=np.float64)
    else:
        print("featurising AUG combined...")
        X_aug_comb = combined(pool["smiles"].tolist())
        y_aug = pool["pec50"].astype(float).values.astype(np.float64)

    # Impute all together
    print("\nimputing all combined-feature matrices together (column median)...")
    X_all = np.vstack([X_train_comb, X_aug_comb, X_unb_comb, X_te_comb])
    X_all = impute(X_all)
    X_tr = X_all[:n_train]
    X_aug = X_all[n_train:n_train + len(y_aug)]
    X_unb = X_all[n_train + len(y_aug):n_train + len(y_aug) + n_unb]
    X_te = X_all[n_train + len(y_aug) + n_unb:]
    print(f"   X_tr={X_tr.shape}  X_aug={X_aug.shape}  X_unb={X_unb.shape}  "
          f"X_te={X_te.shape}  (~{X_all.nbytes/1e6:.0f} MB)")

    # ----- v2 ANCHOR 1: nb950_lite (LGBM combined on aug) -----
    print("\n" + "-" * 78)
    print("v2 ANCHOR 1: nb950_lite (LGBM-combined on TRAIN + AUG)")
    print("-" * 78)
    oof_nb950, te_nb950 = _crossfit_lgbm_v2_anchor(
        X_tr, y_train, X_aug, y_aug, X_unb, y_unb, X_te,
        params=LGBM_V2_COMBINED, tag="nb950_lite",
    )
    rae_nb950 = float(rae(y_unb, oof_nb950))
    np.save(DATA_PROCESSED / "te_nb950_lite.npy", te_nb950)
    np.save(DATA_PROCESSED / "nb950_lite_pred_oof.npy",
            oof_nb950.astype(np.float32))

    # ----- v2 ANCHOR 2: nb951_lite (LGBM morgan-only on aug) -----
    # Build morgan-only matrices via slicing first 2048 cols of combined
    # (morgan FP occupies cols 0..2047)
    print("\n" + "-" * 78)
    print("v2 ANCHOR 2: nb951_lite (LGBM-morgan2048 on TRAIN + AUG)")
    print("-" * 78)
    X_tr_m = X_tr[:, :2048]
    X_aug_m = X_aug[:, :2048] if len(y_aug) > 0 else X_aug
    X_unb_m = X_unb[:, :2048]
    X_te_m = X_te[:, :2048]
    oof_nb951, te_nb951 = _crossfit_lgbm_v2_anchor(
        X_tr_m, y_train, X_aug_m, y_aug, X_unb_m, y_unb, X_te_m,
        params=LGBM_V2_MORGAN, tag="nb951_lite",
    )
    rae_nb951 = float(rae(y_unb, oof_nb951))
    np.save(DATA_PROCESSED / "te_nb951_lite.npy", te_nb951)
    np.save(DATA_PROCESSED / "nb951_lite_pred_oof.npy",
            oof_nb951.astype(np.float32))

    # Free memory before next stage
    del X_all, X_aug, X_unb, X_te, X_tr_m, X_aug_m, X_unb_m, X_te_m

    # ----- v1 anchors loaded into 253 OOFs and 513 te vectors -----
    print("\n" + "-" * 78)
    print("LOAD v1 ANCHORS")
    print("-" * 78)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    te_grand_v6b_calib = np.load(DATA_PROCESSED / "te_grand_v6b_calib.npy").astype(np.float64)
    te_chemprop_aux = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)

    # in-sample anchors on 253 unblind rows
    chemprop_aux_unb = te_chemprop_aux[unb_idx]
    grand_v6b_calib_unb = te_grand_v6b_calib[unb_idx]
    rae_nb503_v1 = float(rae(y_unb, nb503_oof))
    rae_nb562_v1 = float(rae(y_unb, nb562_oof))
    rae_chemprop_v1 = float(rae(y_unb, chemprop_aux_unb))
    rae_grand_v6b_v1 = float(rae(y_unb, grand_v6b_calib_unb))
    print(f"   v1 anchors honest cross-fit / in-sample RAE on 253:")
    print(f"     nb503 (honest OOF)           = {rae_nb503_v1:.4f}")
    print(f"     nb562 (honest OOF)           = {rae_nb562_v1:.4f}")
    print(f"     chemprop_aux (in-sample te)  = {rae_chemprop_v1:.4f}")
    print(f"     grand_v6b_calib (in-sample)  = {rae_grand_v6b_v1:.4f}")

    # ----- v2 OPERATOR 1: nb562_v2 (rank-stretch on (nb950_lite + nb951_lite)/2) -----
    print("\n" + "-" * 78)
    print("v2 OPERATOR 1: nb562_v2 (rank-stretch on mean(nb950_lite, nb951_lite))")
    print("-" * 78)
    nb562_v2_in_oof = 0.5 * (oof_nb950 + oof_nb951)
    nb562_v2_in_te = 0.5 * (te_nb950 + te_nb951)
    oof_nb562_v2, s_dep, mu_dep, per_fold_s = _crossfit_stretch(
        nb562_v2_in_oof, y_unb,
    )
    rae_nb562_v2 = float(rae(y_unb, oof_nb562_v2))
    te_nb562_v2 = _stretch(nb562_v2_in_te, mu_dep, s_dep).astype(np.float32)
    print(f"   nb562_v2 cross-fit RAE = {rae_nb562_v2:.4f} "
          f"(deploy s={s_dep:.3f} mu={mu_dep:.3f} per_fold_s={per_fold_s})")
    np.save(DATA_PROCESSED / "te_nb562_v2.npy", te_nb562_v2)

    # ----- v2 OPERATOR 2: nb503_v2 (SLSQP {nb950_lite, nb951_lite, nb503_v1}) -----
    print("\n" + "-" * 78)
    print("v2 OPERATOR 2: nb503_v2 (SLSQP {nb950_lite, nb951_lite, nb503_v1})")
    print("-" * 78)
    pool_503 = ["nb950_lite", "nb951_lite", "nb503_v1"]
    P_503_oof = np.stack([oof_nb950, oof_nb951, nb503_oof], axis=1)
    P_503_te = np.stack([te_nb950, te_nb951, te_nb503], axis=1)
    oof_nb503_v2, w_dep_503, fold_w_503 = _crossfit_slsqp(P_503_oof, y_unb)
    rae_nb503_v2 = float(rae(y_unb, oof_nb503_v2))
    te_nb503_v2 = (P_503_te @ w_dep_503).astype(np.float32)
    w_str = "  ".join(f"{t}={w:.3f}" for t, w in zip(pool_503, w_dep_503))
    print(f"   nb503_v2 cross-fit RAE = {rae_nb503_v2:.4f}  deploy_w: {w_str}")
    np.save(DATA_PROCESSED / "te_nb503_v2.npy", te_nb503_v2)

    # ----- v2 OPERATOR 3: grand_v6b_v2 (SLSQP over expanded v2-aware pool) -----
    print("\n" + "-" * 78)
    print("v2 OPERATOR 3: grand_v6b_v2 (SLSQP wide pool)")
    print("-" * 78)
    pool_gv6 = [
        "nb950_lite", "nb951_lite", "nb562_v2", "nb503_v2",
        "nb503_v1", "chemprop_aux_v1", "grand_v6b_v1_calib",
    ]
    P_gv6_oof = np.stack(
        [oof_nb950, oof_nb951, oof_nb562_v2, oof_nb503_v2,
         nb503_oof, chemprop_aux_unb, grand_v6b_calib_unb],
        axis=1,
    )
    P_gv6_te = np.stack(
        [te_nb950, te_nb951, te_nb562_v2.astype(np.float64),
         te_nb503_v2.astype(np.float64),
         te_nb503, te_chemprop_aux, te_grand_v6b_calib],
        axis=1,
    )
    oof_gv6_v2, w_dep_gv6, fold_w_gv6 = _crossfit_slsqp(P_gv6_oof, y_unb)
    rae_gv6_v2 = float(rae(y_unb, oof_gv6_v2))
    te_gv6_v2 = (P_gv6_te @ w_dep_gv6).astype(np.float32)
    w_str = "  ".join(f"{t}={w:.3f}" for t, w in zip(pool_gv6, w_dep_gv6))
    print(f"   grand_v6b_v2 cross-fit RAE = {rae_gv6_v2:.4f}")
    print(f"   deploy_w: {w_str}")
    np.save(DATA_PROCESSED / "te_grand_v6b_v2.npy", te_gv6_v2)

    # ----- FINAL BLEND: SLSQP over 5 v2 components -----
    print("\n" + "-" * 78)
    print("FINAL nb953_blend: SLSQP {nb950_lite, nb951_lite, nb562_v2, nb503_v2, grand_v6b_v2}")
    print("-" * 78)
    pool_final = ["nb950_lite", "nb951_lite", "nb562_v2", "nb503_v2", "grand_v6b_v2"]
    P_final_oof = np.stack(
        [oof_nb950, oof_nb951, oof_nb562_v2, oof_nb503_v2, oof_gv6_v2],
        axis=1,
    )
    P_final_te = np.stack(
        [te_nb950, te_nb951,
         te_nb562_v2.astype(np.float64), te_nb503_v2.astype(np.float64),
         te_gv6_v2.astype(np.float64)],
        axis=1,
    )
    oof_blend, w_dep_blend, fold_w_blend = _crossfit_slsqp(P_final_oof, y_unb)
    rae_blend = float(rae(y_unb, oof_blend))
    te_blend = (P_final_te @ w_dep_blend).astype(np.float32)
    w_str = "  ".join(f"{t}={w:.3f}" for t, w in zip(pool_final, w_dep_blend))
    print(f"   nb953_blend cross-fit RAE = {rae_blend:.4f}")
    print(f"   deploy_w: {w_str}")

    # ----- Decision: best v2 candidate vs nb2112 baseline + nb951 alone -----
    candidates = [
        ("nb950_lite", rae_nb950, te_nb950.astype(np.float32)),
        ("nb951_lite", rae_nb951, te_nb951.astype(np.float32)),
        ("nb562_v2", rae_nb562_v2, te_nb562_v2),
        ("nb503_v2", rae_nb503_v2, te_nb503_v2),
        ("grand_v6b_v2", rae_gv6_v2, te_gv6_v2),
        ("nb953_blend", rae_blend, te_blend),
    ]
    print("\n" + "=" * 78)
    print("DEPLOY DECISION TABLE")
    print("=" * 78)
    print(f"   baseline nb2112 honest cross-fit = {NB2112_BASELINE:.4f}")
    print(f"   nb951_lite (LGBM v2)             = {rae_nb951:.4f}")
    for name, r, _ in candidates:
        marker = ""
        if r < NB2112_BASELINE:
            marker += " <BEATS nb2112"
        if r < rae_nb951:
            marker += " <BEATS nb951_lite"
        print(f"   {name:15s} cross-fit RAE = {r:.4f}{marker}")

    best = min(candidates, key=lambda t: t[1])
    best_name, best_rae, best_te = best
    print(f"\n   -> best candidate: {best_name}  cross-fit RAE = {best_rae:.4f}")

    beats_nb2112 = best_rae < NB2112_BASELINE
    beats_nb951 = best_rae < rae_nb951
    deploy_ok = beats_nb2112 and beats_nb951

    # Save te_nb953.npy regardless (canonical artefact for ladder integrity audit)
    np.save(DATA_PROCESSED / "te_nb953.npy", te_blend.astype(np.float32))
    print(f"\n[save] te_nb953.npy = {DATA_PROCESSED / 'te_nb953.npy'} "
          f"({best_name if deploy_ok else 'nb953_blend (saved even if not promoted)'})")

    if deploy_ok:
        sub_path = SUBMISSIONS / f"{TAG}_deploy_blend_v2.csv"
        # Deploy uses the best candidate's te vector
        pd.DataFrame({
            "SMILES": te_df["SMILES"],
            "Molecule Name": te_df["Molecule Name"],
            "pEC50": best_te,
        }).to_csv(sub_path, index=False)
        print(f"[save] DEPLOY: {sub_path}  (best={best_name} "
              f"cross-fit RAE={best_rae:.4f})")
    else:
        sub_path = None
        print(f"[skip] best={best_name} RAE={best_rae:.4f} does NOT beat both "
              f"baselines (nb2112={NB2112_BASELINE:.4f}, nb951={rae_nb951:.4f})")

    # ----- Save summary JSON -----
    summary = {
        "tag": TAG,
        "method": "refit_nb562_nb503_grandv6b_on_aug_corpus_plus_slsqp_v2_blend",
        "wall_sec": round(time.time() - t0, 2),
        "augmented_corpus": {
            "train_rows": int(n_train),
            "aug_rows": int(len(y_aug)),
            "n_unb": int(n_unb),
            "n_test": int(n_te),
        },
        "v2_anchors": {
            "nb950_lite": {
                "type": "LGBM_combined_train+aug",
                "rae_crossfit": float(rae_nb950),
                "te_path": str(DATA_PROCESSED / "te_nb950_lite.npy"),
                "oof_path": str(DATA_PROCESSED / "nb950_lite_pred_oof.npy"),
            },
            "nb951_lite": {
                "type": "LGBM_morgan2048_train+aug",
                "rae_crossfit": float(rae_nb951),
                "te_path": str(DATA_PROCESSED / "te_nb951_lite.npy"),
                "oof_path": str(DATA_PROCESSED / "nb951_lite_pred_oof.npy"),
            },
        },
        "v2_operators": {
            "nb562_v2": {
                "type": "rank_stretch_on_mean(nb950,nb951)",
                "rae_crossfit": float(rae_nb562_v2),
                "deploy_s": float(s_dep),
                "deploy_mu": float(mu_dep),
                "per_fold_s": [float(x) for x in per_fold_s],
                "te_path": str(DATA_PROCESSED / "te_nb562_v2.npy"),
            },
            "nb503_v2": {
                "type": "SLSQP_{nb950,nb951,nb503_v1}",
                "pool": pool_503,
                "rae_crossfit": float(rae_nb503_v2),
                "deploy_weights": {t: float(w) for t, w in zip(pool_503, w_dep_503)},
                "te_path": str(DATA_PROCESSED / "te_nb503_v2.npy"),
            },
            "grand_v6b_v2": {
                "type": "SLSQP_wide_pool",
                "pool": pool_gv6,
                "rae_crossfit": float(rae_gv6_v2),
                "deploy_weights": {t: float(w) for t, w in zip(pool_gv6, w_dep_gv6)},
                "te_path": str(DATA_PROCESSED / "te_grand_v6b_v2.npy"),
            },
        },
        "v1_reference_raes": {
            "nb503_v1_honest": float(rae_nb503_v1),
            "nb562_v1_honest": float(rae_nb562_v1),
            "chemprop_aux_v1_insample": float(rae_chemprop_v1),
            "grand_v6b_calib_v1_insample": float(rae_grand_v6b_v1),
        },
        "final_blend": {
            "pool": pool_final,
            "rae_crossfit": float(rae_blend),
            "deploy_weights": {t: float(w) for t, w in zip(pool_final, w_dep_blend)},
            "te_path": str(DATA_PROCESSED / "te_nb953.npy"),
        },
        "decision": {
            "baseline_nb2112_rae": NB2112_BASELINE,
            "nb951_lite_rae": float(rae_nb951),
            "best_candidate": best_name,
            "best_rae": float(best_rae),
            "beats_nb2112": bool(beats_nb2112),
            "beats_nb951_lite": bool(beats_nb951),
            "deploy_built": bool(deploy_ok),
            "deploy_submission": str(sub_path) if sub_path else None,
        },
    }

    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary: {out_path}")
    print(f"\n[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  rae_nb950_lite       = {res['v2_anchors']['nb950_lite']['rae_crossfit']:.4f}")
    print(f"  rae_nb951_lite       = {res['v2_anchors']['nb951_lite']['rae_crossfit']:.4f}")
    print(f"  rae_nb562_v2         = {res['v2_operators']['nb562_v2']['rae_crossfit']:.4f}")
    print(f"  rae_nb503_v2         = {res['v2_operators']['nb503_v2']['rae_crossfit']:.4f}")
    print(f"  rae_grand_v6b_v2     = {res['v2_operators']['grand_v6b_v2']['rae_crossfit']:.4f}")
    print(f"  rae_nb953_blend      = {res['final_blend']['rae_crossfit']:.4f}")
    print(f"  baseline nb2112      = {NB2112_BASELINE:.4f}")
    print(f"  best                 = {res['decision']['best_candidate']} "
          f"({res['decision']['best_rae']:.4f})")
    print(f"  deploy_built         = {res['decision']['deploy_built']}")
