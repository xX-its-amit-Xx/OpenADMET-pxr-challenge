"""nb2340 -- Avalon-1024 ALONE substrate (orthogonal to Morgan+RDKit-dominant
nb2063 SHAP set).

HYPOTHESIS:
    The 117-col 5-way K-tuned matrix used by nb2063/nb2103/nb2240 contains
    only ~30 Avalon bits, and at the SHAP top-K=28 winner (nb2103: mean-bag
    RAE 0.4737) only 2-3 Avalon bits survived (Mordred=11, ChempropEmbed=8,
    AtomPair=5, Avalon=2, MACCS=1, ChEMBL_kNN=1).  The Avalon family is
    fragment-based and historically captures a different sub-space than
    Morgan-circular ECFP / RDKit physchem descriptors -- it may carry
    PXR-relevant signal that the 117-col SHAP ranking drowns out by
    dominance of Mordred + Chemprop embeddings.

    This script tests Avalon as a STANDALONE substrate: compute Avalon-1024
    on 4139 train + 513 test (incl. 253 unblind), SHAP-rank Avalon bits
    using a chemprop_aux-residual LGBM on the 253 unblind, pick top-28,
    and refit LGBM (5-seed bag) under 5-fold scaffold-CV.  Report
    standalone delta vs nb2103 K=28 (0.4737) and SLSQP blend into the
    nb2240 5-anchor pyramid (which currently sits at pooled_rae_mean_seeds
    0.4598 on 5 scaffold kf_seeds).

PROTOCOL:
    1. Compute Avalon-1024 FPs on 4139 train + 513 test SMILES via
       rdkit.Avalon.pyAvalonTools.GetAvalonFP(nBits=1024).
       (We compute on the union just to verify and cache; the residual
       SHAP/LGBM fit only needs unb_idx slice; the deploy refit uses the
       full 513 test cache.)
    2. Slice to 253 unb_idx -> X_unb_av (253 x 1024).
       Fit full LGBM(MSE) (same hyperparams as nb2063/nb2103) on chemprop_aux
       residual.  Run shap.TreeExplainer -> mean|SHAP| over 1024 bits.
       Pick top-28 bit indices.
    3. Slice X_unb_av and X_te_av to top-28 bits.
       5-seed bag {0, 1, 7, 42, 137} x 5-fold scaffold CV (Bemis-Murcko
       scaffolds on the 253 unb compounds) of LGBM(MSE) residual on
       anchor=chemprop_aux.  Build mean-bag OOF on 253 + deploy refit on
       253 -> predict residual on full 513 -> te_nb2340_K28_avalon.npy.
    4. Compare standalone mean-bag RAE vs nb2103 K=28 ref (0.4737).
    5. Build 6-anchor SLSQP pyramid that adds nb2340 to the existing nb2240
       5-anchor stack {nb2240_K20, chemprop_aux, nb1191, nb503, nb562}
       under 5-fold scaffold CV across 5 kf_seeds {1001..1005} + rank-stretch.
       Report SLSQP weights, delta vs nb2240 ref (0.4598).

Outputs:
    scripts/nb2340_avalon_only.py
    data/processed/nb2340_summary.json
    data/processed/nb2340_avalon_shap_importance.npy   (1024,) float32
    data/processed/nb2340_top28_avalon_bit_idx.npy     (28,)   int32
    data/processed/nb2340_mean_bag_oof_K28.npy         (253,)  float32
    data/processed/te_nb2340_K28_avalon.npy            (513,)  float32
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
import lightgbm as lgb
import shap
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Avalon import pyAvalonTools
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2340"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ---- Avalon ----
N_AVALON = 1024
TOP_K = 28

# ---- Residual LGBM ----
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# ---- SLSQP pyramid (stage 2) ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---- References ----
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_REF = 0.4737        # mean-bag RAE on 117-col SHAP top-28 (task default)
NB2240_PYRAMID_REF = 0.4598    # pooled_rae_mean_seeds, 5 anchors (no nb2340)
DECISION_MARGIN = 0.003


# ---------------------------------------------------------------------------
# Avalon-1024 FP
# ---------------------------------------------------------------------------

def _avalon_fp_batch(smiles: list[str]) -> tuple[np.ndarray, int]:
    """Return (X, n_failed). X shape (n, 1024) uint8."""
    n = len(smiles)
    X = np.zeros((n, N_AVALON), dtype=np.uint8)
    n_failed = 0
    for i, s in enumerate(smiles):
        mol = standardize(s)
        if mol is None:
            n_failed += 1
            continue
        try:
            bv = pyAvalonTools.GetAvalonFP(mol, nBits=N_AVALON)
            arr = np.zeros(N_AVALON, dtype=np.uint8)
            Chem.DataStructs.ConvertToNumpyArray(bv, arr)
            X[i] = arr
        except Exception:
            n_failed += 1
    return X, n_failed


def _lgbm_params(seed: int) -> dict:
    """Same hyperparams as nb2063/nb2103/nb2154 for parity."""
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


def _scaffold_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 scaffolds: list, seed: int) -> np.ndarray:
    n = len(residual)
    splits = scaffold_kfold_indices(scaffolds, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError(f"OOF has NaN for seed {seed}")
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


# ---------------------------------------------------------------------------
# SLSQP pyramid utils (same shape as nb2240 stage 2)
# ---------------------------------------------------------------------------

def slsqp_simplex(P, y):
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_s


# ---------------------------------------------------------------------------
# nb1191 reconstruction (copied from nb2240 stage 2)
# ---------------------------------------------------------------------------

NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb):
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        if not p.exists():
            raise FileNotFoundError(f"missing nb1150 sub-anchor OOF: {p}")
        v = np.load(p).astype(np.float64)
        if v.shape != (n_unb,):
            raise ValueError(f"{p.name} shape {v.shape}")
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)
    return P @ w


def reconstruct_nb1191_oof(n_unb):
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    nb2112_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Avalon-{N_AVALON} ALONE substrate -> SHAP top-{TOP_K} "
          f"-> LGBM residual on anchor={ANCHOR}")
    print(f"          seeds={RESID_SEEDS}  folds={RESID_FOLDS} (scaffold CV)")
    print(f"          ref: nb2103 K=28 mean-bag RAE = {NB2103_K28_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth + anchor ----
    tr = load_train()
    te = load_test()
    n_train = len(tr)
    n_test = len(te)
    train_smiles = tr["smiles"].astype(str).tolist()
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_train={n_train}  n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- STEP 1: compute Avalon-1024 on train + test ----
    print("\n" + "-" * 78)
    print(f"STEP 1: compute Avalon-{N_AVALON} on {n_train} train + {n_test} test")
    print("-" * 78)
    t_fp = time.time()
    X_train_av, n_failed_tr = _avalon_fp_batch(train_smiles)
    X_test_av, n_failed_te = _avalon_fp_batch(test_smiles)
    print(f"   X_train_av {X_train_av.shape}  failed={n_failed_tr}")
    print(f"   X_test_av  {X_test_av.shape}   failed={n_failed_te}")
    print(f"   bit density train = {X_train_av.mean():.4f}  "
          f"row-sum mean = {X_train_av.sum(axis=1).mean():.1f}")
    print(f"   bit density test  = {X_test_av.mean():.4f}  "
          f"row-sum mean = {X_test_av.sum(axis=1).mean():.1f}")
    print(f"   wall = {time.time() - t_fp:.1f}s")

    X_unb_av = X_test_av[unb_idx].astype(np.float32)
    X_te_av = X_test_av.astype(np.float32)
    print(f"   X_unb_av  = {X_unb_av.shape}  (sliced from test by unb_idx)")
    print(f"   X_te_av   = {X_te_av.shape}   (full 513 for deploy refit)")

    # Bemis-Murcko scaffolds on unb compounds (for scaffold CV)
    unb_smiles = [test_smiles[int(i)] for i in unb_idx.tolist()]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_scaf_none = sum(1 for s in unb_scaffolds if s is None or s == "")
    n_scaf_unique = len({s for s in unb_scaffolds if s})
    print(f"   unb scaffolds: {n_scaf_unique} unique, {n_scaf_none} None")

    # ---- STEP 2: SHAP top-28 from full-fit LGBM on residual ----
    print("\n" + "-" * 78)
    print(f"STEP 2: full-fit LGBM(MSE) on {N_AVALON}-d Avalon -> SHAP top-{TOP_K}")
    print("-" * 78)
    t_shap = time.time()
    mdl_full = lgb.LGBMRegressor(**_lgbm_params(seed=0))
    mdl_full.fit(X_unb_av, residual)
    explainer = shap.TreeExplainer(mdl_full)
    shap_vals = explainer.shap_values(X_unb_av)
    shap_imp = np.abs(shap_vals).mean(axis=0).astype(np.float32)
    if shap_imp.shape[0] != N_AVALON:
        raise ValueError(
            f"SHAP imp shape {shap_imp.shape} != N_AVALON {N_AVALON}"
        )
    top_idx = np.argsort(-shap_imp)[:TOP_K].astype(np.int32)
    top_imp = [float(shap_imp[int(i)]) for i in top_idx]
    n_nonzero_imp = int((shap_imp > 0).sum())
    print(f"   full-fit SHAP done   wall = {time.time() - t_shap:.1f}s")
    print(f"   Avalon bits with nonzero SHAP importance: "
          f"{n_nonzero_imp}/{N_AVALON}")
    print(f"   top-{TOP_K} Avalon bit indices (ranked):")
    for rank, (idx, imp) in enumerate(zip(top_idx[:10], top_imp[:10]), 1):
        print(f"     {rank:2d}. Avalon_bit_{int(idx):4d}  |SHAP|={imp:.5f}")

    # Save full SHAP importance + top-28 idx
    shap_path = DATA_PROCESSED / f"{TAG}_avalon_shap_importance.npy"
    top_path = DATA_PROCESSED / f"{TAG}_top28_avalon_bit_idx.npy"
    np.save(shap_path, shap_imp.astype(np.float32))
    np.save(top_path, top_idx.astype(np.int32))
    print(f"   [save] {shap_path}")
    print(f"   [save] {top_path}")

    # ---- STEP 3: 5-seed bag x 5-fold scaffold CV on top-28 ----
    X_unb_top = X_unb_av[:, top_idx].astype(np.float32)
    X_te_top = X_te_av[:, top_idx].astype(np.float32)
    print(f"\n   X_unb_top = {X_unb_top.shape}  X_te_top = {X_te_top.shape}")
    print("\n" + "-" * 78)
    print(f"STEP 3: 5-SEED BAG x 5-FOLD SCAFFOLD CV  (seeds={RESID_SEEDS})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _scaffold_cross_fit_one_seed(
            X_unb_top, residual, unb_scaffolds, s
        )
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        # Deploy refit on all 253 -> predict residual on full 513
        te_resid_s = _train_full_then_predict_te(
            X_unb_top, residual, X_te_top, s
        )
        per_seed_te_resid[i] = te_resid_s
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_K28_513 = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    # Save mean-bag OOF + te deploy
    oof_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{TOP_K}.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}_K{TOP_K}_avalon.npy"
    np.save(oof_path, mean_bag_oof.astype(np.float32))
    np.save(te_path, te_K28_513.astype(np.float32))
    print(f"\n   [save] {oof_path}")
    print(f"   [save] {te_path}")

    # ---- Decorrelation vs nb2103 K=28 ----
    nb2103_oof_path = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    pearson_vs_nb2103_K28 = None
    if nb2103_oof_path.exists():
        nb2103_oof = np.load(nb2103_oof_path).astype(np.float64)
        if nb2103_oof.shape[0] == n_unb:
            pearson_vs_nb2103_K28 = float(
                np.corrcoef(mean_bag_oof, nb2103_oof)[0, 1]
            )
    pearson_vs_anchor = float(np.corrcoef(mean_bag_oof, anchor)[0, 1])
    print("\n" + "-" * 78)
    print("STANDALONE BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}  "
          f"std = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}, "
          f"d_vs_nb2103_K28 = {rae_mean_bag - NB2103_K28_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")
    print(f"   Pearson(mean_bag, anchor)      = {pearson_vs_anchor:.4f}")
    if pearson_vs_nb2103_K28 is not None:
        print(f"   Pearson(mean_bag, nb2103_K28)  = "
              f"{pearson_vs_nb2103_K28:.4f}")

    delta_vs_nb2103 = rae_mean_bag - NB2103_K28_REF
    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb2103 = rae_mean_bag < NB2103_K28_REF - DECISION_MARGIN
    flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN
    if beats_nb2103:
        standalone_verdict = "AVALON_ALONE_BEATS_NB2103_K28"
    elif flat_vs_nb2103:
        standalone_verdict = ("AVALON_ALONE_FLAT_VS_NB2103_K28_"
                              "USE_AS_DECORRELATED_BLEND_PARTNER")
    elif beats_anchor:
        standalone_verdict = ("AVALON_ALONE_BEATS_ANCHOR_BUT_"
                              "WORSE_THAN_NB2103_K28")
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        standalone_verdict = "AVALON_ALONE_FLAT_VS_ANCHOR"
    else:
        standalone_verdict = "AVALON_ALONE_HURTS_ANCHOR"
    print(f"   standalone verdict     = {standalone_verdict}")

    # ============================================================================
    # STAGE 2: SLSQP blend with nb2240 5-anchor pyramid (6 anchors total)
    # ============================================================================
    print("\n" + "=" * 78)
    print("STAGE 2: 6-ANCHOR PYRAMID  (nb2340 added to nb2240 5-anchor stack)")
    print("=" * 78)

    # Required artefacts for the 5-anchor pyramid
    needed = [
        DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
        DATA_PROCESSED / "te_nb2240_K20.npy",
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
        DATA_PROCESSED / "nb503_pred_oof.npy",
        DATA_PROCESSED / "nb562_pred_oof.npy",
        DATA_PROCESSED / "te_nb1191.npy",
        DATA_PROCESSED / "te_nb503.npy",
        DATA_PROCESSED / "te_nb562.npy",
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy",
        DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy",
    ]
    stage2_ready = all(p.exists() for p in needed)
    pyramid_summary = None
    if not stage2_ready:
        missing = [str(p) for p in needed if not p.exists()]
        print(f"   [skip] stage 2 missing artefacts: {missing}")
        pyramid_summary = {"skipped": True, "missing": missing}
    else:
        nb2240_oof = np.load(
            DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
        ).astype(np.float64)
        te_nb2240 = np.load(
            DATA_PROCESSED / "te_nb2240_K20.npy"
        ).astype(np.float64)
        chemprop_oof = np.load(
            DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
        ).astype(np.float64)
        nb503_oof = np.load(
            DATA_PROCESSED / "nb503_pred_oof.npy"
        ).astype(np.float64)
        nb562_oof = np.load(
            DATA_PROCESSED / "nb562_pred_oof.npy"
        ).astype(np.float64)
        te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
        te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
        te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
        te_chemprop_aux = te_anchor_513
        nb1191_oof = reconstruct_nb1191_oof(n_unb)

        anchors_list = [
            ("nb2340_avalon", mean_bag_oof.astype(np.float64),
             te_K28_513.astype(np.float64)),
            ("nb2240_K20",    nb2240_oof,           te_nb2240),
            ("chemprop_aux",  chemprop_oof,         te_chemprop_aux),
            ("nb1191",        nb1191_oof,           te_nb1191),
            ("nb503",         nb503_oof,            te_nb503),
            ("nb562",         nb562_oof,            te_nb562),
        ]
        print("\n[anchors]")
        indiv_rae = {}
        oof_cols, te_cols = [], []
        for disp, oof, te_arr in anchors_list:
            if oof.shape != (n_unb,):
                raise ValueError(f"{disp} oof shape {oof.shape}")
            if te_arr.shape != (n_test,):
                raise ValueError(f"{disp} te shape {te_arr.shape}")
            r = float(rae(y_unb, oof))
            indiv_rae[disp] = r
            oof_cols.append(oof)
            te_cols.append(te_arr)
            print(f"   {disp:14s} oof_RAE={r:.4f}  "
                  f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")
        P_unb = np.column_stack(oof_cols)
        P_te = np.column_stack(te_cols)
        K_anch = P_unb.shape[1]
        print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

        # ---- 5-fold scaffold CV across 5 kf_seeds ----
        print("\n" + "-" * 78)
        print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  "
              f"stretch_grid={STRETCH_GRID}")
        print("-" * 78)
        per_seed = []
        all_oofs = []
        for kf_seed in KF_SEEDS:
            pooled, oof_blend, fw, fs = cv_run_for_seed(
                P_unb, y_unb, unb_scaffolds, kf_seed
            )
            per_seed.append({
                "kf_seed": int(kf_seed),
                "pooled_rae": float(pooled),
                "fold_s": [float(x) for x in fs],
                "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
            })
            all_oofs.append(oof_blend)
            print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
                  f"mean_s={np.mean(fs):.3f}  "
                  f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
        mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
        final_oof_rae = float(rae(y_unb, mean_oof))
        pooled_rae_mean_seeds = float(np.mean(
            [r["pooled_rae"] for r in per_seed]
        ))
        pooled_rae_std_seeds = float(np.std(
            [r["pooled_rae"] for r in per_seed]
        ))
        print(f"\n[cv] mean across 5 seeds  pooled_RAE = "
              f"{pooled_rae_mean_seeds:.4f} (+/- {pooled_rae_std_seeds:.4f})")
        print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

        # ---- Deploy ----
        print("\n" + "-" * 78)
        print("DEPLOY (refit weights on 253; mean(fold_s) across all 5 seeds)")
        print("-" * 78)
        w_deploy = slsqp_simplex(P_unb, y_unb)
        blend_unb = P_unb @ w_deploy
        mu_deploy = float(blend_unb.mean())
        s_deploy = float(np.mean([
            s for r in per_seed for s in r["fold_s"]
        ]))
        in_rae_final = float(rae(
            y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)
        ))
        blend_te = P_te @ w_deploy
        deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(
            np.float32
        )
        te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
        w_str = ", ".join(
            f"{disp}={w:.4f}"
            for (disp, _, _), w in zip(anchors_list, w_deploy)
        )
        print(f"   deploy weights      = {w_str}")
        print(f"   deploy mu / s       = {mu_deploy:.4f} / {s_deploy:.4f}")
        print(f"   in-sample RAE (253) = {in_rae_final:.4f}")
        print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}")
        print(f"   te(513) mean/std    = "
              f"{deploy_te.mean():.3f}/{deploy_te.std():.3f}")

        lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
        lb_low = lb_band_est - 0.05
        lb_high = lb_band_est + 0.05
        print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = "
              f"{lb_band_est:.4f}  [{lb_low:.4f}, {lb_high:.4f}]")

        # ---- Gate vs nb2240 5-anchor pyramid (0.4598) ----
        delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_PYRAMID_REF
        gate_beat = delta_vs_nb2240 < -DECISION_MARGIN
        gate_flat = abs(delta_vs_nb2240) <= DECISION_MARGIN
        print("\n" + "-" * 78)
        print(f"GATE  (vs nb2240 5-anchor pyramid {NB2240_PYRAMID_REF:.4f}, "
              f"margin {DECISION_MARGIN})")
        print("-" * 78)
        print(f"   nb2340 6-anchor OOF (5-seed mean) = "
              f"{pooled_rae_mean_seeds:.4f}")
        print(f"   nb2240 reference                  = "
              f"{NB2240_PYRAMID_REF:.4f}")
        print(f"   delta                             = {delta_vs_nb2240:+.4f}")
        if gate_beat:
            pyramid_verdict = "AVALON_BLEND_BEATS_NB2240_PYRAMID"
        elif gate_flat:
            pyramid_verdict = "AVALON_BLEND_FLAT_VS_NB2240_PYRAMID"
        else:
            pyramid_verdict = "AVALON_BLEND_HURTS_NB2240_PYRAMID"
        print(f"   verdict                           = {pyramid_verdict}")

        pyramid_summary = {
            "skipped": False,
            "anchors": [a[0] for a in anchors_list],
            "indiv_oof_rae": indiv_rae,
            "kf_seeds": KF_SEEDS,
            "n_folds": N_FOLDS,
            "stretch_grid": STRETCH_GRID,
            "per_seed_results": per_seed,
            "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
            "pooled_rae_std_seeds": pooled_rae_std_seeds,
            "rae_of_mean_of_seed_oofs": final_oof_rae,
            "deploy_weights": [
                {"name": disp, "w": float(w)}
                for (disp, _, _), w in zip(anchors_list, w_deploy)
            ],
            "deploy_mu_blend": mu_deploy,
            "deploy_s": s_deploy,
            "in_sample_rae_overfit_bound": in_rae_final,
            "te_unb_rae_in_sample": te_unb_rae,
            "deploy_te_mean": float(deploy_te.mean()),
            "deploy_te_std": float(deploy_te.std()),
            "lb_band_estimate": lb_band_est,
            "lb_band_low": lb_low,
            "lb_band_high": lb_high,
            "lb_band_w_oof": LB_W_OOF,
            "lb_band_w_te": LB_W_TE,
            "compare_nb2240_oof": NB2240_PYRAMID_REF,
            "delta_vs_nb2240": delta_vs_nb2240,
            "gate_margin": DECISION_MARGIN,
            "gate_beat_nb2240": bool(gate_beat),
            "gate_flat_vs_nb2240": bool(gate_flat),
            "verdict_vs_nb2240": pyramid_verdict,
        }

    # ---- Save summary ----
    top28_records = [
        {
            "rank": int(rank),
            "avalon_bit_idx": int(top_idx[rank - 1]),
            "feat_name": f"Avalon1024_bit_{int(top_idx[rank - 1])}",
            "mean_abs_shap": float(top_imp[rank - 1]),
        }
        for rank in range(1, TOP_K + 1)
    ]
    summary = {
        "tag": TAG,
        "method": ("avalon_1024_alone_substrate_shap_top28_lgbm_residual_"
                   "5seed_bag_scaffold_CV_plus_6anchor_slsqp_pyramid"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feature_source": ("Avalon-1024 from rdkit.Avalon.pyAvalonTools."
                           "GetAvalonFP(nBits=1024) on standardize() output"),
        "n_avalon_bits": int(N_AVALON),
        "top_K_shap": int(TOP_K),
        "avalon_bit_density_train": float(X_train_av.mean()),
        "avalon_bit_density_test": float(X_test_av.mean()),
        "avalon_bit_density_unb": float(X_unb_av.mean()),
        "avalon_n_failed_train": int(n_failed_tr),
        "avalon_n_failed_test": int(n_failed_te),
        "shap_n_nonzero_bits": int(n_nonzero_imp),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_scaffolds_unique_unb": int(n_scaf_unique),
        "n_scaffolds_none_unb": int(n_scaf_none),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "cv_kind": "scaffold_kfold_BemisMurcko_5fold_shuffle_per_seed",
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "top28_records": top28_records,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_chemprop_aux": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb2103_K28": delta_vs_nb2103,
        "beats_chemprop_aux": bool(beats_anchor),
        "beats_nb2103_K28": bool(beats_nb2103),
        "flat_vs_nb2103_K28": bool(flat_vs_nb2103),
        "pearson_vs_anchor": pearson_vs_anchor,
        "pearson_vs_nb2103_K28": pearson_vs_nb2103_K28,
        "standalone_verdict": standalone_verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_ref": NB2103_K28_REF,
        "nb2240_pyramid_ref": NB2240_PYRAMID_REF,
        "decision_margin": DECISION_MARGIN,
        "shap_importance_path": str(shap_path),
        "top28_idx_path": str(top_path),
        "mean_bag_oof_path": str(oof_path),
        "te_avalon_path": str(te_path),
        "pyramid_stage2": pyramid_summary,
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
        "n_avalon_bits", "top_K_shap",
        "n_train", "n_test", "n_unb",
        "avalon_bit_density_unb", "shap_n_nonzero_bits",
        "rae_anchor_chemprop_aux", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_chemprop_aux", "delta_mean_bag_vs_nb2103_K28",
        "beats_chemprop_aux", "beats_nb2103_K28", "flat_vs_nb2103_K28",
        "pearson_vs_anchor", "pearson_vs_nb2103_K28",
        "standalone_verdict", "nb2103_K28_ref",
    ):
        print(f"  {k}: {res.get(k)}")
    pyr = res.get("pyramid_stage2", {})
    if pyr and not pyr.get("skipped"):
        print("\n==== STAGE 2 PYRAMID ====")
        for k in (
            "anchors", "indiv_oof_rae",
            "pooled_rae_mean_seeds", "pooled_rae_std_seeds",
            "rae_of_mean_of_seed_oofs",
            "deploy_weights", "deploy_s",
            "in_sample_rae_overfit_bound", "te_unb_rae_in_sample",
            "lb_band_estimate",
            "compare_nb2240_oof", "delta_vs_nb2240",
            "verdict_vs_nb2240",
        ):
            print(f"  {k}: {pyr.get(k)}")
