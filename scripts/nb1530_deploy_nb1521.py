"""nb1530 -- Deploy nb1521 BoB (honest outer-bag of nb1511 blend) to 513-row CSV.

Lifts nb1521's outer-bag protocol to deploy form.  Per outer seed o in
{0, 1, 7, 42, 137}:
    (a) nb1501_o deploy: 5-inner-seed CatBoost(MAE, d4, n200, lr0.05, l2=5)
        on the 112-col 4-way QUAD-PRUNED matrix (top-30 AP + top-20 MACCS +
        top-30 Mordred + top-30 chemprop-embed + pred_chembl + sim).  Each
        inner seed [o*1000+s for s in {0,1,7,42,137}]: fit a single
        CatBoost residual learner on ALL 253 unblind (target = y_unb -
        chemprop_aux[unb_idx]); predict residual on 513.  Mean-bag across
        the 5 inner-seed 513-row residuals.
    (b) nb1484_o deploy: 4 separate per-family LGBM Huber residual
        learners (AtomPair-30, MACCS-20, Mordred-30, ChempropEmbed-30,
        each with pred_chembl + sim appended -> 32/22/32/32 cols).  Per
        family, single LGBM Huber (depth=3, num_leaves=7, n_est=80,
        lr=0.05, huber alpha=1.0) at random_state=o, FIT on all 253
        unblind residuals, PREDICT 513 residual.  Naive 1/4 mean over the
        4 family 513-row residuals.
    (c) blend_o = 0.5 * nb1501_o + 0.5 * nb1484_o   (513,)

Stack 5 per-outer 513-row blends into (5, 513).  Row-level BoB MEAN +
MEDIAN.  Add to chemprop_aux te(513) to produce:
    te_nb1530_mean   = te_chemprop_aux + BoB_mean_resid_513
    te_nb1530_median = te_chemprop_aux + BoB_median_resid_513

Honest LB anchors (PRE-unblind cross-fit RAE on 253 unblind):
    nb1521 BoB MEAN   = 0.5221
    nb1521 BoB MEDIAN = 0.5231
    predicted LB ~0.525

Outputs:
    scripts/nb1530_deploy_nb1521.py                 (this file)
    data/processed/nb1530_summary.json
    data/processed/te_nb1530_mean.npy               (513,) float32
    data/processed/te_nb1530_median.npy             (513,) float32
    submissions/nb1530_deploy_nb1521_mean.csv       (513 rows)
    submissions/nb1530_deploy_nb1521_median.csv     (513 rows)
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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1530"

ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MORDRED_TE_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"

PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1521_SUMMARY = DATA_PROCESSED / "nb1521_summary.json"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
FAMILIES = ("AtomPair", "MACCS", "Mordred", "ChempropEmbed")
BLEND_W = 0.50

TE_MEAN_OUT = DATA_PROCESSED / f"te_{TAG}_mean.npy"
TE_MEDIAN_OUT = DATA_PROCESSED / f"te_{TAG}_median.npy"

SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUB_MEAN_OUT = SUB_DIR / f"{TAG}_deploy_nb1521_mean.csv"
SUB_MEDIAN_OUT = SUB_DIR / f"{TAG}_deploy_nb1521_median.csv"

HONEST_LB_MEAN = 0.5221
HONEST_LB_MEDIAN = 0.5231


def _cat_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        depth=4,
        iterations=200,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _lgbm_params(seed: int) -> dict:
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


def _load_family_te(family: str, n_test: int) -> np.ndarray:
    if family == "AtomPair":
        X = np.load(ATOMPAIR_TE_PATH)
    elif family == "MACCS":
        X = np.load(MACCS_TE_PATH)
    elif family == "Mordred":
        X = np.load(MORDRED_TE_PATH).astype(np.float32)
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
        col_med = np.nanmedian(X, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
        bad = ~np.isfinite(X)
        if bad.any():
            idx_r, idx_c = np.where(bad)
            X[idx_r, idx_c] = col_med[idx_c]
        if X.shape[0] != n_test:
            raise ValueError(f"Mordred cache shape mismatch: {X.shape}")
        return X
    elif family == "ChempropEmbed":
        X = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
        if X.shape[0] != n_test:
            raise ValueError(f"Chemprop embed cache shape mismatch: {X.shape}")
        return X
    else:
        raise ValueError(family)
    if X.shape[0] != n_test:
        raise ValueError(f"{family} cache shape mismatch: {X.shape}")
    return X.astype(np.float32)


def _build_family_X513(X_fam_te: np.ndarray,
                       top_idx: np.ndarray,
                       pred_chembl: np.ndarray,
                       sim_chembl: np.ndarray) -> np.ndarray:
    X_fam_pruned = X_fam_te[:, top_idx].astype(np.float32)
    X = np.concatenate(
        [
            X_fam_pruned,
            pred_chembl.reshape(-1, 1),
            sim_chembl.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    return X


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Deploy nb1521 BoB (honest outer-bag of nb1511 blend) "
          f"to 513 CSV")
    print(f"       outer seeds      = {OUTER_SEEDS}")
    print(f"       inner base seeds = {INNER_BASE_SEEDS}")
    print(f"       inner_seeds(o)   = [o*1000 + s for s in base]  "
          f"(nb1501 inner bag)")
    print(f"       nb1484 family seed(o) = o      (LGBM random_state)")
    print(f"       families         = {FAMILIES}")
    print(f"       blend_w_nb1501   = {BLEND_W:.2f}  "
          f"blend_w_nb1484 = {1.0 - BLEND_W:.2f}")
    print(f"       honest LB anchors: mean={HONEST_LB_MEAN:.4f}  "
          f"median={HONEST_LB_MEDIAN:.4f}")
    print("=" * 78)

    # ---- Load test + truth slice ----
    te_df = load_test()
    n_test = len(te_df)
    print(f"[load] load_test() -> {n_test} rows")

    smiles_col = "smiles" if "smiles" in te_df.columns else "SMILES"
    name_col = None
    for cand in ("molecule_name", "Molecule Name", "name", "compound_id"):
        if cand in te_df.columns:
            name_col = cand
            break
    if name_col is None:
        raise KeyError(
            f"No Molecule-Name column found; cols = {list(te_df.columns)}"
        )
    print(f"[load] smiles col = {smiles_col!r}   name col = {name_col!r}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape != (n_test,):
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape}"
        )
    print(f"[load] te_chemprop_aux: shape={te_anchor_513.shape}  "
          f"mean={te_anchor_513.mean():.4f}  std={te_anchor_513.std():.4f}")

    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unb_idx: {unb_idx.shape}   y_unb: {y_unb.shape}")

    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} in_RAE on unb = {rae_anchor:.4f}")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  "
          f"std={residual_unb.std():.4f}")

    # ---- Load pinned SHAP top-idx per family from nb1484 ----
    if not NB1484_SUMMARY.exists():
        raise FileNotFoundError(f"nb1484 summary missing: {NB1484_SUMMARY}")
    with open(NB1484_SUMMARY) as f:
        nb1484 = json.load(f)
    fam_top_idx: dict[str, np.ndarray] = {}
    for fam_rec in nb1484["families"]:
        fam_top_idx[fam_rec["family"]] = np.asarray(
            fam_rec["top_idx_ranked"], dtype=int
        )
        print(f"[pin][{fam_rec['family']}] top-"
              f"{len(fam_top_idx[fam_rec['family']])} indices loaded "
              f"(n_fam_bits={fam_rec['n_fam_bits']})")
    for fam in FAMILIES:
        if fam not in fam_top_idx:
            raise KeyError(f"nb1484 summary missing family {fam}")

    # ---- Load ChEMBL kNN cached features for 513 ----
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape != (n_test,) or sim_chembl_513.shape != (n_test,):
        raise ValueError(
            f"chembl cache shapes mismatch: "
            f"pred={pred_chembl_513.shape} sim={sim_chembl_513.shape}"
        )
    print(f"[chembl] pred_chembl_513: mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[chembl] sim_chembl_513:  mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Build per-family X_513 + X_unb (pinned cols) -- 4 family matrices ----
    print("\n" + "-" * 78)
    print("BUILD per-family PRUNED feature matrices (pinned SHAP cols)")
    print("-" * 78)
    X_513_by_fam: dict[str, np.ndarray] = {}
    X_unb_by_fam: dict[str, np.ndarray] = {}
    for fam in FAMILIES:
        X_fam_te = _load_family_te(fam, n_test)
        X_513 = _build_family_X513(
            X_fam_te, fam_top_idx[fam],
            pred_chembl_513, sim_chembl_513,
        )
        X_unb = X_513[unb_idx]
        X_513_by_fam[fam] = X_513
        X_unb_by_fam[fam] = X_unb
        print(f"   {fam:<14s} X_513 = {X_513.shape}  X_unb = {X_unb.shape}")

    # ---- Build nb1501 112-col QUAD-PRUNED matrix on 513 ----
    print("\n" + "-" * 78)
    print("BUILD nb1501 112-col QUAD-PRUNED feature matrix")
    print("-" * 78)
    # Get per-family raw pruned slices on 513 (no pred_chembl/sim per family)
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    X_ap_pruned_513 = X_ap_te[:, fam_top_idx["AtomPair"]].astype(np.float32)

    X_maccs_te = np.load(MACCS_TE_PATH)
    X_maccs_pruned_513 = X_maccs_te[:, fam_top_idx["MACCS"]].astype(np.float32)

    X_mord_te = _load_family_te("Mordred", n_test)
    X_mord_pruned_513 = X_mord_te[:, fam_top_idx["Mordred"]].astype(np.float32)

    X_emb_te = _load_family_te("ChempropEmbed", n_test)
    X_emb_pruned_513 = X_emb_te[:, fam_top_idx["ChempropEmbed"]].astype(np.float32)

    X_quad_513 = np.concatenate(
        [
            X_ap_pruned_513,
            X_maccs_pruned_513,
            X_mord_pruned_513,
            X_emb_pruned_513,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    X_quad_unb = X_quad_513[unb_idx]
    n_top_ap = int(len(fam_top_idx["AtomPair"]))
    n_top_maccs = int(len(fam_top_idx["MACCS"]))
    n_top_mord = int(len(fam_top_idx["Mordred"]))
    n_top_embed = int(len(fam_top_idx["ChempropEmbed"]))
    expected_quad = n_top_ap + n_top_maccs + n_top_mord + n_top_embed + 2
    if X_quad_513.shape[1] != expected_quad:
        raise ValueError(
            f"quad feat_dim {X_quad_513.shape[1]} != expected {expected_quad}"
        )
    print(f"   X_quad_513 = {X_quad_513.shape}  "
          f"X_quad_unb = {X_quad_unb.shape}  "
          f"(top-{n_top_ap} AP + top-{n_top_maccs} MACCS + "
          f"top-{n_top_mord} Mordred + top-{n_top_embed} Embed + 2 ChEMBL)")

    # ---- Outer x (nb1501 inner-bag + nb1484 family-bag) deploy ----
    print("\n" + "=" * 78)
    print("OUTER x [nb1501 (5-inner CatBoost) + nb1484 (4-family LGBM)] DEPLOY")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)

    outer_blend_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    outer_nb1501_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    outer_nb1484_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_records = []
    per_outer_inner_seeds_used: list[list[int]] = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds_used.append(inner_seeds)
        print(f"\n   --- outer seed {o}  "
              f"inner CatBoost seeds = {inner_seeds}  "
              f"nb1484 family seed = {o} ---")

        # ----- (a) nb1501_o : 5-inner CatBoost bag on 112-col QUAD ------
        per_inner_resid_513 = np.zeros((n_inner, n_test), dtype=np.float64)
        per_inner_resid_std_513: list[float] = []
        for ii, isd in enumerate(inner_seeds):
            ts = time.time()
            mdl = CatBoostRegressor(**_cat_params(isd))
            mdl.fit(X_quad_unb, residual_unb)
            r_513 = mdl.predict(X_quad_513)
            per_inner_resid_513[ii] = r_513
            per_inner_resid_std_513.append(float(r_513.std()))
            print(f"     [nb1501] outer {o:3d}  inner {isd:6d}: "
                  f"resid_513 mean={r_513.mean():+.4f}  "
                  f"std={r_513.std():.4f}  "
                  f"wall = {time.time() - ts:.1f}s")
        nb1501_o_resid_513 = per_inner_resid_513.mean(axis=0)
        outer_nb1501_513[oi] = nb1501_o_resid_513
        # in-sample diagnostic on unblind
        te_nb1501_o = te_anchor_513 + nb1501_o_resid_513
        in_rae_nb1501_o = float(rae(y_unb, te_nb1501_o[unb_idx]))
        print(f"     [nb1501_o = 5-inner mean_bag] resid_513 "
              f"mean={nb1501_o_resid_513.mean():+.4f}  "
              f"std={nb1501_o_resid_513.std():.4f}  "
              f"in_RAE_unb (in-sample) = {in_rae_nb1501_o:.4f}")

        # ----- (b) nb1484_o : 4 family LGBM Huber, naive 1/4 mean ------
        fam_resid_513: dict[str, np.ndarray] = {}
        fam_records = []
        for fam in FAMILIES:
            X_513 = X_513_by_fam[fam]
            X_unb = X_unb_by_fam[fam]
            ts = time.time()
            mdl = LGBMRegressor(**_lgbm_params(int(o)))
            mdl.fit(X_unb, residual_unb)
            r_513 = mdl.predict(X_513)
            fam_resid_513[fam] = r_513
            # in-sample diagnostic on unblind
            te_fam = te_anchor_513 + r_513
            in_rae_fam = float(rae(y_unb, te_fam[unb_idx]))
            fam_records.append({
                "family": fam,
                "lgbm_seed": int(o),
                "resid_513_mean": float(r_513.mean()),
                "resid_513_std": float(r_513.std()),
                "in_RAE_unb_deploy_in_sample": in_rae_fam,
                "wall_sec": round(time.time() - ts, 2),
            })
            print(f"     [nb1484] outer {o:3d}  fam {fam:<14s} "
                  f"seed = {o:3d}: resid_513 mean={r_513.mean():+.4f}  "
                  f"std={r_513.std():.4f}  "
                  f"in_RAE_unb (in-sample) = {in_rae_fam:.4f}  "
                  f"wall = {time.time() - ts:.1f}s")
        nb1484_o_resid_513 = np.mean(
            np.stack([fam_resid_513[fam] for fam in FAMILIES], axis=0),
            axis=0,
        )
        outer_nb1484_513[oi] = nb1484_o_resid_513
        te_nb1484_o = te_anchor_513 + nb1484_o_resid_513
        in_rae_nb1484_o = float(rae(y_unb, te_nb1484_o[unb_idx]))
        print(f"     [nb1484_o = naive 1/4 mean] resid_513 "
              f"mean={nb1484_o_resid_513.mean():+.4f}  "
              f"std={nb1484_o_resid_513.std():.4f}  "
              f"in_RAE_unb (in-sample) = {in_rae_nb1484_o:.4f}")

        # ----- (c) blend_o on 513 -----
        blend_o_513 = (
            BLEND_W * nb1501_o_resid_513
            + (1.0 - BLEND_W) * nb1484_o_resid_513
        )
        outer_blend_513[oi] = blend_o_513
        te_blend_o = te_anchor_513 + blend_o_513
        in_rae_blend_o = float(rae(y_unb, te_blend_o[unb_idx]))
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds_nb1501": inner_seeds,
            "nb1484_family_seed": int(o),
            "per_inner_resid_std_513_nb1501": per_inner_resid_std_513,
            "nb1501_o_resid_513_mean": float(nb1501_o_resid_513.mean()),
            "nb1501_o_resid_513_std": float(nb1501_o_resid_513.std()),
            "in_RAE_unb_nb1501_o_in_sample": in_rae_nb1501_o,
            "families": fam_records,
            "nb1484_o_resid_513_mean": float(nb1484_o_resid_513.mean()),
            "nb1484_o_resid_513_std": float(nb1484_o_resid_513.std()),
            "in_RAE_unb_nb1484_o_in_sample": in_rae_nb1484_o,
            "blend_w": float(BLEND_W),
            "blend_resid_513_mean": float(blend_o_513.mean()),
            "blend_resid_513_std": float(blend_o_513.std()),
            "in_RAE_unb_blend_o_in_sample": in_rae_blend_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  blend_o = "
              f"{BLEND_W:.2f}*nb1501_o + {1.0 - BLEND_W:.2f}*nb1484_o  "
              f"resid_513 mean={blend_o_513.mean():+.4f}  "
              f"std={blend_o_513.std():.4f}  "
              f"in_RAE_unb (in-sample) = {in_rae_blend_o:.4f}  "
              f"(outer wall = {time.time() - t_outer:.1f}s)")

    # ---- BoB row-level aggregations across 5 per-outer 513-row blends ----
    print("\n" + "=" * 78)
    print("BoB ROW-LEVEL AGGREGATIONS over 5 per-outer 513-row blends")
    print("=" * 78)
    bob_mean_resid_513 = outer_blend_513.mean(axis=0)
    bob_median_resid_513 = np.median(outer_blend_513, axis=0)
    print(f"[bob mean]   resid_513 mean={bob_mean_resid_513.mean():+.4f}  "
          f"std={bob_mean_resid_513.std():.4f}")
    print(f"[bob median] resid_513 mean={bob_median_resid_513.mean():+.4f}  "
          f"std={bob_median_resid_513.std():.4f}")

    te_nb1530_mean = te_anchor_513 + bob_mean_resid_513
    te_nb1530_median = te_anchor_513 + bob_median_resid_513

    te_mean_stats = {
        "mean": float(te_nb1530_mean.mean()),
        "std": float(te_nb1530_mean.std()),
        "min": float(te_nb1530_mean.min()),
        "max": float(te_nb1530_mean.max()),
    }
    te_median_stats = {
        "mean": float(te_nb1530_median.mean()),
        "std": float(te_nb1530_median.std()),
        "min": float(te_nb1530_median.min()),
        "max": float(te_nb1530_median.max()),
    }
    print(f"[te mean]   mean={te_mean_stats['mean']:.4f}  "
          f"std={te_mean_stats['std']:.4f}  "
          f"min={te_mean_stats['min']:.4f}  max={te_mean_stats['max']:.4f}")
    print(f"[te median] mean={te_median_stats['mean']:.4f}  "
          f"std={te_median_stats['std']:.4f}  "
          f"min={te_median_stats['min']:.4f}  max={te_median_stats['max']:.4f}")

    in_rae_mean_unb = float(rae(y_unb, te_nb1530_mean[unb_idx]))
    in_rae_median_unb = float(rae(y_unb, te_nb1530_median[unb_idx]))
    print(f"[in]  te_nb1530_mean   in_RAE on unb (in-sample) = "
          f"{in_rae_mean_unb:.4f}   "
          f"(honest LB anchor = {HONEST_LB_MEAN:.4f})")
    print(f"[in]  te_nb1530_median in_RAE on unb (in-sample) = "
          f"{in_rae_median_unb:.4f}   "
          f"(honest LB anchor = {HONEST_LB_MEDIAN:.4f})")

    # ---- Save te + submission CSVs ----
    np.save(TE_MEAN_OUT, te_nb1530_mean.astype(np.float32))
    np.save(TE_MEDIAN_OUT, te_nb1530_median.astype(np.float32))
    print(f"[save] {TE_MEAN_OUT}")
    print(f"[save] {TE_MEDIAN_OUT}")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    smiles_arr = te_df[smiles_col].astype(str).to_numpy()
    names_arr = te_df[name_col].astype(str).to_numpy()

    out_mean = pd.DataFrame({
        "SMILES": smiles_arr,
        "Molecule Name": names_arr,
        "pEC50": te_nb1530_mean.astype(np.float64),
    })
    if len(out_mean) != n_test:
        raise ValueError(f"CSV row mismatch (mean): {len(out_mean)} vs {n_test}")
    out_mean.to_csv(SUB_MEAN_OUT, index=False)
    print(f"[save] {SUB_MEAN_OUT}  rows={len(out_mean)}  "
          f"cols={list(out_mean.columns)}")

    out_median = pd.DataFrame({
        "SMILES": smiles_arr,
        "Molecule Name": names_arr,
        "pEC50": te_nb1530_median.astype(np.float64),
    })
    if len(out_median) != n_test:
        raise ValueError(
            f"CSV row mismatch (median): {len(out_median)} vs {n_test}"
        )
    out_median.to_csv(SUB_MEDIAN_OUT, index=False)
    print(f"[save] {SUB_MEDIAN_OUT}  rows={len(out_median)}  "
          f"cols={list(out_median.columns)}")

    # ---- Optional: Pearson sanity vs nb1521 BoB OOF (253 only) ----
    pearson_mean_unb_vs_nb1521_mean = None
    pearson_median_unb_vs_nb1521_median = None
    bob_mean_oof_p = DATA_PROCESSED / "nb1521_bob_mean_oof.npy"
    bob_median_oof_p = DATA_PROCESSED / "nb1521_bob_median_oof.npy"
    if bob_mean_oof_p.exists():
        v = np.load(bob_mean_oof_p).astype(np.float64)
        if v.shape[0] == n_unb:
            try:
                pearson_mean_unb_vs_nb1521_mean = float(np.corrcoef(
                    te_nb1530_mean[unb_idx], v
                )[0, 1])
            except Exception:
                pearson_mean_unb_vs_nb1521_mean = None
    if bob_median_oof_p.exists():
        v = np.load(bob_median_oof_p).astype(np.float64)
        if v.shape[0] == n_unb:
            try:
                pearson_median_unb_vs_nb1521_median = float(np.corrcoef(
                    te_nb1530_median[unb_idx], v
                )[0, 1])
            except Exception:
                pearson_median_unb_vs_nb1521_median = None
    if pearson_mean_unb_vs_nb1521_mean is not None:
        print(f"[sanity] Pearson(te_mean[unb],   nb1521_bob_mean_oof)   = "
              f"{pearson_mean_unb_vs_nb1521_mean:.4f}")
    if pearson_median_unb_vs_nb1521_median is not None:
        print(f"[sanity] Pearson(te_median[unb], nb1521_bob_median_oof) = "
              f"{pearson_median_unb_vs_nb1521_median:.4f}")

    summary = {
        "tag": TAG,
        "deploy_of": "nb1521",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te",
        "anchor_path": str(ANCHOR_TE_PATH),
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds_used,
        "families_order": list(FAMILIES),
        "families_top_idx_pinned_from_nb1484": {
            fam: [int(b) for b in fam_top_idx[fam].tolist()]
            for fam in FAMILIES
        },
        "blend_w_nb1501": float(BLEND_W),
        "blend_w_nb1484": float(1.0 - BLEND_W),
        "nb1501_model": "CatBoost(MAE, d4, n200, lr0.05, l2=5)",
        "nb1484_model": "LGBM(huber alpha=1.0, d3, leaves=7, n_est=80, lr=0.05)",
        "nb1501_inner_bag_size": int(n_inner),
        "nb1484_family_aggregation": "naive_1_4_mean",
        "n_top_atompair": n_top_ap,
        "n_top_maccs": n_top_maccs,
        "n_top_mordred": n_top_mord,
        "n_top_chemprop_embed": n_top_embed,
        "feat_dim_nb1501_quad": int(X_quad_513.shape[1]),
        "feat_dim_nb1484_per_family": {
            fam: int(X_513_by_fam[fam].shape[1]) for fam in FAMILIES
        },
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "rae_anchor_chemprop_aux_in_RAE_unb": rae_anchor,
        "residual_mean": float(residual_unb.mean()),
        "residual_std": float(residual_unb.std()),
        "per_outer_records": per_outer_records,
        "te_mean_stats": te_mean_stats,
        "te_median_stats": te_median_stats,
        "in_RAE_unb_deploy_in_sample_mean": in_rae_mean_unb,
        "in_RAE_unb_deploy_in_sample_median": in_rae_median_unb,
        "honest_LB_anchor_mean_nb1521_bob_mean": HONEST_LB_MEAN,
        "honest_LB_anchor_median_nb1521_bob_median": HONEST_LB_MEDIAN,
        "predicted_LB_mean": HONEST_LB_MEAN + 0.003,
        "predicted_LB_median": HONEST_LB_MEDIAN + 0.003,
        "pearson_te_mean_unb_vs_nb1521_bob_mean_oof":
            pearson_mean_unb_vs_nb1521_mean,
        "pearson_te_median_unb_vs_nb1521_bob_median_oof":
            pearson_median_unb_vs_nb1521_median,
        "te_mean_path": str(TE_MEAN_OUT),
        "te_median_path": str(TE_MEDIAN_OUT),
        "csv_mean_path": str(SUB_MEAN_OUT),
        "csv_median_path": str(SUB_MEDIAN_OUT),
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    sum_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {sum_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_test", "n_unb",
        "rae_anchor_chemprop_aux_in_RAE_unb",
        "feat_dim_nb1501_quad",
        "feat_dim_nb1484_per_family",
        "te_mean_stats", "te_median_stats",
        "in_RAE_unb_deploy_in_sample_mean",
        "in_RAE_unb_deploy_in_sample_median",
        "honest_LB_anchor_mean_nb1521_bob_mean",
        "honest_LB_anchor_median_nb1521_bob_median",
        "predicted_LB_mean", "predicted_LB_median",
        "pearson_te_mean_unb_vs_nb1521_bob_mean_oof",
        "pearson_te_median_unb_vs_nb1521_bob_median_oof",
        "te_mean_path", "te_median_path",
        "csv_mean_path", "csv_median_path",
    ):
        print(f"  {k}: {res.get(k)}")
