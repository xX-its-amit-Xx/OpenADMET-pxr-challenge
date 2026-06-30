"""nb1610 -- DEPLOY nb1590 (10-outer-seed CatBoost + LGBM blend) to 513.

COMPACT VARIANT OF nb1600: 10 outer seeds instead of 40, same recipe otherwise.
Original nb1600 40-seed run died at ~outer 10 (no te_nb1600.npy produced).
ORIGINAL DOCSTRING BELOW.

nb1600 -- DEPLOY nb1590 (40-outer-seed CatBoost + LGBM blend) to 513.

PROTOCOL (handoff from nb1590 / nb1561 / nb1560):
    1. Anchor    = te_chemprop_aux.npy on 513 (PRE-unblind, in_RAE 0.6216 on unb).
    2. Features  =
         nb1561 combined 117-col matrix (top-25 AP, top-20 MACCS, top-20 Mordred,
             top-20 ChempropEmbed, top-30 Avalon, pred_chembl_pec50, mean_sim)
             -- on 513 and 253 unblind.
         nb1560 5 per-family pruned matrices (each top-K family bits +
             pred_chembl_pec50 + mean_sim) -- on 513 and 253 unblind.
    3. For each outer seed o in {0, 1, 2, ..., 39}:
         CatBoost track (nb1561 recipe):
             - 5 inner seeds = [o*1000 + s for s in [0, 1, 7, 42, 137]]
             - For each inner seed: fit CatBoost(MAE, d4, n200, lr0.05, l2=5)
               on (X_combined_unb_117, residual=y_unb - anchor_unb),
               then predict residual on 513.
             - cat_resid_513_o = mean of 5 inner residual_513 vectors.
         LGBM track (nb1560 recipe):
             - 5 inner seeds = [o*1000 + s for s in [0, 1, 2, 3, 4]]
             - For each family in [AtomPair-25, MACCS-20, Mordred-20,
               ChempropEmbed-20, Avalon-30]:
                  For each inner seed:
                      fit LGBM(huber alpha=1, d3, leaves7, n80, lr0.05)
                      on (X_family_unb, residual);
                      predict residual on 513.
                  family_resid_513 = mean of 5 inner residual_513 vectors.
             - lgb_resid_513_o = mean over 5 families of family_resid_513.
         Blend:
             blend_o_513 = 0.55 * cat_resid_513_o + 0.45 * lgb_resid_513_o
    4. BoB MEAN  = mean over 40 outer seeds of blend_o_513 (shape (513,))
    5. te_nb1600 = te_chemprop_aux_513 + BoB_mean_resid_513
    6. Save:
         data/processed/te_nb1600.npy                 (513,) float32
         data/processed/nb1600_summary.json
         submissions/nb1600_deploy_nb1590.csv         SMILES, Molecule Name, pEC50

Honest LB anchor (nb1590 BoB MEAN @ 5 outer seeds, 0.55/0.45 blend): 0.5141
   (task brief; summary file shows 0.5191; 40-seed run will report own number.)
Predicted LB ~ 0.517.

If 40 outer seeds is too slow, fall back to 10 outer seeds and report wall time.
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

TAG = "nb1610"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# 10 outer seeds (compact fallback of nb1600's 40).
OUTER_SEEDS = list(range(10))
CAT_INNER_OFFSETS = [0, 1, 7, 42, 137]
LGB_INNER_OFFSETS = [0, 1, 2, 3, 4]

# Blend weight per task brief (matches nb1590 fixed-weight scheme).
W_CAT = 0.55
W_LGB = 0.45

# Cached feature paths.
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

# Top-K summaries.
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR_NB1590 = 0.5141


# ---------------------------------------------------------------------------
# Model params (verbatim from nb1561 / nb1560).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Feature loaders.
# ---------------------------------------------------------------------------
def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X = np.load(mte_p).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X.shape} vs n_test={n_test_expected}"
        )
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _compute_shap_importance(X: np.ndarray, residual: np.ndarray, seed: int = 0):
    """Same SHAP routine as nb1560 for per-family top-K pruning."""
    mdl = LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X, residual)
    try:
        import shap
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        imp = np.abs(sv).mean(axis=0)
        return imp.astype(np.float64), "shap_tree_explainer"
    except Exception as e:
        print(f"   [shap] WARN: shap failed ({e}); falling back to LGBM gain")
        imp = mdl.booster_.feature_importance(importance_type="gain")
        return imp.astype(np.float64), "lgbm_gain_fallback"


def _build_family_pruned_513(family: str,
                              X_fam_513: np.ndarray,
                              pred_chembl_513: np.ndarray,
                              sim_chembl_513: np.ndarray,
                              residual_unb: np.ndarray,
                              unb_idx: np.ndarray,
                              top_k: int) -> tuple[np.ndarray, dict]:
    """Replicates nb1560 _build_family_pruned_matrix but for the full 513 stack.

    Returns:
        X_pruned_513 (513, top_k_eff + 2) float32 -- top-K family bits + ChEMBL.
        meta dict
    """
    n_fam = int(X_fam_513.shape[1])
    X_fam_unb = X_fam_513[unb_idx]
    X_full_unb = np.concatenate(
        [
            X_fam_unb,
            pred_chembl_513[unb_idx].reshape(-1, 1),
            sim_chembl_513[unb_idx].reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    imp_full, imp_src = _compute_shap_importance(X_full_unb, residual_unb,
                                                  seed=0)
    fam_imp = imp_full[:n_fam]
    top_k_eff = min(top_k, n_fam)
    top_order = np.argsort(-fam_imp)
    top_idx = top_order[:top_k_eff].astype(int)
    X_fam_pruned_513 = X_fam_513[:, top_idx]
    X_pruned_513 = np.concatenate(
        [
            X_fam_pruned_513,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    meta = {
        "family": family,
        "n_fam_bits": n_fam,
        "top_k": int(top_k_eff),
        "feat_dim_pruned": int(X_pruned_513.shape[1]),
        "shap_source": imp_src,
        "top_idx_sorted_asc": [int(b) for b in np.sort(top_idx).tolist()],
    }
    return X_pruned_513, meta


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1590 (10-outer CatBoost+LGBM blend) -> 513 [COMPACT]")
    print(f"        outer seeds       = {len(OUTER_SEEDS)}  ({OUTER_SEEDS[:5]} ... "
          f"{OUTER_SEEDS[-1]})")
    print(f"        blend weights     = cat={W_CAT}  lgb={W_LGB}")
    print(f"        honest LB anchor  = {HONEST_LB_ANCHOR_NB1590}")
    print("=" * 78)

    # ---- Test metadata ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    else:
        cand = [c for c in te.columns if "name" in c.lower()]
        if not cand:
            raise KeyError(
                f"No Molecule Name column found in test ({te.columns.tolist()})"
            )
        mol_names = te[cand[0]].astype(str).tolist()
    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"Anchor missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape} vs {n_test}")
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual_unb = (y_unb - anchor_unb).astype(np.float64)
    print(f"[anchor] te_{ANCHOR}  in_RAE(unb_idx)={rae_anchor:.4f}  (ref 0.6216)")
    print(f"[resid]  mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Top-K indices from K-grid winners ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f:
        sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f:
        sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f:
        sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f:
        sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    K_Avalon_used = int(len(top_avalon_bit_idx))

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    print(f"[reuse] top-{n_top_ap}    AtomPair bits  (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}    MACCS bits     (nb1352)")
    print(f"[reuse] top-{n_top_mord}    Mordred cols   (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}    ChempropEmbed  (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}    Avalon bits    (nb1392 SHAP K=30)")

    # ---- ChEMBL kNN features (cached 513) ----
    if not PRED_CHEMBL_513_PATH.exists() or not SIM_CHEMBL_513_PATH.exists():
        raise FileNotFoundError("missing pred_chembl_pec50_513 / sim_chembl_513")
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != n_test or sim_chembl_513.shape[0] != n_test:
        raise ValueError("ChEMBL feature shape mismatch with n_test")
    print(f"[chembl] pred mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[chembl] sim  mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Load raw family caches on 513 ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)

    # ---- Build nb1561 COMBINED 117-col matrix on 513 (and slice 253) ----
    print("\n" + "-" * 78)
    print("BUILD nb1561 COMBINED 117-col matrix on 513 + 253")
    print("-" * 78)
    X_combined_513 = np.concatenate(
        [
            X_ap_te[:, top_ap_bit_idx],
            X_maccs_te[:, top_maccs_bit_idx],
            X_mord_te[:, top_mord_col_idx],
            X_emb_te[:, top_embed_col_idx],
            X_av_te[:, top_avalon_bit_idx],
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    X_combined_unb = X_combined_513[unb_idx].astype(np.float32)
    feat_dim_cat = int(X_combined_513.shape[1])
    expected_dim_cat = (n_top_ap + n_top_maccs + n_top_mord
                        + n_top_embed + n_top_avalon + 2)
    if feat_dim_cat != expected_dim_cat:
        raise ValueError(
            f"feat_dim_cat {feat_dim_cat} != expected {expected_dim_cat}"
        )
    print(f"   X_combined_513 shape = {X_combined_513.shape}  "
          f"(expected dim {expected_dim_cat})")
    print(f"   X_combined_unb shape = {X_combined_unb.shape}")

    # ---- Build nb1560 5 per-family pruned matrices on 513 ----
    print("\n" + "-" * 78)
    print("BUILD nb1560 5 PER-FAMILY PRUNED MATRICES on 513 (SHAP seed=0)")
    print("-" * 78)
    family_specs = [
        ("AtomPair", X_ap_te, 25),
        ("MACCS", X_maccs_te, 20),
        ("Mordred", X_mord_te, 20),
        ("ChempropEmbed", X_emb_te, 20),
        ("Avalon", X_av_te, 30),
    ]
    family_matrices_513: dict[str, np.ndarray] = {}
    family_matrices_unb: dict[str, np.ndarray] = {}
    family_meta: dict[str, dict] = {}
    for fam_name, X_fam_te, top_k in family_specs:
        X_pruned_513, meta = _build_family_pruned_513(
            family=fam_name,
            X_fam_513=X_fam_te,
            pred_chembl_513=pred_chembl_513,
            sim_chembl_513=sim_chembl_513,
            residual_unb=residual_unb,
            unb_idx=unb_idx,
            top_k=top_k,
        )
        family_matrices_513[fam_name] = X_pruned_513
        family_matrices_unb[fam_name] = X_pruned_513[unb_idx].astype(np.float32)
        family_meta[fam_name] = meta
        print(f"   {fam_name:<14s} fam_bits={meta['n_fam_bits']:4d}  "
              f"top_k={meta['top_k']:3d}  "
              f"PRUNED dim={meta['feat_dim_pruned']:3d}  "
              f"shap_src={meta['shap_source']}")

    # ---- Main loop: 40 outer seeds ----
    print("\n" + "=" * 78)
    print(f"OUTER LOOP x {len(OUTER_SEEDS)} seeds  "
          f"(CatBoost 5-inner deploy bag + LGBM 5-inner x 5-family deploy bag)")
    print("=" * 78)

    n_outer = len(OUTER_SEEDS)
    per_outer_records = []
    outer_blend_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    outer_cat_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    outer_lgb_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        cat_inner_seeds = [int(o) * 1000 + int(s) for s in CAT_INNER_OFFSETS]
        lgb_inner_seeds = [int(o) * 1000 + int(s) for s in LGB_INNER_OFFSETS]

        # ---- CatBoost track (deploy fit on all 253) ----
        cat_resid_513 = np.zeros((len(cat_inner_seeds), n_test), dtype=np.float64)
        cat_in_rae: list[float] = []
        for si, s_inner in enumerate(cat_inner_seeds):
            mdl = CatBoostRegressor(**_cat_params(s_inner))
            mdl.fit(X_combined_unb, residual_unb)
            resid_in = mdl.predict(X_combined_unb)
            cat_in_rae.append(float(rae(y_unb, anchor_unb + resid_in)))
            cat_resid_513[si] = mdl.predict(X_combined_513)
        cat_resid_o = cat_resid_513.mean(axis=0)

        # ---- LGBM track (5 families x 5 seeds) ----
        lgb_resid_513 = np.zeros((len(family_specs), n_test), dtype=np.float64)
        lgb_family_in_rae: dict[str, list[float]] = {}
        for fi, (fam_name, _, _) in enumerate(family_specs):
            X_fam_unb = family_matrices_unb[fam_name]
            X_fam_513 = family_matrices_513[fam_name]
            fam_resid_513 = np.zeros((len(lgb_inner_seeds), n_test),
                                      dtype=np.float64)
            fam_in_rae: list[float] = []
            for si, s_inner in enumerate(lgb_inner_seeds):
                mdl = LGBMRegressor(**_lgbm_params(s_inner))
                mdl.fit(X_fam_unb, residual_unb)
                resid_in = mdl.predict(X_fam_unb)
                fam_in_rae.append(float(rae(y_unb, anchor_unb + resid_in)))
                fam_resid_513[si] = mdl.predict(X_fam_513)
            lgb_family_in_rae[fam_name] = fam_in_rae
            lgb_resid_513[fi] = fam_resid_513.mean(axis=0)
        lgb_resid_o = lgb_resid_513.mean(axis=0)

        # ---- Blend ----
        blend_o = W_CAT * cat_resid_o + W_LGB * lgb_resid_o
        outer_cat_resid_513[oi] = cat_resid_o
        outer_lgb_resid_513[oi] = lgb_resid_o
        outer_blend_resid_513[oi] = blend_o

        # ---- In-sample diagnostics on 253 ----
        in_rae_cat_o = float(rae(y_unb, anchor_unb + cat_resid_o[unb_idx]))
        in_rae_lgb_o = float(rae(y_unb, anchor_unb + lgb_resid_o[unb_idx]))
        in_rae_blend_o = float(rae(y_unb, anchor_unb + blend_o[unb_idx]))
        wall_outer = time.time() - t_outer
        per_outer_records.append({
            "outer_seed": int(o),
            "cat_inner_seeds": cat_inner_seeds,
            "lgb_inner_seeds": lgb_inner_seeds,
            "in_rae_cat_o": in_rae_cat_o,
            "in_rae_lgb_o": in_rae_lgb_o,
            "in_rae_blend_o": in_rae_blend_o,
            "cat_per_inner_in_rae": cat_in_rae,
            "lgb_per_family_in_rae": lgb_family_in_rae,
            "wall_sec": round(wall_outer, 2),
        })

        if oi < 5 or oi % 5 == 0 or oi == n_outer - 1:
            print(f"   [outer {o:3d}] in_RAE  cat={in_rae_cat_o:.4f}  "
                  f"lgb={in_rae_lgb_o:.4f}  blend={in_rae_blend_o:.4f}  "
                  f"wall={wall_outer:.1f}s")

    # ---- BoB MEAN over 40 outer seeds ----
    bob_mean_resid_513 = outer_blend_resid_513.mean(axis=0)
    bob_mean_cat_resid_513 = outer_cat_resid_513.mean(axis=0)
    bob_mean_lgb_resid_513 = outer_lgb_resid_513.mean(axis=0)
    te_nb1600 = (te_anchor_513 + bob_mean_resid_513).astype(np.float64)

    in_rae_bob_mean = float(rae(y_unb, te_nb1600[unb_idx]))
    in_rae_bob_cat = float(rae(y_unb, te_anchor_513[unb_idx]
                               + bob_mean_cat_resid_513[unb_idx]))
    in_rae_bob_lgb = float(rae(y_unb, te_anchor_513[unb_idx]
                               + bob_mean_lgb_resid_513[unb_idx]))

    print("\n" + "=" * 78)
    print(f"BoB MEAN OVER {len(OUTER_SEEDS)} OUTER SEEDS")
    print("=" * 78)
    print(f"   te_nb1600  mean={te_nb1600.mean():.4f}  "
          f"std={te_nb1600.std():.4f}  "
          f"min={te_nb1600.min():.4f}  max={te_nb1600.max():.4f}")
    print(f"   in_RAE(unb_idx)  cat_BoB={in_rae_bob_cat:.4f}  "
          f"lgb_BoB={in_rae_bob_lgb:.4f}  blend_BoB={in_rae_bob_mean:.4f}")
    print(f"   d_vs_anchor      {in_rae_bob_mean - rae_anchor:+.4f}")
    print(f"   honest LB anchor (nb1590)  = {HONEST_LB_ANCHOR_NB1590}")

    # ---- Per-outer summary ----
    per_outer_blend = np.array([r["in_rae_blend_o"] for r in per_outer_records])
    per_outer_cat = np.array([r["in_rae_cat_o"] for r in per_outer_records])
    per_outer_lgb = np.array([r["in_rae_lgb_o"] for r in per_outer_records])

    per_outer_summary = {
        "blend": {
            "mean": float(per_outer_blend.mean()),
            "std": float(per_outer_blend.std(ddof=1)),
            "min": float(per_outer_blend.min()),
            "max": float(per_outer_blend.max()),
            "median": float(np.median(per_outer_blend)),
        },
        "cat": {
            "mean": float(per_outer_cat.mean()),
            "std": float(per_outer_cat.std(ddof=1)),
            "min": float(per_outer_cat.min()),
            "max": float(per_outer_cat.max()),
            "median": float(np.median(per_outer_cat)),
        },
        "lgb": {
            "mean": float(per_outer_lgb.mean()),
            "std": float(per_outer_lgb.std(ddof=1)),
            "min": float(per_outer_lgb.min()),
            "max": float(per_outer_lgb.max()),
            "median": float(np.median(per_outer_lgb)),
        },
    }
    print(f"   per-outer in_RAE  blend  mean={per_outer_summary['blend']['mean']:.4f}  "
          f"std={per_outer_summary['blend']['std']:.4f}  "
          f"[{per_outer_summary['blend']['min']:.4f}, "
          f"{per_outer_summary['blend']['max']:.4f}]")

    # ---- Save NPY ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1600.astype(np.float32))
    print(f"\n[save] {te_path}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1590_compact.csv"
    df = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1600.astype(np.float64),
    })
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  rows={len(df)}  cols={list(df.columns)}")

    # ---- Summary JSON ----
    summary = {
        "tag": TAG,
        "parent_method": "nb1590",
        "parent_recipe": "outer-bag blend of nb1561 CatBoost + nb1560 LGBM",
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "anchor_kind": "PRE_unblind_te_513",
        "rae_anchor": rae_anchor,
        "blend_weights": {"cat": W_CAT, "lgb": W_LGB},
        "n_outer_seeds": len(OUTER_SEEDS),
        "outer_seeds": OUTER_SEEDS,
        "cat_inner_offsets": CAT_INNER_OFFSETS,
        "lgb_inner_offsets": LGB_INNER_OFFSETS,
        "cat_inner_seed_formula": "o * 1000 + offset (offsets [0,1,7,42,137])",
        "lgb_inner_seed_formula": "o * 1000 + offset (offsets [0,1,2,3,4])",
        "catboost": {"loss": "MAE", "depth": 4, "iterations": 200,
                      "learning_rate": 0.05, "l2_leaf_reg": 5.0},
        "lgbm": {"objective": "huber", "alpha": 1.0, "depth": 3,
                  "leaves": 7, "n_estimators": 80, "learning_rate": 0.05},
        "K_AP": K_AP_best,
        "K_MACCS": n_top_maccs,
        "K_Mord": K_Mord_best,
        "K_Embed": K_Embed_best,
        "K_Avalon": K_Avalon_used,
        "cat_feat_dim": feat_dim_cat,
        "family_meta": family_meta,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "te_nb1600_stats": {
            "mean": float(te_nb1600.mean()),
            "std": float(te_nb1600.std()),
            "min": float(te_nb1600.min()),
            "max": float(te_nb1600.max()),
        },
        "in_rae_unb_bob_cat": in_rae_bob_cat,
        "in_rae_unb_bob_lgb": in_rae_bob_lgb,
        "in_rae_unb_bob_blend": in_rae_bob_mean,
        "delta_blend_vs_anchor": in_rae_bob_mean - rae_anchor,
        "per_outer_summary": per_outer_summary,
        "per_outer_records": per_outer_records,
        "honest_lb_anchor_nb1590": HONEST_LB_ANCHOR_NB1590,
        "te_npy_path": str(te_path),
        "csv_path": str(csv_path),
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
        "n_test", "n_unb",
        "n_outer_seeds",
        "rae_anchor",
        "te_nb1600_stats",
        "in_rae_unb_bob_cat",
        "in_rae_unb_bob_lgb",
        "in_rae_unb_bob_blend",
        "delta_blend_vs_anchor",
        "per_outer_summary",
        "honest_lb_anchor_nb1590",
        "csv_path", "te_npy_path",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
