"""nb1681 -- DEPLOY nb1673 outer-bag-of-bags (BoB) blend to 513-row CSV.

HANDOFF FROM nb1673 (PRE-unblind cross-fit verdict, 10 outer seeds):
    BoB MEAN     pooled RAE = 0.5116
    BoB MEDIAN   pooled RAE = 0.5121
    per-outer    blend_o    = [0.5139, 0.5170, 0.5126, 0.5118, 0.5108,
                                0.5144, 0.5137, 0.5182, 0.5163, 0.5146]
    blend_o      = 0.55 * nb1561_o + 0.45 * nb1612_o
    nb1612_o     = per-outer best-of(naive 1/6 mean, SLSQP-5fold cross-fit)
                   weights/variant taken from nb1673_summary.json (locked)

PROTOCOL
    For each outer seed o in {0, 1, 2, 7, 42, 99, 137, 250, 500, 750}:
        inner_seeds = [o*1000 + s for s in [0,1,7,42,137]]

        nb1561_o deploy (CatBoost, 117-col 5-way K-tuned):
            For each s' in inner_seeds:
                CatBoost(MAE, d4, n200, lr0.05, l2=5, random_seed=s')
                Fit on ALL 253 unblind, predict residual on 513.
            per-outer mean across 5 inner deploys -> (513,)

        nb1612_o deploy (6-family residual blend, ChemBERTa K=50):
            For each family in {AtomPair K=25, MACCS K=20, Mordred K=20,
                                ChempropEmbed K=20, Avalon K=30, ChemBERTa K=50}:
                SHAP-prune top-K columns from family TE matrix (SHAP fit on
                253-residual against full family + chembl 2-col cap).
                For each s' in inner_seeds:
                    LGBM-Huber(d3, n80, lr0.05, ...) on top-K + chembl 2 cols.
                    Fit on ALL 253 unblind, predict residual on 513.
                per-family residual_513 = mean across 5 inner deploys.

            Pull per-outer locked recipe from nb1673_summary.json:
                if best_variant == "slsqp_5fold":
                    combined_resid_513 = sum_family(w_o[family] * resid_513)
                else:  # naive_1_6_mean
                    combined_resid_513 = mean across 6 families resid_513
            nb1612_o deploy 513 = chemprop_aux_513 + combined_resid_513

        blend_o_513 = 0.55 * nb1561_o_513 + 0.45 * nb1612_o_513

    Stack 10 per-outer blend vectors -> (10, 513)
    Row-level BoB MEAN across outers.

    te_nb1681 = bob_mean_513

OUTPUTS
    scripts/nb1681_deploy_nb1673.py
    data/processed/te_nb1681.npy                 (513,) float32
    data/processed/nb1681_summary.json
    submissions/nb1681_deploy_nb1673.csv         SMILES, Molecule Name, pEC50

Honest LB anchor: 0.5116 (BoB MEAN PRE-unblind cross-fit on 253 unblind).
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

TAG = "nb1681"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Outer/inner schedule -- locked to nb1673 (10 outer seeds).
OUTER_SEEDS = [0, 1, 2, 7, 42, 99, 137, 250, 500, 750]
INNER_OFFSETS = [0, 1, 7, 42, 137]

# Blend weights -- locked to nb1622/nb1632/nb1673 best grid w.
W_NB1561 = 0.55
W_NB1612 = 1.0 - W_NB1561

# nb1673 BoB cross-fit anchor (MEAN over 10 outer).
HONEST_LB_ANCHOR_MEAN = 0.5116

# nb1612 K-tuning (matches nb1612 / nb1623 / nb1632 / nb1673 recipe).
TOP_K_NB1612 = {
    "AtomPair":      25,
    "MACCS":         20,
    "Mordred":       20,
    "ChempropEmbed": 20,
    "Avalon":        30,
    "ChemBERTa":     50,
}
FAMILIES_NB1612 = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed",
                    "Avalon", "ChemBERTa"]

# Family TE caches.
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMBERTA_TE_PATH = DATA_PROCESSED / "chemberta_test_emb.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

# nb1561 5-way K-tuned indices (same summaries used by nb1561/nb1570/nb1632/nb1673).
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

# nb1673 per-outer locked recipe (variant + SLSQP weights for 10 outer seeds).
NB1673_SUMMARY = DATA_PROCESSED / "nb1673_summary.json"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"


# ----------------------------- model recipes ------------------------------
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


# ----------------------------- IO helpers ---------------------------------
def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X = np.load(mte_p).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape mismatch: {X.shape}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy_test(path: Path, n_test_expected: int,
                   zero_fill: bool = True) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    if zero_fill:
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


# ------------------------ per-family residual on 513 ----------------------
def _family_resid_513_5seed_bag(
    family: str,
    X_fam_te: np.ndarray,
    pred_chembl_513: np.ndarray,
    sim_chembl_513: np.ndarray,
    residual_unb: np.ndarray,
    unb_idx: np.ndarray,
    top_k: int,
    inner_seeds: list,
    outer_seed: int,
) -> tuple[np.ndarray, dict]:
    """SHAP-prune per outer (same recipe as nb1673._run_family_nb1612), then
    5-inner-seed LGBM-Huber bag fit on ALL 253; predict residual on 513;
    return mean across seeds.
    """
    n_fam = int(X_fam_te.shape[1])

    X_fam_unb = X_fam_te[unb_idx].astype(np.float32)
    pred_chembl_unb = pred_chembl_513[unb_idx].astype(np.float32)
    sim_chembl_unb = sim_chembl_513[unb_idx].astype(np.float32)

    X_full_unb = np.concatenate(
        [X_fam_unb,
         pred_chembl_unb.reshape(-1, 1),
         sim_chembl_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)

    imp_full, imp_src = _compute_shap_importance(
        X_full_unb, residual_unb, seed=outer_seed
    )
    fam_imp = imp_full[:n_fam]
    top_k_eff = min(top_k, n_fam)
    top_order = np.argsort(-fam_imp)
    top_idx = top_order[:top_k_eff].astype(int)

    X_pruned_unb = np.concatenate(
        [X_fam_unb[:, top_idx],
         pred_chembl_unb.reshape(-1, 1),
         sim_chembl_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_pruned_513 = np.concatenate(
        [X_fam_te[:, top_idx].astype(np.float32),
         pred_chembl_513.reshape(-1, 1),
         sim_chembl_513.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)

    n_test = X_pruned_513.shape[0]
    per_seed_513 = np.zeros((len(inner_seeds), n_test), dtype=np.float64)
    per_seed_train_mae = []
    for i, s in enumerate(inner_seeds):
        mdl = LGBMRegressor(**_lgbm_params(int(s)))
        mdl.fit(X_pruned_unb, residual_unb)
        per_seed_513[i] = mdl.predict(X_pruned_513).astype(np.float64)
        in_sample = mdl.predict(X_pruned_unb)
        per_seed_train_mae.append(
            float(np.mean(np.abs(residual_unb - in_sample)))
        )
    mean_resid_513 = per_seed_513.mean(axis=0)
    info = {
        "family": family,
        "n_fam_bits": n_fam,
        "top_k_eff": int(top_k_eff),
        "shap_source": imp_src,
        "feat_dim_pruned": int(X_pruned_unb.shape[1]),
        "per_seed_train_mae": per_seed_train_mae,
        "resid_513_mean": float(mean_resid_513.mean()),
        "resid_513_std": float(mean_resid_513.std()),
    }
    return mean_resid_513, info


# ------------------------------ main --------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1673 BoB blend (10 outer x [nb1561_o + nb1612_o]) -> 513")
    print(f"         outer seeds   = {OUTER_SEEDS}")
    print(f"         inner offsets = {INNER_OFFSETS}  (inner_seed = o*1000 + offset)")
    print(f"         w_nb1561      = {W_NB1561:.2f}   w_nb1612 = {W_NB1612:.2f}")
    print(f"         honest LB MEAN   anchor = {HONEST_LB_ANCHOR_MEAN}")
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
            raise KeyError(f"No Molecule Name column ({te.columns.tolist()})")
        mol_names = te[cand[0]].astype(str).tolist()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Anchor (PRE-unblind chemprop_aux) ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"Anchor missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch {te_anchor_513.shape}")
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual_unb = y_unb - anchor_unb
    print(f"[anchor] {ANCHOR} in_RAE(unb) = {rae_anchor:.4f}  (ref 0.6216)")
    print(f"[resid]  mean={residual_unb.mean():+.4f}  "
          f"std={residual_unb.std():.4f}")

    # ---- nb1673 per-outer locked recipe ----
    if not NB1673_SUMMARY.exists():
        raise FileNotFoundError(f"nb1673 summary missing: {NB1673_SUMMARY}")
    with open(NB1673_SUMMARY) as f:
        sum_1673 = json.load(f)
    per_outer_lock = {
        int(rec["outer_seed"]): {
            "best_variant": rec["nb1612_best_variant"],
            "slsqp_w_mean_over_folds": np.array(
                rec["slsqp_w_mean_over_folds"], dtype=np.float64
            ),
        }
        for rec in sum_1673["per_outer_records"]
    }
    print("\n[nb1673-lock] per-outer nb1612 variant + SLSQP-mean-over-folds w:")
    for o in OUTER_SEEDS:
        rec = per_outer_lock[int(o)]
        print(f"   outer {o:4d}  variant={rec['best_variant']:<16s}  "
              f"w={[round(x, 4) for x in rec['slsqp_w_mean_over_folds'].tolist()]}")

    # ---- nb1561 5-way K-tuned indices ----
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
    print(f"\n[reuse] AP={n_top_ap}  MACCS={n_top_maccs}  Mord={n_top_mord}  "
          f"Embed={n_top_embed}  Avalon={n_top_avalon}")

    # ---- Load raw 513 feature caches (shared by nb1561 + nb1612) ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test, zero_fill=False)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test, zero_fill=False)
    X_mord_te = _load_mordred_test(n_test)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test, zero_fill=True)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test, zero_fill=False)
    X_cbert_te = _load_npy_test(CHEMBERTA_TE_PATH, n_test, zero_fill=True)

    # ---- ChEMBL kNN features on 513 (cached) ----
    if not PRED_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(f"missing {PRED_CHEMBL_513_PATH}")
    if not SIM_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(f"missing {SIM_CHEMBL_513_PATH}")
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    print(f"[chembl] pred_chembl_513 mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[chembl] sim_chembl_513  mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Build the 5-way K-tuned 117-col 513 matrix for nb1561_o ----
    X_te_1561 = np.concatenate(
        [
            X_ap_te[:, top_ap_bit_idx].astype(np.float32),
            X_maccs_te[:, top_maccs_bit_idx].astype(np.float32),
            X_mord_te[:, top_mord_col_idx].astype(np.float32),
            X_emb_te[:, top_embed_col_idx].astype(np.float32),
            X_av_te[:, top_avalon_bit_idx].astype(np.float32),
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_1561 = X_te_1561.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim_1561 != expected_dim:
        raise ValueError(f"feat_dim {feat_dim_1561} != expected {expected_dim}")
    X_unb_1561 = X_te_1561[unb_idx].astype(np.float32)
    print(f"\n[nb1561] 513 K-tuned matrix: {X_te_1561.shape}  "
          f"X_unb={X_unb_1561.shape}")

    # ---- Preload all 6 nb1612 family X_fam_te ----
    fam_te_dict = {
        "AtomPair":      X_ap_te,
        "MACCS":         X_maccs_te,
        "Mordred":       X_mord_te,
        "ChempropEmbed": X_emb_te,
        "Avalon":        X_av_te,
        "ChemBERTa":     X_cbert_te,
    }
    print("[nb1612] family TE tensors loaded: "
          + ", ".join(f"{k}={fam_te_dict[k].shape[1]}" for k in FAMILIES_NB1612))

    # ---- Outer-bag deploy ----
    print("\n" + "=" * 78)
    print("OUTER-BAG DEPLOY x 10 [nb1561_o = 5-inner CatBoost bag (513-resid)]")
    print("                     [nb1612_o = 6-family per-outer-locked combine]")
    print(f"                     blend_o = {W_NB1561:.2f}*nb1561_o + "
          f"{W_NB1612:.2f}*nb1612_o")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)

    outer_nb1561_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    outer_nb1612_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    outer_blend_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_records = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(s) for s in INNER_OFFSETS]
        print("\n" + "-" * 78)
        print(f"OUTER SEED {o}  ({oi + 1}/{n_outer})  inner_seeds = {inner_seeds}")
        print("-" * 78)

        # ---- nb1561_o deploy: 5-inner CatBoost bag (residual on 513) ----
        t_1561 = time.time()
        inner_resid_513_cat = np.zeros((len(inner_seeds), n_test), dtype=np.float64)
        per_inner_cat = []
        for si, s_inner in enumerate(inner_seeds):
            mdl = CatBoostRegressor(**_cat_params(int(s_inner)))
            mdl.fit(X_unb_1561, residual_unb)
            in_sample_pred = mdl.predict(X_unb_1561)
            corr_in = anchor_unb + in_sample_pred
            in_rae_s = float(rae(y_unb, corr_in))
            resid_513_s = mdl.predict(X_te_1561).astype(np.float64)
            inner_resid_513_cat[si] = resid_513_s
            per_inner_cat.append({
                "inner_seed": int(s_inner),
                "in_sample_rae_253": in_rae_s,
                "resid_513_mean": float(resid_513_s.mean()),
                "resid_513_std": float(resid_513_s.std()),
            })
        nb1561_o_resid_513 = inner_resid_513_cat.mean(axis=0)
        nb1561_o_te_513 = te_anchor_513 + nb1561_o_resid_513
        rae_nb1561_o = float(rae(y_unb, nb1561_o_te_513[unb_idx]))
        print(f"   [nb1561_o]  per-inner in_RAE_253 = "
              f"{[round(r['in_sample_rae_253'], 4) for r in per_inner_cat]}")
        print(f"   [nb1561_o]  mean-bag in_RAE(unb) = {rae_nb1561_o:.4f}  "
              f"(wall {time.time() - t_1561:.1f}s)")

        # ---- nb1612_o deploy: 6 family residual_513, then locked combine ----
        t_1612 = time.time()
        family_resid_513 = {}
        family_info = []
        for family in FAMILIES_NB1612:
            mean_resid_513, info = _family_resid_513_5seed_bag(
                family=family,
                X_fam_te=fam_te_dict[family],
                pred_chembl_513=pred_chembl_513,
                sim_chembl_513=sim_chembl_513,
                residual_unb=residual_unb,
                unb_idx=unb_idx,
                top_k=TOP_K_NB1612[family],
                inner_seeds=inner_seeds,
                outer_seed=int(o),
            )
            family_resid_513[family] = mean_resid_513
            # diagnostic: in_RAE on 253 if this family residual applied alone
            corr_unb_fam = anchor_unb + mean_resid_513[unb_idx]
            in_rae_fam = float(rae(y_unb, corr_unb_fam))
            info["in_rae_unb_alone"] = in_rae_fam
            family_info.append(info)
            print(f"   [nb1612 fam] {family:<14s} (K={info['top_k_eff']:>3})  "
                  f"resid_513 mean={info['resid_513_mean']:+.4f}  "
                  f"std={info['resid_513_std']:.4f}  "
                  f"in_RAE_alone={in_rae_fam:.4f}")

        # apply per-outer-locked variant
        lock = per_outer_lock[int(o)]
        variant = lock["best_variant"]
        if variant == "slsqp_5fold":
            w_vec = lock["slsqp_w_mean_over_folds"]
            if len(w_vec) != len(FAMILIES_NB1612):
                raise ValueError(
                    f"weight length mismatch for outer {o}: "
                    f"{len(w_vec)} vs {len(FAMILIES_NB1612)}"
                )
            combined_resid_513 = np.zeros(n_test, dtype=np.float64)
            for f_i, family in enumerate(FAMILIES_NB1612):
                combined_resid_513 += float(w_vec[f_i]) * family_resid_513[family]
            w_log = {family: float(w_vec[f_i])
                     for f_i, family in enumerate(FAMILIES_NB1612)}
        elif variant == "naive_1_6_mean":
            P = np.stack([family_resid_513[f] for f in FAMILIES_NB1612], axis=0)
            combined_resid_513 = P.mean(axis=0)
            w_log = {family: 1.0 / len(FAMILIES_NB1612)
                     for family in FAMILIES_NB1612}
        else:
            raise ValueError(f"unknown variant for outer {o}: {variant}")

        nb1612_o_resid_513 = combined_resid_513
        nb1612_o_te_513 = te_anchor_513 + nb1612_o_resid_513
        rae_nb1612_o = float(rae(y_unb, nb1612_o_te_513[unb_idx]))
        print(f"   [nb1612_o]  variant={variant}  "
              f"in_RAE(unb)={rae_nb1612_o:.4f}  "
              f"(wall {time.time() - t_1612:.1f}s)")

        # ---- blend_o ----
        blend_o_resid_513 = (W_NB1561 * nb1561_o_resid_513
                             + W_NB1612 * nb1612_o_resid_513)
        blend_o_te_513 = te_anchor_513 + blend_o_resid_513
        rae_blend_o = float(rae(y_unb, blend_o_te_513[unb_idx]))
        print(f"   [blend_o]   {W_NB1561:.2f}*nb1561_o + "
              f"{W_NB1612:.2f}*nb1612_o  in_RAE(unb)={rae_blend_o:.4f}")

        outer_nb1561_resid_513[oi] = nb1561_o_resid_513
        outer_nb1612_resid_513[oi] = nb1612_o_resid_513
        outer_blend_513[oi] = blend_o_te_513

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": [int(x) for x in inner_seeds],
            "per_inner_cat": per_inner_cat,
            "rae_nb1561_o_in_unb": rae_nb1561_o,
            "nb1612_variant_locked": variant,
            "nb1612_w_locked": w_log,
            "family_info": family_info,
            "rae_nb1612_o_in_unb": rae_nb1612_o,
            "rae_blend_o_in_unb": rae_blend_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:4d}  total wall = {time.time() - t_outer:.1f}s")

    # ---- BoB MEAN over 10 outer blend_o vectors (513,) ----
    bob_mean_513 = outer_blend_513.mean(axis=0)
    te_nb1681 = bob_mean_513

    in_rae_mean = float(rae(y_unb, te_nb1681[unb_idx]))

    # diagnostic: per-component BoB
    bob_mean_nb1561_513 = te_anchor_513 + outer_nb1561_resid_513.mean(axis=0)
    bob_mean_nb1612_513 = te_anchor_513 + outer_nb1612_resid_513.mean(axis=0)
    rae_bob_mean_nb1561_only = float(rae(y_unb, bob_mean_nb1561_513[unb_idx]))
    rae_bob_mean_nb1612_only = float(rae(y_unb, bob_mean_nb1612_513[unb_idx]))

    print("\n" + "=" * 78)
    print("BoB DEPLOY SUMMARY  (10 outer-seed blend stack -> row-level MEAN)")
    print("=" * 78)
    per_outer_rae = [rec["rae_blend_o_in_unb"] for rec in per_outer_records]
    print(f"   per-outer blend_o in_RAE(unb) = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    arr = np.array(per_outer_rae)
    print(f"   per-outer mean = {arr.mean():.4f}  std = {arr.std():.4f}  "
          f"min/max = {arr.min():.4f} / {arr.max():.4f}")
    print()
    print(f"   te_nb1681  mean={te_nb1681.mean():.4f}  "
          f"std={te_nb1681.std():.4f}  "
          f"min={te_nb1681.min():.4f}  "
          f"max={te_nb1681.max():.4f}")
    print(f"   in_RAE(unb) MEAN   = {in_rae_mean:.4f}  "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEAN})")
    print(f"   d_mean_vs_anchor   = {in_rae_mean - rae_anchor:+.4f}")
    print(f"   (diag) BoB MEAN nb1561 only in_RAE = {rae_bob_mean_nb1561_only:.4f}")
    print(f"   (diag) BoB MEAN nb1612 only in_RAE = {rae_bob_mean_nb1612_only:.4f}")

    # ---- Save NPY ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1681.astype(np.float32))
    print(f"\n[save] {te_path}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1673.csv"
    df = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1681.astype(np.float64),
    })
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  rows={len(df)}  cols={list(df.columns)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "parent_method": "nb1673",
        "parent_variant": "bob_mean_10seed",
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "anchor_kind": "PRE_unblind_te_513",
        "rae_anchor": rae_anchor,
        "outer_seeds": OUTER_SEEDS,
        "n_outer_seeds": int(n_outer),
        "inner_offsets": INNER_OFFSETS,
        "inner_seed_formula": "o * 1000 + offset",
        "w_nb1561": W_NB1561,
        "w_nb1612": W_NB1612,
        "K_AP_best": K_AP_best,
        "K_Mord_best": K_Mord_best,
        "K_Embed_best": K_Embed_best,
        "K_Avalon_used": K_Avalon_used,
        "K_MACCS_fixed": n_top_maccs,
        "nb1561_feat_dim": int(feat_dim_1561),
        "nb1561_feat_breakdown": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim_1561),
        },
        "nb1612_top_k_config": TOP_K_NB1612,
        "nb1612_families_order": FAMILIES_NB1612,
        "nb1673_per_outer_lock_source": str(NB1673_SUMMARY),
        "per_outer_records": per_outer_records,
        "per_outer_blend_rae_in_unb": per_outer_rae,
        "per_outer_mean_in_unb": float(arr.mean()),
        "per_outer_std_in_unb": float(arr.std()),
        "per_outer_min_in_unb": float(arr.min()),
        "per_outer_max_in_unb": float(arr.max()),
        "per_outer_median_in_unb": float(np.median(arr)),
        "te_nb1681_stats": {
            "mean": float(te_nb1681.mean()),
            "std": float(te_nb1681.std()),
            "min": float(te_nb1681.min()),
            "max": float(te_nb1681.max()),
        },
        "in_rae_unb_mean": in_rae_mean,
        "delta_mean_vs_anchor": in_rae_mean - rae_anchor,
        "rae_bob_mean_nb1561_only_diag": rae_bob_mean_nb1561_only,
        "rae_bob_mean_nb1612_only_diag": rae_bob_mean_nb1612_only,
        "honest_lb_anchor_mean_nb1673": HONEST_LB_ANCHOR_MEAN,
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
        "n_outer_seeds", "rae_anchor",
        "outer_seeds", "inner_offsets",
        "w_nb1561", "w_nb1612",
        "nb1561_feat_dim",
        "per_outer_blend_rae_in_unb",
        "per_outer_mean_in_unb", "per_outer_std_in_unb",
        "te_nb1681_stats",
        "in_rae_unb_mean",
        "delta_mean_vs_anchor",
        "rae_bob_mean_nb1561_only_diag",
        "rae_bob_mean_nb1612_only_diag",
        "honest_lb_anchor_mean_nb1673",
        "te_npy_path", "csv_path",
    ):
        v = res.get(k)
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], float):
            print(f"  {k}: [{', '.join(f'{x:.4f}' for x in v)}]")
        else:
            print(f"  {k}: {v}")
