"""nb1623 -- Outer-bag VALIDATION of nb1612 (6-way ChemBERTa).

PROTOCOL
    1. For 5 outer seeds {0, 1, 7, 42, 137}: rebuild 6 residual learners with
       the chemprop_aux anchor (AtomPair + MACCS + Mordred + Chemprop embed +
       Avalon + ChemBERTa@K=50).  Inner-seed = [o*1000 + s for s in nb1612 seeds].
    2. Per outer: produce per-family mean-bag corrected OOFs, then
         (a) naive 1/6 mean blend,
         (b) 5-fold SLSQP cross-fit on the 6 corrected OOFs (slsqp seed = 42).
       Choose per-outer best variant (same picking rule as nb1612).
    3. Per-outer pooled RAE.
    4. Bag-of-bags (BoB) MEAN OOF = mean across 5 outer best-variant OOFs.
       Bag-of-bags MEDIAN OOF     = median across 5 outer best-variant OOFs.
    5. Verdict NB1612_REPRODUCES if mean(per_outer_rae) within 0.003 of 0.5218.

OUTPUTS
    data/processed/nb1623_summary.json
    data/processed/nb1623_bob_mean_oof.npy    (253,) float32
    data/processed/nb1623_bob_median_oof.npy  (253,) float32
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1623"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# nb1612 reference: rae_slsqp_crossfit = 0.5217555909800832 (best_variant slsqp_5fold)
NB1612_REF = 0.5218
DECISION_MARGIN = 0.003

# Outer bag
OUTER_SEEDS = [0, 1, 7, 42, 137]

# nb1612 inner spec
RESID_FOLDS = 5
RESID_INNER_BASE = [0, 1, 7, 42, 137]
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Same caches as nb1612
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMBERTA_TE_PATH = DATA_PROCESSED / "chemberta_test_emb.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# nb1612 K-tuning (held fixed; ChemBERTa best_K=50 from nb1612_summary)
TOP_K = {
    "AtomPair": 25,
    "MACCS": 20,
    "Mordred": 20,
    "ChempropEmbed": 20,
    "Avalon": 30,
    "ChemBERTa": 50,
}
FAMILIES = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed", "Avalon", "ChemBERTa"]


def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol):
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    frames = []
    p1 = EXT_DIR / "chembl_pxr_CHEMBL3401.parquet"
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
    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        d["src"] = "nr_extended"
        frames.append(d)
    p3 = EXT_DIR / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["smiles", "pec50"]]
        d["src"] = "pxr_all_types"
        frames.append(d)
    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")
    pool = pd.concat(frames, ignore_index=True)
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    return agg


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_pool = b.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if k >= n_pool:
            idx_part = np.argsort(-sim, axis=1)[:, :k]
        else:
            part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            idx_part = part[row_idx, order]
        row_idx = np.arange(e - s)[:, None]
        top_idx[s:e] = idx_part
        top_sim[s:e] = sim[row_idx, idx_part]
    return top_idx, top_sim


def _knn_predict(top_idx: np.ndarray, top_sim: np.ndarray,
                 pool_labels: np.ndarray, fallback: float):
    w = top_sim.copy()
    w = np.clip(w, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


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
    except Exception:
        imp = mdl.booster_.feature_importance(importance_type="gain")
        return imp.astype(np.float64), "lgbm_gain_fallback"


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_family_te(family: str, n_test: int) -> np.ndarray:
    if family == "AtomPair":
        return np.load(ATOMPAIR_TE_PATH).astype(np.float32)
    if family == "MACCS":
        return np.load(MACCS_TE_PATH).astype(np.float32)
    if family == "Mordred":
        return _load_mordred_test(n_test)
    if family == "ChempropEmbed":
        X = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
        return np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    if family == "Avalon":
        return np.load(AVALON_TE_PATH).astype(np.float32)
    if family == "ChemBERTa":
        X = np.load(CHEMBERTA_TE_PATH).astype(np.float32)
        return np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    raise ValueError(f"unknown family: {family}")


def _run_family(family: str, X_fam_unb: np.ndarray,
                pred_chembl_unb: np.ndarray, mean_sim_unb: np.ndarray,
                anchor: np.ndarray, residual: np.ndarray,
                top_k: int, inner_seeds: list, outer_seed: int) -> dict:
    n_fam = int(X_fam_unb.shape[1])
    X_full = np.concatenate(
        [X_fam_unb, pred_chembl_unb.reshape(-1, 1), mean_sim_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    # SHAP uses seed=outer_seed (so each outer gets its own feature pruning)
    imp_full, imp_src = _compute_shap_importance(X_full, residual, seed=outer_seed)
    fam_imp = imp_full[:n_fam]
    top_k_eff = min(top_k, n_fam)
    top_order = np.argsort(-fam_imp)
    top_idx = top_order[:top_k_eff].astype(int)

    X_fam_pruned = X_fam_unb[:, top_idx]
    X_pruned = np.concatenate(
        [X_fam_pruned, pred_chembl_unb.reshape(-1, 1), mean_sim_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)

    n_unb = anchor.shape[0]
    per_seed_corrected = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
    for i, s in enumerate(inner_seeds):
        resid_oof_s = _residual_cross_fit_one_seed(X_pruned, residual, s)
        per_seed_corrected[i] = anchor + resid_oof_s
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    return {
        "family": family,
        "n_fam_bits": n_fam,
        "top_k": int(top_k_eff),
        "shap_source": imp_src,
        "mean_bag_oof": mean_bag_oof.astype(np.float64),
    }


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w):
        r = y_tr - P_tr @ w
        return float(np.mean(r * r))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(_loss, w0, method="SLSQP", bounds=bnds,
                   constraints=cons, options={"ftol": 1e-10, "maxiter": 500})
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return w / s


def _slsqp_cross_fit(P: np.ndarray, y: np.ndarray, n_splits: int, seed: int):
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    folds = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        folds.append({"fold": int(f), "w": [float(x) for x in w.tolist()]})
    return oof, folds


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Outer-bag VALIDATION of nb1612 (6-way ChemBERTa)")
    print(f"          outer_seeds = {OUTER_SEEDS}")
    print(f"          inner_base  = {RESID_INNER_BASE} (offset by outer*1000)")
    print(f"          nb1612_ref  = {NB1612_REF:.4f}  margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Truth + indices ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Anchor ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor

    # ---- ChEMBL pool + kNN (built ONCE; not part of outer-bag perturbation) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN feature build (once)")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Preload all 6 family X_fam_unb tensors ----
    X_fam_unb_dict = {}
    for family in FAMILIES:
        X_fam_te = _load_family_te(family, n_test)
        X_fam_unb_dict[family] = X_fam_te[unb_idx].astype(np.float32)
        print(f"   {family:<14s} unb shape = {X_fam_unb_dict[family].shape}")

    # ---- Outer-bag loop ----
    per_outer_records = []
    per_outer_best_oofs = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_naive_oofs = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_slsqp_oofs = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae = []

    for oi, o in enumerate(OUTER_SEEDS):
        print("\n" + "=" * 78)
        print(f"OUTER SEED {o}  ({oi + 1}/{len(OUTER_SEEDS)})")
        print("=" * 78)
        inner_seeds = [o * 1000 + s for s in RESID_INNER_BASE]
        print(f"   inner_seeds = {inner_seeds}")
        fam_results = []
        for family in FAMILIES:
            r = _run_family(
                family=family,
                X_fam_unb=X_fam_unb_dict[family],
                pred_chembl_unb=pred_chembl_unb,
                mean_sim_unb=mean_sim_unb,
                anchor=anchor,
                residual=residual,
                top_k=TOP_K[family],
                inner_seeds=inner_seeds,
                outer_seed=o,
            )
            rae_fam = float(rae(y_unb, r["mean_bag_oof"]))
            r["rae_mean_bag"] = rae_fam
            fam_results.append(r)
            print(f"   {family:<14s} (K={r['top_k']:>3})  mean_bag RAE = {rae_fam:.4f}")
        # Naive 1/6
        P = np.stack([r["mean_bag_oof"] for r in fam_results], axis=0)  # (6, n_unb)
        naive_oof = P.mean(axis=0)
        rae_naive = float(rae(y_unb, naive_oof))
        # SLSQP cross-fit
        slsqp_oof, slsqp_folds = _slsqp_cross_fit(
            P.T.astype(np.float64), y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED
        )
        rae_slsqp = float(rae(y_unb, slsqp_oof))
        # Pick best (same rule as nb1612: min of the two)
        if rae_naive <= rae_slsqp:
            best_variant = "naive_1_6_mean"
            best_oof = naive_oof
            best_rae = rae_naive
        else:
            best_variant = "slsqp_5fold"
            best_oof = slsqp_oof
            best_rae = rae_slsqp
        print(f"   --> naive_RAE = {rae_naive:.4f}  slsqp_RAE = {rae_slsqp:.4f}  "
              f"best = {best_variant} ({best_rae:.4f})")
        W = np.array([f["w"] for f in slsqp_folds])
        w_mean = W.mean(axis=0).tolist()
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": [int(x) for x in inner_seeds],
            "per_family_rae": {
                r["family"]: r["rae_mean_bag"] for r in fam_results
            },
            "rae_naive_mean_blend": rae_naive,
            "rae_slsqp_crossfit": rae_slsqp,
            "best_variant": best_variant,
            "rae_best": best_rae,
            "slsqp_w_mean_over_folds": w_mean,
        })
        per_outer_best_oofs[oi] = best_oof
        per_outer_naive_oofs[oi] = naive_oof
        per_outer_slsqp_oofs[oi] = slsqp_oof
        per_outer_rae.append(best_rae)

    # ---- BoB MEAN + MEDIAN over per-outer best OOFs ----
    bob_mean_oof = per_outer_best_oofs.mean(axis=0)
    bob_median_oof = np.median(per_outer_best_oofs, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # Also compute pooled aggregates over the naive/slsqp tracks
    rae_bob_mean_naive = float(rae(y_unb, per_outer_naive_oofs.mean(axis=0)))
    rae_bob_median_naive = float(rae(y_unb, np.median(per_outer_naive_oofs, axis=0)))
    rae_bob_mean_slsqp = float(rae(y_unb, per_outer_slsqp_oofs.mean(axis=0)))
    rae_bob_median_slsqp = float(rae(y_unb, np.median(per_outer_slsqp_oofs, axis=0)))

    per_outer_mean = float(np.mean(per_outer_rae))
    per_outer_std = float(np.std(per_outer_rae))
    per_outer_min = float(np.min(per_outer_rae))
    per_outer_max = float(np.max(per_outer_rae))
    delta_per_outer_vs_ref = per_outer_mean - NB1612_REF
    delta_bob_mean_vs_ref = rae_bob_mean - NB1612_REF
    delta_bob_median_vs_ref = rae_bob_median - NB1612_REF

    reproduces = abs(delta_per_outer_vs_ref) < DECISION_MARGIN
    verdict = "NB1612_REPRODUCES" if reproduces else "NB1612_DOES_NOT_REPRODUCE"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   per-outer RAE list   = {[round(x, 4) for x in per_outer_rae]}")
    print(f"   per-outer MEAN       = {per_outer_mean:.4f}  (std {per_outer_std:.4f})")
    print(f"   per-outer MIN/MAX    = {per_outer_min:.4f} / {per_outer_max:.4f}")
    print(f"   BoB MEAN OOF RAE     = {rae_bob_mean:.4f}")
    print(f"   BoB MEDIAN OOF RAE   = {rae_bob_median:.4f}")
    print(f"   nb1612 reference     = {NB1612_REF:.4f}")
    print(f"   d(per-outer mean)    = {delta_per_outer_vs_ref:+.4f}  "
          f"(margin {DECISION_MARGIN})")
    print(f"   d(BoB mean)          = {delta_bob_mean_vs_ref:+.4f}")
    print(f"   d(BoB median)        = {delta_bob_median_vs_ref:+.4f}")
    print(f"   verdict              = {verdict}")
    print("=" * 78)

    # ---- Save ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "rae_anchor_chemprop_aux": rae_anchor,
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "outer_seeds": [int(o) for o in OUTER_SEEDS],
        "inner_seed_base": RESID_INNER_BASE,
        "inner_seed_recipe": "[o*1000 + s for s in inner_base]",
        "resid_folds": RESID_FOLDS,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "families_order": FAMILIES,
        "top_k_config_fixed": TOP_K,
        "chemberta_K_inherited": TOP_K["ChemBERTa"],
        "per_outer_records": per_outer_records,
        "per_outer_rae": [float(x) for x in per_outer_rae],
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "rae_bob_mean_naive_track": rae_bob_mean_naive,
        "rae_bob_median_naive_track": rae_bob_median_naive,
        "rae_bob_mean_slsqp_track": rae_bob_mean_slsqp,
        "rae_bob_median_slsqp_track": rae_bob_median_slsqp,
        "nb1612_ref": NB1612_REF,
        "decision_margin": DECISION_MARGIN,
        "delta_per_outer_mean_vs_ref": delta_per_outer_vs_ref,
        "delta_bob_mean_vs_ref": delta_bob_mean_vs_ref,
        "delta_bob_median_vs_ref": delta_bob_median_vs_ref,
        "reproduces": bool(reproduces),
        "verdict": verdict,
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
        "n_unb", "rae_anchor_chemprop_aux",
        "per_outer_rae",
        "per_outer_mean", "per_outer_std",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_ref",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
