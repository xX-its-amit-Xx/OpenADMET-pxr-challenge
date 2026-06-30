"""nb2370 -- K=21 pyramid verification test (CONDITIONAL on nb2263 K_opt=21).

CONTEXT:
    nb2263 (lucky-seed-aware RFE K=28 -> K=10) selected K_opt by minimising
    30-seed mean RAE at each greedy-drop step (kf_seeds 1116..1145). The
    nb2263 run picks a K_opt that may differ from nb2231's single-seed
    K=20 pick (whose 5-seed mean_bag 0.4763 inflated to 0.4844 on the
    fresh 30-seed pool -- lucky seed exposure).

    nb2370 is a STAND-ALONE INDEPENDENT REPLICATION on a FRESH seed band
    (kf_seeds 1146..1175, NOT used in nb2263 drop-selection nor in
    nb2263 pyramid-wrap test) of the K=21 pyramid vs the nb2240 K=20
    deep-30 reference 0.4601.

PROTOCOL:
    1. Wait for nb2263_summary.json. If missing -> defer to cycle 188.
    2. If K_opt = 20: SKIP (nb2240 already verified K=20 deep-30 0.4601).
    3. If K_opt != 20 (e.g. K=21): take K_opt surviving feature indices
       from nb2263_summary.json["pyramid_wrap_result"]["K_opt_cols"].
    4. Rebuild 117-col 5-way matrix (identical to nb2240 / nb2263:
       AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN),
       slice to K_opt indices on 513 test + 253 unblind.
    5. K_opt residual anchor: chemprop_aux te[unb_idx] anchor +
       LGBM(MSE) on residual, mean-bag over RESID_SEEDS {0,1,7,42,137},
       KFold(n=5, shuffle=True) cross-fit per seed.
    6. Build 5-anchor pyramid {nb2370_Kopt, chemprop_aux, nb1191, nb503,
       nb562}. Per-fold SLSQP convex blend (w>=0, sum=1) + rank-stretch
       (grid 1.000..1.150) under 5-fold scaffold-CV across 30 FRESH
       deep seeds {1146..1175}.
    7. Compare to nb2240 K=20 deep-30 reference 0.4601. Gate margin 0.003.
       If beats by 0.003: build deploy CSV submissions/nb2370_k{K_opt}_pyramid.csv.

Outputs:
    scripts/nb2370_k21_pyramid.py
    data/processed/nb2370_summary.json
    data/processed/nb2370_mean_bag_oof_Kopt.npy  (253,) float32  (on gate)
    data/processed/te_nb2370_Kopt.npy            (513,) float32  (on gate)
    data/processed/te_nb2370.npy                 (513,) float32  (on gate)
    submissions/nb2370_k{K_opt}_pyramid.csv      (on gate beat 0.003)

References:
    nb2263 lucky-aware RFE (provides K_opt + K_opt_cols)
    nb2240 K=20 deep-30 pyramid 0.4601 +/- 0.0017  -- reference
    nb2231 single-seed K=20 pick (lucky-seed-exposed)
    nb2250 K=18/22 pyramid wrap (K=20 stays best; deltas +0.003, +0.007)
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
from rdkit import Chem
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2370"

# ============================================================================
# Constants -- mirror nb2240 / nb2263
# ============================================================================
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2263_SUMMARY = DATA_PROCESSED / "nb2263_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# ============================================================================
# Pyramid params -- FRESH 30 deep seeds disjoint from nb2263's 1116..1145
# ============================================================================
N_FOLDS = 5
KF_SEEDS_DEEP_30 = list(range(1146, 1176))   # 30 FRESH seeds, disjoint
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# Reference for comparison: nb2240 K=20 deep-30 mean
NB2240_K20_DEEP30_REF = 0.4601
GATE_MARGIN = 0.003  # nb2370 must beat 0.4601 by 0.003 to flip ladder

CHEMPROP_AUX_REF = 0.6216

# nb1191 reconstruction (copied verbatim from nb2240 / nb2263)
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


# ============================================================================
# Helpers (copied / adapted from nb2240)
# ============================================================================
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


def _tanimoto_topk(fp_q, fp_pool, k):
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


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
    w = np.clip(top_sim, 0.0, 1.0)
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


def _lgbm_params(seed):
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


def _load_mordred_test(n_test_expected):
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


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


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


def reconstruct_nb1150_oof(n_unb):
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
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


# ============================================================================
# MAIN
# ============================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K=21 pyramid verification (conditional on nb2263 K_opt=21)")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # Step 1: Wait / check nb2263_summary.json
    # ------------------------------------------------------------------------
    if not NB2263_SUMMARY.exists():
        defer_summary = {
            "tag": TAG,
            "status": "DEFERRED_NB2263_NOT_FINISHED",
            "reason": "nb2263_summary.json missing -- waiting on lucky-aware RFE",
            "expected_followup_cycle": 188,
            "nb2263_summary_path": str(NB2263_SUMMARY),
            "wall_sec": round(time.time() - t0, 2),
        }
        out_path = DATA_PROCESSED / f"{TAG}_summary.json"
        with open(out_path, "w") as f:
            json.dump(defer_summary, f, indent=2)
        print(f"[defer] nb2263 not finished -- deferring to cycle 188")
        print(f"[save] {out_path}")
        return defer_summary

    with open(NB2263_SUMMARY) as f:
        nb2263 = json.load(f)
    K_opt = int(nb2263["K_opt_lucky_aware"])
    print(f"[load] nb2263 K_opt_lucky_aware = {K_opt}")
    print(f"[load] nb2263 rae_K_opt_lucky_aware = "
          f"{nb2263.get('rae_K_opt_lucky_aware', float('nan')):.4f}")

    # ------------------------------------------------------------------------
    # Step 2: If K_opt = 20, SKIP (nb2240 already verified)
    # ------------------------------------------------------------------------
    if K_opt == 20:
        skip_summary = {
            "tag": TAG,
            "status": "SKIPPED_K_OPT_IS_20",
            "reason": "nb2240 already verified K=20 deep-30 pyramid 0.4601",
            "K_opt_from_nb2263": 20,
            "nb2240_K20_deep30_ref": NB2240_K20_DEEP30_REF,
            "nb2263_rae_K_opt": float(nb2263.get("rae_K_opt_lucky_aware",
                                                 float("nan"))),
            "wall_sec": round(time.time() - t0, 2),
        }
        out_path = DATA_PROCESSED / f"{TAG}_summary.json"
        with open(out_path, "w") as f:
            json.dump(skip_summary, f, indent=2)
        print(f"[skip] K_opt=20 -- nb2240 already verified")
        print(f"[save] {out_path}")
        return skip_summary

    # ------------------------------------------------------------------------
    # Step 3: Extract K_opt surviving feature indices from nb2263
    # ------------------------------------------------------------------------
    pwr = nb2263.get("pyramid_wrap_result", None)
    if pwr is None or "K_opt_cols" not in pwr:
        # Walk the trajectory to recover surviving idx
        shap_top28 = list(nb2263["shap_top28_idx_in_117"])
        current = list(shap_top28)
        K_opt_cols = None
        for e in nb2263["rfe_trajectory"]:
            if e["feat_dropped"] is not None:
                current.remove(int(e["feat_dropped"]))
            if int(e["K_after"]) == K_opt:
                K_opt_cols = list(current)
                break
        if K_opt_cols is None:
            raise RuntimeError(
                f"Could not recover K_opt={K_opt} subset from nb2263 trajectory"
            )
    else:
        K_opt_cols = list(pwr["K_opt_cols"])
    assert len(K_opt_cols) == K_opt, (
        f"K_opt_cols length {len(K_opt_cols)} != K_opt {K_opt}"
    )
    print(f"[load] K_opt={K_opt} surviving cols from nb2263: {K_opt_cols}")

    # ------------------------------------------------------------------------
    # Step 4: Rebuild 117-col 5-way matrix
    # ------------------------------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values if "name" in te.columns else te["Molecule Name"].values
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor

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

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
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
    std_test_smiles = []
    for m in test_mols:
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"X_te_full shape {X_te_full.shape}"

    # ------------------------------------------------------------------------
    # Step 5: K_opt residual anchor (mean-bag over RESID_SEEDS)
    # ------------------------------------------------------------------------
    X_te_Kopt = X_te_full[:, K_opt_cols].astype(np.float32)
    X_unb_Kopt = X_te_Kopt[unb_idx]
    print(f"[feat] X_unb_Kopt={X_unb_Kopt.shape}  X_te_Kopt={X_te_Kopt.shape}")

    print("\n" + "-" * 78)
    print(f"K={K_opt} RESIDUAL LGBM  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae_corr = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_Kopt, residual, s)
        per_seed_corrected[i] = anchor + resid_oof
        per_seed_rae_corr.append(float(rae(y_unb, anchor + resid_oof)))
        te_resid_s = _train_full_then_predict_te(
            X_unb_Kopt, residual, X_te_Kopt, s
        )
        per_seed_te_resid[i] = te_resid_s
        print(f"   seed={s:3d}: rae_corr={per_seed_rae_corr[-1]:.4f}  "
              f"wall={time.time()-ts:.1f}s")
    mean_bag_oof_Kopt = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid_Kopt = per_seed_te_resid.mean(axis=0)
    te_Kopt_513 = te_anchor_513 + mean_bag_te_resid_Kopt
    rae_Kopt_mean_bag = float(rae(y_unb, mean_bag_oof_Kopt))
    rae_Kopt_per_seed_mean = float(np.mean(per_seed_rae_corr))
    print(f"\n[K{K_opt}] per-seed mean RAE = {rae_Kopt_per_seed_mean:.4f}")
    print(f"[K{K_opt}] mean-bag RAE      = {rae_Kopt_mean_bag:.4f}")

    oof_Kopt_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_Kopt.npy"
    te_Kopt_path = DATA_PROCESSED / f"te_{TAG}_Kopt.npy"
    np.save(oof_Kopt_path, mean_bag_oof_Kopt.astype(np.float32))
    np.save(te_Kopt_path, te_Kopt_513.astype(np.float32))
    print(f"[save] {oof_Kopt_path}")
    print(f"[save] {te_Kopt_path}")

    # ------------------------------------------------------------------------
    # Step 6: 5-anchor pyramid -- 30 deep seeds
    # ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"STAGE 2: 5-ANCHOR PYRAMID (nb2370_K{K_opt} replaces nb2240_K20)")
    print("=" * 78)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)

    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    te_chemprop_aux = te_anchor_513

    anchors_list = [
        (f"nb2370_K{K_opt}", mean_bag_oof_Kopt.astype(np.float64),
         te_Kopt_513.astype(np.float64)),
        ("chemprop_aux", chemprop_oof, te_chemprop_aux),
        ("nb1191", nb1191_oof, te_nb1191),
        ("nb503", nb503_oof, te_nb503),
        ("nb562", nb562_oof, te_nb562),
    ]
    indiv_rae = {}
    oof_cols, te_cols = [], []
    print("\n[anchors]")
    for disp, oof, te_arr in anchors_list:
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_test,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:18s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)

    # ---- Deep-30 scaffold CV ----
    print("\n" + "-" * 78)
    print(f"DEEP-30 SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS_DEEP_30[0]}.."
          f"{KF_SEEDS_DEEP_30[-1]}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed_results = []
    fold_s_all = []
    for kf_seed in KF_SEEDS_DEEP_30:
        pooled, oof_blend, fw, fs = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed_results.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "mean_s": float(np.mean(fs)),
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        fold_s_all.extend([float(x) for x in fs])
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fs):.3f}")
    arr_rae = np.asarray([r["pooled_rae"] for r in per_seed_results])
    deep30_mean = float(arr_rae.mean())
    deep30_std = float(arr_rae.std())
    deep30_min = float(arr_rae.min())
    deep30_max = float(arr_rae.max())
    print(f"\n[deep-30] mean = {deep30_mean:.4f} +/- {deep30_std:.4f}  "
          f"range=[{deep30_min:.4f}, {deep30_max:.4f}]")

    # ------------------------------------------------------------------------
    # Step 7: Gate vs nb2240 K=20 deep-30 reference 0.4601
    # ------------------------------------------------------------------------
    delta_vs_K20 = deep30_mean - NB2240_K20_DEEP30_REF
    gate_beat = delta_vs_K20 < -GATE_MARGIN
    gate_flat = abs(delta_vs_K20) <= GATE_MARGIN
    print("\n" + "-" * 78)
    print(f"GATE: vs nb2240 K=20 deep-30 ref {NB2240_K20_DEEP30_REF:.4f}  "
          f"margin {GATE_MARGIN}")
    print("-" * 78)
    print(f"   nb2370_K{K_opt} deep-30 = {deep30_mean:.4f}")
    print(f"   nb2240_K20  deep-30 = {NB2240_K20_DEEP30_REF:.4f}")
    print(f"   delta            = {delta_vs_K20:+.4f}")
    if gate_beat:
        verdict = f"BEATS_K20_BY_{abs(delta_vs_K20):.4f}"
    elif gate_flat:
        verdict = "FLAT_VS_K20"
    else:
        verdict = "WORSE_THAN_K20"
    print(f"   verdict          = {verdict}")

    # ---- Deploy refit + CSV (only on gate_beat) ----
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(fold_s_all))
    in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae_in_sample = float(rae(y_unb, deploy_te[unb_idx]))

    lb_band_est = LB_W_OOF * deep30_mean + LB_W_TE * te_unb_rae_in_sample
    print(f"\n[deploy] weights = "
          f"{ {disp: float(w) for (disp, _, _), w in zip(anchors_list, w_deploy)} }")
    print(f"[deploy] mu={mu_deploy:.4f}  s={s_deploy:.4f}  "
          f"in_RAE={in_rae_final:.4f}  te_unb_RAE={te_unb_rae_in_sample:.4f}")
    print(f"[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = {lb_band_est:.4f}")

    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    sub_csv_path = SUBMISSIONS / f"{TAG}_k{K_opt}_pyramid.csv"
    if gate_beat:
        np.save(te_npy_path, deploy_te)
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {te_npy_path}  (BEATS_K20)")
        print(f"[save] {sub_csv_path}")
    else:
        print(f"[skip] gate not beat -- no submission CSV written ({verdict})")

    summary = {
        "tag": TAG,
        "status": "RAN",
        "method": "k_opt_pyramid_verification_deep30_fresh_seeds",
        "K_opt_from_nb2263": K_opt,
        "K_opt_cols": [int(j) for j in K_opt_cols],
        "anchor": ANCHOR,
        "rae_anchor_chemprop_aux": rae_anchor,
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "rae_Kopt_per_seed_mean_standalone": rae_Kopt_per_seed_mean,
        "rae_Kopt_mean_bag_standalone": rae_Kopt_mean_bag,
        "delta_Kopt_standalone_vs_anchor": rae_Kopt_mean_bag - rae_anchor,
        "oof_Kopt_path": str(oof_Kopt_path),
        "te_Kopt_path": str(te_Kopt_path),
        "pyramid_anchors": [a[0] for a in anchors_list],
        "anchor_oof_rae_unb": indiv_rae,
        "kf_seeds_deep_30": KF_SEEDS_DEEP_30,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed_results,
        "deep30_mean": deep30_mean,
        "deep30_std": deep30_std,
        "deep30_min": deep30_min,
        "deep30_max": deep30_max,
        "nb2240_K20_deep30_ref": NB2240_K20_DEEP30_REF,
        "delta_vs_nb2240_K20": delta_vs_K20,
        "gate_margin": GATE_MARGIN,
        "gate_beat_K20": bool(gate_beat),
        "gate_flat_K20": bool(gate_flat),
        "verdict_vs_K20": verdict,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(anchors_list, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae_in_sample,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path) if gate_beat else None,
        "submission_csv": str(sub_csv_path) if gate_beat else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K_opt from nb2263      = {K_opt}")
    print(f"   K{K_opt} standalone RAE = {rae_Kopt_mean_bag:.4f}")
    print(f"   pyramid deep-30 RAE    = {deep30_mean:.4f} +/- {deep30_std:.4f}")
    print(f"   delta vs K=20 (0.4601) = {delta_vs_K20:+.4f}")
    print(f"   verdict                = {verdict}")
    print(f"   LB band                = {lb_band_est:.4f}")
    print(f"   wall                   = {summary['wall_sec']:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()
