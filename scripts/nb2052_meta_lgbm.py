"""nb2052 -- Meta-LGBM on nb1162 OOF + top-20 SHAP features.

HYPOTHESIS:
    nb1162 (5-anchor SLSQP stacking pyramid + rank-stretch) is the current
    strongest in-sample blend with pooled scaffold-CV RAE 0.4206 on the 253
    unblind. Can a small meta-LGBM that takes nb1162's *honest* OOF prediction
    plus the top-20 SHAP features from nb2063 squeeze additional signal out
    via non-linear interactions, OR is nb1162 already at the calibration
    ceiling for this anchor set?

PROTOCOL:
    1. "TRAIN" set = 253 unblind (the only labeled scope where nb1162's
       honest OOF lives; nb1162 has no 4139-train OOF since its anchors
       only span 253). Per the task contract, we use nb1162's OOF (not
       in-sample fit) to avoid optimism.
    2. Reconstruct nb1162's honest scaffold-5-fold cross-fit OOF by
       replaying its SLSQP-then-rank-stretch protocol on the 5 anchor
       OOF arrays. The output matches the saved nb1162 pooled
       scaffold-CV RAE.
    3. Build the same 117-col 5-way K-tuned feature matrix as nb2063
       (AtomPair + MACCS + Mordred + ChempropEmbed + Avalon + ChEMBL kNN),
       then slice to the *top-20* SHAP indices from
       nb2063_top50_idx.npy[:20].
    4. Meta features X = [nb1162_oof, 20 SHAP cols] (21 columns total).
    5. Meta-LGBM: scaffold-5-fold cross-fit on 253 -> pooled OOF RAE.
       Hyperparams mirror nb1852/nb2063 (depth=4, leaves=15, n_est=300,
       lr=0.03, min_child=5, reg_lambda=2.0).
    6. Deploy: 513-row test = [te_nb1162, top-20 cols of 117-col on full
       513]. Re-fit meta-LGBM on all 253 with same hyperparams; predict 513.
    7. Gate (must strictly improve): pooled scaffold-CV RAE <=
       nb1162_oof_RAE - 0.01. Only build deploy CSV if gate PASSES.

Outputs:
    scripts/nb2052_meta_lgbm.py
    data/processed/nb2052_summary.json
    data/processed/nb2052_meta_oof.npy        (253,) float32
    data/processed/nb2052_nb1162_oof.npy      (253,) float32
    data/processed/te_nb2052.npy              (513,) float32  (always saved)
    submissions/nb2052_meta_lgbm.csv          (only if gate passes)
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
from scipy.optimize import minimize
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2052"
N_FOLDS = 5
SEED = 42

# Meta-LGBM hyperparams (mirror nb1852/nb2063 conventions)
META_LGBM_KW = dict(
    objective="regression",
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    min_child_samples=5,
    reg_lambda=2.0,
    random_state=0,
    n_jobs=2,
    verbosity=-1,
)

# Gate margin: must strictly improve nb1162 OOF RAE by 0.01 or more.
GATE_MARGIN = 0.01

# -- nb1162 protocol mirror --
NB1162_ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("nb730_honest", "nb730_honest_pred_oof.npy",         "te_nb730_honest.npy"),
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

# -- nb2063 117-col build inputs --
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
NB2063_TOP50_IDX = DATA_PROCESSED / "nb2063_top50_idx.npy"
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"

TOP_K_FEATS = 20  # take top-20 SHAP from nb2063's top-50

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# nb1162 OOF reconstruction (honest scaffold-CV)
# ---------------------------------------------------------------------------

def _slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def _best_stretch(blend_tr: np.ndarray, y_tr: np.ndarray,
                  mu: float, grid) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def build_nb1162_oof_and_te(
    y_unb: np.ndarray, unb_scaffolds, n_te: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (oof_253, te_513, pooled_oof_RAE) replaying nb1162 protocol."""
    oof_cols, te_cols = [], []
    for disp, oof_rel, te_rel in NB1162_ANCHORS:
        oof_p = DATA_PROCESSED / oof_rel
        te_p = DATA_PROCESSED / te_rel
        assert oof_p.exists(), f"missing OOF: {oof_p}"
        assert te_p.exists(),  f"missing te:  {te_p}"
        oof_cols.append(np.load(oof_p).astype(np.float64))
        te_cols.append(np.load(te_p).astype(np.float64))
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    n_unb = len(y_unb)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=SEED,
    )
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_s = []
    for k, (tr_loc, va_loc) in enumerate(splits):
        w_f = _slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = _best_stretch(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_s.append(s_f)
    pooled = float(rae(y_unb, oof_blend))

    # Deploy te: refit on all 253, apply mean(fold_s)
    w_dep = _slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_dep
    mu_dep = float(blend_unb.mean())
    s_dep = float(np.mean(fold_s))
    blend_te = P_te @ w_dep
    te_nb1162 = (mu_dep + s_dep * (blend_te - mu_dep)).astype(np.float64)
    return oof_blend, te_nb1162, pooled


# ---------------------------------------------------------------------------
# nb2063 117-col matrix builder (mirror)
# ---------------------------------------------------------------------------

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
    return agg.rename(columns={"src_first": "src"})


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q, n_pool = a.shape[0], b.shape[0]
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
    return pred, top_sim.mean(axis=1).astype(np.float32)


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    p = MORDRED_DIR / "X_mordred_test.npy"
    if not p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {p}")
    X = np.load(p).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape {X.shape} vs n_test={n_test_expected}")
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
    return np.where(np.isfinite(X), X, 0.0).astype(np.float32)


def _extract_family(sum_dict: dict, fam_key: str) -> np.ndarray:
    for f in sum_dict["families"]:
        if f["family"] == fam_key:
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError(f"{fam_key} not found")


def _extract_best_K(sum_dict: dict, records_key: str) -> dict:
    best_K = int(sum_dict["best_K"])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def build_117col_full(test_smiles: list[str], n_te: int) -> np.ndarray:
    """Build the same 117-col matrix as nb2063 on ALL 513 test rows."""
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
    rec_mord = _extract_best_K(sum_1523, "per_K_records")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_family(sum_1484, "AtomPair")
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    X_ap = _load_npy_test(ATOMPAIR_TE_PATH, n_te)[:, top_ap_bit_idx]
    X_maccs = _load_npy_test(MACCS_TE_PATH, n_te)[:, top_maccs_bit_idx]
    X_mord = _load_mordred_test(n_te)[:, top_mord_col_idx]
    X_emb = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_te)[:, top_embed_col_idx]
    X_av = _load_npy_test(AVALON_TE_PATH, n_te)[:, top_avalon_bit_idx]

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = {ik for m in test_mols
                      if (ik := _safe_inchikey(m)) is not None}
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep = fp_pool.sum(axis=1) > 0
    if not keep.all():
        pool = pool[keep].reset_index(drop=True)
        fp_pool = fp_pool[keep]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))

    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else ""
                       for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_full = np.concatenate(
        [X_ap, X_maccs, X_mord, X_emb, X_av,
         pred_chembl.astype(np.float32).reshape(-1, 1),
         mean_sim.astype(np.float32).reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    return X_full


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Meta-LGBM on nb1162 OOF + top-{TOP_K_FEATS} SHAP features")
    print(f"          gate: scaffold-CV RAE <= nb1162_OOF - {GATE_MARGIN}")
    print("=" * 78)

    # ---- Load 513 test + unblind labels + scaffolds ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Step 1: Reconstruct nb1162 honest scaffold-CV OOF (253) ----
    print("\n" + "-" * 78)
    print("STEP 1: replay nb1162 SLSQP+stretch scaffold-CV -> OOF on 253")
    print("-" * 78)
    nb1162_oof, te_nb1162, nb1162_oof_rae = build_nb1162_oof_and_te(
        y_unb, unb_scaffolds, n_te,
    )
    print(f"   nb1162 reconstructed OOF RAE = {nb1162_oof_rae:.4f}")

    # cross-check against cached te_nb1162.npy
    cached_te_path = DATA_PROCESSED / "te_nb1162.npy"
    if cached_te_path.exists():
        cached_te = np.load(cached_te_path).astype(np.float64)
        if cached_te.shape == te_nb1162.shape:
            diff = float(np.max(np.abs(cached_te - te_nb1162)))
            print(f"   te_nb1162 cached vs replay max|diff| = {diff:.6f}")

    # ---- Step 2: Build top-20 SHAP feature columns (from nb2063) ----
    print("\n" + "-" * 78)
    print(f"STEP 2: build 117-col matrix -> slice to top-{TOP_K_FEATS} SHAP")
    print("-" * 78)
    if not NB2063_TOP50_IDX.exists():
        raise FileNotFoundError(f"missing {NB2063_TOP50_IDX}")
    top50_idx = np.load(NB2063_TOP50_IDX).astype(int)
    top20_idx = top50_idx[:TOP_K_FEATS]
    print(f"   nb2063 top-{TOP_K_FEATS} feat indices in 117: "
          f"{top20_idx.tolist()}")

    X_full_513 = build_117col_full(te_smiles, n_te)
    if X_full_513.shape[1] < 117:
        print(f"   [warn] 117-col build returned {X_full_513.shape[1]} cols")
    X_top20_513 = X_full_513[:, top20_idx].astype(np.float32)
    X_top20_unb = X_top20_513[unb_idx].astype(np.float32)
    print(f"   X_top20_unb={X_top20_unb.shape}  X_top20_513={X_top20_513.shape}")

    # ---- Step 3: Meta-LGBM scaffold-5-fold CV on 253 ----
    print("\n" + "-" * 78)
    print(f"STEP 3: meta-LGBM scaffold {N_FOLDS}-fold cross-fit on 253")
    print("-" * 78)
    X_meta_unb = np.concatenate(
        [nb1162_oof.reshape(-1, 1).astype(np.float32), X_top20_unb],
        axis=1,
    ).astype(np.float32)
    print(f"   X_meta_unb = {X_meta_unb.shape}  "
          f"(col 0 = nb1162_oof, cols 1..{TOP_K_FEATS} = top-{TOP_K_FEATS} SHAP)")

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=SEED,
    )
    meta_oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rows = []
    for k, (tr_loc, va_loc) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**META_LGBM_KW)
        mdl.fit(X_meta_unb[tr_loc], y_unb[tr_loc])
        meta_oof[va_loc] = mdl.predict(X_meta_unb[va_loc])
        rae_va = float(rae(y_unb[va_loc], meta_oof[va_loc]))
        fold_rows.append({
            "fold": k,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "val_rae": rae_va,
        })
        print(f"   fold {k}: n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  "
              f"val_rae={rae_va:.4f}")

    pooled_meta_rae = float(rae(y_unb, meta_oof))
    delta_vs_nb1162 = pooled_meta_rae - nb1162_oof_rae
    print(f"\n   pooled meta-LGBM scaffold-CV RAE = {pooled_meta_rae:.4f}")
    print(f"   delta vs nb1162 OOF              = {delta_vs_nb1162:+.4f}")

    # ---- Step 4: Gate ----
    gate_target = nb1162_oof_rae - GATE_MARGIN
    gate_pass = pooled_meta_rae <= gate_target
    print(f"\n   gate target  = {gate_target:.4f}  (= nb1162_OOF - {GATE_MARGIN})")
    print(f"   gate pass    = {gate_pass}")

    # ---- Step 5: Deploy refit on all 253; predict 513 ----
    print("\n" + "-" * 78)
    print("STEP 5: deploy refit on all 253 -> predict 513")
    print("-" * 78)
    X_meta_513 = np.concatenate(
        [te_nb1162.reshape(-1, 1).astype(np.float32), X_top20_513],
        axis=1,
    ).astype(np.float32)
    mdl_dep = lgb.LGBMRegressor(**META_LGBM_KW)
    mdl_dep.fit(X_meta_unb, y_unb)
    te_meta = mdl_dep.predict(X_meta_513).astype(np.float32)
    in_sample_rae = float(rae(y_unb, mdl_dep.predict(X_meta_unb)))
    print(f"   deploy in-sample RAE (overfit bound) = {in_sample_rae:.4f}")
    print(f"   te(513) mean/std = {te_meta.mean():.3f} / {te_meta.std():.3f}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_meta_oof.npy",
            meta_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_nb1162_oof.npy",
            nb1162_oof.astype(np.float32))
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, te_meta)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_meta_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_nb1162_oof.npy'}")
    print(f"[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_meta_lgbm.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_meta,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(f"[skip] gate FAILED -> no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "meta_lgbm_on_nb1162_oof_plus_top20_shap",
        "anchor": "nb1162",
        "anchor_oof_source": "honest scaffold-CV replay of nb1162 SLSQP+stretch",
        "top_K_feats": TOP_K_FEATS,
        "feat_provenance": "nb2063_top50_idx.npy[:20] (top-20 SHAP of 117-col)",
        "top20_idx_in_117": top20_idx.tolist(),
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "meta_lgbm_hyperparams": {
            k: v for k, v in META_LGBM_KW.items()
            if k != "random_state" and k != "n_jobs" and k != "verbosity"
        },
        "nb1162_oof_rae": nb1162_oof_rae,
        "meta_scaffold_cv_rae": pooled_meta_rae,
        "delta_vs_nb1162_oof": delta_vs_nb1162,
        "fold_results": fold_rows,
        "gate_margin": GATE_MARGIN,
        "gate_target": gate_target,
        "gate_pass": bool(gate_pass),
        "in_sample_rae_overfit_bound": in_sample_rae,
        "deploy_te_mean": float(te_meta.mean()),
        "deploy_te_std": float(te_meta.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_pass else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "nb1162_oof_rae",
        "meta_scaffold_cv_rae",
        "delta_vs_nb1162_oof",
        "gate_target",
        "gate_pass",
        "in_sample_rae_overfit_bound",
        "deploy_te_mean", "deploy_te_std",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
