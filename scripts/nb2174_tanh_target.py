"""nb2174 -- Tanh-transformed residual target on SHAP top-K=28 (117-col).

HYPOTHESIS:
    nb2103 fits LGBM(MSE) directly on the raw chemprop_aux residual
    r = y - anchor (mean ~ -0.13, std ~ 0.70).  Under squared error, the
    LGBM split criterion treats outlier rows symmetrically -- one |r|=2
    residual contributes 4x the gradient of a |r|=1 row, dragging splits
    toward heavy-tail fitting.  If we instead predict z = tanh(r/c) for
    some scale c (so the heavy tails are *compressed* into a finite
    [-1, +1] box) and then untransform via r_hat = c * arctanh(pred), the
    learner sees a bounded, robustified target and the inverse map gives
    back a residual on the real line.  This is the classical "tanh
    squashing" trick used in robust regression, transposed to a
    boosted-trees residual-stack setting.

    The transform is monotonic so rank order is preserved; the scale c
    controls how much heavy-tail compression we apply -- small c
    (e.g. 0.5) aggressively saturates rows with |r| > 1 to +/- 1,
    treating them like binary direction labels; large c (e.g. 2.0)
    barely compresses (tanh is near-linear for |r/c| << 1), so the
    learner behaves close to raw MSE.  We sweep c in
    {0.5, 1.0, 1.5, 2.0} and also test a sign-preserving log transform
    z = sign(r) * log(1 + |r|) (which is even softer than tanh -- it
    only grows logarithmically, not asymptotically) as a fifth variant.

PROTOCOL:
    1. Reuse SHAP top-28 features from nb2063 cached importance on the
       117-col 5-way K-tuned matrix (identical to nb2103 K=28 build).
    2. Load chemprop_aux te[unb_idx] anchor, form r = y_unb - anchor.
    3. For each c in {0.5, 1.0, 1.5, 2.0}: transform target as
       r_tanh = tanh(r / c).  Run 5-seed bag x 5-fold cross-fit LGBM(MSE)
       with K=28 features and the nb2103 hyperparams.  Untransform
       per-fold predictions as
           r_hat = c * arctanh(clip(pred_tanh, -0.99, +0.99))
       and final = anchor + r_hat.  Compute mean-bag and median-bag RAE.
    4. Also fit the sign-log variant
           r_log = sign(r) * log(1 + |r|)
           r_hat = sign(pred_log) * (exp(|pred_log|) - 1)
       (a softer alternative to tanh that doesn't saturate).
    5. Verdict vs nb2103 K=28 (0.4737 mean-bag, 0.4698 median-bag) at
       decision_margin = 0.003.

Outputs:
    scripts/nb2174_tanh_target.py
    data/processed/nb2174_summary.json
    data/processed/nb2174_mean_bag_oof_c{c}.npy   (253,) float32 per c
    data/processed/nb2174_median_bag_oof_c{c}.npy (253,) float32 per c
    data/processed/nb2174_mean_bag_oof_signlog.npy (253,) float32
    data/processed/nb2174_median_bag_oof_signlog.npy (253,) float32
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
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2174"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K_FIXED = 28
C_GRID = [0.5, 1.0, 1.5, 2.0]
ARCTANH_CLIP = 0.99

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
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003


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
    """Identical union as nb2063/nb2081/nb2091/nb2103/nb2166."""
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
        print(f"   [src] CHEMBL3401_raw kept: {len(d)} rows")

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
        print(f"   [src] chembl_nr_extended PXR kept: {len(d)} rows")

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
        print(f"   [src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize union: {len(pool)} rows")
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    print(f"   [pool] after InChIKey dedup (median agg): {len(agg)} unique cpds")
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
    """LGBM(MSE) -- identical to nb2103 K=28 recipe."""
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


def _residual_cross_fit_transformed(
    X: np.ndarray, residual: np.ndarray, seed: int,
    forward, inverse,
) -> np.ndarray:
    """5-fold cross-fit; fit LGBM(MSE) on forward(residual), then invert.

    `forward(r)` returns the transformed target in the model's prediction
    space; `inverse(z)` maps the OOF prediction back to a residual on
    the real line.  Untransform is applied per-fold to the held-out
    prediction vector.
    """
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof_r = np.full(n, np.nan, dtype=np.float64)
    z = forward(residual)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], z[tr_loc])
        z_hat = mdl.predict(X[va_loc])
        oof_r[va_loc] = inverse(z_hat)
    return oof_r


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


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


def _make_tanh_forward_inverse(c: float):
    def forward(r):
        return np.tanh(r / c)

    def inverse(z):
        z_clip = np.clip(z, -ARCTANH_CLIP, ARCTANH_CLIP)
        return c * np.arctanh(z_clip)

    return forward, inverse


def _signlog_forward(r):
    return np.sign(r) * np.log1p(np.abs(r))


def _signlog_inverse(z):
    return np.sign(z) * (np.expm1(np.abs(z)))


def _run_variant(
    label: str, X_topK: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
    y_unb: np.ndarray, n_unb: int, rae_anchor: float,
    nb2103_k28_mean_bag: float, nb2103_k28_median_bag: float,
    forward, inverse, save_tag: str,
) -> dict:
    print(f"\n--- {label} ---")
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_transformed(
            X_topK, residual, s, forward, inverse
        )
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   {label}  seed={s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    delta_mean = rae_mean_bag - nb2103_k28_mean_bag
    delta_median = rae_median_bag - nb2103_k28_median_bag
    beats_mean = rae_mean_bag < nb2103_k28_mean_bag - DECISION_MARGIN
    flat_mean = abs(delta_mean) < DECISION_MARGIN
    beats_median = rae_median_bag < nb2103_k28_median_bag - DECISION_MARGIN
    flat_median = abs(delta_median) < DECISION_MARGIN

    if beats_mean and beats_median:
        verdict_v = "BEATS_NB2103_BOTH"
    elif beats_mean:
        verdict_v = "BEATS_NB2103_MEAN_ONLY"
    elif beats_median:
        verdict_v = "BEATS_NB2103_MEDIAN_ONLY"
    elif flat_mean and flat_median:
        verdict_v = "FLAT_VS_NB2103_BOTH"
    elif flat_mean:
        verdict_v = "FLAT_MEAN_LOSS_MEDIAN"
    elif flat_median:
        verdict_v = "FLAT_MEDIAN_LOSS_MEAN"
    else:
        verdict_v = "HURTS_NB2103"

    print(f"   {label}  per-seed RAE   = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   {label}  per-seed mean  = {rae_per_seed_mean:.4f}  "
          f"std = {rae_per_seed_std:.4f}")
    print(f"   {label}  pooled mean    = {rae_mean_bag:.4f}  "
          f"(d_vs_nb2103 K=28 = {delta_mean:+.4f})")
    print(f"   {label}  pooled median  = {rae_median_bag:.4f}  "
          f"(d_vs_nb2103 K=28 = {delta_median:+.4f})")
    print(f"   {label}  verdict        = {verdict_v}")

    out_mean = DATA_PROCESSED / f"{TAG}_mean_bag_oof_{save_tag}.npy"
    np.save(out_mean, mean_bag_oof.astype(np.float32))
    print(f"   [save] {out_mean}")
    out_median = DATA_PROCESSED / f"{TAG}_median_bag_oof_{save_tag}.npy"
    np.save(out_median, median_bag_oof.astype(np.float32))
    print(f"   [save] {out_median}")

    return {
        "label": label,
        "save_tag": save_tag,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": delta_mean,
        "delta_median_bag_vs_nb2103_K28": delta_median,
        "beats_nb2103_mean": bool(beats_mean),
        "beats_nb2103_median": bool(beats_median),
        "flat_vs_nb2103_mean": bool(flat_mean),
        "flat_vs_nb2103_median": bool(flat_median),
        "verdict": verdict_v,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TANH-TRANSFORMED RESIDUAL TARGET at SHAP K={K_FIXED}")
    print(f"          c-grid={C_GRID}  + sign-log variant")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean-bag {NB2103_K28_MEAN_BAG_REF:.4f} / "
          f"median-bag {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    if not NB2063_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2063_SUMMARY}")
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    with open(NB2063_SUMMARY) as f:
        nb2063_sum = json.load(f)

    nb2103_k28_mean_bag = NB2103_K28_MEAN_BAG_REF
    nb2103_k28_median_bag = NB2103_K28_MEDIAN_BAG_REF
    if NB2103_SUMMARY.exists():
        with open(NB2103_SUMMARY) as f:
            nb2103_sum = json.load(f)
        for r in nb2103_sum.get("per_K_records", []):
            if int(r.get("K", -1)) == K_FIXED:
                nb2103_k28_mean_bag = float(r["rae_mean_bag"])
                nb2103_k28_median_bag = float(r["rae_median_bag"])
                break
    print(f"[ref] nb2103.K={K_FIXED} mean_bag_rae   = {nb2103_k28_mean_bag:.4f}")
    print(f"[ref] nb2103.K={K_FIXED} median_bag_rae = {nb2103_k28_median_bag:.4f}")

    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    topK_idx = full_rank_order[:K_FIXED].astype(np.int32)
    print(f"[topK] K={K_FIXED} top idx (first 8): {topK_idx[:8].tolist()} ...")

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
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
    r_abs = np.abs(residual)
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"|r|_p95={np.percentile(r_abs, 95):.3f}  "
          f"|r|_max={r_abs.max():.3f}")

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
    assert K_Mord_best == int(sum_1523["best_K"])
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
    print(f"[reuse] top-{n_top_ap}  AtomPair / top-{n_top_maccs}  MACCS / "
          f"top-{n_top_mord}  Mordred / top-{n_top_embed}  Embed / "
          f"top-{n_top_avalon}  Avalon")

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)

    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after})")
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")
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
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    if feat_dim != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim {feat_dim} != SHAP importance length "
            f"{shap_imp_full117.shape[0]}"
        )

    feat_names: list[str] = []
    feat_family: list[str] = []
    for b in top_ap_bit_idx:
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for b in top_maccs_bit_idx:
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for c_idx in top_mord_col_idx:
        feat_names.append(f"Mordred_col_{int(c_idx)}")
        feat_family.append("Mordred")
    for d in top_embed_col_idx:
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for b in top_avalon_bit_idx:
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == feat_dim

    topK_family = [feat_family[i] for i in topK_idx]
    fam_counts: dict[str, int] = {}
    for fam in topK_family:
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    print(f"[topK] K={K_FIXED} family breakdown: {fam_counts}")
    X_topK = X_unb[:, topK_idx].astype(np.float32)
    print(f"[topK] X_topK shape = {X_topK.shape}")

    # ---- Sanity check transforms on the actual residual vector ----
    print("\n" + "-" * 78)
    print("TRANSFORM SANITY (on the residual itself; round-trip error)")
    print("-" * 78)
    transform_sanity = []
    for c in C_GRID:
        z = np.tanh(residual / c)
        rr = c * np.arctanh(np.clip(z, -ARCTANH_CLIP, ARCTANH_CLIP))
        rt_err = float(np.mean((rr - residual) ** 2) ** 0.5)
        sat_frac = float(np.mean(np.abs(z) >= ARCTANH_CLIP))
        print(f"   c={c:>4.1f}:  z range [{z.min():+.3f}, {z.max():+.3f}]  "
              f"z std = {z.std():.3f}  "
              f"|z|>=clip frac = {sat_frac:.3f}  "
              f"round-trip RMSE = {rt_err:.4f}")
        transform_sanity.append({
            "kind": f"tanh_c{c}",
            "z_min": float(z.min()),
            "z_max": float(z.max()),
            "z_std": float(z.std()),
            "saturated_frac": sat_frac,
            "round_trip_rmse": rt_err,
        })
    z_log = _signlog_forward(residual)
    rr_log = _signlog_inverse(z_log)
    rt_err_log = float(np.mean((rr_log - residual) ** 2) ** 0.5)
    print(f"   signlog: z range [{z_log.min():+.3f}, {z_log.max():+.3f}]  "
          f"z std = {z_log.std():.3f}  round-trip RMSE = {rt_err_log:.4f}")
    transform_sanity.append({
        "kind": "signlog",
        "z_min": float(z_log.min()),
        "z_max": float(z_log.max()),
        "z_std": float(z_log.std()),
        "saturated_frac": 0.0,
        "round_trip_rmse": rt_err_log,
    })

    # ---- c-grid sweep ----
    print("\n" + "-" * 78)
    print(f"TANH c-GRID SWEEP {C_GRID}")
    print("-" * 78)
    variant_results: list[dict] = []
    for c in C_GRID:
        c_tag = f"{c:0.1f}".replace(".", "_")
        forward, inverse = _make_tanh_forward_inverse(c)
        rec = _run_variant(
            label=f"tanh(c={c:.1f})",
            X_topK=X_topK, residual=residual, anchor=anchor, y_unb=y_unb,
            n_unb=n_unb, rae_anchor=rae_anchor,
            nb2103_k28_mean_bag=nb2103_k28_mean_bag,
            nb2103_k28_median_bag=nb2103_k28_median_bag,
            forward=forward, inverse=inverse, save_tag=f"c{c_tag}",
        )
        rec["c"] = float(c)
        rec["family"] = "tanh"
        variant_results.append(rec)

    # ---- sign-log variant ----
    print("\n" + "-" * 78)
    print("SIGN-LOG VARIANT (z = sign(r) * log(1+|r|))")
    print("-" * 78)
    rec_log = _run_variant(
        label="signlog",
        X_topK=X_topK, residual=residual, anchor=anchor, y_unb=y_unb,
        n_unb=n_unb, rae_anchor=rae_anchor,
        nb2103_k28_mean_bag=nb2103_k28_mean_bag,
        nb2103_k28_median_bag=nb2103_k28_median_bag,
        forward=_signlog_forward, inverse=_signlog_inverse,
        save_tag="signlog",
    )
    rec_log["c"] = None
    rec_log["family"] = "signlog"
    variant_results.append(rec_log)

    # ---- Summary table ----
    print("\n" + "=" * 78)
    print("TANH-TARGET VARIANT SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'variant':>14s}  {'mean_bag':>10s}  {'median_bag':>10s}  "
          f"{'d_mean_K28':>11s}  {'d_med_K28':>11s}  verdict")
    print(f"   {'nb2103_K28':>14s}  {nb2103_k28_mean_bag:>10.4f}  "
          f"{nb2103_k28_median_bag:>10.4f}  {0.0:>+11.4f}  {0.0:>+11.4f}  "
          f"BASELINE")
    for r in variant_results:
        print(f"   {r['label']:>14s}  {r['rae_mean_bag']:>10.4f}  "
              f"{r['rae_median_bag']:>10.4f}  "
              f"{r['delta_mean_bag_vs_nb2103_K28']:>+11.4f}  "
              f"{r['delta_median_bag_vs_nb2103_K28']:>+11.4f}  "
              f"{r['verdict']}")

    sweep_mean = [r["rae_mean_bag"] for r in variant_results]
    sweep_median = [r["rae_median_bag"] for r in variant_results]
    best_mean_i = int(np.argmin(sweep_mean))
    best_median_i = int(np.argmin(sweep_median))
    best_label_mean = variant_results[best_mean_i]["label"]
    best_label_median = variant_results[best_median_i]["label"]
    best_mean_bag = float(sweep_mean[best_mean_i])
    best_median_bag = float(sweep_median[best_median_i])

    all_label = ["nb2103_K28_MSE"] + [r["label"] for r in variant_results]
    all_mean = [nb2103_k28_mean_bag] + sweep_mean
    all_median = [nb2103_k28_median_bag] + sweep_median
    best_overall_mean_i = int(np.argmin(all_mean))
    best_overall_median_i = int(np.argmin(all_median))
    best_overall_mean_label = all_label[best_overall_mean_i]
    best_overall_median_label = all_label[best_overall_median_i]
    best_overall_mean_rae = float(all_mean[best_overall_mean_i])
    best_overall_median_rae = float(all_median[best_overall_median_i])

    print(f"\n   best variant (mean-bag)   = {best_label_mean}  "
          f"RAE {best_mean_bag:.4f}  "
          f"(d_vs_K28 = {best_mean_bag - nb2103_k28_mean_bag:+.4f})")
    print(f"   best variant (median-bag) = {best_label_median}  "
          f"RAE {best_median_bag:.4f}  "
          f"(d_vs_K28 = {best_median_bag - nb2103_k28_median_bag:+.4f})")
    print(f"   best OVERALL mean-bag   = {best_overall_mean_label}  "
          f"RAE {best_overall_mean_rae:.4f}")
    print(f"   best OVERALL median-bag = {best_overall_median_label}  "
          f"RAE {best_overall_median_rae:.4f}")

    if (best_overall_mean_label != "nb2103_K28_MSE"
            and best_overall_mean_rae < nb2103_k28_mean_bag - DECISION_MARGIN):
        global_verdict = (
            f"TANH_TARGET_BEATS_NB2103_K28_MEAN_AT_{best_overall_mean_label}"
        )
    elif (best_overall_median_label != "nb2103_K28_MSE"
            and best_overall_median_rae < nb2103_k28_median_bag - DECISION_MARGIN):
        global_verdict = (
            f"TANH_TARGET_BEATS_NB2103_K28_MEDIAN_ONLY_AT_"
            f"{best_overall_median_label}"
        )
    elif (abs(best_mean_bag - nb2103_k28_mean_bag) < DECISION_MARGIN
            and abs(best_median_bag - nb2103_k28_median_bag) < DECISION_MARGIN):
        global_verdict = "TANH_TARGET_FLAT_VS_NB2103_K28"
    else:
        global_verdict = (
            "TANH_TARGET_DOES_NOT_BEAT_NB2103_K28_MSE_REMAINS_OPTIMAL"
        )
    print(f"\n   global verdict   = {global_verdict}")

    summary = {
        "tag": TAG,
        "method": ("lgbm_mse_tanh_transformed_residual_c_sweep_plus_signlog_"
                   "at_K28_on_117col"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2063 cached SHAP importance + same 117-col "
                        "5-way K-tuned matrix as nb2103/nb2166"),
        "model_family": "LightGBM_MSE_on_transformed_residual",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "K_fixed": int(K_FIXED),
        "c_grid": C_GRID,
        "arctanh_clip": ARCTANH_CLIP,
        "include_signlog": True,
        "topK_idx_in_117": topK_idx.tolist(),
        "topK_family_counts": fam_counts,
        "feat_dim_full": int(feat_dim),
        "feat_breakdown_full": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_abs_p95": float(np.percentile(r_abs, 95)),
        "residual_abs_max": float(r_abs.max()),
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "transform_sanity": transform_sanity,
        "per_variant_records": variant_results,
        "best_variant_mean_bag_label": best_label_mean,
        "best_variant_mean_bag_rae": best_mean_bag,
        "best_variant_median_bag_label": best_label_median,
        "best_variant_median_bag_rae": best_median_bag,
        "best_overall_mean_label": best_overall_mean_label,
        "best_overall_mean_rae": best_overall_mean_rae,
        "best_overall_median_label": best_overall_median_label,
        "best_overall_median_rae": best_overall_median_rae,
        "delta_best_mean_vs_nb2103_K28": best_mean_bag - nb2103_k28_mean_bag,
        "delta_best_median_vs_nb2103_K28": (
            best_median_bag - nb2103_k28_median_bag
        ),
        "verdict": global_verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
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
        "K_fixed",
        "c_grid",
        "include_signlog",
        "feat_dim_full",
        "n_chembl_pool",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref",
        "nb2103_K28_median_bag_ref",
        "best_variant_mean_bag_label", "best_variant_mean_bag_rae",
        "best_variant_median_bag_label", "best_variant_median_bag_rae",
        "best_overall_mean_label", "best_overall_mean_rae",
        "best_overall_median_label", "best_overall_median_rae",
        "delta_best_mean_vs_nb2103_K28",
        "delta_best_median_vs_nb2103_K28",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-VARIANT TABLE ====")
    for r in res["per_variant_records"]:
        print(f"  {r['label']:>14s}  "
              f"mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"per_seed_mean={r['rae_per_seed_mean']:.4f}  "
              f"std={r['rae_per_seed_std']:.4f}  "
              f"d_mean_vs_K28={r['delta_mean_bag_vs_nb2103_K28']:+.4f}  "
              f"d_med_vs_K28={r['delta_median_bag_vs_nb2103_K28']:+.4f}  "
              f"{r['verdict']}")
