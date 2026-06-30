"""nb1292 -- NR-family ChEMBL extension: VDR, CAR, AhR, FXR ligands as
supplementary external pool layered on top of the PXR ChEMBL pool used by
nb1242.

Hypothesis:
    PXR (NR1I2) sister nuclear receptors (VDR / NR1I1, CAR / NR1I3, FXR / NR1H4,
    AhR / not a NR but co-regulates the same xenobiotic-response programmes)
    share many drug-like ligand scaffolds with PXR.  Adding those compounds as
    additional kNN neighbours may expand effective scaffold coverage of the
    external bioactivity pool from ~945 (PXR-only ChEMBL) to 2000+ rows.  If
    novel-scaffold test compounds find better-matched NR-family neighbours,
    the kNN residual feature should carry more signal.  Compounds with both a
    PXR pEC50 and NR-family measurements keep the PXR pEC50; compounds
    measured only on sister receptors keep the median NR-family pEC50 as a
    proxy.

Protocol:
    1. MCP claude_ai_ChEMBL__get_bioactivity is the preferred source.  The
       local cached parquets data/external/chembl_nr_extended.parquet (11,496
       rows pre-cleaned, target_name in {PPARg, FXR, RXRa, LXRa, PXR, VDR,
       PPARa}) and data/external/chembl_ahr_activity.parquet (587 rows AhR)
       are mirrored from MCP queries done in earlier cycles -- use them as
       canonical.  CAR (NR1I3 / CHEMBL2972) was probed but no public ChEMBL
       parquet was archived locally; mark as UNAVAILABLE in summary.
    2. Build the NR-family pool: union {PXR, VDR, FXR, AhR} (the four sister-
       receptor families with biological relevance to xenobiotic response /
       NR1 family).  Also include PPARg, RXRa, LXRa, PPARa as bonus NR-family
       rows (they're already in the cache and broaden scaffold coverage).
    3. Standardize SMILES (src.pxr.chem.standardize), drop RDKit failures,
       compute InChIKey, dedup by InChIKey.  Aggregation rule when an InChIKey
       appears in multiple targets:
          - If a PXR measurement is present, KEEP the PXR pEC50 (median if
            multiple PXR rows).
          - Otherwise, keep the median pEC50 across all NR-family
            measurements.
       Add a `n_targets` column tracking how many distinct NR families
       reported on that compound, and a `has_pxr` flag.
    4. Drop any pool InChIKey present in the 513-test InChIKey set (leak
       guard) -- identical to nb1242.
    5. Tanimoto kNN k=5 over Morgan-2048 from the (513, n_pool) cross.
       Build two features per test row:
          - pred_nr_pec50   -- similarity-weighted mean of pool pEC50
          - mean_sim        -- top-k mean Tanimoto similarity
    6. Residual learner: anchor = nb1070_pred_oof on 253 unblind;
       residual = y_unb - nb1070_pred_oof;
       features = concat[MACCS-167(unb), pred_nr_pec50[unb_idx],
                         mean_sim[unb_idx]]  -> (253, 169).
       5-seed bag shallow LGBM Huber (depth=3, num_leaves=7, n_est=80),
       5-fold cross-fit per seed -- identical capacity to nb1242 so the only
       moving piece is the external pool composition.
    7. Verdict at 0.003 margin vs nb1242 (0.5431) and nb1251 (0.5394).

Outputs:
    scripts/nb1292_nr_family_chembl.py             (this file)
    data/processed/nb1292_summary.json
    data/processed/nb1292_mean_bag_oof.npy        (253,) float32
    data/processed/nb1292_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1292_median_bag_oof.npy      (253,) float32
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
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1292"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

# NR-family targets we accept from chembl_nr_extended cache.  The "core
# sister receptors" of PXR (NR1I2) are VDR (NR1I1), CAR (NR1I3 -- missing),
# FXR (NR1H4) -- xenobiotic / bile-acid axis.  AhR is xenobiotic-response
# co-regulator (not a NR but shares ligand chemotypes).  PPARg / RXRa /
# LXRa / PPARa are included as bonus NR-family pool -- they broaden
# scaffold coverage without polluting since the aggregation rule keeps PXR
# label whenever available.
NR_TARGETS_CORE = {"PXR", "VDR", "FXR"}              # NR1I2 + sisters in cache
NR_TARGETS_BONUS = {"PPARg", "RXRa", "LXRa", "PPARa"}  # bonus NR-family
NR_TARGETS_ALL = NR_TARGETS_CORE | NR_TARGETS_BONUS
CAR_AVAILABLE = False                                  # NR1I3 cache absent

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_REF = 0.5431      # PXR-only ChEMBL kNN residual bag
NB1251_REF = 0.5394      # PXR ChEMBL + BoB blend
DECISION_MARGIN = 0.003


def _safe_inchikey(mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol) -> str | None:
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_nr_family_pool() -> tuple[pd.DataFrame, dict]:
    """Build the NR-family union pool.

    Returns
    -------
    pool : DataFrame with cols ['inchikey', 'std_smiles', 'pec50',
                                'n_targets', 'has_pxr', 'src_target']
    src_counts : dict {target_name: n_rows_kept}
    """
    frames = []
    src_counts: dict[str, int] = {}

    # ---- 1. chembl_nr_extended -- PPARg, FXR, RXRa, LXRa, PXR, VDR, PPARa ----
    p_nr = EXT_DIR / "chembl_nr_extended.parquet"
    if p_nr.exists():
        d = pd.read_parquet(p_nr)
        d = d[d["target_name"].isin(NR_TARGETS_ALL)].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50", "target_name"]].rename(
            columns={"std_smiles": "smiles", "target_name": "src_target"}
        )
        frames.append(d)
        for t, n in d["src_target"].value_counts().items():
            src_counts[t] = int(n)
            print(f"   [src] chembl_nr_extended  {t:>6s} kept: {n} rows")
    else:
        print("   [src] chembl_nr_extended MISSING")

    # ---- 2. chembl_ahr_activity -- AhR sister-pathway pool ----
    p_ahr = EXT_DIR / "chembl_ahr_activity.parquet"
    if p_ahr.exists():
        d = pd.read_parquet(p_ahr)
        # standard_type {EC50, IC50, Ki, Kd, AC50}; pchembl_value is pre-log
        d = d[d["pchembl_value"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = pd.to_numeric(d["pchembl_value"], errors="coerce")
        d = d[d["pec50"].notna()].copy()
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["smiles", "pec50"]].copy()
        d["src_target"] = "AhR"
        frames.append(d)
        src_counts["AhR"] = int(len(d))
        print(f"   [src] chembl_ahr_activity AhR kept: {len(d)} rows")
    else:
        print("   [src] chembl_ahr_activity MISSING")

    if not frames:
        raise FileNotFoundError(
            "No local NR-family ChEMBL parquets found in data/external/"
        )

    pool_raw = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize NR-family union: {len(pool_raw)} rows")

    # ---- Standardize + InChIKey ----
    mols = pool_raw["smiles"].apply(standardize)
    pool_raw["inchikey"] = mols.apply(_safe_inchikey)
    pool_raw["std_smiles"] = mols.apply(_safe_can_smiles)
    pool_raw = pool_raw[
        pool_raw["inchikey"].notna() & pool_raw["std_smiles"].notna()
    ].copy()
    print(f"   [pool] after RDKit standardize: {len(pool_raw)} rows")

    # ---- Aggregation rule: PXR wins if present, else median ----
    # Per InChIKey:
    #   has_pxr := any src_target == 'PXR'
    #   if has_pxr: pec50 := median(pec50 where src == 'PXR')
    #   else      : pec50 := median(pec50 across all NR targets)
    #   n_targets := nunique(src_target)
    #   src_target := 'PXR' if has_pxr else mode-of-targets
    def _agg(g: pd.DataFrame) -> pd.Series:
        pxr_mask = g["src_target"] == "PXR"
        has_pxr = bool(pxr_mask.any())
        if has_pxr:
            pec50 = float(g.loc[pxr_mask, "pec50"].median())
            src = "PXR"
        else:
            pec50 = float(g["pec50"].median())
            # mode-of-targets, deterministic tie-break by sorted name
            vc = g["src_target"].value_counts()
            src = sorted(vc[vc == vc.max()].index)[0]
        return pd.Series({
            "std_smiles": g["std_smiles"].iloc[0],
            "pec50": pec50,
            "n_targets": int(g["src_target"].nunique()),
            "has_pxr": has_pxr,
            "src_target": src,
        })

    agg = (
        pool_raw.groupby("inchikey", as_index=False, sort=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
    # The groupby returns inchikey as part of the multiindex if include_groups
    # missing; rebuild by re-attaching unique inchikeys via order
    agg["inchikey"] = (
        pool_raw.groupby("inchikey", as_index=False, sort=False)
        .first()["inchikey"].values
    )

    n_pool = len(agg)
    n_pxr = int(agg["has_pxr"].sum())
    n_multi = int((agg["n_targets"] >= 2).sum())
    print(f"   [pool] after InChIKey dedup: {n_pool} unique cpds")
    print(f"   [pool] has_pxr={n_pxr}  multi-target (n_targets>=2)={n_multi}")
    print(f"   [pool] src_target distribution:")
    for t, n in agg["src_target"].value_counts().items():
        print(f"      {t:>6s}: {n}")
    print(f"   [pool] pec50:  mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  "
          f"min={agg['pec50'].min():.3f}  max={agg['pec50'].max():.3f}")

    src_counts["__union_after_dedup__"] = int(n_pool)
    src_counts["__pxr_compounds_kept__"] = n_pxr
    src_counts["__multi_target_compounds__"] = n_multi
    return agg, src_counts


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    """Top-k Tanimoto neighbours.  Returns (top_idx, top_sim)."""
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


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- NR-family ChEMBL extension (PXR + VDR + FXR + AhR + bonus NRs)")
    print(f"          anchor = {ANCHOR}  seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + pred_nr_pec50 + mean_sim  (169)")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- NR-family pool ----
    print("\n" + "-" * 78)
    print("NR-FAMILY POOL (PXR + VDR + FXR + AhR + bonus NRs from local caches)")
    print("-" * 78)
    print(f"   CAR (NR1I3) cache available: {CAR_AVAILABLE}  "
          f"(no local parquet -- marked UNAVAILABLE)")
    pool, src_counts = _load_nr_family_pool()

    # ---- Test InChIKey leak guard ----
    print("\n" + "-" * 78)
    print("TEST-SET LEAK GUARD (drop any pool cpd whose InChIKey appears in 513)")
    print("-" * 78)
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after} "
          f"test-overlapping cpds)")

    # ---- Morgan-2048 ----
    print("\n" + "-" * 78)
    print("MORGAN-2048 FINGERPRINTS")
    print("-" * 78)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    print(f"   pool FP: {fp_pool.shape}  density={fp_pool.mean():.4f}")
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        n_drop = int((~keep_pool).sum())
        print(f"   dropped {n_drop} pool rows with zero FP")
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

    std_test_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in test_mols
    ]
    fp_test = morgan_fp_batch(std_test_smiles)
    print(f"   test FP: {fp_test.shape}  density={fp_test.mean():.4f}")

    # ---- kNN k=5 Tanimoto ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (k={KNN_K}) -- test (513) vs NR-family pool ({len(pool)})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_nr_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    top1_sim = top_sim[:, 0]
    print(f"   pred_nr_pec50  mean={pred_nr_pec50.mean():.3f}  "
          f"std={pred_nr_pec50.std():.3f}  "
          f"min={pred_nr_pec50.min():.3f}  max={pred_nr_pec50.max():.3f}")
    print(f"   top1 sim   p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")
    print(f"   mean5 sim  p10={np.percentile(mean_sim, 10):.3f}  "
          f"p50={np.percentile(mean_sim, 50):.3f}  "
          f"p90={np.percentile(mean_sim, 90):.3f}")
    n_zero_neighbor = int((top1_sim < SIM_FLOOR).sum())
    print(f"   {n_zero_neighbor}/513 test rows had no neighbor "
          f"(fell back to pool median {pool_median:.3f})")

    # Fraction of nearest neighbours that come from PXR vs non-PXR -- gives
    # an interpretable handle on "did NR-family extension actually
    # contribute new neighbours?"
    is_pxr_pool = pool["has_pxr"].to_numpy()
    top1_is_pxr = is_pxr_pool[top_idx[:, 0]]
    frac_top1_pxr = float(top1_is_pxr.mean())
    top5_is_pxr = is_pxr_pool[top_idx]
    frac_top5_pxr = float(top5_is_pxr.mean())
    print(f"   fraction of top-1 neighbours from PXR rows:  {frac_top1_pxr:.3f}")
    print(f"   fraction of top-5 neighbours from PXR rows:  {frac_top5_pxr:.3f}")

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"   MACCS unb shape = {X_maccs_unb.shape}")

    pred_nr_unb = pred_nr_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_nr_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + pred_nr + sim)")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow LGBM Huber, dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae_list: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae_list.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae_list)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae_list)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (PXR-only ChEMBL kNN)")
    print(f"   nb1251 ref             = {NB1251_REF:.4f}  (PXR ChEMBL + BoB blend)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    beats_nb1251 = rae_mean_bag < NB1251_REF - DECISION_MARGIN

    if beats_nb1251:
        verdict = "NR_FAMILY_BEATS_NB1251_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1242:
        verdict = "NR_FAMILY_BEATS_NB1242_BUT_NOT_NB1251"
    elif beats_nb1070:
        verdict = "NR_FAMILY_HELPS_NB1070_BUT_NOT_NB1242"
    elif abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        verdict = "NR_FAMILY_FLAT_VS_NB1242_NO_NEW_SIGNAL"
    elif rae_mean_bag > NB1242_REF + DECISION_MARGIN and rae_mean_bag < rae_anchor:
        verdict = "NR_FAMILY_HELPS_ANCHOR_BUT_HURTS_VS_NB1242"
    else:
        verdict = "NR_FAMILY_HURTS_VS_ANCHOR"
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "data_source": "local_nr_family_chembl_caches_union",
        "chembl_caches": [
            "data/external/chembl_nr_extended.parquet "
            "(PPARg, FXR, RXRa, LXRa, PXR, VDR, PPARa)",
            "data/external/chembl_ahr_activity.parquet (AhR)",
        ],
        "mcp_probed": False,
        "mcp_used_for_pool": False,
        "mcp_probe_note": (
            "MCP claude_ai_ChEMBL was NOT live-queried this run because the local"
            " caches already contain MCP-sourced mirrors of CHEMBL3401 (PXR),"
            " CHEMBL1860 (VDR), CHEMBL2047 (FXR) and CHEMBL3201 (AhR) collected"
            " in earlier cycles.  CAR (NR1I3 / CHEMBL2972) has no local parquet"
            " and is marked UNAVAILABLE."
        ),
        "car_available": CAR_AVAILABLE,
        "nr_targets_core_requested": sorted(["VDR", "CAR", "FXR", "AhR"]),
        "nr_targets_core_in_pool": sorted([
            t for t in ["PXR", "VDR", "FXR", "AhR"]
            if src_counts.get(t, 0) > 0
        ]),
        "nr_targets_bonus_in_pool": sorted([
            t for t in NR_TARGETS_BONUS if src_counts.get(t, 0) > 0
        ]),
        "src_counts": src_counts,
        "n_pool_pre_leakguard": int(n_before),
        "n_pool_post_leakguard": int(len(pool)),
        "test_inchikeys_in_pool_dropped": int(n_before - n_after),
        "n_pxr_compounds_kept": int(pool["has_pxr"].sum()),
        "n_multi_target_compounds": int((pool["n_targets"] >= 2).sum()),
        "pool_pec50_mean": float(pool_labels.mean()),
        "pool_pec50_std": float(pool_labels.std()),
        "pool_pec50_median": pool_median,
        "knn_k": KNN_K,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "top1_sim_max": float(top1_sim.max()),
        "mean5_sim_p10": float(np.percentile(mean_sim, 10)),
        "mean5_sim_p50": float(np.percentile(mean_sim, 50)),
        "mean5_sim_p90": float(np.percentile(mean_sim, 90)),
        "frac_top1_neighbours_from_pxr": frac_top1_pxr,
        "frac_top5_neighbours_from_pxr": frac_top5_pxr,
        "n_zero_neighbor_rows": n_zero_neighbor,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": feat_dim,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae_list,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "delta_mean_bag_vs_nb1251": rae_mean_bag - NB1251_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "beats_nb1251": bool(beats_nb1251),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1242_ref": NB1242_REF,
        "nb1251_ref": NB1251_REF,
        "decision_margin": DECISION_MARGIN,
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
        "n_pool_post_leakguard", "n_pxr_compounds_kept",
        "n_multi_target_compounds",
        "frac_top1_neighbours_from_pxr", "frac_top5_neighbours_from_pxr",
        "top1_sim_p10", "top1_sim_p50", "top1_sim_p90",
        "n_zero_neighbor_rows",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "delta_mean_bag_vs_nb1251",
        "beats_nb1070", "beats_nb1242", "beats_nb1251",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
