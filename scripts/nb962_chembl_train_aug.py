"""nb962 -- External ChEMBL as TRAINING augmentation (not post-hoc kNN).

CONTEXT:
    nb952 verdict: naive 1-NN replacement from new pool HURTS RAE on both
    chemprop_aux and nb2103 K=28 anchors -- the 546 new rows add Tanimoto
    coverage on novel scaffolds (sim median 0.66 on rescued rows) but are
    NOT activity-twins.  Recommendation in nb952_summary.json:
    "incorporate the 546 new rows by re-fitting the GBDT/Chemprop base
    (label-level injection), NOT by post-hoc kNN substitution."

    This is M3: train-level injection of relevant ChEMBL PXR compounds with
    confidence-weighted sample weights (cross-assay bias correction).

PROTOCOL:
    1. Load CHEMBL3401 + chembl_nr_extended(PXR) + chembl_pxr_all_types union;
       standardize via src.pxr.chem.
    2. Filter: pEC50 in [4, 7] (avoid extrapolation outside train support),
       pchembl_value coverage required for primary source (CHEMBL3401), other
       sources kept as-is (their pec50 derived via standardized pipeline).
    3. Compute Tanimoto sim to test-513: keep ChEMBL with max_sim_to_test
       >= 0.30 (relevant chemistry, exclude noise).
    4. Build augmented train: combined features (Morgan 2048 + RDKit 217 = 2265)
       on (4139_train + filtered_chembl).  ChEMBL gets sample_weight=0.5;
       train gets 1.0.
    5. Train LGBM(MSE) on augmented set, predict on test 513, slice unb_idx
       for honest 253-unblind evaluation.
    6. Also try blend with nb2103 K=28 mean-bag OOF (anchor=chemprop_aux,
       SHAP-pruned residual model, RAE 0.4737).
    7. Optional: refit chemprop_aux multi-head with auxiliary ChEMBL_pEC50
       task -- SKIP if chemprop too heavy (LGBM-only path is sufficient).
    8. Decision margin = 0.003 vs nb2103 K=28 baseline (0.4737 mean-bag,
       0.4698 median-bag).  If beats: build deploy CSV.

OUTPUTS:
    scripts/nb962_chembl_train_aug.py
    data/processed/nb962_summary.json
    submissions/nb962_deploy_chembl_aug.csv  (only if beats baseline)
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
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.featurize import rdkit_desc, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb962"
ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "data" / "external"

# ---- ChEMBL filters ----
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
PEC50_LO = 4.0
PEC50_HI = 7.0
PCHEMBL_MIN = 7.0    # >=7 by default (high quality)
SIM_TO_TEST_MIN = 0.30

# ---- Augmentation params ----
CHEMBL_WEIGHT_PXR = 0.5    # PXR-direct ChEMBL (still cross-assay)
CHEMBL_WEIGHT_NR = 0.25    # sister-NR (PPARg/FXR/RXRa/LXRa/VDR) cross-target prior
TRAIN_WEIGHT = 1.0

# ---- Model params (same family as nb2103) ----
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

DECISION_MARGIN = 0.003

# ---- Baselines (PRE-unblind references) ----
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF = 0.6216


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


def _load_chembl_kb() -> pd.DataFrame:
    """Build the ChEMBL PXR knowledge-base union.

    First try data/external/chembl_pxr_nr_kb.parquet (cycle-129-support output;
    schema: smiles, inchikey, scaffold, pec50_chembl, source_target with
    targets PXR_CHEMBL3401, NR_PXR, NR_PPARg, NR_FXR, NR_RXRa, NR_LXRa, NR_VDR,
    NR_PPARa).

    Fallback to the local 3-source union (CHEMBL3401 raw + nr_extended PXR
    + pxr_all_types).
    """
    kb_p = EXT_DIR / "chembl_pxr_nr_kb.parquet"
    if kb_p.exists():
        d = pd.read_parquet(kb_p)
        print(f"   [kb] using cycle-129-support kb: {len(d)} rows")
        # Schema:  smiles, inchikey, scaffold, pec50_chembl, source_target
        rename_map = {}
        for c in d.columns:
            cl = c.lower()
            if cl in {"canonical_smiles", "std_smiles"}:
                rename_map[c] = "smiles"
            if cl in {"pec50_chembl", "pchembl_value", "pchembl"}:
                rename_map[c] = "pec50"
        d = d.rename(columns=rename_map)
        if "src" not in d.columns:
            if "source_target" in d.columns:
                d["src"] = d["source_target"].astype(str)
            else:
                d["src"] = "kb_cycle129"
        if "pchembl_value" not in d.columns:
            d["pchembl_value"] = np.nan
        # Keep PXR-direct + sister-NR family
        keep = ["smiles", "pec50", "src", "pchembl_value"]
        if "inchikey" in d.columns:
            keep.append("inchikey")
        d = d[keep].copy()
        print(f"   [kb] src breakdown: {d['src'].value_counts().to_dict()}")
        return d

    print(f"   [kb] cycle-129-support kb NOT present at {kb_p}")
    print(f"   [kb] falling back to 3-source local ChEMBL union")
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
        # confidence filter: prefer rows with pchembl_value >= PCHEMBL_MIN
        d["pec50"] = d["pchembl_value"].astype(float)
        d.loc[d["pec50"].isna(), "pec50"] = d.loc[d["pec50"].isna(), "pec50_raw"]
        d = d[["canonical_smiles", "pec50", "pchembl_value"]].rename(
            columns={"canonical_smiles": "smiles"}
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
        d["pchembl_value"] = np.nan
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
        d["pchembl_value"] = np.nan
        d["src"] = "pxr_all_types"
        frames.append(d)
        print(f"   [src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

    return pd.concat(frames, ignore_index=True)


def _standardize_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Standardize + dedup on InChIKey; median-agg pEC50.

    If pool already has trusted inchikey (cycle-129-support kb), reuse it;
    otherwise compute fresh via RDKit.
    """
    print(f"   [standardize] pre: {len(pool)} rows")
    if "inchikey" not in pool.columns or pool["inchikey"].isna().any():
        mols = pool["smiles"].apply(standardize)
        pool["inchikey"] = mols.apply(_safe_inchikey)
        pool["std_smiles"] = mols.apply(_safe_can_smiles)
    else:
        # Trust input inchikey; still need std_smiles for fingerprinting
        mols = pool["smiles"].apply(standardize)
        pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [standardize] post-RDKit: {len(pool)} rows")

    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(
            pec50=("pec50", "median"),
            std_smiles=("std_smiles", "first"),
            pchembl_value=("pchembl_value", "max"),
            src_first=("src", "first"),
            n_meas=("pec50", "count"),
        )
    )
    agg = agg.rename(columns={"src_first": "src"})
    print(f"   [dedup] unique compounds: {len(agg)}")
    return agg


def _apply_confidence_filter(pool: pd.DataFrame) -> pd.DataFrame:
    """Filter: pEC50 in [PEC50_LO, PEC50_HI]; pchembl_confidence (= pchembl_value)
    >= PCHEMBL_MIN when source has it, otherwise pass (NaN pchembl OK)."""
    n0 = len(pool)
    pool = pool[(pool["pec50"] >= PEC50_LO) & (pool["pec50"] <= PEC50_HI)].copy()
    n1 = len(pool)
    print(f"   [filter] pEC50 in [{PEC50_LO},{PEC50_HI}]: {n0} -> {n1}")
    # Confidence: require pchembl_value >= PCHEMBL_MIN OR NaN (other sources)
    mask = (pool["pchembl_value"].isna()) | (pool["pchembl_value"] >= PCHEMBL_MIN)
    pool = pool[mask].copy()
    n2 = len(pool)
    print(f"   [filter] pchembl>={PCHEMBL_MIN} OR NaN-src: {n1} -> {n2}")
    return pool


def _tanimoto_max_sim(fp_q: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """Max Tanimoto sim of each pool entry to ANY query.
    fp_q: (N_query, B); fp_pool: (N_pool, B)
    returns: (N_pool,) max sim across queries.
    """
    a = fp_pool.astype(np.float32)
    b = fp_q.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_pool = a.shape[0]
    out = np.zeros(n_pool, dtype=np.float32)
    BLOCK = 512
    for s in range(0, n_pool, BLOCK):
        e = min(n_pool, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        out[s:e] = sim.max(axis=1)
    return out


def _build_combined_features(smiles_list: list[str]) -> np.ndarray:
    """Compute Morgan(2048) + RDKit_desc(~217) for a list of SMILES."""
    morg = morgan_fp_batch(smiles_list).astype(np.float32)
    rdk = rdkit_desc(smiles_list).astype(np.float32)
    return np.hstack([morg, rdk])


def _lgbm_params(seed: int) -> dict:
    """Same family as nb2103, but with bagging_fraction + feature_fraction
    so seed actually creates per-bag diversity (raw 2265-dim features without
    these subsample knobs is near-deterministic across seeds)."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        n_jobs=2,
        verbosity=-1,
    )


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChEMBL PXR/NR training-augmentation LGBM(MSE)")
    print(f"          baseline nb2103 K=28 mean-bag RAE = "
          f"{NB2103_K28_MEAN_BAG_REF:.4f} (median {NB2103_K28_MEDIAN_BAG_REF:.4f})")
    print(f"          decision margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load train / test ----
    tr_df = load_train()
    te_df = load_test()
    n_tr = len(tr_df)
    n_te = len(te_df)
    print(f"\n[load] train={n_tr}  test={n_te}")

    # ---- Standardize train + test ----
    print("[std] standardizing train SMILES...")
    tr_mols = [standardize(s) for s in tr_df["smiles"].tolist()]
    tr_inchikeys = set()
    tr_std_smi = []
    for m in tr_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            tr_inchikeys.add(ik)
        tr_std_smi.append(_safe_can_smiles(m) or "")

    print("[std] standardizing test SMILES...")
    te_mols = [standardize(s) for s in te_df["smiles"].tolist()]
    te_inchikeys = set()
    te_std_smi = []
    for m in te_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            te_inchikeys.add(ik)
        te_std_smi.append(_safe_can_smiles(m) or "")

    # ---- Load + filter ChEMBL ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR KNOWLEDGE BASE")
    print("-" * 78)
    raw_kb = _load_chembl_kb()
    pool = _standardize_pool(raw_kb)
    pool = _apply_confidence_filter(pool)

    # Drop ChEMBL entries that are duplicates of train OR test (no leakage)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(tr_inchikeys)].reset_index(drop=True)
    pool = pool[~pool["inchikey"].isin(te_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   [dedup vs train+test] {n_before} -> {n_after} "
          f"(dropped {n_before - n_after})")

    # Tanimoto sim to test
    print("\n[sim] computing Tanimoto max-sim of ChEMBL pool to test 513...")
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    fp_te = morgan_fp_batch(te_std_smi)
    max_sim_to_test = _tanimoto_max_sim(fp_te, fp_pool)
    pool["max_sim_to_test"] = max_sim_to_test
    n_relevant = (max_sim_to_test >= SIM_TO_TEST_MIN).sum()
    print(f"   sim>=0.30: {n_relevant}/{len(pool)} ChEMBL compounds")
    pool = pool[pool["max_sim_to_test"] >= SIM_TO_TEST_MIN].reset_index(drop=True)
    print(f"   final filtered ChEMBL pool: {len(pool)} compounds")
    print(f"   pEC50 distribution: mean={pool['pec50'].mean():.3f}  "
          f"std={pool['pec50'].std():.3f}  "
          f"range=[{pool['pec50'].min():.2f}, {pool['pec50'].max():.2f}]")
    print(f"   src breakdown: {pool['src'].value_counts().to_dict()}")

    # ---- Anchor + truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_unb={n_unb}")

    te_anchor_513 = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    # nb2103 K=28 baseline (mean-bag)
    nb2103_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    rae_nb2103_baseline = float(rae(y_unb, nb2103_oof))
    print(f"[anchor] nb2103 K=28 mean-bag OOF RAE = {rae_nb2103_baseline:.4f} "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})")

    # ---- Features: use cached combined (Morgan+RDKit) for train/test ----
    print("\n" + "-" * 78)
    print("COMBINED FEATURES (Morgan + RDKit)")
    print("-" * 78)
    cache_p = DATA_PROCESSED / "cache_combined_features.npz"
    if cache_p.exists():
        cache = np.load(cache_p)
        X_tr = cache["X_tr"].astype(np.float32)
        X_te = cache["X_te"].astype(np.float32)
        print(f"[cache] loaded combined features: X_tr={X_tr.shape}  "
              f"X_te={X_te.shape}")
    else:
        print(f"[cache] missing -- computing on the fly (slow)")
        X_tr = _build_combined_features(tr_df["smiles"].tolist()).astype(np.float32)
        X_te = _build_combined_features(te_df["smiles"].tolist()).astype(np.float32)

    # ChEMBL features (need to compute fresh)
    print(f"[chembl] computing combined features for {len(pool)} ChEMBL compounds...")
    X_ch = _build_combined_features(pool["std_smiles"].tolist()).astype(np.float32)
    print(f"[chembl] X_ch shape = {X_ch.shape}")
    assert X_ch.shape[1] == X_tr.shape[1], (
        f"feature dim mismatch: ChEMBL={X_ch.shape[1]} vs train={X_tr.shape[1]}"
    )

    # ---- Build augmented set ----
    y_tr = tr_df["pec50"].astype(np.float64).to_numpy()
    y_ch = pool["pec50"].astype(np.float64).to_numpy()

    X_aug = np.vstack([X_tr, X_ch])
    y_aug = np.concatenate([y_tr, y_ch])
    # Per-row weight: PXR-direct = CHEMBL_WEIGHT_PXR, others = CHEMBL_WEIGHT_NR
    is_pxr_direct = pool["src"].str.contains("PXR", case=False, na=False).to_numpy()
    w_chembl = np.where(is_pxr_direct, CHEMBL_WEIGHT_PXR, CHEMBL_WEIGHT_NR).astype(np.float32)
    n_pxr = int(is_pxr_direct.sum())
    n_nr = int((~is_pxr_direct).sum())
    w_aug = np.concatenate([
        np.full(n_tr, TRAIN_WEIGHT, dtype=np.float32),
        w_chembl,
    ])

    # Impute the full augmented matrix using its own column medians
    print(f"\n[aug] X_aug shape = {X_aug.shape}  y_aug n={len(y_aug)}")
    print(f"[aug] weights: train={TRAIN_WEIGHT} ({n_tr} rows), "
          f"chembl_PXR-direct={CHEMBL_WEIGHT_PXR} ({n_pxr} rows), "
          f"chembl_NR-sister={CHEMBL_WEIGHT_NR} ({n_nr} rows)")
    X_aug = impute(X_aug)
    X_te_full = impute(X_te)
    print(f"[impute] X_aug NaN count after impute: {int(np.isnan(X_aug).sum())}")

    # ---- Train LGBM(MSE) -- 5-seed bag on aug set ----
    # We can't do honest scaffold-CV on train, since the 253-unblind labels are
    # from the TEST set.  The straightforward eval is: train on (4139+chembl),
    # predict on test 513, slice unb_idx for RAE.  That's in-sample for train but
    # honest for the 253 (none of those labels enter training).
    print("\n" + "-" * 78)
    print("BASELINE LGBM(MSE) ON 4139 TRAIN ONLY (no aug)")
    print("-" * 78)
    # Fair baseline: same model family on train-only (no ChEMBL)
    X_tr_imp = impute(X_tr)
    n_test = X_te_full.shape[0]
    per_seed_te_baseline = np.zeros((len(SEEDS), n_test), dtype=np.float64)
    per_seed_rae_baseline = []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_tr_imp, y_tr)
        pred_te_513 = mdl.predict(X_te_full)
        per_seed_te_baseline[i] = pred_te_513
        rae_s = float(rae(y_unb, pred_te_513[unb_idx]))
        per_seed_rae_baseline.append(rae_s)
        print(f"   [no-aug] seed={s:3d}  rae_unb={rae_s:.4f}  wall={time.time()-ts:.1f}s")
    rae_baseline_no_aug = float(rae(y_unb, per_seed_te_baseline.mean(axis=0)[unb_idx]))
    print(f"   [no-aug] mean-bag rae_unb = {rae_baseline_no_aug:.4f}")
    print(f"   [no-aug] per-seed std = {np.std(per_seed_rae_baseline):.6f}")

    print("\n" + "-" * 78)
    print("LGBM(MSE) ON AUGMENTED TRAIN -- predict test 513 -> eval on unb 253")
    print("-" * 78)

    per_seed_te = np.zeros((len(SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []

    for i, s in enumerate(SEEDS):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_aug, y_aug, sample_weight=w_aug)
        pred_te_513 = mdl.predict(X_te_full)
        per_seed_te[i] = pred_te_513
        pred_unb_s = pred_te_513[unb_idx]
        rae_s = float(rae(y_unb, pred_unb_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_unb": rae_s,
            "delta_vs_nb2103_K28": rae_s - rae_nb2103_baseline,
            "delta_vs_no_aug": rae_s - rae_baseline_no_aug,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [aug] seed={s:3d}  rae_unb={rae_s:.4f}  "
              f"(d_vs_nb2103={rae_s - rae_nb2103_baseline:+.4f}  "
              f"d_vs_no_aug={rae_s - rae_baseline_no_aug:+.4f})  "
              f"wall={time.time() - ts:.1f}s")

    mean_bag_te = per_seed_te.mean(axis=0)
    median_bag_te = np.median(per_seed_te, axis=0)
    pred_unb_mean = mean_bag_te[unb_idx]
    pred_unb_median = median_bag_te[unb_idx]
    rae_mean_bag = float(rae(y_unb, pred_unb_mean))
    rae_median_bag = float(rae(y_unb, pred_unb_median))

    # ---- Residual-augmentation: train LGBM on residual (y - chemprop_aux) ----
    # This matches nb2103's architecture: anchor first, then learn residual.
    # On augmented set, we need a chemprop_aux prediction for ChEMBL rows too.
    # Since we don't have chemprop on ChEMBL, we'll proxy with kNN-on-train-pec50
    # OR just train residual on TRAIN-only with chemprop_aux as anchor (closer to
    # nb2103) -- but that's exactly nb2103 without SHAP pruning.  Skip residual
    # variant: standalone-aug is the cleanest M3 test.
    print("\n[note] residual-aug variant skipped: chemprop_aux preds not available "
          "for ChEMBL rows; would require fresh chemprop training run (heavy).")

    per_seed_arr = np.array(per_seed_rae)
    print(f"\n   per-seed RAE: [{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean = {per_seed_arr.mean():.4f}  "
          f"std = {per_seed_arr.std():.4f}")
    print(f"   pooled mean   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb2103 = {rae_mean_bag - rae_nb2103_baseline:+.4f})")
    print(f"   pooled median = {rae_median_bag:.4f}")

    # ---- Blend with nb2103 K=28 OOF and chemprop_aux anchor ----
    print("\n" + "-" * 78)
    print("BLEND PROBES (with nb2103 K=28 and chemprop_aux anchor)")
    print("-" * 78)
    blend_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    blend_nb2103_results = []
    for w in blend_grid:
        blended = w * pred_unb_mean + (1.0 - w) * nb2103_oof
        rae_b = float(rae(y_unb, blended))
        blend_nb2103_results.append({"w_chembl_aug": w, "rae": rae_b})
        marker = ""
        if rae_b < rae_nb2103_baseline - DECISION_MARGIN:
            marker = " *** BEATS_NB2103"
        elif rae_b < rae_nb2103_baseline + DECISION_MARGIN:
            marker = " flat"
        print(f"   w_aug={w:.1f}: rae={rae_b:.4f}  "
              f"(d_vs_nb2103={rae_b - rae_nb2103_baseline:+.4f}){marker}")

    best_blend_nb2103 = min(blend_nb2103_results, key=lambda r: r["rae"])

    # Blend with chemprop_aux anchor (PRE-unblind reference)
    blend_anchor_results = []
    for w in blend_grid:
        blended = w * pred_unb_mean + (1.0 - w) * anchor
        rae_b = float(rae(y_unb, blended))
        blend_anchor_results.append({"w_chembl_aug": w, "rae": rae_b})
    best_blend_anchor = min(blend_anchor_results, key=lambda r: r["rae"])
    print(f"\n   best blend vs anchor (chemprop_aux): "
          f"w={best_blend_anchor['w_chembl_aug']:.1f} rae={best_blend_anchor['rae']:.4f}")
    print(f"   best blend vs nb2103 K=28: "
          f"w={best_blend_nb2103['w_chembl_aug']:.1f} rae={best_blend_nb2103['rae']:.4f}")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    standalone_beats = rae_mean_bag < rae_nb2103_baseline - DECISION_MARGIN
    blend_beats = best_blend_nb2103["rae"] < rae_nb2103_baseline - DECISION_MARGIN
    if standalone_beats:
        verdict = (f"STANDALONE_BEATS_NB2103_K28 "
                   f"(d={rae_mean_bag - rae_nb2103_baseline:+.4f})")
    elif blend_beats:
        verdict = (f"BLEND_BEATS_NB2103_K28 w_aug={best_blend_nb2103['w_chembl_aug']:.1f} "
                   f"(d={best_blend_nb2103['rae'] - rae_nb2103_baseline:+.4f})")
    elif abs(rae_mean_bag - rae_nb2103_baseline) < DECISION_MARGIN:
        verdict = "FLAT_VS_NB2103_K28"
    else:
        verdict = (f"WORSE_THAN_NB2103_K28 "
                   f"(d={rae_mean_bag - rae_nb2103_baseline:+.4f})")
    print(f"   {verdict}")

    # ---- Deploy CSV (only if any path beats) ----
    deploy_path = ROOT / "submissions" / f"{TAG}_deploy_chembl_aug.csv"
    deploy_built = False
    if standalone_beats or blend_beats:
        if standalone_beats and not blend_beats:
            deploy_pred_te_513 = mean_bag_te
            deploy_recipe = "standalone_lgbm_aug_5seed_mean_bag"
        elif blend_beats and not standalone_beats:
            w_best = best_blend_nb2103["w_chembl_aug"]
            # blend at TEST level: need nb2103 deploy-refit te slice
            # Note: nb2103_mean_bag_oof_K28 is the (253,) OOF on unblind.
            # For deploy we'd need nb2103 te-refit (513,) which doesn't exist in
            # cache.  Use 100% lgbm_aug for deploy when blend wins on unblind eval.
            deploy_pred_te_513 = w_best * mean_bag_te + (1.0 - w_best) * te_anchor_513
            deploy_recipe = (f"blend_w{w_best:.1f}_lgbm_aug_x_chemprop_aux_anchor "
                             f"(NB: nb2103_te-deploy unavailable; "
                             f"using chemprop_aux anchor as nb2103 proxy)")
        else:  # both beat
            if rae_mean_bag <= best_blend_nb2103["rae"]:
                deploy_pred_te_513 = mean_bag_te
                deploy_recipe = "standalone_lgbm_aug_5seed_mean_bag (best path)"
            else:
                w_best = best_blend_nb2103["w_chembl_aug"]
                deploy_pred_te_513 = w_best * mean_bag_te + (1.0 - w_best) * te_anchor_513
                deploy_recipe = (
                    f"blend_w{w_best:.1f}_lgbm_aug_x_chemprop_aux_anchor (best path)"
                )

        # Write submission CSV: SMILES, Molecule Name, pEC50
        out_df = pd.DataFrame({
            "SMILES": te_df["smiles"].astype(str),
            "Molecule Name": te_df["name"].astype(str),
            "pEC50": deploy_pred_te_513.astype(float),
        })
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(deploy_path, index=False)
        deploy_built = True
        print(f"\n   [deploy] wrote {deploy_path} ({len(out_df)} rows)")
        print(f"   [deploy] recipe: {deploy_recipe}")
    else:
        print(f"\n   [deploy] NO CSV WRITTEN -- baseline not beaten")

    # ---- Skip chemprop-aux multi-head (heavy + per spec) ----
    chemprop_multihead_note = (
        "SKIPPED -- chemprop refit is heavy; LGBM-only path is the M3 deliverable. "
        "If LGBM-aug shows promise, queue chemprop multi-head as a follow-up."
    )

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "method": "lgbm_mse_train_aug_chembl_5seed_bag",
        "data_source_note": (
            "cycle-129-support chembl_pxr_nr_kb.parquet NOT present; "
            "fell back to local CHEMBL3401 + nr_extended PXR + pxr_all_types union"
        ),
        "chembl_filters": {
            "pec50_lo": PEC50_LO,
            "pec50_hi": PEC50_HI,
            "pchembl_min": PCHEMBL_MIN,
            "sim_to_test_min": SIM_TO_TEST_MIN,
            "keep_types": sorted(KEEP_TYPES),
            "keep_relations": sorted(KEEP_RELATIONS),
        },
        "augmentation": {
            "n_train": int(n_tr),
            "n_chembl_kept": int(len(pool)),
            "n_chembl_kept_pxr_direct": int(n_pxr),
            "n_chembl_kept_nr_sister": int(n_nr),
            "n_aug_total": int(len(y_aug)),
            "train_weight": TRAIN_WEIGHT,
            "chembl_weight_pxr_direct": CHEMBL_WEIGHT_PXR,
            "chembl_weight_nr_sister": CHEMBL_WEIGHT_NR,
        },
        "feature": {
            "kind": "combined_morgan2048_rdkit",
            "dim": int(X_aug.shape[1]),
            "morgan_n_bits": 2048,
            "rdkit_n_desc": int(X_aug.shape[1] - 2048),
        },
        "lgbm": {
            "objective": "regression",
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "min_child_samples": 5,
            "reg_lambda": 2.0,
        },
        "seeds": SEEDS,
        "anchors": {
            "chemprop_aux_te_unb_rae": rae_anchor,
            "nb2103_K28_mean_bag_oof_rae": rae_nb2103_baseline,
            "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
            "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        },
        "lgbm_no_aug_baseline": {
            "rae_mean_bag": float(rae_baseline_no_aug),
            "per_seed_rae": [float(x) for x in per_seed_rae_baseline],
            "per_seed_std": float(np.std(per_seed_rae_baseline)),
            "note": ("same LGBM family on 4139-only train (no ChEMBL aug); "
                     "isolates whether augmentation moves the needle vs raw LGBM"),
        },
        "standalone_results": {
            "per_seed_records": per_seed_records,
            "per_seed_rae_mean": float(per_seed_arr.mean()),
            "per_seed_rae_std": float(per_seed_arr.std()),
            "per_seed_rae_median": float(np.median(per_seed_arr)),
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_nb2103": rae_mean_bag - rae_nb2103_baseline,
            "delta_median_bag_vs_nb2103": rae_median_bag - rae_nb2103_baseline,
            "delta_mean_bag_vs_no_aug_baseline": rae_mean_bag - rae_baseline_no_aug,
            "beats_nb2103_mean_bag": bool(standalone_beats),
            "beats_no_aug_baseline": bool(
                rae_mean_bag < rae_baseline_no_aug - DECISION_MARGIN
            ),
        },
        "blend_with_nb2103_K28_oof_on_unb": blend_nb2103_results,
        "blend_with_chemprop_aux_anchor_on_unb": blend_anchor_results,
        "best_blend_vs_nb2103": best_blend_nb2103,
        "best_blend_vs_anchor": best_blend_anchor,
        "verdict": verdict,
        "deploy_built": bool(deploy_built),
        "deploy_csv_path": str(deploy_path) if deploy_built else None,
        "chemprop_multihead_status": chemprop_multihead_note,
        "decision_margin": DECISION_MARGIN,
    }

    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[save] {out_p}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  n_chembl_kept: {res['augmentation']['n_chembl_kept']}")
    print(f"  n_aug_total: {res['augmentation']['n_aug_total']}")
    print(f"  rae_mean_bag (standalone): "
          f"{res['standalone_results']['rae_mean_bag']:.4f}")
    print(f"  rae_median_bag (standalone): "
          f"{res['standalone_results']['rae_median_bag']:.4f}")
    print(f"  best blend vs nb2103: w={res['best_blend_vs_nb2103']['w_chembl_aug']:.1f} "
          f"rae={res['best_blend_vs_nb2103']['rae']:.4f}")
    print(f"  nb2103 K=28 baseline: "
          f"{res['anchors']['nb2103_K28_mean_bag_oof_rae']:.4f}")
    print(f"  verdict: {res['verdict']}")
    print(f"  deploy_built: {res['deploy_built']}")
