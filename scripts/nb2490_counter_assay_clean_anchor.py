"""nb2490 -- Clean counter-assay anchor (replaces nb730 contamination chain).

CONTEXT (per feedback_anchor_contamination_chain.md):
    nb730's te[unb_idx] == nb730_pred_oof bit-identical (coarse lambda grid
    coincidence, NOT honesty). All hybrids built on nb730 anchor carry a
    +0.10-0.15 RAE LB penalty (cf. nb2189/nb2201). Only te_chemprop_aux is
    verified PRE-clean. This script builds a fresh counter-assay-axis anchor
    from scratch using:
      - 2858 counter-assay (PXR-null) DRC rows (dedup by std_smi median)
      - LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03, lambda=2.0)
      - combined features (Morgan-2048 + RDKit-217 = 2265 dims)
      - HONEST 5-fold scaffold-CV (not random KFold per
        feedback_cv_protocol_audit.md)
      - 5-seed mean-bag

PROTOCOL:
    1. Load counter-assay (2858 rows), std SMILES, dedup -> n_uniq rows.
    2. Compute combined features (Morgan+RDKit) + Murcko scaffold.
    3. Counter-assay predictor (target = pec50_null):
       5-fold scaffold-CV (5 seeds {0,1,7,42,137}) -> nb2490_counter_oof.npy
       (n_uniq,) on counter labels.
       Refit on full 2858 + 5 seeds -> te_nb2490_counter.npy (513,) clean
       counter pEC50 predictions on test.
    4. Build K=20 RFE residual on (chemprop_aux + counter_clean) joint anchor:
       - Joint anchor on 253 unb = chemprop_aux + counter_clean[unb_idx]
       - Joint anchor on 513 te = chemprop_aux_te + counter_clean_te
       - residual = y_unb - joint_anchor (NaN-safe)
       - Features = X_unb sliced to K=20 RFE-surviving indices from
         nb2231_summary.json snapshots.20.surviving_idx_in_117
       - 5-fold scaffold-CV LGBM(MSE), 5-seed mean-bag.
    5. nb2490_pred_oof.npy (253,) on 253 unblind (joint_anchor + resid_oof)
       te_nb2490.npy (513,) deploy refit on full 253 (joint_anchor + resid_te)
    6. Gate: mean_rae < 0.4570 -> PROMOTE; < 0.4601 -> MARGINAL_BEAT; else FAIL.

Outputs:
    scripts/nb2490_counter_assay_clean_anchor.py
    data/processed/nb2490_counter_oof.npy        (n_uniq counter,) float32
    data/processed/te_nb2490_counter.npy         (513,) float32
    data/processed/nb2490_pred_oof.npy           (253,) float32
    data/processed/te_nb2490.npy                 (513,) float32
    data/processed/nb2490_summary.json
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
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_counter, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb2490"

# References
CHEMPROP_AUX_TE = DATA_PROCESSED / "te_chemprop_aux.npy"
UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

# Pre-built 117-col K-tuned feature pieces (same as nb2240/nb2171)
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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# Hyperparameters
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

# Gate thresholds
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


def _murcko(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) or ""
    except Exception:
        return ""


def _lgbm_params(seed: int) -> dict:
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


def _scaffold_cross_fit(
    X: np.ndarray,
    y: np.ndarray,
    scaffolds: list,
    seed: int,
) -> np.ndarray:
    folds = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for tr_idx, va_idx in folds:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = mdl.predict(X[va_idx])
    assert not np.any(np.isnan(oof)), "OOF has NaNs"
    return oof


def _refit_full_predict_test(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te).astype(np.float64)


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


def _build_X_te_117(te_smiles, n_test):
    """Build the 117-col 5-way K-tuned feature matrix on the 513 test (same recipe as nb2240)."""
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
    feat_dim = X_te_full.shape[1]
    assert feat_dim == 117, f"feat_dim {feat_dim} != 117"
    return X_te_full


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- clean counter-assay anchor (replaces nb730 contamination)")
    print(f"     gates: PROMOTE<{GATE_PROMOTE}  MARGINAL_BEAT<{GATE_MARGINAL}")
    print("=" * 78)

    summary: dict = {
        "tag": TAG,
        "purpose": "Clean counter-assay anchor (nb730-free) + K=20 RFE residual on joint anchor",
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
    }

    # ---- Step 1: Counter-assay data prep ----
    print("\n[step1] loading counter-assay (PXR-null) data ...")
    co = load_counter()
    co["std_smi"] = co["smiles"].apply(
        lambda s: Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else None
    )
    co_g = (
        co[co["std_smi"].notna() & co["pec50"].notna()]
        .groupby("std_smi", as_index=False)
        .agg(pec50_null=("pec50", "median"))
    )
    smiles_co = co_g["std_smi"].tolist()
    y_null = co_g["pec50_null"].to_numpy(dtype=np.float64)
    scaffolds_co = [_murcko(s) for s in smiles_co]
    n_uniq = len(smiles_co)
    print(f"[step1] counter rows raw      = {len(co)}")
    print(f"[step1] n_uniq_std_smi (pec50)= {n_uniq}")
    print(f"[step1] pec50_null mean/std    = {y_null.mean():.3f}/{y_null.std():.3f}")

    # ---- Step 2: Combined features ----
    print("\n[step2] computing combined (Morgan+RDKit) features ...")
    X_co = impute(combined(smiles_co)).astype(np.float32)
    print(f"[step2] X_co shape = {X_co.shape}")

    te_df = load_test()
    te_smiles_raw = te_df["smiles"].astype(str).tolist()
    te_names = te_df["name"].values if "name" in te_df.columns else te_df["Molecule Name"].values
    n_test = len(te_smiles_raw)
    te_std_smi = [
        Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else s
        for s in te_smiles_raw
    ]
    X_te_combined = impute(combined(te_std_smi)).astype(np.float32)
    print(f"[step2] X_te shape = {X_te_combined.shape}")

    # ---- Step 3: Counter-assay predictor (5-fold scaffold-CV) ----
    print("\n[step3] training counter-assay LGBM (target = pec50_null, scaffold-CV) ...")
    null_oof_per_seed = []
    null_te_per_seed = []
    null_per_seed_rae = []
    for s in SEEDS:
        ts = time.time()
        oof_s = _scaffold_cross_fit(X_co, y_null, scaffolds_co, seed=s)
        null_oof_per_seed.append(oof_s)
        rae_s = float(rae(y_null, oof_s))
        null_per_seed_rae.append(rae_s)
        # Refit on full + predict test
        te_s = _refit_full_predict_test(X_co, y_null, X_te_combined, seed=s)
        null_te_per_seed.append(te_s)
        print(f"   seed={s:3d}  counter OOF RAE={rae_s:.4f}  wall={time.time()-ts:.1f}s")
    null_oof_mean = np.mean(np.stack(null_oof_per_seed, axis=0), axis=0)
    null_te_mean = np.mean(np.stack(null_te_per_seed, axis=0), axis=0)
    null_rae_meanbag = float(rae(y_null, null_oof_mean))
    null_rae_perseed_mean = float(np.mean(null_per_seed_rae))
    print(f"[step3] per-seed mean counter RAE = {null_rae_perseed_mean:.4f}")
    print(f"[step3] mean-bag counter RAE      = {null_rae_meanbag:.4f}")
    print(f"[step3] te counter mean/std        = {null_te_mean.mean():+.3f}/{null_te_mean.std():.3f}")

    # Save counter outputs
    out_counter_oof = DATA_PROCESSED / f"{TAG}_counter_oof.npy"
    out_counter_te = DATA_PROCESSED / f"te_{TAG}_counter.npy"
    np.save(out_counter_oof, null_oof_mean.astype(np.float32))
    np.save(out_counter_te, null_te_mean.astype(np.float32))
    print(f"[save] {out_counter_oof}")
    print(f"[save] {out_counter_te}")

    # ---- Step 4: Joint anchor and K=20 residual on 253 ----
    print("\n[step4] building joint anchor (chemprop_aux + counter_clean) ...")
    te_chemprop_aux_513 = np.load(CHEMPROP_AUX_TE).astype(np.float64)
    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[step4] n_test={n_test}  n_unb={n_unb}")

    anchor_cp_unb = te_chemprop_aux_513[unb_idx]
    anchor_co_unb = null_te_mean.astype(np.float64)[unb_idx]
    # Joint anchor = simple average (chemprop_aux on pec50 axis, counter on null axis)
    joint_anchor_unb = 0.5 * anchor_cp_unb + 0.5 * anchor_co_unb
    joint_anchor_te = 0.5 * te_chemprop_aux_513 + 0.5 * null_te_mean.astype(np.float64)

    rae_cp = float(rae(y_unb, anchor_cp_unb))
    rae_co = float(rae(y_unb, anchor_co_unb))
    rae_joint = float(rae(y_unb, joint_anchor_unb))
    print(f"[step4] chemprop_aux unb RAE     = {rae_cp:.4f}")
    print(f"[step4] counter_clean unb RAE    = {rae_co:.4f}")
    print(f"[step4] joint anchor unb RAE     = {rae_joint:.4f}")

    residual = y_unb - joint_anchor_unb
    print(f"[step4] residual mean/std        = {residual.mean():+.3f}/{residual.std():.3f}")

    # ---- Step 5: Load K=20 RFE features + scaffolds for unblind ----
    print("\n[step5] loading K=20 RFE features (nb2231 surviving idx) ...")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[step5] K=20 surviving feature names (sample): {surviving_K20_names[:5]}...")

    X_te_117 = _build_X_te_117(te_smiles_raw, n_test)
    X_te_K20 = X_te_117[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[step5] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    unb_smiles = [te_smiles_raw[i] for i in unb_idx]
    unb_scaffolds = [_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[step5] n_unique_scaffolds unb    = {n_unique_scaf}")

    # ---- Step 6: 5-fold scaffold-CV K=20 LGBM residual on 253 ----
    print("\n[step6] 5-fold scaffold-CV K=20 LGBM(MSE) on residual ...")
    per_seed_corrected = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        # Scaffold-CV on residual using unb scaffolds
        folds = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=s)
        resid_oof = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in folds:
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_K20[tr_loc], residual[tr_loc])
            resid_oof[va_loc] = mdl.predict(X_unb_K20[va_loc])
        assert not np.any(np.isnan(resid_oof)), "residual OOF has NaNs"
        per_seed_corrected[i] = joint_anchor_unb + resid_oof
        per_seed_rae.append(float(rae(y_unb, joint_anchor_unb + resid_oof)))
        # Deploy refit on full 253 -> predict residual on full 513
        mdl_full = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl_full.fit(X_unb_K20, residual)
        te_resid_s = mdl_full.predict(X_te_K20).astype(np.float64)
        per_seed_te_resid[i] = te_resid_s
        print(f"   seed={s:3d}  rae_corr={per_seed_rae[-1]:.4f}  wall={time.time()-ts:.1f}s")

    pred_oof_meanbag = per_seed_corrected.mean(axis=0)  # (253,) = joint_anchor + mean-bag resid
    te_resid_meanbag = per_seed_te_resid.mean(axis=0)   # (513,)
    deploy_te = (joint_anchor_te + te_resid_meanbag).astype(np.float64)

    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    meanbag_rae = float(rae(y_unb, pred_oof_meanbag))
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[cv] per-seed mean RAE  = {mean_rae:.4f} (+/- {std_rae:.4f})")
    print(f"[cv] mean-bag RAE       = {meanbag_rae:.4f}")
    print(f"[cv] te[unb_idx] RAE    = {te_unb_rae:.4f}  (in-sample deploy)")
    print(f"[cv] deploy_te mean/std = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    # ---- Save artifacts ----
    out_pred_oof = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    out_te = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(out_pred_oof, pred_oof_meanbag.astype(np.float32))
    np.save(out_te, deploy_te.astype(np.float32))
    print(f"\n[save] {out_pred_oof}")
    print(f"[save] {out_te}")

    # ---- Gate decision ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print(f"GATE: mean_rae={mean_rae:.4f}  ->  {verdict}")
    print(f"      PROMOTE<{GATE_PROMOTE}  MARGINAL_BEAT<{GATE_MARGINAL}")
    print("-" * 78)

    summary.update({
        "n_uniq_counter": n_uniq,
        "n_test": n_test,
        "n_unb": n_unb,
        "n_unique_scaffolds_unb": n_unique_scaf,
        "feat_dim_combined": int(X_co.shape[1]),
        "feat_dim_K20": int(X_unb_K20.shape[1]),
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names_first5": surviving_K20_names[:5],
        "counter_per_seed_rae": [float(r) for r in null_per_seed_rae],
        "counter_per_seed_mean_rae": null_rae_perseed_mean,
        "counter_mean_bag_rae": null_rae_meanbag,
        "counter_te_mean": float(null_te_mean.mean()),
        "counter_te_std": float(null_te_mean.std()),
        "out_counter_oof_path": str(out_counter_oof),
        "out_counter_te_path": str(out_counter_te),
        "rae_chemprop_aux_unb": rae_cp,
        "rae_counter_clean_unb": rae_co,
        "rae_joint_anchor_unb": rae_joint,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "resid_per_seed_rae": [float(r) for r in per_seed_rae],
        "mean_rae_per_seed": mean_rae,
        "std_rae_per_seed": std_rae,
        "mean_bag_rae": meanbag_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "out_pred_oof_path": str(out_pred_oof),
        "out_te_path": str(out_te),
        "gate_verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    })

    out_summary = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_summary}")
    print(f"[done] wall = {time.time() - t0:.1f}s")

    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_uniq_counter", "feat_dim_K20",
        "counter_per_seed_mean_rae", "counter_mean_bag_rae",
        "rae_chemprop_aux_unb", "rae_counter_clean_unb", "rae_joint_anchor_unb",
        "mean_rae_per_seed", "std_rae_per_seed", "mean_bag_rae",
        "te_unb_rae_in_sample", "gate_verdict",
    ):
        print(f"  {k}: {res.get(k)}")
