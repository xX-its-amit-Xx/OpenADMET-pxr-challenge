"""nb2482 -- Cross-track: Boltz-2 structure features as activity features.

CONTEXT:
    The structure track produced Boltz-2 5-seed poses for the 513 activity
    test ligands. nb416 generated per-ligand Boltz-2 confidence aggregates
    (44 numeric features in data/processed/boltz_dargason_features_test.parquet
    -- mean/max/std/best of confidence_score, ptm, iptm, ligand_iptm,
    complex_plddt, complex_iplddt, complex_pde, complex_ipde, iptm_0_1,
    iptm_1_0, pair_iptm_A_B_mean). This is a genuinely new SUBSTRATE because
    Boltz-2 sees the 3D protein-ligand interface, which neither chemprop_aux
    nor the K=20/K=28 ligand-only fingerprint features observe.

    nb2240 K=20 anchor: chemprop_aux + LGBM(MSE) on RFE-pruned 20 of 117
    features. We extend the feature set with the 44 Boltz-2 dargason
    features (K=20+44=64 cols) and re-train the residual model with the
    same protocol.

PROTOCOL:
    1. Load Boltz-2 features from boltz_dargason_features_test.parquet
       (513 rows aligned to load_test() Molecule Name).
    2. Reuse nb2240 K=20 features (slice the 117-col matrix to surviving 20).
    3. Concatenate -> 64-col X_K64 on 513.
    4. Build residual = y_unb - chemprop_aux[unb_idx]; LGBM mean-bag over
       5 seeds {0, 1, 7, 42, 137}, KFold(5, shuffle, seed_i).
    5. Scaffold 5-fold CV on the 253 unblind, kf_seeds {1001..1005}.
    6. Gate: mean_rae < 0.4570 -> PROMOTE
            mean_rae < 0.4601 -> MARGINAL_BEAT
            else                -> FAIL
       (Baselines: nb2240 K=20 pyramid 0.4676; chemprop_aux alone 0.6216.)
    7. Save nb2482_summary.json + pred_oof (253) + te (513).

If Boltz-2 features missing -> save {"status": "NO_BOLTZ_DATA"} and exit clean.

Outputs:
    scripts/nb2482_boltz_embed_activity.py
    data/processed/nb2482_summary.json
    data/processed/nb2482_pred_oof.npy    (253,) float32
    data/processed/te_nb2482.npy          (513,) float32
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
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2482"
BOLTZ_PARQUET = DATA_PROCESSED / "boltz_dargason_features_test.parquet"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Residual LGBM
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Same K=20 plumbing as nb2240
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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# Scaffold CV
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates (vs nb2171 0.4676 ceiling family)
PROMOTE_GATE = 0.4570
MARGINAL_GATE = 0.4601


# ============================================================================
# helpers (lifted from nb2240)
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


def _scaffold_cv_one_seed(X_unb, residual, anchor, y_unb, scaffolds, kf_seed):
    """Return scaffold-CV pooled RAE on (anchor + LGBM(residual))."""
    splits = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = len(y_unb)
    oof_corr = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        # mean-bag of RESID_SEEDS on residual
        bag_va = np.zeros(len(va_loc), dtype=np.float64)
        for s in RESID_SEEDS:
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb[tr_loc], residual[tr_loc])
            bag_va += mdl.predict(X_unb[va_loc])
        bag_va /= len(RESID_SEEDS)
        oof_corr[va_loc] = anchor[va_loc] + bag_va
    return float(rae(y_unb, oof_corr)), oof_corr


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Boltz-2 structure features as activity features (K=20 + 44 = 64)")
    print("=" * 78)

    # --- Check Boltz parquet ---
    if not BOLTZ_PARQUET.exists():
        msg = f"Boltz parquet missing at {BOLTZ_PARQUET}"
        print(f"[ABORT] {msg}")
        summary = {"tag": TAG, "status": "NO_BOLTZ_DATA", "reason": msg}
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    df_boltz = pd.read_parquet(BOLTZ_PARQUET)
    print(f"[boltz] {df_boltz.shape}  cols={len(df_boltz.columns) - 1} numeric")

    # --- Load truth + anchor ---
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names_col = "name" if "name" in te.columns else "Molecule Name"
    te_names = te[te_names_col].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[anchor] chemprop_aux in-sample RAE on 253 = {rae_anchor:.4f}")

    # --- Align boltz rows to te by Molecule Name ---
    name_to_row = {n: i for i, n in enumerate(df_boltz["name"].astype(str).tolist())}
    boltz_cols = [c for c in df_boltz.columns if c != "name"]
    n_boltz_feats = len(boltz_cols)
    X_boltz = np.zeros((n_test, n_boltz_feats), dtype=np.float32)
    missing = 0
    for i, nm in enumerate(te_names):
        j = name_to_row.get(nm)
        if j is None:
            missing += 1
            continue
        X_boltz[i] = df_boltz.iloc[j][boltz_cols].to_numpy(dtype=np.float32)
    print(f"[align] boltz feat dim={n_boltz_feats}  missing={missing}/{n_test}")
    if missing > n_test * 0.1:
        msg = f"too many missing boltz rows: {missing}/{n_test}"
        print(f"[ABORT] {msg}")
        summary = {"tag": TAG, "status": "NO_BOLTZ_DATA", "reason": msg}
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary
    # NaN-safe: replace any non-finite with column median
    X_boltz = np.where(np.isfinite(X_boltz), X_boltz, np.nan)
    col_med = np.nanmedian(X_boltz, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_boltz)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_boltz[idx_r, idx_c] = col_med[idx_c]
    X_boltz = X_boltz.astype(np.float32)

    # --- Build K=20 features (same as nb2240) ---
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    print(f"[K20] surviving feats from nb2231: {len(surviving_K20)}")

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
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)

    # --- Concatenate K=20 + Boltz ---
    X_te_K64 = np.concatenate([X_te_K20, X_boltz], axis=1).astype(np.float32)
    X_unb_K64 = X_te_K64[unb_idx]
    print(f"[feat] X_unb_K64 = {X_unb_K64.shape}  X_te_K64 = {X_te_K64.shape}")

    # --- Scaffold-CV unb_scaffolds ---
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]

    # --- Scaffold 5-fold CV across 5 kf_seeds ---
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  resid_seeds={RESID_SEEDS}")
    print("-" * 78)
    per_seed_raes = []
    per_seed_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        pooled, oof_corr = _scaffold_cv_one_seed(
            X_unb_K64, residual, anchor, y_unb, unb_scaffolds, kf_seed
        )
        per_seed_raes.append(pooled)
        per_seed_oofs.append(oof_corr)
        print(f"   kf_seed={kf_seed}  scaffold-CV RAE={pooled:.4f}  wall={time.time()-ts:.1f}s")
    mean_rae = float(np.mean(per_seed_raes))
    std_rae = float(np.std(per_seed_raes))
    print(f"\n[cv] mean={mean_rae:.4f} +/- {std_rae:.4f}  range=[{min(per_seed_raes):.4f}, {max(per_seed_raes):.4f}]")

    pred_oof = np.mean(np.column_stack(per_seed_oofs), axis=1).astype(np.float32)

    # --- Deploy refit on 253; mean-bag over 5 seeds; predict on 513 ---
    print("\n[deploy] refit on all 253, predict 513")
    bag_te = np.zeros(n_test, dtype=np.float64)
    for s in RESID_SEEDS:
        te_resid = _train_full_then_predict_te(X_unb_K64, residual, X_te_K64, s)
        bag_te += te_resid
    bag_te /= len(RESID_SEEDS)
    te_pred = (te_anchor_513 + bag_te).astype(np.float32)
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std = {te_pred.mean():.3f}/{te_pred.std():.3f}")

    # --- Gate ---
    if mean_rae < PROMOTE_GATE:
        verdict = "PROMOTE"
    elif mean_rae < MARGINAL_GATE:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[GATE] mean_rae={mean_rae:.4f}  vs PROMOTE<{PROMOTE_GATE}  MARGINAL<{MARGINAL_GATE}")
    print(f"[GATE] verdict = {verdict}")

    # --- Save ---
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof)
    np.save(te_path, te_pred)

    summary = {
        "tag": TAG,
        "method": "K20_RFE_plus_Boltz2_dargason_44feats_residual_LGBM_meanbag",
        "status": "OK",
        "boltz_parquet": str(BOLTZ_PARQUET),
        "n_boltz_feats": int(n_boltz_feats),
        "boltz_missing_rows": int(missing),
        "anchor": "chemprop_aux",
        "anchor_in_sample_rae_unb": rae_anchor,
        "feat_dim_K64": int(X_unb_K64.shape[1]),
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "per_seed_raes": [float(r) for r in per_seed_raes],
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": float(min(per_seed_raes)),
        "max_rae": float(max(per_seed_raes)),
        "te_unb_in_sample_rae": te_unb_rae,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "promote_gate": PROMOTE_GATE,
        "marginal_gate": MARGINAL_GATE,
        "verdict": verdict,
        "compare_nb2171_oof": 0.4676,
        "delta_vs_nb2171": mean_rae - 0.4676,
        "pred_oof_path": str(oof_path),
        "te_path": str(te_path),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   feat_dim K64                = {X_unb_K64.shape[1]}")
    print(f"   anchor (chemprop_aux) RAE   = {rae_anchor:.4f}")
    print(f"   scaffold-CV mean RAE        = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   delta vs nb2171 0.4676      = {mean_rae - 0.4676:+.4f}")
    print(f"   verdict                     = {verdict}")
    print(f"   wall                        = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== FINAL ====")
    for k in ("status", "mean_rae", "std_rae", "verdict", "delta_vs_nb2171", "n_boltz_feats"):
        print(f"  {k}: {res.get(k)}")
