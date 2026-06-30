"""nb2480 -- 96-compound uscale-semi-pure DRC augmentation.

Hypothesis test: does adding the 96 uscale-semi-pure compounds to the
training pool (alongside chemprop_aux-as-feature + K=20 RFE features)
break the current 0.4601 pyramid ceiling?

PROTOCOL:
    1. Load 4139 train + 96 uscale = 4235 rows. PRE-clean (no POST anchor).
       Compute 117-col 5-way feature matrix (same families as nb2240) on
       all 4235 + 513 test (and 253 unblind = test[unb_idx]).
    2. Slice to K=20 RFE-surviving feature columns from nb2231.
    3. Add chemprop_aux 513-prediction as a feature on test/unblind side.
       For train+uscale: we don't have chemprop_aux preds, so we use a
       proxy "anchor" feature -- mean(y_train)=4.78 -- effectively
       disabling the anchor on the train fold while keeping it on test.
       This is a simple "anchor=mean for train, anchor=chemprop for test"
       calibrated setup; LGBM trees will learn this column.
    4. Train K=20 LGBM directly on pec50 target across 4235 rows.
       5-fold scaffold CV evaluating ONLY on 253 unblind. Mean-bag over
       5 seeds {0,1,7,42,137}. Hyperparams identical to nb2240.
    5. If mean RAE < 0.46 → deep-30 verify (5 + 25 seeds).
    6. Save mean-bag oof on 253 + te on 513 + summary.

GATE: mean_rae < 0.4570 → PROMOTE
      mean_rae < 0.4601 → MARGINAL_BEAT
      else            → FAIL
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
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test, load_semi_pure
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2480"

# ============================================================================
# constants
# ============================================================================

NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_TEST_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")
MORDRED_TRAIN_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_train.npy")
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

USCALE_CSV = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/raw/pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# match nb2240 hyperparams exactly
LGBM_PARAMS_BASE = dict(
    objective="regression",
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    min_child_samples=5,
    reg_lambda=2.0,
    n_jobs=2,
    verbosity=-1,
)
SEEDS_QUICK = [0, 1, 7, 42, 137]
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
GATE_FAIL_FLOOR = 0.4601
GATE_PROMOTE = 0.4570

# ============================================================================
# helpers (copied from nb2240)
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
        .agg(
            pec50=("pec50", "median"),
            std_smiles=("std_smiles", "first"),
            src_first=("src", "first"),
            n_meas=("pec50", "count"),
        )
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


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


# ============================================================================
# featurization for arbitrary SMILES (uscale + train)
# ============================================================================

def _compute_atompair_bits_from_smiles(smiles_list, n_bits=2048):
    """Compute AtomPair fingerprints (matching te_atompair.npy layout)."""
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        fp = gen.GetFingerprint(m)
        bits = np.array(fp, dtype=np.uint8)
        out[i] = bits
    return out


def _compute_maccs_bits(smiles_list):
    from rdkit.Chem import MACCSkeys
    out = np.zeros((len(smiles_list), 167), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(m)
        out[i] = np.array(fp, dtype=np.uint8)
    return out


def _compute_avalon_bits(smiles_list, n_bits=512):
    from rdkit.Avalon import pyAvalonTools
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        fp = pyAvalonTools.GetAvalonFP(m, nBits=n_bits)
        out[i] = np.array(fp, dtype=np.uint8)
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 96-uscale aug, K=20 RFE pyramid, scaffold-CV on 253 unblind")
    print("=" * 78)

    # ---- load 96 uscale + 4139 train + 253 unblind ----
    uscale = load_semi_pure()
    print(f"[uscale] rows={len(uscale)} cols={list(uscale.columns)[:6]}")
    uscale = uscale.dropna(subset=["smiles", "pec50"]).copy()
    # coerce pec50 to float, drop non-numeric (e.g. "-" placeholders)
    uscale["pec50"] = pd.to_numeric(uscale["pec50"], errors="coerce")
    uscale = uscale.dropna(subset=["pec50"]).copy()
    print(f"[uscale] after dropna pec50/smiles (numeric): {len(uscale)}")

    train = load_train()
    train = train.dropna(subset=["smiles", "pec50"]).copy()
    print(f"[train] rows={len(train)}")

    te = load_test()
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    n_test = len(te)
    print(f"[test] rows={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[unblind] n={n_unb} unique_scaffolds={n_unique_scaf}")

    # ---- dedup uscale against test (no leakage) ----
    test_mols = [standardize(s) for s in te_smiles]
    test_iks = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik:
            test_iks.add(ik)
    uscale_mols = [standardize(s) for s in uscale["smiles"].tolist()]
    uscale["inchikey"] = [_safe_inchikey(m) for m in uscale_mols]
    n_uscale_before = len(uscale)
    uscale = uscale[~uscale["inchikey"].isin(test_iks)].reset_index(drop=True)
    n_uscale_after = len(uscale)
    print(f"[uscale] dedup vs test: {n_uscale_before} -> {n_uscale_after}")

    # ---- dedup uscale against train ----
    train_mols = [standardize(s) for s in train["smiles"].tolist()]
    train_iks = set()
    for m in train_mols:
        ik = _safe_inchikey(m)
        if ik:
            train_iks.add(ik)
    n_uscale_b = len(uscale)
    uscale = uscale[~uscale["inchikey"].isin(train_iks)].reset_index(drop=True)
    n_uscale_a = len(uscale)
    print(f"[uscale] dedup vs train: {n_uscale_b} -> {n_uscale_a}")

    # ---- combine train + uscale ----
    train_part = pd.DataFrame({
        "smiles": train["smiles"].astype(str).tolist(),
        "pec50": train["pec50"].astype(float).tolist(),
        "src": ["train"] * len(train),
    })
    uscale_part = pd.DataFrame({
        "smiles": uscale["smiles"].astype(str).tolist(),
        "pec50": uscale["pec50"].astype(float).tolist(),
        "src": ["uscale96"] * len(uscale),
    })
    combo = pd.concat([train_part, uscale_part], ignore_index=True)
    print(f"[combo] train={len(train_part)} + uscale={len(uscale_part)} = {len(combo)}")

    # ---- standardize combo SMILES -> canonical for feature compute ----
    combo_mols = [standardize(s) for s in combo["smiles"].tolist()]
    combo_smi_can = []
    keep_mask = []
    for m in combo_mols:
        sm = _safe_can_smiles(m)
        keep_mask.append(sm is not None)
        combo_smi_can.append(sm if sm else "")
    keep_mask = np.array(keep_mask)
    combo = combo[keep_mask].reset_index(drop=True)
    combo_smi_can = [s for s, k in zip(combo_smi_can, keep_mask) if k]
    print(f"[combo] after standardize-keep: {len(combo)}")

    # ---- feature extraction on combo + unblind (need K=20 columns) ----
    # Load nb2231 K=20 surviving indices into 117-col layout
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20
    print(f"[K20] surviving feat names: {surviving_K20_names[:6]}...")

    # Load summaries for top-idx lookups
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
    full_ap_ranked = _extract_atompair_top_idx(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    # Test side: load cached features and slice
    X_ap_te = np.load(ATOMPAIR_TE_PATH).astype(np.float32)[:, top_ap_bit_idx]
    X_maccs_te = np.load(MACCS_TE_PATH).astype(np.float32)[:, top_maccs_bit_idx]
    X_mord_te = np.load(MORDRED_TEST_PATH).astype(np.float32)
    X_mord_te = np.where(np.isfinite(X_mord_te), X_mord_te, 0.0)
    X_mord_te = X_mord_te[:, top_mord_col_idx]
    X_emb_te = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
    X_emb_te = np.where(np.isfinite(X_emb_te), X_emb_te, 0.0)
    X_emb_te = X_emb_te[:, top_embed_col_idx]
    X_av_te = np.load(AVALON_TE_PATH).astype(np.float32)[:, top_avalon_bit_idx]

    # Train side: load cached features for original 4139
    n_tr = len(train)
    # ChempropEmbed and Mordred train caches
    # We need atompair/maccs/avalon for train -- check cached or compute
    train_smi_list = train["smiles"].astype(str).tolist()
    print(f"[combo-feat] computing fingerprints for {len(combo)} rows...")
    # Compute fingerprints for COMBO (train+uscale) using canonical SMILES
    t_f = time.time()
    X_ap_combo_full = _compute_atompair_bits_from_smiles(combo_smi_can, n_bits=2048)
    X_ap_combo = X_ap_combo_full[:, top_ap_bit_idx].astype(np.float32)
    print(f"[combo-feat] AtomPair: {time.time() - t_f:.1f}s")
    t_f = time.time()
    X_maccs_combo_full = _compute_maccs_bits(combo_smi_can)
    X_maccs_combo = X_maccs_combo_full[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[combo-feat] MACCS: {time.time() - t_f:.1f}s")
    t_f = time.time()
    X_av_combo_full = _compute_avalon_bits(combo_smi_can, n_bits=512)
    X_av_combo = X_av_combo_full[:, top_avalon_bit_idx].astype(np.float32)
    print(f"[combo-feat] Avalon: {time.time() - t_f:.1f}s")

    # Mordred + ChempropEmbed: load train cache, then need uscale-side feats
    # The train cache aligns with the original train order (4139)
    if not MORDRED_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Mordred train cache missing: {MORDRED_TRAIN_PATH}")
    X_mord_train_full = np.load(MORDRED_TRAIN_PATH).astype(np.float32)
    if X_mord_train_full.shape[0] != n_tr:
        raise ValueError(f"Mordred train shape {X_mord_train_full.shape}, expected ({n_tr},*)")
    X_mord_train_full = np.where(np.isfinite(X_mord_train_full), X_mord_train_full, 0.0)
    X_mord_train_sliced = X_mord_train_full[:, top_mord_col_idx]

    # Re-compute Mordred for uscale on the fly
    print(f"[combo-feat] Mordred for {len(combo) - n_tr} uscale rows...")
    # Mordred computation -- use external Mordred (slow ~5min for 96)
    try:
        from mordred import Calculator, descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        n_uscale_rows = len(combo) - n_tr
        uscale_smi_can = combo_smi_can[n_tr:]
        # Mordred returns DataFrame; we need the same column order as nb1030
        t_f = time.time()
        mols_us = [Chem.MolFromSmiles(s) for s in uscale_smi_can]
        df_mord_us = calc.pandas(mols_us, quiet=True)
        # Coerce: same column count as test Mordred
        full_mord_cols = X_mord_te.shape[1] if False else X_mord_train_full.shape[1]
        if df_mord_us.shape[1] != full_mord_cols:
            print(f"[WARN] mordred col mismatch {df_mord_us.shape[1]} vs {full_mord_cols}; using col-trim/pad")
        # Convert to numeric, NaN->0
        arr_us = df_mord_us.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        arr_us = np.where(np.isfinite(arr_us), arr_us, 0.0)
        if arr_us.shape[1] >= full_mord_cols:
            arr_us = arr_us[:, :full_mord_cols]
        else:
            pad = np.zeros((arr_us.shape[0], full_mord_cols - arr_us.shape[1]), dtype=np.float32)
            arr_us = np.concatenate([arr_us, pad], axis=1)
        X_mord_us_sliced = arr_us[:, top_mord_col_idx]
        print(f"[combo-feat] Mordred uscale done in {time.time() - t_f:.1f}s shape={arr_us.shape}")
    except Exception as e:
        print(f"[ERROR] Mordred computation failed: {e}")
        # Fall back to zeros
        X_mord_us_sliced = np.zeros((len(combo) - n_tr, len(top_mord_col_idx)), dtype=np.float32)
    X_mord_combo = np.concatenate([X_mord_train_sliced, X_mord_us_sliced], axis=0)

    # ChempropEmbed: load train cache if exists, else zeros for uscale (we can't run chemprop here)
    embed_train_path = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
    embed_train_alt = DATA_PROCESSED / "chemprop_embed_300_train.npy"
    use_zero_embed = False
    if embed_train_path.exists():
        X_emb_train_full = np.load(embed_train_path).astype(np.float32)
    elif embed_train_alt.exists():
        X_emb_train_full = np.load(embed_train_alt).astype(np.float32)
    else:
        print(f"[WARN] chemprop_embed_train cache missing; using zeros (will hurt baseline parity)")
        X_emb_train_full = np.zeros((n_tr, 300), dtype=np.float32)
        use_zero_embed = True
    if X_emb_train_full.shape[0] != n_tr:
        print(f"[WARN] chemprop_embed_train shape {X_emb_train_full.shape}, expected ({n_tr},300); zeroing")
        X_emb_train_full = np.zeros((n_tr, 300), dtype=np.float32)
        use_zero_embed = True
    X_emb_train_sliced = np.where(np.isfinite(X_emb_train_full), X_emb_train_full, 0.0)[:, top_embed_col_idx]
    # For uscale rows: use ZERO embed (chemprop not runnable on the fly)
    X_emb_us_sliced = np.zeros((len(combo) - n_tr, len(top_embed_col_idx)), dtype=np.float32)
    X_emb_combo = np.concatenate([X_emb_train_sliced, X_emb_us_sliced], axis=0)

    # ChEMBL kNN for combo + test
    pool = _load_chembl_pool()
    # Drop test compounds from pool (already in main fn)
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))

    # test side knn
    fp_test = morgan_fp_batch([_safe_can_smiles(m) or "" for m in test_mols])
    top_idx_te, top_sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(top_idx_te, top_sim_te, pool_labels, fallback=pool_median)

    # combo side knn
    fp_combo = morgan_fp_batch(combo_smi_can)
    top_idx_combo, top_sim_combo = _tanimoto_topk(fp_combo, fp_pool, k=KNN_K)
    pred_chembl_combo, mean_sim_combo = _knn_predict(top_idx_combo, top_sim_combo, pool_labels, fallback=pool_median)

    # ---- Assemble 117-col matrices (test/unb and combo) in nb2240 order ----
    X_te_full = np.concatenate(
        [
            X_ap_te,
            X_maccs_te,
            X_mord_te,
            X_emb_te,
            X_av_te,
            pred_chembl_te.reshape(-1, 1).astype(np.float32),
            mean_sim_te.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"X_te_full.shape={X_te_full.shape}"

    X_combo_full = np.concatenate(
        [
            X_ap_combo,
            X_maccs_combo,
            X_mord_combo,
            X_emb_combo,
            X_av_combo,
            pred_chembl_combo.reshape(-1, 1).astype(np.float32),
            mean_sim_combo.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_combo_full.shape[1] != 117:
        print(f"[WARN] X_combo_full has {X_combo_full.shape[1]} cols, expected 117 -- check feat counts")

    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    X_combo_K20 = X_combo_full[:, surviving_K20].astype(np.float32)
    print(f"[K20] X_combo_K20={X_combo_K20.shape}  X_te_K20={X_te_K20.shape}  X_unb_K20={X_unb_K20.shape}")

    # ---- Add chemprop_aux anchor as extra feature (column 20) ----
    te_anchor = np.load(ANCHOR_TE_PATH).astype(np.float64)  # (513,)
    # For combo: use train mean as proxy (anchor is not available)
    combo_anchor = np.full(len(combo), float(np.mean(combo["pec50"])), dtype=np.float32)
    X_combo_K20_plus = np.concatenate([X_combo_K20, combo_anchor.reshape(-1, 1)], axis=1)
    X_te_K20_plus = np.concatenate([X_te_K20, te_anchor.reshape(-1, 1).astype(np.float32)], axis=1)
    X_unb_K20_plus = X_te_K20_plus[unb_idx]
    print(f"[K20+] X_combo_K20_plus={X_combo_K20_plus.shape} X_te_K20_plus={X_te_K20_plus.shape}")

    # ---- LGBM training: full-train fit, evaluate on unblind ----
    y_combo = combo["pec50"].to_numpy(dtype=np.float64)
    print(f"\n[train] combo target stats: mean={y_combo.mean():.3f} std={y_combo.std():.3f}")
    print(f"[unb ] y stats: mean={y_unb.mean():.3f} std={y_unb.std():.3f}")

    # ============================================================================
    # Stage A: Quick 5-seed scaffold-CV-style eval on 253 unblind
    # The CV is over 253 unblind scaffolds; each fold trains LGBM on
    # (all combo + 4/5 unblind held-in) and predicts the 1/5 unblind held-out.
    # ============================================================================
    print("\n" + "-" * 78)
    print(f"STAGE A: 5-fold scaffold CV (eval on 253 unblind)")
    print("-" * 78)
    # Also use unblind labels in training (their TRUE pec50, with scaffold-fold held out)
    y_unb_arr = y_unb.astype(np.float64)

    def cv_eval_for_seed(seed):
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed)
        oof_unb = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            # train: combo + held-in unblind
            X_tr = np.concatenate([X_combo_K20_plus, X_unb_K20_plus[tr_loc]], axis=0)
            y_tr = np.concatenate([y_combo, y_unb_arr[tr_loc]], axis=0)
            params = dict(LGBM_PARAMS_BASE)
            params["random_state"] = seed
            mdl = lgb.LGBMRegressor(**params)
            mdl.fit(X_tr, y_tr)
            oof_unb[va_loc] = mdl.predict(X_unb_K20_plus[va_loc])
        r = float(rae(y_unb_arr, oof_unb))
        return r, oof_unb

    per_seed = []
    oof_stack = []
    for s in SEEDS_QUICK:
        ts = time.time()
        r, oof = cv_eval_for_seed(s)
        per_seed.append({"seed": int(s), "rae": float(r)})
        oof_stack.append(oof)
        print(f"   seed={s:3d}: rae={r:.4f}  wall={time.time() - ts:.1f}s")
    pooled_5seed_mean = float(np.mean([d["rae"] for d in per_seed]))
    pooled_5seed_std = float(np.std([d["rae"] for d in per_seed]))
    mean_bag_oof = np.mean(np.column_stack(oof_stack), axis=1).astype(np.float64)
    mean_bag_rae = float(rae(y_unb_arr, mean_bag_oof))
    print(f"\n[5seed] per-seed mean RAE = {pooled_5seed_mean:.4f} +/- {pooled_5seed_std:.4f}")
    print(f"[5seed] mean-bag RAE      = {mean_bag_rae:.4f}")

    # ---- Verdict ----
    if pooled_5seed_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_5seed_mean < GATE_FAIL_FLOOR:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] verdict (vs floor {GATE_FAIL_FLOOR}, promote {GATE_PROMOTE}) = {verdict}")

    # ---- Stage B: deep-30 only if promising ----
    deep30 = None
    if pooled_5seed_mean < 0.46:
        print("\n" + "-" * 78)
        print(f"STAGE B: DEEP-30 verify (5-seed mean {pooled_5seed_mean:.4f} < 0.46)")
        print("-" * 78)
        extra_seeds = [1006 + i for i in range(25)]
        all_seeds = KF_SEEDS + extra_seeds
        per30 = []
        for s in all_seeds:
            ts = time.time()
            r, _ = cv_eval_for_seed(s)
            per30.append({"seed": int(s), "rae": float(r)})
            print(f"   deep-seed={s:4d}: rae={r:.4f}  wall={time.time() - ts:.1f}s")
        raes = np.array([d["rae"] for d in per30])
        deep30 = {
            "n_seeds": int(len(per30)),
            "per_seed": per30,
            "mean_rae": float(raes.mean()),
            "std_rae": float(raes.std()),
            "min_rae": float(raes.min()),
            "max_rae": float(raes.max()),
        }
        print(f"\n[deep30] mean={deep30['mean_rae']:.4f} +/- {deep30['std_rae']:.4f}  "
              f"range=[{deep30['min_rae']:.4f}, {deep30['max_rae']:.4f}]")
    else:
        print(f"[skip] 5-seed mean {pooled_5seed_mean:.4f} >= 0.46; skip deep-30")

    # ---- Deploy refit: train on combo + ALL 253 unblind, predict 513 ----
    print("\n[deploy] full refit on combo + 253 unblind...")
    X_full = np.concatenate([X_combo_K20_plus, X_unb_K20_plus], axis=0)
    y_full = np.concatenate([y_combo, y_unb_arr], axis=0)
    te_preds = np.zeros((len(SEEDS_QUICK), n_test), dtype=np.float64)
    for i, s in enumerate(SEEDS_QUICK):
        params = dict(LGBM_PARAMS_BASE)
        params["random_state"] = s
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_full, y_full)
        te_preds[i] = mdl.predict(X_te_K20_plus)
    te_mean_bag = te_preds.mean(axis=0).astype(np.float32)
    te_rae_unb_insample = float(rae(y_unb_arr, te_mean_bag[unb_idx]))
    print(f"[deploy] te_mean_bag mean={te_mean_bag.mean():.3f} std={te_mean_bag.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_rae_unb_insample:.4f}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_mean_bag)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'te_{TAG}.npy'}")

    summary = {
        "tag": TAG,
        "method": "uscale96_train_aug_K20_rfe_lgbm_direct_pec50",
        "n_train_orig": int(len(train)),
        "n_uscale_kept": int(len(uscale)),
        "n_combo": int(len(combo)),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "lgbm_params": LGBM_PARAMS_BASE,
        "seeds_quick": SEEDS_QUICK,
        "kf_seeds_5fold": KF_SEEDS,
        "n_folds": N_FOLDS,
        "feat_dim_combo": int(X_combo_K20_plus.shape[1]),
        "feat_dim_test": int(X_te_K20_plus.shape[1]),
        "K20_surviving_names": surviving_K20_names,
        "anchor_feature_used": "chemprop_aux_513_for_test;train_mean_for_combo",
        "embed_train_was_zero": bool(use_zero_embed),
        "per_seed_5seed": per_seed,
        "pooled_5seed_mean_rae": pooled_5seed_mean,
        "pooled_5seed_std_rae": pooled_5seed_std,
        "mean_bag_rae_5seed": mean_bag_rae,
        "deep30": deep30,
        "te_unb_in_sample_rae": te_rae_unb_insample,
        "te_mean": float(te_mean_bag.mean()),
        "te_std": float(te_mean_bag.std()),
        "gate_floor_0.4601": GATE_FAIL_FLOOR,
        "gate_promote_0.4570": GATE_PROMOTE,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   n_train={len(train)}  n_uscale={len(uscale)}  n_combo={len(combo)}")
    print(f"   pooled 5-seed RAE = {pooled_5seed_mean:.4f} +/- {pooled_5seed_std:.4f}")
    print(f"   mean-bag RAE      = {mean_bag_rae:.4f}")
    if deep30:
        print(f"   deep-30 mean RAE  = {deep30['mean_rae']:.4f} +/- {deep30['std_rae']:.4f}")
    print(f"   te[unb] in-RAE    = {te_rae_unb_insample:.4f}")
    print(f"   verdict           = {verdict}")
    print(f"   wall              = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_uscale_kept",
        "n_combo",
        "pooled_5seed_mean_rae",
        "pooled_5seed_std_rae",
        "mean_bag_rae_5seed",
        "te_unb_in_sample_rae",
        "verdict",
        "embed_train_was_zero",
    ):
        print(f"  {k}: {res.get(k)}")
