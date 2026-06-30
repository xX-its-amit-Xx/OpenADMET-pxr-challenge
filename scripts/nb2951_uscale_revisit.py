"""nb2951 -- uscale-96 revisit: augment nb2240+equal_K substrate, NOT standalone K=20.

NEW PARADIGM:
    Cycle 198 nb2480 failed (5-seed RAE 0.6184) because it tried a STANDALONE
    K=20 LGBM trained on the (4139+96 uscale) augmented data with a stripped
    "anchor=train_mean for combo, anchor=chemprop_aux for test" hack -- which
    breaks the anchor-axis prior the rest of the pyramid depends on.

    This time, replicate the nb2943 BEST combo (0.5*nb2240_K20 + 0.5*equal_K)
    but retrain BOTH legs of that blend on the uscale-augmented data.
    Specifically:
      * nb2240_aug   = K=20 LGBM trained on 4139+96 = 4235  (+ unblind train-fold)
      * equal_K_aug  = mean(K18, K24, K28) where each K_x is a LGBM trained on
                       4139+96 = 4235 (+ unblind train-fold) over its respective
                       117-col K-subset.
      * blend        = 0.5*nb2240_aug + 0.5*equal_K_aug   (the nb2943 best combo)

PROTOCOL:
    1. Re-use nb2480's feature pipeline: load 4139 train + 96 uscale (88 after
       dedup vs train/test), compute 117-col 5-way matrix on (4235 + 513).
    2. Slice into K=18, K=20, K=24, K=28 subsets using cached idx-in-117 maps:
         K18  -> nb2604_summary["k18_idx_in_117col"]   (18 cols)
         K20  -> nb2231_summary["snapshots"]["20"]      (20 cols)  == nb2240 anchor
         K24  -> nb2310_summary["k20_surviving_idx_in_117"]  (20 cols, but
                  this is actually the K24 idx -- see nb2310 / nb2943)
         K28  -> nb2103_summary["per_K_records"][K=28]  (28 cols)
    3. Add chemprop_aux ANCHOR feature column ON TEST/UNBLIND ONLY:
         For combo (4235 rows): anchor_combo = mean(y_combo)
         For test/unb (513 rows): anchor = te_chemprop_aux.npy
       Identical to nb2480 — keeps anchor axis live on inference side.
    4. For each K in {18, 20, 24, 28}:
         5-fold scaffold CV on 253 unblind (kf_seed=1001) using nb2240
         LGBM hyperparams; mean-bag over 5 seeds {0,1,7,42,137}; produce
         oof_unb_K (253,) + te_K (513,).
    5. Blend (df=0, no learning):
         oof_blend = 0.5*oof_K20 + 0.5*((oof_K18 + oof_K24 + oof_K28)/3)
         te_blend  = 0.5*te_K20  + 0.5*((te_K18  + te_K24  + te_K28 )/3)
    6. Report pooled RAE on 253 unblind; save artifacts.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4576 -> BETTER_THAN_NB2943   (nb2943 best pooled = 0.4576)
    else              -> FAIL

References:
    nb2943 fine-ratio grid (no aug)   = 0.4576
    nb2934 5-anchor equal (no aug)    = 0.4580
    nb2171 PRIMARY-1 ceiling          = 0.4682
    nb2480 standalone K=20 aug FAIL   = 0.6184
    chemprop_aux                      = 0.6216

Outputs:
    scripts/nb2951_uscale_revisit.py
    data/processed/nb2951_summary.json
    data/processed/nb2951_pred_oof.npy   (253,) float32  [blend]
    data/processed/te_nb2951.npy         (513,) float32  [blend]
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test, load_semi_pure
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2951"

# --- summary lookups for K-subsets (shared 117-col layout) ---
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"   # K18 idx
NB2310_SUMMARY = DATA_PROCESSED / "nb2310_summary.json"   # K24 idx
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"   # K28 idx

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_TEST_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")
MORDRED_TRAIN_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_train.npy")
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# nb2240 hyperparams
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
KF_SEED = 1001
K_LIST = [18, 20, 24, 28]

GATE_PROMOTE = 0.4570
GATE_BETTER_NB2943 = 0.4576
NB2943_REF = 0.4576
NB2934_REF = 0.4580
NB2171_REF = 0.4682
NB2480_FAIL_REF = 0.6184
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


def _compute_atompair_bits_from_smiles(smiles_list, n_bits=2048):
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


# K-subset extractors from cached summaries
def _load_K_idx_in_117():
    """Returns {K -> idx_array (1d int)} in the shared 117-col layout."""
    out = {}
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    out[20] = np.array(nb2231["snapshots"]["20"]["surviving_idx_in_117"], dtype=int)

    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    out[18] = np.array(nb2604["k18_idx_in_117col"], dtype=int)

    with open(NB2310_SUMMARY) as f:
        nb2310 = json.load(f)
    # nb2310 mis-named the key; per its filename ("K24") this IS the K24 idx
    out[24] = np.array(nb2310["k20_surviving_idx_in_117"], dtype=int)

    with open(NB2103_SUMMARY) as f:
        nb2103 = json.load(f)
    rec_28 = next(r for r in nb2103["per_K_records"] if int(r["K"]) == 28)
    out[28] = np.array(rec_28["top_K_idx_in_117"], dtype=int)

    return out


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- uscale-96 revisit on nb2240+equal_K substrate "
          f"(NOT standalone K=20 like nb2480)")
    print(f"          blend = 0.5*nb2240_K20_aug + 0.5*equal_K_aug "
          f"(nb2943 best combo)")
    print(f"          equal_K_aug = mean(K18_aug, K24_aug, K28_aug)")
    print("=" * 78)

    # ---- load uscale + train + test + unblind ----
    uscale = load_semi_pure()
    uscale = uscale.dropna(subset=["smiles", "pec50"]).copy()
    uscale["pec50"] = pd.to_numeric(uscale["pec50"], errors="coerce")
    uscale = uscale.dropna(subset=["pec50"]).copy()
    print(f"[uscale] after numeric dropna: {len(uscale)}")

    train = load_train()
    train = train.dropna(subset=["smiles", "pec50"]).copy()
    print(f"[train] rows={len(train)}")

    te = load_test()
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    n_test = len(te)
    print(f"[test] rows={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[unblind] n={n_unb} unique_scaffolds={n_unique_scaf}")

    # ---- dedup uscale against test + train ----
    test_mols = [standardize(s) for s in te_smiles]
    test_iks = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik:
            test_iks.add(ik)
    uscale_mols = [standardize(s) for s in uscale["smiles"].tolist()]
    uscale["inchikey"] = [_safe_inchikey(m) for m in uscale_mols]
    n0 = len(uscale)
    uscale = uscale[~uscale["inchikey"].isin(test_iks)].reset_index(drop=True)
    print(f"[uscale] dedup vs test: {n0} -> {len(uscale)}")

    train_mols = [standardize(s) for s in train["smiles"].tolist()]
    train_iks = set()
    for m in train_mols:
        ik = _safe_inchikey(m)
        if ik:
            train_iks.add(ik)
    n1 = len(uscale)
    uscale = uscale[~uscale["inchikey"].isin(train_iks)].reset_index(drop=True)
    print(f"[uscale] dedup vs train: {n1} -> {len(uscale)}")

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

    n_tr = len(train)

    # ---- load K-subset indices ----
    K_idx = _load_K_idx_in_117()
    for k_val in K_LIST:
        print(f"  K={k_val}: {len(K_idx[k_val])} cols  idx[:6]={K_idx[k_val][:6]}")

    # ---- summaries for top-idx lookups (117-col build) ----
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

    # ---- test side: load cached features, slice to top-K per family ----
    X_ap_te = np.load(ATOMPAIR_TE_PATH).astype(np.float32)[:, top_ap_bit_idx]
    X_maccs_te = np.load(MACCS_TE_PATH).astype(np.float32)[:, top_maccs_bit_idx]
    X_mord_te = np.load(MORDRED_TEST_PATH).astype(np.float32)
    X_mord_te = np.where(np.isfinite(X_mord_te), X_mord_te, 0.0)[:, top_mord_col_idx]
    X_emb_te = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
    X_emb_te = np.where(np.isfinite(X_emb_te), X_emb_te, 0.0)[:, top_embed_col_idx]
    X_av_te = np.load(AVALON_TE_PATH).astype(np.float32)[:, top_avalon_bit_idx]

    # ---- combo side: compute fingerprints for 4235 rows ----
    print(f"\n[combo-feat] computing fingerprints for {len(combo)} rows...")
    t_f = time.time()
    X_ap_combo = _compute_atompair_bits_from_smiles(combo_smi_can, n_bits=2048)
    X_ap_combo = X_ap_combo[:, top_ap_bit_idx].astype(np.float32)
    print(f"[combo-feat] AtomPair: {time.time() - t_f:.1f}s")

    t_f = time.time()
    X_maccs_combo = _compute_maccs_bits(combo_smi_can)
    X_maccs_combo = X_maccs_combo[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[combo-feat] MACCS: {time.time() - t_f:.1f}s")

    t_f = time.time()
    X_av_combo = _compute_avalon_bits(combo_smi_can, n_bits=512)
    X_av_combo = X_av_combo[:, top_avalon_bit_idx].astype(np.float32)
    print(f"[combo-feat] Avalon: {time.time() - t_f:.1f}s")

    # ---- Mordred: train cache + on-the-fly for uscale ----
    if not MORDRED_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Mordred train cache missing: {MORDRED_TRAIN_PATH}")
    X_mord_train_full = np.load(MORDRED_TRAIN_PATH).astype(np.float32)
    X_mord_train_full = np.where(np.isfinite(X_mord_train_full), X_mord_train_full, 0.0)
    if X_mord_train_full.shape[0] != n_tr:
        raise ValueError(
            f"Mordred train shape {X_mord_train_full.shape}, expected ({n_tr},*)"
        )
    full_mord_cols = X_mord_train_full.shape[1]
    X_mord_train_sliced = X_mord_train_full[:, top_mord_col_idx]

    n_uscale_rows = len(combo) - n_tr
    if n_uscale_rows > 0:
        print(f"[combo-feat] Mordred for {n_uscale_rows} uscale rows...")
        try:
            from mordred import Calculator, descriptors
            calc = Calculator(descriptors, ignore_3D=True)
            uscale_smi_can = combo_smi_can[n_tr:]
            t_f = time.time()
            mols_us = [Chem.MolFromSmiles(s) for s in uscale_smi_can]
            df_mord_us = calc.pandas(mols_us, quiet=True)
            arr_us = df_mord_us.apply(pd.to_numeric, errors="coerce").to_numpy(
                dtype=np.float32
            )
            arr_us = np.where(np.isfinite(arr_us), arr_us, 0.0)
            if arr_us.shape[1] >= full_mord_cols:
                arr_us = arr_us[:, :full_mord_cols]
            else:
                pad = np.zeros(
                    (arr_us.shape[0], full_mord_cols - arr_us.shape[1]),
                    dtype=np.float32,
                )
                arr_us = np.concatenate([arr_us, pad], axis=1)
            X_mord_us_sliced = arr_us[:, top_mord_col_idx]
            print(f"[combo-feat] Mordred uscale done in {time.time() - t_f:.1f}s")
        except Exception as e:
            print(f"[ERROR] Mordred uscale failed: {e}; using zeros")
            X_mord_us_sliced = np.zeros(
                (n_uscale_rows, len(top_mord_col_idx)), dtype=np.float32
            )
    else:
        X_mord_us_sliced = np.zeros((0, len(top_mord_col_idx)), dtype=np.float32)
    X_mord_combo = np.concatenate([X_mord_train_sliced, X_mord_us_sliced], axis=0)

    # ---- ChempropEmbed: train cache (uscale gets zeros, chemprop not runnable here) ----
    embed_train_path = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
    embed_train_alt = DATA_PROCESSED / "chemprop_embed_300_train.npy"
    use_zero_embed = False
    if embed_train_path.exists():
        X_emb_train_full = np.load(embed_train_path).astype(np.float32)
    elif embed_train_alt.exists():
        X_emb_train_full = np.load(embed_train_alt).astype(np.float32)
    else:
        print(f"[WARN] chemprop_embed_train cache missing; using zeros")
        X_emb_train_full = np.zeros((n_tr, 300), dtype=np.float32)
        use_zero_embed = True
    if X_emb_train_full.shape[0] != n_tr:
        print(f"[WARN] chemprop_embed_train shape mismatch; zeroing")
        X_emb_train_full = np.zeros((n_tr, 300), dtype=np.float32)
        use_zero_embed = True
    X_emb_train_full = np.where(np.isfinite(X_emb_train_full), X_emb_train_full, 0.0)
    X_emb_train_sliced = X_emb_train_full[:, top_embed_col_idx]
    X_emb_us_sliced = np.zeros(
        (n_uscale_rows, len(top_embed_col_idx)), dtype=np.float32
    )
    X_emb_combo = np.concatenate([X_emb_train_sliced, X_emb_us_sliced], axis=0)

    # ---- ChEMBL kNN: test + combo ----
    pool = _load_chembl_pool()
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))

    fp_test = morgan_fp_batch([_safe_can_smiles(m) or "" for m in test_mols])
    top_idx_te, top_sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_te, top_sim_te, pool_labels, fallback=pool_median
    )

    fp_combo = morgan_fp_batch(combo_smi_can)
    top_idx_combo, top_sim_combo = _tanimoto_topk(fp_combo, fp_pool, k=KNN_K)
    pred_chembl_combo, mean_sim_combo = _knn_predict(
        top_idx_combo, top_sim_combo, pool_labels, fallback=pool_median
    )

    # ---- assemble 117-col matrices ----
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
    if X_te_full.shape[1] != 117:
        raise ValueError(f"X_te_full has {X_te_full.shape[1]} cols, expected 117")

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
        print(f"[WARN] X_combo_full has {X_combo_full.shape[1]} cols")

    # ---- anchor column (chemprop_aux on test/unb, train mean on combo) ----
    te_anchor = np.load(ANCHOR_TE_PATH).astype(np.float64)  # (513,)
    combo_anchor = np.full(
        len(combo), float(np.mean(combo["pec50"])), dtype=np.float32
    )

    y_combo = combo["pec50"].to_numpy(dtype=np.float64)
    y_unb_arr = y_unb.astype(np.float64)

    # ---- build per-K matrices (combo, te, unb) with +1 anchor column ----
    per_K_data = {}
    for k_val in K_LIST:
        idx_k = K_idx[k_val]
        X_te_k = X_te_full[:, idx_k]
        X_combo_k = X_combo_full[:, idx_k]
        X_te_kp = np.concatenate(
            [X_te_k, te_anchor.reshape(-1, 1).astype(np.float32)], axis=1
        )
        X_combo_kp = np.concatenate(
            [X_combo_k, combo_anchor.reshape(-1, 1)], axis=1
        )
        X_unb_kp = X_te_kp[unb_idx]
        per_K_data[k_val] = (X_combo_kp, X_te_kp, X_unb_kp)
        print(
            f"[K={k_val}+anchor] combo={X_combo_kp.shape}  te={X_te_kp.shape}  "
            f"unb={X_unb_kp.shape}"
        )

    # ---- 5-fold scaffold CV (kf_seed=1001, single seed per task spec) ----
    print("\n" + "-" * 78)
    print(f"5-fold scaffold CV  kf_seed={KF_SEED}  seeds_lgbm={SEEDS_QUICK}")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED
    )

    def train_K(k_val):
        X_combo_kp, X_te_kp, X_unb_kp = per_K_data[k_val]
        oof_per_seed = np.zeros((len(SEEDS_QUICK), n_unb), dtype=np.float64)
        te_per_seed = np.zeros((len(SEEDS_QUICK), n_test), dtype=np.float64)
        for si, s in enumerate(SEEDS_QUICK):
            params = dict(LGBM_PARAMS_BASE)
            params["random_state"] = s

            # cross-fit OOF on unblind (combo + held-in unblind -> predict held-out)
            oof = np.full(n_unb, np.nan, dtype=np.float64)
            for tr_loc, va_loc in splits:
                X_tr = np.concatenate(
                    [X_combo_kp, X_unb_kp[tr_loc]], axis=0
                )
                y_tr = np.concatenate([y_combo, y_unb_arr[tr_loc]], axis=0)
                mdl = lgb.LGBMRegressor(**params)
                mdl.fit(X_tr, y_tr)
                oof[va_loc] = mdl.predict(X_unb_kp[va_loc])
            oof_per_seed[si] = oof

            # deploy refit (combo + all unblind -> predict full 513)
            X_full = np.concatenate([X_combo_kp, X_unb_kp], axis=0)
            y_full = np.concatenate([y_combo, y_unb_arr], axis=0)
            mdl_full = lgb.LGBMRegressor(**params)
            mdl_full.fit(X_full, y_full)
            te_per_seed[si] = mdl_full.predict(X_te_kp)
        oof_mean = oof_per_seed.mean(axis=0)
        te_mean = te_per_seed.mean(axis=0)
        return oof_mean, te_mean, oof_per_seed, te_per_seed

    per_K_results = {}
    for k_val in K_LIST:
        ts = time.time()
        oof_k, te_k, _, _ = train_K(k_val)
        r = float(rae(y_unb_arr, oof_k))
        per_K_results[k_val] = {
            "oof": oof_k,
            "te": te_k,
            "rae_oof": r,
        }
        print(
            f"   K={k_val:2d}_aug  oof_RAE={r:.4f}  te_mean={te_k.mean():.3f}  "
            f"te_std={te_k.std():.3f}  wall={time.time() - ts:.1f}s"
        )

    # ---- build blends ----
    oof_K20 = per_K_results[20]["oof"]
    oof_K18 = per_K_results[18]["oof"]
    oof_K24 = per_K_results[24]["oof"]
    oof_K28 = per_K_results[28]["oof"]
    te_K20 = per_K_results[20]["te"]
    te_K18 = per_K_results[18]["te"]
    te_K24 = per_K_results[24]["te"]
    te_K28 = per_K_results[28]["te"]

    equal_K_oof = (oof_K18 + oof_K24 + oof_K28) / 3.0
    equal_K_te = (te_K18 + te_K24 + te_K28) / 3.0
    equal_K_rae = float(rae(y_unb_arr, equal_K_oof))

    # blend = nb2943 best combo
    w_K20 = 0.5
    w_eqK = 0.5
    blend_oof = w_K20 * oof_K20 + w_eqK * equal_K_oof
    blend_te = w_K20 * te_K20 + w_eqK * equal_K_te
    pooled_rae = float(rae(y_unb_arr, blend_oof))

    per_fold_rae = []
    for tr_loc, va_loc in splits:
        per_fold_rae.append(float(rae(y_unb_arr[va_loc], blend_oof[va_loc])))
    fold_mean = float(np.mean(per_fold_rae))
    fold_std = float(np.std(per_fold_rae))

    print("\n" + "-" * 78)
    print("Blend results")
    print("-" * 78)
    print(f"  nb2240_K20_aug  oof_RAE = {per_K_results[20]['rae_oof']:.4f}")
    print(f"  equal_K_aug     oof_RAE = {equal_K_rae:.4f}")
    print(
        f"  blend (0.5*K20 + 0.5*equal_K)  oof_RAE = {pooled_rae:.4f}  "
        f"fold_mean={fold_mean:.4f} +/- {fold_std:.4f}"
    )

    # ---- gate ----
    if pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae < GATE_BETTER_NB2943:
        verdict = "BETTER_THAN_NB2943"
    else:
        verdict = "FAIL"
    print(
        f"[gate] pooled={pooled_rae:.4f}  (<{GATE_PROMOTE} PROMOTE / "
        f"<{GATE_BETTER_NB2943} BETTER_THAN_NB2943) -> {verdict}"
    )

    delta_vs_nb2943 = pooled_rae - NB2943_REF
    delta_vs_nb2171 = pooled_rae - NB2171_REF
    delta_vs_nb2480 = pooled_rae - NB2480_FAIL_REF
    print(f"  delta vs nb2943 ({NB2943_REF}) = {delta_vs_nb2943:+.4f}")
    print(f"  delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")
    print(f"  delta vs nb2480 ({NB2480_FAIL_REF}) = {delta_vs_nb2480:+.4f}")

    # ---- save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, blend_oof.astype(np.float32))
    np.save(te_path, blend_te.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    te_unb_in = float(rae(y_unb_arr, blend_te[unb_idx]))

    summary = {
        "tag": TAG,
        "method": "uscale96_aug_substrate_blend_nb2240K20_equalK",
        "paradigm": "uscale_aug_substrate_change_then_2way_scalar_blend",
        "anchors_post_aug": {
            "nb2240_K20_aug": "K=20 LGBM trained on 4139+96 uscale + heldin unblind",
            "equal_K_aug": "mean(K18_aug, K24_aug, K28_aug), each retrained on aug data",
        },
        "blend_weights": {"w_nb2240_K20_aug": w_K20, "w_equal_K_aug": w_eqK},
        "blend_source": "nb2943 best combo (0.5 / 0.5)",
        "n_train_orig": int(len(train)),
        "n_uscale_kept": int(len(uscale)),
        "n_combo": int(len(combo)),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "lgbm_params": LGBM_PARAMS_BASE,
        "seeds_lgbm": SEEDS_QUICK,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "K_list": K_LIST,
        "per_K_oof_rae_unb": {
            str(k): float(per_K_results[k]["rae_oof"]) for k in K_LIST
        },
        "equal_K_oof_rae": equal_K_rae,
        "blend_pooled_rae": pooled_rae,
        "mean_rae": pooled_rae,
        "blend_per_fold_rae": per_fold_rae,
        "blend_fold_mean": fold_mean,
        "blend_fold_std": fold_std,
        "te_mean": float(blend_te.mean()),
        "te_std": float(blend_te.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "gate_promote": GATE_PROMOTE,
        "gate_better_nb2943": GATE_BETTER_NB2943,
        "verdict": verdict,
        "delta_vs_nb2943": delta_vs_nb2943,
        "delta_vs_nb2171": delta_vs_nb2171,
        "delta_vs_nb2480_fail": delta_vs_nb2480,
        "nb2943_ref": NB2943_REF,
        "nb2934_ref": NB2934_REF,
        "nb2171_ref": NB2171_REF,
        "nb2480_fail_ref": NB2480_FAIL_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "embed_train_was_zero": bool(use_zero_embed),
        "anchor_feature_used": "chemprop_aux_513_for_test;train_mean_for_combo",
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    for k in K_LIST:
        print(f"   K={k:2d}_aug oof_RAE = {per_K_results[k]['rae_oof']:.4f}")
    print(f"   equal_K_aug oof_RAE  = {equal_K_rae:.4f}")
    print(f"   blend oof_RAE        = {pooled_rae:.4f}")
    print(f"   blend per-fold       = {per_fold_rae}")
    print(f"   verdict              = {verdict}")
    print(f"   delta nb2943         = {delta_vs_nb2943:+.4f}")
    print(f"   delta nb2480 (fail)  = {delta_vs_nb2480:+.4f}")
    print(f"   te[unb] in-RAE       = {te_unb_in:.4f}")
    print(f"   wall                 = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_uscale_kept",
        "n_combo",
        "per_K_oof_rae_unb",
        "equal_K_oof_rae",
        "blend_pooled_rae",
        "verdict",
        "delta_vs_nb2943",
        "delta_vs_nb2480_fail",
    ):
        print(f"  {k}: {res.get(k)}")
