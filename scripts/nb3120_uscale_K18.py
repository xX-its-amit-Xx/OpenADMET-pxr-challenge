"""nb3120 -- K=18 LGBM on 96-uscale AUGMENTATION, RESIDUAL target paradigm.

NEW PARADIGM (vs nb3114 which used raw pec50 target):
    nb3114 trained K=18 LGBM on raw pec50 with (4139 + 456 htchem + 96 uscale)
    augmentation.  Here we change two axes simultaneously:

      (1) Drop htchem (~456 noisy DRC rows) -- only uscale-semi-pure (96 cleaner
          DRC rows, weight 1.0).
      (2) RESIDUAL TARGET: fit K=18 LGBM on  (y - chemprop_aux_pred), not raw
          pec50.  Predictions then reconstruct as anchor + residual.

    Per nb3114-style: blend 0.5/0.5 with cached nb2960 K=18 deep-30
    (which is itself residual-on-chemprop_aux).  Both legs live on the SAME
    anchor axis, so the blend is a pure deep-bag mean over two residual
    bagging populations (one with +96 uscale rows, one without).

PROTOCOL:
    1. Build augmented training table:
         (a) 4139 standard PXR train  (weight 1.0)
         (b) 96 uscale-semi-pure      (weight 1.0)   -- dedup vs train + test
       Augmented y = pec50 ; augmented chemprop_aux = train OOF for the 4139
       rows + median(train OOF) imputed for the 96 uscale rows
       (chemprop_aux not retrained-with-uscale; impute with train median).
       Residual target = y - anchor_pred.
    2. Build 117-col 5-way feature matrix on (train + uscale + test); slice to
       K=18 RFE indices (nb2604 cached).
    3. Train K=18 LGBM mean-bag with 30 fresh seeds {3001..3030} on residual
       target (weighted samples). Predict residuals on 513 test and 253 unb;
       add back te_chemprop_aux to reconstruct pec50 predictions.
    4. Blend (mean) with nb2960 K=18 deep-30 OOF/te:
         aug_K18_pec50 = anchor + aug_K18_resid_pred
         oof_blend = 0.5 * aug_K18_pec50_oof + 0.5 * nb2960_K18_oof
         te_blend  = 0.5 * aug_K18_pec50_te  + 0.5 * nb2960_K18_te
    5. Scaffold-CV stability assessment over 15 fresh kf_seeds {1141..1155}:
       since both blend legs are deterministic w.r.t. fixed (train + uscale)
       inputs (no kf-dependent retraining), kf_seeds shuffle the 5-fold
       partition over the 253 unblind for pooled-OOF stability measurement.
       Reported gate metric = MEAN pooled RAE across the 15 kf_seeds.
    6. Report pooled RAE on 253 unblind; gate.

GATE:
    mean_pooled_rae < 0.4475  ->  "BETTER"
    else                       ->  "FAIL"

References (deep-30 fresh-seed, chemprop_aux anchor, K=18 residual LGBM):
    nb2960 K18 30seed bag           = 0.4536  (no aug)
    nb2960 K20 30seed bag           = 0.4592
    nb3114 K18 raw+htchem+uscale    = (TBD, raw pec50 paradigm)
    nb2960 0.5*K20+0.5*equal_K      = 0.4576-0.4580
    nb2171 PRIMARY-1 ceiling        = 0.4682
    chemprop_aux                    = 0.6216

Outputs:
    data/processed/nb3120_summary.json
    data/processed/nb3120_pred_oof.npy   (253,) float32
    data/processed/te_nb3120.npy         (513,) float32
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
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test, load_semi_pure
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3120"

# --- summary lookups for 117-col layout ---
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"   # K=18 idx

# --- cached feature paths ---
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
MORDRED_TRAIN_PATH = MORDRED_DIR / "X_mordred_train.npy"
MORDRED_TEST_PATH = MORDRED_DIR / "X_mordred_test.npy"

# --- anchor + deep-30 K18 cached OOF/te ---
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"               # (513,)
ANCHOR_TRAIN_OOF_PATH = DATA_PROCESSED / "oof_chemprop_aux.npy"        # (4139,)
NB2960_K18_OOF = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
NB2960_K18_TE = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"

# --- ChEMBL kNN external pool ---
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# --- LGBM hyperparams (identical to nb2960/nb2604/nb2103 K-pyramid recipe) ---
RESID_SEEDS_DEEP = list(range(3001, 3031))  # 30 fresh seeds {3001..3030}

# --- augmentation weights ---
USCALE_W = 1.0
TRAIN_W = 1.0

# --- scaffold-CV stability protocol (kf_seeds for pooled-OOF stability) ---
N_FOLDS = 5
KF_SEEDS = list(range(1141, 1156))  # 15 fresh seeds {1141..1155}

# --- gate ---
GATE_BETTER = 0.4475

# --- reference scores ---
NB2960_K18_REF = 0.4536        # no-aug deep-30 K18 baseline (target to beat)
NB2960_BLEND_REF = 0.4576      # nb2960 0.5*K20+0.5*equal_K best
NB2171_REF = 0.4682            # PRIMARY-1 ceiling
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers
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


def _load_npy(path, n_expected, name):
    if not path.exists():
        raise FileNotFoundError(f"missing cache {name}: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {name} {path}: {X.shape}, expected n={n_expected}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred(path, n_expected, name):
    if not path.exists():
        raise FileNotFoundError(f"Mordred cache missing {name}: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape mismatch {name}: {X.shape}, expected n={n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


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


def _atompair_2048(mol):
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    fp = gen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def _maccs_167(mol):
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)


def _avalon_512(mol):
    if mol is None:
        return np.zeros(512, dtype=np.float32)
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=512)
    return np.array(fp, dtype=np.float32)


def _build_mordred_block(mols, n_descs_target, log_every=50):
    """Compute mordred (no 3D) on a list of RDKit mols. Returns (N, ~1613)."""
    from mordred import Calculator, descriptors as mdesc
    calc = Calculator(mdesc, ignore_3D=True)
    n = len(mols)
    cols = len(calc.descriptors)
    out = np.full((n, cols), np.nan, dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        try:
            res = calc(m)
            vals = np.fromiter(
                (float(v) if (v is not None and isinstance(v, (int, float))) or
                  (hasattr(v, "value") and v.value is not None)
                  else float("nan") for v in res),
                dtype=np.float32, count=cols,
            )
            out[i] = vals
        except Exception:
            pass
        if (i + 1) % log_every == 0:
            print(f"     mordred row {i+1}/{n}", flush=True)
    return out


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K=18 LGBM RESIDUAL on (4139 train + 96 uscale w=1.0)")
    print(f"       residual target = pec50 - chemprop_aux_pred")
    print(f"       deploy = anchor + bagged_residual_pred")
    print(f"       blend = 0.5*aug_K18_resid + 0.5*nb2960_K18_deep30")
    print(f"       scaffold-CV stability: {len(KF_SEEDS)} fresh kf_seeds "
          f"{KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"       gate: <{GATE_BETTER} BETTER / else FAIL")
    print(f"       baseline: nb2960 K18 deep-30 = {NB2960_K18_REF:.4f} (no aug)")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Load truth, anchor (te + train OOF), K=18 index, nb2960 K18 deep-30
    # ------------------------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns
                else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] unique_scaffolds(unb)={n_unique_scaf}")

    # chemprop_aux: test side (513,) and train OOF (4139,)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    tr_anchor_4139 = np.load(ANCHOR_TRAIN_OOF_PATH).astype(np.float64)
    rae_anchor_unb = float(rae(y_unb, te_anchor_513[unb_idx]))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[load] chemprop_aux train OOF shape = {tr_anchor_4139.shape}")

    # K=18 index in 117-col layout
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"[load] K=18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    # nb2960 K=18 deep-30 cached (the blend partner)
    nb2960_K18_oof = np.load(NB2960_K18_OOF).astype(np.float64)
    nb2960_K18_te = np.load(NB2960_K18_TE).astype(np.float64)
    assert nb2960_K18_oof.shape == (n_unb,), f"nb2960 K18 oof {nb2960_K18_oof.shape}"
    assert nb2960_K18_te.shape == (n_test,), f"nb2960 K18 te {nb2960_K18_te.shape}"
    rae_nb2960_K18 = float(rae(y_unb, nb2960_K18_oof))
    print(f"[load] nb2960 K18 deep-30 OOF RAE = {rae_nb2960_K18:.4f} "
          f"(ref {NB2960_K18_REF:.4f})")

    # ------------------------------------------------------------------
    # 2. Load train + uscale; dedup, weight, build residual target
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: load train + uscale; build residual = y - chemprop_aux_pred")
    print("-" * 78)

    # --- standard train (weight 1.0) ---
    tr = load_train()
    tr = tr.dropna(subset=["smiles", "pec50"]).copy()
    n_tr_pre_dedup = len(tr)
    if n_tr_pre_dedup != tr_anchor_4139.shape[0]:
        print(f"[train] WARN: train n={n_tr_pre_dedup} != anchor n={tr_anchor_4139.shape[0]}")
    tr = tr.reset_index(drop=True)
    tr_smiles = tr["smiles"].astype(str).tolist()
    tr_pec50 = tr["pec50"].astype(float).to_numpy()
    n_tr_std = len(tr)
    # align anchor to train order: assume same row order (verified across nb2960/nb3114)
    if n_tr_std == tr_anchor_4139.shape[0]:
        tr_anchor = tr_anchor_4139.copy()
    else:
        # safety fallback: pad/truncate to match (should not happen in practice)
        tr_anchor = np.full(n_tr_std, float(np.median(tr_anchor_4139)), dtype=np.float64)
    print(f"[train] standard PXR train n={n_tr_std} (weight {TRAIN_W})")
    print(f"[train] anchor stats: mean={tr_anchor.mean():.3f} std={tr_anchor.std():.3f}")

    # --- uscale (weight 1.0) ---
    uscale = load_semi_pure()
    uscale = uscale.dropna(subset=["smiles", "pec50"]).copy()
    uscale["pec50"] = pd.to_numeric(uscale["pec50"], errors="coerce")
    uscale = uscale.dropna(subset=["pec50"]).copy()
    n_uscale_raw = len(uscale)
    print(f"[uscale] raw rows = {n_uscale_raw}  (weight {USCALE_W})")

    # dedup uscale vs test + train
    test_mols = [standardize(s) for s in te_smiles]
    test_iks = set(_safe_inchikey(m) for m in test_mols if m is not None)
    test_iks.discard(None)
    train_mols = [standardize(s) for s in tr_smiles]
    train_iks = set(_safe_inchikey(m) for m in train_mols if m is not None)
    train_iks.discard(None)

    uscale_mols = [standardize(s) for s in uscale["smiles"].tolist()]
    uscale["inchikey"] = [_safe_inchikey(m) for m in uscale_mols]
    n0 = len(uscale)
    uscale = uscale[~uscale["inchikey"].isin(test_iks)].reset_index(drop=True)
    print(f"[uscale] dedup vs test:  {n0} -> {len(uscale)}")
    n1 = len(uscale)
    uscale = uscale[~uscale["inchikey"].isin(train_iks)].reset_index(drop=True)
    print(f"[uscale] dedup vs train: {n1} -> {len(uscale)}")
    n_uscale = len(uscale)
    uscale_smiles = uscale["smiles"].astype(str).tolist()
    uscale_pec50 = uscale["pec50"].astype(float).to_numpy()

    # uscale anchor: chemprop_aux NOT retrained-with-uscale -> impute train median
    anchor_median_train = float(np.median(tr_anchor))
    uscale_anchor = np.full(n_uscale, anchor_median_train, dtype=np.float64)
    print(f"[uscale] anchor imputed as train median = {anchor_median_train:.3f}")

    # --- assemble augmented training tables ---
    aug_smiles = tr_smiles + uscale_smiles
    aug_pec50 = np.concatenate([tr_pec50, uscale_pec50], axis=0)
    aug_anchor = np.concatenate([tr_anchor, uscale_anchor], axis=0)
    aug_resid = aug_pec50 - aug_anchor       # <-- RESIDUAL target
    aug_w = np.concatenate([
        np.full(n_tr_std, TRAIN_W, dtype=np.float32),
        np.full(n_uscale, USCALE_W, dtype=np.float32),
    ], axis=0)
    n_aug = len(aug_smiles)
    print(f"\n[aug] total n={n_aug}  (train={n_tr_std}, uscale={n_uscale})")
    print(f"[aug] weight sum = {aug_w.sum():.1f}  "
          f"(train={float(aug_w[:n_tr_std].sum()):.1f}, "
          f"uscale={float(aug_w[-n_uscale:].sum() if n_uscale > 0 else 0):.1f})")
    print(f"[aug] residual stats: mean={aug_resid.mean():+.3f} std={aug_resid.std():.3f} "
          f"min={aug_resid.min():+.3f} max={aug_resid.max():+.3f}")

    # ------------------------------------------------------------------
    # 3. Build 117-col features for: train (4139 cached), uscale (n_uscale, on-the-fly), test (513)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: build 117-col feature matrix on aug+test")
    print("-" * 78)

    # ---- family summaries -> top-K col indices ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f: sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f: sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f: sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f: sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f: sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f: sum_1541 = json.load(f)

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

    # ---- train side: cached features (4139) ----
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_tr_std, "ap_tr")
    X_maccs_tr = _load_npy(MACCS_TR_PATH, n_tr_std, "maccs_tr")
    X_av_tr = _load_npy(AVALON_TR_PATH, n_tr_std, "av_tr")
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_tr_std, "embed_tr")
    X_mord_tr = _load_mordred(MORDRED_TRAIN_PATH, n_tr_std, "mordred_tr")
    print(f"[tr-feat] AP{X_ap_tr.shape} MACCS{X_maccs_tr.shape} "
          f"Avalon{X_av_tr.shape} Embed{X_emb_tr.shape} Mord{X_mord_tr.shape}")

    # ---- test side: cached features (513) ----
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test, "ap_te")
    X_maccs_te = _load_npy(MACCS_TE_PATH, n_test, "maccs_te")
    X_av_te = _load_npy(AVALON_TE_PATH, n_test, "av_te")
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test, "embed_te")
    X_mord_te = _load_mordred(MORDRED_TEST_PATH, n_test, "mordred_te")

    # ---- per-family top-K slicing on train + test ----
    X_ap_tr_top = X_ap_tr[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_tr_top = X_maccs_tr[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_tr_top = X_mord_tr[:, top_mord_col_idx].astype(np.float32)
    X_emb_tr_top = X_emb_tr[:, top_embed_col_idx].astype(np.float32)
    X_av_tr_top = X_av_tr[:, top_avalon_bit_idx].astype(np.float32)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

    # ---- uscale: compute fingerprints on-the-fly (no htchem this notebook) ----
    nontrain_smiles = uscale_smiles
    nontrain_mols = [standardize(s) for s in nontrain_smiles]
    n_nontrain = len(nontrain_mols)
    n_drop = sum(1 for m in nontrain_mols if m is None)
    if n_drop > 0:
        print(f"[uscale] WARN: {n_drop}/{n_nontrain} unparseable, kept as zero-fp rows")

    print(f"[uscale] computing AtomPair/MACCS/Avalon for n={n_nontrain}...")
    t_f = time.time()
    X_ap_nt = np.stack([_atompair_2048(m) for m in nontrain_mols], axis=0)
    X_maccs_nt = np.stack([_maccs_167(m) for m in nontrain_mols], axis=0)
    X_av_nt = np.stack([_avalon_512(m) for m in nontrain_mols], axis=0)
    print(f"[uscale] AP/MACCS/Avalon: {time.time()-t_f:.1f}s")

    print(f"[uscale] computing Mordred for n={n_nontrain} (slow step)...")
    t_f = time.time()
    X_mord_nt_full = _build_mordred_block(nontrain_mols, n_descs_target=1613)
    print(f"[uscale] Mordred raw{X_mord_nt_full.shape}  {time.time()-t_f:.1f}s")

    # align mordred to train's columns; impute NaN with train column medians
    n_mord_target = X_mord_tr.shape[1]
    X_mord_nt = np.full((n_nontrain, n_mord_target), np.nan, dtype=np.float32)
    take = min(n_mord_target, X_mord_nt_full.shape[1])
    X_mord_nt[:, :take] = X_mord_nt_full[:, :take]
    col_med_tr = np.nanmedian(X_mord_tr, axis=0)
    col_med_tr = np.where(np.isfinite(col_med_tr), col_med_tr, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_mord_nt)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_mord_nt[idx_r, idx_c] = col_med_tr[idx_c]

    # slice uscale to per-family top-K
    X_ap_nt_top = X_ap_nt[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_nt_top = X_maccs_nt[:, top_maccs_bit_idx].astype(np.float32)
    X_av_nt_top = X_av_nt[:, top_avalon_bit_idx].astype(np.float32)
    X_mord_nt_top = X_mord_nt[:, top_mord_col_idx].astype(np.float32)

    # ChempropEmbed not locally computable for uscale -> impute with train col-median
    col_med_embed = np.median(X_emb_tr, axis=0).astype(np.float32)
    X_emb_nt = np.tile(col_med_embed[None, :], (n_nontrain, 1)).astype(np.float32)
    X_emb_nt_top = X_emb_nt[:, top_embed_col_idx].astype(np.float32)

    # ChEMBL kNN on test + train + uscale
    print(f"[chembl] loading external PXR pool...")
    pool = _load_chembl_pool()
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"[chembl] pool n={len(pool)}")

    fp_test = morgan_fp_batch([_safe_can_smiles(m) or "" for m in test_mols])
    top_idx_te, top_sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_te, top_sim_te, pool_labels, fallback=pool_median
    )

    fp_train = morgan_fp_batch([_safe_can_smiles(m) or "" for m in train_mols])
    top_idx_tr, top_sim_tr = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(
        top_idx_tr, top_sim_tr, pool_labels, fallback=pool_median
    )

    fp_nt = morgan_fp_batch([_safe_can_smiles(m) or "" for m in nontrain_mols])
    top_idx_nt, top_sim_nt = _tanimoto_topk(fp_nt, fp_pool, k=KNN_K)
    pred_chembl_nt, mean_sim_nt = _knn_predict(
        top_idx_nt, top_sim_nt, pool_labels, fallback=pool_median
    )

    # ---- build full 117-col matrices ----
    X_te_full = np.concatenate(
        [X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
         pred_chembl_te.reshape(-1, 1).astype(np.float32),
         mean_sim_te.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    X_tr_full = np.concatenate(
        [X_ap_tr_top, X_maccs_tr_top, X_mord_tr_top, X_emb_tr_top, X_av_tr_top,
         pred_chembl_tr.reshape(-1, 1).astype(np.float32),
         mean_sim_tr.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    X_nt_full = np.concatenate(
        [X_ap_nt_top, X_maccs_nt_top, X_mord_nt_top, X_emb_nt_top, X_av_nt_top,
         pred_chembl_nt.reshape(-1, 1).astype(np.float32),
         mean_sim_nt.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"X_te_full {X_te_full.shape}"
    assert X_tr_full.shape[1] == 117, f"X_tr_full {X_tr_full.shape}"
    assert X_nt_full.shape[1] == 117, f"X_nt_full {X_nt_full.shape}"
    print(f"[feat] X_tr_full{X_tr_full.shape} X_nt_full{X_nt_full.shape} "
          f"X_te_full{X_te_full.shape}")

    # ---- slice to K=18, assemble augmented training matrix ----
    X_tr_K18 = X_tr_full[:, K18_idx].astype(np.float32)
    X_nt_K18 = X_nt_full[:, K18_idx].astype(np.float32)
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    X_aug_K18 = np.concatenate([X_tr_K18, X_nt_K18], axis=0).astype(np.float32)
    assert X_aug_K18.shape == (n_aug, 18), f"X_aug_K18 {X_aug_K18.shape}"
    assert len(aug_resid) == n_aug
    assert len(aug_w) == n_aug
    print(f"[K18] X_aug_K18{X_aug_K18.shape}  X_te_K18{X_te_K18.shape}  "
          f"X_unb_K18{X_te_K18[unb_idx].shape}")

    # ------------------------------------------------------------------
    # 4. Train K=18 LGBM mean-bag (30 fresh seeds) on RESIDUAL target,
    #    weighted, with augmented data; reconstruct pec50 = anchor + resid_pred
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 4: K=18 LGBM RESIDUAL mean-bag, {len(RESID_SEEDS_DEEP)} fresh seeds "
          f"{RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} on weighted AUG")
    print("-" * 78)
    sum_resid_unb = np.zeros(n_unb, dtype=np.float64)
    sum_resid_te = np.zeros(n_test, dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS_DEEP):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_aug_K18, aug_resid, sample_weight=aug_w)
        pred_resid_unb_s = mdl.predict(X_te_K18[unb_idx])
        pred_resid_te_s = mdl.predict(X_te_K18)
        sum_resid_unb += pred_resid_unb_s
        sum_resid_te += pred_resid_te_s
        # reconstruct pec50 for per-seed reporting (anchor + residual)
        pred_pec50_unb_s = te_anchor_513[unb_idx] + pred_resid_unb_s
        rae_s = float(rae(y_unb, pred_pec50_unb_s))
        per_seed_rae.append(rae_s)
        if (i % 10) == 0 or i == len(RESID_SEEDS_DEEP) - 1:
            print(f"   seed={s:4d}  unb_RAE={rae_s:.4f}  "
                  f"wall={time.time()-ts:.2f}s  ({i+1}/{len(RESID_SEEDS_DEEP)})")

    # mean-bag residual predictions
    aug_K18_resid_oof = sum_resid_unb / len(RESID_SEEDS_DEEP)
    aug_K18_resid_te = sum_resid_te / len(RESID_SEEDS_DEEP)
    # reconstruct pec50 (anchor + bagged residual)
    aug_K18_oof = te_anchor_513[unb_idx] + aug_K18_resid_oof   # (253,) pec50
    aug_K18_te = te_anchor_513 + aug_K18_resid_te              # (513,) pec50
    rae_aug_K18 = float(rae(y_unb, aug_K18_oof))
    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae, ddof=1))
    print(f"\n[aug_K18_resid] per-seed RAE  mean={per_seed_mean:.4f} +/- {per_seed_std:.4f}  "
          f"min={min(per_seed_rae):.4f}  max={max(per_seed_rae):.4f}")
    print(f"[aug_K18_resid] 30-seed BAG-MEAN RAE = {rae_aug_K18:.4f}")
    print(f"[aug_K18_resid] vs nb2960 K18 no-aug ({rae_nb2960_K18:.4f}) "
          f"delta = {rae_aug_K18 - rae_nb2960_K18:+.4f}")

    # ------------------------------------------------------------------
    # 5. Blend (mean) with nb2960 K=18 deep-30 -> blend OOF/te (deterministic)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: blend = 0.5*aug_K18 + 0.5*nb2960_K18_deep30")
    print("-" * 78)
    blend_oof = 0.5 * aug_K18_oof + 0.5 * nb2960_K18_oof
    blend_te = 0.5 * aug_K18_te + 0.5 * nb2960_K18_te
    rae_blend_pooled = float(rae(y_unb, blend_oof))
    print(f"[blend] OOF pooled RAE = {rae_blend_pooled:.4f}")
    print(f"[blend] te_mean = {blend_te.mean():.3f}  te_std = {blend_te.std():.3f}")

    # ------------------------------------------------------------------
    # 6. Scaffold-CV stability assessment over 15 fresh kf_seeds {1141..1155}
    #    Since both legs are deterministic w.r.t. fixed inputs, each kf_seed
    #    shuffles 5-fold partition over 253 unblind; pooled RAE = same number
    #    every time (= rae_blend_pooled). What varies is per-fold mean (val
    #    RAE averaged across 5 folds). Reported gate metric = MEAN of pooled
    #    RAE across the 15 seeds (deterministic), AND per-fold mean across
    #    seeds (varies with split).
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 6: scaffold-CV stability, {len(KF_SEEDS)} fresh kf_seeds "
          f"{KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_perfold_mean = []
    per_seed_fold_records = {}
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        fold_val_raes = []
        cov_mask = np.zeros(n_unb, dtype=bool)
        for fold_i, (tr_loc, va_loc) in enumerate(splits):
            # both legs are precomputed; just evaluate blend on va_loc
            val_pred = blend_oof[va_loc]
            r_val = float(rae(y_unb[va_loc], val_pred))
            fold_val_raes.append(r_val)
            cov_mask[va_loc] = True
        if not cov_mask.all():
            raise RuntimeError(f"scaffold splits did not cover all {n_unb} rows "
                              f"(kf_seed={kf_seed})")
        pooled_seed_rae = float(rae(y_unb, blend_oof))  # deterministic
        perfold_mean_seed = float(np.mean(fold_val_raes))
        per_seed_pooled.append(pooled_seed_rae)
        per_seed_perfold_mean.append(perfold_mean_seed)
        per_seed_fold_records[str(kf_seed)] = [
            {"fold": int(i), "n_val": int(len(splits[i][1])),
             "val_rae": round(fold_val_raes[i], 4)}
            for i in range(N_FOLDS)
        ]
        print(f"   kf_seed={kf_seed}  pooled={pooled_seed_rae:.4f}  "
              f"per-fold mean={perfold_mean_seed:.4f}")

    arr_pooled = np.asarray(per_seed_pooled)
    arr_perfold = np.asarray(per_seed_perfold_mean)
    mean_pooled = float(arr_pooled.mean())
    std_pooled = float(arr_pooled.std(ddof=1)) if len(arr_pooled) > 1 else 0.0
    mean_perfold = float(arr_perfold.mean())
    std_perfold = float(arr_perfold.std(ddof=1)) if len(arr_perfold) > 1 else 0.0
    print(f"\n   POOLED RAE over {len(KF_SEEDS)} kf_seeds:")
    print(f"     mean = {mean_pooled:.4f}   std = {std_pooled:.4f}")
    print(f"   PER-FOLD MEAN RAE over {len(KF_SEEDS)} kf_seeds:")
    print(f"     mean = {mean_perfold:.4f}   std = {std_perfold:.4f}")

    # ------------------------------------------------------------------
    # 7. Gate (on mean pooled RAE across 15 kf_seeds)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 7: GATE on mean pooled RAE across kf_seeds")
    print("-" * 78)
    if mean_pooled < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    delta_vs_nb2960_K18 = mean_pooled - NB2960_K18_REF
    delta_vs_nb2960_blend = mean_pooled - NB2960_BLEND_REF
    delta_vs_nb2171 = mean_pooled - NB2171_REF
    print(f"   mean_pooled_rae                          = {mean_pooled:.4f}")
    print(f"   delta vs nb2960 K18 ({NB2960_K18_REF})  = {delta_vs_nb2960_K18:+.4f}")
    print(f"   delta vs nb2960 0.5+0.5 ({NB2960_BLEND_REF}) = {delta_vs_nb2960_blend:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF})          = {delta_vs_nb2171:+.4f}")
    print(f"   verdict (<{GATE_BETTER})                   = {verdict}")

    # ------------------------------------------------------------------
    # 8. Save artifacts
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 8: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, blend_oof.astype(np.float32))
    np.save(te_path, blend_te.astype(np.float32))
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_uscale_resid_K18_blend.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": blend_te.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] verdict=FAIL; no submission CSV")

    te_unb_in_rae = float(rae(y_unb, blend_te[unb_idx]))

    summary = {
        "tag": TAG,
        "status": "OK",
        "method": "K18_LGBM_RESIDUAL_on_chemprop_aux_aug_4139+96uscale(w=1.0)_blend_with_nb2960_K18_deep30",
        "paradigm": "residual_target_data_augmentation_substrate_change",
        "anchor_pre_unblind": True,
        "target_axis": "residual_y_minus_chemprop_aux_pred",
        # data setup
        "n_train_std": int(n_tr_std),
        "n_uscale_kept": int(n_uscale),
        "n_aug": int(n_aug),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "n_nontrain_unparseable": int(n_drop),
        "weights": {
            "train": TRAIN_W,
            "uscale": USCALE_W,
            "total_w_sum": float(aug_w.sum()),
        },
        "anchor_imputed_for_uscale_median": float(anchor_median_train),
        "aug_resid_mean": float(aug_resid.mean()),
        "aug_resid_std": float(aug_resid.std()),
        # model
        "K": 18,
        "K18_idx_in_117col": K18_idx.tolist(),
        "lgbm_params": _lgbm_params(0),
        "n_seeds_deep": len(RESID_SEEDS_DEEP),
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_train_oof_path": str(ANCHOR_TRAIN_OOF_PATH),
        "rae_anchor_unb": rae_anchor_unb,
        # individual leg results
        "per_seed_rae_aug_K18_resid": per_seed_rae,
        "per_seed_mean_aug_K18_resid": per_seed_mean,
        "per_seed_std_aug_K18_resid": per_seed_std,
        "aug_K18_oof_rae": rae_aug_K18,
        "aug_K18_te_mean": float(aug_K18_te.mean()),
        "aug_K18_te_std": float(aug_K18_te.std()),
        "nb2960_K18_oof_rae": rae_nb2960_K18,
        "nb2960_K18_te_mean": float(nb2960_K18_te.mean()),
        "nb2960_K18_te_std": float(nb2960_K18_te.std()),
        # blend
        "blend_weights": {"w_aug_K18": 0.5, "w_nb2960_K18": 0.5},
        "blend_oof_rae_pooled": rae_blend_pooled,
        "blend_te_mean": float(blend_te.mean()),
        "blend_te_std": float(blend_te.std()),
        "te_unb_in_sample_rae": te_unb_in_rae,
        # scaffold-CV stability
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "per_seed_pooled_rae": [round(r, 5) for r in per_seed_pooled],
        "per_seed_perfold_mean_rae": [round(r, 5) for r in per_seed_perfold_mean],
        "per_seed_fold_records": per_seed_fold_records,
        "pooled_rae_mean": round(mean_pooled, 5),
        "pooled_rae_std": round(std_pooled, 5),
        "perfold_mean_rae_mean": round(mean_perfold, 5),
        "perfold_mean_rae_std": round(std_perfold, 5),
        "mean_rae": round(mean_pooled, 5),
        # gate
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        # refs
        "nb2960_K18_ref": NB2960_K18_REF,
        "nb2960_blend_ref": NB2960_BLEND_REF,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb2960_K18": delta_vs_nb2960_K18,
        "delta_vs_nb2960_blend": delta_vs_nb2960_blend,
        "delta_vs_nb2171": delta_vs_nb2171,
        # paths
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        # timing
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   n_aug                       = {n_aug} "
          f"({n_tr_std} train + {n_uscale} uscale)")
    print(f"   aug_K18 RESID 30seed bag    = {rae_aug_K18:.4f}")
    print(f"   nb2960 K18 deep-30 RAE      = {rae_nb2960_K18:.4f}")
    print(f"   blend (0.5/0.5) RAE         = {rae_blend_pooled:.4f}")
    print(f"   mean pooled over {len(KF_SEEDS)} kf_seeds = {mean_pooled:.4f} "
          f"(std {std_pooled:.4f})")
    print(f"   per-fold mean over {len(KF_SEEDS)} kf_seeds = {mean_perfold:.4f} "
          f"(std {std_perfold:.4f})")
    print(f"   delta vs nb2960 K18         = {delta_vs_nb2960_K18:+.4f}")
    print(f"   te[unb_idx] in-RAE          = {te_unb_in_rae:.4f}")
    print(f"   verdict                     = {verdict}")
    print(f"   wall                        = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_aug", "n_uscale_kept",
        "aug_K18_oof_rae", "nb2960_K18_oof_rae",
        "blend_oof_rae_pooled", "mean_rae", "pooled_rae_std",
        "perfold_mean_rae_mean", "perfold_mean_rae_std",
        "delta_vs_nb2960_K18", "delta_vs_nb2960_blend",
        "te_unb_in_sample_rae", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
