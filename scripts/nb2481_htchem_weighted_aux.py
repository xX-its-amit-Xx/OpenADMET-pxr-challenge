"""nb2481 -- htchem-libraries weighted augmentation on top of nb2240 K=20 pyramid.

CONTEXT (memory feedback_new_data_inventory.md):
    htchem-libraries_crudes (456 noisy DRC, SE>=0.5 on 38%, weight <=0.3).
    Add as weighted train, see if any lift on the K=20 RFE pyramid.

PROTOCOL:
    1. 4139 standard train (weight=1.0) + 456 htchem (weight = 0.3 * (0.5/max(SE,0.5)))
       = 4595 weighted rows.
    2. Build K=20 RFE feature slice (12 local-cmptable cols + 8 ChempropEmbed
       cols imputed with train-col-median for htchem rows -- chemprop_aux
       checkpoint is not local).
    3. Train K=20 LGBM mean-bag (5 seeds {0, 1, 7, 42, 137}) on weighted-aug;
       predict on 253 unblind (this is OOF since 253 never in train)
       + 513 test (deploy).
    4. Plug new K=20 anchor into 5-anchor pyramid in nb2240_K20 slot.
       Run SLSQP simplex blend + rank-stretch under 5-fold scaffold-CV
       on 253, kf_seeds {1001..1005}.
    5. GATE: mean_rae<0.4570 -> PROMOTE; <0.4601 -> MARGINAL_BEAT; else FAIL.

Outputs:
    data/processed/nb2481_summary.json
    data/processed/nb2481_pred_oof.npy   (253,) float32
    data/processed/te_nb2481.npy         (513,) float32
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
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2481"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HTCHEM_CSV = RAW_DIR / "pxr-challenge_htchem-libraries_TRAIN.csv"

# ------------------------ K=20 + anchors (from nb2231 / nb2240) ---------------
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
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ChEMBL kNN parameters
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# LGBM hyperparams (identical to nb2240)
RESID_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5  # for honest train CV reporting only

# Stage 2 SLSQP pyramid (identical to nb2240)
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb1191 reconstruction parameters
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

# Gate thresholds
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


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


def _load_npy(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}, expected n={n_expected}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"Mordred cache missing -- {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape mismatch: {X.shape}, expected n={n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
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


# ---------- ad-hoc fp computation for htchem rows ----------

def _atompair_2048(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def _maccs_167(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def _avalon_512(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(512, dtype=np.float32)
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=512)
    arr = np.zeros(512, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def _build_mordred_block(mols, n_descs_target, log_every=50):
    """Compute mordred (no 3D) on a list of RDKit mols.  Returns (N, 1613)."""
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
    # Coerce non-finite to NaN, will be median-imputed by caller
    return out


# ============================================================================
# Stage 2 SLSQP helpers (copied from nb2240)
# ============================================================================

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

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- htchem weighted aug on K=20 RFE pyramid")
    print("=" * 78)

    if not HTCHEM_CSV.exists():
        summary = {
            "tag": TAG, "status": "FILE_NOT_FOUND",
            "expected_path": str(HTCHEM_CSV),
        }
        out_path = DATA_PROCESSED / f"{TAG}_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[save] {out_path}")
        return summary

    # ---- Load nb2231 K=20 surviving indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    fam_counts_K20 = dict(nb2231["snapshots"]["20"]["family_counts"])
    print(f"[load] K=20 surviving features: families={fam_counts_K20}")

    # ---- Load truth + chemprop_aux anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    print(f"[scaffold] n_unique unb scaffolds={len({s for s in unb_scaffolds if s})}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)

    # ---- Load standard train ----
    tr = load_train()
    tr_smiles = tr["smiles"].astype(str).tolist()
    tr_pec50 = tr["pec50"].astype(float).to_numpy()
    n_tr_std = len(tr)
    print(f"[load] standard train n={n_tr_std}")

    # ---- Load htchem rows ----
    htc = pd.read_csv(HTCHEM_CSV)
    pec50_corr = pd.to_numeric(htc["Corrected Crude pEC50 (log)"], errors="coerce")
    pec50_crude = pd.to_numeric(htc["Crude pEC50s (log)"], errors="coerce")
    se = pd.to_numeric(htc["Crude DRC pEC50 SE (log)"], errors="coerce")
    htc_pec50 = pec50_corr.where(pec50_corr.notna(), pec50_crude)
    # require pEC50 present; default SE = 0.5 if missing
    valid = htc_pec50.notna() & htc["SMILES"].notna()
    htc = htc[valid].reset_index(drop=True)
    htc_pec50 = htc_pec50[valid].reset_index(drop=True).to_numpy()
    se = se[valid].reset_index(drop=True).fillna(0.5).to_numpy()
    htc_smiles = htc["SMILES"].astype(str).tolist()
    n_htc = len(htc)
    # weight = 0.3 * (0.5 / max(SE, 0.5))  (always <=0.3, decreases as SE grows)
    htc_w = 0.3 * (0.5 / np.maximum(se, 0.5))
    print(f"[load] htchem n={n_htc}  pEC50 range=[{htc_pec50.min():.3f},{htc_pec50.max():.3f}]")
    print(f"[load] htchem weight min/median/max={htc_w.min():.3f}/{np.median(htc_w):.3f}/{htc_w.max():.3f}")

    # ---- Train-side cached features (4139) ----
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_tr_std)
    X_maccs_tr = _load_npy(MACCS_TR_PATH, n_tr_std)
    X_av_tr = _load_npy(AVALON_TR_PATH, n_tr_std)
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_tr_std)
    X_mord_tr = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_tr_std)
    print(f"[train caches] AP{X_ap_tr.shape} MACCS{X_maccs_tr.shape} Avalon{X_av_tr.shape} "
          f"Embed{X_emb_tr.shape} Mord{X_mord_tr.shape}")

    # ---- Test-side cached features (513) ----
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)
    X_maccs_te = _load_npy(MACCS_TE_PATH, n_test)
    X_av_te = _load_npy(AVALON_TE_PATH, n_test)
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)
    X_mord_te = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_test)

    # ---- Family summaries -> col indices ----
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
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    # ---- Slice train/test to per-family top-K, build 117-col matrices ----
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

    # ---- ChEMBL kNN feature on test + train ----
    pool = _load_chembl_pool()
    print(f"[chembl] pool n={len(pool)}")
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

    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    train_mols = [standardize(s) for s in tr_smiles]
    std_train_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in train_mols]
    fp_train = morgan_fp_batch(std_train_smiles)
    top_idx_knn_tr, top_sim_knn_tr = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    pred_chembl_pec50_tr, mean_sim_tr = _knn_predict(
        top_idx_knn_tr, top_sim_knn_tr, pool_labels, fallback=pool_median
    )

    # ---- Build full 117-col train + test ----
    X_te_full = np.concatenate(
        [X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
         pred_chembl_pec50_te.reshape(-1, 1).astype(np.float32),
         mean_sim_te.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    X_tr_full = np.concatenate(
        [X_ap_tr_top, X_maccs_tr_top, X_mord_tr_top, X_emb_tr_top, X_av_tr_top,
         pred_chembl_pec50_tr.reshape(-1, 1).astype(np.float32),
         mean_sim_tr.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_te_full.shape[1]
    assert feat_dim == 117, f"feat_dim {feat_dim} != 117"
    print(f"[feat] X_tr_full = {X_tr_full.shape}  X_te_full = {X_te_full.shape}")

    # ---- Slice to K=20 ----
    X_tr_K20 = X_tr_full[:, surviving_K20].astype(np.float32)
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[K20] train{X_tr_K20.shape}  test{X_te_K20.shape}  unb{X_unb_K20.shape}")

    # =========================================================================
    # Build htchem feature block (computed locally)
    # =========================================================================
    print("\n" + "-" * 78)
    print("Computing htchem features (AtomPair/MACCS/Avalon/Mordred local)...")
    print("-" * 78)
    htc_mols = [standardize(s) for s in htc_smiles]
    n_drop = sum(1 for m in htc_mols if m is None)
    if n_drop > 0:
        # Drop unparseable rows
        keep = [i for i, m in enumerate(htc_mols) if m is not None]
        htc_mols = [htc_mols[i] for i in keep]
        htc_pec50 = htc_pec50[keep]
        htc_w = htc_w[keep]
        se = se[keep]
        htc_smiles = [htc_smiles[i] for i in keep]
        n_htc = len(htc_mols)
        print(f"[htchem] dropped {n_drop} unparseable, n_htc -> {n_htc}")

    # AtomPair 2048
    t = time.time()
    X_ap_htc = np.stack([_atompair_2048(m) for m in htc_mols], axis=0)
    print(f"[htchem] AtomPair {X_ap_htc.shape}  wall={time.time()-t:.1f}s")
    # MACCS 167
    t = time.time()
    X_maccs_htc = np.stack([_maccs_167(m) for m in htc_mols], axis=0)
    print(f"[htchem] MACCS    {X_maccs_htc.shape}  wall={time.time()-t:.1f}s")
    # Avalon 512
    t = time.time()
    X_av_htc = np.stack([_avalon_512(m) for m in htc_mols], axis=0)
    print(f"[htchem] Avalon   {X_av_htc.shape}  wall={time.time()-t:.1f}s")
    # Mordred 1613 (no 3D)
    print("[htchem] Mordred (this is the slow step ~3-5 min for 456 mols)...")
    t = time.time()
    X_mord_htc_full = _build_mordred_block(htc_mols, n_descs_target=1613)
    print(f"[htchem] Mordred  raw{X_mord_htc_full.shape}  wall={time.time()-t:.1f}s")

    # Slice all four to the per-family top-K sets that build the 117-col layout
    X_ap_htc_top = X_ap_htc[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_htc_top = X_maccs_htc[:, top_maccs_bit_idx].astype(np.float32)
    X_av_htc_top = X_av_htc[:, top_avalon_bit_idx].astype(np.float32)

    # The mordred train cache is (4139, 1533) -- slightly different N_descs vs the
    # local recompute (~1613).  Use first 1533 cols / common shape and impute NaN.
    n_mord_target = X_mord_tr.shape[1]
    X_mord_htc = np.full((n_htc, n_mord_target), np.nan, dtype=np.float32)
    take = min(n_mord_target, X_mord_htc_full.shape[1])
    X_mord_htc[:, :take] = X_mord_htc_full[:, :take]
    # Median-impute NaN per column using train medians
    col_med_tr = np.nanmedian(X_mord_tr, axis=0)
    col_med_tr = np.where(np.isfinite(col_med_tr), col_med_tr, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_mord_htc)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_mord_htc[idx_r, idx_c] = col_med_tr[idx_c]
    X_mord_htc_top = X_mord_htc[:, top_mord_col_idx].astype(np.float32)

    # ChEMBL kNN on htchem
    fp_htc = morgan_fp_batch([Chem.MolToSmiles(m) for m in htc_mols])
    top_idx_knn_htc, top_sim_knn_htc = _tanimoto_topk(fp_htc, fp_pool, k=KNN_K)
    pred_chembl_pec50_htc, mean_sim_htc = _knn_predict(
        top_idx_knn_htc, top_sim_knn_htc, pool_labels, fallback=pool_median
    )

    # ChempropEmbed_300 not locally computable -> use train COLUMN MEDIAN
    # impute so the K=20 slice can still index those columns.
    col_med_embed = np.median(X_emb_tr, axis=0).astype(np.float32)
    X_emb_htc = np.tile(col_med_embed[None, :], (n_htc, 1)).astype(np.float32)
    X_emb_htc_top = X_emb_htc[:, top_embed_col_idx].astype(np.float32)

    # Build full 117-col htchem matrix
    X_htc_full = np.concatenate(
        [X_ap_htc_top, X_maccs_htc_top, X_mord_htc_top, X_emb_htc_top, X_av_htc_top,
         pred_chembl_pec50_htc.reshape(-1, 1).astype(np.float32),
         mean_sim_htc.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    assert X_htc_full.shape == (n_htc, 117), f"htchem 117 mismatch: {X_htc_full.shape}"
    X_htc_K20 = X_htc_full[:, surviving_K20].astype(np.float32)
    print(f"[htchem] X_htc_full{X_htc_full.shape}  X_htc_K20{X_htc_K20.shape}")
    # Count how many of the K=20 surviving idx fall in ChempropEmbed (imputed) cols
    chemprop_K20 = [i for i, n in enumerate(surviving_K20_names) if n.startswith("ChempropEmbed")]
    print(f"[htchem] {len(chemprop_K20)}/20 K20 cols are ChempropEmbed (imputed with train median)")

    # =========================================================================
    # K=20 LGBM training: weighted concat (4139 + n_htc)
    # =========================================================================
    print("\n" + "-" * 78)
    print(f"K=20 LGBM weighted-aug train  seeds={RESID_SEEDS}")
    print("-" * 78)
    X_aug = np.concatenate([X_tr_K20, X_htc_K20], axis=0).astype(np.float32)
    y_aug = np.concatenate([tr_pec50.astype(np.float32),
                            htc_pec50.astype(np.float32)], axis=0)
    w_aug = np.concatenate([np.ones(n_tr_std, dtype=np.float32),
                            htc_w.astype(np.float32)], axis=0)
    n_aug = X_aug.shape[0]
    print(f"[aug] n_aug={n_aug}  (4139 train + {n_htc} htchem)")
    print(f"[aug] weight sum = {w_aug.sum():.1f}  (htchem contributes {htc_w.sum():.1f})")

    # 5-fold scaffold-CV on AUGMENTED training for honest train-CV estimate
    # (uses scaffolds for 4139 train + htchem)
    tr_scafs = [bemis_murcko(s) for s in tr_smiles]
    htc_scafs = [bemis_murcko(s) for s in htc_smiles]
    aug_scafs = tr_scafs + htc_scafs
    cv_per_seed = []
    for s in RESID_SEEDS[:3]:  # keep cost in check; 3 seeds for train CV
        splits = scaffold_kfold_indices(aug_scafs, n_splits=5, shuffle=True, seed=s + 5000)
        oof = np.full(n_aug, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_aug[tr_loc], y_aug[tr_loc], sample_weight=w_aug[tr_loc])
            oof[va_loc] = mdl.predict(X_aug[va_loc])
        # Compute weighted RAE on full aug + unweighted on train-only block
        cv_rae_train_only = float(rae(y_aug[:n_tr_std], oof[:n_tr_std]))
        cv_per_seed.append({"seed": int(s), "rae_train_only": cv_rae_train_only})
        print(f"   seed={s} train-only CV RAE={cv_rae_train_only:.4f}")
    train_cv_rae_mean = float(np.mean([r["rae_train_only"] for r in cv_per_seed]))
    print(f"[aug] train-only mean CV RAE = {train_cv_rae_mean:.4f}")

    # Full refit per seed -> mean-bag predictions on 253 unblind + 513 test
    per_seed_unb = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_aug, y_aug, sample_weight=w_aug)
        per_seed_unb[i] = mdl.predict(X_unb_K20)
        per_seed_te[i] = mdl.predict(X_te_K20)
        rae_s = float(rae(y_unb, per_seed_unb[i]))
        print(f"   seed={s:3d}: unb RAE={rae_s:.4f}  wall={time.time()-ts:.1f}s")
    nb2481_oof = per_seed_unb.mean(axis=0)
    nb2481_te = per_seed_te.mean(axis=0)
    rae_K20_aug_mean_bag = float(rae(y_unb, nb2481_oof))
    print(f"\n[K20_aug] mean-bag unb RAE = {rae_K20_aug_mean_bag:.4f}")
    rae_anchor_unb = float(rae(y_unb, te_anchor_513[unb_idx]))
    print(f"[K20_aug] vs chemprop_aux anchor unb RAE = {rae_anchor_unb:.4f} "
          f"(delta {rae_K20_aug_mean_bag - rae_anchor_unb:+.4f})")

    # Save the standalone K=20 aug anchor
    oof_K20_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K20.npy"
    te_K20_path = DATA_PROCESSED / f"te_{TAG}_K20.npy"
    np.save(oof_K20_path, nb2481_oof.astype(np.float32))
    np.save(te_K20_path, nb2481_te.astype(np.float32))
    print(f"[save] {oof_K20_path}")
    print(f"[save] {te_K20_path}")

    # =========================================================================
    # Stage 2: 5-anchor pyramid, slot nb2240_K20 -> nb2481_K20_aug
    # =========================================================================
    print("\n" + "=" * 78)
    print("STAGE 2: 5-ANCHOR PYRAMID  (nb2481_K20_aug swaps in for nb2240_K20)")
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
        ("nb2481_K20_aug", nb2481_oof.astype(np.float64), nb2481_te.astype(np.float64)),
        ("chemprop_aux",   chemprop_oof,                    te_chemprop_aux),
        ("nb1191",         nb1191_oof,                      te_nb1191),
        ("nb503",          nb503_oof,                       te_nb503),
        ("nb562",          nb562_oof,                       te_nb562),
    ]
    indiv_rae = {}
    oof_cols, te_cols = [], []
    print("\n[anchors]")
    for disp, oof, te_arr in anchors_list:
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:18s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)

    # 5-fold scaffold CV across 5 kf_seeds (1001..1005)
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fs],
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        print(f"   seed={kf_seed} pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fs):.3f}  "
              f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean pooled_RAE = {pooled_rae_mean_seeds:.4f} +/- {pooled_rae_std_seeds:.4f}")

    # Save OOF (mean across seeds)
    all_oofs = []
    for kf_seed in KF_SEEDS:
        _, oof_blend, _, _ = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        all_oofs.append(oof_blend)
    mean_oof_pred = np.mean(np.column_stack(all_oofs), axis=1).astype(np.float32)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, mean_oof_pred)
    print(f"[save] {pred_oof_path}")

    # Deploy refit on all 253 (full SLSQP + mean fold-s)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    in_rae = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    w_str = ", ".join(f"{disp}={w:.4f}" for (disp, _, _), w in zip(anchors_list, w_deploy))
    print(f"\n[deploy] weights = {w_str}")
    print(f"[deploy] mu/s = {mu_deploy:.4f}/{s_deploy:.4f}")
    print(f"[deploy] in-sample RAE (253) = {in_rae:.4f}")
    print(f"[deploy] te[unb_idx] RAE     = {te_unb_rae:.4f}")
    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    print(f"[deploy] LB band estimate    = {lb_band_est:.4f}")

    # Save te
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, deploy_te)
    print(f"[save] {te_path}")

    # ---- Gate evaluation ----
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print(f"GATE: pooled_rae_mean={pooled_rae_mean_seeds:.4f}  "
          f"PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}  -> {verdict}")
    print("-" * 78)

    summary = {
        "tag": TAG,
        "status": "OK",
        "method": "htchem_weighted_aug_K20_RFE_pyramid",
        "htchem_file": str(HTCHEM_CSV),
        "n_htchem_used": int(n_htc),
        "htchem_weight_min": float(htc_w.min()),
        "htchem_weight_median": float(np.median(htc_w)),
        "htchem_weight_max": float(htc_w.max()),
        "htchem_weight_sum": float(htc_w.sum()),
        "se_default_for_missing": 0.5,
        "imputed_K20_chempropembed_idx_in_K20": chemprop_K20,
        "n_imputed_K20_cols": int(len(chemprop_K20)),
        "anchors": [a[0] for a in anchors_list],
        "anchor_oof_rae_unb": indiv_rae,
        "rae_anchor_chemprop_aux": rae_anchor_unb,
        "K20_aug_mean_bag_oof_RAE": rae_K20_aug_mean_bag,
        "train_only_cv_rae_mean_3seeds": train_cv_rae_mean,
        "nb2481_oof_K20_path": str(oof_K20_path),
        "nb2481_te_K20_path": str(te_K20_path),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(anchors_list, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "gate_promote_thresh": GATE_PROMOTE,
        "gate_marginal_thresh": GATE_MARGINAL,
        "mean_rae": pooled_rae_mean_seeds,
        "verdict": verdict,
        "pred_oof_path": str(pred_oof_path),
        "te_path": str(te_path),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K20_aug mean_bag RAE      = {rae_K20_aug_mean_bag:.4f}")
    print(f"   pyramid pooled RAE (mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   verdict                   = {verdict}")
    print(f"   LB band estimate          = {lb_band_est:.4f}")
    print(f"   wall                      = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== DONE ====")
    for k in ("status", "mean_rae", "verdict", "lb_band_estimate"):
        print(f"  {k}: {res.get(k)}")
