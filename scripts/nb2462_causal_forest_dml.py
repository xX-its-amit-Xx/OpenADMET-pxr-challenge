"""nb2462 -- Causal Forest (Double-ML) heterogeneous treatment effect.

SUBSTRATE
    Anchor:    te_nb2240_K20.npy (513,)            -- deploy K=20 anchor
               nb2240_mean_bag_oof_K20.npy (253,)  -- OOF K=20 anchor on unblind
               _audit_unblind_y.npy (253,)          -- truth on unblind
    Treatment T (continuous) on 253 = nb_chemprop_aux_oof - nb2240_oof_K20
    Outcome   Y on 253             = y_unb            - nb2240_oof_K20
    Heterogeneity X = X_117 on 253 (5-way AtomPair/MACCS/Mordred/ChempropEmbed/
        Avalon + ChEMBL kNN as in nb2103/nb2240)

PROTOCOL
    5-fold scaffold CV (Bemis-Murcko on unb_smiles).  In each fold:
      fit CausalForestDML(n_estimators=300, max_depth=4,
                          discrete_treatment=False, random_state=42)
      on (Y_tr, T_tr, X_tr) and predict tau_hat on X_va.
    Per-row prediction = anchor_oof + tau_hat * actual_T_oof   (on unblind).
    Deploy: refit on all 253, predict tau on 513 via X_te_117, then
      te_pred = te_nb2240_K20 + tau_te * (te_chemprop_aux - te_nb2240_K20).

GATE
    mean_rae < 0.4570 -> PROMOTE  (5 PRE-clean scaffold seeds)
    else -> FAIL.

OUTPUTS
    data/processed/nb2462_summary.json
    data/processed/nb2462_pred_oof.npy     (253,)  -- mean across kf_seeds
    data/processed/te_nb2462.npy           (513,)  -- deploy refit
    (no submission CSV; ladder ingestion is the caller's job)
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
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

try:
    from econml.dml import CausalForestDML
except Exception as e:  # pragma: no cover
    print(f"[fatal] econml import failed: {e}")
    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "nb2462_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"status": "INSTALL_FAILED", "err": str(e)}, indent=2))
    print("INSTALL_FAILED")
    sys.exit(0)

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2462"

# ------------------------------------------------------------------ paths
ANCHOR_TE_PATH      = DATA_PROCESSED / "te_nb2240_K20.npy"
ANCHOR_OOF_PATH     = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
CHEMPROP_TE_PATH    = DATA_PROCESSED / "te_chemprop_aux.npy"
CHEMPROP_OOF_PATH   = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
UNB_IDX_PATH        = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH          = DATA_PROCESSED / "_audit_unblind_y.npy"

# 117-col feature inputs (same as nb2240 / nb2103)
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH    = DATA_PROCESSED / "te_maccs.npy"
EMBED_TE_PATH    = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH   = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR      = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"  # MACCS top
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"  # Avalon top
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"  # AtomPair ranked
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"  # Mordred
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"  # AtomPair K
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"  # ChempropEmbed K

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# Cycle 174 CF-DML
N_FOLDS  = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
CF_KW    = dict(
    n_estimators=300,
    max_depth=4,
    discrete_treatment=False,
    random_state=42,
)

GATE = 0.4570

# ==========================================================================
# Helpers (subset of nb2240 -- 117-col rebuild)
# ==========================================================================

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


def build_X117_test(n_test, te_smiles):
    """Construct the 117-col 5-way feature matrix on the 513 test set."""
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

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_inchikeys = {ik for ik in (_safe_inchikey(m) for m in test_mols) if ik}
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = [Chem.MolToSmiles(m) if m else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl, mean_sim = _knn_predict(top_idx_knn, top_sim_knn, pool_labels, pool_median)

    X_te_full = np.concatenate(
        [
            X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"
    return X_te_full


# ==========================================================================
# Main
# ==========================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Causal Forest DML on (anchor=nb2240_K20, T=chemprop-anchor)")
    print("=" * 78)

    # ---- Load anchor & outcome ----
    anchor_te_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)   # (513,)
    anchor_oof_253 = np.load(ANCHOR_OOF_PATH).astype(np.float64)  # (253,)
    chemprop_te_513 = np.load(CHEMPROP_TE_PATH).astype(np.float64)
    chemprop_oof_253 = np.load(CHEMPROP_OOF_PATH).astype(np.float64)
    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}  n_test=513")

    # T, Y on unblind (OOF -- honest)
    T_unb = (chemprop_oof_253 - anchor_oof_253).astype(np.float64)
    Y_unb = (y_unb - anchor_oof_253).astype(np.float64)
    rae_anchor = float(rae(y_unb, anchor_oof_253))
    rae_chem = float(rae(y_unb, chemprop_oof_253))
    print(f"[anchor]  oof RAE = {rae_anchor:.4f}")
    print(f"[chem]    oof RAE = {rae_chem:.4f}")
    print(f"[T]   mean={T_unb.mean():+.4f}  std={T_unb.std():.4f}  "
          f"min={T_unb.min():+.4f}  max={T_unb.max():+.4f}")
    print(f"[Y]   mean={Y_unb.mean():+.4f}  std={Y_unb.std():.4f}")

    # ---- Load smiles + scaffolds ----
    te = load_test()
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- X_117 (rebuild) ----
    print("[X117] building 117-col feature matrix on 513 test compounds")
    t1 = time.time()
    X_te_117 = build_X117_test(len(te_smiles), te_smiles)
    X_unb_117 = X_te_117[unb_idx].astype(np.float64)
    print(f"[X117] X_unb_117={X_unb_117.shape}  X_te_117={X_te_117.shape}  "
          f"({time.time() - t1:.1f}s)")

    # ---- CF-DML over scaffold-CV seeds ----
    print("-" * 78)
    print(f"CF-DML  scaffold-{N_FOLDS}fold  seeds={KF_SEEDS}  n_estimators=300  max_depth=4")
    print("-" * 78)

    per_seed_pred = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae  = []
    per_seed_taus = []

    for si, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        tau_oof = np.full(n_unb, np.nan, dtype=np.float64)
        for fi, (tr_loc, va_loc) in enumerate(splits):
            cf = CausalForestDML(**CF_KW)
            cf.fit(
                Y=Y_unb[tr_loc],
                T=T_unb[tr_loc],
                X=X_unb_117[tr_loc],
                W=None,
            )
            tau_oof[va_loc] = np.asarray(
                cf.effect(X_unb_117[va_loc]), dtype=np.float64
            ).reshape(-1)
        # combine: prediction = anchor + tau_hat * actual_T
        pred_oof = anchor_oof_253 + tau_oof * T_unb
        per_seed_pred[si] = pred_oof
        r = float(rae(y_unb, pred_oof))
        per_seed_rae.append(r)
        per_seed_taus.append({
            "kf_seed": int(kf_seed),
            "rae": r,
            "tau_mean": float(np.mean(tau_oof)),
            "tau_std":  float(np.std(tau_oof)),
            "tau_min":  float(np.min(tau_oof)),
            "tau_max":  float(np.max(tau_oof)),
        })
        print(f"   kf_seed={kf_seed}  RAE={r:.4f}  "
              f"tau mean={tau_oof.mean():+.4f} std={tau_oof.std():.4f}  "
              f"wall={time.time() - ts:.1f}s")

    mean_pred_oof = per_seed_pred.mean(axis=0)
    mean_rae = float(rae(y_unb, mean_pred_oof))
    seed_mean_rae = float(np.mean(per_seed_rae))
    seed_std_rae  = float(np.std(per_seed_rae))
    print()
    print(f"[OOF]  per-seed mean RAE = {seed_mean_rae:.4f} +/- {seed_std_rae:.4f}")
    print(f"[OOF]  mean-bag RAE      = {mean_rae:.4f}")
    print(f"[OOF]  anchor RAE        = {rae_anchor:.4f}  "
          f"delta = {mean_rae - rae_anchor:+.4f}")

    status = "PROMOTE" if mean_rae < GATE else "FAIL"
    print(f"[GATE]  threshold={GATE:.4f}  mean_rae={mean_rae:.4f}  -> {status}")

    # ---- Deploy refit: train on ALL 253 -> tau on 513 ----
    print("[deploy] refit CF-DML on all 253 -> tau on 513")
    cf_full = CausalForestDML(**CF_KW)
    cf_full.fit(Y=Y_unb, T=T_unb, X=X_unb_117, W=None)
    tau_te = np.asarray(cf_full.effect(X_te_117.astype(np.float64)),
                        dtype=np.float64).reshape(-1)
    T_te = (chemprop_te_513 - anchor_te_513).astype(np.float64)
    te_pred = anchor_te_513 + tau_te * T_te
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"[deploy] te[unb_idx] RAE (in-sample) = {te_unb_rae:.4f}  "
          f"(expected lower than OOF {mean_rae:.4f})")

    # ---- Save ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path  = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_pred_oof.astype(np.float32))
    np.save(te_path,  te_pred.astype(np.float32))
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "status": status,
        "gate": GATE,
        "anchor": "nb2240_K20",
        "T_def": "chemprop_aux_oof - nb2240_K20_oof",
        "Y_def": "y_unb - nb2240_K20_oof",
        "X_def": "X_117 (AtomPair/MACCS/Mordred/ChempropEmbed/Avalon + ChEMBL kNN)",
        "n_unb": n_unb,
        "n_test": int(len(te_smiles)),
        "n_features_X": int(X_unb_117.shape[1]),
        "cf_kw": {k: (v if not isinstance(v, np.generic) else v.item())
                  for k, v in CF_KW.items()},
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "rae_anchor_oof": rae_anchor,
        "rae_chemprop_oof": rae_chem,
        "rae_mean_bag_oof": mean_rae,
        "rae_per_seed_mean": seed_mean_rae,
        "rae_per_seed_std":  seed_std_rae,
        "per_seed": per_seed_taus,
        "rae_te_unb_in_sample": te_unb_rae,
        "tau_te": {
            "mean": float(np.mean(tau_te)),
            "std":  float(np.std(tau_te)),
            "min":  float(np.min(tau_te)),
            "max":  float(np.max(tau_te)),
        },
        "wall_seconds": float(time.time() - t0),
    }
    sum_path = DATA_PROCESSED / f"{TAG}_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] {sum_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s   status = {status}")


if __name__ == "__main__":
    main()
