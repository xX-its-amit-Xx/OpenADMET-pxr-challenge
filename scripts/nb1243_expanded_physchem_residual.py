"""nb1243 -- Expanded physchem + scaffold-statistic residual feature.

Hypothesis:
    Beyond MACCS substructure dictionary (nb1183), a richer chemistry-grounded
    feature set built from TRAIN-METADATA-ONLY (not extra labels) may add real
    correction signal:
        - physchem (MW, logP, TPSA, HBD, HBA, rotbonds, fsp3, n_rings,
          formal_charge) -- 9 features, derived from RDKit Descriptors.
        - scaffold_train_freq -- count of Bemis-Murcko scaffold occurrences in
          train; 0 means novel scaffold (rare-scaffold failure mode tag).
        - max_train_tanimoto -- nearest-neighbour Morgan-ECFP4 similarity into
          the 4139-row train pool; proxy for "how off-manifold is this row?".
        - knn5_train_pec50_mean -- similarity-weighted mean of train pEC50 for
          the 5 nearest train neighbours.
        - knn5_train_pec50_std  -- similarity-weighted std of those 5 neighbour
          pEC50s; high std = disagreement / cliff region.
    Total: MACCS-167 + 9 physchem + 4 derived = 180 features.

    Anchor = nb1070 (PRE-unblind LB-faithful, pooled RAE 0.5771 on 253 unblind).
    Residual = y_unb - nb1070_oof; 5-seed bag shallow LGBM Huber depth=3,
    5-fold cross-fit per seed; mean-bag pooled RAE compared to:
        nb1183 (MACCS-only residual, 0.5513)  -- ref-mid
        nb1211 (combine-bob blend,        0.5451) -- ref-low
    Decision margin 0.003.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1243_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1243_mean_bag_oof.npy            (253,)   float32
  data/processed/nb1243_summary.json
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

from pxr.data import load_train, load_test
from pxr.chem import (
    standardize,
    compute_physchem,
    morgan_fp_batch,
    bemis_murcko,
)
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED
from rdkit import Chem

TAG = "nb1243"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"   # (513, 167) uint8

PHYSCHEM_KEYS = [
    "mw",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotbonds",
    "fsp3",
    "n_rings",
    "formal_charge",
]
# 9 physchem keys

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1183_MEAN_BAG_REF = 0.5513   # MACCS-only residual on nb1070
NB1211_MEAN_BAG_REF = 0.5451   # combine-bob blend (best PRE-LB analog)
DECISION_MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1183 bag for fair comparison."""
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


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _physchem_matrix(smiles: list[str]) -> np.ndarray:
    """Compute physchem for each SMILES, return (N, 9) float32."""
    n = len(smiles)
    out = np.zeros((n, len(PHYSCHEM_KEYS)), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = standardize(s)
        if mol is None:
            continue
        d = compute_physchem(mol)
        if d is None:
            continue
        for j, k in enumerate(PHYSCHEM_KEYS):
            v = d.get(k)
            if v is not None:
                out[i, j] = float(v)
    return out


def _scaffolds(smiles: list[str]) -> list[str | None]:
    out = []
    for s in smiles:
        mol = standardize(s)
        if mol is None:
            out.append(None)
            continue
        try:
            out.append(bemis_murcko(mol))
        except Exception:
            out.append(None)
    return out


def _tanimoto_matrix(fp_te: np.ndarray, fp_tr: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity between every row of fp_te and fp_tr.

    fp_te: (M, B) uint8 bit vectors
    fp_tr: (N, B) uint8 bit vectors
    returns (M, N) float32.
    """
    a = fp_te.astype(np.uint16)
    b = fp_tr.astype(np.uint16)
    inter = a @ b.T  # (M, N) intersection count
    a_pop = a.sum(axis=1, keepdims=True)  # (M, 1)
    b_pop = b.sum(axis=1, keepdims=True).T  # (1, N)
    union = a_pop + b_pop - inter
    union = np.where(union == 0, 1, union)
    return (inter.astype(np.float32) / union.astype(np.float32))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- residual-LGBM bag on nb1070 anchor with EXPANDED feature set:")
    print(f"          MACCS-167 + 9 physchem + 4 scaffold/sim stats = 180 cols")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          LGBM depth=3 leaves=7 n_est=80 lr=0.05 obj=huber(alpha=1.0)")
    print("=" * 78)

    # ----- Load splits and unblind index -----
    tr = load_train()
    te = load_test()
    n_train, n_test = len(tr), len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_train={n_train}  n_test={n_test}  n_unb={n_unb}")

    # ----- Anchor -----
    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"{anchor_path} missing (run nb1070 first).")
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(f"{anchor_path} shape {anchor_oof.shape} vs n_unb={n_unb}")
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy  pooled RAE = {rae_anchor:.4f}  "
          f"(ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ----- MACCS-167 (cached) -----
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"te_maccs shape {X_maccs_te.shape} vs n_test={n_test}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"[feat] MACCS unb = {X_maccs_unb.shape}  density={X_maccs_unb.mean():.4f}")

    # ----- Physchem (9) on test rows -----
    smi_te = te["smiles"].tolist()
    smi_tr = tr["smiles"].tolist()
    pec50_tr = tr["pec50"].to_numpy(dtype=np.float64)

    print(f"[feat] computing physchem on {n_test} test SMILES ...")
    X_phys_te = _physchem_matrix(smi_te)
    X_phys_unb = X_phys_te[unb_idx]
    print(f"[feat] physchem unb = {X_phys_unb.shape}  "
          f"col means = "
          f"{np.array2string(X_phys_unb.mean(axis=0), precision=2)}")

    # ----- Morgan FPs for train + test (for max_tan, kNN-5 stats) -----
    print(f"[feat] Morgan ECFP4 for {n_train} train + {n_test} test SMILES ...")
    fp_tr = morgan_fp_batch(smi_tr, radius=2, n_bits=2048)
    fp_te = morgan_fp_batch(smi_te, radius=2, n_bits=2048)
    print(f"[feat] fp_tr shape={fp_tr.shape}  fp_te shape={fp_te.shape}")

    # ----- Tanimoto matrix test->train (513 x 4139) -----
    print(f"[feat] Tanimoto matrix test x train ({n_test} x {n_train}) ...")
    sim_te_tr = _tanimoto_matrix(fp_te, fp_tr)
    print(f"[feat] sim mean={sim_te_tr.mean():.4f}  "
          f"max-per-row median={np.median(sim_te_tr.max(axis=1)):.4f}")

    # ----- max_tan + knn5 stats -----
    K = 5
    sim_unb_tr = sim_te_tr[unb_idx]  # (253, 4139)
    max_tan_unb = sim_unb_tr.max(axis=1).astype(np.float32)
    # top-K neighbour indices per unblind row
    topk_idx = np.argpartition(-sim_unb_tr, K, axis=1)[:, :K]
    knn5_mean = np.zeros(n_unb, dtype=np.float32)
    knn5_std = np.zeros(n_unb, dtype=np.float32)
    for i in range(n_unb):
        idx_i = topk_idx[i]
        sims_i = sim_unb_tr[i, idx_i].astype(np.float64)
        pec50_i = pec50_tr[idx_i]
        if sims_i.sum() > 0:
            w = sims_i / sims_i.sum()
            mu = float(np.sum(w * pec50_i))
            var = float(np.sum(w * (pec50_i - mu) ** 2))
            knn5_mean[i] = mu
            knn5_std[i] = float(np.sqrt(max(var, 0.0)))
        else:
            knn5_mean[i] = float(pec50_i.mean())
            knn5_std[i] = float(pec50_i.std())
    print(f"[feat] max_tan unb  mean={max_tan_unb.mean():.4f}  "
          f"median={float(np.median(max_tan_unb)):.4f}")
    print(f"[feat] knn5_mean   mean={knn5_mean.mean():.4f}  "
          f"std-across-rows={knn5_mean.std():.4f}")
    print(f"[feat] knn5_std    mean={knn5_std.mean():.4f}  "
          f"max={knn5_std.max():.4f}")

    # ----- Scaffold train frequency -----
    print(f"[feat] Murcko scaffolds for train + test ...")
    scaf_tr = _scaffolds(smi_tr)
    scaf_te = _scaffolds(smi_te)
    scaf_freq_map: dict[str, int] = {}
    for s in scaf_tr:
        if s is not None:
            scaf_freq_map[s] = scaf_freq_map.get(s, 0) + 1
    scaf_freq_te = np.array(
        [scaf_freq_map.get(s, 0) if s is not None else 0 for s in scaf_te],
        dtype=np.float32,
    )
    scaf_freq_unb = scaf_freq_te[unb_idx]
    novel_share = float((scaf_freq_unb == 0).mean())
    print(f"[feat] scaf_freq unb  mean={scaf_freq_unb.mean():.2f}  "
          f"novel-scaffold share = {novel_share:.3f}")

    # ----- Concatenate feature matrix on unblind -----
    derived = np.stack(
        [scaf_freq_unb, max_tan_unb, knn5_mean, knn5_std], axis=1
    ).astype(np.float32)
    X_unb = np.concatenate(
        [X_maccs_unb, X_phys_unb, derived], axis=1
    ).astype(np.float32)
    print(f"[feat] FINAL X_unb shape = {X_unb.shape}  "
          f"(MACCS {X_maccs_unb.shape[1]} + phys {X_phys_unb.shape[1]} "
          f"+ derived {derived.shape[1]})")

    # ----- Per-seed residual cross-fit -----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (shallow LGBM, expanded {X_unb.shape[1]}-d)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
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

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1183 mean_bag ref    = {NB1183_MEAN_BAG_REF:.4f}  "
          f"(MACCS-only residual)")
    print(f"   nb1211 mean_bag ref    = {NB1211_MEAN_BAG_REF:.4f}  "
          f"(combine-bob blend)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_MEAN_BAG_REF - DECISION_MARGIN
    beats_nb1211 = rae_mean_bag < NB1211_MEAN_BAG_REF - DECISION_MARGIN

    if beats_nb1211:
        verdict = "EXPANDED_BEATS_NB1211_NEW_PRELB_LEADER"
    elif beats_nb1183:
        verdict = "EXPANDED_BEATS_NB1183_BUT_NOT_NB1211"
    elif beats_nb1070:
        verdict = "EXPANDED_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "EXPANDED_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "EXPANDED_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ----- Save outputs -----
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
        "feature_source": "maccs_167+physchem_9+scaf_freq+max_tan+knn5_mean+knn5_std",
        "feature_dim_total": int(X_unb.shape[1]),
        "feature_dim_maccs": int(X_maccs_unb.shape[1]),
        "feature_dim_physchem": int(X_phys_unb.shape[1]),
        "feature_dim_derived": int(derived.shape[1]),
        "physchem_keys": PHYSCHEM_KEYS,
        "knn_k": K,
        "n_unb": n_unb,
        "n_train": n_train,
        "n_test": n_test,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "scaf_freq_mean_unb": float(scaf_freq_unb.mean()),
        "scaf_freq_novel_share_unb": novel_share,
        "max_tan_mean_unb": float(max_tan_unb.mean()),
        "max_tan_median_unb": float(np.median(max_tan_unb)),
        "knn5_mean_mean_unb": float(knn5_mean.mean()),
        "knn5_std_mean_unb": float(knn5_std.mean()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_nb1070": rae_median_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_MEAN_BAG_REF,
        "delta_mean_bag_vs_nb1211": rae_mean_bag - NB1211_MEAN_BAG_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1211": bool(beats_nb1211),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "nb1211_mean_bag_ref": NB1211_MEAN_BAG_REF,
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
        "feature_dim_total",
        "feature_dim_maccs",
        "feature_dim_physchem",
        "feature_dim_derived",
        "rae_anchor_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean",
        "rae_per_seed_median",
        "rae_per_seed_std",
        "rae_mean_bag",
        "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "delta_mean_bag_vs_nb1211",
        "beats_nb1070",
        "beats_nb1183",
        "beats_nb1211",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
