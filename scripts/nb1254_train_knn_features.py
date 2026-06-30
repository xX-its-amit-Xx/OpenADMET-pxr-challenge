"""nb1254 -- CONTROL EXPERIMENT for nb1242: train-only kNN features (no ChEMBL).

Question:
    nb1242 added MACCS-167 + pred_chembl_pec50 + sim (169 cols, ChEMBL-derived)
    and broke 0.5431 against the nb1183 (0.5513) and nb1211 (0.5451) ceilings.
    Was the win driven by the ChEMBL-specific signal (scaffold-diverse external
    chemistry knowledge), or just by the kNN-as-feature pattern in general?

    Control: replace the ChEMBL pool with the INTERNAL training set (4139).
    Same architecture (MACCS-167 + pred_pec50_k5 + sim_k5 = 169 cols).
    Same residual-LGBM bag (5 seeds shallow Huber, 5-fold cross-fit).
    Compare directly to nb1242 (0.5431) and nb1183 (0.5513).

Protocol:
    1. Compute Morgan-2048 (r=2) for ALL 4139 train + 513 test compounds.
    2. For each unblind test row, Tanimoto similarity to all 4139 train,
       top-k=5 (sorted descending by sim).  Features:
          pred_train_pec50  = sum(w_i * pec50_i) / sum(w_i)
          train_knn_sim     = mean of top-5 Tanimoto similarities
       Same recipe as nb1242 (just pool swapped).
    3. Residual learner: anchor = nb1070_pred_oof on 253 unblind rows;
       residual = y_unb - nb1070_pred_oof;
       features = concat[MACCS-167(unb), pred_train_pec50[unb_idx],
                         train_knn_sim[unb_idx]]  -> (253, 169).
    4. 5-seed shallow LGBM Huber bag, 5-fold cross-fit per seed, mean-bag
       pooled RAE.  Identical capacity to nb1183 and nb1242 (head-to-head).
    5. Verdict at 0.003 margin vs nb1242 (0.5431) and nb1183 (0.5513).

Outputs:
    scripts/nb1254_train_knn_features.py             (this file)
    data/processed/nb1254_summary.json
    data/processed/nb1254_mean_bag_oof.npy           (253,)  float32
    data/processed/nb1254_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1254_median_bag_oof.npy         (253,)  float32
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1254"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

KNN_K = 5
SIM_FLOOR = 1e-6

MORGAN_BITS = 2048
MORGAN_RADIUS = 2

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF = 0.5771
NB1183_REF = 0.5513    # MACCS-only residual bag
NB1242_REF = 0.5431    # MACCS + ChEMBL-kNN residual bag (the model being controlled)
NB1211_REF = 0.5451
DECISION_MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical to nb1183/nb1242 (head-to-head)."""
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


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    """Returns (top_idx (n_q, k) int32, top_sim (n_q, k) float32) blockwise."""
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


def _knn_predict(top_idx: np.ndarray, top_sim: np.ndarray,
                 pool_labels: np.ndarray, fallback: float):
    """Similarity-weighted mean of pool_labels at top_idx; rows with all-zero
    sim get the fallback (pool median).  Returns (pred (n_q,), mean_sim (n_q,))."""
    w = np.clip(top_sim, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = float(np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i])
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


def _orthogonality_probe(mean_bag_oof: np.ndarray) -> dict:
    """Pearson vs nb1183 (MACCS-only) and nb1242 (MACCS + ChEMBL-kNN) mean-bag."""
    out: dict = {}
    for ref_tag in ("nb1183", "nb1242"):
        p = DATA_PROCESSED / f"{ref_tag}_mean_bag_oof.npy"
        if not p.exists():
            out[f"pearson_vs_{ref_tag}_mean_bag"] = None
            out[f"{ref_tag}_probe_error"] = f"missing {p}"
            continue
        try:
            ref = np.load(p).astype(np.float64)
            if ref.shape[0] != mean_bag_oof.shape[0]:
                out[f"pearson_vs_{ref_tag}_mean_bag"] = None
                out[f"{ref_tag}_probe_error"] = (
                    f"shape mismatch: ref={ref.shape} vs self={mean_bag_oof.shape}"
                )
                continue
            a = mean_bag_oof.astype(np.float64)
            if a.std() > 0 and ref.std() > 0:
                r = float(np.corrcoef(a, ref)[0, 1])
            else:
                r = float("nan")
            out[f"pearson_vs_{ref_tag}_mean_bag"] = r
        except Exception as e:
            out[f"pearson_vs_{ref_tag}_mean_bag"] = None
            out[f"{ref_tag}_probe_error"] = repr(e)
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CONTROL for nb1242: MACCS + TRAIN-kNN (no ChEMBL); "
          f"shallow residual-LGBM bag on nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + pred_train_pec50_k5 + train_knn_sim_k5 (169)")
    print("=" * 78)

    # ---- Load anchor + truth ----
    tr = load_train()
    te = load_test()
    n_train = len(tr)
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_train={n_train}  n_test={n_test}  n_unb={n_unb}")

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Pool labels (training pEC50) ----
    pool_labels = tr["pec50"].to_numpy(dtype=np.float32)
    if np.isnan(pool_labels).any():
        raise ValueError("train pec50 has NaNs (unexpected)")
    pool_median = float(np.median(pool_labels))
    print(f"\n[pool] train pec50: n={len(pool_labels)}  "
          f"mean={pool_labels.mean():.3f}  std={pool_labels.std():.3f}  "
          f"median={pool_median:.3f}  "
          f"min={pool_labels.min():.3f}  max={pool_labels.max():.3f}")

    # ---- Morgan fingerprints ----
    print("\n" + "-" * 78)
    print(f"MORGAN-{MORGAN_BITS} (r={MORGAN_RADIUS}) FINGERPRINTS")
    print("-" * 78)
    t_fp = time.time()
    fp_pool = morgan_fp_batch(
        tr["smiles"].tolist(), radius=MORGAN_RADIUS, n_bits=MORGAN_BITS
    )
    fp_test = morgan_fp_batch(
        te["smiles"].tolist(), radius=MORGAN_RADIUS, n_bits=MORGAN_BITS
    )
    print(f"   pool FP: {fp_pool.shape}  density={fp_pool.mean():.4f}")
    print(f"   test FP: {fp_test.shape}  density={fp_test.mean():.4f}")
    print(f"   wall_fp = {time.time() - t_fp:.1f}s")

    # ---- kNN k=5 Tanimoto (over ALL 513 test rows; we then slice on unb_idx) ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (k={KNN_K}) -- test (513) vs TRAIN pool ({n_train})")
    print("-" * 78)
    t_knn = time.time()
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_train_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    top1_sim = top_sim[:, 0]
    print(f"   pred_train_pec50  mean={pred_train_pec50.mean():.3f}  "
          f"std={pred_train_pec50.std():.3f}  "
          f"min={pred_train_pec50.min():.3f}  max={pred_train_pec50.max():.3f}")
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
    print(f"   wall_knn = {time.time() - t_knn:.1f}s")

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"\n   MACCS unb shape = {X_maccs_unb.shape}")

    # ---- Build residual feature matrix on 253 ----
    pred_train_unb = pred_train_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_train_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + pred_train_pec50 + train_knn_sim)")
    print(f"   feature head [0..3, last 2 cols]: "
          f"pred_train_pec50_unb mean={pred_train_unb.mean():.3f} "
          f"std={pred_train_unb.std():.3f}; "
          f"sim_unb mean={mean_sim_unb.mean():.3f} std={mean_sim_unb.std():.3f}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow LGBM Huber, dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
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
    print(f"   per-seed min/max       = {rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1183 ref             = {NB1183_REF:.4f}  (MACCS residual bag)")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (MACCS + CHEMBL-kNN bag)  <-- HEAD-TO-HEAD")
    print(f"   nb1211 ref             = {NB1211_REF:.4f}  (SLSQP variants bag)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_REF - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    near_nb1242 = abs(rae_mean_bag - NB1242_REF) <= 0.01

    # Verdict on whether ChEMBL specifically or just kNN-as-feature drove nb1242
    if beats_nb1242:
        verdict = ("TRAIN_KNN_BEATS_CHEMBL_KNN  "
                   "internal kNN-as-feature is the dominant axis; "
                   "ChEMBL adds no incremental value")
        chembl_specific = False
    elif near_nb1242:
        verdict = ("TRAIN_KNN_MATCHES_CHEMBL_KNN  "
                   "nb1242 win was the kNN-as-feature pattern, NOT ChEMBL "
                   "specifically; ChEMBL and train pools carry equivalent signal")
        chembl_specific = False
    elif beats_nb1183:
        verdict = ("TRAIN_KNN_BEATS_NB1183_BUT_NOT_NB1242  "
                   "kNN-as-feature pattern partial credit; ChEMBL pool adds "
                   "scaffold-diverse incremental signal that internal train cannot supply")
        chembl_specific = True
    elif abs(rae_mean_bag - NB1183_REF) <= DECISION_MARGIN:
        verdict = ("TRAIN_KNN_MATCHES_NB1183  "
                   "internal train-kNN equals plain MACCS bag; "
                   "the nb1242 incremental gain over 1183 is entirely ChEMBL-specific")
        chembl_specific = True
    elif beats_nb1070:
        verdict = ("TRAIN_KNN_HELPS_NB1070_BUT_BELOW_NB1183  "
                   "internal kNN features carry weak signal; ChEMBL was the real driver")
        chembl_specific = True
    else:
        verdict = ("TRAIN_KNN_FLAT_OR_HURTS  "
                   "internal kNN-as-feature inert; nb1242 win was ChEMBL-specific")
        chembl_specific = True
    print(f"   verdict                = {verdict}")
    print(f"   chembl_specific_win    = {chembl_specific}")

    # ---- Orthogonality probe ----
    print("\n" + "-" * 78)
    print("ORTHOGONALITY PROBE (corrected mean-bag OOF vs nb1183 / nb1242)")
    print("-" * 78)
    ortho = _orthogonality_probe(mean_bag_oof)
    for k, v in ortho.items():
        if isinstance(v, float):
            print(f"   {k} = {v:+.4f}")
        else:
            print(f"   {k} = {v}")

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
        "control_for": "nb1242 (MACCS + CHEMBL-kNN)",
        "pool_source": "training_set_only_no_chembl",
        "n_pool": int(n_train),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "morgan_bits": MORGAN_BITS,
        "morgan_radius": MORGAN_RADIUS,
        "knn_k": KNN_K,
        "pool_pec50_mean": float(pool_labels.mean()),
        "pool_pec50_std": float(pool_labels.std()),
        "pool_pec50_median": pool_median,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "top1_sim_max": float(top1_sim.max()),
        "mean5_sim_p10": float(np.percentile(mean_sim, 10)),
        "mean5_sim_p50": float(np.percentile(mean_sim, 50)),
        "mean5_sim_p90": float(np.percentile(mean_sim, 90)),
        "n_zero_neighbor_rows": n_zero_neighbor,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(feat_dim),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
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
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "delta_mean_bag_vs_nb1211": rae_mean_bag - NB1211_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1242": bool(beats_nb1242),
        "near_nb1242_within_0p01": bool(near_nb1242),
        "chembl_specific_win": bool(chembl_specific),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1183_ref": NB1183_REF,
        "nb1242_ref": NB1242_REF,
        "nb1211_ref": NB1211_REF,
        "decision_margin": DECISION_MARGIN,
        "orthogonality_probe": ortho,
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
        "n_pool", "top1_sim_p10", "top1_sim_p50", "top1_sim_p90",
        "n_zero_neighbor_rows",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "delta_mean_bag_vs_nb1242",
        "delta_mean_bag_vs_nb1211",
        "beats_nb1070", "beats_nb1183", "beats_nb1242",
        "near_nb1242_within_0p01",
        "chembl_specific_win",
        "verdict",
        "orthogonality_probe",
    ):
        print(f"  {k}: {res.get(k)}")
