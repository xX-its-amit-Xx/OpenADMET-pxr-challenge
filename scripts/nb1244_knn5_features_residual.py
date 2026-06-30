"""nb1244 -- Residual-LGBM bag on top of nb1070 using ONLY 4 kNN5 features.

Hypothesis:
    The residual signal extracted by MACCS-only (nb1183, RAE 0.5513) and the
    Mordred/AtomPair siblings may in fact be a proxy for nearest-neighbour
    lookup -- the trees use the substructure bits to push each unblind
    compound toward its most similar training analogs.  If that is true, a
    residual LGBM fed ONLY kNN-derived statistics (max Tanimoto, k=5
    similarity-weighted mean of train pEC50, k=5 std of train pEC50, k=1
    nearest-neighbour pEC50) should approach the same RAE with just 4
    features -- a diagnostic about WHY MACCS works.

Protocol:
  1.  Morgan-2048 (radius=2) for all 4139 train + 513 test compounds.
  2.  For each unblind test row: Tanimoto similarity to all 4139 train,
      top-5 selected by similarity.  Features:
        f0 = max_sim         (Tanimoto to nearest train compound)
        f1 = top1_pec50      (train pEC50 of nearest neighbour)
        f2 = wmean_pec50_k5  (similarity-weighted mean train pEC50 of top-5)
        f3 = std_pec50_k5    (unweighted std of top-5 train pEC50)
  3.  Residual target: y_unb - nb1070_pred_oof.
  4.  Per seed s in {0, 1, 7, 42, 137}: KFold(5, shuffle=True, seed=s),
      shallow LGBM Huber (depth 3, leaves 7, n_est 80, lr 0.05,
      min_child_samples 20, alpha 1.0) -- identical capacity to nb1183 bag.
  5.  pred_corrected_s = nb1070_oof + residual_oof_s ; pooled RAE.
  6.  Mean-bag pooled RAE compared to:
        nb1070 anchor       0.5771
        nb1183 MACCS bag    0.5513   <-- key comparator
        nb1211 BoB mean     0.5451

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1244_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1244_mean_bag_oof.npy            (253,)   float32
  data/processed/nb1244_median_bag_oof.npy          (253,)   float32
  data/processed/nb1244_summary.json
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

from pxr.chem import morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1244"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

K = 5  # number of nearest neighbours for kNN5 features

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1183_MEAN_BAG_REF = 0.5513   # MACCS-only residual bag on nb1070
NB1211_BOB_MEAN_REF = 0.5451   # best-of-bags MEAN_mean
DECISION_MARGIN = 0.003

MORGAN_BITS = 2048
MORGAN_RADIUS = 2


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical to nb1183 (capacity-matched comparison)."""
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


def _tanimoto_matrix(fp_q: np.ndarray, fp_t: np.ndarray) -> np.ndarray:
    """Tanimoto for query (M, B) uint8 vs train (N, B) uint8 -> (M, N) float32.

    Uses popcount via uint8 dot products.  Memory-safe for our sizes:
    (253, 4139) -> ~4.2 MB float32; bit popcounts fit in int16.
    """
    fp_q = fp_q.astype(np.float32)
    fp_t = fp_t.astype(np.float32)
    # |q AND t| = q @ t.T  (both are 0/1)
    inter = fp_q @ fp_t.T  # (M, N) float32
    sum_q = fp_q.sum(axis=1, keepdims=True)         # (M, 1)
    sum_t = fp_t.sum(axis=1, keepdims=True).T       # (1, N)
    union = sum_q + sum_t - inter
    # Avoid div-by-zero: any zero-bit fingerprint -> sim 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return sim


def _build_knn5_features(
    sim_unb_vs_train: np.ndarray,  # (n_unb, 4139) float32 Tanimoto
    y_train: np.ndarray,           # (4139,) train pEC50
    k: int = 5,
) -> tuple[np.ndarray, dict]:
    """Return (n_unb, 4) float32 features = [max_sim, top1_pec50, wmean_pec50_k5, std_pec50_k5]."""
    n_unb, n_train = sim_unb_vs_train.shape
    # top-k indices per unblind row, sorted descending by similarity
    # argpartition -> top-k unordered; we then sort the k columns
    top_k_idx_unordered = np.argpartition(-sim_unb_vs_train, kth=k - 1, axis=1)[:, :k]
    # Build per-row sorted top-k (desc by sim)
    row_idx = np.arange(n_unb)[:, None]
    top_k_sim_unordered = sim_unb_vs_train[row_idx, top_k_idx_unordered]
    order = np.argsort(-top_k_sim_unordered, axis=1)
    top_k_idx = np.take_along_axis(top_k_idx_unordered, order, axis=1)        # (n_unb, k)
    top_k_sim = np.take_along_axis(top_k_sim_unordered, order, axis=1)        # (n_unb, k)
    top_k_y = y_train[top_k_idx]                                              # (n_unb, k)

    max_sim = top_k_sim[:, 0].astype(np.float32)
    top1_pec50 = top_k_y[:, 0].astype(np.float32)

    # similarity-weighted mean of top-5 train pEC50
    weights = top_k_sim.astype(np.float64)
    w_sum = weights.sum(axis=1, keepdims=True)
    # if all weights zero (e.g. duplicate empty FP), fall back to unweighted mean
    safe_w = np.where(w_sum > 0, weights / np.where(w_sum > 0, w_sum, 1.0), 1.0 / k)
    wmean_pec50_k5 = (safe_w * top_k_y).sum(axis=1).astype(np.float32)

    # unweighted std of top-5 train pEC50
    std_pec50_k5 = top_k_y.std(axis=1, ddof=0).astype(np.float32)

    X = np.column_stack([max_sim, top1_pec50, wmean_pec50_k5, std_pec50_k5]).astype(
        np.float32
    )

    feat_stats = {
        "f0_max_sim":   {"mean": float(max_sim.mean()),       "std": float(max_sim.std()),
                         "min": float(max_sim.min()),         "max": float(max_sim.max())},
        "f1_top1_pec50":{"mean": float(top1_pec50.mean()),    "std": float(top1_pec50.std()),
                         "min": float(top1_pec50.min()),      "max": float(top1_pec50.max())},
        "f2_wmean_pec50_k5":{"mean": float(wmean_pec50_k5.mean()), "std": float(wmean_pec50_k5.std()),
                             "min": float(wmean_pec50_k5.min()),  "max": float(wmean_pec50_k5.max())},
        "f3_std_pec50_k5":  {"mean": float(std_pec50_k5.mean()),  "std": float(std_pec50_k5.std()),
                             "min": float(std_pec50_k5.min()),    "max": float(std_pec50_k5.max())},
    }
    return X, feat_stats


def _orthogonality_probe(mean_bag_oof: np.ndarray) -> dict:
    """Pearson vs the nb1183 MACCS mean-bag corrected OOF.

    High Pearson -> kNN5 features encode the same signal MACCS uses
    (confirms: MACCS is implicitly a neighbour lookup).
    Low Pearson -> kNN5 captures a different axis.
    """
    out: dict = {}
    for ref_tag in ("nb1183",):
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
                    f"shape mismatch: ref={ref.shape} vs "
                    f"self={mean_bag_oof.shape}"
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
    print(f"{TAG} -- SHALLOW residual-LGBM bag on top of nb1070, "
          f"kNN5 features ONLY ({4} cols), {len(RESID_SEEDS)} KFold seeds")
    print(f"          seeds  = {RESID_SEEDS}")
    print(f"          target = y_unb - nb1070_pred_oof")
    print(f"          features = [max_sim, top1_pec50, "
          f"wmean_pec50_k{K}, std_pec50_k{K}]  Morgan-{MORGAN_BITS} r={MORGAN_RADIUS}")
    print(f"          LGBM: max_depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)  [matches nb1183]")
    print("=" * 78)

    tr = load_train()
    te = load_test()
    n_train = len(tr)
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_train={n_train}  n_test={n_test}  n_unb={n_unb}")

    # ---- Anchor ----
    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF (run nb1070 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: {anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Morgan FPs ----
    print(f"\n[feat] computing Morgan-{MORGAN_BITS} (r={MORGAN_RADIUS}) "
          f"for train ({n_train}) + test ({n_test}) ...")
    t_fp = time.time()
    fp_tr = morgan_fp_batch(tr["smiles"].tolist(),
                            radius=MORGAN_RADIUS, n_bits=MORGAN_BITS)  # (4139, 2048) uint8
    fp_te_all = morgan_fp_batch(te["smiles"].tolist(),
                                radius=MORGAN_RADIUS, n_bits=MORGAN_BITS)  # (513, 2048) uint8
    fp_te_unb = fp_te_all[unb_idx]  # (253, 2048) uint8
    print(f"[feat] fp_tr {fp_tr.shape}  fp_te_unb {fp_te_unb.shape}  "
          f"wall_fp = {time.time() - t_fp:.1f}s")

    # ---- Tanimoto similarity matrix ----
    print(f"[feat] Tanimoto similarities (253, {n_train}) ...")
    t_sim = time.time()
    sim = _tanimoto_matrix(fp_te_unb, fp_tr)  # (253, 4139) float32
    print(f"[feat] sim shape = {sim.shape}  dtype={sim.dtype}  "
          f"wall_sim = {time.time() - t_sim:.1f}s")
    print(f"[feat] sim row-max stats: mean={sim.max(axis=1).mean():.4f}  "
          f"median={float(np.median(sim.max(axis=1))):.4f}  "
          f"min={sim.max(axis=1).min():.4f}  max={sim.max(axis=1).max():.4f}")

    # ---- kNN5 features ----
    y_train = tr["pec50"].to_numpy(dtype=np.float64)
    if np.isnan(y_train).any():
        raise ValueError("train pec50 has NaNs (unexpected)")
    X_unb, feat_stats = _build_knn5_features(sim, y_train, k=K)
    print(f"[feat] kNN5 feature matrix shape = {X_unb.shape}  (4 cols)")
    for name, s in feat_stats.items():
        print(f"        {name}: mean={s['mean']:+.4f}  std={s['std']:.4f}  "
              f"min={s['min']:+.4f}  max={s['max']:+.4f}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow, kNN5 4-col)")
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
    print(f"   nb1183 MACCS mean_bag  = {NB1183_MEAN_BAG_REF:.4f}  "
          f"(key comparator)")
    print(f"   nb1211 BoB mean        = {NB1211_BOB_MEAN_REF:.4f}")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_MEAN_BAG_REF - DECISION_MARGIN
    beats_nb1211 = rae_mean_bag < NB1211_BOB_MEAN_REF - DECISION_MARGIN
    near_nb1183 = abs(rae_mean_bag - NB1183_MEAN_BAG_REF) <= 0.01

    if beats_nb1211:
        verdict = ("KNN5_4COL_BEATS_NB1211_BOB_MEAN  "
                   "neighbour-lookup is the dominant residual axis")
    elif beats_nb1183:
        verdict = ("KNN5_4COL_BEATS_NB1183  "
                   "MACCS signal is dominated by implicit neighbour lookup")
    elif near_nb1183:
        verdict = ("KNN5_4COL_MATCHES_NB1183  "
                   "MACCS residual is largely a neighbour-lookup proxy")
    elif beats_nb1070:
        verdict = ("KNN5_4COL_BEATS_NB1070_BUT_NOT_NB1183  "
                   "neighbour stats carry partial residual signal")
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "KNN5_4COL_FLAT_NO_RESIDUAL_SIGNAL"
    else:
        verdict = "KNN5_4COL_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Orthogonality probe (vs nb1183 MACCS) ----
    print("\n" + "-" * 78)
    print("ORTHOGONALITY PROBE (corrected mean-bag OOF vs nb1183 MACCS)")
    print("-" * 78)
    ortho = _orthogonality_probe(mean_bag_oof)
    for k, v in ortho.items():
        if isinstance(v, float):
            print(f"   {k} = {v:+.4f}")
        else:
            print(f"   {k} = {v}")

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
        "feature_source": "knn5_4col",
        "feature_names": ["max_sim", "top1_pec50",
                          "wmean_pec50_k5", "std_pec50_k5"],
        "n_unb": n_unb,
        "n_train": n_train,
        "morgan_bits": MORGAN_BITS,
        "morgan_radius": MORGAN_RADIUS,
        "k": K,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "feat_stats": feat_stats,
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
        "delta_median_bag_vs_nb1070": rae_median_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_MEAN_BAG_REF,
        "delta_mean_bag_vs_nb1211": rae_mean_bag - NB1211_BOB_MEAN_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1211": bool(beats_nb1211),
        "near_nb1183_within_0p01": bool(near_nb1183),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "nb1211_bob_mean_ref": NB1211_BOB_MEAN_REF,
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
    for k in ("rae_anchor_nb1070", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb1070",
              "delta_mean_bag_vs_nb1183",
              "delta_mean_bag_vs_nb1211",
              "beats_nb1070", "beats_nb1183", "beats_nb1211",
              "near_nb1183_within_0p01",
              "verdict", "orthogonality_probe"):
        print(f"  {k}: {res.get(k)}")
