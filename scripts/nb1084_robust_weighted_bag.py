"""nb1084 -- Robust inverse-error weighted bag across (seed, fold) pairs.

nb1060 / nb1070 ran nb1053's per-quantile-bin stretch protocol across 5 KFold
seeds and combined per-seed OOF vectors via mean / median. Both treat all 25
(seed, fold) sub-models equally. But fold quality varies substantially: some
held-out folds are dominated by easy compounds (low val RAE), others by novel
scaffolds (high val RAE). nb1070's MEDIAN bag improves on MEAN by trimming
extremes, but does not reward higher-quality folds.

Hypothesis: weight each of the 25 (seed, fold) deploy-prediction vectors by
1 / val_RAE so that high-quality folds (lower val RAE) dominate the bag. This
should remove the noise contributed by pathological folds (e.g., the seed-7
high-RAE fold and grid-corner s=2.0 picks) while keeping the signal from the
strong folds.

Procedure:
  - For each seed s in {0, 1, 7, 42, 137}:
      * KFold(n=5, shuffle=True, random_state=s) on 253 unblind.
      * For each fold (tr, va):
          - Train-only quantile edges (N_BINS=5).
          - Train-only fit_per_bin_stretch grid scan over s in 0.80..2.00.
          - val_RAE = RAE on held-out fold (the fold-quality signal).
          - Deploy-vector for this (seed, fold): apply the train-fit (mus, ss)
            to ALL 513 test compounds.
          - Record OOF prediction on va_loc for cross-fit RAE accounting.
  - Weights w_{s,f} = 1 / val_RAE_{s,f}, normalised across all 25 pairs.
  - Weighted bag deploy_513 = sum_{s,f} w_{s,f} * deploy_vec_{s,f}.
  - Weighted OOF: for each held-out row i, only 5 (seed, fold) pairs cover it
    (one per seed). Use weights restricted to those 5 pairs, renormalised.
    Pool weighted OOF -> pooled cross-fit RAE.
  - Compare to nb1060 mean (0.5798) and nb1070 median.

Outputs:
  data/processed/te_nb1084.npy
  data/processed/nb1084_summary.json
  submissions/nb1084_robust_weighted_bag.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1084"
ANCHOR = "nb1014"
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

# Reference numbers.
NB1053_SEED42_RAE = 0.5780
NB1060_BAG_MEAN_RAE = 0.5798
NB1070_BAG_MEDIAN_RAE = 0.5798  # honest cross-fit reference for median bag
NB1014_BAGGED_HONEST_RAE = 0.5930


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid-fit (mu_b, s_b) per bin on train data -- nb1053 verbatim."""
    bins_tr = bin_assign(p_train, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        n_b = int(mask.sum())
        if n_b < 2:
            mus[b] = float(p_train.mean())
            ss[b] = 1.0
            continue
        mu_b = float(p_train[mask].mean())
        mus[b] = mu_b
        y_b = y_train[mask]
        p_b = p_train[mask]
        best_s, best_r = 1.0, float("inf")
        for s in STRETCH_GRID:
            stretched = mu_b + s * (p_b - mu_b)
            r = float(rae(y_b, stretched))
            if r < best_r:
                best_r = r
                best_s = float(s)
        ss[b] = best_s
    return mus, ss


def apply_per_bin_stretch(p: np.ndarray, edges: np.ndarray,
                          mus: np.ndarray, ss: np.ndarray) -> np.ndarray:
    bins = bin_assign(p, edges)
    out = np.empty_like(p, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins == b
        if not mask.any():
            continue
        out[mask] = mus[b] + ss[b] * (p[mask] - mus[b])
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- inverse-error WEIGHTED bag of (seed, fold) sub-models")
    print(f"          seeds = {SEEDS}   folds = {N_FOLDS}   "
          f"total submodels = {len(SEEDS) * N_FOLDS}")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    n_513 = preds_513.shape[0]
    n_submodels = len(SEEDS) * N_FOLDS
    print(f"[load] te_{ANCHOR}.npy shape={preds_513.shape}  "
          f"p_unb shape={p_unb.shape}  y shape={y_unb.shape}")

    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"[baseline] in_RAE(te_{ANCHOR} on 253 unblind) = "
          f"{in_rae_anchor:.4f}")

    # =================================================================
    # Build the (seed, fold) sub-model bank.
    # For each (s, f): fit on tr_loc rows of p_unb/y_unb, then predict on:
    #   (a) val rows -> oof prediction & val_RAE (quality signal)
    #   (b) ALL 513 rows -> deploy vector for this sub-model
    # =================================================================
    print("\n" + "-" * 78)
    print("BUILD (seed, fold) BANK")
    print("-" * 78)
    deploy_bank = np.zeros((n_submodels, n_513), dtype=np.float64)
    val_rae_bank = np.zeros(n_submodels, dtype=np.float64)
    # OOF: each row is covered by one fold per seed -> 5 predictions per row.
    oof_per_seed = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    # For each row, which sub-model index covered it (per seed).
    cover_idx = np.zeros((len(SEEDS), n_unb), dtype=np.int64)
    submodel_records: list[dict] = []
    k = 0
    for si, seed in enumerate(SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fi, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
            p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
            p_va, y_va = p_unb[va_loc], y_unb[va_loc]
            qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
            edges = np.quantile(p_tr, qs)
            mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
            # held-out fold prediction
            va_pred = apply_per_bin_stretch(p_va, edges, mus, ss)
            v_rae = float(rae(y_va, va_pred))
            oof_per_seed[si, va_loc] = va_pred
            cover_idx[si, va_loc] = k
            val_rae_bank[k] = v_rae
            # deploy-vector for this sub-model on the full 513
            deploy_bank[k] = apply_per_bin_stretch(preds_513, edges, mus, ss)
            submodel_records.append({
                "submodel_idx": k,
                "seed": seed,
                "fold": fi,
                "val_rae": v_rae,
                "ss": ss.tolist(),
                "n_val": int(len(va_loc)),
            })
            k += 1
        print(f"   seed {seed:>3d}: val_RAEs = "
              + ", ".join(f"{val_rae_bank[si * N_FOLDS + f]:.4f}"
                          for f in range(N_FOLDS)))

    val_rae_arr = val_rae_bank
    print(f"\n[bank] val_RAE  mean={val_rae_arr.mean():.4f}  "
          f"std={val_rae_arr.std():.4f}  "
          f"min={val_rae_arr.min():.4f}  max={val_rae_arr.max():.4f}")

    # =================================================================
    # Inverse-error weights across all 25 sub-models (for deploy).
    # =================================================================
    raw_w = 1.0 / np.clip(val_rae_arr, 1e-6, None)
    w = raw_w / raw_w.sum()
    print(f"[weights] inverse-error weights (25 sub-models): "
          f"min={w.min():.4f}  max={w.max():.4f}  "
          f"effective_n = {1.0 / (w ** 2).sum():.2f}")

    # =================================================================
    # Weighted OOF: each unblind row is covered by 5 sub-models (1 per seed).
    # Restrict the global weights to those 5 covering sub-models, renormalise,
    # and form weighted prediction at that row.
    # =================================================================
    weighted_oof = np.zeros(n_unb, dtype=np.float64)
    # Per-row covering sub-model indices: shape (5, n_unb)
    for i in range(n_unb):
        idxs = cover_idx[:, i]                    # (5,)  sub-model indices
        ws = raw_w[idxs]                          # (5,)  raw weights
        ws = ws / ws.sum()
        # predictions from those 5 covering sub-models at row i
        preds_i = oof_per_seed[np.arange(len(SEEDS)), i]   # (5,)
        weighted_oof[i] = float(np.dot(ws, preds_i))
    pooled_weighted_rae = float(rae(y_unb, weighted_oof))

    # Reference: unweighted (mean across seeds) OOF for sanity vs nb1060.
    unweighted_oof_mean = oof_per_seed.mean(axis=0)
    pooled_unweighted_rae = float(rae(y_unb, unweighted_oof_mean))
    unweighted_oof_median = np.median(oof_per_seed, axis=0)
    pooled_median_rae = float(rae(y_unb, unweighted_oof_median))

    print("\n" + "-" * 78)
    print("WEIGHTED CROSS-FIT RAE")
    print("-" * 78)
    print(f"   pooled WEIGHTED (1/val_RAE)   = {pooled_weighted_rae:.4f}")
    print(f"   pooled UNWEIGHTED MEAN  (ref) = {pooled_unweighted_rae:.4f}  "
          f"(nb1060 = {NB1060_BAG_MEAN_RAE:.4f})")
    print(f"   pooled UNWEIGHTED MEDIAN(ref) = {pooled_median_rae:.4f}  "
          f"(nb1070 = {NB1070_BAG_MEDIAN_RAE:.4f})")

    # =================================================================
    # Weighted deploy_513.
    # =================================================================
    deploy_513 = (deploy_bank * w[:, None]).sum(axis=0).astype(np.float32)
    in_rae_deploy = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))

    # Also report unweighted-mean deploy for sanity.
    deploy_513_unweighted = deploy_bank.mean(axis=0).astype(np.float32)
    in_rae_deploy_unweighted = float(rae(
        y_unb, deploy_513_unweighted[unb_idx].astype(np.float64)))

    print("\n" + "-" * 78)
    print("DEPLOY")
    print("-" * 78)
    print(f"   deploy_513 WEIGHTED   mean={deploy_513.mean():.3f}  "
          f"std={deploy_513.std():.3f}  in-sample 253={in_rae_deploy:.4f}")
    print(f"   deploy_513 UNWEIGHTED mean={deploy_513_unweighted.mean():.3f}  "
          f"std={deploy_513_unweighted.std():.3f}  "
          f"in-sample 253={in_rae_deploy_unweighted:.4f}")
    print(f"   anchor te(513)        mean={preds_513.mean():.3f}  "
          f"std={preds_513.std():.3f}")

    # ---- Save deploy + submission ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_robust_weighted_bag.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    # ---- Verdict ----
    delta_vs_nb1060 = pooled_weighted_rae - NB1060_BAG_MEAN_RAE
    delta_vs_nb1070 = pooled_weighted_rae - NB1070_BAG_MEDIAN_RAE
    delta_vs_nb1053 = pooled_weighted_rae - NB1053_SEED42_RAE
    delta_vs_nb1014 = pooled_weighted_rae - NB1014_BAGGED_HONEST_RAE
    beats_nb1060 = pooled_weighted_rae < NB1060_BAG_MEAN_RAE
    beats_nb1070 = pooled_weighted_rae < NB1070_BAG_MEDIAN_RAE
    if delta_vs_nb1060 <= -0.001:
        verdict = "WEIGHTED_HELPS"
    elif abs(delta_vs_nb1060) < 0.001:
        verdict = "WEIGHTED_TIES_MEAN"
    else:
        verdict = "MEAN_OPTIMAL"

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1053 (seed=42)               = {NB1053_SEED42_RAE:.4f}")
    print(f"   nb1060 mean-bagged OOF         = {NB1060_BAG_MEAN_RAE:.4f}")
    print(f"   nb1070 median-bagged OOF       = {NB1070_BAG_MEDIAN_RAE:.4f}")
    print(f"   nb1084 WEIGHTED-bagged OOF     = {pooled_weighted_rae:.4f}  "
          f"(delta vs nb1060 = {delta_vs_nb1060:+.4f},  "
          f"delta vs nb1070 = {delta_vs_nb1070:+.4f})")
    print(f"   nb1014 bagged honest CV        = {NB1014_BAGGED_HONEST_RAE:.4f}  "
          f"(delta = {delta_vs_nb1014:+.4f})")
    print(f"   beats_nb1060                   = {beats_nb1060}")
    print(f"   beats_nb1070                   = {beats_nb1070}")
    print(f"   verdict                        = {verdict}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "n_submodels": n_submodels,
        "stretch_grid": STRETCH_GRID,
        "in_rae_anchor_on_253": in_rae_anchor,
        "val_rae_bank": val_rae_arr.tolist(),
        "val_rae_mean": float(val_rae_arr.mean()),
        "val_rae_std": float(val_rae_arr.std()),
        "val_rae_min": float(val_rae_arr.min()),
        "val_rae_max": float(val_rae_arr.max()),
        "weights": w.tolist(),
        "weights_min": float(w.min()),
        "weights_max": float(w.max()),
        "effective_n_submodels": float(1.0 / (w ** 2).sum()),
        "pooled_weighted_rae": pooled_weighted_rae,
        "pooled_unweighted_mean_rae": pooled_unweighted_rae,
        "pooled_unweighted_median_rae": pooled_median_rae,
        "in_rae_deploy_weighted_on_253": in_rae_deploy,
        "in_rae_deploy_unweighted_on_253": in_rae_deploy_unweighted,
        "deploy_te_weighted_mean": float(deploy_513.mean()),
        "deploy_te_weighted_std": float(deploy_513.std()),
        "deploy_te_unweighted_mean": float(deploy_513_unweighted.mean()),
        "deploy_te_unweighted_std": float(deploy_513_unweighted.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "delta_vs_nb1060": delta_vs_nb1060,
        "delta_vs_nb1070": delta_vs_nb1070,
        "delta_vs_nb1053": delta_vs_nb1053,
        "delta_vs_nb1014_bagged": delta_vs_nb1014,
        "beats_nb1060": bool(beats_nb1060),
        "beats_nb1070": bool(beats_nb1070),
        "verdict": verdict,
        "nb1053_seed42_rae": NB1053_SEED42_RAE,
        "nb1060_bag_mean_rae": NB1060_BAG_MEAN_RAE,
        "nb1070_bag_median_rae": NB1070_BAG_MEDIAN_RAE,
        "nb1014_bagged_honest_rae": NB1014_BAGGED_HONEST_RAE,
        "plain_submission": str(plain),
        "submodel_records": submodel_records,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("val_rae_mean", "val_rae_std", "val_rae_min", "val_rae_max",
              "effective_n_submodels",
              "pooled_weighted_rae", "pooled_unweighted_mean_rae",
              "pooled_unweighted_median_rae",
              "delta_vs_nb1060", "delta_vs_nb1070",
              "beats_nb1060", "beats_nb1070", "verdict",
              "in_rae_deploy_weighted_on_253", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
