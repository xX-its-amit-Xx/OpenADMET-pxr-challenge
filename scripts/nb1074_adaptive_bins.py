"""nb1074 -- Per-bin rank-stretch using KMeans-adaptive bin boundaries.

Same recipe as nb1053 (per-bin `mu_b + s_b * (p - mu_b)` on te_nb1014 anchor,
honest 5-fold cross-fit on the 253 unblind), but the bin partition is data-
adaptive: a 1-D KMeans(n_clusters=5) clusters the train-fold predictions, and
the bin EDGES are placed at the midpoints between consecutive sorted cluster
centers.

Hypothesis: equal-frequency quantile bins force exactly 50/253 ~= 20% of
samples per bin even when the prediction density is heavily lopsided. nb1014
preds are heavily concentrated around ~5.0 (PXR mean) with a thin upper
tail; quantile bins burn boundary mass on the sparse tail. KMeans-adaptive
bins instead concentrate boundary mass where prediction density is HIGH
(the body), which is also where the per-bin stretch fit has the largest
sample support. The thin tails get a single wide bin each, which is the
right capacity match -- you can't reliably fit a separate `s` on n<10.

Risk: bin counts become uneven (some bins may have <5 samples in val),
which degenerates `fit_per_bin_stretch` into identity for those bins. We
fall back to s=1.0 / global mu in degenerate cases (already handled in
nb1053's helper). Verdict gate: beats nb1014 bagged honest CV (0.5930) by
>=0.005, AND beats nb1053 by any margin.

Procedure:
  1. Load te_nb1014.npy and 253 unblind anchor.
  2. 5-fold KFold(seed=42) on the 253:
       a. Run KMeans(n_clusters=5, seed=42, n_init=10) on train-fold
          predictions reshaped (N, 1).
       b. Sort cluster centers; bin edges = midpoints between consecutive
          sorted centers -> 4 internal edges -> 5 bins.
       c. Grid-fit (mu_b, s_b) on train preds in each bin (same as nb1053).
       d. Apply to val using TRAIN-fold edges (no leakage).
  3. Pooled cross-fit RAE on the 253.
  4. Deploy: fit (centers, edges, mu, s) once on all 253, apply to 513.

Outputs:
  data/processed/te_nb1074.npy
  data/processed/nb1074_summary.json
  submissions/nb1074_adaptive_bins.csv
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
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1074"
ANCHOR = "nb1014"
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42

NB1014_BAGGED_HONEST_RAE = 0.5930
NB1053_HONEST_RAE = None  # filled from nb1053_summary if available


def kmeans_edges(p: np.ndarray, k: int = N_BINS, seed: int = SEED) -> np.ndarray:
    """Cluster 1-D predictions into k groups; return k-1 sorted midpoint edges."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(p.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
    # Midpoints between consecutive sorted centers.
    edges = (centers[:-1] + centers[1:]) / 2.0
    return edges


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values to bins 0..N_BINS-1 given internal edges of length N_BINS-1."""
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid-fit (mu_b, s_b) per bin on train data."""
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
    print(f"{TAG} -- KMeans-adaptive-bin stretch on te_{ANCHOR}")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    print(f"[load] te_{ANCHOR} shape={preds_513.shape}  "
          f"mean={preds_513.mean():.3f}  std={preds_513.std():.3f}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] p_unb={p_unb.shape} y={y_unb.shape}")
    print(f"[load] truth_std={y_unb.std():.4f}  pred_std={p_unb.std():.4f}  "
          f"compression={p_unb.std() / y_unb.std():.3f}")

    in_rae_baseline = float(rae(y_unb, p_unb))
    print(f"\n[baseline] in_RAE(te_{ANCHOR} on 253) = {in_rae_baseline:.4f}")

    # Try to read nb1053 honest RAE for delta reporting.
    nb1053_rae = None
    nb1053_path = DATA_PROCESSED / "nb1053_summary.json"
    if nb1053_path.exists():
        try:
            with open(nb1053_path) as f:
                nb1053_rae = float(json.load(f).get("pooled_cross_fit_rae"))
        except Exception:
            nb1053_rae = None

    print("\n" + "-" * 78)
    print(f"HONEST CROSS-FIT  (KFold seed={SEED}, KMeans bins K={N_BINS})")
    print("-" * 78)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va, y_va = p_unb[va_loc], y_unb[va_loc]
        edges = kmeans_edges(p_tr, k=N_BINS, seed=SEED)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        pred_tr_full = apply_per_bin_stretch(p_tr, edges, mus, ss)
        rae_tr_global = float(rae(y_tr, pred_tr_full))
        pred_va = apply_per_bin_stretch(p_va, edges, mus, ss)
        oof[va_loc] = pred_va
        rae_va = float(rae(y_va, pred_va))
        bins_va = bin_assign(p_va, edges)
        bin_counts_va = [int((bins_va == b).sum()) for b in range(N_BINS)]
        bins_tr = bin_assign(p_tr, edges)
        bin_counts_tr = [int((bins_tr == b).sum()) for b in range(N_BINS)]
        folds.append({
            "fold": k,
            "edges": edges.tolist(),
            "mus": mus.tolist(),
            "ss": ss.tolist(),
            "train_rae": rae_tr_global,
            "val_rae": rae_va,
            "n_va": int(len(va_loc)),
            "bin_counts_tr": bin_counts_tr,
            "bin_counts_va": bin_counts_va,
        })
        ss_str = ",".join(f"{s:.2f}" for s in ss)
        print(f"   fold {k}: tr_RAE={rae_tr_global:.4f}  "
              f"va_RAE={rae_va:.4f}  ss=[{ss_str}]  "
              f"tr_counts={bin_counts_tr}  va_counts={bin_counts_va}")

    pooled = float(rae(y_unb, oof))
    print(f"\n[honest] pooled cross-fit RAE on 253 = {pooled:.4f}")

    # Deploy on all 253 -> apply to 513.
    print("\n" + "-" * 78)
    print("DEPLOY  (fit on all 253, apply to 513)")
    print("-" * 78)
    deploy_edges = kmeans_edges(p_unb, k=N_BINS, seed=SEED)
    deploy_mus, deploy_ss = fit_per_bin_stretch(p_unb, y_unb, deploy_edges)
    bins_unb = bin_assign(p_unb, deploy_edges)
    print("\n[diag] per-bin in-sample fit on 253 (deploy KMeans edges):")
    print(f"   bin  n    p_mean   y_mean   s_best   anchor_AE   stretched_AE")
    for b in range(N_BINS):
        mask = bins_unb == b
        n_b = int(mask.sum())
        if n_b == 0:
            print(f"   {b:>3d}    0    ----     ----     ----     --------   "
                  f"----------")
            continue
        p_b = p_unb[mask]
        y_b = y_unb[mask]
        anchor_ae = float(np.mean(np.abs(p_b - y_b)))
        stretched = deploy_mus[b] + deploy_ss[b] * (p_b - deploy_mus[b])
        stretched_ae = float(np.mean(np.abs(stretched - y_b)))
        print(f"   {b:>3d}  {n_b:>3d}  {p_b.mean():>6.3f}   "
              f"{y_b.mean():>6.3f}   {deploy_ss[b]:>5.2f}   "
              f"{anchor_ae:>8.4f}   {stretched_ae:>10.4f}")

    deploy_513 = apply_per_bin_stretch(preds_513, deploy_edges,
                                        deploy_mus, deploy_ss).astype(np.float32)
    in_rae_deploy = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
    print(f"\n   deploy edges = {deploy_edges.tolist()}")
    print(f"   deploy mus   = {[round(float(x), 3) for x in deploy_mus]}")
    print(f"   deploy ss    = {[round(float(x), 2) for x in deploy_ss]}")
    print(f"   in-sample 253 = {in_rae_deploy:.4f} (overfit lower bound)")
    print(f"   te(513) mean = {deploy_513.mean():.3f}  std={deploy_513.std():.3f}")
    print(f"   anchor(513) mean = {preds_513.mean():.3f}  "
          f"std={preds_513.std():.3f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_adaptive_bins.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_nb1014 = pooled - NB1014_BAGGED_HONEST_RAE
    if delta_vs_nb1014 <= -0.005:
        verdict_nb1014 = "BEATS_NB1014"
    elif abs(delta_vs_nb1014) < 0.005:
        verdict_nb1014 = "TIES_NB1014"
    else:
        verdict_nb1014 = "WORSE_THAN_NB1014"

    delta_vs_nb1053 = (pooled - nb1053_rae) if nb1053_rae is not None else None
    if nb1053_rae is None:
        verdict_nb1053 = "NB1053_NOT_FOUND"
    elif delta_vs_nb1053 < 0:
        verdict_nb1053 = "BEATS_NB1053"
    elif delta_vs_nb1053 == 0:
        verdict_nb1053 = "TIES_NB1053"
    else:
        verdict_nb1053 = "WORSE_THAN_NB1053"

    print(f"\n[verdict] vs nb1014 bagged ({NB1014_BAGGED_HONEST_RAE}): "
          f"delta={delta_vs_nb1014:+.4f}  -> {verdict_nb1014}")
    if nb1053_rae is not None:
        print(f"[verdict] vs nb1053 quantile-bins ({nb1053_rae:.4f}): "
              f"delta={delta_vs_nb1053:+.4f}  -> {verdict_nb1053}")
    else:
        print(f"[verdict] vs nb1053: summary not found")

    per_bin_s_mean = [float(np.mean([f["ss"][b] for f in folds]))
                      for b in range(N_BINS)]
    per_bin_s_std = [float(np.std([f["ss"][b] for f in folds]))
                     for b in range(N_BINS)]
    print(f"\n[bag] mean s per bin (folds) = "
          f"{[round(x, 3) for x in per_bin_s_mean]}")
    print(f"[bag] std  s per bin (folds) = "
          f"{[round(x, 3) for x in per_bin_s_std]}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "binning": "kmeans_midpoint_edges",
        "n_bins": N_BINS,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "in_rae_anchor_on_253": in_rae_baseline,
        "pooled_cross_fit_rae": pooled,
        "in_rae_deploy_on_253": in_rae_deploy,
        "nb1014_bagged_honest_rae": NB1014_BAGGED_HONEST_RAE,
        "nb1053_honest_rae": nb1053_rae,
        "delta_vs_nb1014_bagged": delta_vs_nb1014,
        "delta_vs_nb1053": delta_vs_nb1053,
        "verdict_vs_nb1014": verdict_nb1014,
        "verdict_vs_nb1053": verdict_nb1053,
        "deploy_edges": deploy_edges.tolist(),
        "deploy_mus": deploy_mus.tolist(),
        "deploy_ss": deploy_ss.tolist(),
        "per_bin_s_mean_across_folds": per_bin_s_mean,
        "per_bin_s_std_across_folds": per_bin_s_std,
        "folds": folds,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                = {ANCHOR}")
    print(f"   binning               = KMeans(K={N_BINS}) midpoint edges")
    print(f"   in_RAE anchor on 253  = {in_rae_baseline:.4f}")
    print(f"   pooled cross-fit RAE  = {pooled:.4f}")
    print(f"   in-sample (deploy)    = {in_rae_deploy:.4f}")
    print(f"   nb1014 bagged honest  = {NB1014_BAGGED_HONEST_RAE:.4f}")
    print(f"   nb1053 honest         = "
          f"{nb1053_rae if nb1053_rae is not None else 'n/a'}")
    print(f"   delta vs nb1014       = {delta_vs_nb1014:+.4f}  "
          f"-> {verdict_nb1014}")
    if nb1053_rae is not None:
        print(f"   delta vs nb1053       = {delta_vs_nb1053:+.4f}  "
              f"-> {verdict_nb1053}")
    print(f"   per-bin s (folds mean)= "
          f"{[round(x, 3) for x in per_bin_s_mean]}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("in_rae_anchor_on_253", "pooled_cross_fit_rae",
              "in_rae_deploy_on_253", "delta_vs_nb1014_bagged",
              "delta_vs_nb1053", "verdict_vs_nb1014", "verdict_vs_nb1053",
              "per_bin_s_mean_across_folds", "deploy_ss",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
