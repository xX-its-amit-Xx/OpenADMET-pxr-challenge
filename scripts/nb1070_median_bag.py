"""nb1070 -- Median bag of nb1060 (per-quantile stretch on nb1014, 5 seeds).

nb1060 bagged the per-quantile-bin stretch protocol of nb1053 across 5 KFold
seeds {0, 1, 7, 42, 137} using a MEAN over per-seed OOF predictions. The
per-seed pooled RAE spread is small (mean 0.5814, std 0.0025) but seed 7
spikes to 0.5852 and individual folds occasionally pick stretch grid corners
(s=2.0) that pull the mean. The mean-bagged OOF landed at 0.5798 -- worse
than the lucky seed=42 baseline (0.5780) and not faithful to the typical
seed's behavior.

Hypothesis: replacing mean with MEDIAN across the 5 per-seed OOF predictions
at each compound should be robust to seed-level outliers (fold-4 spike,
grid-corner picks). The median is a more conservative estimator that
discards the two most extreme per-seed predictions and reports the central
one. Expectation: pooled RAE in the 0.578-0.580 band with tighter
distributional behaviour.

Procedure (identical to nb1060 for cross-fit):
  - For each seed s in {0, 1, 7, 42, 137}:
      * KFold(n=5, shuffle=True, random_state=s) on 253 unblind.
      * Train-only quantile edges (5 bins, N_BINS=5).
      * Train-only fit_per_bin_stretch grid scan over s in {0.80..2.00 step 0.05}.
      * Apply per-bin (mu_b, s_b) to held-out fold rows.
      * Accumulate the OOF prediction vector for seed s.
  - Stack per-seed OOFs -> (5, 253), MEDIAN across seed axis -> bagged_oof.
  - Pooled cross-fit RAE on the 253 unblind.
Deploy:
  - For each seed s: fit per-bin (mu, s_b) on ALL 253, apply to 513.
  - Stack per-seed deploy_513 -> (5, 513), MEDIAN across seed axis ->
    deploy_513.
  - Save te_nb1070.npy + submissions/nb1070_median_bag.csv.

Outputs:
  data/processed/te_nb1070.npy
  data/processed/nb1070_summary.json
  submissions/nb1070_median_bag.csv
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

TAG = "nb1070"
ANCHOR = "nb1014"
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

# Reference numbers.
NB1053_SEED42_RAE = 0.5780
NB1060_BAG_MEAN_RAE = 0.5798
NB1014_BAGGED_HONEST_RAE = 0.5930


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin indices 0..N_BINS-1 from internal edges (length N_BINS-1)."""
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


def run_one_seed(seed: int, p_unb: np.ndarray, y_unb: np.ndarray
                 ) -> tuple[float, np.ndarray, list[list[float]]]:
    """Run nb1053's honest 5-fold cross-fit once for KFold(seed)."""
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    per_fold_ss: list[list[float]] = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va = p_unb[va_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        oof[va_loc] = apply_per_bin_stretch(p_va, edges, mus, ss)
        per_fold_ss.append(ss.tolist())
    pooled = float(rae(y_unb, oof))
    return pooled, oof, per_fold_ss


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MEDIAN bag of nb1053 per-quantile-bin stretch across "
          f"{len(SEEDS)} KFold seeds")
    print(f"          seeds = {SEEDS}")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] te_{ANCHOR}.npy shape={preds_513.shape}  "
          f"p_unb shape={p_unb.shape}  y shape={y_unb.shape}")

    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"[baseline] in_RAE(te_{ANCHOR} on 253 unblind) = "
          f"{in_rae_anchor:.4f}")

    # ---- Per-seed honest cross-fit ----
    print("\n" + "-" * 78)
    print("PER-SEED HONEST CROSS-FIT")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_ss_means: list[list[float]] = []
    seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, per_fold_ss = run_one_seed(seed, p_unb, y_unb)
        oof_stack[i] = oof
        per_seed_rae.append(pooled)
        ss_arr = np.array(per_fold_ss)  # (N_FOLDS, N_BINS)
        ss_mean = ss_arr.mean(axis=0).tolist()
        ss_std = ss_arr.std(axis=0).tolist()
        per_seed_ss_means.append(ss_mean)
        seed_records.append({
            "seed": seed,
            "pooled_rae": pooled,
            "per_fold_ss": per_fold_ss,
            "fold_ss_mean": ss_mean,
            "fold_ss_std": ss_std,
        })
        ss_mean_str = ",".join(f"{x:.2f}" for x in ss_mean)
        print(f"   seed {seed:>3d}: pooled_RAE = {pooled:.4f}   "
              f"ss_mean=[{ss_mean_str}]")

    per_seed_rae_arr = np.array(per_seed_rae)
    per_seed_mean = float(per_seed_rae_arr.mean())
    per_seed_std = float(per_seed_rae_arr.std())
    per_seed_min = float(per_seed_rae_arr.min())
    per_seed_max = float(per_seed_rae_arr.max())
    print(f"\n[per-seed] RAE  mean={per_seed_mean:.4f}  "
          f"std={per_seed_std:.4f}  "
          f"min={per_seed_min:.4f}  max={per_seed_max:.4f}")

    # ---- MEDIAN bag across seeds at row level ----
    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"\n[bag] MEDIAN  bagged_oof RAE (median of {len(SEEDS)} OOFs) = "
          f"{bag_median_rae:.4f}")
    print(f"[bag] MEAN    bagged_oof RAE (mean   of {len(SEEDS)} OOFs) = "
          f"{bag_mean_rae:.4f}  (nb1060 reference 0.5798)")

    # Per-row dispersion across seeds: median - mean gap, MAD, etc.
    row_std = oof_stack.std(axis=0)
    row_mad = np.median(np.abs(oof_stack - np.median(oof_stack, axis=0,
                                                     keepdims=True)),
                        axis=0)
    print(f"[disp] per-row across-seed std:  mean={row_std.mean():.4f}  "
          f"max={row_std.max():.4f}")
    print(f"[disp] per-row across-seed MAD:  mean={row_mad.mean():.4f}  "
          f"max={row_mad.max():.4f}")
    median_minus_mean = float(np.mean(np.abs(bagged_median_oof
                                              - bagged_mean_oof)))
    print(f"[disp] mean |median - mean| per row = {median_minus_mean:.4f}")

    # =================================================================
    # Deploy: for each seed s, fit per-bin (mu, s_b) on ALL 253, apply to
    # all 513. Median across the 5 deploy_513 predictions.
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (per-seed fit on all 253, median-bag the 513 predictions)")
    print("-" * 78)
    deploy_stack = np.zeros((len(SEEDS), preds_513.shape[0]),
                            dtype=np.float64)
    per_seed_deploy_ss: list[list[float]] = []
    for i, seed in enumerate(SEEDS):
        # Same protocol but only ONE fit on all 253 -- seed only affects the
        # random_state of KFold; for the "all 253" fit there is no fold split.
        # Hence per-seed deploy fits are deterministic except where they
        # would differ via downstream stochastic choices. Here the protocol
        # is fully deterministic in the all-253 fit, so we add a tiny per-seed
        # quantile-edge jitter: instead, we deploy via the MEAN of the
        # per-fold (mu, s) found inside that seed's KFold, applied with the
        # all-253 deploy edges. This matches nb1060's seed-bag philosophy.
        rec = seed_records[i]
        per_fold_ss_arr = np.array(rec["per_fold_ss"])  # (5, N_BINS)
        ss_seed = per_fold_ss_arr.mean(axis=0)
        # mus: fit once per seed on all 253 using deploy edges (deterministic
        # in p_unb, identical across seeds), but to keep per-seed variation
        # in deploy we re-derive mus from the all-253 fit with the seed-mean
        # ss applied. The mus depend only on bin membership of p_unb, which
        # is seed-invariant, so per-seed deploy variance comes solely from
        # ss_seed.
        qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges_all = np.quantile(p_unb, qs_all)
        mus_all, _ = fit_per_bin_stretch(p_unb, y_unb, edges_all)
        deploy_seed_513 = apply_per_bin_stretch(preds_513, edges_all,
                                                mus_all, ss_seed)
        deploy_stack[i] = deploy_seed_513
        per_seed_deploy_ss.append(ss_seed.tolist())
        ss_str = ",".join(f"{x:.2f}" for x in ss_seed)
        print(f"   seed {seed:>3d}: deploy ss=[{ss_str}]  "
              f"deploy_513 mean={deploy_seed_513.mean():.3f}  "
              f"std={deploy_seed_513.std():.3f}")

    deploy_513_median = np.median(deploy_stack, axis=0).astype(np.float32)
    deploy_513_mean = deploy_stack.mean(axis=0).astype(np.float32)
    in_rae_deploy_median = float(rae(y_unb,
                                      deploy_513_median[unb_idx].astype(
                                          np.float64)))
    in_rae_deploy_mean = float(rae(y_unb,
                                    deploy_513_mean[unb_idx].astype(
                                        np.float64)))
    print(f"\n   deploy_513 MEDIAN-bag  mean={deploy_513_median.mean():.3f}  "
          f"std={deploy_513_median.std():.3f}  "
          f"in-sample 253={in_rae_deploy_median:.4f}")
    print(f"   deploy_513 MEAN-bag    mean={deploy_513_mean.mean():.3f}  "
          f"std={deploy_513_mean.std():.3f}  "
          f"in-sample 253={in_rae_deploy_mean:.4f}")
    print(f"   anchor te(513)         mean={preds_513.mean():.3f}  "
          f"std={preds_513.std():.3f}")

    # ---- Save deploy + submission (median is the headline) ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513_median)
    plain = SUBMISSIONS / f"{TAG}_median_bag.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513_median,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    # ---- Comparisons ----
    delta_vs_nb1053 = bag_median_rae - NB1053_SEED42_RAE
    delta_vs_nb1060_mean = bag_median_rae - NB1060_BAG_MEAN_RAE
    delta_vs_nb1014 = bag_median_rae - NB1014_BAGGED_HONEST_RAE
    beats_nb1053 = bag_median_rae < NB1053_SEED42_RAE
    beats_nb1060 = bag_median_rae < NB1060_BAG_MEAN_RAE
    if delta_vs_nb1060_mean <= -0.001:
        verdict = "MEDIAN_HELPS"
    elif abs(delta_vs_nb1060_mean) < 0.001:
        verdict = "MEDIAN_TIES_MEAN"
    else:
        verdict = "MEAN_OPTIMAL"

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1053 (seed=42)               = {NB1053_SEED42_RAE:.4f}")
    print(f"   nb1060 mean-bagged OOF         = {NB1060_BAG_MEAN_RAE:.4f}")
    print(f"   nb1070 per-seed mean RAE       = {per_seed_mean:.4f}  "
          f"std={per_seed_std:.4f}")
    print(f"   nb1070 MEDIAN-bagged OOF       = {bag_median_rae:.4f}  "
          f"(delta vs nb1053 = {delta_vs_nb1053:+.4f},  "
          f"delta vs nb1060 mean = {delta_vs_nb1060_mean:+.4f})")
    print(f"   nb1070 MEAN-bagged OOF (recap) = {bag_mean_rae:.4f}")
    print(f"   nb1014 bagged honest CV        = {NB1014_BAGGED_HONEST_RAE:.4f}  "
          f"(delta = {delta_vs_nb1014:+.4f})")
    print(f"   beats_nb1053                   = {beats_nb1053}")
    print(f"   beats_nb1060                   = {beats_nb1060}")
    print(f"   verdict                        = {verdict}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stretch_grid": STRETCH_GRID,
        "in_rae_anchor_on_253": in_rae_anchor,
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_std": per_seed_std,
        "per_seed_rae_min": per_seed_min,
        "per_seed_rae_max": per_seed_max,
        "bag_median_rae": bag_median_rae,
        "bag_mean_rae": bag_mean_rae,
        "delta_vs_nb1053": delta_vs_nb1053,
        "delta_vs_nb1060_mean": delta_vs_nb1060_mean,
        "delta_vs_nb1014_bagged": delta_vs_nb1014,
        "beats_nb1053": bool(beats_nb1053),
        "beats_nb1060": bool(beats_nb1060),
        "verdict": verdict,
        "row_across_seed_std_mean": float(row_std.mean()),
        "row_across_seed_std_max": float(row_std.max()),
        "row_across_seed_mad_mean": float(row_mad.mean()),
        "row_across_seed_mad_max": float(row_mad.max()),
        "mean_abs_median_minus_mean": median_minus_mean,
        "per_seed_ss_means_across_folds": per_seed_ss_means,
        "per_seed_deploy_ss": per_seed_deploy_ss,
        "in_rae_deploy_median_on_253": in_rae_deploy_median,
        "in_rae_deploy_mean_on_253": in_rae_deploy_mean,
        "deploy_te_median_mean": float(deploy_513_median.mean()),
        "deploy_te_median_std": float(deploy_513_median.std()),
        "deploy_te_mean_mean": float(deploy_513_mean.mean()),
        "deploy_te_mean_std": float(deploy_513_mean.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "nb1053_seed42_rae": NB1053_SEED42_RAE,
        "nb1060_bag_mean_rae": NB1060_BAG_MEAN_RAE,
        "nb1014_bagged_honest_rae": NB1014_BAGGED_HONEST_RAE,
        "plain_submission": str(plain),
        "seed_records": seed_records,
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
    for k in ("per_seed_rae", "per_seed_rae_mean", "per_seed_rae_std",
              "bag_median_rae", "bag_mean_rae",
              "delta_vs_nb1053", "delta_vs_nb1060_mean",
              "beats_nb1053", "beats_nb1060", "verdict",
              "in_rae_deploy_median_on_253", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
