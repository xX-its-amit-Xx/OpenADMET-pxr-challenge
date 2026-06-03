"""nb1071 -- 15-seed extension of the nb1060 per-quantile-bin stretch bag.

nb1060 (5 seeds {0, 1, 7, 42, 137}) gave per-seed RAE mean=0.5814 std=0.0025
on the 253 unblind, i.e. SE(mean) = std/sqrt(5) ~= 0.0011. With only 5 seeds
the uncertainty on the mean cross-fit RAE is still wide relative to the
deploy decision band.

Hypothesis: extending to 15 KFold seeds
{0, 1, 7, 13, 17, 23, 31, 42, 55, 77, 99, 137, 314, 1024, 1729} should:
  1. Shrink SE(mean) from ~0.0011 to ~0.0006 (sqrt(5/15) = 0.577x), giving a
     tighter LB band for deploy confidence.
  2. Confirm the per-seed RAE distribution is well-characterized -- if mean
     is stable across 5 -> 15 seeds (delta < 1*SE), the nb1060 estimate is
     trustworthy.
  3. Provide a more precise bagged-OOF RAE for the row-level mean across
     more diverse fold partitions.

Protocol is byte-identical to nb1060 except for the seed list. Deploy is NOT
refit -- the deploy te_nb1053.npy from nb1053 remains the deploy artifact.
This is a precision-of-estimate experiment.

Outputs:
  data/processed/nb1071_summary.json
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
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1071"
ANCHOR = "nb1014"
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 13, 17, 23, 31, 42, 55, 77, 99, 137, 314, 1024, 1729]

# nb1060 reference (5-seed pooled cross-fit RAE)
NB1060_PER_SEED_MEAN = 0.5814491511913484
NB1060_PER_SEED_STD = 0.002453094558134132
NB1060_BAG_RAE = 0.5797539000446441
NB1053_SEED42_RAE = 0.5780


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin indices 0..N_BINS-1 from internal edges (length N_BINS-1)."""
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid-fit (mu_b, s_b) per bin on train data -- nb1053/nb1060 verbatim."""
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
    per_fold_ss = []
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
    print(f"{TAG} -- 15-seed bag of nb1053 per-quantile-bin stretch")
    print(f"          seeds = {SEEDS}")
    print("=" * 78)

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
    print("PER-SEED HONEST CROSS-FIT (15 seeds)")
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
        print(f"   seed {seed:>5d}: pooled_RAE = {pooled:.4f}   "
              f"ss_mean=[{ss_mean_str}]")

    per_seed_rae_arr = np.array(per_seed_rae)
    n_seeds = len(SEEDS)
    per_seed_mean = float(per_seed_rae_arr.mean())
    # sample std (ddof=1) for SE computation
    per_seed_std = float(per_seed_rae_arr.std(ddof=1))
    per_seed_std_pop = float(per_seed_rae_arr.std(ddof=0))
    per_seed_min = float(per_seed_rae_arr.min())
    per_seed_max = float(per_seed_rae_arr.max())
    se_mean = per_seed_std / np.sqrt(n_seeds)
    print(f"\n[per-seed] RAE  mean={per_seed_mean:.4f}  "
          f"std(ddof=1)={per_seed_std:.4f}  "
          f"SE(mean)={se_mean:.4f}")
    print(f"           min={per_seed_min:.4f}  max={per_seed_max:.4f}  "
          f"range={per_seed_max - per_seed_min:.4f}")

    # ---- Bag across seeds at row level ----
    bagged_oof = oof_stack.mean(axis=0)
    bag_rae = float(rae(y_unb, bagged_oof))
    print(f"[bag]      bagged_oof RAE (mean of {n_seeds} OOFs) = "
          f"{bag_rae:.4f}")

    # ---- Save bagged OOF for downstream use ----
    bag_path = DATA_PROCESSED / f"{TAG}_bagged_oof_unb.npy"
    np.save(bag_path, bagged_oof)
    print(f"[save] {bag_path}")

    # ---- Comparisons ----
    delta_mean_vs_nb1060 = per_seed_mean - NB1060_PER_SEED_MEAN
    se_mean_nb1060 = NB1060_PER_SEED_STD / np.sqrt(5)
    se_shrinkage = se_mean / se_mean_nb1060
    delta_bag_vs_nb1060_bag = bag_rae - NB1060_BAG_RAE
    delta_vs_nb1053 = bag_rae - NB1053_SEED42_RAE
    beats_nb1060 = bag_rae < NB1060_BAG_RAE

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1053 (seed=42)                = {NB1053_SEED42_RAE:.4f}")
    print(f"   nb1060 per-seed mean (n=5)      = {NB1060_PER_SEED_MEAN:.4f}  "
          f"SE={se_mean_nb1060:.4f}")
    print(f"   nb1071 per-seed mean (n=15)     = {per_seed_mean:.4f}  "
          f"SE={se_mean:.4f}")
    print(f"     -> delta vs nb1060 mean       = "
          f"{delta_mean_vs_nb1060:+.4f}  "
          f"({delta_mean_vs_nb1060 / se_mean:+.2f} sigma_15)")
    print(f"     -> SE shrinkage ratio         = "
          f"{se_shrinkage:.3f}  (target ~0.577)")
    print(f"   nb1060 bagged OOF               = {NB1060_BAG_RAE:.4f}")
    print(f"   nb1071 bagged OOF (15 seeds)    = {bag_rae:.4f}  "
          f"(delta = {delta_bag_vs_nb1060_bag:+.4f})")
    print(f"   beats_nb1060                    = {beats_nb1060}")
    print(f"   95%% CI(mean) ~= {per_seed_mean:.4f} +/- "
          f"{1.96 * se_mean:.4f}  "
          f"-> [{per_seed_mean - 1.96 * se_mean:.4f}, "
          f"{per_seed_mean + 1.96 * se_mean:.4f}]")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "n_seeds": n_seeds,
        "stretch_grid": STRETCH_GRID,
        "in_rae_anchor_on_253": in_rae_anchor,
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_std": per_seed_std,
        "per_seed_rae_std_pop": per_seed_std_pop,
        "per_seed_rae_se_mean": se_mean,
        "per_seed_rae_min": per_seed_min,
        "per_seed_rae_max": per_seed_max,
        "bag_mean_rae": bag_rae,
        "delta_bag_vs_nb1060_bag": delta_bag_vs_nb1060_bag,
        "delta_mean_vs_nb1060_mean": delta_mean_vs_nb1060,
        "se_shrinkage_ratio_vs_nb1060": se_shrinkage,
        "delta_vs_nb1053": delta_vs_nb1053,
        "beats_nb1060": bool(beats_nb1060),
        "nb1053_seed42_rae": NB1053_SEED42_RAE,
        "nb1060_per_seed_mean": NB1060_PER_SEED_MEAN,
        "nb1060_per_seed_std": NB1060_PER_SEED_STD,
        "nb1060_bag_rae": NB1060_BAG_RAE,
        "ci95_lo": per_seed_mean - 1.96 * se_mean,
        "ci95_hi": per_seed_mean + 1.96 * se_mean,
        "per_seed_ss_means_across_folds": per_seed_ss_means,
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
              "per_seed_rae_se_mean", "bag_mean_rae",
              "delta_bag_vs_nb1060_bag", "beats_nb1060",
              "ci95_lo", "ci95_hi"):
        print(f"  {k}: {res.get(k)}")
