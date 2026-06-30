"""nb1194 -- Per-quantile rank stretch on nb1181 (cycle 19 protocol on new anchor).

HYPOTHESIS:
  nb1181 is a blend of three residual-corrected predictions; each component
  has its own variance-decompression baked in. But the BLEND step (1/3 mean)
  re-compresses variance (averaging shrinks toward mean). A per-quantile
  stretch on the blended OOF may recover some variance.

PROTOCOL (cycle 19 nb1134 protocol verbatim on new anchor):
  1. Anchor = nb1181_mean_oof.npy (253 rows, RAE 0.5566).
  2. Per-seed in {0,1,7,42,137}, KFold(5, shuffle, random_state=seed):
       * Train-only quantile edges (N_BINS=5).
       * Train-only fit_per_bin_stretch grid (s in {0.80..2.00 step 0.05}).
       * Apply per-bin (mu_b, s_b) to held-out fold rows.
  3. Stack per-seed OOF -> (5, 253), MEAN-bag AND MEDIAN-bag.
  4. Per-bin stretch factor consistency check (mean across seeds).
  5. Verdict at 0.003 margin vs nb1181 anchor RAE.

OUTPUTS:
  data/processed/nb1194_summary.json
  data/processed/nb1194_mean_bag_oof.npy
  data/processed/nb1194_median_bag_oof.npy
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

TAG = "nb1194"
ANCHOR = "nb1181_mean_oof"

# nb1070 per-quantile-stretch protocol (verbatim).
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

# Reference (from nb1181_summary.json).
NB1181_REF_POOLED = 0.5566


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
        y_b, p_b = y_train[mask], p_train[mask]
        best_s, best_r = 1.0, float("inf")
        for s in STRETCH_GRID:
            stretched = mu_b + s * (p_b - mu_b)
            r = float(rae(y_b, stretched))
            if r < best_r:
                best_r, best_s = r, float(s)
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
    """Run nb1053-style honest 5-fold cross-fit once for KFold(seed)."""
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
    print(f"{TAG} -- per-quantile rank stretch on {ANCHOR}")
    print(f"          seeds = {SEEDS}   N_BINS = {N_BINS}   "
          f"stretch_grid = {STRETCH_GRID[0]:.2f}..{STRETCH_GRID[-1]:.2f} "
          f"step 0.05")
    print("=" * 78)

    # ---- Load anchor + truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    anchor_path = DATA_PROCESSED / f"{ANCHOR}.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"missing {anchor_path}")
    p_unb = np.load(anchor_path).astype(np.float64)
    if p_unb.shape[0] != n_unb:
        raise ValueError(f"anchor shape {p_unb.shape} != y_unb shape {y_unb.shape}")

    in_rae_anchor = float(rae(y_unb, p_unb))
    pred_std = float(p_unb.std())
    truth_std = float(y_unb.std())
    print(f"[load] {ANCHOR}.npy shape={p_unb.shape}  "
          f"y_unb shape={y_unb.shape}")
    print(f"[base] pooled RAE(nb1181_mean OOF, 253)  = {in_rae_anchor:.4f}   "
          f"(ref nb1181 {NB1181_REF_POOLED:.4f})")
    print(f"[diag] pred_std = {pred_std:.4f}   truth_std = {truth_std:.4f}   "
          f"ratio = {pred_std / truth_std:.4f}")

    # ---- Per-seed honest cross-fit on the corrected predictor ----
    print("\n" + "-" * 78)
    print(f"PER-SEED HONEST CROSS-FIT  (anchor = {ANCHOR})")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_ss_means: list[list[float]] = []
    seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, per_fold_ss = run_one_seed(seed, p_unb, y_unb)
        oof_stack[i] = oof
        per_seed_rae.append(pooled)
        ss_arr = np.array(per_fold_ss)
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

    # ---- Bag across seeds at row level ----
    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"\n[bag] MEDIAN  bagged_oof RAE (median of {len(SEEDS)} OOFs) = "
          f"{bag_median_rae:.4f}")
    print(f"[bag] MEAN    bagged_oof RAE (mean   of {len(SEEDS)} OOFs) = "
          f"{bag_mean_rae:.4f}")

    # ---- Per-bin consistency across seeds ----
    ss_means_arr = np.array(per_seed_ss_means)  # (n_seeds, N_BINS)
    ss_mean_across_seeds = ss_means_arr.mean(axis=0).tolist()
    ss_std_across_seeds = ss_means_arr.std(axis=0).tolist()
    print(f"\n[bin-consistency] per-bin stretch factor mean across seeds:")
    print(f"   bins (low -> high): "
          f"[{', '.join(f'{x:.3f}' for x in ss_mean_across_seeds)}]")
    print(f"   std  across seeds : "
          f"[{', '.join(f'{x:.3f}' for x in ss_std_across_seeds)}]")

    # ---- Verdict at 0.003 margin ----
    best_bag_rae = min(bag_median_rae, bag_mean_rae)
    delta_vs_nb1181_median = bag_median_rae - in_rae_anchor
    delta_vs_nb1181_mean = bag_mean_rae - in_rae_anchor
    best_delta = best_bag_rae - in_rae_anchor
    beats_nb1181 = best_bag_rae < (in_rae_anchor - 0.003)
    if best_delta <= -0.003:
        verdict = "RECAL_HELPS"
    elif abs(best_delta) < 0.003:
        verdict = "RECAL_NEUTRAL"
    else:
        verdict = "RECAL_HURTS"

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            bagged_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            bagged_median_oof.astype(np.float32))
    print(f"\n[save] {TAG}_mean_bag_oof.npy")
    print(f"[save] {TAG}_median_bag_oof.npy")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1181 ref pooled               = {NB1181_REF_POOLED:.4f}")
    print(f"   nb1181 OOF in_RAE this run      = {in_rae_anchor:.4f}")
    print(f"   nb1194 per-seed mean RAE        = {per_seed_mean:.4f}  "
          f"std={per_seed_std:.4f}")
    print(f"   nb1194 MEAN-bag pooled OOF      = {bag_mean_rae:.4f}  "
          f"(delta vs nb1181 = {delta_vs_nb1181_mean:+.4f})")
    print(f"   nb1194 MEDIAN-bag pooled OOF    = {bag_median_rae:.4f}  "
          f"(delta vs nb1181 = {delta_vs_nb1181_median:+.4f})")
    print(f"   beats_nb1181 (0.003 margin)     = {beats_nb1181}")
    print(f"   verdict                         = {verdict}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stretch_grid": STRETCH_GRID,
        "in_rae_anchor_on_253": in_rae_anchor,
        "pred_std_anchor": pred_std,
        "truth_std_unblind": truth_std,
        "std_ratio_pred_over_truth": pred_std / truth_std,
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_std": per_seed_std,
        "per_seed_rae_min": per_seed_min,
        "per_seed_rae_max": per_seed_max,
        "bag_median_rae": bag_median_rae,
        "bag_mean_rae": bag_mean_rae,
        "delta_vs_nb1181_median": delta_vs_nb1181_median,
        "delta_vs_nb1181_mean": delta_vs_nb1181_mean,
        "best_bag_rae": best_bag_rae,
        "best_delta_vs_nb1181": best_delta,
        "beats_nb1181": bool(beats_nb1181),
        "verdict": verdict,
        "per_seed_ss_means_across_folds": per_seed_ss_means,
        "per_bin_stretch_mean_across_seeds": ss_mean_across_seeds,
        "per_bin_stretch_std_across_seeds": ss_std_across_seeds,
        "nb1181_ref_pooled": NB1181_REF_POOLED,
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
    for k in ("in_rae_anchor_on_253", "pred_std_anchor", "truth_std_unblind",
              "per_seed_rae", "per_seed_rae_mean", "per_seed_rae_std",
              "bag_mean_rae", "bag_median_rae",
              "delta_vs_nb1181_mean", "delta_vs_nb1181_median",
              "beats_nb1181", "verdict",
              "per_bin_stretch_mean_across_seeds"):
        print(f"  {k}: {res.get(k)}")
