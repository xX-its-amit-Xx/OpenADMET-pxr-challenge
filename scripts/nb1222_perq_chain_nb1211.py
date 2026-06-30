"""nb1222 -- Chain nb1211 (MEAN_mean BoB blend) -> per-quantile rank stretch.

HYPOTHESIS:
  nb1211 is a 0.5/0.5 mean of two BoBs (nb1190_mean + nb1200_mean). Averaging
  compresses variance below truth_std. A per-quantile (per-bin) stretch may
  recover variance per-bin and reduce RAE further.

PROTOCOL (nb1134 reference):
  1. Load nb1211_mean_oof.npy (253 rows, pooled RAE 0.5451).
  2. Per-quantile stretch:
       N_BINS = 5
       stretch grid 0.80..2.00 step 0.05
       KFold(5) x 5 seeds (0, 1, 7, 42, 137)
  3. For each (bin, seed, fold):
       * Train-fold: compute per-bin mu and grid-search per-bin s minimizing MAE
         (== minimizing RAE since denom is constant per training fold).
       * Held-out fold: apply per-bin (mu_b, s_b).
  4. Pool RAE; mean-bag + median-bag across 5 seeds.
  5. Per-bin stretch factor consistency across seeds.
  6. Verdict at 0.003 margin vs nb1211 (0.5451).
  7. pred_std nb1211 vs truth_std on unblind 253.

Outputs:
  data/processed/nb1222_summary.json
  data/processed/nb1222_mean_bag_oof.npy
  data/processed/nb1222_median_bag_oof.npy
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

TAG = "nb1222"
ANCHOR = "nb1211_mean"   # MEAN_mean BoB blend variant

N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

NB1211_REF_POOLED = 0.5451


# ---------- per-quantile stretch primitives (nb1070/nb1134 verbatim) ----------

def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
                 ) -> tuple[float, np.ndarray, list[list[float]],
                            list[list[float]]]:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    per_fold_ss: list[list[float]] = []
    per_fold_mus: list[list[float]] = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va = p_unb[va_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        oof[va_loc] = apply_per_bin_stretch(p_va, edges, mus, ss)
        per_fold_ss.append(ss.tolist())
        per_fold_mus.append(mus.tolist())
    pooled = float(rae(y_unb, oof))
    return pooled, oof, per_fold_ss, per_fold_mus


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-quantile stretch chain on nb1211_mean_oof "
          f"(5 bins, 5-seed bag)")
    print(f"       seeds = {SEEDS}   N_BINS = {N_BINS}   "
          f"grid = {STRETCH_GRID[0]:.2f}..{STRETCH_GRID[-1]:.2f} step 0.05")
    print("=" * 78)

    # ---- Load anchor + truth ----
    anchor_path = DATA_PROCESSED / f"{ANCHOR}_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"missing {anchor_path}")
    p_unb = np.load(anchor_path).astype(np.float64)
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert p_unb.shape[0] == n_unb, (p_unb.shape, n_unb)

    in_rae_anchor = float(rae(y_unb, p_unb))
    pred_std = float(p_unb.std(ddof=0))
    truth_std = float(y_unb.std(ddof=0))
    pred_mean = float(p_unb.mean())
    truth_mean = float(y_unb.mean())
    print(f"[load] {ANCHOR}_oof.npy  shape={p_unb.shape}")
    print(f"[base] pooled RAE(nb1211 OOF, 253) = {in_rae_anchor:.4f}   "
          f"(ref {NB1211_REF_POOLED:.4f})")
    print(f"[diag] nb1211 pred  mean={pred_mean:.4f}  std={pred_std:.4f}")
    print(f"[diag] truth        mean={truth_mean:.4f}  std={truth_std:.4f}")
    print(f"[diag] std-ratio    pred/truth = {pred_std/truth_std:.4f}   "
          f"(<1 => compressed; stretch hypothesis valid)")

    # ---- Per-seed honest cross-fit ----
    print("\n" + "-" * 78)
    print("PER-SEED HONEST CROSS-FIT  (anchor = nb1211_mean)")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_ss_means: list[list[float]] = []
    per_seed_ss_stds: list[list[float]] = []
    seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, per_fold_ss, per_fold_mus = run_one_seed(seed, p_unb, y_unb)
        oof_stack[i] = oof
        per_seed_rae.append(pooled)
        ss_arr = np.array(per_fold_ss)
        ss_mean = ss_arr.mean(axis=0).tolist()
        ss_std = ss_arr.std(axis=0).tolist()
        per_seed_ss_means.append(ss_mean)
        per_seed_ss_stds.append(ss_std)
        seed_records.append({
            "seed": seed,
            "pooled_rae": pooled,
            "per_fold_ss": per_fold_ss,
            "per_fold_mus": per_fold_mus,
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

    # ---- Bags ----
    bagged_mean_oof = oof_stack.mean(axis=0)
    bagged_median_oof = np.median(oof_stack, axis=0)
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    print(f"\n[bag] MEAN-bag   pooled OOF RAE = {bag_mean_rae:.4f}")
    print(f"[bag] MEDIAN-bag pooled OOF RAE = {bag_median_rae:.4f}")

    # Per-bin stretch consistency across seeds
    ss_mean_arr = np.array(per_seed_ss_means)  # (n_seeds, N_BINS)
    perbin_ss_mean_across_seeds = ss_mean_arr.mean(axis=0).tolist()
    perbin_ss_std_across_seeds = ss_mean_arr.std(axis=0).tolist()
    print(f"\n[consistency] per-bin ss (mean of seed-means)  = "
          f"[{', '.join(f'{x:.3f}' for x in perbin_ss_mean_across_seeds)}]")
    print(f"[consistency] per-bin ss (std  of seed-means)  = "
          f"[{', '.join(f'{x:.3f}' for x in perbin_ss_std_across_seeds)}]")

    # Bagged-OOF std diagnostics
    bag_mean_std = float(bagged_mean_oof.std(ddof=0))
    bag_median_std = float(bagged_median_oof.std(ddof=0))
    print(f"[diag] bag_mean   std = {bag_mean_std:.4f}   "
          f"(was anchor {pred_std:.4f})   truth {truth_std:.4f}")
    print(f"[diag] bag_median std = {bag_median_std:.4f}")

    # Verdict
    delta_mean_vs_ref = bag_mean_rae - NB1211_REF_POOLED
    delta_median_vs_ref = bag_median_rae - NB1211_REF_POOLED
    delta_mean_vs_anchor = bag_mean_rae - in_rae_anchor
    delta_median_vs_anchor = bag_median_rae - in_rae_anchor
    best_bag_rae = min(bag_mean_rae, bag_median_rae)
    best_bag_tag = "MEAN_bag" if bag_mean_rae <= bag_median_rae else "MEDIAN_bag"
    beats_nb1211 = best_bag_rae < NB1211_REF_POOLED - 0.003
    if best_bag_rae < NB1211_REF_POOLED - 0.003:
        verdict = f"PERQ_HELPS ({best_bag_tag} @ {best_bag_rae:.4f})"
    elif abs(best_bag_rae - NB1211_REF_POOLED) <= 0.003:
        verdict = f"PERQ_NEUTRAL ({best_bag_tag} @ {best_bag_rae:.4f})"
    else:
        verdict = f"PERQ_HURTS ({best_bag_tag} @ {best_bag_rae:.4f})"

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            bagged_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            bagged_median_oof.astype(np.float32))
    print(f"\n[save] {TAG}_mean_bag_oof.npy")
    print(f"[save] {TAG}_median_bag_oof.npy")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stretch_grid": STRETCH_GRID,
        "in_rae_anchor_on_253": in_rae_anchor,
        "anchor_pred_mean": pred_mean,
        "anchor_pred_std": pred_std,
        "truth_mean": truth_mean,
        "truth_std": truth_std,
        "anchor_pred_std_over_truth_std": pred_std / truth_std,
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_std": per_seed_std,
        "per_seed_rae_min": per_seed_min,
        "per_seed_rae_max": per_seed_max,
        "bag_mean_rae": bag_mean_rae,
        "bag_median_rae": bag_median_rae,
        "bag_mean_std": bag_mean_std,
        "bag_median_std": bag_median_std,
        "per_seed_ss_means_across_folds": per_seed_ss_means,
        "per_seed_ss_stds_across_folds": per_seed_ss_stds,
        "perbin_ss_mean_across_seeds": perbin_ss_mean_across_seeds,
        "perbin_ss_std_across_seeds": perbin_ss_std_across_seeds,
        "delta_bag_mean_vs_nb1211_ref": delta_mean_vs_ref,
        "delta_bag_median_vs_nb1211_ref": delta_median_vs_ref,
        "delta_bag_mean_vs_anchor": delta_mean_vs_anchor,
        "delta_bag_median_vs_anchor": delta_median_vs_anchor,
        "best_bag_tag": best_bag_tag,
        "best_bag_rae": best_bag_rae,
        "beats_nb1211": bool(beats_nb1211),
        "nb1211_ref_pooled": NB1211_REF_POOLED,
        "verdict": verdict,
        "seed_records": seed_records,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1211 ref pooled          = {NB1211_REF_POOLED:.4f}")
    print(f"   nb1211 in-RAE this run     = {in_rae_anchor:.4f}")
    print(f"   nb1222 MEAN-bag pooled OOF = {bag_mean_rae:.4f}   "
          f"(delta vs nb1211 ref = {delta_mean_vs_ref:+.4f})")
    print(f"   nb1222 MEDIAN-bag pooled   = {bag_median_rae:.4f}   "
          f"(delta vs nb1211 ref = {delta_median_vs_ref:+.4f})")
    print(f"   pred_std / truth_std       = "
          f"{pred_std:.4f} / {truth_std:.4f} = {pred_std/truth_std:.4f}")
    print(f"   beats_nb1211 (>=0.003)     = {beats_nb1211}")
    print(f"   verdict                    = {verdict}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("in_rae_anchor_on_253", "anchor_pred_std", "truth_std",
              "per_seed_rae", "per_seed_rae_mean", "per_seed_rae_std",
              "bag_mean_rae", "bag_median_rae",
              "perbin_ss_mean_across_seeds", "perbin_ss_std_across_seeds",
              "best_bag_tag", "best_bag_rae",
              "delta_bag_mean_vs_nb1211_ref",
              "delta_bag_median_vs_nb1211_ref",
              "beats_nb1211", "verdict"):
        print(f"  {k}: {res.get(k)}")
