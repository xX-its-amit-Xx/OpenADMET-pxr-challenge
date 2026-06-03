"""nb1151 -- Outer-seed bag of nb1143 (per-quantile median-bag on nb1130 mean-bag).

Hypothesis test:
    nb1143 reported MEDIAN-bag pooled cross-fit RAE = 0.5649 using a SINGLE
    inner seed set {0, 1, 7, 42, 137}. That number sits below the
    nb1130 mean-bag anchor (0.5673) by -0.0024, but the per-seed spread
    inside nb1143 was 0.5624..0.5746 (std 0.0043). A single inner-seed-set
    could itself be lucky -- if we re-roll the inner KFold seeds, does the
    median-bag pooled RAE stay near 0.5649?

Protocol:
    OUTER_SEEDS = {0, 1, 7, 42, 137}.
    For each outer seed o, build a derived inner seed set
        inner_seeds(o) = [o*1000 + s for s in (0, 1, 7, 42, 137)]
    so o=0 reproduces nb1143 verbatim (inner seeds 0, 1, 7, 42, 137 since
    0*1000+s=s), and o>0 gives 5 disjoint inner-seed sets (offsets
    1000/7000/42000/137000) that the nb1143 protocol has never seen.

    For each outer seed:
      * Run the full nb1143 per-q median-bag protocol with its 5 derived
        inner seeds (verbatim run_one_seed + median row-level bag).
      * Record the pooled cross-fit RAE of that outer seed's median-bag
        OOF on the 253 unblind = per-outer-seed pooled cross-fit RAE.

    Report: mean / std / min / max of the 5 per-outer-seed pooled RAEs and
    the row-level median bag across the 5 outer-seed median-bag OOFs.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1151_pred_oof.npy   (253,) float32   row-level median
                                                        across the 5
                                                        outer-seed median
                                                        bags
  data/processed/nb1151_summary.json
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

TAG = "nb1151"
ANCHOR_FILE = "nb1130_mean_bag_oof.npy"

# nb1143 / nb1070 per-quantile-stretch protocol (verbatim).
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
INNER_SEED_BASE = [0, 1, 7, 42, 137]
OUTER_SEEDS = [0, 1, 7, 42, 137]

# Reference numbers.
NB1070_REF = 0.5790
NB1123_SINGLE_SEED_REF = 0.5704
NB1130_MEAN_BAG_REF = 0.5673
NB1143_BAG_MEDIAN_REF = 0.5649


# ---------- nb1143 / nb1070 per-quantile-stretch primitives (verbatim) ----

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


def run_one_inner_seed(inner_seed: int, p_unb: np.ndarray,
                       y_unb: np.ndarray) -> tuple[float, np.ndarray]:
    """Single nb1143 inner-seed pass: KFold + per-bin stretch grid -> OOF."""
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=inner_seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va = p_unb[va_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        oof[va_loc] = apply_per_bin_stretch(p_va, edges, mus, ss)
    pooled = float(rae(y_unb, oof))
    return pooled, oof


def run_one_outer_seed(outer_seed: int, p_unb: np.ndarray,
                       y_unb: np.ndarray
                       ) -> tuple[float, np.ndarray, list[int], list[float]]:
    """nb1143 protocol with 5 derived inner seeds, median row-level bag."""
    inner_seeds = [outer_seed * 1000 + s for s in INNER_SEED_BASE]
    n = len(y_unb)
    inner_oof_stack = np.zeros((len(inner_seeds), n), dtype=np.float64)
    inner_per_seed_rae: list[float] = []
    for j, isd in enumerate(inner_seeds):
        r_j, oof_j = run_one_inner_seed(isd, p_unb, y_unb)
        inner_oof_stack[j] = oof_j
        inner_per_seed_rae.append(r_j)
    median_oof = np.median(inner_oof_stack, axis=0)
    bag_median_rae = float(rae(y_unb, median_oof))
    return bag_median_rae, median_oof, inner_seeds, inner_per_seed_rae


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- outer-seed bag of nb1143 (per-q median-bag on nb1130)")
    print(f"          OUTER_SEEDS = {OUTER_SEEDS}")
    print(f"          inner_seeds(o) = [o*1000 + s for s in "
          f"{INNER_SEED_BASE}]")
    print(f"          N_BINS = {N_BINS}   N_FOLDS = {N_FOLDS}   "
          f"grid {STRETCH_GRID[0]:.2f}..{STRETCH_GRID[-1]:.2f} step 0.05")
    print("=" * 78)

    # ---- Load 253 truth + nb1130 mean-bag anchor ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    anchor_path = DATA_PROCESSED / ANCHOR_FILE
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"missing {anchor_path} -- run nb1130_bag_nb1123.py first")
    p_unb = np.load(anchor_path).astype(np.float64)
    if p_unb.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape {p_unb.shape} vs n_unb={n_unb}")
    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"[load] {ANCHOR_FILE} shape={p_unb.shape}  "
          f"y_unb shape={y_unb.shape}")
    print(f"[base] pooled RAE(nb1130 mean-bag OOF, 253) = "
          f"{in_rae_anchor:.4f}   (ref nb1130 {NB1130_MEAN_BAG_REF:.4f})")

    # ---- Per-outer-seed honest cross-fit + bag ----
    print("\n" + "-" * 78)
    print("PER-OUTER-SEED nb1143 PROTOCOL")
    print("-" * 78)
    outer_oof_stack = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae: list[float] = []
    outer_records: list[dict] = []
    for i, o in enumerate(OUTER_SEEDS):
        bag_r, median_oof, inner_seeds, inner_rae = run_one_outer_seed(
            o, p_unb, y_unb)
        outer_oof_stack[i] = median_oof
        per_outer_rae.append(bag_r)
        outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "inner_per_seed_rae": inner_rae,
            "inner_rae_mean": float(np.mean(inner_rae)),
            "inner_rae_std": float(np.std(inner_rae)),
            "outer_bag_median_rae": bag_r,
        })
        inner_rae_str = ",".join(f"{r:.4f}" for r in inner_rae)
        print(f"   outer {o:>3d}: inner_seeds={inner_seeds}  "
              f"inner_RAE=[{inner_rae_str}]")
        print(f"              outer-median-bag pooled RAE = {bag_r:.4f}")

    per_outer_arr = np.array(per_outer_rae)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    print(f"\n[per-outer] pooled RAE  mean={per_outer_mean:.4f}  "
          f"std={per_outer_std:.4f}  "
          f"min={per_outer_min:.4f}  max={per_outer_max:.4f}")

    # ---- Row-level median across the 5 outer-seed median-bag OOFs ----
    outer_median_oof = np.median(outer_oof_stack, axis=0)
    outer_mean_oof = outer_oof_stack.mean(axis=0)
    outer_median_rae = float(rae(y_unb, outer_median_oof))
    outer_mean_rae = float(rae(y_unb, outer_mean_oof))
    print(f"\n[bag-of-bags] MEDIAN across 5 outer median-bag OOFs = "
          f"{outer_median_rae:.4f}")
    print(f"[bag-of-bags] MEAN   across 5 outer median-bag OOFs = "
          f"{outer_mean_rae:.4f}")

    # ---- Hypothesis verdict ----
    delta_vs_nb1143 = per_outer_mean - NB1143_BAG_MEDIAN_REF
    delta_vs_anchor = per_outer_mean - in_rae_anchor
    beats_nb1143_mean = per_outer_mean < NB1143_BAG_MEDIAN_REF - 0.003
    beats_anchor_mean = per_outer_mean < in_rae_anchor - 0.003
    if abs(delta_vs_nb1143) < 0.003:
        verdict = "NB1143_REPRODUCES_UNDER_OUTER_SEEDS"
    elif delta_vs_nb1143 > 0.003:
        verdict = "NB1143_WAS_LUCKY_SINGLE_INNER_SEED_SET"
    else:
        verdict = "NB1143_WAS_UNLUCKY_OUTER_BAG_IMPROVES"

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            outer_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1070 ref pooled               = {NB1070_REF:.4f}")
    print(f"   nb1123 single-seed ref          = {NB1123_SINGLE_SEED_REF:.4f}")
    print(f"   nb1130 mean-bag ref             = {NB1130_MEAN_BAG_REF:.4f}")
    print(f"   nb1143 bag_median ref           = {NB1143_BAG_MEDIAN_REF:.4f}")
    print(f"   nb1151 per-outer-seed RAE mean  = {per_outer_mean:.4f}  "
          f"std={per_outer_std:.4f}")
    print(f"   nb1151 outer-MEDIAN bag-of-bags = {outer_median_rae:.4f}")
    print(f"   nb1151 outer-MEAN bag-of-bags   = {outer_mean_rae:.4f}")
    print(f"   delta(mean vs nb1143)           = {delta_vs_nb1143:+.4f}")
    print(f"   delta(mean vs nb1130 anchor)    = {delta_vs_anchor:+.4f}")
    print(f"   beats_nb1143_mean (>=0.003)     = {beats_nb1143_mean}")
    print(f"   beats_anchor_mean (>=0.003)     = {beats_anchor_mean}")
    print(f"   verdict                         = {verdict}")

    summary = {
        "tag": TAG,
        "anchor_file": ANCHOR_FILE,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "outer_seeds": OUTER_SEEDS,
        "inner_seed_base": INNER_SEED_BASE,
        "inner_seed_formula": "inner_seed = outer_seed * 1000 + base_seed",
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "in_rae_anchor_on_253": in_rae_anchor,
        "per_outer_seed_rae": per_outer_rae,
        "per_outer_seed_rae_mean": per_outer_mean,
        "per_outer_seed_rae_std": per_outer_std,
        "per_outer_seed_rae_min": per_outer_min,
        "per_outer_seed_rae_max": per_outer_max,
        "outer_bag_of_bags_median_rae": outer_median_rae,
        "outer_bag_of_bags_mean_rae": outer_mean_rae,
        "delta_per_outer_mean_vs_nb1143": delta_vs_nb1143,
        "delta_per_outer_mean_vs_anchor": delta_vs_anchor,
        "beats_nb1143_mean": bool(beats_nb1143_mean),
        "beats_anchor_mean": bool(beats_anchor_mean),
        "verdict": verdict,
        "outer_records": outer_records,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1123_single_seed_ref": NB1123_SINGLE_SEED_REF,
        "nb1130_mean_bag_ref": NB1130_MEAN_BAG_REF,
        "nb1143_bag_median_ref": NB1143_BAG_MEDIAN_REF,
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
    for k in ("in_rae_anchor_on_253", "per_outer_seed_rae",
              "per_outer_seed_rae_mean", "per_outer_seed_rae_std",
              "outer_bag_of_bags_median_rae", "outer_bag_of_bags_mean_rae",
              "delta_per_outer_mean_vs_nb1143", "beats_nb1143_mean",
              "verdict"):
        print(f"  {k}: {res.get(k)}")
