"""nb1143 -- Per-quantile median-bag stretch applied to nb1130 mean-bag OOF.

Hypothesis test:
    nb1134 chained per-q stretch on top of the SINGLE-SEED nb1123 corrected
    predictor and found the chain hurts (perq_chain RAE > nb1123 0.5704).
    nb1130 is the BAGGED version of nb1123 (mean across 5 KFold seeds of the
    residual-LGBM corrected predictor) -- pooled RAE 0.5673 (mean-bag) vs
    nb1070 anchor 0.5771. The bag re-smooths the corrected-predictor
    distribution (per-seed std collapses by ~0.02 vs single-seed); maybe THAT
    distribution responds to perq stretch where the single-seed one didn't.

Anchor in this run = `nb1130_mean_bag_oof.npy` on the 253 unblind (the bagged
mean across 5 KFold seeds of nb1070 + residual_LGBM_seed_s OOF). It's the
exact predictor whose pooled cross-fit RAE is 0.5673 in nb1130's summary.

Protocol (verbatim nb1070 / nb1134 per-q stretch):
  N_BINS=5, STRETCH_GRID=0.80..2.00 step 0.05, KFold(5, shuffle), seeds in
  {0, 1, 7, 42, 137}, per-seed honest cross-fit, median bag of 5 per-seed
  OOFs at row level -> nb1143 OOF, pooled RAE on 253.

NO deploy (513) refit -- this is a 253-only honest cross-fit diagnostic to
test the hypothesis. The corresponding 513 deploy would require
te_nb1130_mean_bag.npy which doesn't exist on disk (nb1130 is 253-only).

Outputs:
  data/processed/nb1143_pred_oof.npy            (253,) float32
  data/processed/nb1143_summary.json
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

TAG = "nb1143"
ANCHOR_FILE = "nb1130_mean_bag_oof.npy"

# nb1070 per-quantile-stretch protocol (verbatim).
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

# Reference numbers.
NB1070_REF = 0.5790
NB1123_SINGLE_SEED_REF = 0.5704
NB1130_MEAN_BAG_REF = 0.5673
NB1130_MEDIAN_BAG_REF = 0.5696


# ---------- nb1070 per-quantile-stretch primitives (verbatim) ----------

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
                 ) -> tuple[float, np.ndarray, list[list[float]]]:
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
    print(f"{TAG} -- per-quantile median-bag stretch on nb1130 mean-bag OOF")
    print(f"          seeds = {SEEDS}   N_BINS = {N_BINS}   "
          f"stretch_grid = {STRETCH_GRID[0]:.2f}..{STRETCH_GRID[-1]:.2f} step 0.05")
    print("=" * 78)

    # ---- Load 253 truth + nb1130 mean-bag anchor ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    anchor_path = DATA_PROCESSED / ANCHOR_FILE
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"missing {anchor_path} -- run scripts/nb1130_bag_nb1123.py first")
    p_unb = np.load(anchor_path).astype(np.float64)
    if p_unb.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape {p_unb.shape} vs n_unb={n_unb}")
    print(f"[load] {ANCHOR_FILE} shape={p_unb.shape}  "
          f"y_unb shape={y_unb.shape}")

    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"[base] pooled RAE(nb1130 mean-bag OOF, 253) = "
          f"{in_rae_anchor:.4f}   (ref nb1130 {NB1130_MEAN_BAG_REF:.4f})")
    print(f"[diag] nb1130 OOF std={p_unb.std():.4f}   "
          f"truth std={y_unb.std():.4f}   "
          f"compression ratio={p_unb.std() / y_unb.std():.3f}")

    # ---- Per-seed honest cross-fit ----
    print("\n" + "-" * 78)
    print("PER-SEED HONEST CROSS-FIT  (anchor = nb1130 mean-bag OOF)")
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
            "seed": int(seed),
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
          f"{bag_mean_rae:.4f}")

    delta_vs_anchor = bag_median_rae - in_rae_anchor
    delta_vs_nb1070_ref = bag_median_rae - NB1070_REF
    delta_vs_nb1123_ref = bag_median_rae - NB1123_SINGLE_SEED_REF
    beats_anchor = bag_median_rae < in_rae_anchor - 0.003
    if delta_vs_anchor <= -0.003:
        verdict = "PERQ_HELPS_ON_BAGGED"
    elif abs(delta_vs_anchor) < 0.003:
        verdict = "PERQ_NEUTRAL_ON_BAGGED"
    else:
        verdict = "PERQ_HURTS_ON_BAGGED"

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            bagged_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1070 ref pooled              = {NB1070_REF:.4f}")
    print(f"   nb1123 single-seed ref         = {NB1123_SINGLE_SEED_REF:.4f}")
    print(f"   nb1130 mean-bag ref            = {NB1130_MEAN_BAG_REF:.4f}")
    print(f"   nb1130 mean-bag OOF this run   = {in_rae_anchor:.4f}")
    print(f"   nb1143 per-seed mean RAE       = {per_seed_mean:.4f}  "
          f"std={per_seed_std:.4f}")
    print(f"   nb1143 MEDIAN-bag pooled OOF   = {bag_median_rae:.4f}  "
          f"(delta vs nb1130 anchor = {delta_vs_anchor:+.4f},  "
          f"delta vs nb1123 ref = {delta_vs_nb1123_ref:+.4f},  "
          f"delta vs nb1070 ref = {delta_vs_nb1070_ref:+.4f})")
    print(f"   nb1143 MEAN-bag pooled OOF     = {bag_mean_rae:.4f}")
    print(f"   beats_anchor                   = {beats_anchor}")
    print(f"   verdict                        = {verdict}")

    summary = {
        "tag": TAG,
        "anchor_file": ANCHOR_FILE,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "in_rae_anchor_on_253": in_rae_anchor,
        "anchor_oof_std": float(p_unb.std()),
        "truth_std": float(y_unb.std()),
        "anchor_compression_ratio": float(p_unb.std() / y_unb.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_std": per_seed_std,
        "per_seed_rae_min": per_seed_min,
        "per_seed_rae_max": per_seed_max,
        "bag_median_rae": bag_median_rae,
        "bag_mean_rae": bag_mean_rae,
        "delta_vs_anchor": delta_vs_anchor,
        "delta_vs_nb1070_ref": delta_vs_nb1070_ref,
        "delta_vs_nb1123_ref": delta_vs_nb1123_ref,
        "beats_anchor": bool(beats_anchor),
        "verdict": verdict,
        "per_seed_ss_means_across_folds": per_seed_ss_means,
        "seed_records": seed_records,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1123_single_seed_ref": NB1123_SINGLE_SEED_REF,
        "nb1130_mean_bag_ref": NB1130_MEAN_BAG_REF,
        "nb1130_median_bag_ref": NB1130_MEDIAN_BAG_REF,
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
    for k in ("in_rae_anchor_on_253", "per_seed_rae", "per_seed_rae_mean",
              "per_seed_rae_std", "bag_median_rae", "bag_mean_rae",
              "delta_vs_anchor", "delta_vs_nb1070_ref",
              "delta_vs_nb1123_ref", "beats_anchor", "verdict"):
        print(f"  {k}: {res.get(k)}")
