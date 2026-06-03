"""nb1164 -- Chain nb1153 mean-bag residual-corrected predictor -> nb1070
per-quantile stretch median bag.

Hypothesis:
    nb1153 (depth=3 LGBM Huber on Mordred-only of nb1070 residual, 5-seed
    mean bag) cross-fits to RAE 0.5640 on the 253 unblind by adding a
    Mordred-orthogonal residual chunk on top of nb1070's 0.5790 anchor.
    The residual correction RE-SHAPES the predictor distribution: per-seed
    resid_oof.std ~ 0.25 vs truth-residual.std 0.67 -- shrunk -- so the
    corrected predictor may now be variance-compressed or quantile-skewed
    in a non-uniform way the original (nb1070-tuned) per-quantile stretch
    doesn't see.

    Re-running nb1070's per-quantile stretch protocol (5 bins, 5-seed
    KFold median bag, stretch grid 0.80..2.00 step 0.05) on the
    nb1153 mean-bag OOF as the NEW anchor tests whether quantile-aware
    decompression COMPOUNDS with the residual correction (chain helps,
    nb1143-style) or whether nb1153 already exhausted the quantile
    decompression capacity nb1070's per-q stretch was extracting (chain
    neutral/hurts, nb1134-style).

    Cycle 19 nb1134 (chain nb1123 residual -> per-q stretch) HURT vs
    nb1123 anchor by +0.005 RAE. nb1153 residual is Mordred-only, leaves
    a different residual distribution than nb1123 (Morgan+RDKit), so the
    outcome is not pre-determined.

Procedure (mirrors nb1134, 253-only -- NO 513 deploy):
  1. Anchor = nb1153_mean_bag_oof.npy (253-vec, pooled RAE ~0.5640).
  2. Per seed s in {0, 1, 7, 42, 137}:
       KFold(5, shuffle, random_state=s):
         * Train-only quantile edges (5 bins).
         * Train-only fit_per_bin_stretch grid (s in {0.80..2.00 step 0.05}).
         * Apply per-bin (mu_b, s_b) to held-out fold rows -> seed OOF.
  3. Stack -> (5, 253), MEDIAN across seed -> nb1164 OOF.
  4. Pooled RAE on the 253 unblind.

Outputs:
  data/processed/nb1164_pred_oof.npy             (253,) float32 -- median bag
  data/processed/nb1164_per_seed_oof.npy         (5, 253) float32
  data/processed/nb1164_summary.json
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

TAG = "nb1164"
ANCHOR_TAG = "nb1153"
ANCHOR_OOF_FILE = f"{ANCHOR_TAG}_mean_bag_oof.npy"

# nb1070 per-quantile-stretch protocol (verbatim).
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

# Reference numbers.
NB1070_REF_POOLED = 0.5790
NB1153_MEAN_BAG_REF = 0.5640


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
    """nb1053-style honest 5-fold cross-fit once for KFold(seed)."""
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
    print(f"{TAG} -- nb1070 per-quantile stretch (5 bins, 5-seed MEDIAN bag) "
          f"on nb1153 mean-bag predictor")
    print(f"          seeds = {SEEDS}   N_BINS = {N_BINS}   "
          f"stretch_grid = {STRETCH_GRID[0]:.2f}..{STRETCH_GRID[-1]:.2f} "
          f"step 0.05")
    print(f"          253-only honest cross-fit; no 513 deploy")
    print("=" * 78)

    # ---- Load 253 unblind truth + nb1153 anchor ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    anchor_path = DATA_PROCESSED / ANCHOR_OOF_FILE
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"missing {anchor_path} -- run scripts/nb1153_residual_mordred_features.py "
            f"first"
        )
    p_unb = np.load(anchor_path).astype(np.float64)
    if p_unb.shape[0] != n_unb:
        raise ValueError(
            f"anchor shape mismatch: {p_unb.shape} vs n_unb={n_unb}"
        )
    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"[load] {ANCHOR_OOF_FILE}  shape={p_unb.shape}  "
          f"pooled RAE = {in_rae_anchor:.4f}  "
          f"(ref nb1153 {NB1153_MEAN_BAG_REF:.4f})")

    # Distribution diagnostics: anchor vs truth
    nb1070_oof_path = DATA_PROCESSED / "nb1070_pred_oof.npy"
    if nb1070_oof_path.exists():
        nb1070_oof = np.load(nb1070_oof_path).astype(np.float64)
        if nb1070_oof.shape[0] == n_unb:
            print(f"[diag] nb1070 OOF  std={nb1070_oof.std():.4f}   "
                  f"nb1153 anchor std={p_unb.std():.4f}   "
                  f"truth       std={y_unb.std():.4f}")
            print(f"[diag] corr(nb1070_oof, nb1153_anchor) = "
                  f"{np.corrcoef(nb1070_oof, p_unb)[0, 1]:.4f}")
        else:
            print(f"[diag] nb1070 OOF shape mismatch -- skipping diag corr")
    else:
        print(f"[diag] nb1070 OOF not found -- skipping diag corr")
    print(f"[diag] anchor mean={p_unb.mean():.4f}  truth mean={y_unb.mean():.4f}")

    # ---- Per-seed honest cross-fit ----
    print("\n" + "-" * 78)
    print("PER-SEED HONEST CROSS-FIT (per-quantile stretch on nb1153 anchor)")
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
            "delta_vs_anchor": pooled - in_rae_anchor,
            "per_fold_ss": per_fold_ss,
            "fold_ss_mean": ss_mean,
            "fold_ss_std": ss_std,
        })
        ss_mean_str = ",".join(f"{x:.2f}" for x in ss_mean)
        print(f"   seed {seed:>3d}: pooled_RAE = {pooled:.4f}  "
              f"(d_vs_anchor = {pooled - in_rae_anchor:+.4f})  "
              f"ss_mean=[{ss_mean_str}]")

    per_seed_rae_arr = np.array(per_seed_rae)
    per_seed_mean = float(per_seed_rae_arr.mean())
    per_seed_median = float(np.median(per_seed_rae_arr))
    per_seed_std = float(per_seed_rae_arr.std())
    per_seed_min = float(per_seed_rae_arr.min())
    per_seed_max = float(per_seed_rae_arr.max())
    print(f"\n[per-seed] RAE  mean={per_seed_mean:.4f}  "
          f"median={per_seed_median:.4f}  "
          f"std={per_seed_std:.4f}  "
          f"min={per_seed_min:.4f}  max={per_seed_max:.4f}")

    # ---- MEDIAN bag (primary aggregator per protocol) and MEAN bag ----
    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"\n[bag] MEDIAN bagged_oof RAE (median of {len(SEEDS)} OOFs) = "
          f"{bag_median_rae:.4f}")
    print(f"[bag] MEAN   bagged_oof RAE (mean   of {len(SEEDS)} OOFs) = "
          f"{bag_mean_rae:.4f}")

    delta_vs_anchor = bag_median_rae - in_rae_anchor
    delta_vs_nb1070_ref = bag_median_rae - NB1070_REF_POOLED
    beats_anchor = bag_median_rae < in_rae_anchor - 0.003
    if delta_vs_anchor <= -0.003:
        verdict = "PERQ_CHAIN_HELPS_ON_NB1153"
    elif abs(delta_vs_anchor) < 0.003:
        verdict = "PERQ_CHAIN_NEUTRAL_ON_NB1153"
    else:
        verdict = "PERQ_CHAIN_HURTS_ON_NB1153"

    # ---- Save artefacts (253-only; no 513 deploy this notebook) ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            bagged_median_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_oof.npy",
            oof_stack.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}  "
          f"(median-bag OOF)")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_seed_oof.npy'}  "
          f"(per-seed OOF stack)")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1070 ref pooled               = {NB1070_REF_POOLED:.4f}")
    print(f"   nb1153 mean-bag ref             = {NB1153_MEAN_BAG_REF:.4f}")
    print(f"   nb1153 anchor in_RAE this run   = {in_rae_anchor:.4f}")
    print(f"   nb1164 per-seed mean RAE        = {per_seed_mean:.4f}  "
          f"std={per_seed_std:.4f}")
    print(f"   nb1164 MEDIAN-bag pooled OOF    = {bag_median_rae:.4f}  "
          f"(d_vs_nb1153 anchor = {delta_vs_anchor:+.4f},  "
          f"d_vs_nb1070 ref = {delta_vs_nb1070_ref:+.4f})")
    print(f"   nb1164 MEAN-bag pooled OOF      = {bag_mean_rae:.4f}")
    print(f"   beats_nb1153 (margin 0.003)     = {beats_anchor}")
    print(f"   verdict                         = {verdict}")

    summary = {
        "tag": TAG,
        "anchor_tag": ANCHOR_TAG,
        "anchor_oof_file": ANCHOR_OOF_FILE,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "in_rae_anchor_on_253": in_rae_anchor,
        "anchor_mean": float(p_unb.mean()),
        "anchor_std": float(p_unb.std()),
        "truth_mean": float(y_unb.mean()),
        "truth_std": float(y_unb.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_median": per_seed_median,
        "per_seed_rae_std": per_seed_std,
        "per_seed_rae_min": per_seed_min,
        "per_seed_rae_max": per_seed_max,
        "bag_median_rae": bag_median_rae,
        "bag_mean_rae": bag_mean_rae,
        "delta_vs_nb1153_anchor": delta_vs_anchor,
        "delta_vs_nb1070_ref": delta_vs_nb1070_ref,
        "beats_nb1153": bool(beats_anchor),
        "verdict": verdict,
        "per_seed_ss_means_across_folds": per_seed_ss_means,
        "seed_records": seed_records,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
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
    for k in ("in_rae_anchor_on_253", "per_seed_rae",
              "per_seed_rae_mean", "per_seed_rae_std",
              "bag_median_rae", "bag_mean_rae",
              "delta_vs_nb1153_anchor", "delta_vs_nb1070_ref",
              "beats_nb1153", "verdict"):
        print(f"  {k}: {res.get(k)}")
