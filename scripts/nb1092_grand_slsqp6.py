"""nb1092 -- Grand 6-way SLSQP blend with scalar stretch, multi-seed median bag,
followed by a per-quantile-bin median bag stage.

Pool (K=6):
  0. chemprop_aux         (PRE-unblind LB anchor)
  1. nb972_long_train     (deep+stretch winner from nb1014)
  2. nb1030_mordred_lgbm  (Mordred descriptors LGBM)
  3. nb914_persistence_homology
  4. nb960_pseudo_self_train
  5. nb1081_chemprop_ensemble

Protocol (nb1014 lineage, scaled to K=6):
  For each seed in SEEDS={0,1,7,42,137}:
    KFold(n_splits=5, shuffle=True, random_state=seed) on the 253 unblind:
      For each fold f:
        a. SLSQP fit w = (w0..w5), simplex (sum=1, in [0,1]), minimizing SSE
           on the 4 train folds for the 6-way blend vs y.
        b. blend_tr = P_unb[tr] @ w; mu_tr = blend_tr.mean().
           Grid-scan scalar s in STRETCH_GRID on the train folds (no per-bin
           scan in this stage -- that is the second protocol below).
        c. Apply (w, s, mu_tr) to held-out fold; record OOF.
    Per-seed pooled OOF -> oof_stack[i].
  Bag stage 1: median across the 5 OOFs -> bagged_oof_blend (253,).
  Pool pooled RAE on the 253 = bag_blend_rae.

Stage 2 (per-quantile median bag, on top of the bagged blend OOF):
  Treat bagged_oof_blend as the anchor on the 253 (and apply the bagged
  per-seed deploy blend on the 513 as the anchor on the 513).
  For each seed in SEEDS:
    KFold(5, shuffle=True, random_state=seed) on the 253:
      For each fold f:
        - Train-only quantile edges (N_BINS=5).
        - Train-only fit_per_bin_stretch grid scan in STRETCH_GRID2.
        - Apply per-bin (mu_b, s_b) to held-out fold rows; collect OOF.
  Bag stage 2: median across the 5 OOFs -> bagged_oof_perq (253,).
  Pooled RAE on 253 = bag_perq_rae.

The reported "final_perq_rae" is bag_perq_rae. Anchor reference is nb1070's
median-bag RAE 0.5771. We compare bag_perq_rae against 0.5771 to set
beats_nb1070.

Hypothesis: at K=6 SLSQP can find an off-manifold weight vector that
captures incremental signal (nb1030 Mordred, nb914 PH, nb960 pseudo,
nb1081 ensemble may each carry novel structure). Risk: 6 weights + 1
scalar = 7 hyperparameters per fold -- with 4/5 of 253 (~202) train rows
this is comfortable, but the multi-collinearity of 6 chem predictors can
collapse SLSQP into a 2-way support and waste capacity.

Outputs:
  data/processed/nb1092_summary.json
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1092"
CANDIDATES = [
    "chemprop_aux",
    "nb972_long_train",
    "nb1030",                    # te_nb1030.npy (Mordred LGBM)
    "nb914",                     # te_nb914.npy (Persistence Homology)
    "nb960",                     # te_nb960.npy (Pseudo self-train)
    "nb1081_chemprop_ensemble",  # te_nb1081_chemprop_ensemble.npy
]
CAND_LABEL = [
    "chemprop_aux",
    "nb972_long_train",
    "nb1030_mordred_lgbm",
    "nb914_persistence_homology",
    "nb960_pseudo_self_train",
    "nb1081_chemprop_ensemble",
]
K = len(CANDIDATES)
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
STRETCH_GRID2 = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_BINS = 5

NB1070_BAG_MEDIAN_RAE = 0.5771
NB1014_BAGGED_HONEST_RAE = 0.5930


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SLSQP fit of K weights on the simplex (sum=1, w_i in [0,1])
    minimizing SSE(P @ w - y). Returns w (K,)."""
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)
    cons = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        w0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = float(w.sum())
    if s <= 0.0:
        return np.full(k, 1.0 / k)
    return w / s


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float, grid: list[float]) -> float:
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins_tr = bin_assign(p_train, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        if int(mask.sum()) < 2:
            mus[b] = float(p_train.mean())
            ss[b] = 1.0
            continue
        mu_b = float(p_train[mask].mean())
        mus[b] = mu_b
        y_b = y_train[mask]
        p_b = p_train[mask]
        best_s, best_r = 1.0, float("inf")
        for s in STRETCH_GRID2:
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


def run_blend_seed(P_unb: np.ndarray, y_unb: np.ndarray,
                   seed: int) -> tuple[float, np.ndarray, list[dict]]:
    """Stage-1 per-seed honest cross-fit: SLSQP 6-way + scalar stretch."""
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    folds: list[dict] = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        folds.append({
            "fold": k,
            "w": w_f.tolist(),
            "s": s_f,
            "mu_tr": mu_tr,
            "val_rae": rae_va,
            "n_va": int(len(va_loc)),
        })
    pooled = float(rae(y_unb, oof))
    return pooled, oof, folds


def run_perq_seed(p_unb: np.ndarray, y_unb: np.ndarray,
                  seed: int) -> tuple[float, np.ndarray, list[list[float]]]:
    """Stage-2 per-seed honest cross-fit: per-quantile-bin stretch on the
    already-bagged blend anchor (p_unb is the bagged-blend OOF on 253)."""
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
    print(f"{TAG} -- grand SLSQP-6 + scalar stretch + median seed-bag + "
          f"per-quantile median bag")
    print(f"   K={K}  N_FOLDS={N_FOLDS}  seeds={SEEDS}  N_BINS={N_BINS}")
    print("=" * 78)

    # ---- Load 513 te ----
    te = load_test()
    te_names = te["name"].values
    preds_513 = np.column_stack(
        [load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")
    for j, lab in enumerate(CAND_LABEL):
        col = preds_513[:, j]
        print(f"   te[{j}] {lab:30s}: "
              f"mean={col.mean():.3f}  std={col.std():.3f}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"\n[load] P_unb shape = {P_unb.shape}  y_unb shape = {y_unb.shape}")

    # ---- Individual in_RAE on 253 ----
    indiv_rae = {}
    print("\n[indiv] in_RAE on 253 unblind:")
    for j, lab in enumerate(CAND_LABEL):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[lab] = r
        print(f"   {lab:30s}: {r:.4f}")

    # =================================================================
    # STAGE 1 — Multi-seed SLSQP-6 + scalar stretch, median bag.
    # =================================================================
    print("\n" + "-" * 78)
    print("STAGE 1 — per-seed honest cross-fit (SLSQP-6 + scalar stretch)")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    all_w = []     # (n_seeds*n_folds, K)
    all_s: list[float] = []
    seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, folds = run_blend_seed(P_unb, y_unb, seed)
        oof_stack[i] = oof
        per_seed_rae.append(pooled)
        for f in folds:
            all_w.append(f["w"])
            all_s.append(f["s"])
        w_mean = np.mean([f["w"] for f in folds], axis=0)
        s_mean = float(np.mean([f["s"] for f in folds]))
        seed_records.append({
            "seed": seed,
            "pooled_rae": pooled,
            "fold_w_mean": w_mean.tolist(),
            "fold_s_mean": s_mean,
            "folds": folds,
        })
        w_str = ",".join(f"{x:.2f}" for x in w_mean)
        print(f"   seed {seed:>3d}: pooled_RAE={pooled:.4f}  "
              f"w_mean=[{w_str}]  s_mean={s_mean:.2f}")

    per_seed_rae_arr = np.array(per_seed_rae)
    per_seed_mean = float(per_seed_rae_arr.mean())
    per_seed_std = float(per_seed_rae_arr.std())
    per_seed_min = float(per_seed_rae_arr.min())
    per_seed_max = float(per_seed_rae_arr.max())
    print(f"\n[stage1] per-seed RAE  mean={per_seed_mean:.4f}  "
          f"std={per_seed_std:.4f}  min={per_seed_min:.4f}  "
          f"max={per_seed_max:.4f}")

    bag_blend_median_oof = np.median(oof_stack, axis=0)
    bag_blend_mean_oof = oof_stack.mean(axis=0)
    bag_blend_median_rae = float(rae(y_unb, bag_blend_median_oof))
    bag_blend_mean_rae = float(rae(y_unb, bag_blend_mean_oof))
    print(f"[stage1] MEDIAN-bag pooled RAE (across {len(SEEDS)} OOFs) = "
          f"{bag_blend_median_rae:.4f}")
    print(f"[stage1] MEAN-bag   pooled RAE (across {len(SEEDS)} OOFs) = "
          f"{bag_blend_mean_rae:.4f}")

    all_w_arr = np.array(all_w)  # (25, 6)
    mean_w = all_w_arr.mean(axis=0)
    std_w = all_w_arr.std(axis=0)
    mean_s = float(np.mean(all_s))
    print(f"\n[stage1] mean weights across {len(all_s)} folds:")
    for j, lab in enumerate(CAND_LABEL):
        print(f"   w[{j}] {lab:30s}: "
              f"mean={mean_w[j]:.3f}  std={std_w[j]:.3f}")
    print(f"   scalar s: mean={mean_s:.3f}  std={np.std(all_s):.3f}")

    # =================================================================
    # STAGE 2 — per-quantile-bin median bag on the bagged-blend OOF.
    # =================================================================
    print("\n" + "-" * 78)
    print("STAGE 2 — per-quantile median bag on bagged-blend anchor")
    print("-" * 78)
    # Anchor for stage 2 = stage-1 MEDIAN-bagged OOF on the 253 unblind.
    p_anchor_unb = bag_blend_median_oof.copy()
    print(f"[stage2] anchor (bagged-blend median OOF) on 253: "
          f"in_RAE={float(rae(y_unb, p_anchor_unb)):.4f}")

    oof2_stack = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae2: list[float] = []
    per_seed_ss2: list[list[list[float]]] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, per_fold_ss = run_perq_seed(p_anchor_unb, y_unb, seed)
        oof2_stack[i] = oof
        per_seed_rae2.append(pooled)
        per_seed_ss2.append(per_fold_ss)
        ss_arr = np.array(per_fold_ss)  # (5, N_BINS)
        ss_mean_str = ",".join(f"{x:.2f}" for x in ss_arr.mean(axis=0))
        print(f"   seed {seed:>3d}: pooled_RAE={pooled:.4f}  "
              f"ss_mean=[{ss_mean_str}]")

    per_seed_rae2_arr = np.array(per_seed_rae2)
    per_seed_mean2 = float(per_seed_rae2_arr.mean())
    per_seed_std2 = float(per_seed_rae2_arr.std())
    bag_perq_median_oof = np.median(oof2_stack, axis=0)
    bag_perq_mean_oof = oof2_stack.mean(axis=0)
    bag_perq_median_rae = float(rae(y_unb, bag_perq_median_oof))
    bag_perq_mean_rae = float(rae(y_unb, bag_perq_mean_oof))
    print(f"\n[stage2] per-seed RAE  mean={per_seed_mean2:.4f}  "
          f"std={per_seed_std2:.4f}")
    print(f"[stage2] MEDIAN-bag pooled RAE = {bag_perq_median_rae:.4f}")
    print(f"[stage2] MEAN-bag   pooled RAE = {bag_perq_mean_rae:.4f}")

    # ---- Verdicts ----
    final_perq_rae = bag_perq_median_rae
    delta_vs_nb1070 = final_perq_rae - NB1070_BAG_MEDIAN_RAE
    beats_nb1070 = final_perq_rae < NB1070_BAG_MEDIAN_RAE
    if delta_vs_nb1070 < -0.005:
        verdict = "BEATS_NB1070"
    elif abs(delta_vs_nb1070) <= 0.005:
        verdict = "TIES_NB1070"
    else:
        verdict = "WORSE_THAN_NB1070"

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   anchor: nb1070 median-bag      = {NB1070_BAG_MEDIAN_RAE:.4f}")
    print(f"   stage-1 bagged-blend (median)  = {bag_blend_median_rae:.4f}")
    print(f"   stage-2 final perq (median)    = {final_perq_rae:.4f}")
    print(f"   delta vs nb1070                = {delta_vs_nb1070:+.4f}  "
          f"-> {verdict}")
    print(f"   beats_nb1070                   = {beats_nb1070}")

    summary = {
        "tag": TAG,
        "candidates": CANDIDATES,
        "candidate_labels": CAND_LABEL,
        "K": K,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "n_bins": N_BINS,
        "stretch_grid_stage1": STRETCH_GRID,
        "stretch_grid_stage2": STRETCH_GRID2,
        "indiv_in_rae_on_253": indiv_rae,
        # Stage 1
        "stage1_per_seed_rae": per_seed_rae,
        "stage1_per_seed_rae_mean": per_seed_mean,
        "stage1_per_seed_rae_std": per_seed_std,
        "stage1_per_seed_rae_min": per_seed_min,
        "stage1_per_seed_rae_max": per_seed_max,
        "stage1_bag_median_rae": bag_blend_median_rae,
        "stage1_bag_mean_rae": bag_blend_mean_rae,
        "stage1_mean_w": mean_w.tolist(),
        "stage1_std_w": std_w.tolist(),
        "stage1_mean_s": mean_s,
        "stage1_std_s": float(np.std(all_s)),
        # Stage 2
        "stage2_per_seed_rae": per_seed_rae2,
        "stage2_per_seed_rae_mean": per_seed_mean2,
        "stage2_per_seed_rae_std": per_seed_std2,
        "stage2_bag_median_rae": bag_perq_median_rae,
        "stage2_bag_mean_rae": bag_perq_mean_rae,
        # Headline
        "final_perq_rae": final_perq_rae,
        "anchor_rae_nb1070": NB1070_BAG_MEDIAN_RAE,
        "delta_vs_nb1070": delta_vs_nb1070,
        "beats_nb1070": bool(beats_nb1070),
        "verdict": verdict,
        "nb1014_bagged_honest_rae": NB1014_BAGGED_HONEST_RAE,
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
    for k in (
        "stage1_per_seed_rae",
        "stage1_per_seed_rae_mean",
        "stage1_bag_median_rae",
        "stage1_mean_w",
        "stage1_mean_s",
        "stage2_per_seed_rae",
        "stage2_bag_median_rae",
        "final_perq_rae",
        "anchor_rae_nb1070",
        "delta_vs_nb1070",
        "beats_nb1070",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
