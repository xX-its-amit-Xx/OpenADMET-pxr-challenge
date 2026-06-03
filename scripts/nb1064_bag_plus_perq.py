"""nb1064 -- FULL integration: SLSQP blend + per-quantile-bin stretch + multi-seed bag.

Combines two protocols that were previously evaluated separately:
  - nb1014: per-fold SLSQP w0 (chemprop_aux vs nb972) + scalar stretch s,
    bagged across 5 KFold seeds.
  - nb1053/nb1060: per-fold per-quantile-bin stretch (5 bins, learn 5 s_b),
    bagged across 5 KFold seeds.

nb1064 fuses them INSIDE each fold: SLSQP w0 first to get a blended train
fold, then 5-bin per-quantile stretch on the blended train fold. Per-fold
capacity = 1 (w0) + 5 (s_b) = 6 hyperparameters, with edges/mu_b/s_b ALL
fit train-only to keep cross-fit honest.

Procedure (per seed s in SEEDS):
  KFold(N_FOLDS=5, shuffle=True, random_state=s) on the 253 unblind:
    For each fold f:
      a. SLSQP for w0_f on the 4 train folds (chemprop_aux vs nb972).
      b. blend_tr = w0_f*p0_tr + (1-w0_f)*p1_tr.
      c. Quantile bin edges from blend_tr (20/40/60/80 percentiles).
      d. For each bin b in {0..4}: mu_b = blend_tr[bin_tr==b].mean();
         grid-scan s_b in STRETCH_GRID minimizing train-fold RAE on bin.
      e. blend_va = w0_f*p0_va + (1-w0_f)*p1_va.
      f. Apply per-bin stretch to blend_va using TRAIN edges, mu_b, s_b.
  Per-seed pooled cross-fit RAE on the 253; OOF vector cached.
Bag:
  Stack 5 OOF vectors -> (5, 253); mean across seed axis -> bagged_oof;
  RAE(y_unb, bagged_oof) = bag_mean_rae.

This is purely diagnostic -- no deploy CSV/te.npy is emitted. Gate vs the
prior bagged-honest reference: nb1053 (single-seed) pooled 0.5780 and nb1060
(per-quantile bag) bagged 0.5780. The integration is "FULL" only if it beats
these without inflating overfit.

Outputs:
  data/processed/nb1064_summary.json
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

TAG = "nb1064"
CANDIDATES = ["chemprop_aux", "nb972_long_train"]
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

NB1053_SEED42_RAE = 0.5780
NB1060_BAG_RAE = 0.5780
NB1014_BAGGED_HONEST_RAE = 0.5930


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_w0(p0: np.ndarray, p1: np.ndarray, y: np.ndarray) -> float:
    P = np.column_stack([p0, p1])
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0), (0.0, 1.0)]
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.array([0.5, 0.5]),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return float(res.x[0])


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


def run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray, seed: int
                 ) -> tuple[float, np.ndarray, list[dict]]:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        # Step 1: SLSQP blend weight on train fold.
        w0_f = slsqp_w0(P_unb[tr_loc, 0], P_unb[tr_loc, 1], y_unb[tr_loc])
        blend_tr = w0_f * P_unb[tr_loc, 0] + (1.0 - w0_f) * P_unb[tr_loc, 1]
        blend_va = w0_f * P_unb[va_loc, 0] + (1.0 - w0_f) * P_unb[va_loc, 1]
        # Step 2: per-quantile-bin stretch on blended train fold.
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(blend_tr, qs)
        mus, ss = fit_per_bin_stretch(blend_tr, y_unb[tr_loc], edges)
        # Step 3: apply train-fit (edges, mus, ss) to held-out fold.
        oof[va_loc] = apply_per_bin_stretch(blend_va, edges, mus, ss)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        fold_records.append({
            "fold": k, "w0": w0_f, "edges": edges.tolist(),
            "mus": mus.tolist(), "ss": ss.tolist(),
            "val_rae": rae_va, "n_va": int(len(va_loc)),
        })
    pooled = float(rae(y_unb, oof))
    return pooled, oof, fold_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- FULL integration: SLSQP blend + per-quantile stretch + bag")
    print(f"   pool={CANDIDATES}  N_BINS={N_BINS}  N_FOLDS={N_FOLDS}  "
          f"seeds={SEEDS}")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] preds_513={preds_513.shape}  P_unb={P_unb.shape}  "
          f"y_unb={y_unb.shape}")

    indiv_rae = {c: float(rae(y_unb, P_unb[:, j]))
                 for j, c in enumerate(CANDIDATES)}
    for c, r in indiv_rae.items():
        print(f"   in_RAE {c:30s} = {r:.4f}")

    print("\n" + "-" * 78)
    print(f"PER-SEED HONEST CROSS-FIT (6 params/fold: 1 w0 + {N_BINS} s_b)")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, folds = run_one_seed(P_unb, y_unb, seed)
        oof_stack[i] = oof
        per_seed_rae.append(pooled)
        w0_arr = np.array([f["w0"] for f in folds])
        ss_arr = np.array([f["ss"] for f in folds])  # (N_FOLDS, N_BINS)
        ss_mean_str = ",".join(f"{x:.2f}" for x in ss_arr.mean(axis=0))
        seed_records.append({
            "seed": seed,
            "pooled_rae": pooled,
            "w0_mean": float(w0_arr.mean()),
            "w0_std": float(w0_arr.std()),
            "ss_mean": ss_arr.mean(axis=0).tolist(),
            "ss_std": ss_arr.std(axis=0).tolist(),
            "folds": folds,
        })
        print(f"   seed {seed:>3d}: pooled_RAE = {pooled:.4f}  "
              f"w0_mean={w0_arr.mean():.3f}  ss_mean=[{ss_mean_str}]")

    per_seed_rae_arr = np.array(per_seed_rae)
    per_seed_mean = float(per_seed_rae_arr.mean())
    per_seed_std = float(per_seed_rae_arr.std())
    per_seed_min = float(per_seed_rae_arr.min())
    per_seed_max = float(per_seed_rae_arr.max())
    print(f"\n[per-seed] mean={per_seed_mean:.4f}  std={per_seed_std:.4f}  "
          f"min={per_seed_min:.4f}  max={per_seed_max:.4f}")

    bagged_oof = oof_stack.mean(axis=0)
    bag_rae = float(rae(y_unb, bagged_oof))
    print(f"[bag]      bagged_oof RAE (mean of {len(SEEDS)} OOFs) = "
          f"{bag_rae:.4f}")

    delta_vs_nb1053 = bag_rae - NB1053_SEED42_RAE
    delta_vs_nb1060 = bag_rae - NB1060_BAG_RAE
    delta_vs_nb1014 = bag_rae - NB1014_BAGGED_HONEST_RAE
    beats_nb1053 = bag_rae < NB1053_SEED42_RAE

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1053 (seed=42)        = {NB1053_SEED42_RAE:.4f}")
    print(f"   nb1060 bag              = {NB1060_BAG_RAE:.4f}")
    print(f"   nb1014 bag (no perq)    = {NB1014_BAGGED_HONEST_RAE:.4f}")
    print(f"   nb1064 per-seed mean    = {per_seed_mean:.4f}  "
          f"(delta vs nb1053 = {per_seed_mean - NB1053_SEED42_RAE:+.4f})")
    print(f"   nb1064 bagged OOF       = {bag_rae:.4f}  "
          f"(delta vs nb1053 = {delta_vs_nb1053:+.4f}, "
          f"vs nb1060 = {delta_vs_nb1060:+.4f}, "
          f"vs nb1014 = {delta_vs_nb1014:+.4f})")
    print(f"   beats_nb1053            = {beats_nb1053}")

    summary = {
        "tag": TAG,
        "candidates": CANDIDATES,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stretch_grid": STRETCH_GRID,
        "indiv_in_rae_on_253": indiv_rae,
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_std": per_seed_std,
        "per_seed_rae_min": per_seed_min,
        "per_seed_rae_max": per_seed_max,
        "bag_mean_rae": bag_rae,
        "delta_vs_nb1053": delta_vs_nb1053,
        "delta_vs_nb1060": delta_vs_nb1060,
        "delta_vs_nb1014_bagged": delta_vs_nb1014,
        "beats_nb1053": bool(beats_nb1053),
        "nb1053_seed42_rae": NB1053_SEED42_RAE,
        "nb1060_bag_rae": NB1060_BAG_RAE,
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
    for k in ("per_seed_rae", "per_seed_rae_mean", "per_seed_rae_std",
              "bag_mean_rae", "delta_vs_nb1053", "delta_vs_nb1060",
              "beats_nb1053"):
        print(f"  {k}: {res.get(k)}")
