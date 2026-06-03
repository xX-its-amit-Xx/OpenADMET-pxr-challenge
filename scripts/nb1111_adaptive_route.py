"""nb1111 -- Adaptive routing by top-1 Tanimoto similarity.

Hypothesis: chemprop_aux (nb1014 backbone) is trained on the 4139 train
distribution, where the prediction quality decays sharply for test compounds
whose nearest train neighbor has low Tanimoto similarity (off-manifold).
For those LOW-sim compounds, a chemistry-only ensemble of simple but
structurally complementary predictors (kNN Morgan + Avalon FP LGBM + nb1014)
gives a less-confident but more honest signal than nb1014 alone. For
HIGH-sim (>= 0.5) compounds, nb1014 already routes through neighbor
manifolds and its chemprop multi-task head dominates.

Procedure:
  1. Compute 513x4139 ECFP4 Tanimoto, take per-test top-1 sim.
  2. LOW group: top1 < 0.5  ; HIGH group: top1 >= 0.5.
  3. Routed prediction (513):
       LOW  -> mean(te_nb1104, te_nb1042, te_nb1014)
       HIGH -> te_nb1014
  4. Apply per-quantile median bag (nb1070 protocol, anchor = routed pred,
     5 seeds x 5 folds, 5 bins, stretch grid 0.80..2.00) globally on the
     combined output.
  5. Honest 5-fold cross-fit RAE on 253 unblind, compare to nb1070 (0.5771).

Outputs:
  data/processed/te_nb1111_adaptive_route.npy
  data/processed/nb1111_summary.json
  submissions/nb1111_adaptive_route.csv
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

from pxr.chem import morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1111_adaptive_route"
SIM_THRESH = 0.5
LOW_MEMBERS = ["nb1104", "nb1042", "nb1014"]
HIGH_ANCHOR = "nb1014"

# nb1070 per-quantile median-bag knobs
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

NB1070_REFERENCE = 0.5771


# ---------- Tanimoto top-1 ----------

def tanimoto_513x4139(fp_te: np.ndarray, fp_tr: np.ndarray) -> np.ndarray:
    """Vectorized Tanimoto via dot product on uint8 bit-vectors."""
    a = fp_te.astype(np.float32)
    b = fp_tr.astype(np.float32)
    inter = a @ b.T
    sa = a.sum(axis=1, keepdims=True)
    sb = b.sum(axis=1, keepdims=True).T
    union = sa + sb - inter
    union[union == 0] = 1.0
    return inter / union


# ---------- nb1070 per-quantile median-bag helpers ----------

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
                 ) -> tuple[float, np.ndarray, list]:
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
    return float(rae(y_unb, oof)), oof, per_fold_ss


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- adaptive route by top-1 Tanimoto + global median bag")
    print("=" * 78)

    te = load_test()
    tr = load_train()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    # ---- 1. Compute 513x4139 Tanimoto top-1 ----
    print("\n[sim] computing 513x4139 Morgan Tanimoto -> top-1 per test row")
    t_f = time.time()
    fp_tr = morgan_fp_batch(tr["smiles"].tolist())
    fp_te = morgan_fp_batch(te["smiles"].tolist())
    print(f"[sim] fp_tr={fp_tr.shape}  fp_te={fp_te.shape}  "
          f"({time.time()-t_f:.1f}s)")
    t_s = time.time()
    sim = tanimoto_513x4139(fp_te, fp_tr)
    top1 = sim.max(axis=1).astype(np.float64)
    print(f"[sim] top1 shape={top1.shape}  mean={top1.mean():.3f}  "
          f"median={np.median(top1):.3f}  min={top1.min():.3f}  "
          f"max={top1.max():.3f}  ({time.time()-t_s:.1f}s)")

    # ---- 2. LOW / HIGH membership ----
    low_mask = top1 < SIM_THRESH
    high_mask = ~low_mask
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())
    print(f"\n[route] LOW (top1 < {SIM_THRESH}): n={n_low}  "
          f"HIGH (top1 >= {SIM_THRESH}): n={n_high}")

    # ---- 3. Load member predictions, build routed te(513) ----
    te_1104 = np.load(DATA_PROCESSED / "te_nb1104.npy").astype(np.float64)
    te_1042 = np.load(DATA_PROCESSED / "te_nb1042.npy").astype(np.float64)
    te_1014 = np.load(DATA_PROCESSED / "te_nb1014.npy").astype(np.float64)
    print(f"[load] te_nb1104 mean/std = {te_1104.mean():.3f}/{te_1104.std():.3f}")
    print(f"[load] te_nb1042 mean/std = {te_1042.mean():.3f}/{te_1042.std():.3f}")
    print(f"[load] te_nb1014 mean/std = {te_1014.mean():.3f}/{te_1014.std():.3f}")

    low_blend = (te_1104 + te_1042 + te_1014) / 3.0
    routed = np.where(low_mask, low_blend, te_1014).astype(np.float64)
    print(f"[route] routed mean/std = {routed.mean():.3f}/{routed.std():.3f}")

    # ---- 4. Cross-fit diagnostics on 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    print(f"\n[load] unb_idx n={len(unb_idx)}  y_unb n={len(y_unb)}")

    in_rae_routed = float(rae(y_unb, routed[unb_idx]))
    in_rae_1014 = float(rae(y_unb, te_1014[unb_idx]))
    in_rae_low_blend = float(rae(y_unb, low_blend[unb_idx]))
    print(f"\n[in_rae] te_nb1014          = {in_rae_1014:.4f}")
    print(f"[in_rae] low_blend (3-way)  = {in_rae_low_blend:.4f}")
    print(f"[in_rae] routed             = {in_rae_routed:.4f}")

    # Per-group in-sample RAE on overlap with 253
    unb_low = low_mask[unb_idx]
    unb_high = high_mask[unb_idx]
    n_unb_low = int(unb_low.sum())
    n_unb_high = int(unb_high.sum())
    if n_unb_low >= 2:
        rae_low_route = float(rae(y_unb[unb_low], routed[unb_idx][unb_low]))
        rae_low_1014 = float(rae(y_unb[unb_low], te_1014[unb_idx][unb_low]))
    else:
        rae_low_route = rae_low_1014 = float("nan")
    if n_unb_high >= 2:
        rae_high_route = float(rae(y_unb[unb_high], routed[unb_idx][unb_high]))
        rae_high_1014 = float(rae(y_unb[unb_high], te_1014[unb_idx][unb_high]))
    else:
        rae_high_route = rae_high_1014 = float("nan")
    print(f"\n[group] LOW  (n={n_unb_low:3d}):  route={rae_low_route:.4f}  "
          f"nb1014={rae_low_1014:.4f}  delta={rae_low_route-rae_low_1014:+.4f}")
    print(f"[group] HIGH (n={n_unb_high:3d}):  route={rae_high_route:.4f}  "
          f"nb1014={rae_high_1014:.4f}  delta={rae_high_route-rae_high_1014:+.4f}")

    # ---- 5. Global per-quantile median bag on routed anchor ----
    print("\n" + "-" * 78)
    print("PER-QUANTILE MEDIAN BAG (global, nb1070 protocol)")
    print(f"  seeds={SEEDS}  n_folds={N_FOLDS}  n_bins={N_BINS}")
    print("-" * 78)
    p_unb = routed[unb_idx]
    oof_stack = np.zeros((len(SEEDS), len(y_unb)), dtype=np.float64)
    per_seed_rae = []
    seed_records = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, per_fold_ss = run_one_seed(seed, p_unb, y_unb)
        oof_stack[i] = oof
        per_seed_rae.append(pooled)
        seed_records.append({
            "seed": seed,
            "pooled_rae": pooled,
            "fold_ss_mean": np.array(per_fold_ss).mean(axis=0).tolist(),
        })
        print(f"   seed {seed:>3d}: pooled_RAE = {pooled:.4f}")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"\n[bag] per-seed mean={per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[bag] MEDIAN  bagged_oof RAE = {bag_median_rae:.4f}")
    print(f"[bag] MEAN    bagged_oof RAE = {bag_mean_rae:.4f}")

    # ---- 6. Deploy median bag on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY  (per-seed fit on all 253, median bag of 513 preds)")
    print("-" * 78)
    qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    edges_all = np.quantile(p_unb, qs_all)
    mus_all, _ = fit_per_bin_stretch(p_unb, y_unb, edges_all)
    deploy_stack = np.zeros((len(SEEDS), n_te), dtype=np.float64)
    for i, rec in enumerate(seed_records):
        ss_seed = np.array(rec["fold_ss_mean"])
        deploy_stack[i] = apply_per_bin_stretch(routed, edges_all,
                                                 mus_all, ss_seed)
    deploy_513 = np.median(deploy_stack, axis=0).astype(np.float32)
    in_rae_deploy = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
    print(f"   deploy_513 mean/std = "
          f"{deploy_513.mean():.3f}/{deploy_513.std():.3f}")
    print(f"   in-sample 253 RAE   = {in_rae_deploy:.4f}")

    # ---- Save ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    # ---- Verdict ----
    delta = bag_median_rae - NB1070_REFERENCE
    beats = bag_median_rae < NB1070_REFERENCE
    if delta < -0.005:
        verdict = "BEATS_NB1070"
    elif abs(delta) <= 0.005:
        verdict = "TIES_NB1070"
    else:
        verdict = "WORSE_THAN_NB1070"
    print(f"\n[verdict] vs nb1070 ({NB1070_REFERENCE:.4f}): "
          f"delta={delta:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "sim_thresh": SIM_THRESH,
        "low_members": LOW_MEMBERS,
        "high_anchor": HIGH_ANCHOR,
        "n_low": n_low,
        "n_high": n_high,
        "top1_mean": float(top1.mean()),
        "top1_median": float(np.median(top1)),
        "top1_min": float(top1.min()),
        "top1_max": float(top1.max()),
        "in_rae_routed_on_253": in_rae_routed,
        "in_rae_nb1014_on_253": in_rae_1014,
        "in_rae_low_blend_on_253": in_rae_low_blend,
        "n_unb_low": n_unb_low,
        "n_unb_high": n_unb_high,
        "rae_low_route_unb": rae_low_route,
        "rae_low_nb1014_unb": rae_low_1014,
        "rae_high_route_unb": rae_high_route,
        "rae_high_nb1014_unb": rae_high_1014,
        "per_seed_rae": per_seed_rae,
        "per_seed_mean": per_seed_mean,
        "per_seed_std": per_seed_std,
        "bag_median_rae": bag_median_rae,
        "bag_mean_rae": bag_mean_rae,
        "in_rae_deploy_on_253": in_rae_deploy,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "nb1070_reference": NB1070_REFERENCE,
        "delta_vs_nb1070": delta,
        "beats_nb1070": bool(beats),
        "verdict": verdict,
        "seed_records": seed_records,
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG.split('_')[0]}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   n_low / n_high             = {n_low} / {n_high}")
    print(f"   routed in_RAE (253)        = {in_rae_routed:.4f}")
    print(f"   per-seed mean / std        = "
          f"{per_seed_mean:.4f} / {per_seed_std:.4f}")
    print(f"   bag MEDIAN  cross-fit RAE  = {bag_median_rae:.4f}")
    print(f"   bag MEAN    cross-fit RAE  = {bag_mean_rae:.4f}")
    print(f"   nb1070 reference           = {NB1070_REFERENCE:.4f}")
    print(f"   delta                      = {delta:+.4f}  -> {verdict}")
    print(f"   in-sample deploy (253)     = {in_rae_deploy:.4f}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("n_low", "n_high", "rae_low_route_unb", "rae_low_nb1014_unb",
              "rae_high_route_unb", "rae_high_nb1014_unb",
              "per_seed_rae", "bag_median_rae", "bag_mean_rae",
              "delta_vs_nb1070", "beats_nb1070", "verdict",
              "in_rae_deploy_on_253", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
