"""nb1150 -- DEPLOY artifact for nb1143 (per-quantile median-bag on nb1140).

Companion to nb1143:
  - nb1140 is the 513 deploy vector of nb1130 (mean-bag of 5 residual-LGBM
    seeds on top of nb1070).
  - nb1143 ran the nb1070 per-quantile median-bag stretch protocol on the
    nb1130 mean-bag OOF (253) as an honest cross-fit diagnostic.
  - nb1150 = the matching DEPLOY artefact: apply that same per-quantile
    median-bag stretch protocol to the nb1140 deploy predictions (513).

Protocol (matches nb1070 deploy logic):
  N_BINS=5, STRETCH_GRID=0.80..2.00 step 0.05, KFold(5, shuffle), seeds in
  {0, 1, 7, 42, 137}.

  For each seed s:
    1. KFold(n=5, shuffle=True, random_state=s) on the 253 unblind subset
       of te_nb1140 (= te_nb1140[unb_idx]).
    2. For each fold, fit train-only quantile edges + per-bin (mu_b, s_b)
       via grid search minimising fold-train RAE.
    3. Mean the per-fold ss across the 5 folds -> ss_seed (5,).
    4. Deploy edges = quantile of the full p_unb (253). Deploy mus from
       fit_per_bin_stretch on all 253 with those edges. Apply
       (edges, mus, ss_seed) to the full 513 te_nb1140 -> deploy_seed_513.
  Median-bag across the 5 seed deploy vectors at row level -> te_nb1150.

Outputs:
  data/processed/te_nb1150.npy             (513,) float32
  submissions/nb1150_deploy_nb1143.csv     (SMILES, Molecule Name, pEC50)
  data/processed/nb1150_summary.json

Caveat (per feedback_lb_two_regime_calibration): this is a POST-unblind
deploy artefact -- in_RAE on te_nb1150[unb_idx] is in-sample and
optimistic. The LB-faithful number is the nb1143 honest cross-fit RAE
(`bag_median_rae` in nb1143_summary.json).
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

TAG = "nb1150"
ANCHOR = "nb1140"           # deploy companion of nb1130
PERQ_REF = "nb1143"         # honest cross-fit reference

# nb1070 per-quantile-stretch protocol (verbatim).
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]


# ---------- nb1070 per-quantile-stretch primitives (verbatim) ----------

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


def run_one_seed_cv(seed: int, p_unb: np.ndarray, y_unb: np.ndarray
                    ) -> tuple[float, np.ndarray, list[list[float]]]:
    """Honest KFold(seed) cross-fit on 253; returns pooled RAE, OOF, per-fold ss."""
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
    print(f"{TAG} -- DEPLOY per-quantile median-bag stretch on te_{ANCHOR}")
    print(f"          seeds = {SEEDS}   N_BINS = {N_BINS}   "
          f"stretch_grid = {STRETCH_GRID[0]:.2f}..{STRETCH_GRID[-1]:.2f} step 0.05")
    print(f"          honest cross-fit reference: {PERQ_REF}_summary.json")
    print("=" * 78)

    # ---- Load 513 test names + anchor deploy + unblind index/truth ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)

    te_anchor_path = DATA_PROCESSED / f"te_{ANCHOR}.npy"
    unb_idx_path = DATA_PROCESSED / "_audit_unblind_idx.npy"
    y_unb_path = DATA_PROCESSED / "_audit_unblind_y.npy"
    if not te_anchor_path.exists():
        raise FileNotFoundError(
            f"missing {te_anchor_path} -- run scripts/nb{ANCHOR[2:]}_deploy_nb1130.py first")

    te_anchor = np.load(te_anchor_path).astype(np.float64)
    unb_idx = np.load(unb_idx_path)
    y_unb = np.load(y_unb_path).astype(np.float64)
    n_unb = len(y_unb)
    assert te_anchor.shape[0] == n_test, (
        f"te_{ANCHOR} shape {te_anchor.shape} mismatch n_test={n_test}")

    p_unb = te_anchor[unb_idx]
    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"[load] te_{ANCHOR}.npy shape={te_anchor.shape}  "
          f"p_unb shape={p_unb.shape}  y shape={y_unb.shape}")
    print(f"[base] in_RAE(te_{ANCHOR}[unb_idx]) = {in_rae_anchor:.4f}")
    print(f"[diag] anchor p_unb std={p_unb.std():.4f}   "
          f"truth std={y_unb.std():.4f}   "
          f"compression ratio={p_unb.std() / y_unb.std():.3f}")

    # ---- Per-seed honest cross-fit (diagnostic + ss source for deploy) ----
    print("\n" + "-" * 78)
    print("PER-SEED HONEST CROSS-FIT  (anchor = te_{}[unb_idx])".format(ANCHOR))
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_ss_means: list[list[float]] = []
    seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, per_fold_ss = run_one_seed_cv(seed, p_unb, y_unb)
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
    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"\n[per-seed] RAE mean={per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[bag]      OOF MEDIAN bagged_RAE = {bag_median_rae:.4f}")
    print(f"[bag]      OOF MEAN   bagged_RAE = {bag_mean_rae:.4f}")

    # =================================================================
    # DEPLOY: per-seed deploy_513 = apply(edges_all, mus_all, ss_seed) to
    # te_anchor (513). Median-bag across 5 seeds.
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (per-seed fit on all 253, median-bag the 513 predictions)")
    print("-" * 78)
    qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    edges_all = np.quantile(p_unb, qs_all)
    mus_all, _ = fit_per_bin_stretch(p_unb, y_unb, edges_all)
    print(f"   deploy edges (all 253): {[f'{e:.3f}' for e in edges_all]}")
    print(f"   deploy mus   (all 253): {[f'{m:.3f}' for m in mus_all]}")

    deploy_stack = np.zeros((len(SEEDS), n_test), dtype=np.float64)
    per_seed_deploy_ss: list[list[float]] = []
    per_seed_deploy_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        rec = seed_records[i]
        per_fold_ss_arr = np.array(rec["per_fold_ss"])      # (5, N_BINS)
        ss_seed = per_fold_ss_arr.mean(axis=0)
        deploy_seed_513 = apply_per_bin_stretch(te_anchor, edges_all,
                                                mus_all, ss_seed)
        deploy_stack[i] = deploy_seed_513
        per_seed_deploy_ss.append(ss_seed.tolist())
        in_rae_deploy_seed = float(rae(y_unb,
                                       deploy_seed_513[unb_idx]))
        per_seed_deploy_records.append({
            "seed": int(seed),
            "ss_seed": ss_seed.tolist(),
            "deploy_513_mean": float(deploy_seed_513.mean()),
            "deploy_513_std": float(deploy_seed_513.std()),
            "in_rae_253": in_rae_deploy_seed,
        })
        ss_str = ",".join(f"{x:.2f}" for x in ss_seed)
        print(f"   seed {seed:>3d}: deploy ss=[{ss_str}]  "
              f"deploy_513 mean={deploy_seed_513.mean():.3f}  "
              f"std={deploy_seed_513.std():.3f}  "
              f"in_RAE(253)={in_rae_deploy_seed:.4f}")

    # ---- MEDIAN bag across 5 seed deploy vectors ----
    te_nb1150 = np.median(deploy_stack, axis=0)
    deploy_mean_513 = deploy_stack.mean(axis=0)
    in_rae_deploy_median = float(rae(y_unb, te_nb1150[unb_idx]))
    in_rae_deploy_mean = float(rae(y_unb, deploy_mean_513[unb_idx]))
    delta_vs_anchor_te = in_rae_deploy_median - in_rae_anchor
    print(f"\n[bag] te_nb1150 MEDIAN-bag  mean={te_nb1150.mean():.3f}  "
          f"std={te_nb1150.std():.3f}  in_RAE(253)={in_rae_deploy_median:.4f}")
    print(f"[bag] deploy MEAN-bag        mean={deploy_mean_513.mean():.3f}  "
          f"std={deploy_mean_513.std():.3f}  in_RAE(253)={in_rae_deploy_mean:.4f}")
    print(f"[bag] anchor te_{ANCHOR}        mean={te_anchor.mean():.3f}  "
          f"std={te_anchor.std():.3f}  in_RAE(253)={in_rae_anchor:.4f}")
    print(f"[bag] delta in-sample vs anchor = {delta_vs_anchor_te:+.4f}")

    # ---- Save te artefact ----
    te_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_out, te_nb1150.astype(np.float32))
    print(f"\n[save] {te_out}")

    # ---- Save submission CSV ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_nb1150.astype(np.float64),
    })
    sub_path = SUBMISSIONS / f"{TAG}_deploy_nb1143.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[save] {sub_path}  rows={len(sub)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "deploy_companion_of": PERQ_REF,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stretch_grid": STRETCH_GRID,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "in_rae_anchor_te_253": in_rae_anchor,
        "anchor_p_unb_std": float(p_unb.std()),
        "truth_std": float(y_unb.std()),
        "anchor_compression_ratio": float(p_unb.std() / y_unb.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_std": per_seed_std,
        "bag_median_oof_rae": bag_median_rae,
        "bag_mean_oof_rae": bag_mean_rae,
        "per_seed_ss_means_across_folds": per_seed_ss_means,
        "per_seed_deploy_ss": per_seed_deploy_ss,
        "deploy_edges_all_253": edges_all.tolist(),
        "deploy_mus_all_253": mus_all.tolist(),
        "in_rae_deploy_median_253": in_rae_deploy_median,
        "in_rae_deploy_mean_253": in_rae_deploy_mean,
        "delta_in_sample_vs_anchor": delta_vs_anchor_te,
        "te_nb1150_mean": float(te_nb1150.mean()),
        "te_nb1150_std": float(te_nb1150.std()),
        "anchor_te_mean": float(te_anchor.mean()),
        "anchor_te_std": float(te_anchor.std()),
        "te_path": str(te_out),
        "submission_path": str(sub_path),
        "seed_records": seed_records,
        "per_seed_deploy_records": per_seed_deploy_records,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "DEPLOY artefact: per-seed KFold(seed) on 253 gives per-fold ss; "
            "ss_seed = mean across folds; deploy edges/mus fit on all 253; "
            "apply (edges, mus, ss_seed) to 513 te_anchor; median-bag across "
            "5 seeds. in_RAE is in-sample and optimistic; LB-faithful number "
            "is nb1143 honest cross-fit RAE (bag_median_rae in nb1143_summary)."
        ),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "in_rae_anchor_te_253",
        "per_seed_rae",
        "per_seed_rae_mean",
        "per_seed_rae_std",
        "bag_median_oof_rae",
        "bag_mean_oof_rae",
        "in_rae_deploy_median_253",
        "in_rae_deploy_mean_253",
        "delta_in_sample_vs_anchor",
        "te_nb1150_mean",
        "te_nb1150_std",
        "te_path",
        "submission_path",
    ):
        print(f"  {k}: {res.get(k)}")
