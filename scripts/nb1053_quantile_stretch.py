"""nb1053 -- Per-quantile-bin rank-stretch on the nb1014 deploy point.

nb562's scalar stretch `mu + s*(p - mu)` decompresses variance globally with
one parameter. That works when compression is roughly uniform across the
prediction range, but variance compression on PXR is suspected to be heaviest
at the tails -- low-pec50 over-predictions get squeezed up toward the mean
and high-pec50 under-predictions get squeezed down. A scalar s averages
across all of that and under-corrects the tails.

Hypothesis: split te_nb1014 into 5 quantile bins by prediction value and
fit a separate stretch factor `s_b` per bin. Each bin gets its own
within-bin mean `mu_b` and stretch `s_b`, applied as
    p' = mu_b + s_b * (p - mu_b)
The blend's global anchor is preserved while tails get a heavier multiplier.

5 parameters at n ~= 50/bin per held-out fold is risky -- this is the same
regime where nb571/572/580 per-quantile tries previously overfit. Honest
5-fold cross-fit on the 253 unblind is mandatory; per-bin s is grid-fit on
the train folds only, bin edges are also computed train-only to avoid
leakage. Verdict gate: beats nb1014 bagged honest CV (0.5930) by >=0.005.

Procedure:
  1. Load te_nb1014.npy (513 deploy predictions).
  2. Subset to 253 unblind indices (in-sample anchor predictions).
  3. 5-fold KFold(seed=42) on the 253:
       a. From train-fold predictions, compute 5 quantile bin edges
          (20th/40th/60th/80th percentiles).
       b. For each of the 5 bins:
            i. Compute mu_b = mean of train preds in this bin.
           ii. Grid-scan s_b in {0.80, 0.85, ..., 2.00} minimizing
                train-fold RAE on the bin.
       c. Apply (mu_b, s_b) to held-out fold using TRAIN-fold bin edges
          (no leakage from val edges).
  4. Pooled cross-fit RAE on the 253.
  5. Deploy: fit (edges, mu_b, s_b) once on the 253 unblind, apply to all 513.
  6. Save te_nb1053.npy, summary, submission CSV.

Outputs:
  data/processed/te_nb1053.npy
  data/processed/nb1053_summary.json
  submissions/nb1053_quantile_stretch.csv
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

TAG = "nb1053"
ANCHOR = "nb1014"
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42

# Reference numbers from prior honest cross-fit work.
NB1014_BAGGED_HONEST_RAE = 0.5930  # mean over 5 seeds in nb1014_summary
NB1014_SEED42_RAE = 0.5994         # single-seed nb1014 / nb1001
NB562_HONEST_RAE = 0.5065          # scalar-stretch on nb503 (different anchor)


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values to bins 0..N_BINS-1 given internal edges of length N_BINS-1.

    Values <= edges[0] go to bin 0; values > edges[-1] go to bin N_BINS-1.
    np.digitize with right=False gives this exact behavior.
    """
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid-fit (mu_b, s_b) per bin on train data.

    Returns
    -------
    mus : (N_BINS,) array of per-bin train means
    ss  : (N_BINS,) array of per-bin best stretch factors
    """
    bins_tr = bin_assign(p_train, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        n_b = int(mask.sum())
        if n_b < 2:
            # Degenerate bin -- identity transform.
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
    """Apply per-bin (mu_b, s_b) using the given bin edges."""
    bins = bin_assign(p, edges)
    out = np.empty_like(p, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins == b
        if not mask.any():
            continue
        out[mask] = mus[b] + ss[b] * (p[mask] - mus[b])
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-quantile-bin stretch on te_{ANCHOR}")
    print("=" * 78)

    # ---- 513 test load ----
    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    print(f"[load] te_{ANCHOR} shape = {preds_513.shape}  "
          f"mean={preds_513.mean():.3f}  std={preds_513.std():.3f}")

    # ---- 253 unblind subset ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] p_unb shape = {p_unb.shape}, y shape = {y_unb.shape}")
    print(f"[load] truth_std={y_unb.std():.4f}  pred_std={p_unb.std():.4f}  "
          f"(compression ratio = {p_unb.std() / y_unb.std():.3f})")

    in_rae_baseline = float(rae(y_unb, p_unb))
    print(f"\n[baseline] in_RAE(te_{ANCHOR} on 253 unblind) = "
          f"{in_rae_baseline:.4f}")

    # =================================================================
    # Honest 5-fold cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"HONEST CROSS-FIT  (KFold seed={SEED}, N_BINS={N_BINS}, "
          f"grid {STRETCH_GRID[0]}..{STRETCH_GRID[-1]} step 0.05)")
    print("-" * 78)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va, y_va = p_unb[va_loc], y_unb[va_loc]
        # Train-only quantile edges -- no val contamination.
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        rae_tr_global = float(rae(y_tr,
                                   apply_per_bin_stretch(p_tr, edges, mus, ss)))
        pred_va = apply_per_bin_stretch(p_va, edges, mus, ss)
        oof[va_loc] = pred_va
        rae_va = float(rae(y_va, pred_va))
        # Per-bin val counts for diagnostics.
        bins_va = bin_assign(p_va, edges)
        bin_counts_va = [int((bins_va == b).sum()) for b in range(N_BINS)]
        folds.append({
            "fold": k,
            "edges": edges.tolist(),
            "mus": mus.tolist(),
            "ss": ss.tolist(),
            "train_rae": rae_tr_global,
            "val_rae": rae_va,
            "n_va": int(len(va_loc)),
            "bin_counts_va": bin_counts_va,
        })
        ss_str = ",".join(f"{s:.2f}" for s in ss)
        print(f"   fold {k}: train_RAE={rae_tr_global:.4f}  "
              f"val_RAE={rae_va:.4f}  ss=[{ss_str}]  "
              f"val_bin_counts={bin_counts_va}")

    pooled = float(rae(y_unb, oof))
    print(f"\n[honest] pooled cross-fit RAE on 253 unblind = {pooled:.4f}")

    # =================================================================
    # Per-bin diagnostic: where does this gain (or lose) vs anchor?
    # =================================================================
    # Refit per-bin s on ALL 253 just for diagnostic decomposition.
    qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    edges_all = np.quantile(p_unb, qs_all)
    mus_all, ss_all = fit_per_bin_stretch(p_unb, y_unb, edges_all)
    bins_unb = bin_assign(p_unb, edges_all)
    print("\n[diag] per-bin in-sample fit on 253 (deploy bin edges):")
    print(f"   bin  n    p_mean   y_mean   s_best   anchor_AE   stretched_AE")
    for b in range(N_BINS):
        mask = bins_unb == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        p_b = p_unb[mask]
        y_b = y_unb[mask]
        anchor_ae = float(np.mean(np.abs(p_b - y_b)))
        stretched = mus_all[b] + ss_all[b] * (p_b - mus_all[b])
        stretched_ae = float(np.mean(np.abs(stretched - y_b)))
        print(f"   {b:>3d}  {n_b:>3d}  {p_b.mean():>6.3f}   "
              f"{y_b.mean():>6.3f}   {ss_all[b]:>5.2f}   "
              f"{anchor_ae:>8.4f}   {stretched_ae:>10.4f}")

    # =================================================================
    # Deploy: fit edges + per-bin (mu, s) once on all 253, apply to 513.
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (fit on all 253, apply to 513)")
    print("-" * 78)
    deploy_edges = edges_all
    deploy_mus = mus_all
    deploy_ss = ss_all
    deploy_513 = apply_per_bin_stretch(preds_513, deploy_edges,
                                        deploy_mus, deploy_ss).astype(np.float32)
    in_rae_deploy = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
    print(f"   deploy edges   = {deploy_edges.tolist()}")
    print(f"   deploy mus     = "
          f"{[round(float(x), 3) for x in deploy_mus]}")
    print(f"   deploy ss      = "
          f"{[round(float(x), 2) for x in deploy_ss]}")
    print(f"   in-sample 253  = {in_rae_deploy:.4f}  (overfit lower bound)")
    print(f"   te(513) mean   = {deploy_513.mean():.3f}  "
          f"std={deploy_513.std():.3f}")
    print(f"   anchor te(513) mean   = {preds_513.mean():.3f}  "
          f"std={preds_513.std():.3f}")

    # =================================================================
    # Save artifacts
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_quantile_stretch.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_nb1014 = pooled - NB1014_BAGGED_HONEST_RAE
    if delta_vs_nb1014 <= -0.005:
        verdict = "BEATS_NB1014"
    elif abs(delta_vs_nb1014) < 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"
    print(f"\n[verdict] pooled CV vs nb1014 bagged ({NB1014_BAGGED_HONEST_RAE}): "
          f"delta={delta_vs_nb1014:+.4f}  -> {verdict}")

    per_bin_s_mean = [float(np.mean([f["ss"][b] for f in folds]))
                      for b in range(N_BINS)]
    per_bin_s_std = [float(np.std([f["ss"][b] for f in folds]))
                     for b in range(N_BINS)]
    print(f"\n[bag] mean s per bin across {N_FOLDS} folds = "
          f"{[round(x, 3) for x in per_bin_s_mean]}")
    print(f"[bag] std  s per bin across {N_FOLDS} folds = "
          f"{[round(x, 3) for x in per_bin_s_std]}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_bins": N_BINS,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "in_rae_anchor_on_253": in_rae_baseline,
        "pooled_cross_fit_rae": pooled,
        "in_rae_deploy_on_253": in_rae_deploy,
        "nb1014_bagged_honest_rae": NB1014_BAGGED_HONEST_RAE,
        "nb1014_seed42_rae": NB1014_SEED42_RAE,
        "delta_vs_nb1014_bagged": delta_vs_nb1014,
        "verdict": verdict,
        "deploy_edges": deploy_edges.tolist(),
        "deploy_mus": deploy_mus.tolist(),
        "deploy_ss": deploy_ss.tolist(),
        "per_bin_s_mean_across_folds": per_bin_s_mean,
        "per_bin_s_std_across_folds": per_bin_s_std,
        "folds": folds,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                     = {ANCHOR}")
    print(f"   in_RAE anchor on 253       = {in_rae_baseline:.4f}")
    print(f"   pooled cross-fit RAE       = {pooled:.4f}")
    print(f"   in-sample (deploy)         = {in_rae_deploy:.4f}")
    print(f"   nb1014 bagged honest CV    = {NB1014_BAGGED_HONEST_RAE:.4f}")
    print(f"   delta vs nb1014 bagged     = {delta_vs_nb1014:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   per-bin s (folds mean)     = "
          f"{[round(x, 3) for x in per_bin_s_mean]}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("in_rae_anchor_on_253", "pooled_cross_fit_rae",
              "in_rae_deploy_on_253", "delta_vs_nb1014_bagged", "verdict",
              "per_bin_s_mean_across_folds", "deploy_ss",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
