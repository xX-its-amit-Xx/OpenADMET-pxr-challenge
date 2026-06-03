"""nb1134 -- Chain nb1123 residual correction -> nb1070 per-quantile stretch.

Hypothesis: nb1123 (shallow LGBM Huber on residual of nb1070, honest cross-fit
RAE 0.5704) adds signal but also re-shapes the predictor distribution: the
residual correction shrinks toward zero (residual_oof.std ~0.27 vs truth
residual.std ~0.67), which DECREASES the corrected predictor's variance below
truth in a non-uniform way. The original per-quantile stretch tuned on
nb1014 / nb1070 may no longer be optimal for this re-shaped distribution --
each quantile bin's optimal s could shift.

So: re-run nb1070's exact per-quantile stretch protocol (5 bins, 5-seed median
bag) on `nb1123_pred_oof` as the NEW anchor and compare pooled cross-fit RAE
to nb1123's 0.5704 base.

Procedure:
  1. Anchor = nb1123_pred_oof (corrected predictor on 253; pre-computed).
  2. Per-seed in {0,1,7,42,137}, KFold(5, shuffle, random_state=seed):
       * Train-only quantile edges (5 bins, N_BINS=5) on the corrected anchor.
       * Train-only fit_per_bin_stretch grid (s in {0.80..2.00 step 0.05}).
       * Apply per-bin (mu_b, s_b) to held-out fold rows.
  3. Stack per-seed OOF -> (5, 253), MEDIAN across seed -> nb1134 OOF.
  4. Pooled RAE on the 253 unblind.

Deploy:
  * Need a 513-vector for the corrected predictor. te_nb1123 doesn't exist on
    disk, so we derive it: refit nb1123's shallow LGBM Huber on all 253 unblind
    residuals (truth - nb1070_oof), apply to combined(Morgan+RDKit) features
    of all 513, ADD to te_nb1070 -> te_nb1123_corrected_513.
  * Then nb1070-style deploy: per-seed fit (mu, ss) on all 253 corrected anchor
    with each seed's KFold-mean ss, apply to 513, median across the 5
    deploys.

Outputs:
  data/processed/te_nb1134.npy
  data/processed/nb1134_pred_oof.npy
  data/processed/nb1134_summary.json
  submissions/nb1134_residual_perq_chain.csv
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
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1134"
RESID_ANCHOR = "nb1123"       # corrected predictor (chain input)
NB1070_ANCHOR = "nb1070"      # uncorrected predictor (for residual definition + 513 base)
NB1014_ANCHOR = "nb1014"      # for nb1070_oof regen on 253

# nb1070 per-quantile-stretch protocol (verbatim).
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

# nb1123 residual model params (verbatim from nb1123_residual_gb.py).
RESID_SEED = 42
LGBM_PARAMS = dict(
    objective="huber",
    alpha=1.0,
    learning_rate=0.05,
    n_estimators=80,
    max_depth=3,
    num_leaves=7,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    verbosity=-1,
    random_state=RESID_SEED,
    n_jobs=2,
)

# Reference numbers (from prior runs).
NB1070_REF_POOLED = 0.5790
NB1123_REF_POOLED = 0.5704


# ---------- nb1070 per-quantile-stretch primitives ----------

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


# ---------- nb1070 OOF (re-)generation for residual definition ----------

def _nb1070_seed_oof(seed: int, p_unb: np.ndarray,
                     y_unb: np.ndarray) -> np.ndarray:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        oof[va_loc] = apply_per_bin_stretch(p_unb[va_loc], edges, mus, ss)
    return oof


def regenerate_nb1070_oof(p_unb: np.ndarray, y_unb: np.ndarray) -> np.ndarray:
    stack = np.zeros((len(SEEDS), len(y_unb)), dtype=np.float64)
    for i, s in enumerate(SEEDS):
        stack[i] = _nb1070_seed_oof(s, p_unb, y_unb)
    return np.median(stack, axis=0)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- nb1070 per-quantile stretch (5 bins, 5-seed median bag) "
          f"on nb1123-corrected predictor")
    print(f"          seeds = {SEEDS}   N_BINS = {N_BINS}   "
          f"stretch_grid = {STRETCH_GRID[0]:.2f}..{STRETCH_GRID[-1]:.2f} "
          f"step 0.05")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    # ---- Load nb1123 corrected predictor on 253 (anchor) ----
    nb1123_oof_path = DATA_PROCESSED / f"{RESID_ANCHOR}_pred_oof.npy"
    if not nb1123_oof_path.exists():
        raise FileNotFoundError(
            f"missing {nb1123_oof_path} -- run scripts/nb1123_residual_gb.py "
            f"first")
    p_unb = np.load(nb1123_oof_path).astype(np.float64)
    print(f"[load] {RESID_ANCHOR}_pred_oof.npy shape={p_unb.shape}  "
          f"y_unb shape={y_unb.shape}")

    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"[base] pooled RAE(nb1123 OOF, 253)  = {in_rae_anchor:.4f}   "
          f"(ref nb1123 {NB1123_REF_POOLED:.4f})")

    # Distribution diagnostics: corrected vs un-corrected anchor
    nb1070_oof_path = DATA_PROCESSED / f"{NB1070_ANCHOR}_pred_oof.npy"
    if nb1070_oof_path.exists():
        nb1070_oof_for_diag = np.load(nb1070_oof_path).astype(np.float64)
        if nb1070_oof_for_diag.shape[0] != n_unb:
            nb1070_oof_for_diag = None
    else:
        nb1070_oof_for_diag = None
    if nb1070_oof_for_diag is not None:
        print(f"[diag] nb1070 OOF  std={nb1070_oof_for_diag.std():.4f}   "
              f"nb1123 OOF  std={p_unb.std():.4f}   "
              f"truth      std={y_unb.std():.4f}")
        print(f"[diag] corr(nb1070_oof, nb1123_oof) = "
              f"{np.corrcoef(nb1070_oof_for_diag, p_unb)[0, 1]:.4f}")

    # ---- Per-seed honest cross-fit on the corrected predictor ----
    print("\n" + "-" * 78)
    print("PER-SEED HONEST CROSS-FIT  (anchor = nb1123 corrected predictor)")
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

    # ---- MEDIAN bag across seeds at row level ----
    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"\n[bag] MEDIAN  bagged_oof RAE (median of {len(SEEDS)} OOFs) = "
          f"{bag_median_rae:.4f}")
    print(f"[bag] MEAN    bagged_oof RAE (mean   of {len(SEEDS)} OOFs) = "
          f"{bag_mean_rae:.4f}")

    delta_vs_nb1123 = bag_median_rae - in_rae_anchor
    delta_vs_nb1070_ref = bag_median_rae - NB1070_REF_POOLED
    beats_nb1123 = bag_median_rae < in_rae_anchor - 0.003
    if delta_vs_nb1123 <= -0.003:
        verdict = "RECAL_HELPS"
    elif abs(delta_vs_nb1123) < 0.003:
        verdict = "RECAL_NEUTRAL"
    else:
        verdict = "RECAL_HURTS"

    # =================================================================
    # Deploy on 513.
    #   Step A: build corrected te_513 = te_nb1070 + residual_513_refit
    #           where residual_513 comes from refitting nb1123's LGBM on
    #           all 253 unblind residuals (truth - nb1070_oof).
    #   Step B: nb1070-style per-seed mean-ss deploy on this corrected
    #           513 anchor, median across 5 seeds.
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (build corrected te_513 -> per-q stretch median bag on 513)")
    print("-" * 78)

    te_nb1070_path = DATA_PROCESSED / f"te_{NB1070_ANCHOR}.npy"
    te_nb1070_513 = np.load(te_nb1070_path).astype(np.float64)
    print(f"[load] te_{NB1070_ANCHOR}.npy shape={te_nb1070_513.shape}  "
          f"mean={te_nb1070_513.mean():.3f}  std={te_nb1070_513.std():.3f}")

    # Need nb1070_oof on 253 to define residual targets for the all-253
    # refit of the nb1123 residual LGBM. Use saved file if shape OK.
    if nb1070_oof_for_diag is not None:
        nb1070_oof = nb1070_oof_for_diag
        print(f"[load] nb1070_pred_oof.npy on 253 (for residual targets)")
    else:
        # Regen from te_nb1014.npy
        nb1014_path = DATA_PROCESSED / f"te_{NB1014_ANCHOR}.npy"
        if not nb1014_path.exists():
            raise FileNotFoundError(
                f"need te_{NB1014_ANCHOR}.npy to regenerate nb1070 OOF")
        preds_513_nb1014 = np.load(nb1014_path).astype(np.float64)
        p_unb_nb1014 = preds_513_nb1014[unb_idx]
        print(f"[regen] nb1070_oof from te_{NB1014_ANCHOR}.npy")
        nb1070_oof = regenerate_nb1070_oof(p_unb_nb1014, y_unb)
        np.save(nb1070_oof_path, nb1070_oof)
        print(f"[save] {nb1070_oof_path}")

    residual_train = y_unb - nb1070_oof  # signed residual targets
    print(f"[resid] residual mean={residual_train.mean():+.4f}  "
          f"std={residual_train.std():.4f}")

    # Features: combined Morgan+RDKit on ALL 513
    print(f"[feat] computing combined(Morgan+RDKit) on n=513 SMILES")
    X_all = impute(combined(te_smiles.tolist()))
    print(f"[feat] X_all shape={X_all.shape}")
    X_unb_feat = X_all[unb_idx]

    # Refit nb1123 residual LGBM on all 253 -> predict residual on 513
    mdl_full = LGBMRegressor(**LGBM_PARAMS)
    mdl_full.fit(X_unb_feat, residual_train)
    residual_513 = mdl_full.predict(X_all).astype(np.float64)
    print(f"[refit] residual_513 mean={residual_513.mean():+.4f}  "
          f"std={residual_513.std():.4f}")

    corrected_513 = te_nb1070_513 + residual_513
    in_rae_corrected_513 = float(rae(y_unb,
                                      corrected_513[unb_idx].astype(np.float64)))
    print(f"[base] corrected te_513  mean={corrected_513.mean():.3f}  "
          f"std={corrected_513.std():.3f}   "
          f"in-sample 253={in_rae_corrected_513:.4f}")

    # nb1070-style per-seed deploy on the corrected 513 anchor.
    # Per nb1070 protocol: mus are computed on all-253 corrected OOF anchor
    # using deploy edges (seed-invariant); ss is each seed's mean-of-folds
    # stretch vector; apply to 513.
    qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    edges_all = np.quantile(p_unb, qs_all)
    mus_all, _ = fit_per_bin_stretch(p_unb, y_unb, edges_all)
    print(f"[deploy] edges (on nb1123 OOF, 253) = "
          f"[{', '.join(f'{e:.3f}' for e in edges_all)}]")
    print(f"[deploy] mus (all-253 corrected anchor) = "
          f"[{', '.join(f'{m:.3f}' for m in mus_all)}]")

    deploy_stack = np.zeros((len(SEEDS), corrected_513.shape[0]),
                            dtype=np.float64)
    per_seed_deploy_ss: list[list[float]] = []
    for i, seed in enumerate(SEEDS):
        rec = seed_records[i]
        per_fold_ss_arr = np.array(rec["per_fold_ss"])  # (5, N_BINS)
        ss_seed = per_fold_ss_arr.mean(axis=0)
        per_seed_deploy_ss.append(ss_seed.tolist())
        deploy_seed_513 = apply_per_bin_stretch(corrected_513, edges_all,
                                                mus_all, ss_seed)
        deploy_stack[i] = deploy_seed_513
        ss_str = ",".join(f"{x:.2f}" for x in ss_seed)
        print(f"   seed {seed:>3d}: deploy ss=[{ss_str}]  "
              f"deploy_513 mean={deploy_seed_513.mean():.3f}  "
              f"std={deploy_seed_513.std():.3f}")

    deploy_513_median = np.median(deploy_stack, axis=0).astype(np.float32)
    deploy_513_mean = deploy_stack.mean(axis=0).astype(np.float32)
    in_rae_deploy_median = float(rae(
        y_unb, deploy_513_median[unb_idx].astype(np.float64)))
    in_rae_deploy_mean = float(rae(
        y_unb, deploy_513_mean[unb_idx].astype(np.float64)))
    print(f"\n   deploy_513 MEDIAN-bag  mean={deploy_513_median.mean():.3f}  "
          f"std={deploy_513_median.std():.3f}  "
          f"in-sample 253={in_rae_deploy_median:.4f}")
    print(f"   deploy_513 MEAN-bag    mean={deploy_513_mean.mean():.3f}  "
          f"std={deploy_513_mean.std():.3f}  "
          f"in-sample 253={in_rae_deploy_mean:.4f}")
    print(f"   anchor (nb1070) te(513)         "
          f"mean={te_nb1070_513.mean():.3f}  std={te_nb1070_513.std():.3f}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513_median)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            bagged_median_oof.astype(np.float32))
    plain = SUBMISSIONS / f"{TAG}_residual_perq_chain.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_513_median,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {TAG}_pred_oof.npy")
    print(f"[save] {plain}")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1070 ref pooled               = {NB1070_REF_POOLED:.4f}")
    print(f"   nb1123 ref pooled               = {NB1123_REF_POOLED:.4f}")
    print(f"   nb1123 OOF in_RAE this run      = {in_rae_anchor:.4f}")
    print(f"   nb1134 per-seed mean RAE        = {per_seed_mean:.4f}  "
          f"std={per_seed_std:.4f}")
    print(f"   nb1134 MEDIAN-bag pooled OOF    = {bag_median_rae:.4f}  "
          f"(delta vs nb1123 anchor = {delta_vs_nb1123:+.4f},  "
          f"delta vs nb1070 ref = {delta_vs_nb1070_ref:+.4f})")
    print(f"   nb1134 MEAN-bag pooled OOF      = {bag_mean_rae:.4f}")
    print(f"   beats_nb1123                    = {beats_nb1123}")
    print(f"   verdict                         = {verdict}")

    summary = {
        "tag": TAG,
        "resid_anchor": RESID_ANCHOR,
        "nb1070_anchor": NB1070_ANCHOR,
        "nb1014_anchor": NB1014_ANCHOR,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stretch_grid": STRETCH_GRID,
        "in_rae_anchor_on_253": in_rae_anchor,
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": per_seed_mean,
        "per_seed_rae_std": per_seed_std,
        "per_seed_rae_min": per_seed_min,
        "per_seed_rae_max": per_seed_max,
        "bag_median_rae": bag_median_rae,
        "bag_mean_rae": bag_mean_rae,
        "delta_vs_nb1123_anchor": delta_vs_nb1123,
        "delta_vs_nb1070_ref": delta_vs_nb1070_ref,
        "beats_nb1123": bool(beats_nb1123),
        "verdict": verdict,
        "per_seed_ss_means_across_folds": per_seed_ss_means,
        "per_seed_deploy_ss": per_seed_deploy_ss,
        "deploy_mus_all_253": mus_all.tolist(),
        "deploy_edges_all_253": edges_all.tolist(),
        "in_rae_deploy_median_on_253": in_rae_deploy_median,
        "in_rae_deploy_mean_on_253": in_rae_deploy_mean,
        "in_rae_corrected_513_on_253": in_rae_corrected_513,
        "deploy_te_median_mean": float(deploy_513_median.mean()),
        "deploy_te_median_std": float(deploy_513_median.std()),
        "deploy_te_mean_mean": float(deploy_513_mean.mean()),
        "deploy_te_mean_std": float(deploy_513_mean.std()),
        "nb1070_te_mean": float(te_nb1070_513.mean()),
        "nb1070_te_std": float(te_nb1070_513.std()),
        "residual_513_mean": float(residual_513.mean()),
        "residual_513_std": float(residual_513.std()),
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1123_ref_pooled": NB1123_REF_POOLED,
        "plain_submission": str(plain),
        "seed_records": seed_records,
        "lgbm_params": LGBM_PARAMS,
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
              "delta_vs_nb1123_anchor", "delta_vs_nb1070_ref",
              "beats_nb1123", "verdict",
              "in_rae_deploy_median_on_253", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
