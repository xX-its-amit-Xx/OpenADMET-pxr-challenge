"""nb1080 -- 3-way SLSQP+stretch anchor (chemprop_aux + nb972 + nb1030)
                + nb1070 per-quantile median bag on top.

Motivation
----------
nb1014 anchor is 2-way {chemprop_aux, nb972_long_train}. nb1070 (per-quantile
median bag on that anchor) hit 0.5771. nb1051 quad SLSQP added Mordred (nb1030)
+ Avalon (nb1042) but SLSQP zero-weighted Mordred in the simplex at K=4 --
likely because Avalon and Mordred fought over the same descriptor-axis
residual.

Hypothesis: at K=3 (drop Avalon), the SLSQP simplex is cleaner and Mordred --
Pearson ~0.93 to nb972 but with a different residual structure -- might
earn a non-zero weight, yielding a stronger anchor for nb1070's per-quantile
median bag to operate on.

Procedure
---------
STAGE 1 -- New 3-way anchor (nb1014 protocol with K=3):
    Pool = {chemprop_aux, nb972_long_train, nb1030_mordred}.
    For each KFold seed in {0, 1, 7, 42, 137}:
        5-fold KFold(shuffle=True, random_state=seed) on 253 unblind.
        Per fold:
            a. SLSQP simplex w in [0,1]^3, sum=1, on the 4 train folds (SSE).
            b. Grid-scan s in {1.00..2.00 step 0.05} on train-fold blend.
            c. Apply (w_f, s_f, mu_tr) to held-out fold; collect OOF.
        Record pooled cross-fit RAE per seed.
    Deploy point: mean (w, s) across 25 folds; mu = global blend mean on 253.
    Output anchor: te_nb1080_anchor.npy on 513.

STAGE 2 -- nb1070 per-quantile median bag on the new anchor:
    Anchor input: stage-1 deploy_513 (3-way bag).
    For each seed in {0, 1, 7, 42, 137}:
        5-fold KFold(shuffle=True, random_state=seed) on 253 unblind.
        Per fold:
            * Train-only quantile edges (N_BINS=5).
            * Grid-fit per-bin (mu_b, s_b) on train fold (stretch grid
              0.80..2.00 step 0.05).
            * Apply per-bin transform to held-out fold; collect OOF.
        Record per-seed pooled OOF.
    Stack (5, 253) -> MEDIAN across seeds -> bagged_oof.
    Pooled cross-fit RAE = headline.
    Deploy: per-seed fit on ALL 253 -> apply to 513; median across 5 vectors.

Verdicts vs reference points
----------------------------
    nb1014 bagged honest CV       = 0.5930  (2-way anchor reference)
    nb1051 quad bagged honest CV  = ~0.59-0.60 (Mordred zero-weighted)
    nb1070 per-q median bag       = 0.5771  (current PRIMARY-CANDIDATE)
    Beats nb1070  ?  -> headline check.

Outputs
-------
    data/processed/te_nb1080_anchor.npy        (stage-1 3-way bag deploy)
    data/processed/te_nb1080.npy               (stage-2 per-q median bag deploy)
    data/processed/nb1080_summary.json
    submissions/nb1080_3way_anchor_perq.csv
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

TAG = "nb1080"

# Stage-1 pool: (display_name, te_npy_stem, submission_csv_stem)
CANDIDATES = [
    ("chemprop_aux",      "chemprop_aux",      "chemprop_aux"),
    ("nb972_long_train",  "nb972_long_train",  "nb972_long_train_optim"),
    ("nb1030_mordred",    "nb1030",            "nb1030_mordred_lgbm"),
]

# Stage-1 grids
S1_STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
# Stage-2 grids (mirror nb1070 verbatim)
S2_STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()

N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]
N_BINS = 5

# Reference numbers
NB1014_BAG_REF = 0.5930        # nb1014 bagged honest CV (2-way anchor)
NB1051_QUAD_REF = 0.5994       # nb1051 quad-ortho bagged honest CV
NB1070_BAG_MEDIAN_REF = 0.5771 # nb1070 median bag (current per-q champion)
NB1070_PER_SEED_MEAN = 0.5814


# =====================================================================
# Loaders
# =====================================================================
def load_te(stem: str, csv_stem: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{stem}.npy"
    if npy.exists():
        arr = np.load(npy).astype(np.float64)
        if arr.shape[0] == len(te_names):
            return arr
        print(f"   [warn] te_{stem}.npy shape {arr.shape}, expected "
              f"{len(te_names)}; falling back to csv")
    sub = pd.read_csv(SUBMISSIONS / f"{csv_stem}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{csv_stem}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


# =====================================================================
# Stage 1: 3-way SLSQP+stretch multi-seed bag (nb1014 protocol)
# =====================================================================
def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float, grid) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def stage1_one_seed(P_unb: np.ndarray, y_unb: np.ndarray, seed: int) -> dict:
    n_unb = len(y_unb)
    K = P_unb.shape[1]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr,
                                      S1_STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        folds.append({
            "fold": k, "w": w_f.tolist(), "s": s_f, "mu_tr": mu_tr,
            "train_rae": rae_tr, "val_rae": rae_va, "n_va": int(len(va_loc)),
        })
    pooled = float(rae(y_unb, oof))
    return {"seed": seed, "folds": folds, "pooled_rae": pooled, "oof": oof}


# =====================================================================
# Stage 2: nb1070 per-quantile median bag (verbatim protocol)
# =====================================================================
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
        y_b = y_train[mask]
        p_b = p_train[mask]
        best_s, best_r = 1.0, float("inf")
        for s in S2_STRETCH_GRID:
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


def stage2_one_seed(seed: int, p_unb: np.ndarray, y_unb: np.ndarray
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


# =====================================================================
# Main
# =====================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 3-way SLSQP+stretch anchor + nb1070 per-q median bag")
    print(f"      pool = {[c[0] for c in CANDIDATES]}")
    print(f"      seeds = {SEEDS}  N_FOLDS = {N_FOLDS}  N_BINS = {N_BINS}")
    print("=" * 78)

    # ---- Load 513 ----
    te = load_test()
    te_names = te["name"].values
    preds_513 = np.column_stack([
        load_te(stem, csv_stem, te_names)
        for _, stem, csv_stem in CANDIDATES
    ])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    K = P_unb.shape[1]
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # ---- Individual in_RAE + corr ----
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, (name, _, _) in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[name] = r
        print(f"   {name:24s}: {r:.4f}")
    print("\n[corr] pairwise Pearson on 253:")
    corr = np.corrcoef(P_unb.T)
    names = [c[0] for c in CANDIDATES]
    for i in range(K):
        for j in range(i + 1, K):
            print(f"   {names[i]:24s} vs {names[j]:24s}: "
                  f"r={corr[i, j]:+.3f}")

    # =================================================================
    # STAGE 1 -- 3-way multi-seed cross-fit
    # =================================================================
    print("\n" + "=" * 78)
    print("STAGE 1 -- 3-way SLSQP+stretch multi-seed bag")
    print("=" * 78)

    s1_seed_results = []
    s1_per_seed_rae = []
    all_w = []
    all_s = []
    for seed in SEEDS:
        res = stage1_one_seed(P_unb, y_unb, seed)
        s1_seed_results.append({
            "seed": seed, "pooled_rae": res["pooled_rae"],
            "folds": res["folds"],
        })
        s1_per_seed_rae.append(res["pooled_rae"])
        for f in res["folds"]:
            all_w.append(f["w"])
            all_s.append(f["s"])
        fold_vals = [round(f["val_rae"], 3) for f in res["folds"]]
        print(f"   seed {seed:>3d}: pooled_RAE = {res['pooled_rae']:.4f}  "
              f"(folds: {fold_vals})")

    s1_mean_rae = float(np.mean(s1_per_seed_rae))
    s1_std_rae = float(np.std(s1_per_seed_rae))
    print(f"\n[stage1] mean pooled CV RAE = {s1_mean_rae:.4f}  "
          f"(std {s1_std_rae:.4f})")
    print(f"[stage1] vs nb1014 ref (2-way) = {NB1014_BAG_REF:.4f}  "
          f"delta = {s1_mean_rae - NB1014_BAG_REF:+.4f}")
    print(f"[stage1] vs nb1051 ref (4-way) = {NB1051_QUAD_REF:.4f}  "
          f"delta = {s1_mean_rae - NB1051_QUAD_REF:+.4f}")

    W = np.array(all_w)
    s_arr = np.array(all_s)
    mean_w = W.mean(axis=0)
    mean_w = np.clip(mean_w, 0.0, 1.0)
    mean_w = mean_w / mean_w.sum() if mean_w.sum() > 0 else np.full(K, 1.0/K)
    mean_s = float(s_arr.mean())
    print(f"\n[stage1] mean w across 25 folds:")
    for j, (name, _, _) in enumerate(CANDIDATES):
        print(f"   {name:24s}: {mean_w[j]:.4f}  "
              f"(std {W[:, j].std():.4f})")
    print(f"[stage1] mean s  across 25 folds = {mean_s:.4f}  "
          f"(std {s_arr.std():.4f})")

    # Stage-1 deploy: apply (mean_w, mean_s) with global blend mean as anchor
    blend_unb_all = P_unb @ mean_w
    mu_deploy = float(blend_unb_all.mean())
    s1_anchor_unb = mu_deploy + mean_s * (blend_unb_all - mu_deploy)
    s1_in_rae = float(rae(y_unb, s1_anchor_unb))

    blend_513 = preds_513 @ mean_w
    s1_anchor_513 = (mu_deploy + mean_s * (blend_513 - mu_deploy)).astype(
        np.float64)
    print(f"\n[stage1] deploy mu (blend on 253) = {mu_deploy:.4f}")
    print(f"[stage1] in-sample RAE (253)       = {s1_in_rae:.4f}  "
          "(overfit lower bound)")
    print(f"[stage1] anchor_513 mean/std       = "
          f"{s1_anchor_513.mean():.3f} / {s1_anchor_513.std():.3f}")

    # Save stage-1 anchor for downstream reuse
    s1_anchor_path = DATA_PROCESSED / f"te_{TAG}_anchor.npy"
    np.save(s1_anchor_path, s1_anchor_513.astype(np.float32))
    print(f"[save] {s1_anchor_path}")

    # =================================================================
    # STAGE 2 -- nb1070 per-quantile median bag on the new anchor
    # =================================================================
    print("\n" + "=" * 78)
    print("STAGE 2 -- nb1070 per-quantile median bag on stage-1 anchor")
    print("=" * 78)
    # Use the stage-1 deploy_513 as the anchor input for stage-2.
    # On the 253 we use s1_anchor_unb (= s1_anchor_513[unb_idx]).
    p_unb_s2 = s1_anchor_513[unb_idx].astype(np.float64)
    print(f"[stage2] anchor in_RAE(253) = {float(rae(y_unb, p_unb_s2)):.4f}")

    oof_stack = np.zeros((len(SEEDS), len(y_unb)), dtype=np.float64)
    s2_per_seed_rae: list[float] = []
    per_seed_ss_means: list[list[float]] = []
    seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        pooled, oof, per_fold_ss = stage2_one_seed(seed, p_unb_s2, y_unb)
        oof_stack[i] = oof
        s2_per_seed_rae.append(pooled)
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

    s2_per_seed_arr = np.array(s2_per_seed_rae)
    s2_per_seed_mean = float(s2_per_seed_arr.mean())
    s2_per_seed_std = float(s2_per_seed_arr.std())
    s2_per_seed_min = float(s2_per_seed_arr.min())
    s2_per_seed_max = float(s2_per_seed_arr.max())
    print(f"\n[stage2] per-seed RAE  mean={s2_per_seed_mean:.4f}  "
          f"std={s2_per_seed_std:.4f}  "
          f"min={s2_per_seed_min:.4f}  max={s2_per_seed_max:.4f}")

    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"[stage2] MEDIAN-bag OOF RAE = {bag_median_rae:.4f}")
    print(f"[stage2] MEAN-bag OOF RAE   = {bag_mean_rae:.4f}")

    # ---- Stage-2 deploy: per-seed fit on all 253, median across 5 vectors
    print("\n" + "-" * 78)
    print("STAGE 2 DEPLOY  (per-seed fit on all 253, median over 513 preds)")
    print("-" * 78)
    deploy_stack = np.zeros((len(SEEDS), s1_anchor_513.shape[0]),
                            dtype=np.float64)
    per_seed_deploy_ss: list[list[float]] = []
    qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    edges_all = np.quantile(p_unb_s2, qs_all)
    mus_all, _ = fit_per_bin_stretch(p_unb_s2, y_unb, edges_all)
    for i, seed in enumerate(SEEDS):
        rec = seed_records[i]
        per_fold_ss_arr = np.array(rec["per_fold_ss"])
        ss_seed = per_fold_ss_arr.mean(axis=0)
        deploy_seed_513 = apply_per_bin_stretch(s1_anchor_513,
                                                edges_all, mus_all, ss_seed)
        deploy_stack[i] = deploy_seed_513
        per_seed_deploy_ss.append(ss_seed.tolist())
        ss_str = ",".join(f"{x:.2f}" for x in ss_seed)
        print(f"   seed {seed:>3d}: deploy ss=[{ss_str}]  "
              f"deploy_513 mean={deploy_seed_513.mean():.3f}  "
              f"std={deploy_seed_513.std():.3f}")

    deploy_513_median = np.median(deploy_stack, axis=0).astype(np.float32)
    deploy_513_mean = deploy_stack.mean(axis=0).astype(np.float32)
    in_rae_deploy_median = float(rae(y_unb,
                                      deploy_513_median[unb_idx].astype(
                                          np.float64)))
    in_rae_deploy_mean = float(rae(y_unb,
                                    deploy_513_mean[unb_idx].astype(
                                        np.float64)))
    print(f"\n   deploy_513 MEDIAN  mean={deploy_513_median.mean():.3f}  "
          f"std={deploy_513_median.std():.3f}  "
          f"in-sample 253={in_rae_deploy_median:.4f}")
    print(f"   deploy_513 MEAN    mean={deploy_513_mean.mean():.3f}  "
          f"std={deploy_513_mean.std():.3f}  "
          f"in-sample 253={in_rae_deploy_mean:.4f}")

    # ---- Save deploy + submission (median is headline) ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513_median)
    plain = SUBMISSIONS / f"{TAG}_3way_anchor_perq.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513_median,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    # ---- Verdicts ----
    delta_vs_nb1070 = bag_median_rae - NB1070_BAG_MEDIAN_REF
    delta_vs_nb1014 = bag_median_rae - NB1014_BAG_REF
    delta_vs_nb1051 = bag_median_rae - NB1051_QUAD_REF
    beats_nb1070 = bool(bag_median_rae < NB1070_BAG_MEDIAN_REF)
    if delta_vs_nb1070 < -0.003:
        verdict = "BEATS_NB1070"
    elif abs(delta_vs_nb1070) <= 0.003:
        verdict = "TIES_NB1070"
    else:
        verdict = "WORSE_THAN_NB1070"

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   stage-1 mean pooled CV RAE     = {s1_mean_rae:.4f}  "
          f"(std {s1_std_rae:.4f})")
    print(f"   stage-2 per-seed mean RAE      = {s2_per_seed_mean:.4f}  "
          f"std {s2_per_seed_std:.4f}")
    print(f"   stage-2 MEDIAN-bagged OOF RAE  = {bag_median_rae:.4f}  "
          "(HEADLINE)")
    print(f"   nb1070 reference               = {NB1070_BAG_MEDIAN_REF:.4f}  "
          f"delta = {delta_vs_nb1070:+.4f}")
    print(f"   nb1014 reference               = {NB1014_BAG_REF:.4f}  "
          f"delta = {delta_vs_nb1014:+.4f}")
    print(f"   nb1051 reference               = {NB1051_QUAD_REF:.4f}  "
          f"delta = {delta_vs_nb1051:+.4f}")
    print(f"   beats_nb1070                   = {beats_nb1070}")
    print(f"   verdict                        = {verdict}")

    summary = {
        "tag": TAG,
        "candidates": [c[0] for c in CANDIDATES],
        "indiv_in_rae": indiv_rae,
        "pairwise_corr": corr.tolist(),
        "n_folds": N_FOLDS,
        "n_bins": N_BINS,
        "seeds": SEEDS,
        "stage1_stretch_grid": S1_STRETCH_GRID,
        "stage2_stretch_grid": S2_STRETCH_GRID,
        # stage 1
        "stage1_per_seed_pooled_rae": s1_per_seed_rae,
        "stage1_mean_pooled_rae": s1_mean_rae,
        "stage1_std_pooled_rae": s1_std_rae,
        "stage1_mean_w": mean_w.tolist(),
        "stage1_std_w_across_folds": W.std(axis=0).tolist(),
        "stage1_mean_s": mean_s,
        "stage1_std_s_across_folds": float(s_arr.std()),
        "stage1_deploy_mu_blend": mu_deploy,
        "stage1_in_sample_rae": s1_in_rae,
        "stage1_anchor_te_mean": float(s1_anchor_513.mean()),
        "stage1_anchor_te_std": float(s1_anchor_513.std()),
        "stage1_anchor_npy": str(s1_anchor_path),
        "stage1_seed_results": s1_seed_results,
        # stage 2
        "stage2_per_seed_rae": s2_per_seed_rae,
        "stage2_per_seed_rae_mean": s2_per_seed_mean,
        "stage2_per_seed_rae_std": s2_per_seed_std,
        "stage2_per_seed_rae_min": s2_per_seed_min,
        "stage2_per_seed_rae_max": s2_per_seed_max,
        "stage2_bag_median_rae": bag_median_rae,
        "stage2_bag_mean_rae": bag_mean_rae,
        "stage2_per_seed_ss_means_across_folds": per_seed_ss_means,
        "stage2_per_seed_deploy_ss": per_seed_deploy_ss,
        "stage2_in_rae_deploy_median_on_253": in_rae_deploy_median,
        "stage2_in_rae_deploy_mean_on_253": in_rae_deploy_mean,
        "stage2_deploy_te_median_mean": float(deploy_513_median.mean()),
        "stage2_deploy_te_median_std": float(deploy_513_median.std()),
        "stage2_deploy_te_mean_mean": float(deploy_513_mean.mean()),
        "stage2_deploy_te_mean_std": float(deploy_513_mean.std()),
        "stage2_seed_records": seed_records,
        # references / verdicts
        "nb1014_bag_reference": NB1014_BAG_REF,
        "nb1051_quad_reference": NB1051_QUAD_REF,
        "nb1070_bag_median_reference": NB1070_BAG_MEDIAN_REF,
        "nb1070_per_seed_mean_reference": NB1070_PER_SEED_MEAN,
        "delta_vs_nb1070_median": delta_vs_nb1070,
        "delta_vs_nb1014": delta_vs_nb1014,
        "delta_vs_nb1051_quad": delta_vs_nb1051,
        "beats_nb1070": beats_nb1070,
        "verdict": verdict,
        "plain_submission": str(plain),
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
    for k in ("stage1_per_seed_pooled_rae", "stage1_mean_pooled_rae",
              "stage1_mean_w", "stage1_mean_s",
              "stage2_per_seed_rae", "stage2_per_seed_rae_mean",
              "stage2_bag_median_rae", "stage2_bag_mean_rae",
              "delta_vs_nb1070_median", "delta_vs_nb1014",
              "beats_nb1070", "verdict",
              "stage2_in_rae_deploy_median_on_253", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
