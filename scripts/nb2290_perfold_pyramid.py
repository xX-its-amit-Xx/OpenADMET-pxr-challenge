"""nb2290 -- Per-fold pyramid: rebuild nb2240 SLSQP weights INSIDE each
fold's train portion only.

CONTEXT:
    nb2240 deploys a 5-anchor SLSQP pyramid {nb2240_K20, chemprop_aux, nb1191,
    nb503, nb562} and reports pooled scaffold-CV RAE 0.4598 (5 kf_seeds in
    {1001..1005}) and deep-30 mean 0.4601 across {2001..2025}.  Both numbers
    already use PER-FOLD SLSQP (weights refit on each fold's train portion).
    However, the *deploy* artefact is built from a GLOBAL SLSQP refit on all
    253 labels (line 669 of nb2240).  The legitimate worry is that the
    headline "0.4601" CV number is anchored on a single 5-seed window that
    might not be representative under per-fold weights, and that the deploy
    blend (global w) could be over-fit to the full 253 in a way the CV
    doesn't expose.

THIS SCRIPT:
    1. Loads the same 5 anchors (cached OOFs + te arrays).
    2. For each of 30 FRESH scaffold-CV seeds in {1146..1175}:
         - 5 scaffold folds.
         - PER-FOLD SLSQP weights on (fold-train rows only) -> apply to
           fold-val rows.  Per-fold stretch grid search inside fold-train,
           applied to fold-val.  Pool across folds.
         - GLOBAL SLSQP weights on ALL 253 -> apply to fold-val rows.
           Per-fold stretch grid search inside fold-train, applied to
           fold-val.  Pool across folds.
    3. Per-seed: report (rae_perfold, rae_global, delta = perfold - global).
    4. Aggregate: mean / std / range / paired-delta across 30 seeds.
    5. Compare vs nb2240 5-seed reference 0.4598 and deep-30 reference 0.4601.
    6. NO new submission CSV -- this is a diagnostic only.

OUTPUTS:
    scripts/nb2290_perfold_pyramid.py
    data/processed/nb2290_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2290"

# Reference numbers from nb2240
NB2240_REF_5SEED = 0.4598   # pooled_rae_mean_seeds across {1001..1005}
NB2240_REF_DEEP30 = 0.4601  # deep_30_verify.mean_rae across 5+25 seeds

# 30 FRESH kf_seeds disjoint from nb2240's {1001..1005} and {2001..2025}
KF_SEEDS = list(range(1146, 1176))
N_FOLDS = 5
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]


# ----------------------------- pyramid sub-anchors --------------------------
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb):
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)
    return P @ w


def reconstruct_nb1191_oof(n_unb):
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    nb2112_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


# -------------------------------- SLSQP utils -------------------------------

def slsqp_simplex(P, y):
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


# -------------------------------- CV protocols ------------------------------

def cv_perfold_weights(P_unb, y_unb, unb_scaffolds, kf_seed):
    """SLSQP weights refit on EACH fold's train portion only."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_s


def cv_global_weights(P_unb, y_unb, unb_scaffolds, kf_seed, w_global):
    """SLSQP weights fit ONCE on all 253; per-fold stretch still done in
    fold-train.  This isolates the marginal effect of GLOBAL-vs-PERFOLD
    weight estimation, holding scaffold-split structure fixed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_s = []
    for tr_loc, va_loc in splits:
        blend_tr = P_unb[tr_loc] @ w_global
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_global
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_s


# ------------------------------------ MAIN -----------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold vs global SLSQP weights on nb2240 5-anchor pyramid")
    print(f"        30 fresh kf_seeds {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print("=" * 78)

    # ---- Load truth + scaffolds ----
    te = load_test()
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")

    # ---- Reconstruct the 5 anchors EXACTLY as nb2240 stage-2 input ----
    nb2240_K20_oof = np.load(
        DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
    ).astype(np.float64)
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)

    anchor_names = ["nb2240_K20", "chemprop_aux", "nb1191", "nb503", "nb562"]
    oof_cols = [nb2240_K20_oof, chemprop_oof, nb1191_oof, nb503_oof, nb562_oof]
    indiv_rae = {nm: float(rae(y_unb, v)) for nm, v in zip(anchor_names, oof_cols)}
    P_unb = np.column_stack(oof_cols)
    K = P_unb.shape[1]
    print("\n[anchors]")
    for nm in anchor_names:
        print(f"   {nm:14s} oof_RAE={indiv_rae[nm]:.4f}")
    print(f"[stack] P_unb {P_unb.shape}  K={K}")

    # ---- Global SLSQP weights on full 253 (for the GLOBAL protocol) ----
    w_global = slsqp_simplex(P_unb, y_unb)
    print("\n[global-w] SLSQP fit on all 253:")
    for nm, w in zip(anchor_names, w_global):
        print(f"   {nm:14s} w={float(w):.4f}")

    # ---- Per-seed sweep (both protocols) ----
    print("\n" + "-" * 78)
    print(f"30-SEED SCAFFOLD-CV SWEEP  (per-fold weights vs global weights)")
    print("-" * 78)
    per_seed = []
    raes_perfold, raes_global = [], []
    fold_w_arrays_all_seeds = []   # for w-variability analysis
    for kf_seed in KF_SEEDS:
        rae_pf, _oof_pf, fw_pf, fs_pf = cv_perfold_weights(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        rae_gl, _oof_gl, fs_gl = cv_global_weights(
            P_unb, y_unb, unb_scaffolds, kf_seed, w_global,
        )
        delta = rae_pf - rae_gl
        fw_mat = np.asarray(fw_pf, dtype=float)
        fold_w_arrays_all_seeds.append(fw_mat)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "rae_perfold": float(rae_pf),
            "rae_global":  float(rae_gl),
            "delta_perfold_minus_global": float(delta),
            "fold_w_mean_perfold": [float(x) for x in fw_mat.mean(axis=0)],
            "fold_w_std_perfold":  [float(x) for x in fw_mat.std(axis=0)],
            "fold_s_perfold": [float(x) for x in fs_pf],
            "fold_s_global":  [float(x) for x in fs_gl],
        })
        raes_perfold.append(rae_pf)
        raes_global.append(rae_gl)
        print(f"   seed={kf_seed}  perfold={rae_pf:.4f}  global={rae_gl:.4f}  "
              f"delta={delta:+.4f}")

    raes_perfold = np.asarray(raes_perfold)
    raes_global = np.asarray(raes_global)
    deltas = raes_perfold - raes_global

    # Across-seed weight-variability (per-fold weights pooled)
    fw_stack = np.concatenate(fold_w_arrays_all_seeds, axis=0)  # (n_seeds*n_folds, K)
    fw_mean_acrossall = fw_stack.mean(axis=0)
    fw_std_acrossall  = fw_stack.std(axis=0)

    # ---- Compare vs nb2240 reference ----
    perfold_mean = float(raes_perfold.mean())
    perfold_std  = float(raes_perfold.std())
    global_mean  = float(raes_global.mean())
    global_std   = float(raes_global.std())
    delta_mean   = float(deltas.mean())
    delta_std    = float(deltas.std())
    n_perfold_better = int((deltas < 0).sum())

    delta_vs_nb2240 = perfold_mean - NB2240_REF_5SEED
    delta_vs_deep30 = perfold_mean - NB2240_REF_DEEP30

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   n_seeds                                = {len(KF_SEEDS)}")
    print(f"   PERFOLD pooled RAE (mean +/- std)      = {perfold_mean:.4f} +/- {perfold_std:.4f}")
    print(f"   GLOBAL  pooled RAE (mean +/- std)      = {global_mean:.4f} +/- {global_std:.4f}")
    print(f"   delta (perfold - global) mean+/-std    = {delta_mean:+.4f} +/- {delta_std:.4f}")
    print(f"   seeds where perfold beats global       = {n_perfold_better}/{len(KF_SEEDS)}")
    print(f"   perfold range                          = [{raes_perfold.min():.4f}, {raes_perfold.max():.4f}]")
    print(f"   global  range                          = [{raes_global.min():.4f}, {raes_global.max():.4f}]")
    print(f"   delta vs nb2240 5-seed ref ({NB2240_REF_5SEED:.4f}) = {delta_vs_nb2240:+.4f}")
    print(f"   delta vs nb2240 deep-30 ref ({NB2240_REF_DEEP30:.4f}) = {delta_vs_deep30:+.4f}")
    print(f"   per-fold w mean (across all seeds*folds)= "
          f"{np.round(fw_mean_acrossall, 4).tolist()}")
    print(f"   per-fold w std  (across all seeds*folds)= "
          f"{np.round(fw_std_acrossall, 4).tolist()}")
    print(f"   global w (fit on all 253)              = "
          f"{np.round(w_global, 4).tolist()}")
    print(f"   wall                                   = {time.time() - t0:.1f}s")
    print("=" * 78)

    if delta_mean < -0.001:
        verdict = "PERFOLD_BETTER"
    elif delta_mean > 0.001:
        verdict = "GLOBAL_BETTER"
    else:
        verdict = "EQUIVALENT"

    summary = {
        "tag": TAG,
        "method": "perfold_vs_global_SLSQP_diagnostic_on_nb2240_5anchor_pyramid",
        "anchors": anchor_names,
        "anchor_oof_rae_unb": indiv_rae,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "stretch_grid": STRETCH_GRID,
        "global_w_on_all_253": [float(x) for x in w_global],
        "per_seed": per_seed,
        "perfold_pooled_rae_mean": perfold_mean,
        "perfold_pooled_rae_std":  perfold_std,
        "perfold_pooled_rae_min":  float(raes_perfold.min()),
        "perfold_pooled_rae_max":  float(raes_perfold.max()),
        "global_pooled_rae_mean":  global_mean,
        "global_pooled_rae_std":   global_std,
        "global_pooled_rae_min":   float(raes_global.min()),
        "global_pooled_rae_max":   float(raes_global.max()),
        "delta_perfold_minus_global_mean": delta_mean,
        "delta_perfold_minus_global_std":  delta_std,
        "n_seeds_perfold_beats_global": n_perfold_better,
        "compare_nb2240_5seed_ref": NB2240_REF_5SEED,
        "delta_perfold_vs_nb2240_5seed_ref": delta_vs_nb2240,
        "compare_nb2240_deep30_ref": NB2240_REF_DEEP30,
        "delta_perfold_vs_nb2240_deep30_ref": delta_vs_deep30,
        "fold_w_mean_across_all_seeds_folds_perfold":
            [float(x) for x in fw_mean_acrossall],
        "fold_w_std_across_all_seeds_folds_perfold":
            [float(x) for x in fw_std_acrossall],
        "verdict_perfold_vs_global": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "perfold_pooled_rae_mean",
        "perfold_pooled_rae_std",
        "global_pooled_rae_mean",
        "global_pooled_rae_std",
        "delta_perfold_minus_global_mean",
        "n_seeds_perfold_beats_global",
        "delta_perfold_vs_nb2240_5seed_ref",
        "delta_perfold_vs_nb2240_deep30_ref",
        "verdict_perfold_vs_global",
        "fold_w_mean_across_all_seeds_folds_perfold",
        "global_w_on_all_253",
    ):
        print(f"  {k}: {res.get(k)}")
