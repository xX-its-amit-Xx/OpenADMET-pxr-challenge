"""nb2100 -- Deep verify nb2095 with 30 fresh kf_seeds {1086..1115}.

Motivation
----------
nb2095 reported pooled scaffold-CV RAE = 0.4703 +/- 0.0009 across 5 kf_seeds
{1001..1005}. The std=0.0009 is suspiciously low; prior experience (nb2060
5-seed vs 30-seed) shows the 5-seed std jumped from 0.00087 -> 0.00408 when
expanded to 30 seeds (~5x under-dispersion). nb2095 may have the same
problem because:

  - the per-fold SLSQP refit + per-fold rank-stretch grid are coarse (7-point
    stretch grid, simplex-projected weights) -> small movements between
    adjacent seeds are absorbed into the same discrete operating point;
  - only 180 unique scaffolds across 253 unblind rows -> scaffold 5-fold splits
    permute a small number of macro-clusters; 5 seeds undersample the cluster
    permutation space;
  - nb2095 deploy weights collapse to 3-of-5 anchors (chemprop_aux ~0, nb1014 ~0).

This script reloads the exact nb2095 pipeline (5 anchors with reconstructed
nb1150 OOF, SLSQP convex blend per fold, per-fold rank-stretch grid in
{1.000..1.150}) and runs it across 30 *fresh* kf_seeds {1086..1115} that
were NOT touched by any prior nb20xx run. The per-seed pooled RAE, mean,
std, and 95% CI are reported, then the dispersion is compared to nb2095's
claim and to the nb2060 under-dispersion precedent.

Gate
----
  PASS  : mean(pooled_RAE) <= 0.4720 AND std(pooled_RAE) <= 0.012
            -> confirm nb2095 PROMOTE (deep-verified, dispersion bounded)
  FAIL  : either condition violated
            -> "HOLD_DEEP_FAIL": keep nb2095 as candidate but flag dispersion
               (do NOT demote outright; record under_dispersion ratio so the
                ladder operator can decide whether to gate it behind nb1191).

Outputs
-------
  data/processed/nb2100_summary.json
  (no submission CSV; this is a verification-only run; nb2095 te file is
   already on disk and is the deploy artefact regardless of gate outcome.)
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2100"
N_FOLDS = 5
KF_SEEDS = list(range(1086, 1116))           # 30 fresh seeds
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb2095 reference numbers (from data/processed/nb2095_summary.json)
NB2095_CLAIM_MEAN = 0.4703
NB2095_CLAIM_STD = 0.0009
NB2095_KF_SEEDS_CLAIM = [1001, 1002, 1003, 1004, 1005]

# nb2060 under-dispersion precedent: 5-seed std 0.00087 -> 30-seed std 0.00408
NB2060_5SEED_STD = 0.00087
NB2060_30SEED_STD = 0.00408
NB2060_UNDER_DISP_RATIO = NB2060_30SEED_STD / NB2060_5SEED_STD  # ~4.69

# Gate thresholds (deep-30 verification of nb2095)
GATE_MEAN = 0.4720
GATE_STD = 0.012

# Anchors -- identical to nb2095
ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",      "te_nb2112.npy"),
    ("nb1014",       "nb1133_nb1014_pred_oof.npy",       "te_nb1014.npy"),
]

# nb1150 reconstructed OOF -- identical to nb2095
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)
    return P @ w


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
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
            best_r = r
            best_s = float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
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
        s_f, _ = best_stretch_on(
            blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    pooled = float(rae(y_unb, oof_blend))
    return pooled, oof_blend, fold_w, fold_s


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEEP verify nb2095 across 30 fresh kf_seeds")
    print(f"       seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]} (n={len(KF_SEEDS)})")
    print(f"       nb2095 claim: mean={NB2095_CLAIM_MEAN:.4f}  std="
          f"{NB2095_CLAIM_STD:.4f}  (5 seeds)")
    print("=" * 78)

    te = load_test()
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Anchors (identical to nb2095) ----
    print("\n[anchors]  (nb2095 pipeline: 5 anchors, chemprop_aux + nb1150 + "
          "nb1158 + nb2112 + nb1014)")
    oof_cols, te_cols, indiv_rae = [], [], {}
    for disp, oof_rel, te_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
        else:
            oof_p = DATA_PROCESSED / oof_rel
            assert oof_p.exists(), f"missing OOF: {oof_p}"
            oof = np.load(oof_p).astype(np.float64)
        te_p = DATA_PROCESSED / te_rel
        assert te_p.exists(), f"missing te: {te_p}"
        te_arr = np.load(te_p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_te,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K}")

    # ---- Run all 30 seeds ----
    print("\n" + "-" * 78)
    print(f"DEEP CV: {len(KF_SEEDS)} kf_seeds  scaffold {N_FOLDS}-fold")
    print("-" * 78)
    per_seed = []
    for kf_seed in KF_SEEDS:
        pooled, _, fold_w, fold_s = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "fold_s": [float(x) for x in fold_s],
            "fold_w_mean": [float(x) for x in np.mean(fold_w, axis=0)],
        })
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fold_s):.3f}")

    pooled_vec = np.asarray([r["pooled_rae"] for r in per_seed])
    mean_seeds = float(pooled_vec.mean())
    std_seeds = float(pooled_vec.std(ddof=1))
    se_seeds = std_seeds / np.sqrt(len(pooled_vec))
    ci_lo = float(mean_seeds - 1.96 * se_seeds)
    ci_hi = float(mean_seeds + 1.96 * se_seeds)
    min_seed = float(pooled_vec.min())
    max_seed = float(pooled_vec.max())
    iqr_lo, iqr_hi = float(np.percentile(pooled_vec, 25)), \
                     float(np.percentile(pooled_vec, 75))
    median_seeds = float(np.median(pooled_vec))

    print(f"\n[stats] n_seeds={len(pooled_vec)}")
    print(f"        mean   = {mean_seeds:.4f}")
    print(f"        std    = {std_seeds:.4f}  (sample, ddof=1)")
    print(f"        SE     = {se_seeds:.5f}")
    print(f"        95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"        median = {median_seeds:.4f}")
    print(f"        IQR    = [{iqr_lo:.4f}, {iqr_hi:.4f}]")
    print(f"        min/max= [{min_seed:.4f}, {max_seed:.4f}]")

    # ---- Compare to nb2095 5-seed claim + nb2060 precedent ----
    under_disp_ratio = std_seeds / NB2095_CLAIM_STD if NB2095_CLAIM_STD > 0 \
        else float("inf")
    delta_mean = mean_seeds - NB2095_CLAIM_MEAN
    print("\n" + "-" * 78)
    print("DISPERSION COMPARISON")
    print("-" * 78)
    print(f"   nb2095 5-seed claim   mean={NB2095_CLAIM_MEAN:.4f}  "
          f"std={NB2095_CLAIM_STD:.4f}")
    print(f"   nb2100 30-seed result mean={mean_seeds:.4f}  "
          f"std={std_seeds:.4f}")
    print(f"   delta_mean (30s - 5s) = {delta_mean:+.4f}")
    print(f"   under_dispersion_ratio (std_30 / std_5) = "
          f"{under_disp_ratio:.2f}x")
    print(f"   nb2060 precedent ratio                  = "
          f"{NB2060_UNDER_DISP_RATIO:.2f}x  (5s=0.00087 -> 30s=0.00408)")
    if under_disp_ratio >= 3.0:
        print(f"   -> nb2095 std DEFINITELY under-reported (>= 3x dispersion "
              f"on deep-30); matches nb2060 precedent.")
    elif under_disp_ratio >= 1.5:
        print(f"   -> nb2095 std mildly under-reported (1.5-3x dispersion).")
    else:
        print(f"   -> nb2095 std looks honest (within 1.5x).")

    # ---- Deploy refit (same as nb2095) ----
    print("\n" + "-" * 78)
    print("DEPLOY refit (same pipeline as nb2095; mean(fold_s) across 30 seeds)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(
        [s for r in per_seed for s in r["fold_s"]]
    ))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(
        np.float32
    )
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    lb_band_est = LB_W_OOF * mean_seeds + LB_W_TE * te_unb_rae
    print(f"   deploy weights = " + ", ".join(
        f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy)
    ))
    print(f"   mu / s         = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   te[unb] RAE    = {te_unb_rae:.4f}  (in-sample on 253)")
    print(f"   LB-band est    = {LB_W_OOF:.2f}*{mean_seeds:.4f} + "
          f"{LB_W_TE:.2f}*{te_unb_rae:.4f} = {lb_band_est:.4f}")

    # ---- Gate evaluation ----
    gate_mean_pass = mean_seeds <= GATE_MEAN
    gate_std_pass = std_seeds <= GATE_STD
    gate_pass = gate_mean_pass and gate_std_pass
    decision = "PROMOTE_CONFIRMED" if gate_pass else "HOLD_DEEP_FAIL"

    print("\n" + "-" * 78)
    print("GATE EVALUATION (deep-30 verification of nb2095)")
    print("-" * 78)
    print(f"   mean: {mean_seeds:.4f} <= {GATE_MEAN:.4f}  "
          f"-> {'PASS' if gate_mean_pass else 'FAIL'}")
    print(f"   std : {std_seeds:.4f} <= {GATE_STD:.4f}  "
          f"-> {'PASS' if gate_std_pass else 'FAIL'}")
    print(f"   overall: {'PASS' if gate_pass else 'FAIL'}  -> "
          f"{decision}")

    if gate_pass:
        print("\n   -> nb2095 deep-verified; CONFIRM as PRIMARY-1-PRE-PYRAMID.")
    else:
        print("\n   -> nb2095 HOLD: keep as candidate but flag dispersion;")
        print("      ladder operator should compare deep-30 mean to nb1191 "
              "before LB submission.")

    summary = {
        "tag": TAG,
        "method": "deep30_verification_of_nb2095_pipeline",
        "purpose": "test_underdispersion_vs_nb2095_5seed_claim_0.4703pm0.0009",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": mean_seeds,
        "pooled_rae_std_seeds_ddof1": std_seeds,
        "pooled_rae_se_seeds": se_seeds,
        "pooled_rae_ci95_lo": ci_lo,
        "pooled_rae_ci95_hi": ci_hi,
        "pooled_rae_median": median_seeds,
        "pooled_rae_iqr_lo": iqr_lo,
        "pooled_rae_iqr_hi": iqr_hi,
        "pooled_rae_min": min_seed,
        "pooled_rae_max": max_seed,
        "nb2095_claim_mean_5seed": NB2095_CLAIM_MEAN,
        "nb2095_claim_std_5seed": NB2095_CLAIM_STD,
        "nb2095_claim_kf_seeds": NB2095_KF_SEEDS_CLAIM,
        "delta_mean_30s_minus_5s": delta_mean,
        "under_dispersion_ratio_std30_over_std5": under_disp_ratio,
        "nb2060_5seed_std_precedent": NB2060_5SEED_STD,
        "nb2060_30seed_std_precedent": NB2060_30SEED_STD,
        "nb2060_under_dispersion_ratio_precedent": NB2060_UNDER_DISP_RATIO,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(ANCHORS, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s_mean_over_30_seeds": s_deploy,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "gate_mean_target": GATE_MEAN,
        "gate_std_target": GATE_STD,
        "gate_mean_pass": bool(gate_mean_pass),
        "gate_std_pass": bool(gate_std_pass),
        "gate_pass": bool(gate_pass),
        "decision": decision,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   nb2095 5-seed claim   = {NB2095_CLAIM_MEAN:.4f} +/- "
          f"{NB2095_CLAIM_STD:.4f}")
    print(f"   nb2100 30-seed result = {mean_seeds:.4f} +/- "
          f"{std_seeds:.4f}")
    print(f"   95% CI                = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"   delta_mean            = {delta_mean:+.4f}")
    print(f"   under_disp_ratio      = {under_disp_ratio:.2f}x  "
          f"(precedent {NB2060_UNDER_DISP_RATIO:.2f}x)")
    print(f"   gate                  = {gate_pass}  -> {decision}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds_ddof1",
        "pooled_rae_ci95_lo",
        "pooled_rae_ci95_hi",
        "delta_mean_30s_minus_5s",
        "under_dispersion_ratio_std30_over_std5",
        "te_unb_rae_in_sample",
        "lb_band_estimate",
        "gate_pass",
        "decision",
    ):
        print(f"  {k}: {res.get(k)}")
