"""nb1200 -- Verify nb1191 PRE-unblind pyramid under STRICTER fresh kf_seeds.

Spec:
  1. Reload nb1191 pipeline (4 anchors: chemprop_aux, nb1150, nb1158_K32, nb2112_K28;
     SLSQP simplex blend + per-fold rank-stretch on STRETCH_GRID).
  2. Run fresh kf_seeds {1006, 1007, 1008, 1009, 1010} (disjoint from
     nb1191's 1001-1005).
  3. Report per-seed pooled RAE, mean + std.
  4. Verify nb1191's claimed mean 0.4703 reproduces under the new seeds.
  5. Re-compute LB-band estimate with new seeds (should match 0.3688 band
     [0.319, 0.419]).
  6. Gate: mean OOF <= 0.4730 AND std <= 0.010 -> confirm PROMOTE.
  7. If reproducible: nb1191_deploy_pre_pyramid.csv stays as LADDER
     PRIMARY-1-LB-SAFE (file already on disk from the nb1191 run).
  Output: data/processed/nb1200_summary.json

All anchors / paths / SLSQP / stretch grid / LB-band weights mirror nb1191
verbatim; only KF_SEEDS differ.
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1200"
PARENT_TAG = "nb1191"
N_FOLDS = 5
KF_SEEDS_FRESH = [1006, 1007, 1008, 1009, 1010]   # disjoint from nb1191
KF_SEEDS_ORIGINAL = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# Gates per spec
GATE_MEAN_OOF = 0.4730
GATE_STD = 0.010

# Reference (nb1191 claim) for the reproduce check
NB1191_CLAIM_MEAN = 0.4703
NB1191_CLAIM_LB_BAND_MID = 0.3688
NB1191_CLAIM_LB_BAND_LOW = 0.319
NB1191_CLAIM_LB_BAND_HIGH = 0.419
REPRODUCE_MEAN_TOL = 0.005     # mean OOF must match within +/- 0.005
REPRODUCE_LB_BAND_TOL = 0.020  # LB-band must match within +/- 0.020

# Anchors -- IDENTICAL to nb1191
ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",      "te_nb2112.npy"),
]

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
    print(f"{TAG} -- Verify {PARENT_TAG} under fresh kf_seeds {KF_SEEDS_FRESH}")
    print("=" * 78)

    te = load_test()
    te_smiles = te["smiles"].values
    n_te = len(te)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Anchors (IDENTICAL to nb1191) ----
    print("\n[anchors]")
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
        print(f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K}")

    # ---- Stage 2+3: scaffold 5-fold CV across FRESH kf_seeds ----
    print("\n" + "-" * 78)
    print(f"FRESH SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS_FRESH}  "
          f"stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS_FRESH:
        pooled, oof_blend, fold_w, fold_s = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "fold_s": [float(x) for x in fold_s],
            "fold_w_mean": [float(x) for x in np.mean(fold_w, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fold_s):.3f}  "
              f"w_mean={np.round(np.mean(fold_w, axis=0), 3).tolist()}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean_seed_oof = float(rae(y_unb, mean_oof))
    pooled_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    pooled_min = float(np.min([r["pooled_rae"] for r in per_seed]))
    pooled_max = float(np.max([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv-fresh] pooled_RAE mean = {pooled_mean:.4f}  "
          f"std = {pooled_std:.4f}  [min={pooled_min:.4f}, "
          f"max={pooled_max:.4f}]")
    print(f"[cv-fresh] RAE of mean-of-seed OOFs = {rae_of_mean_seed_oof:.4f}")

    # ---- Deploy (identical to nb1191) ----
    print("\n" + "-" * 78)
    print("DEPLOY REFIT (full 253; mean(fold_s) across all 5 FRESH seeds)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(
        [s for r in per_seed for s in r["fold_s"]]
    ))
    in_rae = float(rae(
        y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)
    ))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float64)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))

    w_str = ", ".join(
        f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy)
    )
    print(f"   deploy weights        = {w_str}")
    print(f"   deploy mu / s         = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   in-sample RAE (253)   = {in_rae:.4f}")
    print(f"   te[unb_idx] RAE       = {te_unb_rae:.4f}  (in-sample on 253)")
    print(f"   te(513) mean / std    = {deploy_te.mean():.3f} / "
          f"{deploy_te.std():.3f}")

    # LB-band estimate per memo
    lb_band_est = LB_W_OOF * pooled_mean + LB_W_TE * te_unb_rae
    lb_band_low = lb_band_est - 0.05
    lb_band_high = lb_band_est + 0.05
    print(f"\n[LB-band fresh] {LB_W_OOF:.2f}*OOF({pooled_mean:.4f}) + "
          f"{LB_W_TE:.2f}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}")
    print(f"               band = [{lb_band_low:.4f}, {lb_band_high:.4f}]")

    # ---- Reproduce check vs nb1191 claim ----
    print("\n" + "-" * 78)
    print(f"REPRODUCE CHECK vs {PARENT_TAG} claim")
    print("-" * 78)
    mean_delta = pooled_mean - NB1191_CLAIM_MEAN
    lb_delta = lb_band_est - NB1191_CLAIM_LB_BAND_MID
    mean_reproduces = abs(mean_delta) <= REPRODUCE_MEAN_TOL
    lb_reproduces = abs(lb_delta) <= REPRODUCE_LB_BAND_TOL
    print(f"   mean OOF fresh = {pooled_mean:.4f}  "
          f"vs claim {NB1191_CLAIM_MEAN:.4f}  "
          f"delta = {mean_delta:+.4f}  (tol +/-{REPRODUCE_MEAN_TOL}) -> "
          f"{'OK' if mean_reproduces else 'MISMATCH'}")
    print(f"   LB-band fresh  = {lb_band_est:.4f}  "
          f"vs claim {NB1191_CLAIM_LB_BAND_MID:.4f}  "
          f"delta = {lb_delta:+.4f}  (tol +/-{REPRODUCE_LB_BAND_TOL}) -> "
          f"{'OK' if lb_reproduces else 'MISMATCH'}")

    # ---- Promote gate ----
    print("\n" + "-" * 78)
    print("PROMOTE GATE")
    print("-" * 78)
    gate_a = pooled_mean <= GATE_MEAN_OOF
    gate_b = pooled_std <= GATE_STD
    gate_pass = gate_a and gate_b
    print(f"   A: mean OOF {pooled_mean:.4f} <= {GATE_MEAN_OOF} -> "
          f"{'PASS' if gate_a else 'FAIL'}")
    print(f"   B: std OOF  {pooled_std:.4f} <= {GATE_STD}  -> "
          f"{'PASS' if gate_b else 'FAIL'}")
    print(f"   overall promote = {'CONFIRM' if gate_pass else 'HOLD'}")

    sub_csv_path = SUBMISSIONS / f"{PARENT_TAG}_deploy_pre_pyramid.csv"
    sub_csv_exists = sub_csv_path.exists()
    print(f"\n[deploy-file] {sub_csv_path}  exists={sub_csv_exists}")
    if gate_pass and sub_csv_exists:
        print(f"[ladder] -> add {sub_csv_path.name} as PRIMARY-1-LB-SAFE")
        ladder_action = "ADD_PRIMARY_1_LB_SAFE"
    elif gate_pass and not sub_csv_exists:
        print(f"[ladder] gate PASS but deploy CSV missing -- run "
              f"{PARENT_TAG} first")
        ladder_action = "GATE_PASS_DEPLOY_MISSING"
    else:
        print("[ladder] gate FAIL -- no ladder change")
        ladder_action = "HOLD_NO_CHANGE"

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "verify_PRE_unblind_pyramid_fresh_seeds",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds_fresh": KF_SEEDS_FRESH,
        "kf_seeds_original_nb1191": KF_SEEDS_ORIGINAL,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results_fresh": per_seed,
        "pooled_rae_mean_fresh": pooled_mean,
        "pooled_rae_std_fresh": pooled_std,
        "pooled_rae_min_fresh": pooled_min,
        "pooled_rae_max_fresh": pooled_max,
        "rae_of_mean_of_seed_oofs_fresh": rae_of_mean_seed_oof,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(ANCHORS, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate_fresh": lb_band_est,
        "lb_band_low_fresh": lb_band_low,
        "lb_band_high_fresh": lb_band_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "nb1191_claim_mean": NB1191_CLAIM_MEAN,
        "nb1191_claim_lb_band_mid": NB1191_CLAIM_LB_BAND_MID,
        "nb1191_claim_lb_band_low": NB1191_CLAIM_LB_BAND_LOW,
        "nb1191_claim_lb_band_high": NB1191_CLAIM_LB_BAND_HIGH,
        "mean_delta_vs_claim": mean_delta,
        "lb_band_delta_vs_claim": lb_delta,
        "reproduce_mean_tolerance": REPRODUCE_MEAN_TOL,
        "reproduce_lb_band_tolerance": REPRODUCE_LB_BAND_TOL,
        "mean_reproduces_within_tol": bool(mean_reproduces),
        "lb_band_reproduces_within_tol": bool(lb_reproduces),
        "reproduces_overall": bool(mean_reproduces and lb_reproduces),
        "gate_mean_target": GATE_MEAN_OOF,
        "gate_std_target": GATE_STD,
        "gate_a_mean_le_target": bool(gate_a),
        "gate_b_std_le_target": bool(gate_b),
        "gate_pass": bool(gate_pass),
        "deploy_csv_path": str(sub_csv_path),
        "deploy_csv_exists": bool(sub_csv_exists),
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-seed pooled RAE:")
    for r in per_seed:
        print(f"      seed={r['kf_seed']}  pooled_RAE={r['pooled_rae']:.4f}")
    print(f"   mean +/- std       = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   nb1191 claim       = {NB1191_CLAIM_MEAN:.4f}  "
          f"(delta {mean_delta:+.4f}, reproduces={mean_reproduces})")
    print(f"   LB-band fresh      = {lb_band_est:.4f}  "
          f"vs claim {NB1191_CLAIM_LB_BAND_MID:.4f}  "
          f"(delta {lb_delta:+.4f}, reproduces={lb_reproduces})")
    print(f"   gate mean<={GATE_MEAN_OOF} std<={GATE_STD} -> "
          f"{'CONFIRM PROMOTE' if gate_pass else 'HOLD'}")
    print(f"   ladder_action      = {ladder_action}")
    print(f"   wall               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_fresh",
        "pooled_rae_std_fresh",
        "mean_reproduces_within_tol",
        "lb_band_estimate_fresh",
        "lb_band_reproduces_within_tol",
        "gate_pass",
        "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
