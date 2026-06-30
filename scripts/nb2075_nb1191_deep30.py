"""nb2075 -- DEEP verify of nb1191 (PRE-unblind pyramid) with 30 fresh kf_seeds.

Cycle 149 (nb1210) ran 15 fresh disjoint seeds {1011-1025} and reported
pooled-RAE mean = 0.4718 +/- 0.00245 (n=15, t-CI95 [0.4705, 0.4732]).
That passed nb1210's GATE-A (mean <= 0.4720) and GATE-B (std <= 0.012)
and re-promoted nb1191 to PRIMARY-1-LB-SAFE.

This script doubles the seed budget for higher-confidence verdict:
  - kf_seeds {1026..1055} (30 fresh seeds, DISJOINT from 1001-1025)
  - identical anchors / pipeline / stretch grid as nb1191 / nb1200 / nb1210
  - per-seed pooled RAE; mean/std/median/95%-CI (t, df=29)
  - compare against nb1210's 15-seed claim (0.4718 +/- 0.00245)
  - final PROMOTE verdict: mean <= 0.4720 AND std <= 0.012

Pipeline (IDENTICAL to nb1210):
  4 anchors -> per-fold SLSQP convex blend -> per-fold rank-stretch
  STRETCH_GRID = {1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150}
  scaffold_kfold_indices(unb_scaffolds, n_splits=5, shuffle=True, seed=kf_seed)

Output:
  data/processed/nb2075_summary.json
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
from scipy.stats import t as student_t

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2075"
PARENT_TAG = "nb1191"
PRIOR_TAG = "nb1210"
N_FOLDS = 5

# Fresh disjoint seed band: nb1191 used 1001-1005, nb1200 used 1006-1010,
# nb1210 used 1011-1025. nb2075 uses 1026-1055 (30 brand-new seeds).
KF_SEEDS_DEEP = list(range(1026, 1056))   # 30 seeds: 1026..1055
KF_SEEDS_NB1210 = list(range(1011, 1026)) # 15 seeds (cycle 149 claim)
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# DEEP promote gates per task spec
GATE_MEAN_OOF = 0.4720
GATE_STD = 0.012

# nb1210 (cycle 149) claim
NB1210_CLAIM_MEAN = 0.47182021877300634
NB1210_CLAIM_STD = 0.0024508702197786974
NB1210_CLAIM_N = 15
NB1191_CLAIM_MEAN = 0.4703

# Anchors -- IDENTICAL to nb1191 / nb1200 / nb1210
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
    print(f"{TAG} -- DEEP re-verify of {PARENT_TAG}  "
          f"({len(KF_SEEDS_DEEP)} fresh disjoint seeds)")
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

    # ---- DEEP scaffold 5-fold CV over 30 fresh seeds ----
    print("\n" + "-" * 78)
    print(f"DEEP SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS_DEEP[0]}..{KF_SEEDS_DEEP[-1]}  "
          f"(n={len(KF_SEEDS_DEEP)})  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS_DEEP:
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

    pooled_arr = np.array([r["pooled_rae"] for r in per_seed], dtype=np.float64)
    n_seeds = len(pooled_arr)
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1))   # sample std for CI
    pooled_min = float(pooled_arr.min())
    pooled_max = float(pooled_arr.max())
    pooled_median = float(np.median(pooled_arr))
    pooled_p025 = float(np.percentile(pooled_arr, 2.5))
    pooled_p975 = float(np.percentile(pooled_arr, 97.5))

    # 95% CI on the mean (t-distribution, df = n-1 = 29)
    sem = pooled_std / np.sqrt(n_seeds)
    t_crit = float(student_t.ppf(0.975, df=n_seeds - 1))
    ci_low = pooled_mean - t_crit * sem
    ci_high = pooled_mean + t_crit * sem

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean_seed_oof = float(rae(y_unb, mean_oof))
    print(f"\n[cv-deep]  mean = {pooled_mean:.4f}  "
          f"std(ddof=1) = {pooled_std:.4f}  "
          f"median = {pooled_median:.4f}")
    print(f"[cv-deep]  min  = {pooled_min:.4f}  max  = {pooled_max:.4f}")
    print(f"[cv-deep]  95%CI on mean (t, df={n_seeds - 1}) = "
          f"[{ci_low:.4f}, {ci_high:.4f}]")
    print(f"[cv-deep]  p2.5%/p97.5% band = "
          f"[{pooled_p025:.4f}, {pooled_p975:.4f}]")
    print(f"[cv-deep]  RAE of mean-of-seed OOFs = "
          f"{rae_of_mean_seed_oof:.4f}")

    # ---- Compare against nb1210 (cycle 149) 15-seed claim ----
    print("\n" + "-" * 78)
    print("COMPARISON: nb1210 (15 fresh seeds, cycle 149) vs nb2075 (30 fresh seeds)")
    print("-" * 78)
    mean_shift = pooled_mean - NB1210_CLAIM_MEAN
    std_ratio = pooled_std / NB1210_CLAIM_STD if NB1210_CLAIM_STD > 0 else float("nan")

    # Welch two-sample t test: nb2075 (n=30) vs nb1210 (n=15)
    sem_pool = float(
        np.sqrt(pooled_std**2 / n_seeds + NB1210_CLAIM_STD**2 / NB1210_CLAIM_N)
    )
    if sem_pool > 0:
        welch_t = mean_shift / sem_pool
    else:
        welch_t = 0.0
    # Welch-Satterthwaite df
    num = (pooled_std**2 / n_seeds + NB1210_CLAIM_STD**2 / NB1210_CLAIM_N) ** 2
    den = (
        (pooled_std**2 / n_seeds) ** 2 / (n_seeds - 1)
        + (NB1210_CLAIM_STD**2 / NB1210_CLAIM_N) ** 2 / (NB1210_CLAIM_N - 1)
    )
    welch_df = num / den if den > 0 else float("nan")
    welch_p = 2.0 * (1.0 - float(student_t.cdf(abs(welch_t), df=welch_df))) \
        if not np.isnan(welch_df) else float("nan")

    print(f"   nb1210 mean +/- std = {NB1210_CLAIM_MEAN:.4f} +/- "
          f"{NB1210_CLAIM_STD:.5f}  (n={NB1210_CLAIM_N})")
    print(f"   nb2075 mean +/- std = {pooled_mean:.4f} +/- "
          f"{pooled_std:.5f}  (n={n_seeds})")
    print(f"   shift (2075 - 1210) = {mean_shift:+.4f}")
    print(f"   std ratio (30/15)   = {std_ratio:.2f}x")
    print(f"   Welch t test        = t={welch_t:+.3f}, df~{welch_df:.1f}, "
          f"p~{welch_p:.4f}")

    # Does nb2075 CI contain nb1210 claim?
    nb1210_in_deep_ci = ci_low <= NB1210_CLAIM_MEAN <= ci_high
    print(f"   nb1210 mean in nb2075 95%CI?  {nb1210_in_deep_ci}")

    # ---- Deploy refit (mean fold_s across deep seeds) ----
    print("\n" + "-" * 78)
    print(f"DEPLOY REFIT (full 253; mean(fold_s) across all {n_seeds} deep seeds)")
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
    print(f"   deploy weights      = {w_str}")
    print(f"   deploy mu / s       = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   in-sample RAE (253) = {in_rae:.4f}")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}")
    print(f"   te(513) mean / std  = {deploy_te.mean():.3f} / "
          f"{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * pooled_mean + LB_W_TE * te_unb_rae
    lb_band_low = lb_band_est - 0.05
    lb_band_high = lb_band_est + 0.05
    print(f"\n[LB-band deep] {LB_W_OOF:.2f}*OOF({pooled_mean:.4f}) + "
          f"{LB_W_TE:.2f}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}")
    print(f"               band = [{lb_band_low:.4f}, {lb_band_high:.4f}]")

    # ---- DEEP promote gate ----
    print("\n" + "-" * 78)
    print("DEEP PROMOTE GATE")
    print("-" * 78)
    gate_a = pooled_mean <= GATE_MEAN_OOF
    gate_b = pooled_std <= GATE_STD
    gate_pass = gate_a and gate_b
    print(f"   A: mean OOF {pooled_mean:.4f} <= {GATE_MEAN_OOF}  -> "
          f"{'PASS' if gate_a else 'FAIL'}")
    print(f"   B: std OOF  {pooled_std:.4f} <= {GATE_STD}   -> "
          f"{'PASS' if gate_b else 'FAIL'}")
    ci_low_below_gate = ci_low <= GATE_MEAN_OOF
    ci_high_below_gate = ci_high <= GATE_MEAN_OOF
    print(f"   diag: 95%CI low  {ci_low:.4f}  <= {GATE_MEAN_OOF}  -> "
          f"{'YES' if ci_low_below_gate else 'NO'}")
    print(f"   diag: 95%CI high {ci_high:.4f}  <= {GATE_MEAN_OOF}  -> "
          f"{'YES' if ci_high_below_gate else 'NO'}")

    sub_csv_path = SUBMISSIONS / f"{PARENT_TAG}_deploy_pre_pyramid.csv"
    sub_csv_exists = sub_csv_path.exists()
    print(f"\n[deploy-file] {sub_csv_path}  exists={sub_csv_exists}")

    if gate_pass and sub_csv_exists:
        ladder_action = "PROMOTE_PRIMARY_1_LB_SAFE_DEEP_CONFIRMED"
        verdict = "PROMOTE"
        print(f"[ladder] -> PROMOTE {sub_csv_path.name} to "
              f"PRIMARY-1-LB-SAFE (deep 30-seed confirmed)")
    elif gate_pass and not sub_csv_exists:
        ladder_action = "GATE_PASS_DEPLOY_MISSING"
        verdict = "PROMOTE_BUT_FILE_MISSING"
        print(f"[ladder] gate PASS but deploy CSV missing -- run "
              f"{PARENT_TAG} first")
    else:
        ladder_action = "HOLD_PRIMARY_2_ONLY"
        verdict = "HOLD"
        print("[ladder] DEEP HOLD; nb1191 stays PRIMARY-2 only")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "prior_tag": PRIOR_TAG,
        "method": "deep_seed_reverify_PRE_unblind_pyramid_30seed",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds_deep": KF_SEEDS_DEEP,
        "kf_seeds_nb1210": KF_SEEDS_NB1210,
        "n_folds": N_FOLDS,
        "n_seeds_deep": n_seeds,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results_deep": per_seed,
        "pooled_rae_mean_deep": pooled_mean,
        "pooled_rae_std_deep": pooled_std,
        "pooled_rae_median_deep": pooled_median,
        "pooled_rae_min_deep": pooled_min,
        "pooled_rae_max_deep": pooled_max,
        "pooled_rae_p025_deep": pooled_p025,
        "pooled_rae_p975_deep": pooled_p975,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "sem_deep": float(sem),
        "t_crit_df29": t_crit,
        "rae_of_mean_of_seed_oofs_deep": rae_of_mean_seed_oof,
        "nb1210_claim_mean": NB1210_CLAIM_MEAN,
        "nb1210_claim_std": NB1210_CLAIM_STD,
        "nb1210_claim_n": NB1210_CLAIM_N,
        "mean_shift_vs_nb1210": float(mean_shift),
        "std_ratio_vs_nb1210": float(std_ratio),
        "welch_t_vs_nb1210": float(welch_t),
        "welch_df_vs_nb1210": float(welch_df) if not np.isnan(welch_df) else None,
        "welch_p_vs_nb1210": float(welch_p) if not np.isnan(welch_p) else None,
        "nb1210_mean_in_deep_ci95": bool(nb1210_in_deep_ci),
        "nb1191_claim_mean": NB1191_CLAIM_MEAN,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(ANCHORS, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate_deep": lb_band_est,
        "lb_band_low_deep": lb_band_low,
        "lb_band_high_deep": lb_band_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "gate_mean_target": GATE_MEAN_OOF,
        "gate_std_target": GATE_STD,
        "gate_a_mean_le_target": bool(gate_a),
        "gate_b_std_le_target": bool(gate_b),
        "gate_pass": bool(gate_pass),
        "ci95_low_below_gate": bool(ci_low_below_gate),
        "ci95_high_below_gate": bool(ci_high_below_gate),
        "deploy_csv_path": str(sub_csv_path),
        "deploy_csv_exists": bool(sub_csv_exists),
        "ladder_action": ladder_action,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   deep seeds            = {KF_SEEDS_DEEP[0]}..{KF_SEEDS_DEEP[-1]} "
          f"(n={n_seeds})")
    print(f"   mean +/- std (ddof=1) = {pooled_mean:.4f} +/- {pooled_std:.5f}")
    print(f"   95% CI on mean        = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   min / max             = {pooled_min:.4f} / {pooled_max:.4f}")
    print(f"   nb1210 claim (n=15)   = {NB1210_CLAIM_MEAN:.4f} +/- "
          f"{NB1210_CLAIM_STD:.5f}")
    print(f"   shift vs nb1210       = {mean_shift:+.4f}  "
          f"(Welch p~{welch_p:.4f})")
    print(f"   gate mean<={GATE_MEAN_OOF} std<={GATE_STD} -> "
          f"{'PROMOTE' if gate_pass else 'HOLD'}")
    print(f"   ladder_action         = {ladder_action}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_deep",
        "pooled_rae_std_deep",
        "ci95_low",
        "ci95_high",
        "mean_shift_vs_nb1210",
        "welch_p_vs_nb1210",
        "gate_pass",
        "ladder_action",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
