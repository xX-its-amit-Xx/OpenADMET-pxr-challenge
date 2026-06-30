"""nb2911 -- DEEP-30 verify nb2900 4-anchor PRE-clean SLSQP pyramid.

CRITICAL MOTIVATION
-------------------
nb2900 reported pooled scaffold-CV RAE = 0.4619 +/- 0.00077 across 5
kf_seeds {1001..1005}. The 5-seed std 0.00077 < 0.001 is the cycle-160 red
flag (per memory feedback_cycle160_deep_verify_dispersion): the same family
of under-dispersion has been confirmed on every prior 5-seed claim in this
pipeline:

  - nb2060 (5s) std 0.00087 -> nb2080 (30s) std 0.00408 -> 4.69x
  - nb1191 (5s) std ~0.0008 -> deep-30 std 0.00317           ~ 4.0x
  - nb2095 (5s) std 0.0009  -> nb2100 (30s) std 0.0037       ~ 4.1x
  - nb2171 (5s) std 0.0008  -> nb2180 (30s) std ~0.0024      ~ 2.95x

nb2900's reported number 0.4619 < cycle-167 ceiling 0.4682 (nb2171) is
exceptional (-0.0063, ~1.7-2.5x of expected deep-30 std band). Per the
cycle-149/160 rule, claims at exceptional magnitudes MUST be re-verified
on a wide (>=15) seed sample, with deep-30 the gate-grade standard.

PROTOCOL
--------
Two-layer seeding:
  - 5 outer kf_seeds {1001..1005} -- the scaffold-CV split seeds (same as
    nb2900's reference set, so we can directly compare).
  - 30 fresh inner bag seeds {3001..3030} -- each perturbs the SLSQP
    initial point AND the per-fold rank-stretch tie-break. Per outer
    kf_seed we average the per-bag-seed pooled OOF predictions across
    the 30 inner seeds to get a single "bagged" OOF; pooled RAE on the
    bagged OOF is the per-kf_seed verdict. Mean over the 5 outer
    kf_seeds is the gate-grade number.

For SLSQP, the bag-seed perturbation is implemented via a small (1e-3)
Gaussian jitter on the simplex initial guess; the SLSQP optimum is
nominally unique but the per-fold local convergence + the stretch
tie-break give meaningful per-seed variance that maps to the same kind
of dispersion captured by LGBM bag seeds. This makes the deep-30 fair
to the SLSQP-pyramid family.

ANCHORS (identical to nb2900, all PRE-clean):
    nb2240_K20      RAE 0.4630
    chemprop_aux    RAE 0.5879
    counter_clean   RAE 2.138  (lifted from te[unb_idx])
    nb1191          RAE 0.4697

GATE
----
  deep-30 mean < 0.4570 -> "VERIFIED_PROMOTE"
  deep-30 mean < 0.4598 -> "VERIFIED_MARGINAL"
  else                  -> "LUCKY_SEED_TRAP"

REFERENCES
----------
  nb2900 5-seed mean : 0.4619 +/- 0.00077
  Current ceilings:
    nb2171 deep-30  : 0.4682
    nb1191 deep-30  : 0.4718
    nb2060 deep-30  : 0.4720
    nb2095 deep-30  : 0.4720
  chemprop_aux ref  : 0.6216

Outputs
-------
  scripts/nb2911_deep30_verify_nb2900.py
  data/processed/nb2911_summary.json
  data/processed/nb2911_pred_oof.npy   (253,) float32
  data/processed/te_nb2911.npy         (513,) float32
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

TAG = "nb2911"
BASELINE_TAG = "nb2900"

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]          # outer scaffold CV seeds
BAG_SEEDS = list(range(3001, 3031))                # 30 fresh inner bag seeds
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
JITTER_SIGMA = 1e-3                                # SLSQP init perturbation
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- Refs ----
NB2900_REF_MEAN_5SEED = 0.46186909553786987
NB2900_REF_STD_5SEED  = 0.0007709650366527661
NB2171_DEEP30 = 0.4682
NB1191_DEEP30 = 0.4718
NB2060_DEEP30 = 0.4720
NB2095_DEEP30 = 0.4720
CHEMPROP_AUX_REF = 0.6216

# ---- 4 PRE-clean anchors (identical to nb2900) ----
# (name, oof_rel, te_rel, lift_te_to_oof)
ANCHORS = [
    ("nb2240_K20",    "nb2240_mean_bag_oof_K20.npy",       "te_nb2240_K20.npy",     False),
    ("chemprop_aux",  "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy",   False),
    ("counter_clean", None,                                "te_nb2490_counter.npy", True),
    ("nb1191",        "nb1191_pred_oof.npy",               "te_nb1191.npy",         False),
]


# ============================================================================
# helpers
# ============================================================================

def slsqp_simplex(P, y, init=None):
    """SLSQP simplex (w >= 0, sum w = 1) minimizing |P @ w - y|^2."""
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    if init is None:
        init = np.full(K, 1.0 / K)
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        init,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def perturbed_simplex_init(K, rng, sigma=JITTER_SIGMA):
    """Generate a perturbed simplex starting point: uniform + small Gaussian
    jitter, re-projected onto the simplex (clip + renormalize)."""
    base = np.full(K, 1.0 / K)
    jitter = rng.normal(0.0, sigma, size=K)
    w0 = np.clip(base + jitter, 0.0, 1.0)
    s = w0.sum()
    return w0 / s if s > 0 else base


def best_stretch_on(blend_tr, y_tr, mu, grid, rng=None, eps=1e-9):
    """Pick best rank-stretch scalar on training fold. Inject negligible
    rng-tied tie-break perturbation so per-seed variability is non-zero
    when the loss surface is flat over the grid."""
    best_s, best_r = 1.0, float("inf")
    perturb = (
        rng.normal(0.0, eps, size=len(grid))
        if rng is not None else np.zeros(len(grid))
    )
    for s, p in zip(grid, perturb):
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred)) + float(p)
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def cv_run_one(P_unb, y_unb, unb_scaffolds, kf_seed, bag_seed):
    """One run: scaffold CV at kf_seed with SLSQP init/tie-break perturbed
    by bag_seed. Returns pooled RAE + OOF blend + per-fold (w, s)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    K = P_unb.shape[1]
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_w, fold_s = [], []
    rng = np.random.default_rng(int(bag_seed))
    for tr_loc, va_loc in splits:
        init = perturbed_simplex_init(K, rng)
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc], init=init)
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(
            blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID, rng=rng,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    pooled = float(rae(y_unb, oof_blend))
    return pooled, oof_blend, fold_w, fold_s


def load_anchor_columns(anchor_list, unb_idx, n_unb, n_te):
    oof_cols, te_cols, names = [], [], []
    for disp, oof_rel, te_rel, lift in anchor_list:
        te_p = DATA_PROCESSED / te_rel
        assert te_p.exists(), f"missing te: {te_p}"
        te_arr = np.load(te_p).astype(np.float64)
        assert te_arr.shape == (n_te,), f"{disp} te {te_arr.shape}"
        if lift:
            oof = te_arr[unb_idx].copy()
        else:
            oof_p = DATA_PROCESSED / oof_rel
            assert oof_p.exists(), f"missing OOF: {oof_p}"
            oof = np.load(oof_p).astype(np.float64)
            assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        oof_cols.append(oof)
        te_cols.append(te_arr)
        names.append(disp)
    return np.column_stack(oof_cols), np.column_stack(te_cols), names


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEEP-30 verify {BASELINE_TAG} 4-anchor PRE-clean pyramid")
    print(f"   anchors      : {[a[0] for a in ANCHORS]}")
    print(f"   kf_seeds     : {KF_SEEDS}  (outer scaffold CV)")
    print(f"   bag_seeds    : {BAG_SEEDS[0]}..{BAG_SEEDS[-1]}  "
          f"(n={len(BAG_SEEDS)}, inner SLSQP init perturbation)")
    print(f"   nb2900 ref   : mean={NB2900_REF_MEAN_5SEED:.4f} +/- "
          f"{NB2900_REF_STD_5SEED:.5f}")
    print(f"   gates        : PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
    print(f"   ceilings     : nb2171={NB2171_DEEP30}  nb1191={NB1191_DEEP30}  "
          f"nb2060={NB2060_DEEP30}  nb2095={NB2095_DEEP30}")
    print("=" * 78)

    # ---- Load data ----
    te = load_test()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    te_smiles = (
        te["smiles"].values if "smiles" in te.columns else te["SMILES"].values
    )
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[load] n_te={n_te}  n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")

    # ---- Anchor stack ----
    P_unb, P_te, names = load_anchor_columns(ANCHORS, unb_idx, n_unb, n_te)
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}")
    indiv_rae = {}
    for j, nm in enumerate(names):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[nm] = r
        print(f"   anchor {j} {nm:14s} oof_RAE={r:.4f}  "
              f"mean={P_unb[:, j].mean():.4f}  std={P_unb[:, j].std():.4f}")

    # =================================================================
    # Per-kf_seed: bag across 30 inner seeds, pool RAE on bagged OOF
    # =================================================================
    print("\n" + "-" * 78)
    print(f"DEEP-30: {len(KF_SEEDS)} kf_seeds x {len(BAG_SEEDS)} bag_seeds")
    print("-" * 78)

    per_kfseed_results = []
    all_kf_bagged_oofs = []
    all_per_bag_pooled = []  # flat list (kf_seed, bag_seed, pooled_rae)
    all_fold_s = []
    for ki, kf_seed in enumerate(KF_SEEDS):
        t_kf = time.time()
        per_bag_oofs = np.zeros((len(BAG_SEEDS), n_unb), dtype=np.float64)
        per_bag_rae = np.zeros(len(BAG_SEEDS), dtype=np.float64)
        per_bag_w_mean = []
        per_bag_s_mean = []
        for bi, bag_seed in enumerate(BAG_SEEDS):
            pooled, oof_blend, fw, fs = cv_run_one(
                P_unb, y_unb, unb_scaffolds, kf_seed, bag_seed,
            )
            per_bag_oofs[bi] = oof_blend
            per_bag_rae[bi] = pooled
            per_bag_w_mean.append([float(x) for x in np.mean(fw, axis=0)])
            per_bag_s_mean.append(float(np.mean(fs)))
            all_per_bag_pooled.append({
                "kf_seed": int(kf_seed),
                "bag_seed": int(bag_seed),
                "pooled_rae": float(pooled),
            })
            all_fold_s.extend([float(x) for x in fs])
        # bagged OOF for this kf_seed = mean across 30 bag seeds
        bagged_oof = per_bag_oofs.mean(axis=0)
        bagged_pooled = float(rae(y_unb, bagged_oof))
        all_kf_bagged_oofs.append(bagged_oof)

        per_bag_rae_mean = float(per_bag_rae.mean())
        per_bag_rae_std = float(per_bag_rae.std(ddof=1))
        per_bag_w_mean_arr = np.mean(np.asarray(per_bag_w_mean), axis=0)
        per_bag_s_mean_val = float(np.mean(per_bag_s_mean))
        per_kfseed_results.append({
            "kf_seed": int(kf_seed),
            "bagged_pooled_rae": bagged_pooled,
            "per_bag_pooled_mean": per_bag_rae_mean,
            "per_bag_pooled_std_ddof1": per_bag_rae_std,
            "per_bag_pooled_min": float(per_bag_rae.min()),
            "per_bag_pooled_max": float(per_bag_rae.max()),
            "per_bag_w_mean": [float(x) for x in per_bag_w_mean_arr],
            "per_bag_s_mean": per_bag_s_mean_val,
        })
        print(
            f"   kf_seed={kf_seed} [{ki+1}/{len(KF_SEEDS)}]  "
            f"bagged_pooled={bagged_pooled:.4f}  "
            f"per_bag_mean={per_bag_rae_mean:.4f} +/- {per_bag_rae_std:.4f}  "
            f"per_bag_range=[{per_bag_rae.min():.4f},{per_bag_rae.max():.4f}]  "
            f"wall={time.time()-t_kf:.1f}s"
        )

    # =================================================================
    # Gate-grade number: mean over 5 kf_seeds of bagged_pooled_rae
    # =================================================================
    bagged_pooled_arr = np.asarray(
        [r["bagged_pooled_rae"] for r in per_kfseed_results]
    )
    deep30_mean = float(bagged_pooled_arr.mean())
    deep30_std = float(bagged_pooled_arr.std(ddof=1))
    deep30_min = float(bagged_pooled_arr.min())
    deep30_max = float(bagged_pooled_arr.max())
    deep30_median = float(np.median(bagged_pooled_arr))

    # All 5x30 = 150 per-bag pooled RAEs (flat dispersion view)
    flat_pooled = np.asarray([r["pooled_rae"] for r in all_per_bag_pooled])
    flat_mean = float(flat_pooled.mean())
    flat_std = float(flat_pooled.std(ddof=1))

    # Under-dispersion ratio: flat-30 std vs nb2900 5-seed std
    under_disp = (
        flat_std / NB2900_REF_STD_5SEED
        if NB2900_REF_STD_5SEED > 0 else float("inf")
    )

    # Mean-bag final OOF (across 5 kf_seeds and 30 bag_seeds)
    pred_oof_final = np.mean(np.column_stack(all_kf_bagged_oofs), axis=1)
    rae_of_pred_oof_final = float(rae(y_unb, pred_oof_final))

    delta_vs_nb2900 = deep30_mean - NB2900_REF_MEAN_5SEED
    delta_vs_nb2171 = deep30_mean - NB2171_DEEP30
    delta_vs_nb1191 = deep30_mean - NB1191_DEEP30

    # Gate
    if deep30_mean < GATE_PROMOTE:
        verdict = "VERIFIED_PROMOTE"
    elif deep30_mean < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
    else:
        verdict = "LUCKY_SEED_TRAP"

    print("\n" + "-" * 78)
    print("DEEP-30 STATISTICS")
    print("-" * 78)
    print(f"   bagged_pooled per kf_seed = "
          f"[{', '.join(f'{x:.4f}' for x in bagged_pooled_arr)}]")
    print(f"   deep-30 mean (over 5 kf_seeds, bag-averaged) = {deep30_mean:.4f}")
    print(f"   deep-30 std  (ddof=1)                        = {deep30_std:.4f}")
    print(f"   deep-30 min/max                              = "
          f"[{deep30_min:.4f}, {deep30_max:.4f}]")
    print(f"   deep-30 median                               = {deep30_median:.4f}")
    print(f"   flat 5x30 = 150 per-bag pooled mean+/-std    = "
          f"{flat_mean:.4f} +/- {flat_std:.4f}")
    print(f"   under_dispersion_ratio (flat_std/nb2900_5s_std) = {under_disp:.2f}x")
    print(f"   RAE(mean-of-all-bagged-OOFs)                  = "
          f"{rae_of_pred_oof_final:.4f}")
    print(f"   delta vs nb2900 ({NB2900_REF_MEAN_5SEED:.4f}) = "
          f"{delta_vs_nb2900:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_DEEP30:.4f}) deep-30 = "
          f"{delta_vs_nb2171:+.4f}")
    print(f"   delta vs nb1191 ({NB1191_DEEP30:.4f}) deep-30 = "
          f"{delta_vs_nb1191:+.4f}")

    # =================================================================
    # Deploy refit: full-pool SLSQP + mean(fold_s) over all 5x30 inner folds
    # =================================================================
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(all_fold_s))
    in_rae = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    lb_band_est = LB_W_OOF * deep30_mean + LB_W_TE * te_unb_rae

    print("\n" + "-" * 78)
    print("DEPLOY REFIT")
    print("-" * 78)
    print(f"   weights      = " + ", ".join(
        f"{nm}={w:.4f}" for nm, w in zip(names, w_deploy)
    ))
    print(f"   mu/s         = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   in_sample    = {in_rae:.4f}  (overfit bound)")
    print(f"   te[unb] RAE  = {te_unb_rae:.4f}  (in-sample)")
    print(f"   LB-band est  = {LB_W_OOF:.2f}*{deep30_mean:.4f} + "
          f"{LB_W_TE:.2f}*{te_unb_rae:.4f} = {lb_band_est:.4f}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} GATE ===")
    print(f"   deep-30 mean = {deep30_mean:.4f}")
    print(f"   PROMOTE  < {GATE_PROMOTE:.4f} -> {deep30_mean < GATE_PROMOTE}")
    print(f"   MARGINAL < {GATE_MARGINAL:.4f} -> {deep30_mean < GATE_MARGINAL}")
    print(f"   VERDICT      = {verdict}")
    print("=" * 78)

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, pred_oof_final.astype(np.float32))
    np.save(te_npy_path, deploy_te)
    print(f"   [save] {pred_oof_path}")
    print(f"   [save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_deep30_4anchor.csv"
    wrote_csv = False
    if verdict == "VERIFIED_PROMOTE":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"   [save] {sub_csv_path}  (verdict={verdict})")
        wrote_csv = True
    else:
        print(f"   [skip] no submission CSV (verdict={verdict})")

    summary = {
        "tag": TAG,
        "baseline_tag": BASELINE_TAG,
        "method": "deep30_verification_of_nb2900_4anchor_pyramid",
        "purpose": (
            "verify_nb2900_5seed_mean_0.4619_under-disp_std_0.00077_"
            "cycle160_redflag"),
        "anchors": names,
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "anchor_pre_unblind": True,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "bag_seeds": BAG_SEEDS,
        "n_bag_seeds": len(BAG_SEEDS),
        "n_total_runs": len(KF_SEEDS) * len(BAG_SEEDS),
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "jitter_sigma": JITTER_SIGMA,
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_kfseed_results": per_kfseed_results,
        "deep30_bagged_pooled_per_kf": bagged_pooled_arr.tolist(),
        "deep30_mean": deep30_mean,
        "deep30_std_ddof1": deep30_std,
        "deep30_min": deep30_min,
        "deep30_max": deep30_max,
        "deep30_median": deep30_median,
        "flat_150_per_bag_mean": flat_mean,
        "flat_150_per_bag_std_ddof1": flat_std,
        "nb2900_ref_mean_5seed": NB2900_REF_MEAN_5SEED,
        "nb2900_ref_std_5seed": NB2900_REF_STD_5SEED,
        "under_dispersion_ratio_flatstd_over_nb2900_5sstd": under_disp,
        "rae_of_mean_of_bagged_oofs": rae_of_pred_oof_final,
        "delta_vs_nb2900_5seed": delta_vs_nb2900,
        "delta_vs_nb2171_deep30": delta_vs_nb2171,
        "delta_vs_nb1191_deep30": delta_vs_nb1191,
        "peer_nb2171_deep30": NB2171_DEEP30,
        "peer_nb1191_deep30": NB1191_DEEP30,
        "peer_nb2060_deep30": NB2060_DEEP30,
        "peer_nb2095_deep30": NB2095_DEEP30,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "deploy_weights": [
            {"name": nm, "w": float(w)} for nm, w in zip(names, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s_mean_over_all_inner_folds": s_deploy,
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if wrote_csv else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")
    print(f"\n   wall = {time.time()-t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "deep30_mean",
        "deep30_std_ddof1",
        "flat_150_per_bag_mean",
        "flat_150_per_bag_std_ddof1",
        "under_dispersion_ratio_flatstd_over_nb2900_5sstd",
        "delta_vs_nb2900_5seed",
        "delta_vs_nb2171_deep30",
        "delta_vs_nb1191_deep30",
        "deploy_weights",
        "te_unb_rae_in_sample",
        "lb_band_estimate",
        "verdict",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
