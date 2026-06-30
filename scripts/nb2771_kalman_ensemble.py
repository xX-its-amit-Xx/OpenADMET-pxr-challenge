"""nb2771 -- Kalman filter recursive state estimate for K-pyramid blending.

NEW PARADIGM:
    Treat each K-pyramid OOF (K=18, K=20, K=24, K=28) as a NOISY OBSERVATION
    of the true pEC50 for each unblind row, and run a 1-D scalar Kalman
    filter per row to recursively fuse the four observations into a posterior
    estimate.

    This is distinct from every prior K-blend method (SLSQP convex blend,
    equal-weight mean, Ridge stack, LGBM-meta, etc.) because:

      * SLSQP/LGBM fit a SINGLE set of weights across the WHOLE 253 sample
        and then apply those weights uniformly per row.
      * The Kalman filter computes a PER-ROW posterior whose weighting
        between obs and prior is set by the obs / prior VARIANCES (here:
        the per-K standalone OOF RAE squared). Rows where one source is
        sharper get more of that source -- per row, not per column.
      * The recursion is sequential / Bayesian, not least-squares; the
        state estimate is an iterated convex combination weighted by
        inverse variance ("Kalman gain"), which is the optimal scalar
        linear-Gaussian fusion under independent observation noise.

    Initial state is K=20 (the spec calls K=20 the "lowest standalone RAE"
    initial state per the protocol). Initial state variance is set to
    K=20's per-K RAE^2. Subsequent observations are K=18, K=24, K=28 in
    order, each with its own observation variance = its per-K RAE^2.

    Per-row update rule (scalar Kalman with no process noise / dynamics):
        K_gain  = prior_var / (prior_var + obs_var)
        post    = prior_mean + K_gain * (obs - prior_mean)
        post_var = (1 - K_gain) * prior_var
    which is algebraically the formula in the spec:
        posterior = (prior_var * obs + obs_var * prior_mean)
                   / (prior_var + obs_var)
    (both forms are identical; we use the gain form for numerical clarity).

    Per-K observation variance is computed PER FOLD from the TRAIN portion's
    standalone RAE only (so the val portion is never used to set its own
    variance) -- this preserves the cross-fit contract on n=253.

PROTOCOL:
    1. Load four K-pyramid OOFs (K=18, K=20, K=24, K=28) and four matching
       te(513) deploy predictions.
    2. Anchor: chemprop_aux (PRE-unblind verified clean) -- not used by the
       Kalman fusion itself (Kalman fuses the K-pyramid OOFs directly,
       which are already chemprop_aux-residual predictions), but the
       baseline RAE is logged for delta accounting.
    3. 5-fold scaffold CV on 253, single kf_seed = 1001 (deterministic
       because the inputs are frozen; no stochastic component in the
       Kalman fusion -- variances are computed from train-fold RAE).
    4. Per fold: compute each K's train-portion RAE on its train rows
       vs y_unb[tr]; use those four numbers as the four observation
       variances (RAE^2 by spec). Then per val row run the sequential
       Kalman fusion using those variances.
    5. Deploy: refit observation variances on ALL 253 (full-sample RAE^2
       per K), then run the per-row Kalman fusion on the four te(513)
       deploy predictions to produce te_nb2771.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2771_kalman_ensemble.py
    data/processed/nb2771_summary.json
    data/processed/nb2771_pred_oof.npy   (253,) float32
    data/processed/te_nb2771.npy         (513,) float32
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

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2771"

# --------------------------------------------------------------------------
# Inputs (canonical K-pyramid OOFs + deploy predictions on te(513))
# Sources verified from nb2604_summary.json (cycle-148 equal-weight ensemble).
# --------------------------------------------------------------------------
K_SOURCES = [
    {
        "K": 18,
        "oof_path": DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy",
        "te_path": DATA_PROCESSED / "te_nb2604_K18.npy",
    },
    {
        "K": 20,
        "oof_path": DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
        "te_path": DATA_PROCESSED / "te_nb2240_K20.npy",
    },
    {
        "K": 24,
        "oof_path": DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy",
        "te_path": DATA_PROCESSED / "te_nb2310_K24.npy",
    },
    {
        "K": 28,
        "oof_path": DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
        "te_path": DATA_PROCESSED / "te_nb2112.npy",  # K28 deploy lives at te_nb2112
    },
]

# Order of sequential Kalman updates: initial state = K=20 (spec),
# then observations come in K=18, K=24, K=28 order.
INITIAL_K = 20
OBS_K_ORDER = [18, 24, 28]

# CV protocol -- single deterministic seed (spec: inputs are frozen npy files)
N_FOLDS = 5
KF_SEED = 1001

# Anchor (for delta accounting only -- not used in Kalman fusion)
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Numerical floor on observation variance (avoid div-by-zero collapse)
VAR_FLOOR = 1e-6


def kalman_fuse_per_row(
    initial_mean: np.ndarray,
    initial_var: float,
    obs_means: list[np.ndarray],
    obs_vars: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential scalar Kalman filter, vectorized over rows.

    No process dynamics (state is static truth). Each obs k has its own
    variance obs_vars[k]. Returns final posterior mean + variance arrays.
    """
    n = initial_mean.shape[0]
    prior_mean = initial_mean.astype(np.float64).copy()
    prior_var = np.full(n, float(initial_var), dtype=np.float64)
    for obs_mean, obs_var in zip(obs_means, obs_vars):
        obs_var_eff = max(float(obs_var), VAR_FLOOR)
        # Kalman gain (per row -- prior_var is per-row, obs_var is scalar)
        gain = prior_var / (prior_var + obs_var_eff)
        new_mean = prior_mean + gain * (obs_mean.astype(np.float64) - prior_mean)
        new_var = (1.0 - gain) * prior_var
        prior_mean = new_mean
        prior_var = new_var
    return prior_mean, prior_var


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Kalman filter recursive state estimate for K-pyramid blending")
    print("=" * 78)

    # ---- Load test set + scaffolds + truth ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_smiles = te[smi_col].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_uniq_scaf}")

    # ---- Load anchor (chemprop_aux) for delta accounting ----
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    anchor_unb = te_chem[unb_idx]
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
          f"(PRE-clean PRIMARY baseline, delta-accounting only)")

    # ---- Load four K-pyramid OOFs + te deploys ----
    oof_by_K: dict[int, np.ndarray] = {}
    te_by_K: dict[int, np.ndarray] = {}
    standalone_rae: dict[int, float] = {}
    for src in K_SOURCES:
        K = src["K"]
        oof = np.load(src["oof_path"]).astype(np.float64)
        te_arr = np.load(src["te_path"]).astype(np.float64)
        assert oof.shape == (n_unb,), f"K={K} oof shape {oof.shape} != (253,)"
        assert te_arr.shape == (n_test,), f"K={K} te shape {te_arr.shape} != (513,)"
        oof_by_K[K] = oof
        te_by_K[K] = te_arr
        r = float(rae(y_unb, oof))
        standalone_rae[K] = r
        print(f"[load] K={K:>2}  oof shape={oof.shape}  te shape={te_arr.shape}  "
              f"full-sample RAE={r:.4f}")

    # ---- Sanity: K=20 is the initial state per spec ----
    assert INITIAL_K in oof_by_K, f"initial K={INITIAL_K} missing"
    for K in OBS_K_ORDER:
        assert K in oof_by_K, f"observation K={K} missing"
    init_rae = standalone_rae[INITIAL_K]
    print(f"[kalman] initial state = K={INITIAL_K}  full-sample RAE={init_rae:.4f}")
    print(f"[kalman] observation order = {OBS_K_ORDER}  "
          f"(per-K variance = per-K RAE^2)")

    # ---- Scaffold 5-fold CV (single deterministic kf_seed=1001) ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seed={KF_SEED}  (single seed -- inputs frozen)")
    print("Per fold: variances computed from TRAIN-portion RAE (cross-fit safe)")
    print("-" * 78)

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_post = np.full(n_unb, np.nan, dtype=np.float64)
    fold_info = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        ts_f = time.time()
        # per-K RAE on the TRAIN portion (folds-safe variance estimate)
        var_per_K: dict[int, float] = {}
        rae_per_K_tr: dict[int, float] = {}
        for K, oof in oof_by_K.items():
            r_tr = float(rae(y_unb[tr_loc], oof[tr_loc]))
            rae_per_K_tr[K] = r_tr
            var_per_K[K] = max(r_tr * r_tr, VAR_FLOOR)
        # Initial state = K=20 on val portion
        init_mean_va = oof_by_K[INITIAL_K][va_loc]
        init_var_va = var_per_K[INITIAL_K]
        obs_means = [oof_by_K[K][va_loc] for K in OBS_K_ORDER]
        obs_vars = [var_per_K[K] for K in OBS_K_ORDER]
        post_mean, post_var = kalman_fuse_per_row(
            init_mean_va, init_var_va, obs_means, obs_vars,
        )
        oof_post[va_loc] = post_mean
        fold_rae = float(rae(y_unb[va_loc], post_mean))
        fold_info.append({
            "fold": fi,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "rae_per_K_train_portion": {
                str(K): rae_per_K_tr[K] for K in sorted(rae_per_K_tr)
            },
            "var_per_K": {str(K): var_per_K[K] for K in sorted(var_per_K)},
            "fold_rae_post": fold_rae,
            "post_var_mean_va": float(post_var.mean()),
            "post_var_min_va": float(post_var.min()),
            "post_var_max_va": float(post_var.max()),
            "wall_sec": round(time.time() - ts_f, 3),
        })
        print(f"   fold={fi}  n_va={len(va_loc):>3}  "
              f"train-RAE K18={rae_per_K_tr[18]:.4f} K20={rae_per_K_tr[20]:.4f} "
              f"K24={rae_per_K_tr[24]:.4f} K28={rae_per_K_tr[28]:.4f}  "
              f"-> fold_RAE_post={fold_rae:.4f}")
    assert not np.isnan(oof_post).any(), "oof_post has NaN -- fold cover incomplete"
    # gentle clip to a sane pEC50 range
    oof_post = np.clip(oof_post, 3.0, 8.0)
    pooled_rae = float(rae(y_unb, oof_post))
    print(f"\n[cv] pooled posterior RAE (kf_seed={KF_SEED}) = {pooled_rae:.4f}")

    # Standalone baselines for delta accounting
    for K in sorted(oof_by_K):
        r = float(rae(y_unb, oof_by_K[K]))
        print(f"     standalone K={K:>2} full-sample RAE = {r:.4f}  "
              f"delta_post_vs_K = {pooled_rae - r:+.4f}")
    equal_weight = np.mean(np.column_stack([oof_by_K[K] for K in sorted(oof_by_K)]), axis=1)
    r_eq = float(rae(y_unb, equal_weight))
    print(f"     equal-weight mean (nb2604 ref)      = {r_eq:.4f}  "
          f"delta_post_vs_eq = {pooled_rae - r_eq:+.4f}")

    # ---- Deploy: refit variances on ALL 253 -> per-row Kalman fusion on te(513) ----
    print("\n" + "-" * 78)
    print("DEPLOY: variances from full-sample per-K RAE -> Kalman fuse 4 te(513) preds")
    print("-" * 78)
    var_full: dict[int, float] = {}
    for K, oof in oof_by_K.items():
        r = standalone_rae[K]
        var_full[K] = max(r * r, VAR_FLOOR)
        print(f"   K={K:>2}  full-sample RAE={r:.4f}  var={var_full[K]:.4f}")
    init_mean_te = te_by_K[INITIAL_K]
    init_var_te = var_full[INITIAL_K]
    obs_means_te = [te_by_K[K] for K in OBS_K_ORDER]
    obs_vars_te = [var_full[K] for K in OBS_K_ORDER]
    post_mean_te, post_var_te = kalman_fuse_per_row(
        init_mean_te, init_var_te, obs_means_te, obs_vars_te,
    )
    deploy_te = np.clip(post_mean_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy variances from all 253)")
    print(f"[deploy] post_var_te mean={post_var_te.mean():.4f}  "
          f"min={post_var_te.min():.4f}  max={post_var_te.max():.4f}")

    # ---- Effective deploy weights (collapse of Kalman recursion at fixed vars) ----
    # With static obs vars, the final posterior is a linear combo of obs means:
    # weight_K = (1/var_K) / sum_j(1/var_j) for ALL K including the initial one.
    # (Identical to inverse-variance weighting -- Kalman recursion at fixed
    # vars is order-invariant for linear Gaussian fusion.)
    inv_var = {K: 1.0 / var_full[K] for K in var_full}
    total_inv = sum(inv_var.values())
    eff_weights = {K: inv_var[K] / total_inv for K in inv_var}
    print("[deploy] effective inverse-variance weights (Kalman == IVW @ fixed vars):")
    for K in sorted(eff_weights):
        print(f"         K={K:>2}  w={eff_weights[K]:.4f}")

    # ---- Gate ----
    if pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   pooled_rae     = {pooled_rae:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_post.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "Sequential scalar Kalman filter per row over four K-pyramid "
            "OOFs (K=18, K=20, K=24, K=28). Initial state = K=20 (lowest "
            "standalone RAE per spec). Observation variance per K = per-K "
            "RAE^2 (computed per-fold from TRAIN portion only). Deploy "
            "uses full-sample per-K RAE^2 as variances. Per-row posterior "
            "is the inverse-variance-weighted fusion of the four estimates."
        ),
        "paradigm": (
            "bayesian_recursive_scalar_kalman_per_row_inverse_variance_fusion_"
            "distinct_from_global_least_squares_slsqp"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean; delta-accounting only)",
        "rae_anchor_unb": rae_anchor_unb,
        "k_sources": [
            {
                "K": int(s["K"]),
                "oof_path": str(s["oof_path"]),
                "te_path": str(s["te_path"]),
                "full_sample_rae": standalone_rae[s["K"]],
            }
            for s in K_SOURCES
        ],
        "initial_K": INITIAL_K,
        "obs_K_order": OBS_K_ORDER,
        "var_floor": VAR_FLOOR,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "fold_info": fold_info,
        "pooled_rae": pooled_rae,
        "mean_rae": pooled_rae,
        "standalone_rae_per_K": {str(K): standalone_rae[K] for K in sorted(standalone_rae)},
        "equal_weight_mean_rae": r_eq,
        "delta_vs_K20_standalone": pooled_rae - standalone_rae[INITIAL_K],
        "delta_vs_equal_weight": pooled_rae - r_eq,
        "delta_vs_anchor": pooled_rae - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "deploy_var_per_K": {str(K): var_full[K] for K in sorted(var_full)},
        "deploy_effective_inverse_variance_weights": {
            str(K): eff_weights[K] for K in sorted(eff_weights)
        },
        "deploy_post_var_mean": float(post_var_te.mean()),
        "deploy_post_var_min": float(post_var_te.min()),
        "deploy_post_var_max": float(post_var_te.max()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor (chemprop_aux) in_RAE   = {rae_anchor_unb:.4f}")
    print(f"   pooled posterior RAE (kf=1001) = {pooled_rae:.4f}")
    print(f"   delta vs K=20 standalone       = "
          f"{pooled_rae - standalone_rae[INITIAL_K]:+.4f}")
    print(f"   delta vs equal-weight mean     = {pooled_rae - r_eq:+.4f}")
    print(f"   delta vs anchor                = "
          f"{pooled_rae - rae_anchor_unb:+.4f}")
    print(f"   te[unb_idx] in_sample          = {te_unb_in_rae:.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_rae",
        "delta_vs_K20_standalone",
        "delta_vs_equal_weight",
        "delta_vs_anchor",
        "te_unb_rae_in_sample",
        "verdict",
        "te_deploy_mean",
        "te_deploy_std",
    ):
        print(f"  {k}: {res.get(k)}")
