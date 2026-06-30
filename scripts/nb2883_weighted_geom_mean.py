"""nb2883 -- Weighted geometric mean of 3 PRE-clean anchors (SLSQP simplex in log-space).

NEW PARADIGM (vs cycle 208 nb2580):
    nb2580 SLSQP-log-space-geometric-mean across 3-4 anchors collapsed to
    a near-equal-weight degenerate blend (effectively reproducing nb2240).
    This script narrows the anchor pool to exactly the 3 verified-clean
    PRE-unblind anchors and re-runs SLSQP in log-space, keeping the same
    paradigm but tightening the substrate:

        yhat = exp( sum_k w_k * log(a_k) )    s.t. w_k >= 0, sum w_k = 1

ANCHORS (all PRE-unblind clean, no nb730/POST-unblind contamination):
    1. nb2240_K20     -- chemprop_aux + K=20 RFE LGBM residual mean-bag
    2. chemprop_aux   -- 4139 PRE-unblind multitask MPNN (frozen anchor)
    3. counter_clean  -- nb2490 counter-assay joint K=20 residual (nb730-free)

PROTOCOL:
    - Clip per-anchor predictions at EPSILON=1.0 (truth_min = 1.745).
    - For each fold/seed: SLSQP simplex on log-space residuals
        minimize  || P_log[tr] @ w - log(y)[tr] ||^2
        s.t.      w_k >= 0, sum_k w_k = 1
      Apply on fold-val:
        pred_va = exp( P_log[va] @ w )  clipped to [Y_MIN, Y_MAX] = [3.0, 8.0]
    - 5-fold scaffold CV on 253 unblind across 5 kf_seeds {42, 1, 7, 137, 1009}.
    - Per-seed pooled RAE in pEC50 space; mean_rae averaged across 5 seeds.

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT
    else               -> FAIL

Outputs:
    data/processed/nb2883_summary.json
    data/processed/nb2883_pred_oof.npy   (253,) float32
    data/processed/te_nb2883.npy         (513,) float32
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
from rdkit import RDLogger
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2883"

# ---- CV ----
N_FOLDS = 5
KF_SEEDS = [42, 1, 7, 137, 1009]

# ---- Geometric mean clip ----
EPSILON = 1.0          # log floor; truth_min observed 1.745, all anchors > 1.4
Y_MIN = 3.0            # final pEC50 clip lower
Y_MAX = 8.0            # final pEC50 clip upper

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- 3 PRE-clean anchors (no POST-unblind contamination, no nb730 chain) ----
ANCHOR_SPEC = [
    ("nb2240_K20",
     DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
     DATA_PROCESSED / "te_nb2240_K20.npy",
     "PRE-clean K=20 RFE residual stack on chemprop_aux"),
    ("chemprop_aux",
     DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
     DATA_PROCESSED / "te_chemprop_aux.npy",
     "PRE-clean (4139 PRE-unblind only; verified frozen anchor)"),
    ("counter_clean",
     DATA_PROCESSED / "nb2490_pred_oof.npy",
     DATA_PROCESSED / "te_nb2490.npy",
     "PRE-clean counter-assay K=20 residual joint anchor (nb730-free)"),
]


# ============================================================================
# helpers
# ============================================================================

def slsqp_simplex_logspace(P_log: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    """SLSQP on log-space residuals: minimize ||P_log @ w - y_log||^2."""
    K = P_log.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P_log @ w - y_log) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def clip_then_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, EPSILON))


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- weighted geometric mean of 3 PRE-clean anchors (log-space SLSQP)")
    print("=" * 78)

    # ---- load test + unblind ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- load anchors (all 3 must exist; no graceful fallback) ----
    anchor_names = []
    anchor_prov = {}
    oof_cols, te_cols = [], []
    for name, oof_path, te_path, prov in ANCHOR_SPEC:
        if not oof_path.exists() or not te_path.exists():
            raise FileNotFoundError(
                f"Required PRE-clean anchor missing: {name}  "
                f"oof={oof_path.exists()} te={te_path.exists()}"
            )
        oof = np.load(oof_path).astype(np.float64)
        te_v = np.load(te_path).astype(np.float64)
        if oof.shape[0] != n_unb:
            raise ValueError(f"{name} pred_oof shape {oof.shape} != ({n_unb},)")
        if te_v.shape[0] != n_test:
            raise ValueError(f"{name} te shape {te_v.shape} != ({n_test},)")
        anchor_names.append(name)
        anchor_prov[name] = prov
        oof_cols.append(oof)
        te_cols.append(te_v)
    K = len(anchor_names)
    assert K == 3, f"Expected exactly 3 anchors, got K={K}"
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)

    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    std_anchors = {k: float(P_unb[:, i].std()) for i, k in enumerate(anchor_names)}
    truth_std = float(y_unb.std())
    print(f"[anchors] K={K}  truth_std={truth_std:.3f}")
    for k in anchor_names:
        ratio = std_anchors[k] / truth_std if truth_std > 0 else float("nan")
        print(f"   {k:14s}  in_RAE={rae_anchors[k]:.4f}  std={std_anchors[k]:.3f}  "
              f"std/truth_std={ratio:.3f}  [{anchor_prov[k]}]")

    # ---- precompute log anchors / log truth (EPSILON-clipped) ----
    P_unb_log = clip_then_log(P_unb)
    P_te_log = clip_then_log(P_te)
    y_unb_log = clip_then_log(y_unb)
    print(f"[log] log(anchor) ranges: min={P_unb_log.min():.3f}  "
          f"max={P_unb_log.max():.3f}")
    print(f"[log] log(truth)  range: min={y_unb_log.min():.3f}  "
          f"max={y_unb_log.max():.3f}  std={y_unb_log.std():.3f}")

    # ---- in-sample equal-weight diagnostics (arith vs geo) ----
    geo_eq = np.exp(P_unb_log.mean(axis=1))
    arith_eq = P_unb.mean(axis=1)
    rae_geo_eq = float(rae(y_unb, geo_eq))
    rae_arith_eq = float(rae(y_unb, arith_eq))
    print(f"[diag] equal-weight in-sample:  arith RAE={rae_arith_eq:.4f}  "
          f"geo RAE={rae_geo_eq:.4f}  delta(geo-arith)={rae_geo_eq-rae_arith_eq:+.4f}")

    # ---- scaffold CV ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}  K_anchors={K}")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_weights = []
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        t_seed = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
        fold_weights = []
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            # SLSQP simplex weights on log-space residuals (fold-train)
            w = slsqp_simplex_logspace(
                P_unb_log[tr_loc, :], y_unb_log[tr_loc],
            )
            # Apply on fold-val
            log_pred_va = P_unb_log[va_loc, :] @ w
            pred_va = np.exp(log_pred_va)
            pred_va = np.clip(pred_va, Y_MIN, Y_MAX)
            oof_blend[va_loc] = pred_va
            fold_weights.append(w.tolist())

        if np.isnan(oof_blend).any():
            raise RuntimeError("OOF has NaN -- scaffold splits did not cover all rows")
        pooled = float(rae(y_unb, oof_blend))
        per_seed_pooled.append(pooled)
        per_seed_fold_weights.append(fold_weights)
        oof_seed_stack[s_idx] = oof_blend
        mean_w = np.mean(np.asarray(fold_weights), axis=0)
        print(
            f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  "
            f"mean_w=[" + ", ".join(f"{nm}={mw:.3f}" for nm, mw in zip(anchor_names, mean_w))
            + f"]  wall={time.time()-t_seed:.1f}s"
        )

    mean_rae = float(np.mean(per_seed_pooled))
    std_rae = float(np.std(per_seed_pooled))
    oof_final = oof_seed_stack.mean(axis=0)
    rae_mean_oof = float(rae(y_unb, oof_final))
    print(f"\n[wide-seed] mean pooled RAE = {mean_rae:.4f} +/- {std_rae:.4f} "
          f"(n_seeds={len(KF_SEEDS)})")
    print(f"[wide-seed] RAE(seed-mean OOF) = {rae_mean_oof:.4f}")
    print(f"[diag] oof_final std = {oof_final.std():.3f}  (truth {truth_std:.3f})")

    # ---- gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL)  "
          f"->  {verdict}")

    # ---- deploy: refit on all 253 (log-space), apply to 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit log-space SLSQP simplex on all 253, predict 513")
    print("-" * 78)
    w_deploy = slsqp_simplex_logspace(P_unb_log, y_unb_log)
    log_te_pred = P_te_log @ w_deploy
    te_pred = np.exp(log_te_pred).astype(np.float32)
    te_pred = np.clip(te_pred, Y_MIN, Y_MAX).astype(np.float32)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy weights = "
          + ", ".join(f"{nm}={w:.4f}" for nm, w in zip(anchor_names, w_deploy)))
    print(f"   te mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << mean_rae)")

    # ---- save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_final.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_weighted_geom_mean.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "weighted_geometric_mean_3_PRE_clean_anchors_log_space_slsqp_simplex",
        "paradigm": "weighted_geometric_mean_vs_arithmetic_convex_blend",
        "anchor_spec": [a[0] for a in ANCHOR_SPEC],
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_prov,
        "anchor_in_rae": rae_anchors,
        "anchor_std": std_anchors,
        "anchor_pre_unblind": True,
        "truth_std": truth_std,
        "epsilon_clip_floor": EPSILON,
        "y_min_pec50_clip": Y_MIN,
        "y_max_pec50_clip": Y_MAX,
        "rae_equal_weight_arith_in_sample": rae_arith_eq,
        "rae_equal_weight_geo_in_sample": rae_geo_eq,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_fold_weights": per_seed_fold_weights,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_oof": rae_mean_oof,
        "oof_final_std": float(oof_final.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_weights": {nm: float(w) for nm, w in zip(anchor_names, w_deploy)},
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K anchors                = {K}  ({anchor_names})")
    print(f"   per-seed pooled RAE      = "
          f"{[float('%.4f' % r) for r in per_seed_pooled]}")
    print(f"   MEAN pooled RAE (5 sd)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   RAE(seed-mean OOF)       = {rae_mean_oof:.4f}")
    print(f"   gate                     = <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL  ->  {verdict}")
    print(f"   deploy weights           = "
          + ", ".join(f"{nm}={w:.3f}" for nm, w in zip(anchor_names, w_deploy)))
    print(f"   te[unb_idx] in-sample    = {te_unb_in:.4f}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "rae_of_mean_oof",
        "verdict",
        "deploy_weights",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
