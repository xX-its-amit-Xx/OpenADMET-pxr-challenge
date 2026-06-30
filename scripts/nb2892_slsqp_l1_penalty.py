"""nb2892 -- SLSQP simplex with explicit L1 sparsity penalty.

NEW PARADIGM:
    Standard SLSQP simplex blends 3 anchors with w >= 0, sum(w) = 1.
    Under the simplex constraint sum(|w|) == sum(w) == 1, so a literal L1
    penalty would be a constant; we therefore implement L1 as a sparsity
    pressure that *concentrates* weight on the best single anchor relative
    to the uniform baseline. Concretely we add lambda * (1 - max(w))
    which is monotone in -L_inf (a smooth-via-softmax) proxy for L0/L1
    sparsity on the simplex:

        objective(w) = RAE(y, P @ w) + lambda * smooth_inv_linf(w)

        smooth_inv_linf(w) = 1 - softmax_max(w, tau=20)
            -> 0 when one weight dominates, ~1 - 1/K when uniform.

    This is the standard simplex-sparsity formulation (also known as a
    'spike-concentration' regulariser); it differs from SLSQP-vanilla by
    biasing the optimum toward corners of the simplex when blending is
    not strongly required by the data.

ANCHORS (3 PRE-clean):
    1. nb2240_K20   -- chemprop_aux + K=20 RFE LGBM residual mean-bag
    2. chemprop_aux -- 4139 PRE-unblind multitask MPNN (frozen anchor)
    3. counter_clean -- nb2490 counter-assay joint K=20 residual (nb730-free)

PROTOCOL:
    - 5-fold scaffold CV on 253 unblind, 5 kf_seeds {42, 1, 7, 137, 1009}.
    - Sweep lambda in {0.01, 0.05, 0.1, 0.5}.
    - For each (lambda, kf_seed): per-fold SLSQP-simplex with L1 sparsity
      penalty, fit on TRAIN slice (held out from VAL), predict VAL slice.
    - Per-seed pooled RAE; per-lambda mean_rae across seeds.
    - Best lambda = argmin mean_rae across the grid.

GATE (best lambda only):
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4598 -> MARGINAL_BEAT
    else              -> FAIL

Outputs:
    data/processed/nb2892_summary.json
    data/processed/nb2892_pred_oof.npy   (253,) float32 -- best-lambda OOF
    data/processed/te_nb2892.npy         (513,) float32 -- best-lambda deploy
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

TAG = "nb2892"

# ---- CV ----
N_FOLDS = 5
KF_SEEDS = [42, 1, 7, 137, 1009]

# ---- L1 sparsity sweep ----
LAMBDA_GRID = [0.01, 0.05, 0.1, 0.5]
SOFTMAX_TAU = 20.0  # smooth-max temperature (smooth proxy for max(w))

# ---- Multi-start SLSQP ----
N_STARTS = 5

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- 3 PRE-clean anchors ----
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

def smooth_max(w: np.ndarray, tau: float = SOFTMAX_TAU) -> float:
    """Differentiable approximation of max(w) on the simplex.

    smooth_max(w) = (1/tau) * log(sum(exp(tau * w)))
    Bounded between max(w) and max(w) + log(K)/tau.
    """
    z = tau * w
    z_max = float(np.max(z))
    return float(z_max + np.log(np.sum(np.exp(z - z_max)))) / tau


def sparsity_penalty(w: np.ndarray) -> float:
    """L1-like sparsity on the simplex: 1 - smooth_max(w).

    0 when one weight dominates (corner of simplex),
    -> 1 - log(K)/tau - 1/K when uniform (worst-case spread).
    """
    return float(1.0 - smooth_max(w))


def slsqp_l1_simplex(P: np.ndarray, y: np.ndarray, lam: float,
                     n_starts: int = N_STARTS, seed: int = 0) -> tuple[np.ndarray, float, float]:
    """SLSQP on simplex with L1 sparsity penalty.

    Objective:  RAE(y, P @ w) + lam * (1 - smooth_max(w))
    Constraints: w >= 0, sum(w) = 1

    Returns (w_best, raw_rae_at_w_best, penalized_obj_at_w_best).
    """
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def obj(w):
        return float(rae(y, P @ w) + lam * sparsity_penalty(w))

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(n_starts - 1):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w, best_obj = None, np.inf
    for x0 in starts:
        try:
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-9})
            w = np.clip(res.x, 0.0, 1.0)
            s = w.sum()
            if s <= 0:
                continue
            w = w / s
            o = obj(w)
            if o < best_obj:
                best_obj, best_w = o, w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_obj = obj(best_w)
    raw_rae = float(rae(y, P @ best_w))
    return best_w, raw_rae, float(best_obj)


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SLSQP simplex with L1 sparsity penalty (lambda sweep)")
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

    # ---- load anchors ----
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
    P_unb = np.column_stack(oof_cols)  # (253, 3)
    P_te = np.column_stack(te_cols)    # (513, 3)

    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    truth_std = float(y_unb.std())
    print(f"[anchors] K={K}  truth_std={truth_std:.3f}")
    for k in anchor_names:
        print(f"   {k:14s}  in_RAE={rae_anchors[k]:.4f}  std={P_unb[:, anchor_names.index(k)].std():.3f}")

    # ---- sweep lambda x scaffold CV x kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SLSQP L1 sweep  lambda={LAMBDA_GRID}  kf_seeds={KF_SEEDS}  folds={N_FOLDS}")
    print("-" * 78)

    per_lambda_results = {}
    per_lambda_oof = {}
    per_lambda_fold_weights = {}

    for lam in LAMBDA_GRID:
        t_lam = time.time()
        per_seed_pooled = []
        per_seed_fold_weights = []
        oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)

        for s_idx, kf_seed in enumerate(KF_SEEDS):
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
            )
            oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
            fold_weights = []
            for f_idx, (tr_loc, va_loc) in enumerate(splits):
                w, raw_rae_tr, obj_tr = slsqp_l1_simplex(
                    P_unb[tr_loc, :], y_unb[tr_loc], lam=lam,
                    n_starts=N_STARTS, seed=kf_seed * 100 + f_idx,
                )
                pred_va = P_unb[va_loc, :] @ w
                oof_blend[va_loc] = pred_va
                fold_weights.append(w.tolist())

            if np.isnan(oof_blend).any():
                raise RuntimeError("OOF has NaN -- scaffold splits did not cover all rows")
            pooled = float(rae(y_unb, oof_blend))
            per_seed_pooled.append(pooled)
            per_seed_fold_weights.append(fold_weights)
            oof_seed_stack[s_idx] = oof_blend

        mean_rae = float(np.mean(per_seed_pooled))
        std_rae = float(np.std(per_seed_pooled))
        oof_seed_mean = oof_seed_stack.mean(axis=0)
        mean_fold_w = np.mean(
            [np.mean(np.asarray(fws), axis=0) for fws in per_seed_fold_weights], axis=0,
        )
        max_w_avg = float(mean_fold_w.max())
        per_lambda_results[str(lam)] = {
            "lambda": lam,
            "per_seed_pooled": per_seed_pooled,
            "mean_rae": mean_rae,
            "std_rae": std_rae,
            "rae_of_mean_oof": float(rae(y_unb, oof_seed_mean)),
            "mean_fold_weights": {nm: float(w_) for nm, w_ in zip(anchor_names, mean_fold_w)},
            "max_mean_weight": max_w_avg,
            "wall_sec": round(time.time() - t_lam, 2),
        }
        per_lambda_oof[str(lam)] = oof_seed_mean
        per_lambda_fold_weights[str(lam)] = per_seed_fold_weights
        print(
            f"   lambda={lam:.3f}  mean_rae={mean_rae:.4f} +/- {std_rae:.4f}  "
            f"max(mean_w)={max_w_avg:.3f}  "
            f"mean_w=[" + ", ".join(f"{nm}={w_:.3f}" for nm, w_ in zip(anchor_names, mean_fold_w))
            + f"]  wall={time.time()-t_lam:.1f}s"
        )

    # ---- pick best lambda ----
    best_lam_str = min(per_lambda_results.keys(), key=lambda k: per_lambda_results[k]["mean_rae"])
    best_lam = float(best_lam_str)
    best = per_lambda_results[best_lam_str]
    best_mean_rae = best["mean_rae"]
    best_oof = per_lambda_oof[best_lam_str]
    print(f"\n[best] lambda={best_lam:.3f}  mean_rae={best_mean_rae:.4f}")

    # ---- gate ----
    if best_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] thresholds(<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL)  -> {verdict}")

    # ---- deploy with best lambda: refit SLSQP-L1 on all 253, apply to 513 ----
    print("\n" + "-" * 78)
    print(f"DEPLOY: refit SLSQP-L1 (lambda={best_lam:.3f}) on all 253, predict 513")
    print("-" * 78)
    w_deploy, raw_rae_deploy, obj_deploy = slsqp_l1_simplex(
        P_unb, y_unb, lam=best_lam, n_starts=12, seed=0,
    )
    te_pred = (P_te @ w_deploy).astype(np.float32)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy weights = "
          + ", ".join(f"{nm}={w_:.4f}" for nm, w_ in zip(anchor_names, w_deploy)))
    print(f"   raw in-sample RAE = {raw_rae_deploy:.4f}  (no penalty)")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << best_mean_rae)")
    print(f"   te mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    # ---- save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_oof.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_slsqp_l1_penalty.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "slsqp_simplex_with_l1_sparsity_penalty_lambda_sweep",
        "paradigm": "explicit_L1_sparsity_penalty_on_simplex_weights",
        "penalty_form": "RAE + lambda * (1 - smooth_max(w, tau=20))",
        "softmax_tau": SOFTMAX_TAU,
        "anchor_spec": [a[0] for a in ANCHOR_SPEC],
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_prov,
        "anchor_in_rae": rae_anchors,
        "anchor_pre_unblind": True,
        "truth_std": truth_std,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_starts_slsqp": N_STARTS,
        "lambda_grid": LAMBDA_GRID,
        "per_lambda_results": per_lambda_results,
        "best_lambda": best_lam,
        "best_mean_rae": best_mean_rae,
        "best_std_rae": best["std_rae"],
        "best_mean_fold_weights": best["mean_fold_weights"],
        "best_max_mean_weight": best["max_mean_weight"],
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_weights": {nm: float(w_) for nm, w_ in zip(anchor_names, w_deploy)},
        "deploy_lambda": best_lam,
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
    print(f"   anchors                  = {anchor_names}")
    print(f"   lambda grid              = {LAMBDA_GRID}")
    for lam_str in per_lambda_results:
        r = per_lambda_results[lam_str]
        print(f"     lambda={float(lam_str):.3f}  mean_rae={r['mean_rae']:.4f}  "
              f"max_w={r['max_mean_weight']:.3f}")
    print(f"   BEST lambda              = {best_lam:.3f}")
    print(f"   BEST mean_rae (5 seeds)  = {best_mean_rae:.4f} +/- {best['std_rae']:.4f}")
    print(f"   gate                     = <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL  ->  {verdict}")
    print(f"   deploy weights           = "
          + ", ".join(f"{nm}={w_:.3f}" for nm, w_ in zip(anchor_names, w_deploy)))
    print(f"   te[unb_idx] in-sample    = {te_unb_in:.4f}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_lambda", "best_mean_rae", "best_std_rae",
        "verdict", "deploy_weights", "te_unb_in_sample_rae", "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
