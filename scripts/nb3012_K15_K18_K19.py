"""nb3012 -- Per-fold SLSQP simplex on {K15, K18, K19_deep30}.

NEW PARADIGM:
    Test smaller K=15 cached anchor alongside K18 and K19_deep30. The
    cycle-160+ ceiling work has clustered around K18-K28; this probes
    whether a smaller K=15 LGBM residual brings orthogonal value to the
    pyramid bag. K=15 is cached from nb2261 (mean-bag OOF on chemprop_aux
    residual using top-15 SHAP-ranked features); K=18 is the cycle-160
    canonical (nb2604); K=19 is the recent deep-30 rebuild (nb3000).

ANCHORS (3 PRE-clean, all chemprop_aux residual K-band):
    1. K15_nb2261       -- 15-feat SHAP residual mean-bag (PRE)
    2. K18_nb2604       -- 18-feat residual mean-bag (PRE, cycle-160 canon)
    3. K19_nb3000       -- 19-feat residual deep-30 seed bag (PRE)

PROTOCOL:
    - 5-fold scaffold CV on 253 unblind, single kf_seed=1001.
    - Per-fold SLSQP simplex w (sum=1, w>=0) on fold-TRAIN OOF rows.
    - Apply fold-w to fold-VAL rows -> per-row OOF prediction.
    - Pool across folds -> pooled RAE = mean (single seed per spec).
    - NO rank-stretch.  NO bias shift.

GATE:
    mean_rae < 0.4511 -> BETTER
    else              -> FAIL/SKIP

Outputs:
    data/processed/nb3012_summary.json
    data/processed/nb3012_pred_oof.npy   (253,) float32 -- OOF
    data/processed/te_nb3012.npy         (513,) float32 -- deploy refit on all 253
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

TAG = "nb3012"

# ---- CV ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Multi-start SLSQP ----
N_STARTS = 8

# ---- Gates ----
GATE_BETTER = 0.4511

# ---- 3 PRE-clean K-band anchors ----
ANCHOR_SPEC = [
    ("K15_nb2261",
     DATA_PROCESSED / "nb2261_mean_bag_oof_K15.npy",
     DATA_PROCESSED / "te_nb2261_K15.npy",
     "PRE-clean K=15 SHAP-ranked LGBM residual mean-bag on chemprop_aux"),
    ("K18_nb2604",
     DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy",
     DATA_PROCESSED / "te_nb2604_K18.npy",
     "PRE-clean K=18 LGBM residual mean-bag on chemprop_aux (cycle-160 canon)"),
    ("K19_nb3000",
     DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
     DATA_PROCESSED / "te_nb3000_K19.npy",
     "PRE-clean K=19 LGBM residual deep-30 seed bag on chemprop_aux"),
]


# ============================================================================
# helpers
# ============================================================================

def slsqp_simplex(P: np.ndarray, y: np.ndarray,
                  n_starts: int = N_STARTS, seed: int = 0) -> tuple[np.ndarray, float]:
    """SLSQP on simplex; multi-start with seeded Dirichlet inits + uniform.

    Returns (w_best, rae_at_w_best).
    """
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def obj(w):
        return float(rae(y, P @ w))

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
    return best_w, float(best_obj)


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-fold SLSQP on {{K15, K18, K19_deep30}} (kf_seed={KF_SEED})")
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

    # ---- check K=15 availability (skip-with-marker if missing) ----
    k15_oof_path = ANCHOR_SPEC[0][1]
    k15_te_path = ANCHOR_SPEC[0][2]
    if not k15_oof_path.exists() or not k15_te_path.exists():
        skip_summary = {
            "tag": TAG,
            "method": "per_fold_slsqp_K15_K18_K19_deep30",
            "verdict": "SKIPPED_NO_K15",
            "reason": f"K=15 cached OOF/te missing: oof={k15_oof_path.exists()} te={k15_te_path.exists()}",
            "k15_oof_path": str(k15_oof_path),
            "k15_te_path": str(k15_te_path),
            "kf_seed": KF_SEED,
            "wall_sec": round(time.time() - t0, 2),
        }
        out_path = DATA_PROCESSED / f"{TAG}_summary.json"
        with open(out_path, "w") as f:
            json.dump(skip_summary, f, indent=2)
        print(f"[skip] {out_path}")
        return skip_summary

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
    for i, k in enumerate(anchor_names):
        print(f"   {k:14s}  in_RAE={rae_anchors[k]:.4f}  std={P_unb[:, i].std():.3f}")

    # ---- per-fold SLSQP simplex (single seed) ----
    print("\n" + "-" * 78)
    print(f"Per-fold SLSQP simplex  kf_seed={KF_SEED}  folds={N_FOLDS}  starts={N_STARTS}")
    print("-" * 78)

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_weights = []
    fold_rae_tr = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        w, rae_tr = slsqp_simplex(
            P_unb[tr_loc, :], y_unb[tr_loc],
            n_starts=N_STARTS, seed=KF_SEED * 100 + f_idx,
        )
        pred_va = P_unb[va_loc, :] @ w
        oof_blend[va_loc] = pred_va
        fold_weights.append(w.tolist())
        fold_rae_tr.append(rae_tr)
        print(f"   fold={f_idx}  train_rae={rae_tr:.4f}  "
              + "w=[" + ", ".join(f"{nm}={w_:.3f}"
                                  for nm, w_ in zip(anchor_names, w)) + "]")

    if np.isnan(oof_blend).any():
        raise RuntimeError("OOF has NaN -- scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    mean_fold_w = np.mean(np.asarray(fold_weights), axis=0)
    print(f"\n[summary]  pooled_rae={pooled:.4f}")
    print(f"[mean fold w] "
          + ", ".join(f"{nm}={w_:.3f}" for nm, w_ in zip(anchor_names, mean_fold_w)))

    mean_rae = pooled

    # ---- gate ----
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    print(f"[gate] threshold(<{GATE_BETTER} BETTER) -> {verdict}")

    # ---- deploy: refit single SLSQP simplex on all 253, apply to 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit SLSQP simplex on all 253, predict 513")
    print("-" * 78)
    w_deploy, rae_deploy = slsqp_simplex(P_unb, y_unb, n_starts=16, seed=0)
    te_pred = (P_te @ w_deploy).astype(np.float32)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy weights = "
          + ", ".join(f"{nm}={w_:.4f}" for nm, w_ in zip(anchor_names, w_deploy)))
    print(f"   raw in-sample RAE = {rae_deploy:.4f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << mean_rae)")
    print(f"   te mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    # ---- save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_blend.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_K15_K18_K19.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "per_fold_slsqp_K15_K18_K19_deep30",
        "paradigm": "small_K_probe_per_fold_simplex_no_post_hoc",
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
        "kf_seed": KF_SEED,
        "n_starts_slsqp": N_STARTS,
        "fold_weights": fold_weights,
        "fold_train_rae": fold_rae_tr,
        "mean_fold_weights": {nm: float(w_) for nm, w_ in zip(anchor_names, mean_fold_w)},
        "mean_rae": mean_rae,
        "pooled_rae": pooled,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "deploy_weights": {nm: float(w_) for nm, w_ in zip(anchor_names, w_deploy)},
        "deploy_in_sample_rae": rae_deploy,
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
    print(f"   K                        = {K}")
    print(f"   pooled_rae (kf_seed={KF_SEED}) = {mean_rae:.4f}")
    print(f"   gate                     = <{GATE_BETTER} BETTER  ->  {verdict}")
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
        "mean_rae", "verdict", "deploy_weights",
        "te_unb_in_sample_rae", "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
