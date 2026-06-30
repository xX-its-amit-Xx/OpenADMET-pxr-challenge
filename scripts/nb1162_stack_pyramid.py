"""nb1162 -- Stacking pyramid: 5 anchors -> SLSQP -> rank-stretch.

Stage 1 anchors (OOF on the 253 unblind, cached):
    0. nb2103_K28      data/processed/nb2103_mean_bag_oof_K28.npy
    1. chemprop_aux    data/processed/nb1133_chemprop_aux_pred_oof.npy
    2. nb730_honest    data/processed/nb730_honest_pred_oof.npy
    3. nb503           data/processed/nb503_pred_oof.npy
    4. nb562           data/processed/nb562_pred_oof.npy

For deploy on the 513 test compounds we use the matched te files:
    nb2103   -> te_chemprop_aux.npy + per-row nb2103 residual is NOT cached;
                 fall back to te_chemprop_aux.npy as nb2103 anchor proxy.
                 (nb2103 is the K=28 SHAP-pruned LGBM residual stack on top of
                 chemprop_aux; cached te file is not present so deploy uses the
                 chemprop_aux te as its proxy for the stretched blend.)
    chemprop_aux -> te_chemprop_aux.npy
    nb730_honest -> te_nb730_honest.npy
    nb503        -> te_nb503.npy
    nb562        -> te_nb562.npy

Stage 2 SLSQP: convex blend (w >= 0, sum = 1) under scaffold 5-fold CV on
the 253 unblind. Loss = SSE.

Stage 3 rank-stretch (per-fold): grid s in {1.00, 1.025, 1.05, 1.075, 1.10,
1.125, 1.15}. For deploy on the 513, apply mean(per-fold s) around the
blend mean of the deploy blend (mu = mean of full-253 deploy blend).

Final RAE = pooled cross-fit RAE on 253.

Gates:
    A) final_rae <= 0.5027
    B) deploy_stretch_mean_s > 1.0   (rank-stretch alive)
If BOTH gates pass -> build deploy CSV submissions/nb1162_stack_pyramid.csv.

Outputs:
    scripts/nb1162_stack_pyramid.py (this file)
    data/processed/nb1162_summary.json
    data/processed/te_nb1162.npy
    submissions/nb1162_stack_pyramid.csv  (only if gate passes)
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

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1162"
GATE_RAE = 0.5027
N_FOLDS = 5
SEED = 42
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

# (display_name, oof_path_relative_to_DATA_PROCESSED, te_path_relative)
# te paths describe how to fan deploy onto the 513-row test set.
# For nb2103 we use te_chemprop_aux as the proxy (nb2103 is the residual
# stack anchored on chemprop_aux but its 513-row deploy te is not cached).
ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("nb730_honest", "nb730_honest_pred_oof.npy",         "te_nb730_honest.npy"),
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex blend weights minimizing SSE of (P @ w) vs y."""
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


def best_stretch_on(blend_tr: np.ndarray, y_tr: np.ndarray,
                    mu: float, grid) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 5-anchor stacking pyramid (SLSQP + rank-stretch)")
    print("=" * 78)

    # ---- Load 513 test + scaffolds for the 253 unblind ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    n_singleton = sum(1 for s in unb_scaffolds if not s)
    print(f"[scaffold] unique={n_unique_scaf}  singleton(no-ring)={n_singleton}")

    # ---- Load anchor OOFs (253) and te (513) ----
    oof_cols = []
    te_cols = []
    indiv_rae = {}
    print("\n[anchors]")
    for disp, oof_rel, te_rel in ANCHORS:
        oof_p = DATA_PROCESSED / oof_rel
        te_p = DATA_PROCESSED / te_rel
        assert oof_p.exists(), f"missing OOF: {oof_p}"
        assert te_p.exists(),  f"missing te:  {te_p}"
        oof = np.load(oof_p).astype(np.float64)
        te_arr = np.load(te_p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof shape {oof.shape}"
        assert te_arr.shape == (n_te,), f"{disp} te shape {te_arr.shape}"
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

    # =================================================================
    # Stage 2+3: SLSQP + rank-stretch under scaffold 5-fold CV
    # =================================================================
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV   SLSQP simplex + stretch grid {STRETCH_GRID}")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=SEED,
    )
    oof_blend = np.full(n_unb, np.nan)
    fold_rows = []
    fold_s = []
    fold_w = []
    for k, (tr_loc, va_loc) in enumerate(splits):
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(
            blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof_blend[va_loc]))
        fold_w.append(w_f)
        fold_s.append(s_f)
        fold_rows.append({
            "fold": k,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "w": [float(x) for x in w_f],
            "s": float(s_f),
            "mu_tr": mu_tr,
            "train_rae": rae_tr,
            "val_rae": rae_va,
        })
        w_str = ",".join(f"{x:.3f}" for x in w_f)
        print(f"   fold {k}: n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  "
              f"w=[{w_str}]  s={s_f:.3f}  mu={mu_tr:.3f}  "
              f"train={rae_tr:.4f}  val={rae_va:.4f}")

    pooled_rae = float(rae(y_unb, oof_blend))
    s_mean = float(np.mean(fold_s))
    print(f"\n[cv] pooled scaffold-CV RAE on 253 = {pooled_rae:.4f}")
    print(f"[cv] per-fold s = {[round(x,3) for x in fold_s]}  mean={s_mean:.4f}")

    # =================================================================
    # Gate evaluation
    # =================================================================
    gate_a = pooled_rae <= GATE_RAE
    gate_b = s_mean > 1.0
    gate_pass = gate_a and gate_b
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   gate A: pooled RAE {pooled_rae:.4f} <= {GATE_RAE:.4f}  -> "
          f"{'PASS' if gate_a else 'FAIL'}")
    print(f"   gate B: mean(fold_s) {s_mean:.4f} > 1.0              -> "
          f"{'PASS' if gate_b else 'FAIL'}")
    print(f"   overall: {'PASS' if gate_pass else 'FAIL'}")

    # =================================================================
    # Deploy: convex weights refit on all 253; stretch = mean of fold s
    #         applied around deploy-blend mean.
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY (refit weights on 253, apply mean(fold_s) to 513)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = s_mean
    in_rae_final = float(rae(
        y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)
    ))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)

    w_str = ", ".join(
        f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy)
    )
    print(f"   deploy weights = {w_str}")
    print(f"   deploy mu      = {mu_deploy:.4f}")
    print(f"   deploy s       = {s_deploy:.4f}")
    print(f"   in-sample RAE  = {in_rae_final:.4f}  (overfit lower bound)")
    print(f"   te(513) mean/std = {deploy_te.mean():.3f} / "
          f"{deploy_te.std():.3f}")

    # Save te artefact (always; useful even if gate fails)
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)

    sub_csv_path = SUBMISSIONS / f"{TAG}_stack_pyramid.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"\n[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(f"\n[skip] gate FAILED -> no submission CSV written "
              f"(would be {sub_csv_path})")

    summary = {
        "tag": TAG,
        "method": "5-anchor_stack_SLSQP_then_rank_stretch_scaffoldCV",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae": indiv_rae,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "n_singleton_scaffolds": n_singleton,
        "fold_results": fold_rows,
        "fold_s_per_fold": [float(x) for x in fold_s],
        "fold_s_mean": s_mean,
        "pooled_scaffold_cv_rae": pooled_rae,
        "gate_rae_target": GATE_RAE,
        "gate_a_pooled_le_target": bool(gate_a),
        "gate_b_stretch_alive_s_gt_1": bool(gate_b),
        "gate_pass": bool(gate_pass),
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(ANCHORS, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_final,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_pass else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled scaffold-CV RAE = {pooled_rae:.4f}")
    print(f"   mean fold s            = {s_mean:.4f}")
    print(f"   gate A (<= {GATE_RAE})    = {gate_a}")
    print(f"   gate B (s>1)           = {gate_b}")
    print(f"   gate overall           = {gate_pass}")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_scaffold_cv_rae",
        "fold_s_mean",
        "gate_pass",
        "deploy_weights",
        "deploy_s",
        "in_sample_rae_overfit_bound",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
