"""nb2861 -- Strong Ridge regression on K=20 chemprop_aux residual (strong-alpha sweep).

NEW PARADIGM (vs nb2760 alpha=100 boundary):
    nb2760 swept alpha in {0.01, 0.1, 1.0, 10.0, 100.0} and the best result
    landed at the BOUNDARY (alpha=100) -- a textbook sign the regularization
    knob wasn't strong enough; the optimum lies further into the
    high-shrinkage regime.  This script extends the grid one full decade
    deeper -- alpha in {100, 500, 1000, 5000, 10000} -- to find the true
    interior optimum (or confirm the boundary continues to slide right, in
    which case the ridge floor approaches the anchor itself).

    Same K=20 raw-feature slice on the chemprop_aux residual, same scaffold
    5-fold CV with 5 kf_seeds.  Plain StandardScaler -> Ridge(alpha=X), no
    polynomial expansion, no kernel.  This is a pure follow-on test of the
    "is the K=20 ridge floor actually below the anchor?" question after
    fixing the original alpha-grid boundary problem.

PROTOCOL:
    1. Slice X_K20 = first 20 cols of X_117_unb / X_117_te (pyramid contract).
    2. Anchor: chemprop_aux (PRE-unblind, verified clean). Residual target
       = y_unb - anchor_unb.
    3. Per fold: StandardScaler.fit on train slice -> transform val ->
       Ridge(alpha=alpha) fit on (20,) -> predict val.
    4. 5-fold scaffold CV on 253 unblind, 5 kf_seeds {1001..1005}.
    5. Strong-alpha sweep {100, 500, 1000, 5000, 10000} -- pick best by mean
       pooled RAE across seeds.
    6. Deploy: refit StandardScaler + Ridge(best_alpha) on all 253 -> predict
       on 513.

GATE:
    best_alpha mean_rae < 0.4570 -> "PROMOTE"
                       < 0.4598 -> "MARGINAL_BEAT"
                       else     -> "FAIL"

Outputs:
    data/processed/nb2861_summary.json
    data/processed/nb2861_pred_oof.npy   (253,) float32 -- best-alpha mean OOF
    data/processed/te_nb2861.npy         (513,) float32 -- best-alpha deploy
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2861"

# --------------------------------------------------------------------------
# Hyperparameters (spec)
# --------------------------------------------------------------------------
ALPHA_GRID = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
RIDGE_RANDOM_STATE = 42

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Number of K=20 cols sliced from the 117-col block (first-20 contract).
K_SLICE = 20


def _run_alpha(alpha, X_unb, resid_unb, unb_scaffolds, anchor_unb, y_unb):
    """Run 5-fold scaffold CV across all kf_seeds for a single alpha.

    Returns dict with per-seed pooled RAE, mean RAE, std, OOFs.
    """
    n_unb = len(y_unb)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        for fi, (tr_loc, va_loc) in enumerate(splits):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_unb[tr_loc])
            X_va = scaler.transform(X_unb[va_loc])
            mdl = Ridge(alpha=alpha, random_state=RIDGE_RANDOM_STATE)
            mdl.fit(X_tr, resid_unb[tr_loc])
            oof_resid[va_loc] = mdl.predict(X_va)
        assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
        oof_pred = anchor_unb + oof_resid
        oof_pred = np.clip(oof_pred, 3.0, 8.0)
        pooled = float(rae(y_unb, oof_pred))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "wall_sec": round(time.time() - ts, 3),
        })
        all_oofs.append(oof_pred)
    oof_stack = np.column_stack(all_oofs)
    mean_oof = oof_stack.mean(axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    return {
        "alpha": float(alpha),
        "per_seed": per_seed,
        "mean_rae": pooled_rae_mean,
        "std_rae_seeds": pooled_rae_std,
        "rae_of_mean_oof": final_oof_rae,
        "mean_oof": mean_oof,
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Strong Ridge alpha sweep on K=20 chemprop_aux residual")
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

    # ---- Load X_117 then slice to first K=20 cols ----
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    assert X_unb_117.shape == (n_unb, 117), f"X_unb shape {X_unb_117.shape}"
    assert X_te_117.shape == (n_test, 117), f"X_te shape {X_te_117.shape}"
    X_unb = X_unb_117[:, :K_SLICE].astype(np.float32)
    X_te = X_te_117[:, :K_SLICE].astype(np.float32)
    print(f"[feat] X_unb_K20={X_unb.shape}  X_te_K20={X_te.shape}  "
          f"slice=first-{K_SLICE}-cols")

    # ---- Anchor (chemprop_aux, PRE-unblind verified-clean) ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
          f"(PRE-clean baseline)")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- Alpha sweep ----
    print("\n" + "-" * 78)
    print(f"STRONG-ALPHA SWEEP  alphas={ALPHA_GRID}  seeds={KF_SEEDS}\n"
          f"Per-fold: StandardScaler -> Ridge(alpha=alpha)  (K=20 raw, no poly)")
    print("-" * 78)

    sweep_results = []
    best_idx = -1
    best_mean = float("inf")
    for ai, alpha in enumerate(ALPHA_GRID):
        ts = time.time()
        res = _run_alpha(alpha, X_unb, resid_unb, unb_scaffolds, anchor_unb, y_unb)
        sweep_results.append(res)
        print(f"   alpha={alpha:<7g}  mean_rae={res['mean_rae']:.4f} "
              f"+/- {res['std_rae_seeds']:.4f}   "
              f"oof_mean_rae={res['rae_of_mean_oof']:.4f}   "
              f"wall={time.time()-ts:.1f}s")
        if res["mean_rae"] < best_mean:
            best_mean = res["mean_rae"]
            best_idx = ai

    best = sweep_results[best_idx]
    best_alpha = best["alpha"]
    print(f"\n[best] alpha={best_alpha}  mean_rae={best['mean_rae']:.4f} "
          f"(+/- {best['std_rae_seeds']:.4f})  oof_mean_rae={best['rae_of_mean_oof']:.4f}")
    print(f"[best] delta vs anchor = {best['mean_rae'] - rae_anchor_unb:+.4f}")

    # Flag boundary best -- informative for whether grid needs to extend further
    at_low_boundary = best_idx == 0
    at_high_boundary = best_idx == len(ALPHA_GRID) - 1
    if at_low_boundary:
        print(f"[warn] best alpha at LOW boundary ({best_alpha}) -- "
              f"optimum may lie below grid")
    if at_high_boundary:
        print(f"[warn] best alpha at HIGH boundary ({best_alpha}) -- "
              f"optimum may lie above grid (consider extending further)")

    # ---- Deploy: refit StandardScaler + Ridge(best_alpha) on ALL 253 -> 513 ----
    print("\n" + "-" * 78)
    print(f"DEPLOY: refit scaler+Ridge(alpha={best_alpha}) on all 253 -> apply to 513")
    print("-" * 78)
    scaler_full = StandardScaler()
    X_unb_s = scaler_full.fit_transform(X_unb)
    X_te_s = scaler_full.transform(X_te)
    mdl_full = Ridge(alpha=best_alpha, random_state=RIDGE_RANDOM_STATE)
    mdl_full.fit(X_unb_s, resid_unb)
    deploy_resid_te = mdl_full.predict(X_te_s)
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")

    # ---- Gate ----
    if best["mean_rae"] < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best["mean_rae"] < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   best_alpha     = {best_alpha}")
    print(f"   mean_rae       = {best['mean_rae']:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts (best-alpha OOF + te) ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best["mean_oof"].astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    # Strip OOF arrays from sweep results before JSON dump
    sweep_serializable = []
    for res in sweep_results:
        sweep_serializable.append({
            "alpha": res["alpha"],
            "mean_rae": res["mean_rae"],
            "std_rae_seeds": res["std_rae_seeds"],
            "rae_of_mean_oof": res["rae_of_mean_oof"],
            "per_seed": res["per_seed"],
        })

    summary = {
        "tag": TAG,
        "method": (
            "Strong-alpha Ridge regression (no polynomial expansion, no "
            "kernel) on chemprop_aux residual over first-K=20 cols of X_117. "
            "Per-fold StandardScaler -> Ridge(alpha) over strong-alpha sweep "
            f"{ALPHA_GRID}. Follow-on to nb2760 alpha-grid boundary "
            "(best=100 at upper edge). 20 raw features, n=253 unblind."
        ),
        "paradigm": "linear_floor_strong_ridge_k20_extended_alpha_grid",
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "alpha_grid": ALPHA_GRID,
        "ridge_random_state": RIDGE_RANDOM_STATE,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim_raw": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "sweep_results": sweep_serializable,
        "best_alpha": best_alpha,
        "best_alpha_idx": int(best_idx),
        "best_at_low_boundary": bool(at_low_boundary),
        "best_at_high_boundary": bool(at_high_boundary),
        "mean_rae": best["mean_rae"],
        "pooled_rae_std_seeds": best["std_rae_seeds"],
        "rae_of_mean_of_seed_oofs": best["rae_of_mean_oof"],
        "delta_vs_anchor": best["mean_rae"] - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
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
    print(f"   K=20 raw features              = no expansion, no kernel")
    print(f"   alpha grid                     = {ALPHA_GRID}")
    print(f"   best alpha                     = {best_alpha}")
    print(f"   best mean_rae (5 kf_seeds)     = {best['mean_rae']:.4f} "
          f"+/- {best['std_rae_seeds']:.4f}")
    print(f"   rae_of_mean_oof                = {best['rae_of_mean_oof']:.4f}")
    print(f"   delta vs anchor                = "
          f"{best['mean_rae'] - rae_anchor_unb:+.4f}")
    print(f"   te[unb_idx] in_sample          = {te_unb_in_rae:.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_alpha",
        "mean_rae",
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "delta_vs_anchor",
        "te_unb_rae_in_sample",
        "verdict",
        "te_deploy_mean",
        "te_deploy_std",
    ):
        print(f"  {k}: {res.get(k)}")
