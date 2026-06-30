"""nb2742 -- sklearn HistGradientBoostingRegressor with monotonic constraints on K=20.

NEW PARADIGM:
    sklearn.ensemble.HistGradientBoostingRegressor (LightGBM-inspired, sklearn-
    native) brings TWO distinct properties vs the LGBM / CatBoost zoo already
    swept on the (anchor=chemprop_aux, K=20, n=253) substrate:

      1. BINNING / SPLIT ALGORITHM differs:
         - LGBM: GOSS sampling + histogram with leaf-wise growth + EFB.
         - CatBoost (Plain & Ordered): symmetric/oblivious trees,
           target statistic + ordered boosting.
         - HistGB: per-feature 255-bin histograms, level-wise growth,
           no target leakage path, sklearn-canonical regularization
           (L2 on leaf values, early stopping on validation slice).
         These are different inductive biases over the SAME histogram
         primitive; HistGB has not been visited on the K=20 substrate yet.

      2. MONOTONIC CONSTRAINTS encode PXR biology directly:
         PXR is a hydrophobic-pocket nuclear receptor (~1300 cubic-A LBD);
         many literature SAR studies show LogP + lipophilic surface area
         correlate POSITIVELY with PXR activation up to a solubility wall.
         monotonic_cst = +1 on a LogP-proxy column forces every tree split
         on that feature to respect non-decreasing pred-as-feature-grows.
         All other K=20 cols left at 0 (unconstrained).

K=20 LOGP-PROXY DETECTION:
    The first 20 cols of X_117 are AtomPair FP bits (per nb1020
    build_5way_117col_matrix), which carry no LogP semantics by themselves.
    We therefore use the Pearson correlation between each K=20 column and
    the y_unb truth as a proxy: the single most-positively-correlated column
    is treated as a LogP-up surrogate and given monotonic_cst=+1; all
    others remain 0. This treats monotonic_cst as a SOFT regularizer on the
    single most LogP-aligned axis (positive activity correlation = the
    "more lipophilic, more PXR active" direction in the data).
    The choice is logged in the summary for auditability.

PROTOCOL:
    1. Slice X_K20 = first 20 cols of X_117_unb / X_117_te (standard contract).
    2. Anchor: chemprop_aux (PRE-unblind, verified clean).
       Residual target = y_unb - anchor_unb.
    3. Detect LogP-proxy column inside K=20 by argmax Pearson(col, y_unb).
       Build monotonic_cst vector of length 20, +1 at proxy idx, 0 elsewhere.
    4. Model: HistGradientBoostingRegressor(
                max_iter=300, max_depth=4, learning_rate=0.05,
                l2_regularization=1.0, monotonic_cst=...,
                random_state=42).
       Fit on residual.
    5. 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}.
    6. Deploy: refit on all 253 -> predict on 513.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2742_histgradboost_monotonic.py
    data/processed/nb2742_summary.json
    data/processed/nb2742_pred_oof.npy   (253,) float32
    data/processed/te_nb2742.npy         (513,) float32
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
from sklearn.ensemble import HistGradientBoostingRegressor

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2742"

# --------------------------------------------------------------------------
# HistGB hyperparameters (per spec)
# --------------------------------------------------------------------------
HGB_MAX_ITER = 300
HGB_MAX_DEPTH = 4
HGB_LEARNING_RATE = 0.05
HGB_L2_REGULARIZATION = 1.0
HGB_RANDOM_STATE = 42

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


def _detect_logp_proxy_col(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Return (col_idx, pearson_r) of the column with max +ve corr with y.

    Falls back to col 0 if every correlation is non-finite (e.g. zero-var col).
    """
    n_cols = X.shape[1]
    rs = np.full(n_cols, -np.inf, dtype=np.float64)
    y_mean = y.mean()
    y_demeaned = y - y_mean
    y_norm = float(np.sqrt(np.sum(y_demeaned ** 2)))
    for j in range(n_cols):
        col = X[:, j].astype(np.float64)
        col_demeaned = col - col.mean()
        col_norm = float(np.sqrt(np.sum(col_demeaned ** 2)))
        denom = col_norm * y_norm
        if denom <= 0 or not np.isfinite(denom):
            continue
        r = float(np.sum(col_demeaned * y_demeaned) / denom)
        if np.isfinite(r):
            rs[j] = r
    if not np.any(np.isfinite(rs)):
        return 0, 0.0
    idx = int(np.argmax(rs))
    return idx, float(rs[idx])


def _new_hgb(monotonic_cst, seed: int = HGB_RANDOM_STATE) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=HGB_MAX_ITER,
        max_depth=HGB_MAX_DEPTH,
        learning_rate=HGB_LEARNING_RATE,
        l2_regularization=HGB_L2_REGULARIZATION,
        monotonic_cst=monotonic_cst,
        random_state=int(seed),
    )


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HistGradientBoostingRegressor + monotonic_cst on K=20 "
          f"chemprop_aux residual")
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
    # Sanitize NaN/Inf carried in cache
    X_unb = np.where(np.isfinite(X_unb), X_unb, 0.0).astype(np.float32)
    X_te = np.where(np.isfinite(X_te), X_te, 0.0).astype(np.float32)
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
          f"(PRE-clean PRIMARY-1 baseline)")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- LogP-proxy detection inside K=20 ----
    logp_idx, logp_r = _detect_logp_proxy_col(X_unb, y_unb)
    monotonic_cst = [0] * K_SLICE
    monotonic_cst[logp_idx] = 1
    print(f"[mono] LogP-proxy col within K=20 -> idx={logp_idx} "
          f"(Pearson r={logp_r:+.4f} vs y_unb)")
    print(f"[mono] monotonic_cst vector (len={K_SLICE}): "
          f"{monotonic_cst}")

    # ---- Scaffold 5-fold CV across kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"HistGB: max_iter={HGB_MAX_ITER}  max_depth={HGB_MAX_DEPTH}  "
          f"lr={HGB_LEARNING_RATE}  l2={HGB_L2_REGULARIZATION}  "
          f"seed={HGB_RANDOM_STATE}  monotonic_idx={logp_idx}")
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        fold_info = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            mdl = _new_hgb(
                monotonic_cst=monotonic_cst,
                seed=HGB_RANDOM_STATE + kf_seed + fi,
            )
            mdl.fit(X_unb[tr_loc], resid_unb[tr_loc])
            oof_resid[va_loc] = mdl.predict(X_unb[va_loc])
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(oof_resid[va_loc].mean()),
                "resid_pred_std": float(oof_resid[va_loc].std()),
            })
        assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
        oof_pred = anchor_unb + oof_resid
        # gentle clip to a sane pEC50 range
        oof_pred = np.clip(oof_pred, 3.0, 8.0)
        pooled = float(rae(y_unb, oof_pred))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "folds": fold_info,
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(oof_pred)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  wall={time.time()-ts:.1f}s")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] pooled RAE mean across seeds = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs      = {final_oof_rae:.4f}")
    print(f"[cv] delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")

    # ---- Deploy: refit on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit on all 253 unblind -> apply to 513")
    print("-" * 78)
    deploy_mdl = _new_hgb(monotonic_cst=monotonic_cst, seed=HGB_RANDOM_STATE)
    deploy_mdl.fit(X_unb, resid_unb)
    deploy_resid_te = deploy_mdl.predict(X_te)
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")

    # ---- Gate ----
    if pooled_rae_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {pooled_rae_mean:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "sklearn.ensemble.HistGradientBoostingRegressor with monotonic "
            "constraint +1 on a LogP-proxy column (detected by argmax "
            "Pearson(col, y_unb) within K=20 first-20 of X_117) and 0 "
            "elsewhere; fit on chemprop_aux residual."
        ),
        "paradigm": (
            "histogram_gbdt_sklearn_native_with_monotonic_constraint_logp_"
            "proxy_pos1_pxr_biology_lipophilic_pocket"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "hgb_max_iter": HGB_MAX_ITER,
        "hgb_max_depth": HGB_MAX_DEPTH,
        "hgb_learning_rate": HGB_LEARNING_RATE,
        "hgb_l2_regularization": HGB_L2_REGULARIZATION,
        "hgb_random_state": HGB_RANDOM_STATE,
        "logp_proxy_col_in_k20": int(logp_idx),
        "logp_proxy_pearson_r": logp_r,
        "monotonic_cst": list(monotonic_cst),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "per_seed": per_seed,
        "mean_rae": pooled_rae_mean,
        "pooled_rae_std_seeds": pooled_rae_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "delta_vs_anchor": pooled_rae_mean - rae_anchor_unb,
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
    print(f"   mean_rae (5 seeds)            = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"   delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")
    print(f"   logp_proxy_col / r            = {logp_idx} / {logp_r:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "rae_anchor_unb",
        "delta_vs_anchor",
        "logp_proxy_col_in_k20",
        "logp_proxy_pearson_r",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
