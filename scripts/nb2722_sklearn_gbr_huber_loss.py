"""nb2722 -- sklearn GradientBoostingRegressor with Huber loss on K=20 substrate.

NEW PARADIGM (vs LGBM default + vs sklearn GBR default least_squares):
    The K=20 chemprop_aux residual on n=253 has a heavy-tailed distribution
    (failure-mode tail of novel-scaffold rows where chemprop_aux severely
    under/over-predicts).  Least-squares regression -- the default for both
    LGBM `objective='regression'` and sklearn GBR -- weights residuals by
    r^2, so the gradient signal is dominated by the worst few rows.  On
    n=253 with a sparse failure tail this leads to over-fitting the tail
    and compressing predictions toward the mean for the rest of the rows.

    Huber loss is least-squares inside |r| < alpha*sigma and linear (L1)
    outside.  This bounds the per-row gradient magnitude so a few extreme
    residuals can no longer dominate the boosting updates.  The Huber
    quantile parameter alpha=0.9 says: 90% of training residuals receive
    L2 treatment, the worst 10% are clipped to L1.

    sklearn GradientBoostingRegressor differs from LGBM at the tree-construction
    layer too -- sklearn GBR uses true greedy CART splits (no histogram
    binning), so on small n=253 it is sometimes more conservative on
    high-cardinality continuous splits.  Combined with Huber loss, this is
    a two-axis paradigm change vs nb2240 (LGBM + L2) and vs default sklearn
    GBR (CART + L2): both the loss and the tree-builder are swapped.

    Hypothesis: if the chemprop_aux residual on 253 is heavy-tailed (which
    cycle-149 wide-seed analysis suggests via the 0.4718-0.4720 ceiling
    convergence -- residual structure is signal-dominated outside the tail),
    Huber loss should yield OOF residual predictions whose RAE on the bulk
    is preserved while the tail is no longer dragging the fit.  Predicted
    deep-30 ceiling band: 0.467-0.472.  This script runs 5-seed only, so
    treat as hypothesis-generation tier.

PROTOCOL:
    1. Load X_117 substrate -> slice K=20 surviving columns from
       nb2240 summary (idx_in_117 + names mirror nb2231 RFE result).
    2. residual = y_unb - chemprop_aux_te[unb_idx]  (only PRE-clean anchor).
    3. sklearn GradientBoostingRegressor(loss='huber', alpha=0.9,
       n_estimators=300, max_depth=4, learning_rate=0.05,
       random_state=42).  K=20 features, residual target.
    4. 5-fold scaffold CV (`scaffold_kfold_indices`), 5 kf_seeds
       {1001..1005}.
    5. Deploy: refit GBR on full 253 per seed -> predict 513 te residual;
       mean-bag aggregate.

GATE (mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2722_sklearn_gbr_huber_loss.py
    data/processed/nb2722_summary.json
    data/processed/nb2722_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2722.npy         (513,) float32 deploy refit
    submissions/nb2722_sklearn_gbr_huber_loss.csv
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
from sklearn.ensemble import GradientBoostingRegressor

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2722"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# sklearn GBR + Huber hyperparams as requested in task spec
GBR_LOSS = "huber"
GBR_ALPHA = 0.9
GBR_N_ESTIMATORS = 300
GBR_MAX_DEPTH = 4
GBR_LEARNING_RATE = 0.05
GBR_RANDOM_STATE = 42  # spec-fixed; we still vary the kf_seed for fold splits

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # LGBM K=20 baseline (L2 loss)


def _gbr_params() -> dict:
    """sklearn GBR hyperparams (Huber loss) as per task spec."""
    return dict(
        loss=GBR_LOSS,
        alpha=GBR_ALPHA,
        n_estimators=GBR_N_ESTIMATORS,
        max_depth=GBR_MAX_DEPTH,
        learning_rate=GBR_LEARNING_RATE,
        random_state=GBR_RANDOM_STATE,
    )


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> np.ndarray:
    """One scaffold-CV pass: fit per-fold sklearn GBR (Huber loss).

    Returns oof_residual_pred (shape n,).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        mdl = GradientBoostingRegressor(**_gbr_params())
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
) -> np.ndarray:
    """Fit sklearn GBR (Huber loss) on full 253; predict 513 residual."""
    mdl = GradientBoostingRegressor(**_gbr_params())
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- sklearn GBR (Huber loss alpha={GBR_ALPHA}) on K=20 substrate "
          f"(chemprop_aux residual)")
    print(f"        n_est={GBR_N_ESTIMATORS}  max_depth={GBR_MAX_DEPTH}  "
          f"lr={GBR_LEARNING_RATE}  random_state={GBR_RANDOM_STATE}")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 LGBM (L2 loss) = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor + scaffolds ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"anchor te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"abs_p90={np.quantile(np.abs(residual), 0.9):.4f}")

    # ---- Load X_117 substrate ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape} expected ({n_unb},117)")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape} expected ({n_test},117)")
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)
    print(f"[feat] X117_unb = {X117_unb.shape}  X117_te = {X117_te.shape}")

    # ---- Slice K=20 columns from nb2240 RFE ----
    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"
    print(f"[K20] loaded {len(k20_idx)} surviving indices from nb2240")

    X_unb = X117_unb[:, k20_idx].astype(np.float32)
    X_te = X117_te[:, k20_idx].astype(np.float32)
    feat_dim = X_unb.shape[1]
    assert feat_dim == 20, f"feat_dim {feat_dim} != 20"
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # ---- Scaffolds ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  dim={feat_dim}")
    print(f"  loss={GBR_LOSS}  alpha={GBR_ALPHA}  n_est={GBR_N_ESTIMATORS}  "
          f"depth={GBR_MAX_DEPTH}  lr={GBR_LEARNING_RATE}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof = _scaffold_cv_one_seed(
            X_unb, residual, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        te_resid = _deploy_te_one_seed(X_unb, residual, X_te)
        per_seed_te_resid[i] = te_resid
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"wall={time.time() - ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_resid = per_seed_oof_resid.mean(axis=0)
    median_bag_resid = np.median(per_seed_oof_resid, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag   RAE = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE = {rae_median_bag:.4f}")
    print(f"[cv] anchor     RAE = {rae_anchor:.4f}  "
          f"(d_mean_bag = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"[cv] reference  nb2240 K=20 LGBM (L2) = "
          f"{NB2240_K20_REF:.4f}  "
          f"(d = {rae_mean_bag - NB2240_K20_REF:+.4f})")

    # ---- Deploy te (mean-bag corrected) ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_resid).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_sklearn_gbr_huber_loss.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_bag_rae        = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "sklearn_GBR_Huber_K20_residual_on_chemprop_aux",
        "rationale": (
            "sklearn GradientBoostingRegressor with Huber loss (alpha=0.9) "
            "on K=20 substrate; loss-axis paradigm change vs LGBM L2 and "
            "vs sklearn GBR default least_squares; bounds gradient magnitude "
            "for tail residuals so boosting updates aren't dominated by "
            "novel-scaffold failure rows"
        ),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "x117_unb_path": str(X117_UNB_PATH),
        "x117_te_path": str(X117_TE_PATH),
        "k20_idx_source": str(NB2240_SUMMARY),
        "k20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "k20_surviving_names": k20_names,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "feat_dim": int(feat_dim),
        "model_class": "sklearn.ensemble.GradientBoostingRegressor",
        "gbr_params": _gbr_params(),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_abs_p90": float(np.quantile(np.abs(residual), 0.9)),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "delta_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_vs_nb2240_K20_lgbm": rae_mean_bag - NB2240_K20_REF,
        "nb2240_K20_lgbm_ref": NB2240_K20_REF,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_lgbm",
        "te_unb_in_sample_rae",
        "residual_abs_p90",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
