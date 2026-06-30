"""nb2794 -- sklearn HistGradientBoostingRegressor with Tweedie/Poisson loss on K=20.

NEW PARADIGM:
    Tweedie loss generalizes Gaussian / Poisson / Gamma into a single
    exponential-dispersion family indexed by the power parameter p:
        p=0  -> Gaussian (squared error)
        p=1  -> Poisson  (variance proportional to mean)
        p=2  -> Gamma    (variance proportional to mean^2)
        1<p<2 -> Compound Poisson-Gamma (mass at 0 + continuous right tail)
    sklearn.ensemble.HistGradientBoostingRegressor does NOT expose a generic
    'tweedie' loss, but its 'poisson' loss IS the Tweedie family at p=1
    (the deviance of a Poisson likelihood equals Tweedie deviance with p=1).
    Poisson loss therefore brings:
      1. A right-skewed, log-link-implied conditional mean (more stable
         tails than squared error on bounded-below non-negative targets).
      2. A variance-mean coupling that the Gaussian-loss LGBM / CatBoost
         zoo on this K=20 substrate has never carried.
    The target must be NON-NEGATIVE for Poisson loss. We therefore fit on
    SHIFTED residual:   z = (y_unb - anchor_unb) - min(y_unb - anchor_unb) + 0.1
    then UN-SHIFT the prediction back to the residual scale before adding
    the anchor to recover pEC50.

PROTOCOL:
    1. Slice X_K20 = first 20 cols of X_117_unb / X_117_te (standard contract).
    2. Anchor: chemprop_aux (PRE-unblind, verified clean).
       Residual target r = y_unb - anchor_unb.
       Shift constant s = -min(r) + 0.1  (ensures z = r + s > 0 always).
       Tweedie / Poisson target z = r + s.
    3. Model: HistGradientBoostingRegressor(
                loss='poisson', max_iter=300, max_depth=4,
                learning_rate=0.05, l2_regularization=1.0, random_state=42).
       Fit on z (shifted residual).
    4. Un-shift at inference: r_hat = z_hat - s, then pred = anchor + r_hat.
    5. 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}.
    6. Deploy: refit on all 253 -> predict on 513 (anchor + un-shifted z_hat).

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2794_histgb_tweedie.py
    data/processed/nb2794_summary.json
    data/processed/nb2794_pred_oof.npy   (253,) float32
    data/processed/te_nb2794.npy         (513,) float32
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

TAG = "nb2794"

# --------------------------------------------------------------------------
# HistGB hyperparameters (per spec)
# --------------------------------------------------------------------------
HGB_LOSS = "poisson"  # Tweedie p=1; sklearn doesn't expose generic 'tweedie'
HGB_MAX_ITER = 300
HGB_MAX_DEPTH = 4
HGB_LEARNING_RATE = 0.05
HGB_L2_REGULARIZATION = 1.0
HGB_RANDOM_STATE = 42

# Shift offset to guarantee strictly positive target for Poisson loss
SHIFT_EPS = 0.1

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


def _new_hgb(seed: int = HGB_RANDOM_STATE) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=HGB_LOSS,
        max_iter=HGB_MAX_ITER,
        max_depth=HGB_MAX_DEPTH,
        learning_rate=HGB_LEARNING_RATE,
        l2_regularization=HGB_L2_REGULARIZATION,
        random_state=int(seed),
    )


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HistGradientBoostingRegressor loss='poisson' (Tweedie p=1)"
          f" on K=20 chemprop_aux residual (shifted to non-negative)")
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

    # ---- Residual target + shift to strictly positive for Poisson loss ----
    resid_unb = y_unb - anchor_unb
    resid_min = float(resid_unb.min())
    shift_const = float(-resid_min + SHIFT_EPS)  # ensures z = r + s > 0
    z_unb = resid_unb + shift_const
    assert (z_unb > 0).all(), f"z_unb has non-positive entries: min={z_unb.min()}"
    print(f"[resid] r mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")
    print(f"[shift] shift_const = -min(r) + eps = {shift_const:.4f}  "
          f"(eps={SHIFT_EPS})")
    print(f"[shift] z=r+s    mean={z_unb.mean():.3f}  std={z_unb.std():.3f}  "
          f"min={z_unb.min():.3f}  max={z_unb.max():.3f}")

    # ---- Scaffold 5-fold CV across kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"HistGB: loss={HGB_LOSS}  max_iter={HGB_MAX_ITER}  "
          f"max_depth={HGB_MAX_DEPTH}  lr={HGB_LEARNING_RATE}  "
          f"l2={HGB_L2_REGULARIZATION}  seed={HGB_RANDOM_STATE}  "
          f"shift={shift_const:.4f}")
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_z = np.full(n_unb, np.nan, dtype=np.float64)
        fold_info = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            mdl = _new_hgb(seed=HGB_RANDOM_STATE + kf_seed + fi)
            mdl.fit(X_unb[tr_loc], z_unb[tr_loc])
            oof_z[va_loc] = mdl.predict(X_unb[va_loc])
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "z_pred_mean": float(oof_z[va_loc].mean()),
                "z_pred_std": float(oof_z[va_loc].std()),
            })
        assert not np.isnan(oof_z).any(), "oof_z has NaN -- fold cover incomplete"
        oof_resid = oof_z - shift_const  # un-shift back to residual scale
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
    deploy_mdl = _new_hgb(seed=HGB_RANDOM_STATE)
    deploy_mdl.fit(X_unb, z_unb)
    deploy_z_te = deploy_mdl.predict(X_te)
    deploy_resid_te = deploy_z_te - shift_const  # un-shift
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
            "sklearn.ensemble.HistGradientBoostingRegressor with loss='poisson'"
            " (Tweedie family at power p=1) on K=20 first-20-cols of X_117 "
            "fit on chemprop_aux residual shifted to strictly positive by "
            "s = -min(resid) + 0.1; un-shifted at inference."
        ),
        "paradigm": (
            "histogram_gbdt_sklearn_native_tweedie_poisson_loss_p1_"
            "compound_poisson_gamma_right_skew_non_negative_shifted_residual"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "hgb_loss": HGB_LOSS,
        "hgb_max_iter": HGB_MAX_ITER,
        "hgb_max_depth": HGB_MAX_DEPTH,
        "hgb_learning_rate": HGB_LEARNING_RATE,
        "hgb_l2_regularization": HGB_L2_REGULARIZATION,
        "hgb_random_state": HGB_RANDOM_STATE,
        "shift_eps": SHIFT_EPS,
        "shift_const": shift_const,
        "resid_min": resid_min,
        "resid_max": float(resid_unb.max()),
        "z_min": float(z_unb.min()),
        "z_max": float(z_unb.max()),
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
    print(f"   shift_const                   = {shift_const:.4f}")
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
        "shift_const",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
