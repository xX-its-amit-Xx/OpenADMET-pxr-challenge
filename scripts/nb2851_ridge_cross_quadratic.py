"""nb2851 -- Ridge with cross-quadratic features on TOP-5 K=20 chemprop_aux residual.

NEW PARADIGM (vs nb2740 230-feature blowup):
    Ridge on top-5 K=20 features ranked by SHAP importance + their pairwise
    cross-products + squares only.  Total feature count = 20 (matches nb2740
    raw-K=20 count) but with INTERACTION STRUCTURE on the 5 highest-signal
    features instead of raw 20 with NO interactions.

    nb2740 (230 features) blew up degrees-of-freedom past n=253 -> Ridge
    alpha=1.0 over-shrinks the genuine top interactions while still wasting
    capacity on 195 low-SHAP noise terms.  This script restricts the
    interaction block to the SHAP top-5 only:

        5 raw + 5 squares + C(5, 2)=10 cross-products = 20 features

    Same downstream Ridge(alpha=1.0) as nb2740.  Same Scaler -> Poly -> Ridge
    pipeline.  Same fold/seed protocol.  ONLY difference: top-5-by-SHAP
    pre-selection of the polynomial substrate.

TOP-5 BY SHAP (from nb2263_summary.json shap_top28_idx_in_117 intersected with
nb2231_summary.json snapshots.20.surviving_idx_in_117):

    Position in K=20 (nb2240 slice order)  idx_in_117  name
    --------------------------------------  ---------- ---------------------
     0                                       45         Mordred_col_292
     1                                       67         ChempropEmbed_dim_14
     2                                       66         ChempropEmbed_dim_9
     3                                       68         ChempropEmbed_dim_32
     4                                       65         ChempropEmbed_dim_259

PROTOCOL:
    1. Load X_117_unb / X_117_te.  Slice to K=20 (nb2240 RFE survivors).
    2. Slice to TOP-5 by SHAP ranking (first 5 cols of nb2240's K=20).
    3. Anchor: chemprop_aux (PRE-unblind, verified clean).  Residual target
       = y_unb - anchor_unb.
    4. Per fold: StandardScaler.fit on train -> transform val.
       PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
       -> 5 raw + 5 squares + 10 cross = 20 features.
       Ridge(alpha=1.0, random_state=42) fit on poly features.
    5. 5-fold scaffold CV on 253 unblind, 5 kf_seeds {1001..1005}.
    6. Deploy: refit scaler + poly + Ridge on ALL 253 -> predict on 513.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2851_ridge_cross_quadratic.py
    data/processed/nb2851_summary.json
    data/processed/nb2851_pred_oof.npy   (253,) float32
    data/processed/te_nb2851.npy         (513,) float32
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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2851"

# --------------------------------------------------------------------------
# Ridge / Polynomial hyperparameters (match nb2740 spec for apples-to-apples)
# --------------------------------------------------------------------------
POLY_DEGREE = 2
POLY_INTERACTION_ONLY = False
POLY_INCLUDE_BIAS = False
RIDGE_ALPHA = 1.0
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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"
NB2263_SUMMARY = DATA_PROCESSED / "nb2263_summary.json"

TOP_N = 5  # top-5 by SHAP


def _build_poly_pipeline(X_train, X_val=None):
    """Fit StandardScaler then PolynomialFeatures on TRAIN; transform both."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    poly = PolynomialFeatures(
        degree=POLY_DEGREE,
        interaction_only=POLY_INTERACTION_ONLY,
        include_bias=POLY_INCLUDE_BIAS,
    )
    X_train_p = poly.fit_transform(X_train_s)
    X_val_p = None
    if X_val is not None:
        X_val_s = scaler.transform(X_val)
        X_val_p = poly.transform(X_val_s)
    return X_train_p, X_val_p, scaler, poly


def _select_top5_by_shap_in_K20():
    """Resolve TOP-5 K=20 features by SHAP importance.

    Source of truth:
      - nb2263_summary.json `shap_top28_idx_in_117` is the SHAP-ranked list
        (descending) of 28 columns in the 117-col pyramid matrix.
      - nb2231_summary.json `snapshots.20.surviving_idx_in_117` is the K=20
        RFE-greedy-backward survivor set (used by nb2240).

    Top-5 by SHAP within K=20 = the first 5 entries of the SHAP-28 list that
    also appear in the K=20 survivor set.

    Returns:
      top5_cols_in_117  (5,) ints -- column indices in the 117-col matrix
      top5_cols_in_K20  (5,) ints -- column indices in the K=20 slice
      top5_names        (5,) strs -- feature names
    """
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    with open(NB2263_SUMMARY) as f:
        nb2263 = json.load(f)

    k20_idx_in_117 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    k20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(k20_idx_in_117) == 20, (
        f"expected 20 K=20 survivors, got {len(k20_idx_in_117)}"
    )
    name_by_117 = dict(zip(k20_idx_in_117, k20_names))

    shap_28_in_117 = list(nb2263["shap_top28_idx_in_117"])
    k20_set = set(k20_idx_in_117)
    top5_in_117 = [j for j in shap_28_in_117 if j in k20_set][:TOP_N]
    assert len(top5_in_117) == TOP_N, (
        f"only {len(top5_in_117)} of the SHAP-28 survived K=20 RFE; need {TOP_N}"
    )
    top5_in_K20 = [k20_idx_in_117.index(j) for j in top5_in_117]
    top5_names = [name_by_117[j] for j in top5_in_117]
    return top5_in_117, top5_in_K20, top5_names, k20_idx_in_117


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Ridge cross-quadratic on TOP-5 K=20 chemprop_aux residual")
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

    # ---- Resolve TOP-5 K=20 by SHAP ----
    top5_in_117, top5_in_K20, top5_names, k20_in_117 = _select_top5_by_shap_in_K20()
    print(f"[shap] TOP-5 K=20 (ranked by SHAP from nb2263):")
    for r, (j117, jK20, nm) in enumerate(zip(top5_in_117, top5_in_K20, top5_names)):
        print(f"   rank={r}  idx_in_117={j117:3d}  K20_col={jK20:2d}  {nm}")

    # ---- Load X_117 -> slice K=20 -> slice top-5 ----
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    assert X_unb_117.shape == (n_unb, 117), f"X_unb shape {X_unb_117.shape}"
    assert X_te_117.shape == (n_test, 117), f"X_te shape {X_te_117.shape}"
    # direct slice in 117-space (avoids needing the K20 cache file)
    top5_cols = np.asarray(top5_in_117, dtype=int)
    X_unb = X_unb_117[:, top5_cols].astype(np.float32)
    X_te = X_te_117[:, top5_cols].astype(np.float32)
    print(f"[feat] X_unb_top5={X_unb.shape}  X_te_top5={X_te.shape}")

    # ---- Sanity-check polynomial feature count ----
    _probe_poly = PolynomialFeatures(
        degree=POLY_DEGREE,
        interaction_only=POLY_INTERACTION_ONLY,
        include_bias=POLY_INCLUDE_BIAS,
    )
    _probe_poly.fit(X_unb[:2])
    n_poly_features = _probe_poly.transform(X_unb[:2]).shape[1]
    print(f"[poly] degree={POLY_DEGREE}  interaction_only={POLY_INTERACTION_ONLY}  "
          f"include_bias={POLY_INCLUDE_BIAS}  -> n_features={n_poly_features}")
    # 5 raw + 5 squares + C(5, 2)=10 cross = 20
    assert n_poly_features == 20, f"expected 20 poly features, got {n_poly_features}"

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

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"Ridge: alpha={RIDGE_ALPHA}  random_state={RIDGE_RANDOM_STATE}\n"
          f"Poly:  degree={POLY_DEGREE}  -> n_features={n_poly_features}\n"
          f"Scaler -> Poly -> Ridge (per-fold fit on train slice only)")
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
            X_tr_p, X_va_p, _, _ = _build_poly_pipeline(
                X_unb[tr_loc], X_unb[va_loc]
            )
            mdl = Ridge(alpha=RIDGE_ALPHA, random_state=RIDGE_RANDOM_STATE)
            mdl.fit(X_tr_p, resid_unb[tr_loc])
            oof_resid[va_loc] = mdl.predict(X_va_p)
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(oof_resid[va_loc].mean()),
                "resid_pred_std": float(oof_resid[va_loc].std()),
            })
        assert not np.isnan(oof_resid).any(), (
            "oof_resid has NaN -- fold cover incomplete"
        )
        oof_pred = anchor_unb + oof_resid
        # gentle clip to a sane pEC50 range (mirror nb2740)
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

    # ---- Deploy: refit scaler + poly + Ridge on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit scaler+poly+Ridge on all 253 unblind -> apply to 513")
    print("-" * 78)
    scaler_full = StandardScaler()
    X_unb_s = scaler_full.fit_transform(X_unb)
    X_te_s = scaler_full.transform(X_te)
    poly_full = PolynomialFeatures(
        degree=POLY_DEGREE,
        interaction_only=POLY_INTERACTION_ONLY,
        include_bias=POLY_INCLUDE_BIAS,
    )
    X_unb_p = poly_full.fit_transform(X_unb_s)
    X_te_p = poly_full.transform(X_te_s)
    mdl_full = Ridge(alpha=RIDGE_ALPHA, random_state=RIDGE_RANDOM_STATE)
    mdl_full.fit(X_unb_p, resid_unb)
    deploy_resid_te = mdl_full.predict(X_te_p)
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
            "Ridge cross-quadratic on TOP-5 K=20 chemprop_aux residual. "
            "PolynomialFeatures(degree=2, interaction_only=False, "
            "include_bias=False) on 5 SHAP-top features -> 5 raw + 5 squares + "
            "10 cross-products = 20 features. Per-fold StandardScaler -> "
            f"PolynomialFeatures -> Ridge(alpha={RIDGE_ALPHA}). 5-fold scaffold "
            f"CV on n=253 across {len(KF_SEEDS)} kf_seeds."
        ),
        "paradigm": (
            "linear_top5_quadratic_interactions_ridge_"
            "distinct_from_nb2740_K20_blowup_and_tree_axis_aligned_and_kernel_radial"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "top5_shap_in_117": [int(j) for j in top5_in_117],
        "top5_shap_in_K20": [int(j) for j in top5_in_K20],
        "top5_names": top5_names,
        "k20_idx_in_117": [int(j) for j in k20_in_117],
        "shap_source_nb": "nb2263_summary.json:shap_top28_idx_in_117",
        "k20_source_nb": "nb2231_summary.json:snapshots.20.surviving_idx_in_117",
        "poly_degree": POLY_DEGREE,
        "poly_interaction_only": POLY_INTERACTION_ONLY,
        "poly_include_bias": POLY_INCLUDE_BIAS,
        "n_poly_features": int(n_poly_features),
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_random_state": RIDGE_RANDOM_STATE,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim_raw": int(X_unb.shape[1]),
        "top_n_from_K20_by_shap": TOP_N,
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
    print(f"   anchor (chemprop_aux) in_RAE   = {rae_anchor_unb:.4f}")
    print(f"   poly degree                    = {POLY_DEGREE}  -> {n_poly_features} feat")
    print(f"   ridge alpha                    = {RIDGE_ALPHA}")
    print(f"   mean_rae (5 kf_seeds)          = {pooled_rae_mean:.4f} "
          f"+/- {pooled_rae_std:.4f}")
    print(f"   rae_of_mean_oof                = {final_oof_rae:.4f}")
    print(f"   delta vs anchor                = "
          f"{pooled_rae_mean - rae_anchor_unb:+.4f}")
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
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "delta_vs_anchor",
        "te_unb_rae_in_sample",
        "verdict",
        "te_deploy_mean",
        "te_deploy_std",
    ):
        print(f"  {k}: {res.get(k)}")
