"""nb2221 -- Counter-assay anchor RESCALE: retrain pec50_null with LGBM(MSE)
on combined Morgan+RDKit (2265-dim), then plug into K=28+1 stack.

Cycle 170 finding (nb2152): ChemBERTa-77M-MTR Ridge on pEC50_null gives RAE
0.9440 -- essentially a mean predictor -- so the +1 feature carries near-zero
signal (Pearson(null_hat, y_unb) = 0.13) and K=28+1 LOSES by +0.0064 vs K=28.

Hypothesis: the failure was the predictor, not the axis. A combined Morgan+RDKit
LGBM(MSE) on pEC50_null should hit OOF RAE ~0.40-0.50 (nb1240 memo: 0.4350),
which is a >0.50 RAE improvement over the Ridge. Plug that stronger null_hat
as the +1 feature on top of cached X_unb_28_nb2103.npy, re-run the same
5-seed bag KFold(5) residual cross-fit, and compare against:
    a) K=28 reference        : 0.4737 (nb2103 mean-bag), 0.4698 (median-bag)
    b) nb2171 ladder ceiling : 0.4682 (5-anchor SLSQP+stretch mean-of-seeds)
    c) nb2095 baseline PRE   : 0.4720
Gate margin 0.003.

Additionally Stage 7: weighted-shrink experiment -- higher predicted null
(promiscuity proxy) -> down-weight pEC50 toward population mean. Formula:
    pred_shrunk[i] = mu_pop + (1 - lambda * z_null[i]) * (pred[i] - mu_pop)
where z_null is the per-row standardized null_hat and lambda in a small grid.

Inputs:
    data/processed/X_unb_28_nb2103.npy   (253, 28)  float32
    data/processed/nb1133_chemprop_aux_pred_oof.npy  (253,)
    data/processed/te_chemprop_aux.npy   (513,)
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy

Outputs:
    scripts/nb2221_counter_rescale.py
    data/processed/nb2221_summary.json
    data/processed/nb2221_null_hat_lgbm_unb.npy   (253,) float32
    data/processed/nb2221_null_hat_lgbm_te.npy    (513,) float32
    data/processed/nb2221_K28_plus1_lgbm_oof.npy  (253,) float32

References:
    nb2152 Ridge counter RAE     = 0.9440 (mean predictor)
    nb1240 LGBM combined memo    = 0.4350 OOF
    nb2103 K=28 mean-bag         = 0.4737
    nb2103 K=28 median-bag       = 0.4698
    nb2171 5-anchor stack mean   = 0.4682 (deploy w=0.94 on nb1191)
    nb2095 baseline PRE          = 0.4720
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
import lightgbm as lgb
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko, to_inchikey
from pxr.data import load_test, load_train, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined as featurize_combined
from pxr.featurize import impute
from pxr.paths import DATA_PROCESSED

TAG = "nb2221"

# Hyperparameters
N_FOLDS = 5
KF_SEED_COUNTER = 1001
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
DECISION_MARGIN = 0.003

# References
NB2095_BASELINE_OOF = 0.4720
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
NB2171_CEILING_OOF = 0.4682
NB2152_COUNTER_RIDGE_RAE = 0.9440
CHEMPROP_AUX_REF_UNB = 0.5879

# Shrink experiment
SHRINK_LAMBDAS = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20]


def _lgbm_params_null(seed: int) -> dict:
    """LGBM(MSE) for counter-axis pEC50_null prediction on combined 2265 features.
    Slightly larger capacity than residual-LGBM since the input is 2265-dim and
    the target carries real signal (n_dual ~ 2,600)."""
    return dict(
        objective="regression",
        max_depth=6,
        num_leaves=63,
        n_estimators=500,
        learning_rate=0.05,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _lgbm_params_residual(seed: int) -> dict:
    """LGBM(MSE) for residual stack -- identical to nb2103/nb2152."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                  seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params_residual(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM(MSE) on COUNTER axis (pEC50_null) using combined 2265")
    print(f"        K=28+1 LGBM bag + shrink-by-null sweep")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 0. Test set + unblind
    # ------------------------------------------------------------------
    te = load_test()
    te_names = te["name"].astype(str).values
    te_smiles = te["smiles"].astype(str).values
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    # ------------------------------------------------------------------
    # 1. Build dual-labelled (train + counter inner-join on InChIKey)
    # ------------------------------------------------------------------
    print("\n--- Step 1: dual-labelled set (InChIKey inner-join) ---")
    tr_df = load_train()
    ca_df = load_counter()
    tr_df["ik"] = tr_df["smiles"].apply(to_inchikey)
    ca_df["ik"] = ca_df["smiles"].apply(to_inchikey)
    tr_uniq = (
        tr_df.dropna(subset=["ik", "pec50"])
             .drop_duplicates("ik", keep="first")[["ik", "smiles", "pec50", "name"]]
    )
    ca_uniq = (
        ca_df.dropna(subset=["ik", "pec50"])
             .drop_duplicates("ik", keep="first")[["ik", "pec50"]]
             .rename(columns={"pec50": "pec50_null"})
    )
    dual = (
        pd.merge(tr_uniq, ca_uniq, on="ik", how="inner")
          .dropna(subset=["pec50_null"])
          .reset_index(drop=True)
    )
    n_dual = len(dual)
    y_null = dual["pec50_null"].astype(np.float64).values
    print(f"   dual rows = {n_dual}  "
          f"pec50_null mean={y_null.mean():.3f}  std={y_null.std():.3f}")

    # ------------------------------------------------------------------
    # 2. Featurize -- combined Morgan(2048) + RDKit(~217) = 2265
    # ------------------------------------------------------------------
    print("\n--- Step 2: combined Morgan+RDKit (2265-dim) ---")
    X_dual_raw = featurize_combined(dual["smiles"].tolist()).astype(np.float64)
    X_te_raw = featurize_combined(te_smiles.tolist()).astype(np.float64)
    print(f"   X_dual_raw = {X_dual_raw.shape}  X_te_raw = {X_te_raw.shape}")
    # Impute column-medians on the joint matrix to keep test/train consistent
    n_dual_rows = X_dual_raw.shape[0]
    X_joint = np.vstack([X_dual_raw, X_te_raw])
    X_joint = impute(X_joint)
    X_dual = X_joint[:n_dual_rows]
    X_te = X_joint[n_dual_rows:]
    X_unb = X_te[unb_idx]
    print(f"   after impute   X_dual={X_dual.shape}  "
          f"X_te={X_te.shape}  X_unb={X_unb.shape}")

    # ------------------------------------------------------------------
    # 3. LGBM(MSE) on pEC50_null -- scaffold 5-fold cross-fit
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"LGBM(MSE) COUNTER cross-fit  folds={N_FOLDS}  kf_seed={KF_SEED_COUNTER}")
    print("-" * 78)

    dual_scaf = [bemis_murcko(s) for s in dual["smiles"]]
    splits_dual = scaffold_kfold_indices(
        dual_scaf, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED_COUNTER,
    )

    oof_null = np.full(n_dual, np.nan, dtype=np.float64)
    for fold_i, (tr_loc, va_loc) in enumerate(splits_dual):
        mdl = lgb.LGBMRegressor(**_lgbm_params_null(KF_SEED_COUNTER))
        mdl.fit(X_dual[tr_loc], y_null[tr_loc])
        oof_null[va_loc] = mdl.predict(X_dual[va_loc])
        print(f"   fold {fold_i+1}/{N_FOLDS}  "
              f"|tr|={len(tr_loc)}  |va|={len(va_loc)}  "
              f"va_RAE={rae(y_null[va_loc], oof_null[va_loc]):.4f}")

    counter_oof_rae = float(rae(y_null, oof_null))
    pearson_oof = float(np.corrcoef(y_null, oof_null)[0, 1])
    print(f"\n[counter] LGBM(MSE) OOF RAE = {counter_oof_rae:.4f}  "
          f"(nb2152 Ridge ref {NB2152_COUNTER_RIDGE_RAE:.4f})")
    print(f"[counter] Pearson(y_null, oof_null) = {pearson_oof:+.4f}")

    # ------------------------------------------------------------------
    # 4. Deploy refit on full dual -> null_hat for 253 unb + 513 te
    # ------------------------------------------------------------------
    print("\n--- Step 4: deploy refit (full dual, 5-seed bag) ---")
    null_te_stack = np.zeros((len(RESID_SEEDS), n_te), dtype=np.float64)
    for k, sd in enumerate(RESID_SEEDS):
        m = lgb.LGBMRegressor(**_lgbm_params_null(sd))
        m.fit(X_dual, y_null)
        null_te_stack[k] = m.predict(X_te)
    null_hat_te = null_te_stack.mean(axis=0).astype(np.float32)
    null_hat_unb = null_hat_te[unb_idx]
    np.save(DATA_PROCESSED / f"{TAG}_null_hat_lgbm_unb.npy", null_hat_unb)
    np.save(DATA_PROCESSED / f"{TAG}_null_hat_lgbm_te.npy", null_hat_te)
    print(f"   null_hat_unb  mean={null_hat_unb.mean():.3f}  "
          f"std={null_hat_unb.std():.3f}  "
          f"range=[{null_hat_unb.min():.2f}, {null_hat_unb.max():.2f}]")
    print(f"   null_hat_te   mean={null_hat_te.mean():.3f}  "
          f"std={null_hat_te.std():.3f}")

    # ------------------------------------------------------------------
    # 5. K=28+1 residual cross-fit -- identical protocol to nb2103/nb2152
    # ------------------------------------------------------------------
    print("\n--- Step 5: K=28+1 LGBM bag residual cross-fit on 253 unblind ---")
    X_unb_28 = np.load(DATA_PROCESSED / "X_unb_28_nb2103.npy").astype(np.float32)
    assert X_unb_28.shape == (n_unb, 28), f"X_unb_28 shape {X_unb_28.shape}"
    chemprop_te = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    assert chemprop_te.shape == (n_te,)
    anchor = chemprop_te[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"   chemprop_aux te[unb] RAE = {rae_anchor:.4f}  "
          f"(in-sample; OOF ref {CHEMPROP_AUX_REF_UNB:.4f})")

    X_unb_29 = np.column_stack(
        [X_unb_28, null_hat_unb.astype(np.float32)]
    )
    assert X_unb_29.shape == (n_unb, 29)
    print(f"   X_unb_29 = {X_unb_29.shape}  (K=28 + lgbm_null_hat)")

    print(f"   5-seed bag (seeds={RESID_SEEDS}) KFold({RESID_FOLDS}) cross-fit:")
    oof_stack_28 = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    oof_stack_29 = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    for k, sd in enumerate(RESID_SEEDS):
        oof_stack_28[k] = _residual_cross_fit_one_seed(X_unb_28, residual, sd)
        oof_stack_29[k] = _residual_cross_fit_one_seed(X_unb_29, residual, sd)
    pred_28_stack = anchor[None, :] + oof_stack_28
    pred_29_stack = anchor[None, :] + oof_stack_29

    rae_28_per_seed = [float(rae(y_unb, pred_28_stack[k]))
                       for k in range(len(RESID_SEEDS))]
    rae_29_per_seed = [float(rae(y_unb, pred_29_stack[k]))
                       for k in range(len(RESID_SEEDS))]
    pred_28_mean = pred_28_stack.mean(axis=0)
    pred_29_mean = pred_29_stack.mean(axis=0)
    pred_28_median = np.median(pred_28_stack, axis=0)
    pred_29_median = np.median(pred_29_stack, axis=0)
    rae_28_mean_bag = float(rae(y_unb, pred_28_mean))
    rae_29_mean_bag = float(rae(y_unb, pred_29_mean))
    rae_28_median_bag = float(rae(y_unb, pred_28_median))
    rae_29_median_bag = float(rae(y_unb, pred_29_median))

    print(f"\n   K=28  per_seed_RAE   = {[f'{r:.4f}' for r in rae_28_per_seed]}")
    print(f"   K=28  mean_bag_RAE   = {rae_28_mean_bag:.4f}  "
          f"(nb2103 ref {NB2103_K28_MEAN_BAG_REF:.4f})")
    print(f"   K=28  median_bag_RAE = {rae_28_median_bag:.4f}  "
          f"(nb2103 ref {NB2103_K28_MEDIAN_BAG_REF:.4f})")
    print(f"   K=29  per_seed_RAE   = {[f'{r:.4f}' for r in rae_29_per_seed]}")
    print(f"   K=29  mean_bag_RAE   = {rae_29_mean_bag:.4f}")
    print(f"   K=29  median_bag_RAE = {rae_29_median_bag:.4f}")

    delta_mean_bag = rae_29_mean_bag - rae_28_mean_bag
    delta_median_bag = rae_29_median_bag - rae_28_median_bag
    delta_vs_nb2095 = rae_29_mean_bag - NB2095_BASELINE_OOF
    delta_vs_nb2171 = rae_29_mean_bag - NB2171_CEILING_OOF
    print(f"\n   delta(K=29 - K=28)  mean_bag   = {delta_mean_bag:+.4f}")
    print(f"   delta(K=29 - K=28)  median_bag = {delta_median_bag:+.4f}")
    print(f"   delta(K=29 - nb2095_baseline) = {delta_vs_nb2095:+.4f}")
    print(f"   delta(K=29 - nb2171_ceiling)  = {delta_vs_nb2171:+.4f}  "
          f"(nb2171={NB2171_CEILING_OOF:.4f})")

    np.save(DATA_PROCESSED / f"{TAG}_K28_plus1_lgbm_oof.npy",
            pred_29_mean.astype(np.float32))

    # ------------------------------------------------------------------
    # 6. Shrink experiment -- high null = high promiscuity = down-weight pEC50
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STAGE 7: shrink-by-null sweep (high null_hat -> shrink toward mu_pop)")
    print("-" * 78)
    # Use the K=29 mean-bag prediction as the candidate to shrink
    base_pred = pred_29_mean.copy()
    mu_pop = float(base_pred.mean())
    # Standardize null_hat on the 253 unblind, clip to [-3, 3]
    z_null = (null_hat_unb - null_hat_unb.mean()) / (null_hat_unb.std() + 1e-9)
    z_null = np.clip(z_null, -3.0, 3.0)

    shrink_results = []
    for lam in SHRINK_LAMBDAS:
        # shrink_factor in [-2, 2] -> blend in [-1, 3] roughly; we clip to [0.2, 1.5]
        shrink_factor = np.clip(1.0 - lam * z_null, 0.2, 1.5)
        pred_shrunk = mu_pop + shrink_factor * (base_pred - mu_pop)
        r = float(rae(y_unb, pred_shrunk))
        shrink_results.append({
            "lambda": float(lam),
            "rae": r,
            "delta_vs_base": float(r - rae_29_mean_bag),
            "mean_shrink_factor": float(shrink_factor.mean()),
            "std_shrink_factor": float(shrink_factor.std()),
        })
        print(f"   lambda={lam:.3f}  shrink_factor_mean={shrink_factor.mean():.3f}  "
              f"RAE={r:.4f}  delta={r - rae_29_mean_bag:+.4f}")
    best_shrink = min(shrink_results, key=lambda d: d["rae"])
    print(f"\n[shrink] best lambda = {best_shrink['lambda']:.3f}  "
          f"RAE = {best_shrink['rae']:.4f}  "
          f"delta vs base = {best_shrink['delta_vs_base']:+.4f}")

    # ------------------------------------------------------------------
    # 7. Gates: K=29 must beat K=28 by margin AND <= nb2171 ceiling
    # ------------------------------------------------------------------
    gate_beats_K28 = delta_mean_bag < -DECISION_MARGIN
    gate_beats_nb2171 = rae_29_mean_bag <= NB2171_CEILING_OOF - DECISION_MARGIN
    gate_beats_nb2095 = rae_29_mean_bag <= NB2095_BASELINE_OOF
    shrink_helps = best_shrink["delta_vs_base"] < -DECISION_MARGIN
    overall_gate = gate_beats_K28 and gate_beats_nb2171
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   K=29 beats K=28 by margin (delta < -{DECISION_MARGIN:.3f})  "
          f"-> {'PASS' if gate_beats_K28 else 'FAIL'}")
    print(f"   K=29 <= nb2171_ceiling - margin ({NB2171_CEILING_OOF - DECISION_MARGIN:.4f}) "
          f"-> {'PASS' if gate_beats_nb2171 else 'FAIL'}")
    print(f"   K=29 <= nb2095 baseline ({NB2095_BASELINE_OOF:.4f})    "
          f"-> {'PASS' if gate_beats_nb2095 else 'FAIL'}")
    print(f"   shrink lambda helps (delta < -{DECISION_MARGIN:.3f})       "
          f"-> {'PASS' if shrink_helps else 'FAIL'}")
    print(f"   overall (K28 + nb2171)                                "
          f"-> {'PASS' if overall_gate else 'FAIL'}")

    # ------------------------------------------------------------------
    # 8. Diagnostics
    # ------------------------------------------------------------------
    pearson_null_vs_y = float(np.corrcoef(null_hat_unb, y_unb)[0, 1])
    pearson_null_vs_anchor = float(np.corrcoef(null_hat_unb, anchor)[0, 1])
    pearson_null_vs_resid = float(np.corrcoef(null_hat_unb, residual)[0, 1])
    print("\n--- Diagnostics on 253 unblind ---")
    print(f"   Pearson(null_hat_unb, y_unb)         = {pearson_null_vs_y:+.4f}  "
          f"(nb2152: +0.1263)")
    print(f"   Pearson(null_hat_unb, chemprop_anch) = {pearson_null_vs_anchor:+.4f}  "
          f"(nb2152: +0.1361)")
    print(f"   Pearson(null_hat_unb, resid)         = {pearson_null_vs_resid:+.4f}  "
          f"(nb2152: +0.0629)")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    summary = {
        "tag": TAG,
        "method": "lgbm_mse_combined_2265_on_pec50_null_then_K28_plus1_LGBM_bag_plus_shrink",
        "cycle_hypothesis": "Cycle 170 nb2152 failure was Ridge predictor, not axis. "
                            "LGBM(MSE) on combined 2265 should beat 0.94 RAE "
                            "(nb1240 memo: ~0.4350).",
        "n_dual": int(n_dual),
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "kf_seed_counter": KF_SEED_COUNTER,
        "counter_oof_rae_lgbm": float(counter_oof_rae),
        "counter_oof_pearson": float(pearson_oof),
        "counter_oof_rae_ridge_nb2152": float(NB2152_COUNTER_RIDGE_RAE),
        "counter_oof_rae_delta_vs_ridge": float(counter_oof_rae - NB2152_COUNTER_RIDGE_RAE),
        "null_hat_unb_mean": float(null_hat_unb.mean()),
        "null_hat_unb_std": float(null_hat_unb.std()),
        "null_hat_te_mean": float(null_hat_te.mean()),
        "null_hat_te_std": float(null_hat_te.std()),
        "chemprop_aux_te_unb_in_rae": float(rae_anchor),
        "K28_per_seed_rae": rae_28_per_seed,
        "K28_mean_bag_rae": rae_28_mean_bag,
        "K28_median_bag_rae": rae_28_median_bag,
        "K29_per_seed_rae": rae_29_per_seed,
        "K29_mean_bag_rae": rae_29_mean_bag,
        "K29_median_bag_rae": rae_29_median_bag,
        "delta_K29_minus_K28_mean_bag": float(delta_mean_bag),
        "delta_K29_minus_K28_median_bag": float(delta_median_bag),
        "delta_K29_vs_nb2095_baseline": float(delta_vs_nb2095),
        "delta_K29_vs_nb2171_ceiling": float(delta_vs_nb2171),
        "nb2095_baseline_oof": NB2095_BASELINE_OOF,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "nb2171_ceiling_oof": NB2171_CEILING_OOF,
        "decision_margin": DECISION_MARGIN,
        "gate_beats_K28": bool(gate_beats_K28),
        "gate_beats_nb2095": bool(gate_beats_nb2095),
        "gate_beats_nb2171": bool(gate_beats_nb2171),
        "gate_overall": bool(overall_gate),
        "shrink_lambdas": SHRINK_LAMBDAS,
        "shrink_results": shrink_results,
        "shrink_best_lambda": float(best_shrink["lambda"]),
        "shrink_best_rae": float(best_shrink["rae"]),
        "shrink_best_delta_vs_base": float(best_shrink["delta_vs_base"]),
        "shrink_helps_gate": bool(shrink_helps),
        "diagnostics_pearson_null_vs_y": pearson_null_vs_y,
        "diagnostics_pearson_null_vs_anchor": pearson_null_vs_anchor,
        "diagnostics_pearson_null_vs_resid": pearson_null_vs_resid,
        "null_hat_unb_path": str(DATA_PROCESSED / f"{TAG}_null_hat_lgbm_unb.npy"),
        "null_hat_te_path": str(DATA_PROCESSED / f"{TAG}_null_hat_lgbm_te.npy"),
        "K28_plus1_oof_path": str(DATA_PROCESSED / f"{TAG}_K28_plus1_lgbm_oof.npy"),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   counter LGBM OOF RAE = {counter_oof_rae:.4f}  "
          f"(Ridge ref {NB2152_COUNTER_RIDGE_RAE:.4f}  "
          f"delta {counter_oof_rae - NB2152_COUNTER_RIDGE_RAE:+.4f})")
    print(f"   K=28 mean_bag        = {rae_28_mean_bag:.4f}  "
          f"(nb2103 ref {NB2103_K28_MEAN_BAG_REF:.4f})")
    print(f"   K=28+1 mean_bag      = {rae_29_mean_bag:.4f}  "
          f"(delta {delta_mean_bag:+.4f})")
    print(f"   nb2171 ceiling       = {NB2171_CEILING_OOF:.4f}  "
          f"(delta {delta_vs_nb2171:+.4f})")
    print(f"   best shrink lambda   = {best_shrink['lambda']:.3f}  "
          f"RAE = {best_shrink['rae']:.4f}")
    print(f"   gates: K28={gate_beats_K28}  nb2171={gate_beats_nb2171}  "
          f"overall={overall_gate}")
    print(f"   wall                 = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "counter_oof_rae_lgbm",
        "counter_oof_rae_delta_vs_ridge",
        "K28_mean_bag_rae",
        "K29_mean_bag_rae",
        "delta_K29_minus_K28_mean_bag",
        "delta_K29_vs_nb2171_ceiling",
        "shrink_best_lambda",
        "shrink_best_rae",
        "gate_overall",
    ):
        print(f"  {k}: {res.get(k)}")
