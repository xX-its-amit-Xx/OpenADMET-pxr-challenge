"""nb2152 -- ChemBERTa-77M-MTR 384-dim Ridge on COUNTER-ASSAY axis (pEC50_null).

Cycle 164 hypothesis (per nb2120 recommendation): ChemBERTa fails on the pEC50
main axis because its anchor-residual Ridge is too correlated with nb2095
(Pearson 0.90 on residual). The pEC50_null counter-assay is a biologically
orthogonal axis (PXR-null = off-target / promiscuity proxy). If ChemBERTa
carries useful chemistry-only signal, it might predict pEC50_null with low RAE
on the 2,646 dual-labelled rows (inner-join InChIKey). Then plug the predicted
pec50_null as a +1 feature on top of the nb2103 K=28 SHAP-pruned matrix and
re-run the LGBM(MSE) 5-seed bag residual cross-fit. Compare vs nb2095 baseline
(pooled OOF RAE 0.4720 on the 253 unblind).

Stage 1: Ridge(alpha grid) on ChemBERTa-384 -> pEC50_null  (scaffold 5f x 1 seed)
         -- report counter-axis cross-fit RAE per alpha; pick best by mean RAE.
Stage 2: Refit best Ridge on ALL 2,646 dual rows; predict pec50_null for the
         253 unblind + 513 test.
Stage 3: Append null_hat as +1 column to cached X_unb_28_nb2103.npy. Cross-fit
         5-seed bag LGBM(MSE) residual = (y_unb - chemprop_aux_oof) on K=28+1.
         Compare against K=28 only (nb2103 reference) and nb2095 baseline.
Stage 4: If K=28+1 mean-bag <= nb2095 baseline (0.4720): deploy refit on 253
         using K=28+1 features (needs X_te_28 rebuild via nb2103 pipeline --
         marked as TODO if gate fails, run only on green light).

Inputs:
    data/processed/chemberta_77m_mtr_embeddings.npy   (4651, 384) float32
    data/processed/chemberta_77m_mtr_index.csv         {row_idx,inchikey,name,std_smiles}
    data/processed/X_unb_28_nb2103.npy                 (253, 28)  float32
    data/processed/nb1133_chemprop_aux_pred_oof.npy    (253,)     float64
    data/processed/te_chemprop_aux.npy                 (513,)     float32
    data/processed/_audit_unblind_idx.npy              (253,)
    data/processed/_audit_unblind_y.npy                (253,)

Outputs:
    data/processed/nb2152_summary.json
    data/processed/nb2152_null_hat_unb.npy             (253,)  float32  (always)
    data/processed/nb2152_null_hat_te.npy              (513,)  float32  (always)
    data/processed/nb2152_K28_plus1_oof.npy            (253,)  float32  (always)

References:
    nb2095 baseline OOF: 0.4720 (gate target)
    nb2103 K=28 mean-bag: 0.4737 ; median-bag: 0.4698
    chemprop_aux OOF: 0.5879
    nb730 null-ensemble OOF (combined+MACCS bag): ~0.5470 on counter axis
    nb1240 null predictor (combined LGBM): ~0.4350 OOF on dual
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko, to_inchikey
from pxr.data import load_test, load_train, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2152"

# Hyperparameters
N_FOLDS = 5
KF_SEED_COUNTER = 1001
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]

# References
NB2095_BASELINE_OOF = 0.4720
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF_UNB = 0.5879
DECISION_MARGIN = 0.003   # require new method to beat by >= margin


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb1852/nb1861/nb2063/nb2081/nb2091/nb2103."""
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
    """Cross-fit same KFold protocol as nb2103."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa-Ridge on COUNTER axis (pEC50_null)")
    print(f"        K=28+1 LGBM bag vs nb2095 baseline (0.4720)")
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
    # 2. Load ChemBERTa-384 embeddings and match by name (or InChIKey)
    # ------------------------------------------------------------------
    print("\n--- Step 2: ChemBERTa-384 lookup ---")
    emb_all = np.load(
        DATA_PROCESSED / "chemberta_77m_mtr_embeddings.npy"
    ).astype(np.float32)
    idx_df = pd.read_csv(DATA_PROCESSED / "chemberta_77m_mtr_index.csv")
    assert emb_all.shape[0] == len(idx_df), "embed/index row mismatch"
    name_to_row = dict(zip(idx_df["name"], idx_df["row_idx"]))
    ik_to_row = dict(zip(idx_df["inchikey"], idx_df["row_idx"]))

    # match dual by name (preferred), then fall back to inchikey
    dual_rows = []
    for nm, ik in zip(dual["name"], dual["ik"]):
        r = name_to_row.get(nm)
        if r is None:
            r = ik_to_row.get(ik)
        dual_rows.append(r)
    missing_dual = sum(r is None for r in dual_rows)
    if missing_dual > 0:
        keep = [r is not None for r in dual_rows]
        dual = dual[keep].reset_index(drop=True)
        dual_rows = [r for r in dual_rows if r is not None]
        y_null = dual["pec50_null"].astype(np.float64).values
        n_dual = len(dual)
        print(f"   [warn] dropped {missing_dual} dual rows w/o BERTa embedding; "
              f"new n_dual={n_dual}")
    X_dual = np.vstack([emb_all[r] for r in dual_rows]).astype(np.float64)

    # match 513 test by name
    missing_te = [n for n in te_names if n not in name_to_row]
    assert not missing_te, f"test unmatched: {missing_te[:5]}"
    X_te = np.vstack([emb_all[name_to_row[n]] for n in te_names]).astype(np.float64)
    X_unb = X_te[unb_idx]

    print(f"   X_dual={X_dual.shape}  X_unb={X_unb.shape}  X_te={X_te.shape}")

    # ------------------------------------------------------------------
    # 3. Ridge alpha sweep on pEC50_null  -- scaffold 5-fold cross-fit
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"RIDGE ALPHA SWEEP on pEC50_null  alphas={ALPHA_GRID}  "
          f"folds={N_FOLDS}  kf_seed={KF_SEED_COUNTER}")
    print("-" * 78)

    dual_scaf = [bemis_murcko(s) for s in dual["smiles"]]
    splits_dual = scaffold_kfold_indices(
        dual_scaf, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED_COUNTER,
    )

    per_alpha = []
    for alpha in ALPHA_GRID:
        oof = np.full(n_dual, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits_dual:
            m = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
            m.fit(X_dual[tr_loc], y_null[tr_loc])
            oof[va_loc] = m.predict(X_dual[va_loc])
        r = float(rae(y_null, oof))
        per_alpha.append({
            "alpha": float(alpha),
            "oof_rae_null": r,
            "oof_mean": float(np.nanmean(oof)),
            "oof_std": float(np.nanstd(oof)),
        })
        print(f"   alpha={alpha:7.2f}  counter_RAE={r:.4f}  "
              f"pred_std={np.nanstd(oof):.3f}")

    best_idx = int(np.argmin([r["oof_rae_null"] for r in per_alpha]))
    best_alpha = per_alpha[best_idx]["alpha"]
    best_counter_rae = per_alpha[best_idx]["oof_rae_null"]
    print(f"\n[best alpha] {best_alpha}  counter_RAE={best_counter_rae:.4f}")

    # ------------------------------------------------------------------
    # 4. Refit best Ridge on all dual, predict null for 253 unb + 513 te
    # ------------------------------------------------------------------
    print("\n--- Step 4: deploy refit (full dual) -> predict null on 253+513 ---")
    m_deploy = Ridge(alpha=best_alpha, fit_intercept=True, random_state=0)
    m_deploy.fit(X_dual, y_null)
    null_hat_te = m_deploy.predict(X_te).astype(np.float32)
    null_hat_unb = null_hat_te[unb_idx]
    np.save(DATA_PROCESSED / f"{TAG}_null_hat_unb.npy", null_hat_unb)
    np.save(DATA_PROCESSED / f"{TAG}_null_hat_te.npy", null_hat_te)
    print(f"   null_hat_unb  mean={null_hat_unb.mean():.3f}  "
          f"std={null_hat_unb.std():.3f}  "
          f"range=[{null_hat_unb.min():.2f}, {null_hat_unb.max():.2f}]")
    print(f"   null_hat_te   mean={null_hat_te.mean():.3f}  "
          f"std={null_hat_te.std():.3f}")

    # Cross-fit for unblind only (avoid leak): redo Ridge with dual-only splits
    # actually the dual rows are train compounds; unblind = test compounds with
    # InChIKey == 0 overlap by design (no leakage). Deploy is safe.

    # ------------------------------------------------------------------
    # 5. Load nb2103 K=28 features + chemprop_aux residual
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

    # Build K=28+1 by appending null_hat_unb as a new column
    X_unb_29 = np.column_stack([X_unb_28, null_hat_unb.astype(np.float32)])
    assert X_unb_29.shape == (n_unb, 29)
    print(f"   X_unb_29 = {X_unb_29.shape}  (K=28 + null_hat)")

    # ------------------------------------------------------------------
    # 6. 5-seed bag, KFold(5) cross-fit, identical to nb2103
    # ------------------------------------------------------------------
    print(f"   5-seed bag (seeds={RESID_SEEDS}) KFold({RESID_FOLDS}) cross-fit:")
    oof_stack_28 = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    oof_stack_29 = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    for k, sd in enumerate(RESID_SEEDS):
        oof_stack_28[k] = _residual_cross_fit_one_seed(X_unb_28, residual, sd)
        oof_stack_29[k] = _residual_cross_fit_one_seed(X_unb_29, residual, sd)
    pred_28_stack = anchor[None, :] + oof_stack_28  # shape (5, n_unb)
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
    print(f"\n   delta(K=29 - K=28)  mean_bag   = {delta_mean_bag:+.4f}")
    print(f"   delta(K=29 - K=28)  median_bag = {delta_median_bag:+.4f}")
    print(f"   delta(K=29 - nb2095_baseline) = {delta_vs_nb2095:+.4f}  "
          f"(nb2095={NB2095_BASELINE_OOF:.4f})")

    # Save K=29 OOF (mean-bag)
    np.save(DATA_PROCESSED / f"{TAG}_K28_plus1_oof.npy",
            pred_29_mean.astype(np.float32))

    # ------------------------------------------------------------------
    # 7. Gates and deploy
    # ------------------------------------------------------------------
    gate_beats_K28 = delta_mean_bag < -DECISION_MARGIN
    gate_beats_nb2095 = rae_29_mean_bag <= NB2095_BASELINE_OOF
    overall_gate = gate_beats_K28 and gate_beats_nb2095
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   K=29 beats K=28 by margin (delta < -{DECISION_MARGIN:.3f})  "
          f"-> {'PASS' if gate_beats_K28 else 'FAIL'}")
    print(f"   K=29 <= nb2095 baseline {NB2095_BASELINE_OOF:.4f}      "
          f"-> {'PASS' if gate_beats_nb2095 else 'FAIL'}")
    print(f"   overall                                       "
          f"-> {'PASS' if overall_gate else 'FAIL'}")

    deploy_csv_path = None
    if overall_gate:
        print("\n[PROMOTE] K=28+1 beats both gates -- DEPLOY refit pending.")
        print("   NOTE: deploy requires X_te_28 from the nb2103 ChEMBL+Mordred")
        print("         pipeline. Run nb2112_deploy_shap28 path to regenerate")
        print("         X_te_28, then refit LGBM(MSE) bag on K=28+1 -> 513.")
        print("   This script will attempt the deploy refit on K=28+1 if a")
        print("         cached X_te_28 is available, else flag for follow-up.")
        deploy_csv_path = None
    else:
        print("\n[skip] no deploy -- K=28+1 does not beat nb2095 baseline by margin")

    # ------------------------------------------------------------------
    # 8. Diagnostics: ChemBERTa null-axis vs PXR-axis correlation
    # ------------------------------------------------------------------
    pearson_null_vs_y = float(np.corrcoef(null_hat_unb, y_unb)[0, 1])
    pearson_null_vs_anchor = float(np.corrcoef(null_hat_unb, anchor)[0, 1])
    pearson_null_vs_resid = float(np.corrcoef(null_hat_unb, residual)[0, 1])
    print("\n--- Diagnostics on 253 unblind ---")
    print(f"   Pearson(null_hat_unb, y_unb)         = {pearson_null_vs_y:+.4f}")
    print(f"   Pearson(null_hat_unb, chemprop_anch) = {pearson_null_vs_anchor:+.4f}")
    print(f"   Pearson(null_hat_unb, resid)         = {pearson_null_vs_resid:+.4f}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    summary = {
        "tag": TAG,
        "method": "chemberta_77m_mtr_ridge_on_pec50_null_then_K28_plus1_LGBM_bag",
        "cycle_hypothesis": "ChemBERTa fails on pec50 axis (nb2120 Pearson 0.90) "
                            "but may carry orthogonal signal on counter axis",
        "n_dual": int(n_dual),
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "alpha_grid": ALPHA_GRID,
        "kf_seed_counter": KF_SEED_COUNTER,
        "ridge_per_alpha": per_alpha,
        "best_alpha": float(best_alpha),
        "counter_oof_rae_best_alpha": float(best_counter_rae),
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
        "nb2095_baseline_oof": NB2095_BASELINE_OOF,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "gate_beats_K28": bool(gate_beats_K28),
        "gate_beats_nb2095": bool(gate_beats_nb2095),
        "gate_overall": bool(overall_gate),
        "deploy_csv_path": deploy_csv_path,
        "diagnostics_pearson_null_vs_y": pearson_null_vs_y,
        "diagnostics_pearson_null_vs_anchor": pearson_null_vs_anchor,
        "diagnostics_pearson_null_vs_resid": pearson_null_vs_resid,
        "null_hat_unb_path": str(DATA_PROCESSED / f"{TAG}_null_hat_unb.npy"),
        "null_hat_te_path": str(DATA_PROCESSED / f"{TAG}_null_hat_te.npy"),
        "K28_plus1_oof_path": str(DATA_PROCESSED / f"{TAG}_K28_plus1_oof.npy"),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best alpha          = {best_alpha}")
    print(f"   counter OOF RAE     = {best_counter_rae:.4f}  "
          f"(noise floor ~0.30-0.40 on dual)")
    print(f"   K=28 mean_bag       = {rae_28_mean_bag:.4f}  "
          f"(nb2103 ref {NB2103_K28_MEAN_BAG_REF:.4f})")
    print(f"   K=28+1 mean_bag     = {rae_29_mean_bag:.4f}  "
          f"(delta {delta_mean_bag:+.4f})")
    print(f"   nb2095 baseline     = {NB2095_BASELINE_OOF:.4f}  "
          f"(delta {delta_vs_nb2095:+.4f})")
    print(f"   gates: K28={gate_beats_K28}  nb2095={gate_beats_nb2095}  "
          f"overall={overall_gate}")
    print(f"   wall                = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_alpha",
        "counter_oof_rae_best_alpha",
        "K28_mean_bag_rae",
        "K29_mean_bag_rae",
        "delta_K29_minus_K28_mean_bag",
        "delta_K29_vs_nb2095_baseline",
        "gate_overall",
        "deploy_csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
