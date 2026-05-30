"""nb184 -- Grand v16: Refine 7-model SLSQP with more starts + expand pool.

nb183 breakthrough:
  QReg(0.5, alpha=0.0012) on degree-2 poly of top-10 OOFs: RAE=0.300023 (nb183)
  7-model SLSQP (6 original + nb183): RAE=0.299711  ratio=0.5801  PASS

This script:
  A: Refine 7-model SLSQP with 500 starts (verify convergence)
  B: 8-model SLSQP (add each pool model one at a time)
  C: Multi-start SLSQP over top-15 pool including nb183
  D: Save final grand ensemble OOF
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

COLLAPSE_THRESH = 0.58
SEED = 42

SIX_MODELS = [
    "nb167_xgboost_mae",
    "nb156_catboost_mae",
    "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool",
    "nb165_multiseed_162c",
    "nb149_meta_maeloss",
]

# Models to try as 8th in the pool (ordered by OOF RAE)
EXTRA_CANDIDATES = [
    "nb166_catboost_v2",
    "nb168_multiseed_catboost",
    "nb169_rf_et_mae",
    "nb171_catboost_extended",
    "nb175_bayes_blend",
    "nb170_grand_v15",
    "nb176_optuna_weights",
]

# Stems to exclude from grand pool (meta-learners / grand ensembles)
META_STEMS = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12", "nb154_lgbm_mae_filtered",
    "nb155_grand_v13", "nb156_catboost_mae", "nb157_optuna_lgbm_mae",
    "nb158_collapse_fix", "nb159_variance_scaled_ensemble", "nb160_pca_meta",
    "nb161_neural_meta", "nb162_mixed_pool", "nb163_lgbm_colsample_low",
    "nb163_lgbm_colsample",
    "nb164_grand_v14", "nb165_multiseed_162c", "nb166_catboost_v2",
    "nb167_xgboost_mae",
    "nb168_multiseed_catboost", "nb169_rf_et_mae",
    "nb170_grand_v15", "nb171_catboost_extended",
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb174_top10_lgbm", "nb175_bayes_blend", "nb176_optuna_weights",
    "nb177_xgb_histgb", "nb178_xgb_10seed", "nb179_xgb_collapse_fix",
    "nb180_nonlinear_6model", "nb181_quantile_poly",
    "nb182_qreg_alpha", "nb183_qreg_poly10",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}


def load_oof(stem, n_tr):
    p = DATA_PROCESSED / f"oof_{stem}.npy"
    if not p.exists():
        return None, None
    for pref in ("te_", "te_oof_"):
        te_p = DATA_PROCESSED / f"{pref}{stem}.npy"
        if te_p.exists():
            break
    else:
        return None, None
    oof = np.load(p).astype(np.float64).flatten()
    te  = np.load(te_p).astype(np.float64).flatten()
    if len(oof) != n_tr:
        return None, None
    oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
    te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
    return oof, te


def slsqp_multistart(X_tr, y_tr, X_te, n_starts, label):
    k = X_tr.shape[1]
    mean_pred = y_tr.mean()
    def obj(w): return np.mean(np.abs(y_tr - X_tr@w)) / np.mean(np.abs(y_tr - mean_pred))
    best_r, best_w = 1e9, None
    for _ in range(n_starts):
        w0 = np.random.dirichlet(np.ones(k))
        r = minimize(obj, w0, method='SLSQP',
                     bounds=[(0,1)]*k,
                     constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                     options={'ftol':1e-12, 'maxiter':2000})
        if r.fun < best_r:
            best_r, best_w = r.fun, r.x
    oof_b = X_tr@best_w; te_b = X_te@best_w
    ratio = te_b.std()/oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label}] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]  w={best_w.round(4)}")
    return best_r, oof_b, te_b, ratio, best_w


def main():
    print("=== nb184: Grand v16 -- Refine 7-model SLSQP + expand pool ===\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    # Load 6 original SLSQP models
    oofs6, tes6 = [], []
    for stem in SIX_MODELS:
        oof, te = load_oof(stem, n_tr)
        if oof is not None:
            oofs6.append(oof); tes6.append(te)
    X6_tr = np.column_stack(oofs6); X6_te = np.column_stack(tes6)
    print(f"Loaded {X6_tr.shape[1]}/6 original models")

    # Load nb183 (QReg poly-10 alpha=0.0012)
    oof183, te183 = load_oof("nb183_qreg_poly10", n_tr)
    if oof183 is None:
        print("ERROR: nb183 not found! Run nb183_qreg_save.py first.")
        return
    r183 = rae(y_tr, oof183); ratio183 = te183.std()/oof183.std()
    print(f"nb183 standalone: RAE={r183:.6f}  ratio={ratio183:.4f}")

    # 7-model matrix: 6 originals + nb183
    X7_tr = np.column_stack([X6_tr, oof183])
    X7_te = np.column_stack([X6_te, te183])

    # --- A: 7-model SLSQP with 500 starts ---
    print(f"\n--- A: 7-model SLSQP (500 starts) ---")
    r_a, oof_a, te_a, ratio_a, w_a = slsqp_multistart(X7_tr, y_tr, X7_te, 500, "7model_500starts")
    if r_a < 0.3001:
        print(f"  *** BEATS 0.3001! ***")

    # --- B: 8-model SLSQP (add each candidate one at a time) ---
    print(f"\n--- B: 8-model SLSQP (add candidate, 100 starts each) ---")
    best_b_r, best_b_oof, best_b_te, best_b_ratio, best_b_stem = 1e9, None, None, None, None
    for stem in EXTRA_CANDIDATES:
        oof_e, te_e = load_oof(stem, n_tr)
        if oof_e is None:
            continue
        X8_tr = np.column_stack([X7_tr, oof_e])
        X8_te = np.column_stack([X7_te, te_e])
        r_e, oof8, te8, ratio8, w8 = slsqp_multistart(
            X8_tr, y_tr, X8_te, 100, f"7+{stem.split('_')[0]}"
        )
        if ratio8 >= COLLAPSE_THRESH and r_e < best_b_r:
            best_b_r, best_b_oof, best_b_te, best_b_ratio, best_b_stem = r_e, oof8, te8, ratio8, stem
        if r_e < 0.3001 and ratio8 >= COLLAPSE_THRESH:
            print(f"  *** BEATS 0.3001 with {stem}! ***")

    if best_b_stem:
        print(f"  Best 8-model: +{best_b_stem}  RAE={best_b_r:.6f}  ratio={best_b_ratio:.4f}")

    # --- C: Grand pool search including nb183 ---
    print(f"\n--- C: Grand pool (include nb183, top-20 by RAE, 200 starts) ---")
    # Load all OOF files, exclude meta-stems
    all_stems = []
    for f in sorted(DATA_PROCESSED.glob("oof_nb*.npy")):
        stem = f.stem[4:]  # remove 'oof_'
        if any(stem.startswith(ms) for ms in META_STEMS):
            continue
        all_stems.append(stem)
    # Also include nb183 (it's in META_STEMS exclusion above, so we manually add back)
    if "nb183_qreg_poly10" not in all_stems:
        all_stems.append("nb183_qreg_poly10")

    pool_oofs, pool_tes, pool_stems = [], [], []
    for stem in all_stems:
        oof, te = load_oof(stem, n_tr)
        if oof is None:
            continue
        pool_oofs.append(oof); pool_tes.append(te); pool_stems.append(stem)

    # Sort by standalone RAE, take top-20
    stem_raes = [(rae(y_tr, oof), stem, oof, te) for stem, oof, te in zip(pool_stems, pool_oofs, pool_tes)]
    stem_raes.sort(key=lambda x: x[0])
    top20 = stem_raes[:20]
    print(f"  Pool size: {len(pool_stems)} models, using top-{len(top20)}")
    for r_s, stem, _, _ in top20[:10]:
        print(f"    {stem:50s}  RAE={r_s:.4f}")

    X20_tr = np.column_stack([v[2] for v in top20])
    X20_te = np.column_stack([v[3] for v in top20])

    r_c, oof_c, te_c, ratio_c, w_c = slsqp_multistart(
        X20_tr, y_tr, X20_te, 200, "top20_grand"
    )
    if r_c < 0.3001 and ratio_c >= COLLAPSE_THRESH:
        print(f"  *** BEATS 0.3001! ***")

    # Show non-zero weights
    for i, (_, stem, _, _) in enumerate(top20):
        if w_c[i] > 0.001:
            print(f"    {stem:50s}  w={w_c[i]:.4f}")

    # === Summary ===
    print(f"\n=== Summary ===")
    results = []
    if ratio_a >= COLLAPSE_THRESH:
        results.append((r_a, oof_a, te_a, ratio_a, "7model_500starts"))
    if best_b_r < 1e9 and best_b_ratio >= COLLAPSE_THRESH:
        results.append((best_b_r, best_b_oof, best_b_te, best_b_ratio, f"8model+{best_b_stem}"))
    if ratio_c >= COLLAPSE_THRESH:
        results.append((r_c, oof_c, te_c, ratio_c, "top20_grand"))

    print(f"  Linear ref (6-model):    RAE=0.300101  ratio=0.580  PASS")
    print(f"  nb183 standalone:        RAE={r183:.6f}  ratio={ratio183:.4f}")
    for r, _, _, ratio, label in sorted(results, key=lambda x: x[0]):
        beat = " *** NEW BEST ***" if r < 0.2999 else (" *** BEATS 0.3001 ***" if r < 0.3001 else "")
        print(f"  {label:35s}  RAE={r:.6f}  ratio={ratio:.4f}  PASS{beat}")

    if not results:
        print("  Using 7-model result from nb183 (0.299711)")
        return

    best_r, best_oof, best_te, best_ratio, best_label = min(results, key=lambda x: x[0])
    print(f"\nBEST: {best_label}  RAE={best_r:.6f}  ratio={best_ratio:.4f}")

    # Save grand ensemble OOF/TE
    np.save(DATA_PROCESSED / "oof_nb184_grand_v16.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb184_grand_v16.npy",  best_te)

    best_te_clip = np.clip(best_te, y_tr.min()-0.5, y_tr.max()+0.5)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
    sub.to_csv(SUBMISSIONS / "184_grand_v16.csv", index=False)
    print(f"Saved: submissions/184_grand_v16.csv  OOF RAE={best_r:.6f}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
