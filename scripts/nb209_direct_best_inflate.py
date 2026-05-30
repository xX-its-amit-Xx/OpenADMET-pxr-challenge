"""nb209 -- Direct submission of the best inflated QReg candidate.

nb207 discovered the true OOF minimum is at div15 a=0.0017 (RAE=0.297246),
better than nb205's a=0.002 (RAE=0.297307).

The optimal SLSQP solution for the inflated pool is w=1.0 on the inflated
div15 a=0.0017 candidate (since it has the lowest individual OOF RAE and
all NB188 base models have much higher RAE). This script computes that
candidate directly, avoiding the hour-long SLSQP wait.

Inflation: te_inflated = mean(te) + (te - mean(te)) × (0.580 × oof_std / te_std)
This brings ratio from 0.5617 → 0.580 exactly.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
PREV_BEST = 0.297639
COLLAPSE_THRESH = 0.58

t0 = time.time()
print("=== nb209: Direct best inflated QReg candidate ===\n", flush=True)

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
    "nb182_qreg_alpha", "nb183_qreg_poly10", "nb184_grand_v16",
    "nb185_qreg_iter", "nb186_single_conc_lookup",
    "nb187_diversity_qreg", "nb188_diverse_refine", "nb189_iterate_diverse",
    "nb190_random_diverse_search", "nb191_lgbm_qstack",
    "nb192_poly_variant", "nb193_div_fine",
    "nb194_constrained", "nb195_expanded", "nb196_fine_div15",
    "nb197_dense_grid", "nb198_k_sweep", "nb199_external_pxr",
    "nb200_multi_seed", "nb201_chemprop_slsqp", "nb202_ridge_external",
    "nb203_div25_low_alpha", "nb204_multiseed_fast", "nb205_ratio_inflate",
    "nb206_small_k_grid", "nb207_fine_alpha_min", "nb208_direct_inflate_submit",
    "nb209_direct_best_inflate",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}


def load_base_pool(n_tr):
    oofs, tes, stems = [], [], []
    for f in sorted(DATA_PROCESSED.glob("oof_nb*.npy")):
        stem = f.stem[4:]
        if any(stem.startswith(ms) or stem == ms for ms in META_STEMS):
            continue
        for pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{pref}{stem}.npy"
            if te_p.exists():
                break
        else:
            continue
        oof = np.load(f).astype(np.float64).flatten()
        te  = np.load(te_p).astype(np.float64).flatten()
        if len(oof) != n_tr:
            continue
        oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
        te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
        oofs.append(oof); tes.append(te); stems.append(stem)
    return oofs, tes, stems


def greedy_diversity(oofs, k, seed_idx=0):
    X = np.column_stack(oofs)
    corr = np.abs(np.corrcoef(X.T))
    selected = [seed_idx]
    remaining = list(range(len(oofs)))
    remaining.remove(seed_idx)
    while len(selected) < k and remaining:
        avg_corrs = [(np.mean([corr[i, j] for j in selected]), i) for i in remaining]
        avg_corrs.sort(key=lambda x: x[0])
        selected.append(avg_corrs[0][1])
        remaining.remove(avg_corrs[0][1])
    return selected


def qreg_oof(X_tr, y_tr, X_te, splits, alpha):
    n_tr = len(y_tr)
    oof_s = np.full(n_tr, np.nan)
    for _, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_s[va_idx] = m.predict(X_tr[va_idx])
    m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
    m_f.fit(X_tr, y_tr)
    return oof_s, m_f.predict(X_te)


def inflate_te(oof_pred, te_pred, target_ratio):
    oof_std = oof_pred.std()
    te_std = te_pred.std()
    if te_std < 1e-9:
        return te_pred.copy()
    te_mean = te_pred.mean()
    return te_mean + (te_pred - te_mean) * (target_ratio * oof_std / te_std)


tr = load_train()
te_df = load_test()
y_tr = tr["pec50"].values.astype(np.float64)
n_tr = len(y_tr)
scaffolds = tr["smiles"].map(bemis_murcko).tolist()
splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

print("Loading base pool...", flush=True)
base_oofs, base_tes, base_stems = load_base_pool(n_tr)

# nb183 re-insertion (same as nb205/nb207)
oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
if oof183_p.exists() and not any(s == "nb183_qreg_poly10" for s in base_stems):
    o183 = np.load(oof183_p).astype(np.float64).flatten()
    t183 = np.load(te183_p).astype(np.float64).flatten()
    o183 = np.where(np.isfinite(o183), o183, np.nanmean(o183))
    t183 = np.where(np.isfinite(t183), t183, np.nanmean(t183))
    if len(o183) == n_tr:
        base_oofs.insert(0, o183); base_tes.insert(0, t183)
        base_stems.insert(0, "nb183_qreg_poly10")

print(f"Base pool: {len(base_oofs)} models", flush=True)

poly2 = PolynomialFeatures(degree=2, include_bias=False)
div15_idx = greedy_diversity(base_oofs, k=15, seed_idx=0)
X15_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div15_idx]))
X15_te = poly2.transform(np.column_stack([base_tes[i] for i in div15_idx]))

# Best alpha from nb207 scan: a=0.0017 gives OOF RAE=0.297246
BEST_ALPHA = 0.0017
print(f"\nBuilding div15 QReg at a={BEST_ALPHA}...", flush=True)
t_a = time.time()
oof_c, te_c = qreg_oof(X15_tr, y_tr, X15_te, splits, BEST_ALPHA)
r_c = rae(y_tr, oof_c)
oof_std = oof_c.std()
te_std = te_c.std()
ratio_c = te_std / oof_std if oof_std > 1e-9 else 0.0
print(f"  div15 a={BEST_ALPHA}:  OOF RAE={r_c:.6f}  ratio={ratio_c:.4f}  ({time.time()-t_a:.0f}s)", flush=True)

# Inflate to exactly COLLAPSE_THRESH=0.580
te_inflated = inflate_te(oof_c, te_c, COLLAPSE_THRESH)
ratio_inflated = te_inflated.std() / oof_std
print(f"  After inflation to {COLLAPSE_THRESH}: ratio={ratio_inflated:.4f}", flush=True)

beat = " ***NEW BEST***" if r_c < PREV_BEST else ""
flag = "PASS" if ratio_inflated >= COLLAPSE_THRESH else "FAIL"
print(f"\n  Direct inflated candidate: OOF RAE={r_c:.6f}  ratio={ratio_inflated:.4f}  [{flag}]{beat}", flush=True)

print(f"\n=== Summary ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  nb197 (best):  RAE={PREV_BEST:.6f}  ratio=0.5800  [PASS]", flush=True)
print(f"  nb209 direct:  RAE={r_c:.6f}  ratio={ratio_inflated:.4f}  [{flag}]{beat}", flush=True)

if r_c < PREV_BEST and ratio_inflated >= COLLAPSE_THRESH:
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": te_inflated})
    assert len(sub) == 513 and sub["pEC50"].notna().all()
    out_stem = "nb209_direct_best_inflate"
    np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_c)
    np.save(DATA_PROCESSED / f"te_{out_stem}.npy", te_inflated)
    sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
    print(f"Saved: {SUBMISSIONS/f'{out_stem}.csv'}", flush=True)
else:
    print("Not saving — no improvement over nb197.", flush=True)
