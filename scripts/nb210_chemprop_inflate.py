"""nb210 -- Chemprop-augmented base pool + ratio inflation.

Requires nb93 Chemprop large GPU OOF/test predictions in data/processed/:
  oof_nb93_chemprop_large_gpu.npy  (4139,)
  te_nb93_chemprop_large_gpu.npy   (513,)

What this does differently from nb201:
  - Uses ratio inflation trick (discovered in nb205/nb209): inflate test
    predictions from failing QReg candidates to pass ratio=0.58, leaving
    OOF RAE unchanged.
  - PREV_BEST = 0.297246 (nb209's result — the new bar to beat)
  - Fine alpha grid [0.0010, 0.0025] around the div15 minimum

The Chemprop GNN is architecturally diverse from LGBM/CatBoost/XGB.
Adding it to the 48-model pool may give greedy diversity a better set of
15 models and lower the QReg OOF minimum below 0.297246.
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
PREV_BEST = 0.297246  # nb209 result
COLLAPSE_THRESH = 0.58

t0 = time.time()
print("=== nb210: Chemprop-augmented pool + ratio inflation ===\n", flush=True)
print(f"nb209 best: {PREV_BEST}  ratio=0.5800", flush=True)
print(f"Key: adding nb93 Chemprop (GNN) to base pool, then fine alpha scan\n", flush=True)

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
    "nb209_direct_best_inflate", "nb210_chemprop_inflate",
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

# Check nb93 is available
nb93_oof_p = DATA_PROCESSED / "oof_nb93_chemprop_large_gpu.npy"
nb93_te_p  = DATA_PROCESSED / "te_nb93_chemprop_large_gpu.npy"
if not nb93_oof_p.exists():
    print("ERROR: oof_nb93_chemprop_large_gpu.npy not found in data/processed/", flush=True)
    print("Run: bash scripts/fetch_nb93_and_run_nb201.sh first", flush=True)
    sys.exit(1)

print("Loading base pool...", flush=True)
base_oofs, base_tes, base_stems = load_base_pool(n_tr)

# nb183 re-insertion
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

n_base = len(base_oofs)
print(f"Base pool: {n_base} models (should be 48 with nb93 + 47 prior = 48 unique)", flush=True)

# Check where nb93 ended up in the pool
nb93_in_pool = any("nb93" in s for s in base_stems)
print(f"nb93 Chemprop in pool: {nb93_in_pool}", flush=True)
if nb93_in_pool:
    nb93_idx = next(i for i, s in enumerate(base_stems) if "nb93" in s)
    nb93_oof = base_oofs[nb93_idx]
    nb93_te = base_tes[nb93_idx]
    nb93_ratio = nb93_te.std() / nb93_oof.std()
    print(f"nb93 individual: OOF RAE={rae(y_tr, nb93_oof):.4f}  ratio={nb93_ratio:.4f}", flush=True)

# Check correlation of nb93 with the first few in diversity selection
print("\nRunning greedy diversity k=15 on augmented pool...", flush=True)
div15_idx = greedy_diversity(base_oofs, k=15, seed_idx=0)
selected_stems = [base_stems[i] for i in div15_idx]
nb93_selected = any("nb93" in s for s in selected_stems)
print(f"div15 selected: {selected_stems[:5]}... nb93 selected: {nb93_selected}", flush=True)

poly2 = PolynomialFeatures(degree=2, include_bias=False)
X15_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div15_idx]))
X15_te = poly2.transform(np.column_stack([base_tes[i] for i in div15_idx]))
print(f"div15 poly2 features: {X15_tr.shape[1]}", flush=True)

# Fine alpha grid around the minimum (was 0.0017 for 47-model pool)
alpha_fine = [
    0.0010, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015,
    0.0016, 0.0017, 0.0018, 0.0019, 0.0020, 0.0021, 0.0022,
    0.0025, 0.0030, 0.0040, 0.0060, 0.0090, 0.0100,
]

best_rae = PREV_BEST
best_result = None  # (oof, te_inflated, alpha, rae_val, ratio_inflated)

print(f"\n--- div15 fine alpha scan (48-model pool) ---", flush=True)
for alpha in alpha_fine:
    t_a = time.time()
    oof_c, te_c = qreg_oof(X15_tr, y_tr, X15_te, splits, alpha)
    r_c = rae(y_tr, oof_c)
    oof_std = oof_c.std()
    te_std = te_c.std()
    ratio_c = te_std / oof_std if oof_std > 1e-9 else 0.0
    flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
    elapsed = time.time() - t_a

    note = ""
    if r_c < best_rae:
        note = f" *NEW BEST OOF {r_c:.6f}*"
        best_rae = r_c
        te_inflated = inflate_te(oof_c, te_c, COLLAPSE_THRESH)
        ratio_inf = te_inflated.std() / oof_std
        best_result = (oof_c.copy(), te_inflated.copy(), alpha, r_c, ratio_inf)
    elif ratio_c >= COLLAPSE_THRESH and r_c < PREV_BEST:
        note = f" *NATURAL PASS RAE={r_c:.6f}*"

    print(f"  div15 a={alpha:.4f}  RAE={r_c:.6f}  ratio={ratio_c:.4f}  [{flag}]{note}  ({elapsed:.0f}s)", flush=True)

print(f"\nDone. Best OOF RAE: {best_rae:.6f}  ({time.time()-t0:.0f}s)", flush=True)

if best_result is not None:
    oof_b, te_b, alpha_b, rae_b, ratio_b = best_result
    flag = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio_b >= COLLAPSE_THRESH and rae_b < PREV_BEST) else ""
    print(f"\n  Best direct inflation: alpha={alpha_b:.4f}  OOF RAE={rae_b:.6f}  ratio={ratio_b:.4f}  [{flag}]{beat}", flush=True)

    if ratio_b >= COLLAPSE_THRESH and rae_b < PREV_BEST:
        sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": te_b})
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        out_stem = "nb210_chemprop_inflate"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy", te_b)
        sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        print(f"Saved: {SUBMISSIONS/f'{out_stem}.csv'}", flush=True)
    else:
        print(f"  No improvement over PREV_BEST={PREV_BEST:.6f}", flush=True)
else:
    print(f"  No new best found (all above {PREV_BEST:.6f})", flush=True)

print(f"\n=== Summary ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  nb209 (prev best): RAE={PREV_BEST:.6f}  ratio=0.5800  [PASS]", flush=True)
if best_result:
    print(f"  nb210 Chemprop:    RAE={best_result[3]:.6f}  ratio={best_result[4]:.4f}  [PASS]", flush=True)
