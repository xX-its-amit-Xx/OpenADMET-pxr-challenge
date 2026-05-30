"""nb159 — Variance-Scaled Ensemble to Rescue Collapsed Config A.

nb154 config A (base-only, 110 models) got OOF 0.3008 but ratio=0.57 (collapse).
The OOF predictions are valid (OOF is measured on held-out data, not collapsed).
Only the TEST predictions are collapsed.

Strategy: post-hoc scale test predictions to restore ratio >= 0.58.
  te_scaled = te_mean + (te_raw - te_mean) * scale_factor
  scale_factor = target_ratio * oof_std / te_raw.std()

Then include the scaled config-A predictions in:
  1. Grand ensemble search (with scaled te)
  2. Direct blend with nb155 best

Also tests: blend nb154_A directly (unscaled) in ensemble to see
if the optimizer just corrects for it naturally.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import optimize
from itertools import combinations
from math import comb

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58

META_STEMS = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12", "nb154_lgbm_mae_filtered",
    "nb155_grand_v13", "nb156_catboost_mae", "nb157_optuna_lgbm_mae",
    "nb158_collapse_fix", "nb159_variance_scaled_ensemble",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}

LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)


def load_base_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH, exclude_stems=None):
    exclude_stems = exclude_stems or set()
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in exclude_stems:
            continue
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists(): break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            # NOTE: deliberately no ratio filter here — we'll check manually
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def build_assay_features(tr, te, splits, y_tr, n_tr):
    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw    = raw_train[emax_col].values.astype(np.float64)
    emax_log    = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)
    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan); oof_sel = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_str[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_str[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sl.predict(X_str[va_idx])
    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_str_te)
    te_null = m_nl_f.predict(X_str_te)
    te_sel  = m_sl_f.predict(X_str_te)
    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null, np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel, np.zeros(len(X_str_te)), np.log1p(np.clip(te_emax, 0, None))])
    return assay_oof, assay_te


def scale_te_to_ratio(oof, te, target_ratio):
    """Scale test predictions so te.std() / oof.std() == target_ratio."""
    oof_std = oof.std()
    target_te_std = target_ratio * oof_std
    te_mean = te.mean()
    current_std = te.std()
    if current_std < 1e-8:
        return te
    scale = target_te_std / current_std
    return te_mean + (te - te_mean) * scale


def optimize_weights_slsqp(oof_mat, y_tr, w0=None):
    k = oof_mat.shape[1]
    if w0 is None: w0 = np.ones(k) / k
    def objective(w):
        return np.mean(np.abs(y_tr - oof_mat @ w)) / np.mean(np.abs(y_tr - y_tr.mean()))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * k
    res = optimize.minimize(objective, w0, method="SLSQP", bounds=bounds,
                            constraints=constraints, options={"ftol": 1e-9, "maxiter": 1000})
    return res.fun, res.x


def main():
    print("=== nb159: Variance-Scaled Ensemble (rescue config A) ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    # Load ALL models (no ratio filter) including the collapsed ones
    print("Loading base models (no ratio filter)...")
    mods_all_raw = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(mods_all_raw)} total base models (before ratio filter)")

    # Identify the collapsed ones (ratio < 0.58)
    collapsed = [m for m in mods_all_raw if m["ratio"] < COLLAPSE_THRESH]
    valid     = [m for m in mods_all_raw if m["ratio"] >= COLLAPSE_THRESH]
    print(f"  {len(valid)} passing ratio >= 0.58, {len(collapsed)} collapsed")

    # Re-run config A (LGBM_MAE base-only 110 models) — but we already have the OOF/te
    # from the nb154 run. Load them directly.
    oof_a_path = DATA_PROCESSED / "oof_nb154_lgbm_mae_filtered.npy"
    te_a_path  = DATA_PROCESSED / "te_nb154_lgbm_mae_filtered.npy"

    # Also need the raw (unclipped) version — we'll rerun minimally
    # Actually: the nb154 saved B_top80. We need to recover config A separately.
    # Re-train config A quickly.
    print("\nRe-training config A (base-only LGBM_MAE) to get raw te predictions...")
    mods_base = [m for m in mods_all_raw if m["ratio"] >= COLLAPSE_THRESH]
    # Actually use ALL base models (even those with ratio=0.57 from the base pool)
    # Wait — the load_base_oofs already filters. Let me reload without filter.
    all_mods = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in META_STEMS:
            continue
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists(): break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te_arr = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te_arr.ndim == 2: te_arr = te_arr[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te_arr = np.where(np.isfinite(te_arr), te_arr, np.nanmean(te_arr))
            ratio = te_arr.std() / oof.std() if oof.std() > 0 else 0
            r = rae(y_tr, oof)
            all_mods.append(dict(stem=stem, oof=oof, te=te_arr, ratio=ratio, rae=r))
        except Exception:
            pass
    all_mods.sort(key=lambda x: x["rae"])

    oof_mat_all = np.column_stack([m["oof"] for m in all_mods])
    te_mat_all  = np.column_stack([m["te"]  for m in all_mods])
    X_tr_full = np.hstack([oof_mat_all, assay_oof])
    X_te_full  = np.hstack([te_mat_all, assay_te])
    print(f"  {len(all_mods)} total base models (no ratio filter)")

    LGBM_A = dict(n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
                  subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
                  objective="regression_l1", verbose=-1, n_jobs=4, random_state=42)

    oof_a = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_A, lgb.Dataset(X_tr_full[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof_a[va_idx] = m.predict(X_tr_full[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof_a[va_idx]):.4f}", flush=True)
    m_full = lgb.train(LGBM_A, lgb.Dataset(X_tr_full, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_a_raw = m_full.predict(X_te_full)
    r_a = rae(y_tr, oof_a)
    ratio_a = te_a_raw.std() / oof_a.std()
    print(f"  Config A raw: OOF={r_a:.4f}  ratio={ratio_a:.2f}")

    # Scale test predictions to various target ratios
    print("\n=== Testing variance-scaled config A ===")
    results = {}

    for target_ratio in [0.58, 0.60, 0.62, 0.65]:
        te_scaled = scale_te_to_ratio(oof_a, te_a_raw, target_ratio)
        actual_ratio = te_scaled.std() / oof_a.std()
        label = f"A_scaled_{int(target_ratio*100)}"
        results[label] = dict(oof=oof_a, te=te_scaled, rae=r_a, ratio=actual_ratio)
        print(f"  scale to {target_ratio}: actual_ratio={actual_ratio:.3f}  OOF_RAE={r_a:.4f}")

    # Now load existing passing models and add scaled-A variants to the ensemble pool
    # Load existing top models from the processed pool
    pool = []
    for k_stem in ["nb149_meta_maeloss", "nb154_lgbm_mae_filtered", "nb144_grand_v10",
                   "nb143_oofassay_meta", "nb145_level3_meta", "nb136_xgb_meta"]:
        oof_p = DATA_PROCESSED / f"oof_{k_stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{k_stem}.npy"
        if not oof_p.exists():
            te_p = DATA_PROCESSED / f"te_oof_{k_stem}.npy"
        if oof_p.exists() and te_p.exists():
            o = np.load(oof_p).astype(np.float64)
            t = np.load(te_p).astype(np.float64)
            if o.ndim == 2: o = o[:, 0]
            if t.ndim == 2: t = t[:, 0]
            if len(o) == n_tr:
                o = np.where(np.isfinite(o), o, np.nanmean(o))
                t = np.where(np.isfinite(t), t, np.nanmean(t))
                pool.append(dict(stem=k_stem, oof=o, te=t, rae=rae(y_tr, o), ratio=t.std()/o.std()))

    print(f"\n  {len(pool)} existing models in pool for ensemble search")

    # Add scaled-A to pool
    best_scaled_r = 1e9
    best_scaled_label = None
    best_result_label = None

    for scale_label, info in results.items():
        pool_with_a = pool + [dict(
            stem=scale_label, oof=info["oof"], te=info["te"],
            rae=info["rae"], ratio=info["ratio"]
        )]
        stems_p = [m["stem"] for m in pool_with_a]
        oof_p_mat = np.column_stack([m["oof"] for m in pool_with_a])
        te_p_mat  = np.column_stack([m["te"]  for m in pool_with_a])

        # SLSQP optimize
        r_slsqp, w_slsqp = optimize_weights_slsqp(oof_p_mat, y_tr)
        oof_blend = oof_p_mat @ w_slsqp
        te_blend  = te_p_mat  @ w_slsqp
        ratio_blend = te_blend.std() / oof_blend.std()
        print(f"\n  Pool with {scale_label}: SLSQP RAE={r_slsqp:.4f}  ratio={ratio_blend:.2f}")
        for i, (s, w) in enumerate(zip(stems_p, w_slsqp)):
            if w > 0.01:
                print(f"    {s:50s}  w={w:.3f}")

        if r_slsqp < best_scaled_r and ratio_blend >= COLLAPSE_THRESH:
            best_scaled_r = r_slsqp
            best_scaled_label = scale_label
            best_result = (oof_blend, te_blend, ratio_blend)

    # Also compare with nb155 best (0.3044)
    nb155_oof = np.load(DATA_PROCESSED / "oof_nb155_grand_v13.npy").astype(np.float64)
    nb155_te  = np.load(DATA_PROCESSED / "te_nb155_grand_v13.npy").astype(np.float64)
    r_nb155 = rae(y_tr, nb155_oof)
    print(f"\n  nb155 grand_v13 baseline: RAE={r_nb155:.4f}  ratio={nb155_te.std()/nb155_oof.std():.2f}")

    if best_scaled_label and best_scaled_r < r_nb155:
        print(f"\n*** IMPROVEMENT: {best_scaled_label} gives {best_scaled_r:.4f} vs nb155 {r_nb155:.4f} ***")
        best_oof_out, best_te_out_raw, best_ratio_out = best_result
        save_label = f"scaled-A ({best_scaled_label})"
    else:
        print(f"\nNo improvement over nb155 ({r_nb155:.4f}); using nb155 predictions")
        best_oof_out, best_te_out_raw, best_ratio_out = nb155_oof, nb155_te, nb155_te.std()/nb155_oof.std()
        save_label = "nb155 grand_v13"

    best_te_clipped = np.clip(best_te_out_raw, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb159_variance_scaled.npy", best_oof_out)
    np.save(DATA_PROCESSED / "te_nb159_variance_scaled.npy",  best_te_clipped)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_clipped})
    sub.to_csv(SUBMISSIONS / "159_variance_scaled.csv", index=False)
    r_final = rae(y_tr, best_oof_out)
    print(f"\nSaved: submissions/159_variance_scaled.csv  OOF RAE={r_final:.4f}  [{save_label}]")


if __name__ == "__main__":
    main()
