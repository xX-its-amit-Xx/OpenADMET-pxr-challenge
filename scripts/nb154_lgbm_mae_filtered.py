"""nb154 — LGBM MAE Meta-Stack with Filtered OOF Inputs.

nb149 found LGBM_MAE on OOF+assay = 0.3069 (118 features from 113 models).
nb152 added meta-stack OOF files → 120 models → 0.3104 (WORSE!).

Hypothesis: meta-stack OOF files (nb143, nb144, nb149, nb151 etc.) add circular
features that confuse the meta-learner. Solution: filter to only "base" models
OR filter by OOF RAE quality threshold.

Tests:
  A: All models passing ratio >= 0.58 (same as nb149 = 113ish models)
     BUT exclude known meta-stack stems (nb13x, nb14x, nb15x)
  B: Only top-K models by RAE (K=50, K=80, K=100)
  C: Exclude models with RAE > 0.45 (strict quality filter)
  D: Original nb149 config — just load exactly what nb149 loaded (re-reproduce)

Also: multi-seed ensemble of LGBM_MAE (seeds 42, 123, 456, 789, 1234) for variance reduction.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58

LGBM_MAE = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
    subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05, objective="regression_l1",
    verbose=-1, n_jobs=4, random_state=42
)
LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)

# Known meta-stack stems to exclude for clean Level-2 stacking
META_STEMS = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12",
    # Grand ensemble outputs
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}


def load_filtered_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH, exclude_stems=None,
                       max_rae=None, top_k=None):
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
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            if max_rae is not None and r > max_rae: continue
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    if top_k is not None:
        results = results[:top_k]
    return results


def cv_lgbm_mae(cfg, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(cfg, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
    m_full = lgb.train(cfg, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    print(f"  [{label:55s}] RAE={r:.4f}  n_feats={X_tr.shape[1]}  ratio={ratio:.2f}", flush=True)
    return r, oof, te_pred, ratio


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

    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                  np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel,
                                  np.zeros(len(X_str_te)),
                                  np.log1p(np.clip(te_emax, 0, None))])
    return assay_oof, assay_te


def main():
    print("=== nb154: LGBM MAE Meta-Stack with Filtered OOF ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    results = {}

    # A: Exclude meta-stack stems (clean Level-2)
    print("\n--- A: Base models only (exclude meta-stacks) ---")
    mods = load_filtered_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(mods)} models loaded (excluding meta-stacks)")
    if mods:
        oof_mat = np.column_stack([m["oof"] for m in mods])
        te_mat  = np.column_stack([m["te"]  for m in mods])
        X_tr = np.hstack([oof_mat, assay_oof]); X_te = np.hstack([te_mat, assay_te])
        r, oof, te_pred, ratio = cv_lgbm_mae(LGBM_MAE, X_tr, y_tr, X_te, splits, "A: base-only+assay")
        results["A_base_only"] = (r, oof, te_pred, ratio)

    # B: Top-80 models by RAE
    print("\n--- B: Top-80 models by OOF RAE ---")
    mods_80 = load_filtered_oofs(n_tr, y_tr, top_k=80)
    print(f"  Top-80 loaded, worst RAE: {mods_80[-1]['rae']:.4f}")
    oof_mat_80 = np.column_stack([m["oof"] for m in mods_80])
    te_mat_80  = np.column_stack([m["te"]  for m in mods_80])
    X_tr = np.hstack([oof_mat_80, assay_oof]); X_te = np.hstack([te_mat_80, assay_te])
    r, oof, te_pred, ratio = cv_lgbm_mae(LGBM_MAE, X_tr, y_tr, X_te, splits, "B: top80+assay")
    results["B_top80"] = (r, oof, te_pred, ratio)

    # C: RAE < 0.45 (only strong models)
    print("\n--- C: Models with RAE < 0.45 ---")
    mods_45 = load_filtered_oofs(n_tr, y_tr, max_rae=0.45)
    print(f"  {len(mods_45)} models loaded (RAE < 0.45)")
    if len(mods_45) >= 2:
        oof_mat_45 = np.column_stack([m["oof"] for m in mods_45])
        te_mat_45  = np.column_stack([m["te"]  for m in mods_45])
        X_tr = np.hstack([oof_mat_45, assay_oof]); X_te = np.hstack([te_mat_45, assay_te])
        r, oof, te_pred, ratio = cv_lgbm_mae(LGBM_MAE, X_tr, y_tr, X_te, splits, "C: RAE<0.45+assay")
        results["C_rae_lt45"] = (r, oof, te_pred, ratio)

    # D: Multi-seed ensemble (5 seeds, base-only config)
    print("\n--- D: Multi-seed ensemble (5 seeds, base models) ---")
    if "A_base_only" in results:
        mods_base = load_filtered_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
        oof_mat_b = np.column_stack([m["oof"] for m in mods_base])
        te_mat_b  = np.column_stack([m["te"]  for m in mods_base])
        X_tr = np.hstack([oof_mat_b, assay_oof]); X_te = np.hstack([te_mat_b, assay_te])
        seed_oofs = []; seed_tes = []
        for seed in [42, 123, 456, 789, 1234]:
            cfg = {**LGBM_MAE, "random_state": seed}
            oof_s = np.full(n_tr, np.nan)
            for fold, (tr_idx, va_idx) in enumerate(splits):
                m = lgb.train(cfg, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                              callbacks=[lgb.log_evaluation(-1)])
                oof_s[va_idx] = m.predict(X_tr[va_idx])
            m_full = lgb.train(cfg, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
            seed_oofs.append(oof_s); seed_tes.append(m_full.predict(X_te))
        oof_d = np.mean(seed_oofs, axis=0)
        te_d  = np.mean(seed_tes, axis=0)
        r_d = rae(y_tr, oof_d)
        ratio_d = te_d.std() / oof_d.std()
        print(f"  [D: multi-seed 5x] RAE={r_d:.4f}  ratio={ratio_d:.2f}")
        results["D_multiseed"] = (r_d, oof_d, te_d, ratio_d)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:50s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, _ = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}  (nb149 baseline: 0.3069)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb154_lgbm_mae_filtered.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb154_lgbm_mae_filtered.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "154_lgbm_mae_filtered.csv", index=False)
    print(f"\nSaved: submissions/154_lgbm_mae_filtered.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
