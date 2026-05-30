"""nb174 — LGBM_MAE on top-10 best OOF models.

nb173 found that softmax weighting of top-10 models achieves 0.3009 OOF RAE
WITHOUT any training. This motivates training LGBM_MAE directly on these
top-10 OOF vectors as features — replacing simple linear softmax with a
nonlinear meta-learner.

Key difference from nb149 (113 models): using only top-10 reduces feature
dimensionality dramatically, preventing LGBM from over-fitting specific
feature combinations that cause test collapse.

Tests:
  A: LGBM_MAE on top-10 OOF (no assay), 5 seeds
  B: LGBM_MAE on top-10 OOF + 5 assay features, 5 seeds
  C: LGBM_MAE on top-5 OOF + 5 assay features
  D: CatBoost depth=8 on top-10 OOF + 5 assay features
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58

EXCLUDE_METAS = {
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb164_grand_v14", "nb170_grand_v15",
}

LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)

LGBM_MAE = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
    subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05, reg_lambda=0.0,
    objective="regression_l1", verbose=-1, n_jobs=4, random_state=42
)


def load_top_models(n_tr, y_tr, top_n, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in EXCLUDE_METAS: continue
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
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results[:top_n]


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


def cv_lgbm_multiseed(cfg, X_tr, y_tr, X_te, splits, label, seeds=(42, 123, 456, 789, 1234)):
    n_tr = len(y_tr)
    seed_oofs = []; seed_tes = []
    for seed in seeds:
        cfg_s = {**cfg, "random_state": seed}
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = lgb.train(cfg_s, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        m_f = lgb.train(cfg_s, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
        seed_oofs.append(oof_s); seed_tes.append(m_f.predict(X_te))
        print(f"    seed {seed}: RAE={rae(y_tr, oof_s):.4f}", flush=True)
    oof_ms = np.mean(seed_oofs, axis=0); te_ms = np.mean(seed_tes, axis=0)
    r_ms = rae(y_tr, oof_ms); ratio_ms = te_ms.std() / oof_ms.std()
    flag = "PASS" if ratio_ms >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label:55s}] RAE={r_ms:.4f}  ratio={ratio_ms:.2f}  [{flag}]", flush=True)
    return r_ms, oof_ms, te_ms, ratio_ms


def main():
    print("=== nb174: LGBM_MAE on Top-10 OOF Pool ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("\nTop-10 OOF models:")
    mods_10 = load_top_models(n_tr, y_tr, 10)
    for m in mods_10:
        print(f"  {m['stem']:55s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")

    oof10 = np.column_stack([m["oof"] for m in mods_10])
    te10  = np.column_stack([m["te"]  for m in mods_10])
    X_tr_10    = oof10
    X_te_10    = te10
    X_tr_10_a  = np.hstack([oof10, assay_oof])
    X_te_10_a  = np.hstack([te10, assay_te])

    results = {}

    # A: LGBM_MAE top-10 OOF only
    print(f"\n--- A: LGBM_MAE top-10 OOF only ({X_tr_10.shape[1]} features) ---")
    r_a, oof_a, te_a, ratio_a = cv_lgbm_multiseed(
        LGBM_MAE, X_tr_10, y_tr, X_te_10, splits, "LGBM_top10_oofonly")
    results["A_top10_no_assay"] = (r_a, oof_a, te_a, ratio_a)

    # B: LGBM_MAE top-10 OOF + assay
    print(f"\n--- B: LGBM_MAE top-10 OOF + assay ({X_tr_10_a.shape[1]} features) ---")
    r_b, oof_b, te_b, ratio_b = cv_lgbm_multiseed(
        LGBM_MAE, X_tr_10_a, y_tr, X_te_10_a, splits, "LGBM_top10_with_assay")
    results["B_top10_with_assay"] = (r_b, oof_b, te_b, ratio_b)

    # C: LGBM_MAE top-5 OOF + assay
    print(f"\nTop-5 OOF models:")
    mods_5 = load_top_models(n_tr, y_tr, 5)
    for m in mods_5:
        print(f"  {m['stem']:55s}  RAE={m['rae']:.4f}")
    oof5 = np.column_stack([m["oof"] for m in mods_5])
    te5  = np.column_stack([m["te"]  for m in mods_5])
    X_tr_5_a = np.hstack([oof5, assay_oof]); X_te_5_a = np.hstack([te5, assay_te])
    print(f"\n--- C: LGBM_MAE top-5 OOF + assay ({X_tr_5_a.shape[1]} features) ---")
    r_c, oof_c, te_c, ratio_c = cv_lgbm_multiseed(
        LGBM_MAE, X_tr_5_a, y_tr, X_te_5_a, splits, "LGBM_top5_with_assay")
    results["C_top5_with_assay"] = (r_c, oof_c, te_c, ratio_c)

    # D: CatBoost depth=8 on top-10 OOF + assay
    print(f"\n--- D: CatBoost depth=8 top-10 OOF + assay ---")
    CAT_D8 = dict(loss_function="MAE", eval_metric="MAE", iterations=600,
                  learning_rate=0.04, depth=8, l2_leaf_reg=5.0, subsample=0.8,
                  thread_count=4, allow_writing_files=False, verbose=0)
    oof_d_list = []; te_d_list = []
    for seed in [42, 123, 456, 789, 1234]:
        cfg_s = {**CAT_D8, "random_seed": seed}
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = CatBoostRegressor(**cfg_s)
            m.fit(X_tr_10_a[tr_idx], y_tr[tr_idx])
            oof_s[va_idx] = m.predict(X_tr_10_a[va_idx])
        m_f = CatBoostRegressor(**cfg_s)
        m_f.fit(X_tr_10_a, y_tr)
        oof_d_list.append(oof_s); te_d_list.append(m_f.predict(X_te_10_a))
        print(f"    seed {seed}: RAE={rae(y_tr, oof_s):.4f}", flush=True)
    oof_d = np.mean(oof_d_list, axis=0); te_d = np.mean(te_d_list, axis=0)
    r_d = rae(y_tr, oof_d); ratio_d = te_d.std() / oof_d.std()
    flag_d = "PASS" if ratio_d >= COLLAPSE_THRESH else "FAIL"
    print(f"  [CatBoost_top10] RAE={r_d:.4f}  ratio={ratio_d:.2f}  [{flag_d}]")
    results["D_cat_top10"] = (r_d, oof_d, te_d, ratio_d)

    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb164 grand v14: 0.3013, nb173 softmax blend: 0.3006)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb174_top10_lgbm.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb174_top10_lgbm.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "174_top10_lgbm.csv", index=False)
    print(f"Saved: submissions/174_top10_lgbm.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
