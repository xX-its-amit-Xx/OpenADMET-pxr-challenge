"""nb147 — OOF + RDKit Descriptors + Assay Meta-Stack.

nb141 ablation:
  OOF only (111):             RAE=0.3215
  OOF + assay (116):          RAE=0.3186  <- best
  OOF + str + assay (2381):   RAE=0.3379  <- WORSE

Structural features = Morgan FP (2048) + RDKit descriptors (217) = 2265.
Hypothesis: Morgan FP (sparse binary, 2048 bits, 91% of structural) is the noise source.
RDKit descriptors (217 continuous) may be neutral or helpful.

Test: OOF + RDKit-only (217) + assay (5) = 333 features.

Compare against:
  A: OOF + assay only (116)  — nb143/nb141-B baseline
  B: OOF + RDKit + assay (333)
  C: OOF + Morgan + assay (2164)  — to confirm Morgan is the culprit
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats

from pxr.data import load_train, load_test
from pxr.featurize import rdkit_desc, morgan, combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
COLLAPSE_THRESH = 0.58
LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
XGB_BEST = dict(
    n_estimators=800, max_depth=6, learning_rate=0.04, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.05, reg_lambda=0.5,
    tree_method="hist", device="cpu", verbosity=0, n_jobs=4, random_state=SEED
)


def load_model_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
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
    return results


def cv_xgb(params, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr[tr_idx], y_tr[tr_idx], verbose=False)
        oof[va_idx] = m.predict(X_tr[va_idx])
    m_full = xgb.XGBRegressor(**params)
    m_full.fit(X_tr, y_tr, verbose=False)
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    print(f"  [{label:50s}] RAE={r:.4f}  feats={X_tr.shape[1]}  ratio={ratio:.2f}", flush=True)
    return r, oof, te_pred, ratio


def main():
    print("=== nb147: OOF + RDKit Descriptors + Assay Meta-Stack ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

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

    print("Computing structural features...")
    X_rdkit = impute(rdkit_desc(tr["smiles"].tolist()))
    X_morgan = impute(morgan(tr["smiles"].tolist()))
    X_rdkit_te = impute(rdkit_desc(te["smiles"].tolist()))
    X_morgan_te = impute(morgan(te["smiles"].tolist()))
    X_str = impute(combined(tr["smiles"].tolist()))  # for aux OOF
    X_str_te = impute(combined(te["smiles"].tolist()))
    print(f"  RDKit: {X_rdkit.shape}  Morgan: {X_morgan.shape}")

    print("Loading OOF predictions...")
    models = load_model_oofs(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded")
    oof_all = np.column_stack([m["oof"] for m in models])
    te_all  = np.column_stack([m["te"]  for m in models])

    print("Aux OOF (assay features)...")
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

    results = {}
    print("\n=== Ablation: Feature Combinations ===")

    # A: OOF + assay only (Variant B baseline)
    r, oof, te_p, ratio = cv_xgb(XGB_BEST,
        np.hstack([oof_all, assay_oof]), y_tr,
        np.hstack([te_all,  assay_te]), splits, "A: OOF+assay (baseline)")
    results["A_oof_assay"] = (r, oof, te_p, ratio)

    # B: OOF + RDKit + assay
    r, oof, te_p, ratio = cv_xgb(XGB_BEST,
        np.hstack([oof_all, X_rdkit,  assay_oof]), y_tr,
        np.hstack([te_all,  X_rdkit_te, assay_te]), splits, "B: OOF+RDKit+assay")
    results["B_oof_rdkit_assay"] = (r, oof, te_p, ratio)

    # C: OOF + Morgan + assay
    r, oof, te_p, ratio = cv_xgb(XGB_BEST,
        np.hstack([oof_all, X_morgan,  assay_oof]), y_tr,
        np.hstack([te_all,  X_morgan_te, assay_te]), splits, "C: OOF+Morgan+assay")
    results["C_oof_morgan_assay"] = (r, oof, te_p, ratio)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, _ = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb147_oofrdkit_meta.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb147_oofrdkit_meta.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "147_oofrdkit_meta.csv", index=False)
    print(f"\nSaved: submissions/147_oofrdkit_meta.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
