"""nb165 — Multi-Seed Ensemble of nb162 Config C.

nb162 config C (base 110 models + 6 meta-stack anchors, 121 features) achieved
OOF RAE=0.3071, ratio=0.58 PASS. This nearly ties nb149's 0.3069.

Goal: reduce variance via multi-seed averaging (5-10 seeds).
Expected: OOF RAE ≈ 0.3060-0.3070, possibly improving over nb149's 0.3069.

Also tests: extended hyperparameter variants around config C's setup.
  v1: same config, 5 seeds
  v2: n_estimators=1000, lr=0.025 (slightly more capacity)
  v3: num_leaves=95, same other params (between 63 and 127)
  v4: min_child_samples=3 (allow smaller leaves)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58

BASE_EXCLUDE = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12", "nb154_lgbm_mae_filtered",
    "nb155_grand_v13", "nb156_catboost_mae", "nb157_optuna_lgbm_mae",
    "nb158_collapse_fix", "nb159_variance_scaled_ensemble", "nb160_pca_meta",
    "nb161_neural_meta", "nb162_mixed_pool", "nb163_lgbm_colsample_low",
    "nb164_grand_v14", "nb165_multiseed_162c",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}

ANCHOR_STEMS = [
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb136_xgb_meta", "nb134_grand_v9", "nb141_xgb_ablation",
]

LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)

BASE_MAE_CFG = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
    subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
    objective="regression_l1", verbose=-1, n_jobs=4, random_state=42
)


def load_stem(stem, n_tr):
    oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
    for te_pref in ("te_", "te_oof_"):
        te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
        if te_p.exists(): break
    if not oof_p.exists() or not te_p.exists():
        return None
    oof = np.load(oof_p).astype(np.float64)
    te  = np.load(te_p).astype(np.float64)
    if oof.ndim == 2: oof = oof[:, 0]
    if te.ndim == 2:  te  = te[:, 0]
    if len(oof) != n_tr: return None
    oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
    te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
    return dict(stem=stem, oof=oof, te=te)


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
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
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


def build_mixed_pool(base_mods, anchor_stems, n_tr, assay_oof, assay_te):
    anchors = []
    for s in anchor_stems:
        data = load_stem(s, n_tr)
        if data is not None:
            anchors.append(data)
    oof_parts = [m["oof"] for m in base_mods] + [a["oof"] for a in anchors]
    te_parts  = [m["te"]  for m in base_mods] + [a["te"]  for a in anchors]
    oof_mat = np.column_stack(oof_parts)
    te_mat  = np.column_stack(te_parts)
    X_tr = np.hstack([oof_mat, assay_oof])
    X_te = np.hstack([te_mat, assay_te])
    return X_tr, X_te


def train_multi_seed(cfg, X_tr, y_tr, X_te, splits, seeds, label):
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
        r_s = rae(y_tr, oof_s)
        seed_oofs.append(oof_s); seed_tes.append(m_f.predict(X_te))
        print(f"  seed {seed}: RAE={r_s:.4f}", flush=True)
    oof_avg = np.mean(seed_oofs, axis=0)
    te_avg  = np.mean(seed_tes, axis=0)
    r = rae(y_tr, oof_avg); ratio = te_avg.std() / oof_avg.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label:55s}] RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]", flush=True)
    return r, oof_avg, te_avg, ratio


def main():
    print("=== nb165: Multi-Seed nb162-C + Hyper Variants ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("Loading base models (exclude meta-stacks)...")
    base_mods = load_base_oofs(n_tr, y_tr, exclude_stems=BASE_EXCLUDE)
    print(f"  {len(base_mods)} base models loaded")

    X_tr, X_te = build_mixed_pool(base_mods, ANCHOR_STEMS, n_tr, assay_oof, assay_te)
    print(f"  Mixed pool features: {X_tr.shape} (base+6 anchors+assay)")

    results = {}

    # V1: Multi-seed (5 seeds) with base config (same as nb162-C)
    print("\n--- V1: Multi-seed 5x (base config = nb162-C) ---")
    r, oof, te_pred, ratio = train_multi_seed(
        BASE_MAE_CFG, X_tr, y_tr, X_te, splits,
        seeds=[42, 123, 456, 789, 1234], label="V1: 5-seed base config")
    results["V1_5seed"] = (r, oof, te_pred, ratio)

    # V2: More trees (n=1000, lr=0.025)
    print("\n--- V2: More capacity (n=1000, lr=0.025), 5 seeds ---")
    cfg_v2 = {**BASE_MAE_CFG, "n_estimators": 1000, "learning_rate": 0.025}
    r, oof, te_pred, ratio = train_multi_seed(
        cfg_v2, X_tr, y_tr, X_te, splits,
        seeds=[42, 123, 456, 789, 1234], label="V2: 5-seed n=1000,lr=0.025")
    results["V2_1000trees_5seed"] = (r, oof, te_pred, ratio)

    # V3: num_leaves=95 (more leaves, less than 127)
    print("\n--- V3: num_leaves=95, 5 seeds ---")
    cfg_v3 = {**BASE_MAE_CFG, "num_leaves": 95}
    r, oof, te_pred, ratio = train_multi_seed(
        cfg_v3, X_tr, y_tr, X_te, splits,
        seeds=[42, 123, 456, 789, 1234], label="V3: 5-seed leaves=95")
    results["V3_leaves95_5seed"] = (r, oof, te_pred, ratio)

    # V4: min_child_samples=3 (allow smaller leaves)
    print("\n--- V4: min_child_samples=3, 5 seeds ---")
    cfg_v4 = {**BASE_MAE_CFG, "min_child_samples": 3}
    r, oof, te_pred, ratio = train_multi_seed(
        cfg_v4, X_tr, y_tr, X_te, splits,
        seeds=[42, 123, 456, 789, 1234], label="V4: 5-seed min_child=3")
    results["V4_minchild3_5seed"] = (r, oof, te_pred, ratio)

    # V5: 10 seeds (most thorough)
    print("\n--- V5: 10 seeds (most thorough) ---")
    r, oof, te_pred, ratio = train_multi_seed(
        BASE_MAE_CFG, X_tr, y_tr, X_te, splits,
        seeds=[42, 123, 456, 789, 1234, 2468, 3691, 4812, 5935, 7058],
        label="V5: 10-seed base config")
    results["V5_10seed"] = (r, oof, te_pred, ratio)

    # V6: Ensemble of V1 + V2 + V3
    if all(k in results for k in ["V1_5seed", "V2_1000trees_5seed", "V3_leaves95_5seed"]):
        oof_v6 = np.mean([results["V1_5seed"][1], results["V2_1000trees_5seed"][1],
                          results["V3_leaves95_5seed"][1]], axis=0)
        te_v6  = np.mean([results["V1_5seed"][2], results["V2_1000trees_5seed"][2],
                          results["V3_leaves95_5seed"][2]], axis=0)
        r_v6 = rae(y_tr, oof_v6); ratio_v6 = te_v6.std() / oof_v6.std()
        flag = "PASS" if ratio_v6 >= COLLAPSE_THRESH else "FAIL"
        print(f"\n  [V6: V1+V2+V3 ensemble] RAE={r_v6:.4f}  ratio={ratio_v6:.2f}  [{flag}]")
        results["V6_ensemble_v1v2v3"] = (r_v6, oof_v6, te_v6, ratio_v6)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:55s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb149: 0.3069, nb162-C: 0.3071, nb155 ensemble: 0.3044)")
    if best_r < 0.3069: print("*** NEW BEST SINGLE MODEL! ***")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb165_multiseed_162c.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb165_multiseed_162c.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "165_multiseed_162c.csv", index=False)
    print(f"\nSaved: submissions/165_multiseed_162c.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
