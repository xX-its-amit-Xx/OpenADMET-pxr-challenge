"""nb801 — Huber LGBM with EXPANDED assay-decomp feature set.

Same base recipe as nb120/nb800 (Huber alpha=2.0, LightGBM, scaffold 5-fold,
meta-OOF gating by std-ratio >= 0.55), but adds 6 new train-only assay
features to the augmented matrix. Train-only features are leak-prone, so
each must be modeled with an auxiliary LGBM and predicted onto the
validation fold (and the test set) — never use the raw label directly on
the train side without OOF cross-fit.

New features:
  1. counter_assay_disagreement  (pec50_pxr - pec50_null)
  2. single_conc_log2FC          (mean log2_fc_estimate per compound)
  3. single_conc_fdr             (mean fdr_bh per compound)
  4. single_conc_concentration_M (mean assay concentration per compound)
  5. emax_se                     (Emax standard error from train)
  6. emax_high                   (binary: Emax.vs.pos.ctrl > 1.0)

For each feature we:
  - Compute the raw value on train (where available)
  - Median-impute missing values
  - Train an auxiliary LGBM per-fold to predict the value from Morgan+RDKit
  - Use the OOF auxiliary prediction on val, full-train aux on test

Target: in_RAE < 0.62 on the 253-row unblind (beat chemprop_aux 0.6216).
"""
import os, sys, warnings, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
HUBER_ALPHA = 2.0

BASE_PARAMS = dict(
    n_estimators=1500, num_leaves=64, learning_rate=0.03,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def load_meta(stem, n_tr):
    for op in ("oof_", ""):
        for tp in ("te_", "te_oof_"):
            of = DATA_PROCESSED / f"{op}{stem}.npy"
            tf = DATA_PROCESSED / f"{tp}{stem}.npy"
            if of.exists() and tf.exists():
                oof = np.load(of); te = np.load(tf)
                if oof.ndim == 2: oof = oof[:, 0]
                if te.ndim == 2:  te  = te[:, 0]
                if len(oof) == n_tr: return oof, te
    return None, None


def fold_aux_predict(X_tr, X_te, y_aux, splits, label):
    """OOF + full-train auxiliary LGBM predictions for one feature."""
    n_tr = len(y_aux)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=y_aux[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
    m_full = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=y_aux),
                       callbacks=[lgb.log_evaluation(-1)])
    te_pred = m_full.predict(X_te)
    print(f"  aux[{label:30s}]  oof_std={oof.std():.3f}  te_std={te_pred.std():.3f}")
    return oof, te_pred


def main():
    print("=== nb801: Huber LGBM with EXPANDED assay-decomp ===\n")

    unblind_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unblind = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unblind_idx) == len(y_unblind) == 253, "unblind shapes off"

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    raw_sc      = pd.read_csv("data/raw/pxr-challenge_single_concentration_TRAIN.csv")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ---- ORIGINAL assay-decomp labels ----
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_se_col = "Emax.vs.pos.ctrl_std.error (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_median  = np.nanmedian(pec50_null)
    null_imputed = np.where(np.isnan(pec50_null), null_median, pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    # ---- NEW: counter_assay_disagreement (= -selectivity sign-flipped; explicit) ----
    counter_disagree = y_tr - null_imputed  # same as selectivity; keep as a clean alias
    # (kept as a separate feature for explicit naming; aux model trained independently)

    # ---- NEW: single_conc aggregates (mean across replicates per molecule) ----
    sc_agg = raw_sc.groupby("Molecule Name").agg(
        sc_log2fc=("log2_fc_estimate", "mean"),
        sc_fdr=("fdr_bh", "mean"),
        sc_conc=("concentration_M", "mean"),
    ).reset_index()
    sc_map_log2fc = dict(zip(sc_agg["Molecule Name"], sc_agg["sc_log2fc"]))
    sc_map_fdr    = dict(zip(sc_agg["Molecule Name"], sc_agg["sc_fdr"]))
    sc_map_conc   = dict(zip(sc_agg["Molecule Name"], sc_agg["sc_conc"]))

    sc_log2fc_raw = np.array([sc_map_log2fc.get(n, np.nan) for n in mol_names])
    sc_fdr_raw    = np.array([sc_map_fdr.get(n,    np.nan) for n in mol_names])
    sc_conc_raw   = np.array([sc_map_conc.get(n,   np.nan) for n in mol_names])
    sc_log2fc_imp = np.where(np.isnan(sc_log2fc_raw), np.nanmedian(sc_log2fc_raw), sc_log2fc_raw)
    sc_fdr_imp    = np.where(np.isnan(sc_fdr_raw),    np.nanmedian(sc_fdr_raw),    sc_fdr_raw)
    sc_conc_imp   = np.where(np.isnan(sc_conc_raw),   np.nanmedian(sc_conc_raw),   sc_conc_raw)
    sc_conc_log   = np.log10(np.clip(sc_conc_imp, 1e-10, None))  # log10(molarity)

    n_with_sc = int((~np.isnan(sc_log2fc_raw)).sum())
    print(f"  single-conc coverage: {n_with_sc}/{n_tr} ({100*n_with_sc/n_tr:.1f}%)")

    # ---- NEW: emax_se and emax_high ----
    emax_se_raw = raw_train[emax_se_col].values.astype(np.float64)
    emax_se_imp = np.where(np.isnan(emax_se_raw), np.nanmedian(emax_se_raw), emax_se_raw)
    emax_se_log = np.log10(np.clip(emax_se_imp, 1e-4, None))
    emax_high   = (emax_raw > 1.0).astype(np.float32)  # >100% baseline

    print("Computing structural features (Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # ---- Meta-OOFs gated by std-ratio >= 0.55 ----
    meta_candidates = [
        "nb107_assay_decomp", "nb109_deep_meta_stack", "nb101_delta_base",
        "nb99_sc_bio_fp", "grand_v6b", "lgbm_tuned", "catboost",
    ]
    meta_oofs, meta_tes, meta_names = [], [], []
    for stem in meta_candidates:
        oof, te_m = load_meta(stem, n_tr)
        if oof is not None:
            oof_f = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te_f  = np.where(np.isfinite(te_m), te_m, np.nanmean(te_m))
            ratio = te_f.std() / oof_f.std() if oof_f.std() > 0 else 0
            if ratio >= 0.55:
                meta_oofs.append(oof_f); meta_tes.append(te_f)
                meta_names.append(stem)
                print(f"  meta KEEP {stem}: RAE={rae(y_tr, oof_f):.4f}  ratio={ratio:.2f}")
            else:
                print(f"  meta DROP {stem}: ratio={ratio:.2f}")

    # ---- Stage 1: ALL auxiliary OOFs (original 3 + 6 new) ----
    print("\nStage 1: auxiliary OOF for assay-decomp features...")
    # original 3
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
    fold_aux = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_tr[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_tr[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx]  = m_sl.predict(X_tr[va_idx])
        fold_aux.append((tr_idx, va_idx,
                         oof_emax[va_idx].copy(), oof_null[va_idx].copy(),
                         oof_sel[va_idx].copy()))
    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    # NEW 6 auxiliary OOFs
    print("\nStage 1b: NEW auxiliary OOFs (6 expanded features)...")
    oof_disagree, te_disagree = fold_aux_predict(X_tr, X_te, counter_disagree, splits, "counter_disagree")
    oof_sclog2,   te_sclog2   = fold_aux_predict(X_tr, X_te, sc_log2fc_imp,    splits, "sc_log2FC")
    oof_scfdr,    te_scfdr    = fold_aux_predict(X_tr, X_te, sc_fdr_imp,       splits, "sc_fdr")
    oof_scconc,   te_scconc   = fold_aux_predict(X_tr, X_te, sc_conc_log,      splits, "sc_conc_log")
    oof_emaxse,   te_emaxse   = fold_aux_predict(X_tr, X_te, emax_se_log,      splits, "emax_se_log")
    oof_emaxhi,   te_emaxhi   = fold_aux_predict(X_tr, X_te, emax_high.astype(np.float64), splits, "emax_high")

    # ---- Build augmented train/test matrix ----
    # original 5: emax, null, sel, has_null, log1p(emax)
    # meta_oofs (gated)
    # NEW 6: disagree, sc_log2, sc_fdr, sc_conc_log, emax_se_log, emax_high
    assay_oof = np.column_stack([
        oof_emax, oof_null, oof_sel, has_null, np.log1p(np.clip(oof_emax, 0, None)),
        oof_disagree, oof_sclog2, oof_scfdr, oof_scconc, oof_emaxse, oof_emaxhi,
    ] + meta_oofs)
    assay_te = np.column_stack([
        te_emax, te_null, te_sel, np.zeros(len(X_te)), np.log1p(np.clip(te_emax, 0, None)),
        te_disagree, te_sclog2, te_scfdr, te_scconc, te_emaxse, te_emaxhi,
    ] + meta_tes)
    X_tr_aug = np.hstack([X_tr, assay_oof])
    X_te_aug = np.hstack([X_te, assay_te])
    print(f"\nAugmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")
    print(f"  (5 original + 6 new + {len(meta_oofs)} meta-OOFs = {5+6+len(meta_oofs)} aug cols)")

    # ---- Stage 2: Huber LGBM (alpha=2.0) on augmented matrix ----
    print(f"\n=== Stage 2: Huber alpha={HUBER_ALPHA} ===")
    params = dict(BASE_PARAMS, objective="huber", alpha=HUBER_ALPHA)
    oof = np.full(n_tr, np.nan)

    for fold, (tr_idx, va_idx, em_va, nl_va, sl_va) in enumerate(fold_aux):
        # Reuse cached fold aux preds for original 3.
        # For NEW 6: oof_disagree[va_idx] etc. are already OOF (computed via splits above).
        va_assay = np.column_stack([
            em_va, nl_va, sl_va, has_null[va_idx], np.log1p(np.clip(em_va, 0, None)),
            oof_disagree[va_idx], oof_sclog2[va_idx], oof_scfdr[va_idx],
            oof_scconc[va_idx],   oof_emaxse[va_idx], oof_emaxhi[va_idx],
        ] + [o[va_idx] for o in meta_oofs])
        X_va = np.hstack([X_tr[va_idx], va_assay])

        m = lgb.train(params, lgb.Dataset(X_tr_aug[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(X_va, label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(80, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_va)
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    oof_rae = rae(y_tr, oof)
    print(f"\n  OOF RAE = {oof_rae:.4f}")

    # ---- Final model on full train ----
    m_final = lgb.train(dict(params, n_estimators=1000),
                        lgb.Dataset(X_tr_aug, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te_aug),
                       y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = te_preds.std() / oof.std()
    in_r = in_rae(y_unblind, te_preds[unblind_idx])
    print(f"\n  TEST  med={np.median(te_preds):.3f}  std={te_preds.std():.3f}  ratio={ratio:.2f}")
    print(f"  in_RAE(253) = {in_r:.4f}   (target < 0.6216)")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / "oof_nb801.npy", oof)
    np.save(DATA_PROCESSED / "te_nb801.npy",  te_preds)

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_preds,
    })
    sub_path = SUBMISSIONS / "nb801_huber_assay_decomp_plus.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved te_nb801.npy and {sub_path}")

    summary = {
        "oof_rae": float(oof_rae),
        "in_rae": float(in_r),
        "te_std": float(te_preds.std()),
        "oof_std": float(oof.std()),
        "ratio": float(ratio),
        "n_new_features": 6,
        "n_aug_cols": int(assay_oof.shape[1]),
        "n_meta_kept": len(meta_oofs),
        "meta_kept": meta_names,
        "beats_chemprop_aux_0_6216": bool(in_r < 0.6216),
    }
    with open(DATA_PROCESSED / "nb801_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {DATA_PROCESSED/'nb801_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
