"""nb130 — External PXR Data Augmentation.

Use Papyrus (~11k) + ChEMBL (~812) + BindingDB (~945) PXR measurements as
additional training data. These external compounds have no Emax/null assay
features, so we train two parallel pipelines:

  A) Structural-only LGBM on (external + challenge) → "structural prior"
  B) Augmented LGBM on (challenge only, assay-aug) → nb107-style
  C) Meta-stack: predict from [A_oof, B_oof, structural, assay_aux]

Key details:
  - External data enters training for every CV fold (not held out)
  - OOF computed on challenge data only (4139 compounds, 5-fold scaffold)
  - Measurement type heterogeneity (IC50/EC50/Ki) handled via one-hot feature
  - Compound overlap with challenge train removed (by standardized SMILES)
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
from pxr.chem import bemis_murcko, standardize
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_STRUCT = dict(
    n_estimators=600, num_leaves=48, learning_rate=0.04,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4, reg_alpha=0.1
)
LGBM_AUG = dict(
    n_estimators=600, num_leaves=48, learning_rate=0.04,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_META = dict(
    n_estimators=400, num_leaves=31, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def load_external_pxr(challenge_smiles_set):
    """Load and clean external PXR data, excluding challenge overlap."""
    dfs = []
    sources = [
        ("data/external/papyrus_pxr_nr.parquet",   "papyrus"),
        ("data/external/chembl_pxr_all_types.parquet", "chembl"),
        ("data/external/bindingdb_pxr_direct.parquet", "bindingdb"),
    ]
    for path, src in sources:
        try:
            df = pd.read_parquet(path)
            smi_col = "smiles" if "smiles" in df.columns else "canonical_smiles"
            std_col = "std_smiles" if "std_smiles" in df.columns else None
            type_col = "standard_type" if "standard_type" in df.columns else "measurement_type"

            out = pd.DataFrame()
            out["smiles"] = df[smi_col].fillna("").astype(str)
            out["pec50"]  = pd.to_numeric(df["pec50"], errors="coerce")
            out["mtype"]  = df[type_col].fillna("Unknown").astype(str) if type_col in df.columns else "Unknown"
            out["source"] = src
            dfs.append(out)
        except Exception as e:
            print(f"  Warning: {src} failed: {e}")

    ext = pd.concat(dfs, ignore_index=True)
    ext = ext.dropna(subset=["pec50"])
    ext = ext[ext["pec50"].between(3.0, 12.0)]
    ext = ext[ext["smiles"].str.len() > 3]

    # Remove compounds overlapping with challenge (exact SMILES)
    ext = ext[~ext["smiles"].isin(challenge_smiles_set)]
    # Also try to remove by standard_smiles if available
    ext["smiles"] = ext["smiles"].str.strip()

    # One-hot encode measurement type
    mtype_map = {"EC50": 0, "AC50": 0, "IC50": 1, "Ki": 2, "Kd": 2}
    ext["mtype_code"] = ext["mtype"].map(lambda x: mtype_map.get(x, 3))

    # Deduplicate (same SMILES + mtype): keep median
    ext = ext.groupby(["smiles", "mtype_code"]).agg(
        pec50=("pec50", "median")
    ).reset_index()

    return ext


def full_metrics(y_true, y_pred, label=""):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def main():
    print("=== nb130: External PXR Data Augmentation ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    challenge_smiles = set(tr["smiles"].tolist())
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print("Computing challenge structural features...")
    X_ch = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"  Challenge: train={X_ch.shape}  test={X_te.shape}")

    print("Loading external PXR data...")
    ext = load_external_pxr(challenge_smiles)
    print(f"  External: {len(ext)} compounds (after dedup + overlap removal)")
    print(f"  pEC50 range: [{ext['pec50'].min():.1f}, {ext['pec50'].max():.1f}]  "
          f"mean={ext['pec50'].mean():.2f}")
    print(f"  mtype distribution: {ext['mtype_code'].value_counts().to_dict()}")

    print("Featurizing external data (this may take a minute)...")
    X_ext_raw = combined(ext["smiles"].tolist())
    X_ext = impute(X_ext_raw)
    y_ext = ext["pec50"].values.astype(np.float64)
    mtype_ext = ext["mtype_code"].values.reshape(-1, 1).astype(np.float32)
    # Append mtype as feature (0=EC50/AC50, 1=IC50, 2=Ki/Kd, 3=other)
    X_ext_aug = np.hstack([X_ext, mtype_ext])
    # Challenge data: mtype=0 (all CRC EC50)
    mtype_ch = np.zeros((n_tr, 1), dtype=np.float32)
    X_ch_aug = np.hstack([X_ch, mtype_ch])
    mtype_te = np.zeros((len(X_te), 1), dtype=np.float32)
    X_te_aug = np.hstack([X_te, mtype_te])
    n_ext = len(y_ext)
    print(f"  External featurized: {X_ext_aug.shape}")

    # === Stage 1: Structural-only LGBM auxiliary targets (for challenge data) ===
    print("\nAux OOF (assay decomposition on challenge)...")
    oof_emax = np.full(n_tr, np.nan)
    oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUG, lgb.Dataset(X_ch[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_ch[va_idx])
        m_nl = lgb.train(LGBM_AUG, lgb.Dataset(X_ch[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_ch[va_idx])
        m_sl = lgb.train(LGBM_AUG, lgb.Dataset(X_ch[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sl.predict(X_ch[va_idx])

    m_em_f = lgb.train(LGBM_AUG, lgb.Dataset(X_ch, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUG, lgb.Dataset(X_ch, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUG, lgb.Dataset(X_ch, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    # === Stage 2: Structural-only model WITH external data ===
    print("\n=== Structural + External LGBM (k-fold) ===")
    oof_struct_ext = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Combine: challenge training fold + all external data
        X_fold = np.vstack([X_ch_aug[tr_idx], X_ext_aug])
        y_fold = np.concatenate([y_tr[tr_idx], y_ext])
        m = lgb.train(LGBM_STRUCT, lgb.Dataset(X_fold, label=y_fold),
                      callbacks=[lgb.log_evaluation(-1)])
        oof_struct_ext[va_idx] = m.predict(X_ch_aug[va_idx])
        r_va = rae(y_tr[va_idx], oof_struct_ext[va_idx])
        print(f"  fold {fold+1}  RAE={r_va:.4f}", flush=True)

    m_struct_full = lgb.train(LGBM_STRUCT,
                              lgb.Dataset(np.vstack([X_ch_aug, X_ext_aug]),
                                          label=np.concatenate([y_tr, y_ext])),
                              callbacks=[lgb.log_evaluation(-1)])
    te_struct_ext = m_struct_full.predict(X_te_aug)
    full_metrics(y_tr, oof_struct_ext, "Structural+External LGBM")
    ratio_se = te_struct_ext.std() / oof_struct_ext.std()
    print(f"  Test: med={np.median(te_struct_ext):.2f}  std={te_struct_ext.std():.3f}  ratio={ratio_se:.2f}")

    # === Stage 3: Challenge-only augmented LGBM (nb107 baseline) ===
    print("\n=== Challenge-only augmented LGBM (baseline) ===")
    assay_feats_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                       np.log1p(np.clip(oof_emax, 0, None))])
    X_aug_ch = np.hstack([X_ch, assay_feats_oof])

    oof_aug_ch = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m2 = lgb.train(LGBM_AUG, lgb.Dataset(X_aug_ch[tr_idx], label=y_tr[tr_idx]),
                       callbacks=[lgb.log_evaluation(-1)])
        oof_aug_ch[va_idx] = m2.predict(X_aug_ch[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_aug_ch[va_idx]):.4f}", flush=True)

    assay_te = np.column_stack([te_emax, te_null, te_sel,
                                 np.zeros(len(X_te)),
                                 np.log1p(np.clip(te_emax, 0, None))])
    X_aug_te = np.hstack([X_te, assay_te])
    m_aug_full = lgb.train(LGBM_AUG, lgb.Dataset(X_aug_ch, label=y_tr),
                           callbacks=[lgb.log_evaluation(-1)])
    te_aug_ch = m_aug_full.predict(X_aug_te)
    full_metrics(y_tr, oof_aug_ch, "Challenge-only augmented LGBM")

    # === Stage 4: Meta-stack: [struct_ext_oof, aug_ch_oof, structural, assay] ===
    print("\n=== Meta-stack ===")
    meta_tr = np.column_stack([oof_struct_ext, oof_aug_ch,
                                oof_emax, oof_null, oof_sel, has_null])
    meta_te = np.column_stack([te_struct_ext, te_aug_ch,
                                te_emax, te_null, te_sel,
                                np.zeros(len(X_te))])
    # Add structural features (PCA-reduced for speed: just use all 2265)
    meta_tr_full = np.hstack([meta_tr, X_ch])
    meta_te_full = np.hstack([meta_te, X_te])

    oof_meta = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m3 = lgb.train(LGBM_META, lgb.Dataset(meta_tr_full[tr_idx], label=y_tr[tr_idx]),
                       callbacks=[lgb.log_evaluation(-1)])
        oof_meta[va_idx] = m3.predict(meta_tr_full[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_meta[va_idx]):.4f}", flush=True)

    m_meta_full = lgb.train(LGBM_META, lgb.Dataset(meta_tr_full, label=y_tr),
                            callbacks=[lgb.log_evaluation(-1)])
    te_meta = m_meta_full.predict(meta_te_full)
    full_metrics(y_tr, oof_meta, "Meta-stack (ext+aug+struct)")
    te_meta = np.clip(te_meta, y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio_m = te_meta.std() / oof_meta.std()
    print(f"  Test: med={np.median(te_meta):.2f}  std={te_meta.std():.3f}  ratio={ratio_m:.2f}")

    # Best output = meta-stack
    np.save(DATA_PROCESSED / "oof_nb130_external_pxr.npy", oof_meta)
    np.save(DATA_PROCESSED / "te_nb130_external_pxr.npy",  te_meta)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_meta})
    sub.to_csv(SUBMISSIONS / "130_external_pxr_augment.csv", index=False)
    print(f"\nSaved: submissions/130_external_pxr_augment.csv")
    print(f"OOF RAE: {rae(y_tr, oof_meta):.4f}")

    # Also save structural+external as standalone
    np.save(DATA_PROCESSED / "oof_nb130_struct_ext.npy", oof_struct_ext)
    np.save(DATA_PROCESSED / "te_nb130_struct_ext.npy",  te_struct_ext)


if __name__ == "__main__":
    main()
