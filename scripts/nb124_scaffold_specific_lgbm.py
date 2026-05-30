"""nb124 — Scaffold-Specific LGBM + Universal Model Blend.

Hypothesis: scaffold-specific models that are trained only on compounds
sharing the same Murcko scaffold family can better learn scaffold-specific
SAR patterns. But since most scaffolds have < 20 compounds, use a
hierarchical approach:

1. For each scaffold family with >= 10 members: train a scaffold-specific
   model using full training + local scaffold data (like a custom prior)
2. For test compounds: if they have a matching scaffold, use the scaffold-
   specific model prediction + (1-w)*universal model
3. For test compounds without matching scaffold: use universal model only

This is essentially local QSAR vs global QSAR. The universal model is nb109.
The local models use a narrow set of features and strong regularization.

Implementation:
- Universal base: nb109 OOF/TE predictions
- Scaffold groups with >= 10 members in train: ~49 groups
- For each group: retrain LGBM using only group members
- Scaffold-specific weight w: function of group size (bigger = higher weight)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from collections import defaultdict

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
MIN_SCAFFOLD_SIZE = 8  # minimum to train a local model
MAX_LOCAL_WEIGHT = 0.3  # maximum weight for local model

LGBM_LOCAL = dict(
    n_estimators=200, num_leaves=16, learning_rate=0.05,
    min_child_samples=3, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=2
)
LGBM_GLOBAL = dict(
    n_estimators=1200, num_leaves=64, learning_rate=0.03,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def tanimoto_maxsim(fps_q, fps_db, batch=256):
    fps_q = fps_q.astype(np.float32); fps_db = fps_db.astype(np.float32)
    db_norm = (fps_db ** 2).sum(axis=1)
    n_q = len(fps_q); max_sims = np.zeros(n_q, dtype=np.float32)
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        q_f = fps_q[start:end]
        q_norm = (q_f ** 2).sum(axis=1, keepdims=True)
        dot = q_f @ fps_db.T
        union = q_norm + db_norm[None, :] - dot
        max_sims[start:end] = np.where(union > 0, dot / union, 0.0).max(axis=1)
    return max_sims


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


def main():
    print("=== nb124: Scaffold-Specific LGBM + Universal Model Blend ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds_tr = tr["smiles"].map(bemis_murcko).tolist()
    scaffolds_te = te["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds_tr, N_FOLDS, SEED)

    # Load universal base (nb109)
    oof_uni = np.load(DATA_PROCESSED / "oof_nb109_deep_meta_stack.npy").astype(np.float64)
    te_uni  = np.load(DATA_PROCESSED / "te_nb109_deep_meta_stack.npy").astype(np.float64)
    print(f"Universal (nb109): OOF RAE={rae(y_tr, oof_uni):.4f}  "
          f"te_std={te_uni.std():.3f}")

    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print("Computing features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    fps_tr = morgan_fp_batch(tr["smiles"].tolist())
    fps_te = morgan_fp_batch(te["smiles"].tolist())

    # Scaffold group assignments
    sc_to_idx_tr = defaultdict(list)
    for i, sc in enumerate(scaffolds_tr):
        sc_to_idx_tr[sc].append(i)
    sc_groups = {sc: idx for sc, idx in sc_to_idx_tr.items()
                 if len(idx) >= MIN_SCAFFOLD_SIZE}
    print(f"\nScaffold groups with >= {MIN_SCAFFOLD_SIZE} members: {len(sc_groups)}")
    total_covered = sum(len(v) for v in sc_groups.values())
    print(f"Train compounds in groups: {total_covered}/{n_tr} ({100*total_covered/n_tr:.1f}%)")

    te_sc_match = {i: scaffolds_te[i] if scaffolds_te[i] in sc_groups else None
                   for i in range(len(te["smiles"]))}
    te_matched = sum(1 for v in te_sc_match.values() if v is not None)
    print(f"Test compounds with matching group: {te_matched}/{len(te['smiles'])} "
          f"({100*te_matched/len(te['smiles']):.1f}%)")

    # Augmented features for global use
    print("\nBuilding augmented feature matrix...")
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
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

    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    # Global augmented model (for smooth baseline in each scaffold)
    meta_oofs, meta_tes = [], []
    for stem in ["nb107_assay_decomp", "nb111_selectivity_primary", "nb99_sc_bio_fp"]:
        oof, te_m = load_meta(stem, n_tr)
        if oof is not None:
            oof_f = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te_f  = np.where(np.isfinite(te_m), te_m, np.nanmean(te_m))
            if te_f.std() / oof_f.std() >= 0.55:
                meta_oofs.append(oof_f); meta_tes.append(te_f)

    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                  np.log1p(np.clip(oof_emax, 0, None))] + meta_oofs)
    assay_te  = np.column_stack([te_emax, te_null, te_sel, np.zeros(len(X_te)),
                                  np.log1p(np.clip(te_emax, 0, None))] + meta_tes)
    X_tr_aug = np.hstack([X_tr, assay_oof, oof_uni.reshape(-1, 1)])
    X_te_aug = np.hstack([X_te, assay_te,  te_uni.reshape(-1, 1)])
    print(f"Augmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # Scaffold CV: compute local model corrections
    print("\n=== Scaffold CV: local model blend ===")
    oof_local = oof_uni.copy()  # start from universal predictions
    for fold, (tr_idx, va_idx) in enumerate(splits):
        sc_in_fold = {sc: [i for i in idx if i in set(va_idx)]
                      for sc, idx in sc_groups.items()}
        for sc, va_sc_idx in sc_in_fold.items():
            if len(va_sc_idx) < 3:
                continue
            # Train: all other training compounds in this scaffold group
            tr_sc_idx = [i for i in sc_groups[sc] if i not in set(va_idx)]
            if len(tr_sc_idx) < MIN_SCAFFOLD_SIZE:
                continue
            # Local model: trained on scaffold group + full training fold
            combined_tr = list(set(list(tr_idx) + tr_sc_idx))
            m_local = lgb.train(LGBM_LOCAL,
                                lgb.Dataset(X_tr_aug[combined_tr], label=y_tr[combined_tr]),
                                callbacks=[lgb.log_evaluation(-1)])
            local_pred = m_local.predict(X_tr_aug[va_sc_idx])
            # Local weight: sqrt(n) / (sqrt(n) + sqrt(20))
            n_local = len(tr_sc_idx)
            local_w = min(MAX_LOCAL_WEIGHT, np.sqrt(n_local) / (np.sqrt(n_local) + np.sqrt(30)))
            oof_local[va_sc_idx] = (1 - local_w) * oof_uni[va_sc_idx] + local_w * local_pred

        va_rae = rae(y_tr[va_idx], oof_local[va_idx])
        uni_rae = rae(y_tr[va_idx], oof_uni[va_idx])
        print(f"  fold {fold+1}  RAE={va_rae:.4f} (universal={uni_rae:.4f})", flush=True)

    print(f"\nScaffold-specific blend OOF RAE: {rae(y_tr, oof_local):.4f}")
    print(f"Universal (nb109) OOF RAE:       {rae(y_tr, oof_uni):.4f}")

    # Final test predictions
    # Train global model on full data
    m_global = lgb.train(LGBM_GLOBAL, lgb.Dataset(X_tr_aug, label=y_tr),
                         callbacks=[lgb.log_evaluation(-1)])
    te_global = m_global.predict(X_te_aug)

    # Apply scaffold-specific local corrections to test
    te_blended = te_uni.copy()
    for i in range(len(te["smiles"])):
        sc = te_sc_match[i]
        if sc is None:
            continue
        sc_idx = sc_groups[sc]
        n_local = len(sc_idx)
        m_local_f = lgb.train(LGBM_LOCAL,
                               lgb.Dataset(X_tr_aug[sc_idx], label=y_tr[sc_idx]),
                               callbacks=[lgb.log_evaluation(-1)])
        local_pred = m_local_f.predict(X_te_aug[[i]])[0]
        local_w = min(MAX_LOCAL_WEIGHT, np.sqrt(n_local) / (np.sqrt(n_local) + np.sqrt(30)))
        te_blended[i] = (1 - local_w) * te_uni[i] + local_w * local_pred

    te_blended = np.clip(te_blended, y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"\nTest blended: min={te_blended.min():.2f}  med={np.median(te_blended):.2f}  "
          f"max={te_blended.max():.2f}  std={te_blended.std():.3f}")

    np.save(DATA_PROCESSED / "oof_nb124_scaffold_specific.npy", oof_local)
    np.save(DATA_PROCESSED / "te_nb124_scaffold_specific.npy",  te_blended)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_blended})
    sub.to_csv(SUBMISSIONS / "124_scaffold_specific_blend.csv", index=False)
    print(f"\nSaved: submissions/124_scaffold_specific_blend.csv")
    print(f"OOF RAE: {rae(y_tr, oof_local):.4f}")


if __name__ == "__main__":
    main()
