"""nb800 — Huber LGBM variants with new alphas {0.3, 0.7, 1.5, 3.0}.

Replicates the nb120 Huber recipe (Morgan + RDKit + assay-decomp + meta-OOF
augmentation gated by std-ratio >= 0.55, scaffold 5-fold CV, fold-rebuilt
auxiliary models) and trains four additional Huber variants. Each variant is
saved as te_nb800_huber_{alpha}.npy with matching OOF arrays and submission CSVs.

Also evaluates:
  * 4-way mean ensemble across the new alphas: te_nb800_huber_ens
  * 7-way mean ensemble combining new + nb120 originals (0.5/1.0/2.0)

Each variant + ensemble is scored by in-sample RAE on the 253 Phase-1 unblind
labels (data/processed/nb472_unblind_idx.npy, _audit_unblind_y.npy).
"""
import os, sys, warnings, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5

NEW_ALPHAS = [0.3, 0.7, 1.5, 3.0]
OLD_ALPHAS = [0.5, 1.0, 2.0]   # already in data/processed (te_nb120_huber_*)

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


def alpha_tag(a):
    return f"{a}".replace(".", "_")


def main():
    print("=== nb800: Huber LGBM variants {0.3, 0.7, 1.5, 3.0} ===\n")

    # ---- Load truth + unblind index for in_RAE ----
    unblind_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unblind = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unblind_idx) == len(y_unblind) == 253, "unblind shapes off"

    # ---- Raw + train splits ----
    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ---- Assay decomposition labels ----
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_median  = np.nanmedian(pec50_null)
    null_imputed = np.where(np.isnan(pec50_null), null_median, pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print("Computing structural features (Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # ---- Filtered meta-OOFs (std-ratio >= 0.55) ----
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

    # ---- Stage 1: auxiliary OOF for assay-decomp features ----
    print("\nStage 1: full-train auxiliary OOF for test-side aug...")
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
        # cache per-fold aux predictions on va_idx for reuse below
        fold_aux.append((tr_idx, va_idx,
                         oof_emax[va_idx].copy(), oof_null[va_idx].copy(), oof_sel[va_idx].copy()))

    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    # train-side augmented matrix
    assay_oof = np.column_stack([
        oof_emax, oof_null, oof_sel, has_null,
        np.log1p(np.clip(oof_emax, 0, None))
    ] + meta_oofs)
    assay_te = np.column_stack([
        te_emax, te_null, te_sel, np.zeros(len(X_te)),
        np.log1p(np.clip(te_emax, 0, None))
    ] + meta_tes)
    X_tr_aug = np.hstack([X_tr, assay_oof])
    X_te_aug = np.hstack([X_te, assay_te])
    print(f"Augmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # ---- Train new Huber alphas ----
    saved = {}   # alpha -> (oof, te)
    for a in NEW_ALPHAS:
        print(f"\n=== Huber alpha={a} ===")
        params = dict(BASE_PARAMS, objective="huber", alpha=a)
        oof = np.full(n_tr, np.nan)

        for fold, (tr_idx, va_idx, em_va, nl_va, sl_va) in enumerate(fold_aux):
            va_assay = np.column_stack([
                em_va, nl_va, sl_va, has_null[va_idx],
                np.log1p(np.clip(em_va, 0, None))
            ] + [o[va_idx] for o in meta_oofs])
            X_va = np.hstack([X_tr[va_idx], va_assay])

            m = lgb.train(params, lgb.Dataset(X_tr_aug[tr_idx], label=y_tr[tr_idx]),
                          valid_sets=[lgb.Dataset(X_va, label=y_tr[va_idx])],
                          callbacks=[lgb.early_stopping(80, verbose=False),
                                     lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_va)
            print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

        oof_rae = rae(y_tr, oof)
        print(f"  OOF RAE={oof_rae:.4f}")

        m_final = lgb.train(dict(params, n_estimators=1000),
                            lgb.Dataset(X_tr_aug, label=y_tr),
                            callbacks=[lgb.log_evaluation(-1)])
        te_preds = np.clip(m_final.predict(X_te_aug),
                           y_tr.min() - 0.5, y_tr.max() + 0.5)
        ratio = te_preds.std() / oof.std()
        in_r = in_rae(y_unblind, te_preds[unblind_idx])
        print(f"  TEST med={np.median(te_preds):.2f} std={te_preds.std():.3f} ratio={ratio:.2f}"
              f"  in_RAE(253)={in_r:.4f}")

        saved[a] = (oof, te_preds, oof_rae, in_r)

        tag = alpha_tag(a)
        np.save(DATA_PROCESSED / f"oof_nb800_huber_{tag}.npy", oof)
        np.save(DATA_PROCESSED / f"te_nb800_huber_{tag}.npy",  te_preds)

        sub = pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": te_preds,
        })
        sub.to_csv(SUBMISSIONS / f"nb800_huber_{tag}.csv", index=False)

    # ---- 4-way mean ensemble across new alphas ----
    te_stack_new = np.stack([saved[a][1] for a in NEW_ALPHAS], axis=0)
    te_ens4 = te_stack_new.mean(axis=0)
    in_r_ens4 = in_rae(y_unblind, te_ens4[unblind_idx])
    np.save(DATA_PROCESSED / "te_nb800_huber_ens.npy", te_ens4)
    pd.DataFrame({"SMILES": te["smiles"].values, "Molecule Name": te["name"].values,
                  "pEC50": te_ens4}).to_csv(
        SUBMISSIONS / "nb800_huber_ens4.csv", index=False)
    print(f"\n4-way Huber ensemble (new alphas): in_RAE={in_r_ens4:.4f}")

    # ---- 7-way mean ensemble: new + nb120 originals ----
    old_tes = []
    for a in OLD_ALPHAS:
        p = DATA_PROCESSED / f"te_nb120_huber_{alpha_tag(a)}.npy"
        old_tes.append(np.load(p))
    te_stack_all = np.concatenate([te_stack_new, np.stack(old_tes, axis=0)], axis=0)
    te_ens7 = te_stack_all.mean(axis=0)
    in_r_ens7 = in_rae(y_unblind, te_ens7[unblind_idx])
    np.save(DATA_PROCESSED / "te_nb800_huber_ens7.npy", te_ens7)
    pd.DataFrame({"SMILES": te["smiles"].values, "Molecule Name": te["name"].values,
                  "pEC50": te_ens7}).to_csv(
        SUBMISSIONS / "nb800_huber_ens7.csv", index=False)
    print(f"7-way Huber ensemble (new + nb120 originals): in_RAE={in_r_ens7:.4f}")

    # ---- Summary ----
    print("\n=== nb800 SUMMARY ===")
    print(f"{'variant':>22s}  {'OOF_RAE':>8s}  {'in_RAE':>8s}")
    for a in NEW_ALPHAS:
        oof, _, oof_rae, in_r = saved[a]
        print(f"{'huber_'+alpha_tag(a):>22s}  {oof_rae:>8.4f}  {in_r:>8.4f}")
    print(f"{'nb800_huber_ens (4-way)':>22s}  {'-':>8s}  {in_r_ens4:>8.4f}")
    print(f"{'nb800_huber_ens7 (7-way)':>22s}  {'-':>8s}  {in_r_ens7:>8.4f}")

    # also print pre-loaded old in_RAEs for context
    print("\nReference (already saved):")
    for a in OLD_ALPHAS:
        p = np.load(DATA_PROCESSED / f"te_nb120_huber_{alpha_tag(a)}.npy")
        print(f"  nb120_huber_{alpha_tag(a)}  in_RAE={in_rae(y_unblind, p[unblind_idx]):.4f}")

    # Write a small JSON for the orchestrator
    summary = {
        "per_alpha_in_rae": {f"huber_{a}": float(saved[a][3]) for a in NEW_ALPHAS},
        "per_alpha_oof_rae": {f"huber_{a}": float(saved[a][2]) for a in NEW_ALPHAS},
        "ens4_in_rae": float(in_r_ens4),
        "ens7_in_rae": float(in_r_ens7),
    }
    with open(DATA_PROCESSED / "nb800_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {DATA_PROCESSED/'nb800_summary.json'}")


if __name__ == "__main__":
    main()
