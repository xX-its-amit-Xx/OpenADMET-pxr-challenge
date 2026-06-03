"""nb984 - Counter-assay kNN-Tanimoto imputation as a promiscuity proxy feature.

Hypothesis: filling in the counter-assay pec50_null for ALL 4139 train + 513
test compounds (via kNN-Tanimoto, k=5, weighted mean from the 2858 labeled
compounds) gives the model a dense "promiscuity proxy" feature - similar in
spirit to chemprop_aux's dual-head trick, but as an explicit imputed column.

Pipeline:
  1. Build Morgan FPs for train (4139) + test (513).
  2. Identify the 2858 train compounds with real pec50_null. Use them as the
     kNN library. The other 1281 train + 513 test compounds get kNN-imputed
     pec50_null.
  3. Build the standard nb120/nb801 augmented matrix (Morgan+RDKit+assay-decomp
     +meta-OOFs), with one tweak: replace median-imputed null with kNN-imputed
     null wherever the raw value is missing.
  4. LGBM Huber alpha=2.0, scaffold 5-fold CV. Compute in_RAE on the 253
     unblind. Target: beat chemprop_aux 0.6216 -> predicted LB ~0.62.

Wall-time budget: <10 minutes.
"""
import os, sys, warnings, json, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
HUBER_ALPHA = 2.0
K_NN = 5

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


def tanimoto_knn_impute(fp_query, fp_lib, y_lib, k=5):
    """For each row in fp_query, k=top-Tanimoto neighbors in fp_lib, weighted mean of y_lib.

    Uses bitwise representation. fp_query and fp_lib are (N, 2048) uint8 0/1.
    Returns shape (N_query,) imputed values.
    """
    # Tanimoto = |A & B| / |A| + |B| - |A & B|
    q = fp_query.astype(np.float32)
    L = fp_lib.astype(np.float32)
    qn = q.sum(axis=1, keepdims=True)        # (Nq, 1)
    Ln = L.sum(axis=1, keepdims=True).T      # (1, Nl)
    inter = q @ L.T                          # (Nq, Nl)
    denom = qn + Ln - inter
    sim = np.where(denom > 0, inter / np.maximum(denom, 1e-9), 0.0)
    # top-k indices per row
    idx_topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    out = np.zeros(len(fp_query), dtype=np.float64)
    for i in range(len(fp_query)):
        idx = idx_topk[i]
        w = sim[i, idx]
        if w.sum() <= 0:
            out[i] = y_lib.mean()
        else:
            out[i] = float((w * y_lib[idx]).sum() / w.sum())
    return out, sim.max(axis=1)


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
    t0 = time.time()
    print("=== nb984: counter-assay kNN-Tanimoto imputation ===\n")

    unblind_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unblind = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unblind_idx) == len(y_unblind) == 253

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null_raw = np.array([counter_map.get(n, np.nan) for n in mol_names])
    has_null = ~np.isnan(pec50_null_raw)
    print(f"  paired CRC+counter: {int(has_null.sum())}/{n_tr}  "
          f"missing: {int((~has_null).sum())}")

    print("\nStage 0: Morgan FPs for kNN-Tanimoto...")
    fps_tr = morgan_fp_batch(tr["smiles"].tolist())
    fps_te = morgan_fp_batch(te["smiles"].tolist())
    print(f"  fps_tr={fps_tr.shape}  fps_te={fps_te.shape}")

    # ---- kNN library: 2858 paired compounds ----
    lib_mask = has_null
    fps_lib = fps_tr[lib_mask]
    y_lib = pec50_null_raw[lib_mask]
    print(f"\nStage 1: kNN-Tanimoto k={K_NN} impute  lib={len(y_lib)}")

    # impute the 1281 missing train rows
    missing_idx = np.where(~has_null)[0]
    imp_tr_missing, sim_tr_missing = tanimoto_knn_impute(
        fps_tr[missing_idx], fps_lib, y_lib, k=K_NN
    )
    pec50_null_full = pec50_null_raw.copy()
    pec50_null_full[missing_idx] = imp_tr_missing
    print(f"  train-missing imputed: n={len(missing_idx)}  "
          f"mean={imp_tr_missing.mean():.3f}  std={imp_tr_missing.std():.3f}  "
          f"top1_sim med={np.median(sim_tr_missing):.3f}")

    # impute all 513 test rows
    imp_te, sim_te = tanimoto_knn_impute(fps_te, fps_lib, y_lib, k=K_NN)
    print(f"  test imputed:  n=513  mean={imp_te.mean():.3f}  "
          f"std={imp_te.std():.3f}  top1_sim med={np.median(sim_te):.3f}")

    # ---- Structural features ----
    print("\nStage 2: Morgan + RDKit combined features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # ---- assay-decomp pieces (nb801-style, but using kNN-imputed null) ----
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    selectivity = y_tr - pec50_null_full  # uses kNN-imputed null when missing
    has_null_flag = has_null.astype(np.float32)

    # ---- meta-OOFs (gated like nb801) ----
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

    # ---- Stage 3: aux OOF for assay-decomp targets ----
    print("\nStage 3: aux OOF for emax/null/selectivity...")
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
    fold_aux = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_tr[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=pec50_null_full[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_tr[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx]  = m_sl.predict(X_tr[va_idx])
        fold_aux.append((tr_idx, va_idx,
                         oof_emax[va_idx].copy(), oof_null[va_idx].copy(),
                         oof_sel[va_idx].copy()))
    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=pec50_null_full), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    # ---- Augmented matrix ----
    # NEW: the kNN-imputed pec50_null as its OWN feature (a direct promiscuity proxy)
    knn_null_tr = pec50_null_full.astype(np.float32)  # raw where available, kNN where missing
    knn_null_te = imp_te.astype(np.float32)
    knn_sim_te  = sim_te.astype(np.float32)
    knn_sim_tr  = np.ones(n_tr, dtype=np.float32)
    knn_sim_tr[missing_idx] = sim_tr_missing.astype(np.float32)

    assay_oof = np.column_stack([
        oof_emax, oof_null, oof_sel, has_null_flag,
        np.log1p(np.clip(oof_emax, 0, None)),
        knn_null_tr, knn_sim_tr,
    ] + meta_oofs)
    assay_te = np.column_stack([
        te_emax, te_null, te_sel, np.zeros(len(X_te)),
        np.log1p(np.clip(te_emax, 0, None)),
        knn_null_te, knn_sim_te,
    ] + meta_tes)
    X_tr_aug = np.hstack([X_tr, assay_oof])
    X_te_aug = np.hstack([X_te, assay_te])
    print(f"\nAugmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")
    print(f"  (5 base + 2 kNN-null + {len(meta_oofs)} meta)")

    # ---- Stage 4: Huber LGBM ----
    print(f"\nStage 4: Huber alpha={HUBER_ALPHA}")
    params = dict(BASE_PARAMS, objective="huber", alpha=HUBER_ALPHA)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx, em_va, nl_va, sl_va) in enumerate(fold_aux):
        va_assay = np.column_stack([
            em_va, nl_va, sl_va, has_null_flag[va_idx],
            np.log1p(np.clip(em_va, 0, None)),
            knn_null_tr[va_idx], knn_sim_tr[va_idx],
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

    m_final = lgb.train(dict(params, n_estimators=1000),
                        lgb.Dataset(X_tr_aug, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te_aug),
                       y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = te_preds.std() / oof.std()
    in_r = in_rae(y_unblind, te_preds[unblind_idx])
    print(f"\n  TEST  med={np.median(te_preds):.3f}  std={te_preds.std():.3f}  ratio={ratio:.2f}")
    print(f"  in_RAE(253) = {in_r:.4f}   (target < 0.6216)")

    np.save(DATA_PROCESSED / "oof_nb984.npy", oof)
    np.save(DATA_PROCESSED / "te_nb984.npy",  te_preds)

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_preds,
    })
    sub_path = SUBMISSIONS / "nb984_counter_impute.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved te_nb984.npy and {sub_path}")

    summary = {
        "oof_rae": float(oof_rae),
        "in_rae": float(in_r),
        "te_std": float(te_preds.std()),
        "oof_std": float(oof.std()),
        "ratio": float(ratio),
        "n_paired": int(has_null.sum()),
        "n_imputed_train": int((~has_null).sum()),
        "n_imputed_test": 513,
        "knn_k": K_NN,
        "test_top1_sim_median": float(np.median(sim_te)),
        "train_missing_top1_sim_median": float(np.median(sim_tr_missing)),
        "n_meta_kept": len(meta_oofs),
        "meta_kept": meta_names,
        "beats_chemprop_aux_0_6216": bool(in_r < 0.6216),
        "wall_time_s": float(time.time() - t0),
    }
    with open(DATA_PROCESSED / "nb984_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote nb984_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
