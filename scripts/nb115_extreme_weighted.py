"""nb115 — Extreme-Weighted LGBM for Hit and Inactive Prediction.

Problem: hits (pEC50 > 6.0) = 1.6% of training, but they dominate the test set.
  Model shrinks extreme predictions toward the training mean.
  OOF error on extreme compounds: |error| > 2.5 for both high and low ends.

Fix: Assign higher sample weights to extreme pEC50 compounds.
  - Compounds with |pEC50 - mean| > 1.5: weight = distance_weight
  - This forces the model to fit the extremes better
  - Cost: slightly worse fit for middle-of-range compounds

Distance-based weighting scheme:
  w_i = max(1.0, k * abs(y_i - y_mean) / y_std)
  k = tuned by scaffold CV

Also tests: inverse-density weighting based on scaffold size.
  - Compounds in small scaffold families get higher weight
  - (Rare scaffolds → model has less data → more likely to fail)
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
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_BASE = dict(
    n_estimators=1500, num_leaves=64, learning_rate=0.04,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=600, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def distance_weights(y, k=3.0):
    """Assign higher weights to extreme values: w = max(1, k * |y - mean| / std)."""
    mean_y = y.mean(); std_y = y.std()
    return np.maximum(1.0, k * np.abs(y - mean_y) / std_y)


def scaffold_inverse_density_weights(scaffolds, y, min_w=1.0, max_w=5.0):
    """Compounds in rare scaffold families get higher weight."""
    sc_counts = defaultdict(int)
    for sc in scaffolds: sc_counts[sc] += 1
    weights = np.array([1.0 / np.sqrt(sc_counts[sc]) for sc in scaffolds])
    # Normalize to [min_w, max_w]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
    return min_w + (max_w - min_w) * weights


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def main():
    print("=== nb115: Extreme-Weighted LGBM ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    y_mean = y_tr.mean(); y_std = y_tr.std()
    n_hits   = (y_tr >= 6.0).sum()
    n_inact  = (y_tr <= 2.5).sum()
    print(f"Training: n={n_tr}, mean={y_mean:.3f}, std={y_std:.3f}")
    print(f"  Hits (pEC50 >= 6.0): {n_hits} ({100*n_hits/n_tr:.1f}%)")
    print(f"  Very inactive (pEC50 <= 2.5): {n_inact} ({100*n_inact/n_tr:.1f}%)")

    # Assay features
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_median  = np.nanmedian(pec50_null)
    null_imputed = np.where(np.isnan(pec50_null), null_median, pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    # Structural features
    print("\nComputing features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # Load best meta-OOF as additional features
    meta_oofs, meta_tes = [], []
    for stem in ["nb107_assay_decomp", "nb109_deep_meta_stack",
                 "nb101_delta_base", "nb99_sc_bio_fp", "grand_v6b"]:
        of = DATA_PROCESSED / f"oof_{stem}.npy"
        tf = DATA_PROCESSED / f"te_{stem}.npy"
        if of.exists() and tf.exists():
            o = np.load(of); t = np.load(tf)
            if o.ndim == 2: o = o[:, 0]
            if t.ndim == 2: t = t[:, 0]
            if len(o) == n_tr:
                o = np.where(np.isfinite(o), o, np.nanmean(o))
                t = np.where(np.isfinite(t), t, np.nanmean(t))
                # Only include non-collapsed
                if t.std() / o.std() >= 0.58:
                    meta_oofs.append(o); meta_tes.append(t)

    # ── Stage 1: OOF auxiliary features ──────────────────────────────────────
    print("\nStage 1: OOF auxiliary predictors...")
    oof_emax = np.full(n_tr, np.nan)
    oof_null = np.full(n_tr, np.nan)
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
        oof_sel[va_idx] = m_sl.predict(X_tr[va_idx])

    assay_oof_cols = np.column_stack([
        oof_emax, oof_null, oof_sel, has_null,
        np.log1p(np.clip(oof_emax, 0, None)),
    ] + meta_oofs)

    # Full-data auxiliary predictors for test
    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    assay_te_cols = np.column_stack([
        te_emax, te_null, te_sel, np.zeros(len(X_te)),
        np.log1p(np.clip(te_emax, 0, None)),
    ] + meta_tes)

    X_tr_aug = np.hstack([X_tr, assay_oof_cols])
    X_te_aug = np.hstack([X_te, assay_te_cols])
    print(f"Augmented features: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # ── Stage 2: Sweep weighting schemes ─────────────────────────────────────
    print("\n=== Stage 2: Weighting scheme sweep ===")

    weighting_schemes = {
        "uniform":     np.ones(n_tr),
        "dist_k2":     distance_weights(y_tr, k=2.0),
        "dist_k3":     distance_weights(y_tr, k=3.0),
        "dist_k5":     distance_weights(y_tr, k=5.0),
        "scaffold_inv": scaffold_inverse_density_weights(scaffolds, y_tr),
        "combined_k3": distance_weights(y_tr, k=3.0) * scaffold_inverse_density_weights(scaffolds, y_tr),
    }

    best_overall_r = 1.0
    best_stem_name = ""
    all_results = {}

    for scheme_name, weights_all in weighting_schemes.items():
        oof_scheme = np.full(n_tr, np.nan)

        for fold, (tr_idx, va_idx) in enumerate(splits):
            # Recompute fold-level auxiliary features
            m_em2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                              callbacks=[lgb.log_evaluation(-1)])
            m_nl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                              callbacks=[lgb.log_evaluation(-1)])
            m_sl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                              callbacks=[lgb.log_evaluation(-1)])

            em_va = 10.0 ** m_em2.predict(X_tr[va_idx])
            nl_va = m_nl2.predict(X_tr[va_idx])
            sl_va = m_sl2.predict(X_tr[va_idx])

            va_assay = np.column_stack([
                em_va, nl_va, sl_va, has_null[va_idx],
                np.log1p(np.clip(em_va, 0, None)),
            ] + [o[va_idx] for o in meta_oofs])

            X_va_fold = np.hstack([X_tr[va_idx], va_assay])
            X_tr_fold = X_tr_aug[tr_idx]

            m = lgb.train(
                LGBM_BASE,
                lgb.Dataset(X_tr_fold, label=y_tr[tr_idx],
                            weight=weights_all[tr_idx]),
                valid_sets=[lgb.Dataset(X_va_fold, label=y_tr[va_idx])],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)]
            )
            oof_scheme[va_idx] = m.predict(X_va_fold)

        r = rae(y_tr, oof_scheme)
        print(f"  {scheme_name:20s}: OOF RAE={r:.4f}  (weights range: "
              f"[{weights_all.min():.1f}, {weights_all.max():.1f}])")
        all_results[scheme_name] = (r, oof_scheme)
        if r < best_overall_r:
            best_overall_r = r
            best_stem_name = scheme_name

    print(f"\nBest weighting scheme: {best_stem_name} (OOF RAE={best_overall_r:.4f})")

    # ── Train final model with best weighting ────────────────────────────────
    best_weights = weighting_schemes[best_stem_name]
    m_final = lgb.train(
        LGBM_BASE,
        lgb.Dataset(X_tr_aug, label=y_tr, weight=best_weights),
        callbacks=[lgb.log_evaluation(-1)]
    )
    te_preds = m_final.predict(X_te_aug)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)

    print(f"\nTest: min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  "
          f"max={te_preds.max():.2f}  std={te_preds.std():.3f}")

    # Compare extreme accuracy for best model
    best_oof = all_results[best_stem_name][1]
    uniform_oof = all_results["uniform"][1]
    for label, oof_v in [("uniform", uniform_oof), (best_stem_name, best_oof)]:
        hits_mask = y_tr >= 6.0
        inact_mask = y_tr <= 2.5
        hits_rae = rae(y_tr[hits_mask], oof_v[hits_mask]) if hits_mask.sum() > 5 else np.nan
        inact_rae = rae(y_tr[inact_mask], oof_v[inact_mask]) if inact_mask.sum() > 5 else np.nan
        print(f"\n  {label}: hits RAE={hits_rae:.4f}  inactives RAE={inact_rae:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(DATA_PROCESSED / "oof_nb115_extreme_weighted.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb115_extreme_weighted.npy", te_preds)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    sub.to_csv(SUBMISSIONS / "115_extreme_weighted.csv", index=False)
    print(f"\nSaved: submissions/115_extreme_weighted.csv")
    print(f"OOF RAE ({best_stem_name}) = {best_overall_r:.4f}")


if __name__ == "__main__":
    main()
