"""nb221 -- Assay-noise-aware training + counter-assay-resolved PXR signal.

Two ideas:

  1. NOISE-AWARE WEIGHTING: every CRC compound has pec50_se (median 0.24).
     Compounds with high SE are less reliable labels. Weight training by
     1/(SE^2 + noise_floor^2). High-noise compounds contribute less.
     This is theoretically a generalized least squares -> MAE/Huber mapping.

  2. COUNTER-ASSAY DECONFOUNDING: 2,859 CRC compounds have BOTH PXR and PXR-null
     (counter-assay) measurements. The TRUE PXR-specific signal is:
        pec50_pxr_specific = pec50_pxr - pec50_null
     Compounds with high counter-assay activity (false positives in original)
     get downweighted. The pec50_specific target may be a cleaner objective.

  3. NOISE DISTRIBUTION FROM CHEMBL PXR: 945 ChEMBL PXR records give us a second
     independent view. Compounds appearing in both with similar pEC50 confirm
     low noise. Compounds with large disagreement are noisy/assay-dependent.

Strategy:
  A. Train standard LGBM with sample_weight = 1/(SE^2 + 0.15^2) (noise floor)
  B. Train delta-target LGBM on pec50 - pec50_null, where available
  C. Multi-task: predict both pec50_pxr and pec50_null jointly
  D. Compare blends with nb197 grand ensemble; require collapse-check pass
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

from pxr.data import load_train, load_test, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

COLLAPSE_THRESH = 0.58
NOISE_FLOOR = 0.15  # known PXR assay noise floor (median SE = 0.24, floor estimate)

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def main():
    print("=== nb221: Noise-aware + counter-deconfounded training ===\n")

    tr = load_train()
    te_df = load_test()
    co = load_counter()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    se_tr = tr["pec50_se"].fillna(tr["pec50_se"].median()).values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    # Match counter-assay by SMILES (after standardization)
    co["std_smiles"] = co["smiles"]   # already canonical from load_counter
    co_map = dict(zip(co["smiles"], co["pec50"]))
    co_se_map = dict(zip(co["smiles"], co["pec50_se"].fillna(co["pec50_se"].median())))

    y_null = np.array([co_map.get(s, np.nan) for s in tr["smiles"]])
    se_null = np.array([co_se_map.get(s, NOISE_FLOOR) for s in tr["smiles"]])
    has_null = np.isfinite(y_null)
    print(f"Compounds with counter-assay measurement: {has_null.sum()}/{len(y_tr)}")

    # PXR-specific signal where available
    y_specific = np.where(has_null, y_tr - y_null, y_tr)
    print(f"y_specific (pxr - null) mean ± std for matched: "
          f"{y_specific[has_null].mean():.3f} ± {y_specific[has_null].std():.3f}")
    print(f"  (vs raw pEC50: mean={y_tr.mean():.3f} std={y_tr.std():.3f})")

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"\nBase nb197: OOF RAE={base_rae:.4f}  ratio={te_base.std()/oof_base.std():.3f}\n")

    # ── Featurize ─────────────────────────────────────────────────────────────
    print("Featurizing...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    # ── Sample weights ────────────────────────────────────────────────────────
    w_inv_se = 1.0 / (se_tr ** 2 + NOISE_FLOOR ** 2)
    w_inv_se /= w_inv_se.mean()  # normalize to mean=1
    print(f"Inv-SE weights: mean={w_inv_se.mean():.3f}  range=[{w_inv_se.min():.3f}, {w_inv_se.max():.3f}]")

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    # ── Run multiple configs ──────────────────────────────────────────────────
    configs = [
        ("base_uniform",  y_tr, None),
        ("noise_weighted", y_tr, w_inv_se),
    ]

    results = {}
    for name, y_target, sw in configs:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            kwargs = dict(eval_set=[(X_tr[va_idx], y_tr[va_idx])],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                      lgb.log_evaluation(-1)])
            if sw is not None:
                m.fit(X_tr[tr_idx], y_target[tr_idx], sample_weight=sw[tr_idx], **kwargs)
            else:
                m.fit(X_tr[tr_idx], y_target[tr_idx], **kwargs)
            oof[va_idx] = m.predict(X_tr[va_idx])
            te_preds.append(m.predict(X_te))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"\n[{name}]")
        print(f"  OOF RAE={r:.4f}  te_std={te_pred.std():.4f}  ratio={ratio:.3f}  [{flag}]")
        results[name] = (oof, te_pred, r, ratio)

    # ── Counter-deconfounded model ────────────────────────────────────────────
    print("\n[counter_decon] Two-stage model:")
    print("  Stage 1: predict pec50_null for compounds without counter")
    print("  Stage 2: predict (pec50_pxr - pec50_null) using all CRC data")
    print("  Final:   stage2_pred + (true null OR stage1_pred null)")

    # Stage 1: train LGBM on compounds with counter -> predict null pEC50
    has_null_idx = np.where(has_null)[0]
    X_null = X_tr[has_null_idx]
    y_null_observed = y_null[has_null_idx]

    print(f"  Stage 1 training on {len(y_null_observed)} compounds with counter-assay")

    # Predict null for all train+test (impute when missing)
    null_pred_tr = np.zeros(len(y_tr))
    null_pred_te = np.zeros(len(X_te))
    null_te_preds = []

    # 5-fold CV for null predictions on train
    from sklearn.model_selection import KFold
    null_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for null_tr, null_va in null_kf.split(X_null):
        m_null = lgb.LGBMRegressor(**LGBM_BASE)
        m_null.fit(X_null[null_tr], y_null_observed[null_tr],
                    eval_set=[(X_null[null_va], y_null_observed[null_va])],
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        null_pred_tr[has_null_idx[null_va]] = m_null.predict(X_null[null_va])

    # Use observed nulls where we have them; predicted where we don't
    null_pred_tr_final = np.where(has_null, y_null, null_pred_tr)
    # train a full null model for test predictions
    m_null_full = lgb.LGBMRegressor(**LGBM_BASE)
    m_null_full.fit(X_null, y_null_observed, callbacks=[lgb.log_evaluation(-1)])
    null_pred_te_final = m_null_full.predict(X_te)
    print(f"  Predicted null on test: mean={null_pred_te_final.mean():.3f} std={null_pred_te_final.std():.3f}")

    # Stage 2: train on (pec50_pxr - pec50_null) target where both exist
    y_delta = y_tr - null_pred_tr_final
    print(f"  Stage 2 target (pxr - null) for {has_null.sum()} compounds: "
          f"mean={y_delta[has_null].mean():.3f} std={y_delta[has_null].std():.3f}")

    oof_delta = np.zeros(len(y_tr))
    delta_te_preds = []
    for tr_idx, va_idx in folds:
        m = lgb.LGBMRegressor(**LGBM_BASE)
        m.fit(X_tr[tr_idx], y_delta[tr_idx],
                eval_set=[(X_tr[va_idx], y_delta[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        oof_delta[va_idx] = m.predict(X_tr[va_idx])
        delta_te_preds.append(m.predict(X_te))

    delta_te = np.mean(delta_te_preds, axis=0)
    # Reconstruct full pEC50 prediction = delta_pred + null_pred
    oof_decon = oof_delta + null_pred_tr_final
    te_decon  = delta_te + null_pred_te_final
    r_decon = rae(y_tr, oof_decon)
    ratio_decon = te_decon.std() / oof_decon.std()
    flag = "PASS" if ratio_decon >= COLLAPSE_THRESH else "FAIL"
    print(f"  Counter-deconfounded: OOF RAE={r_decon:.4f}  te_std={te_decon.std():.4f}  "
          f"ratio={ratio_decon:.3f}  [{flag}]")
    results["counter_decon"] = (oof_decon, te_decon, r_decon, ratio_decon)

    # ── Blend each with nb197 ─────────────────────────────────────────────────
    print("\n── Blends with nb197 ──")
    saved = []
    for name, (oof_v, te_v, r_v, ratio_v) in results.items():
        if name == "base_uniform":
            continue
        best_w, best_r_bl, best_ratio_bl = None, 999, 0
        best_oof_bl, best_te_bl = None, None
        for w in np.arange(0.05, 0.75, 0.05):
            oof_bl = (1-w)*oof_base + w*oof_v
            te_bl  = (1-w)*te_base  + w*te_v
            r_bl   = rae(y_tr, oof_bl)
            ratio_bl = te_bl.std() / oof_bl.std()
            if ratio_bl >= COLLAPSE_THRESH and r_bl < best_r_bl:
                best_r_bl = r_bl; best_ratio_bl = ratio_bl
                best_w = w; best_oof_bl = oof_bl; best_te_bl = te_bl

        print(f"\n  Best blend with {name}: ", end="")
        if best_w is not None:
            print(f"w={best_w:.2f}  OOF={best_r_bl:.4f}  ratio={best_ratio_bl:.3f}")
            if best_r_bl < base_rae:
                out_name = f"221_{name}_w{int(best_w*100):02d}"
                np.save(DATA_PROCESSED / f"oof_{out_name}.npy", best_oof_bl)
                np.save(DATA_PROCESSED / f"te_{out_name}.npy", best_te_bl)
                sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": best_te_bl})
                sub.to_csv(SUBMISSIONS / f"{out_name}.csv", index=False)
                saved.append(f"{out_name} OOF={best_r_bl:.4f}")
        else:
            print("no passing blend")

    print(f"\n=== Saved: {saved or ['none']}")


if __name__ == "__main__":
    main()
