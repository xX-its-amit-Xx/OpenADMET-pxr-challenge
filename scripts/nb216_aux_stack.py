"""nb216 -- Multi-task auxiliary-target stack.

Train auxiliary LGBM models to predict:
- log(pEC50 SE)        -- assay reliability proxy
- Emax (PXR)           -- efficacy
- pEC50_null           -- counter-assay activity (PXR-specificity proxy)

Use their OOF + test predictions as additional features for a main pEC50 LGBM.

The hypothesis: predictions become more diverse from existing pool because the
features now encode "is this compound's label trustworthy", "would it be a strong
activator", "is it likely PXR-specific" -- all signals not in raw morgan/rdkit.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_train, load_test, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.featurize import combined as feat_combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
COLLAPSE_THRESH = 0.58
PREV_BEST = 0.296172

t0 = time.time()


def cv_lgbm(X_tr, y_tr, X_te, splits, objective="regression_l1", n_estimators=1500):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    te_pred = np.zeros(X_te.shape[0])
    for fi, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=n_estimators, num_leaves=64, learning_rate=0.03,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, objective=objective,
            random_state=SEED, verbose=-1,
        )
        m.fit(
            X_tr[tr_idx], y_tr[tr_idx],
            eval_set=[(X_tr[va_idx], y_tr[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        te_pred += m.predict(X_te) / N_FOLDS
    return oof, te_pred


def main():
    print("=== nb216: Multi-task auxiliary stack ===\n", flush=True)

    tr_df = load_train()
    te_df = load_test()
    cn_df = load_counter()
    print(f"Train: {len(tr_df)}, Test: {len(te_df)}, Counter: {len(cn_df)}\n", flush=True)

    y_pec = tr_df["pec50"].values.astype(np.float64)
    n_tr = len(tr_df)

    scaffolds = tr_df["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ----- Build counter-assay pEC50 lookup (for compounds in both) -----
    cn_lookup = dict(zip(cn_df["smiles"], cn_df["pec50"]))
    cn_emax_lookup = dict(zip(cn_df["smiles"], cn_df["emax"]))

    # ----- Base features -----
    print("Computing base features...", flush=True)
    X_tr = impute(feat_combined(tr_df["smiles"].tolist())).astype(np.float32)
    X_te = impute(feat_combined(te_df["smiles"].tolist())).astype(np.float32)
    print(f"  base shape: {X_tr.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ----- Aux target 1: log(pEC50 SE) ---------------------------
    # Lower SE = more reliable label
    y_se = tr_df["pec50_se"].values.astype(np.float64)
    y_logse = np.log(np.where(y_se > 1e-3, y_se, 1e-3))
    mask_se = ~np.isnan(y_logse)
    print(f"\nAux 1: log(pEC50_SE)  -- {mask_se.sum()}/{n_tr} valid", flush=True)

    oof_logse = np.full(n_tr, np.nan)
    te_logse = np.zeros(len(te_df))
    if mask_se.sum() > n_tr * 0.5:
        # Use only valid rows for training; predict for all
        valid_idx = np.where(mask_se)[0]
        # Subset splits to valid indices
        for fi, (tr_idx, va_idx) in enumerate(splits):
            tr_v = np.intersect1d(tr_idx, valid_idx)
            va_v = np.intersect1d(va_idx, valid_idx)
            if len(tr_v) < 100: continue
            m = lgb.LGBMRegressor(
                n_estimators=800, num_leaves=32, learning_rate=0.05,
                min_child_samples=20, random_state=SEED, verbose=-1,
            )
            m.fit(X_tr[tr_v], y_logse[tr_v],
                  eval_set=[(X_tr[va_v], y_logse[va_v])] if len(va_v) > 0 else None,
                  callbacks=[lgb.early_stopping(40, verbose=False)] if len(va_v) > 0 else None)
            # Predict for ALL va_idx (not just valid), so feature is dense
            oof_logse[va_idx] = m.predict(X_tr[va_idx])
            te_logse += m.predict(X_te) / N_FOLDS
        print(f"  OOF MAE: {np.nanmean(np.abs(y_logse - oof_logse)):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        print("  SKIP (insufficient data)", flush=True)

    # ----- Aux target 2: Emax (PXR) -------------------------------
    y_emax = tr_df["emax"].values.astype(np.float64)
    mask_emax = ~np.isnan(y_emax)
    print(f"\nAux 2: Emax (PXR)  -- {mask_emax.sum()}/{n_tr} valid", flush=True)

    oof_emax = np.full(n_tr, np.nan)
    te_emax = np.zeros(len(te_df))
    if mask_emax.sum() > n_tr * 0.5:
        valid_idx = np.where(mask_emax)[0]
        for fi, (tr_idx, va_idx) in enumerate(splits):
            tr_v = np.intersect1d(tr_idx, valid_idx)
            if len(tr_v) < 100: continue
            m = lgb.LGBMRegressor(
                n_estimators=800, num_leaves=32, learning_rate=0.05,
                min_child_samples=20, random_state=SEED, verbose=-1,
            )
            va_v = np.intersect1d(va_idx, valid_idx)
            m.fit(X_tr[tr_v], y_emax[tr_v],
                  eval_set=[(X_tr[va_v], y_emax[va_v])] if len(va_v) > 0 else None,
                  callbacks=[lgb.early_stopping(40, verbose=False)] if len(va_v) > 0 else None)
            oof_emax[va_idx] = m.predict(X_tr[va_idx])
            te_emax += m.predict(X_te) / N_FOLDS
        print(f"  OOF MAE: {np.nanmean(np.abs(y_emax[mask_emax] - oof_emax[mask_emax])):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ----- Aux target 3: counter-assay pEC50 (predict for non-overlap + test) -----
    # Use real value when available, predicted otherwise
    print("\nAux 3: counter-assay pEC50", flush=True)
    cn_real_tr = np.array([cn_lookup.get(s, np.nan) for s in tr_df["smiles"]])
    cn_real_te = np.array([cn_lookup.get(s, np.nan) for s in te_df["smiles"]])
    print(f"  cn-overlap train: {(~np.isnan(cn_real_tr)).sum()}/{n_tr}", flush=True)
    print(f"  cn-overlap test: {(~np.isnan(cn_real_te)).sum()}/{len(te_df)}", flush=True)

    cn_valid_idx = np.where(~np.isnan(cn_real_tr))[0]
    oof_cn = np.full(n_tr, np.nan)
    te_cn = np.zeros(len(te_df))
    for fi, (tr_idx, va_idx) in enumerate(splits):
        tr_v = np.intersect1d(tr_idx, cn_valid_idx)
        va_v = np.intersect1d(va_idx, cn_valid_idx)
        if len(tr_v) < 100: continue
        m = lgb.LGBMRegressor(
            n_estimators=800, num_leaves=32, learning_rate=0.05,
            min_child_samples=20, random_state=SEED, verbose=-1,
        )
        m.fit(X_tr[tr_v], cn_real_tr[tr_v],
              eval_set=[(X_tr[va_v], cn_real_tr[va_v])] if len(va_v) > 0 else None,
              callbacks=[lgb.early_stopping(40, verbose=False)] if len(va_v) > 0 else None)
        oof_cn[va_idx] = m.predict(X_tr[va_idx])
        te_cn += m.predict(X_te) / N_FOLDS

    # Hybrid: real where available, predicted otherwise (for the FEATURE)
    cn_feat_tr = np.where(np.isnan(cn_real_tr), oof_cn, cn_real_tr)
    cn_feat_te = np.where(np.isnan(cn_real_te), te_cn, cn_real_te)

    # ----- Aux target 4: PXR-specificity (delta = pec50 - pec50_null) -----
    delta_pec = np.where(np.isnan(cn_real_tr), 0.0, y_pec - np.where(np.isnan(cn_real_tr), 0, cn_real_tr))
    # Just compute predicted delta from features (proxy for PXR-specificity)
    print("\nAux 4: PXR-specificity (pec50 - pec50_null) for overlap", flush=True)
    delta_valid_idx = cn_valid_idx
    oof_delta = np.full(n_tr, np.nan)
    te_delta = np.zeros(len(te_df))
    if len(delta_valid_idx) > 100:
        for fi, (tr_idx, va_idx) in enumerate(splits):
            tr_v = np.intersect1d(tr_idx, delta_valid_idx)
            va_v = np.intersect1d(va_idx, delta_valid_idx)
            if len(tr_v) < 100: continue
            tgt = y_pec[tr_v] - cn_real_tr[tr_v]
            m = lgb.LGBMRegressor(
                n_estimators=800, num_leaves=32, learning_rate=0.05,
                min_child_samples=20, random_state=SEED, verbose=-1,
            )
            m.fit(X_tr[tr_v], tgt,
                  eval_set=[(X_tr[va_v], y_pec[va_v]-cn_real_tr[va_v])] if len(va_v) > 0 else None,
                  callbacks=[lgb.early_stopping(40, verbose=False)] if len(va_v) > 0 else None)
            oof_delta[va_idx] = m.predict(X_tr[va_idx])
            te_delta += m.predict(X_te) / N_FOLDS

    # ----- Build augmented feature matrix -----
    print("\nBuilding augmented features...", flush=True)
    X_tr_aux = np.hstack([
        oof_logse.reshape(-1, 1),
        oof_emax.reshape(-1, 1),
        cn_feat_tr.reshape(-1, 1),
        oof_delta.reshape(-1, 1),
    ])
    X_te_aux = np.hstack([
        te_logse.reshape(-1, 1),
        te_emax.reshape(-1, 1),
        cn_feat_te.reshape(-1, 1),
        te_delta.reshape(-1, 1),
    ])
    X_tr_aux = impute(X_tr_aux)
    X_te_aux = impute(X_te_aux)

    X_tr_full = np.hstack([X_tr, X_tr_aux]).astype(np.float32)
    X_te_full = np.hstack([X_te, X_te_aux]).astype(np.float32)
    print(f"  augmented shape: {X_tr_full.shape}  (added 4 aux columns)\n", flush=True)

    # ----- Train MAIN pEC50 model -----
    print("Training MAIN LGBM (pEC50) with aux features...", flush=True)
    oof, te_pred = cv_lgbm(X_tr_full, y_pec, X_te_full, splits, objective="regression_l1",
                           n_estimators=2000)

    r = rae(y_pec, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***BEATS PREV BEST***" if (ratio >= COLLAPSE_THRESH and r < PREV_BEST) else ""
    print(f"\n=== nb216 OOF: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat} ===", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    out_stem = "nb216_aux_stack"
    np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{out_stem}.npy", te_pred)
    sub = pd.DataFrame({
        "SMILES": te_df["smiles"].values,
        "Molecule Name": te_df["name"].values,
        "pEC50": te_pred,
    })
    sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
    print(f"Saved: {out_stem}", flush=True)


if __name__ == "__main__":
    main()
