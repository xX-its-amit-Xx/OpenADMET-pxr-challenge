"""nb145 -- Non-linear meta-learner / stacker.

SLSQP can only find linear combinations. A non-linear stacker (XGBoost,
LightGBM, MLP) can capture cross-model interactions. Use all our OOF arrays
as features for a meta-learner trained via scaffold CV to predict pEC50.

If XGBoost stacker OOF < nb224's 0.289087, we have a new winner.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# All candidate stems to stack
STEMS = [
    # Original anchor pool
    "nb167_xgboost_mae", "nb156_catboost_mae", "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool", "nb165_multiseed_162c", "nb149_meta_maeloss",
    "nb183_qreg_poly10", "nb187_diversity_qreg",
    # New candidates from this campaign
    "nb219_aug_30pct", "nb228_medchem", "nb132_tanimoto",
    "nb133_interactions", "nb144_full_papyrus",
    "nb230_ecfp8", "nb230_atom_pair", "nb230_top_torsion", "nb230_maccs",
    # Older ones that might add signal
    "nb215_chemist_features", "nb213_chemberta_mtr", "nb217_pxr_modes",
    # The reference best
    "nb197_dense_grid", "nb224_pool_plus_2",
]

COLLAPSE_THRESH = 0.58


def main():
    print("=== nb145: Non-linear stacker ===\n")
    print("Target to beat: nb224 OOF 0.289087")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    folds = scaffold_kfold_indices(scaffolds, 5, 42)

    cols_oof, cols_te, names = [], [], []
    for stem in STEMS:
        op = DATA_PROCESSED / f"oof_{stem}.npy"
        tp = DATA_PROCESSED / f"te_{stem}.npy"
        if not op.exists() or not tp.exists():
            print(f"  MISSING: {stem}")
            continue
        o = np.load(op).astype(np.float64).flatten()
        t = np.load(tp).astype(np.float64).flatten()
        if len(o) != n_tr or len(t) != len(te_df):
            print(f"  WRONG SIZE: {stem}")
            continue
        o = np.where(np.isfinite(o), o, np.nanmean(o))
        t = np.where(np.isfinite(t), t, np.nanmean(t))
        cols_oof.append(o); cols_te.append(t); names.append(stem)
        r = rae(y_tr, o); ratio = t.std() / o.std()
        print(f"  {stem:30s}: OOF={r:.4f}  ratio={ratio:.3f}")

    X_tr = np.column_stack(cols_oof)
    X_te = np.column_stack(cols_te)
    print(f"\nStacking matrix: train={X_tr.shape}  test={X_te.shape}")

    # ── XGBoost stacker ─────────────────────────────────────────────────
    print("\n[XGBoost stacker]")
    XGB_KW = dict(
        n_estimators=2000, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, objective="reg:absoluteerror",
        n_jobs=4, random_state=42, verbosity=0,
    )
    oof_xgb = np.zeros(len(y_tr))
    te_preds_xgb = []
    for tr_idx, va_idx in folds:
        m = xgb.XGBRegressor(**XGB_KW, early_stopping_rounds=80)
        m.fit(X_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr[va_idx], y_tr[va_idx])],
              verbose=False)
        oof_xgb[va_idx] = m.predict(X_tr[va_idx])
        te_preds_xgb.append(m.predict(X_te))
    te_xgb = np.mean(te_preds_xgb, axis=0)
    r_x = rae(y_tr, oof_xgb); ratio_x = te_xgb.std() / oof_xgb.std()
    print(f"  OOF RAE={r_x:.6f}  ratio={ratio_x:.3f}  te_std={te_xgb.std():.4f}")

    # ── LightGBM stacker ────────────────────────────────────────────────
    print("\n[LightGBM stacker]")
    LGB_KW = dict(
        n_estimators=2000, num_leaves=15, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_lambda=1.0, objective="mae", n_jobs=4, random_state=42, verbose=-1,
    )
    oof_lgb = np.zeros(len(y_tr))
    te_preds_lgb = []
    for tr_idx, va_idx in folds:
        m = lgb.LGBMRegressor(**LGB_KW)
        m.fit(X_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr[va_idx], y_tr[va_idx])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_lgb[va_idx] = m.predict(X_tr[va_idx])
        te_preds_lgb.append(m.predict(X_te))
    te_lgb = np.mean(te_preds_lgb, axis=0)
    r_l = rae(y_tr, oof_lgb); ratio_l = te_lgb.std() / oof_lgb.std()
    print(f"  OOF RAE={r_l:.6f}  ratio={ratio_l:.3f}  te_std={te_lgb.std():.4f}")

    # ── Save winners ────────────────────────────────────────────────────
    saved = []
    te_raw = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smap = dict(zip(te_raw["Molecule Name"], te_raw["SMILES"]))

    for name, oof, te, r, ratio in [("xgb_stack", oof_xgb, te_xgb, r_x, ratio_x),
                                     ("lgb_stack", oof_lgb, te_lgb, r_l, ratio_l)]:
        np.save(DATA_PROCESSED / f"oof_nb145_{name}.npy", oof)
        np.save(DATA_PROCESSED / f"te_nb145_{name}.npy", te)
        if ratio >= COLLAPSE_THRESH and r < 0.289087:
            sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te})
            sub["SMILES"] = sub["Molecule Name"].map(smap)
            sub = sub[["SMILES", "Molecule Name", "pEC50"]]
            sub.to_csv(SUBMISSIONS / f"145_{name}.csv", index=False)
            saved.append(f"145_{name}.csv OOF={r:.4f}  ratio={ratio:.3f}")

    print(f"\n=== Saved as ensemble candidates. New submissions:\n  " + ('\n  '.join(saved) if saved else 'none beat nb224 (0.289087)'))


if __name__ == "__main__":
    main()
