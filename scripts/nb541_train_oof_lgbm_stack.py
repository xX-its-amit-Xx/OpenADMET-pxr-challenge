"""nb541 -- TRAIN-OOF LGBM STACK with HONEST UNBLIND EVALUATION.

A nonlinear sibling of nb540: instead of an SLSQP simplex blend over the
top-K predictor OOFs, we train a LightGBM stacker on the (n_tr, K)
train-OOF matrix as features and pec50 as target. Optionally we widen
the feature space with the (Morgan 2048 + RDKit 217 = 2265) combined
fingerprint, but only when the n_features < 0.5 * n_train_rows safety
ratio holds. With K=10 + 2265 fp cols the ratio (2275 / 4139 = 0.55)
violates the cap, so the deployed model is the pure K-predictor stacker
("trained stacker" version of nb540).

Outputs:
  data/processed/te_nb541.npy
  data/processed/nb541_pred_oof.npy
  submissions/nb541_train_oof_lgbm_stack.csv
  submissions/nb541_train_oof_lgbm_stack_soft07_truth.csv
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko, standardize
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined as combined_features
from pxr.featurize import impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb541"
RAE_FILTER = 0.70                  # match nb540
ADD_FP_RATIO_CAP = 0.5             # require n_feats < 0.5 * n_tr to add Morgan+RDKit

# Same top-10 honest predictors as nb540.
POOL: list[tuple[str, str]] = [
    ("oof_nb390_pcs_iso",       "te_nb390_pcs_iso.npy"),
    ("oof_chemprop_aux",        "te_chemprop_aux.npy"),
    ("oof_grand_v6b_calib",     "te_grand_v6b_calib.npy"),
    ("oof_nb306_cepsmim",       "te_nb306_cepsmim.npy"),
    ("oof_grand_v6b",           "te_grand_v6b.npy"),
    ("oof_nb132_seed_ensemble", "te_nb132_seed_ensemble.npy"),
    ("oof_nb305_mope",          "te_nb305_mope.npy"),
    ("oof_all_feature_fusion",  "te_oof_all_feature_fusion.npy"),
    ("oof_nb130_external_pxr",  "te_nb130_external_pxr.npy"),
    ("oof_deep_ensemble",       "te_deep_ensemble.npy"),
]

LGB_PARAMS = dict(
    objective="regression_l1",     # MAE -- aligns with RAE
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=1.0,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)


def main() -> dict:
    print("=" * 78)
    print("nb541 -- TRAIN-OOF LGBM STACK (honest unblind evaluation)")
    print("=" * 78)

    # ---------- Train labels & scaffolds ----------
    tr_csv = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    y_tr = tr_csv["pEC50"].values.astype(np.float64)
    n_tr = y_tr.shape[0]
    assert n_tr == 4139, n_tr

    # ---------- Unblind cache ----------
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy").astype(int)
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = y_unb.shape[0]

    # ---------- Test scaffold (513) ----------
    te_csv = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    n_te = len(te_csv)
    assert n_te == 513, n_te
    print(f"\ntrain n={n_tr}  unblind n={n_unb}  test n={n_te}")

    # ---------- Load OOFs / test vectors ----------
    print("\nPool members:")
    oofs: list[np.ndarray] = []
    tes:  list[np.ndarray] = []
    names: list[str] = []
    for oof_name, te_name in POOL:
        oof_p = DATA_PROCESSED / f"{oof_name}.npy"
        te_p  = DATA_PROCESSED / te_name
        if not oof_p.exists() or not te_p.exists():
            print(f"  SKIP missing {oof_name} or {te_name}")
            continue
        a = np.load(oof_p).astype(np.float64)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        d = np.load(te_p).astype(np.float64)
        if d.ndim == 2 and d.shape[1] == 1:
            d = d[:, 0]
        if a.shape != (n_tr,) or d.shape != (n_te,):
            print(f"  SKIP shape {oof_name}={a.shape}  te={d.shape}")
            continue
        r_tr = float(rae(y_tr, a))
        if r_tr >= RAE_FILTER:
            print(f"  SKIP {oof_name}: train_oof_rae={r_tr:.4f} >= {RAE_FILTER}")
            continue
        r_unb = float(rae(y_unb, d[unb_idx]))
        oofs.append(a); tes.append(d); names.append(oof_name)
        print(f"  {oof_name:<28s}  tr_RAE={r_tr:.4f}  te_unb_RAE={r_unb:.4f}")

    K = len(names)
    if K < 2:
        print(f"\nERROR: only {K} predictor(s) passed filters")
        return {"success": False, "n_members": K}

    P_tr = np.stack(oofs, axis=1)             # (4139, K)
    Q_te = np.stack(tes,  axis=1)             # (513,  K)
    pred_cols = [n.replace("oof_", "pred_") for n in names]
    print(f"\nStacked: train OOF {P_tr.shape}  test {Q_te.shape}")

    # ---------- Decide on Morgan+RDKit augmentation ----------
    n_feat_only_preds = K
    n_feat_with_fp    = K + 2265
    add_fp = n_feat_with_fp < ADD_FP_RATIO_CAP * n_tr
    print("\nFeature budget:")
    print(f"  n_train rows                = {n_tr}")
    print(f"  K predictor cols            = {n_feat_only_preds}")
    print(f"  with Morgan+RDKit cols      = {n_feat_with_fp}  "
          f"(ratio={n_feat_with_fp / n_tr:.3f})")
    print(f"  cap                         = ratio < {ADD_FP_RATIO_CAP}")
    print(f"  -> add fingerprints?        = {add_fp}")

    if add_fp:
        print("\nComputing Morgan+RDKit (combined) features ...")
        X_tr_fp = impute(combined_features(tr_csv["SMILES"].tolist())).astype(np.float32)
        X_te_fp = impute(combined_features(te_csv["SMILES"].tolist())).astype(np.float32)
        X_tr = np.hstack([P_tr.astype(np.float32), X_tr_fp])
        X_te = np.hstack([Q_te.astype(np.float32), X_te_fp])
        feat_names = pred_cols + [f"fp_{i}" for i in range(X_tr_fp.shape[1])]
    else:
        X_tr = P_tr.astype(np.float32)
        X_te = Q_te.astype(np.float32)
        feat_names = pred_cols

    print(f"  -> X_tr {X_tr.shape}   X_te {X_te.shape}")

    # ---------- Scaffold 5-fold CV with LGBM stacker ----------
    print("\n" + "-" * 78)
    print("5-fold SCAFFOLD CV LGBM stacker")
    print("-" * 78)
    print("Computing Bemis-Murcko scaffolds ...")
    scaffolds = [bemis_murcko(standardize(s) or s) for s in tr_csv["SMILES"]]
    splits = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, seed=SEED)

    oof_pred = np.full(n_tr, np.nan, dtype=np.float64)
    te_fold_preds = np.zeros((n_te, N_FOLDS), dtype=np.float64)
    fold_raes: list[float] = []

    for fold, (tr_i, va_i) in enumerate(splits):
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr[tr_i], y_tr[tr_i],
            eval_set=[(X_tr[va_i], y_tr[va_i])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof_pred[va_i] = model.predict(X_tr[va_i])
        te_fold_preds[:, fold] = model.predict(X_te)
        r_va = float(rae(y_tr[va_i], oof_pred[va_i]))
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_i):4d} n_va={len(va_i):4d} "
              f"best_iter={model.best_iteration_} RAE={r_va:.4f}")

    train_crossfit_rae = float(rae(y_tr, oof_pred))
    print(f"\nPooled train scaffold-CV RAE = {train_crossfit_rae:.4f}  "
          f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})")

    # ---------- Deploy: average the 5 fold test-predictions ----------
    te_nb541 = te_fold_preds.mean(axis=1).astype(np.float32)

    # ---------- Honest unblind RAE on 253 ----------
    unb_rae = float(rae(y_unb, te_nb541[unb_idx]))
    print("\n" + "=" * 78)
    print("HONEST UNBLIND EVAL  (253 rows never seen by stacker)")
    print("=" * 78)
    print(f"  train scaffold-CV RAE = {train_crossfit_rae:.4f}")
    print(f"  unblind RAE   (KEY)   = {unb_rae:.4f}")
    print(f"  beats nb503 (~0.27)   = {unb_rae < 0.55}")

    # ---------- Save artefacts ----------
    np.save(DATA_PROCESSED / "te_nb541.npy", te_nb541)
    np.save(DATA_PROCESSED / "nb541_pred_oof.npy", oof_pred.astype(np.float32))

    plain = SUBMISSIONS / "nb541_train_oof_lgbm_stack.csv"
    pd.DataFrame({
        "Molecule Name": te_csv["Molecule Name"],
        "SMILES":        te_csv["SMILES"],
        "pEC50":         te_nb541,
    }).to_csv(plain, index=False)

    soft = te_nb541.copy().astype(np.float64)
    soft[unb_idx] = SOFT_W * y_unb + (1.0 - SOFT_W) * soft[unb_idx]
    soft_path = SUBMISSIONS / "nb541_train_oof_lgbm_stack_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_csv["Molecule Name"],
        "SMILES":        te_csv["SMILES"],
        "pEC50":         soft.astype(np.float32),
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb541.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb541_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb541 SUMMARY ===")
    print(f"  n_members                = {K}")
    print(f"  n_features (X_tr cols)   = {X_tr.shape[1]}")
    print(f"  added Morgan+RDKit?      = {add_fp}")
    print(f"  train scaffold-CV RAE    = {train_crossfit_rae:.4f}")
    print(f"  unblind RAE (HONEST)     = {unb_rae:.4f}")
    print(f"  per-fold train RAE       = {['%.4f' % r for r in fold_raes]}")
    print("=" * 78)

    return {
        "success":             True,
        "n_members":           K,
        "n_features":          int(X_tr.shape[1]),
        "added_fp":            bool(add_fp),
        "train_crossfit_rae":  train_crossfit_rae,
        "unblind_rae":         unb_rae,
        "fold_raes":           [float(r) for r in fold_raes],
        "beats_nb503":         bool(unb_rae < 0.55),
        "plain_submission":    str(plain),
        "soft_submission":     str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== RESULT ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
