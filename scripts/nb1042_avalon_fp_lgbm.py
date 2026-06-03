"""nb1042 -- Avalon FP (512-bit) + LightGBM Huber alpha=2.0.

Avalon is a different substructure encoding than Morgan/MACCS. Hypothesis:
the predictions will be decorrelated enough (Pearson < 0.95) to nb972
(Morgan/RDKit-based) that we can bag with chemprop_aux for an LB gain
versus nb1014.

Pipeline:
  1. Load (cached) Avalon 512-bit FP for 4139 train + 513 test.
  2. LightGBM Huber alpha=2.0, n_est=1500, num_leaves=64, lr=0.03 on Avalon
     only -- NO Morgan, NO RDKit, NO assay decomp.
  3. Scaffold 5-fold CV; full refit for 513 deploy.
  4. Compute in_RAE on 253 unblind.
  5. Pearson(te_nb1042, te_nb972_long_train) on the 513.
  6. If Pearson < 0.95: 50/50 bag with chemprop_aux, save submission.

Outputs (always):
  data/processed/te_nb1042.npy
  data/processed/oof_nb1042.npy
  data/processed/nb1042_summary.json
  submissions/nb1042_avalon_fp_lgbm.csv

Conditional output (if Pearson < 0.95):
  submissions/nb1042_avalon_bag_chemprop.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1042"
SEED = 42
N_FOLDS = 5
PEARSON_THRESHOLD = 0.95
BAG_WEIGHT = 0.5  # 50/50 bag with chemprop_aux if Pearson < threshold

LGBM_PARAMS = dict(
    objective="huber",
    alpha=2.0,
    n_estimators=1500,
    num_leaves=64,
    learning_rate=0.03,
    min_child_samples=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.1,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)


def in_rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Avalon FP (512-bit) + LGBM Huber alpha=2.0")
    print("=" * 78)

    # ---- Load cached Avalon FPs ----
    tr_av_path = DATA_PROCESSED / "tr_avalon512.npy"
    te_av_path = DATA_PROCESSED / "te_avalon512.npy"
    assert tr_av_path.exists() and te_av_path.exists(), (
        "Avalon FP cache missing; expected tr_avalon512.npy + te_avalon512.npy")
    X_tr = np.load(tr_av_path).astype(np.float32)
    X_te = np.load(te_av_path).astype(np.float32)
    print(f"[load] X_tr={X_tr.shape}  X_te={X_te.shape}")
    assert X_tr.shape[1] == 512 and X_te.shape[1] == 512, "expected 512-bit FP"

    # ---- Load datasets / scaffolds / unblind labels ----
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    assert n_tr == X_tr.shape[0] == 4139, f"train mismatch: {n_tr}"
    assert len(te) == X_te.shape[0] == 513, f"test mismatch: {len(te)}"

    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    unblind_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unblind = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    assert len(unblind_idx) == len(y_unblind) == 253

    # ---- Scaffold 5-fold CV (OOF) ----
    print("\n[cv] scaffold 5-fold Huber LGBM on Avalon FP ...")
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        params = dict(LGBM_PARAMS)
        m = lgb.train(
            params,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        fr = rae(y_tr[va_idx], oof[va_idx])
        print(f"   fold {fold+1}: RAE={fr:.4f}  n_va={len(va_idx)}", flush=True)
    oof_rae = float(rae(y_tr, oof))
    print(f"[cv] OOF RAE = {oof_rae:.4f}")

    # ---- Final refit on full train for 513 deploy ----
    print("\n[deploy] full-train refit ...")
    params_full = dict(LGBM_PARAMS, n_estimators=1000)
    m_full = lgb.train(
        params_full,
        lgb.Dataset(X_tr, label=y_tr),
        callbacks=[lgb.log_evaluation(-1)],
    )
    te_pred = np.clip(
        m_full.predict(X_te), y_tr.min() - 0.5, y_tr.max() + 0.5
    ).astype(np.float64)
    print(f"   te(513) mean/std = {te_pred.mean():.3f} / {te_pred.std():.3f}")

    # ---- in_RAE on 253 unblind ----
    in_r = in_rae(y_unblind, te_pred[unblind_idx])
    print(f"[eval] in_RAE on 253 unblind = {in_r:.4f}")

    # ---- Pearson vs nb972 on 513 ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    pear, _ = pearsonr(te_pred, te_nb972)
    pear = float(pear)
    print(f"[pearson] te_nb1042 vs te_nb972_long_train = {pear:.4f}")

    # ---- nb1014 reference ----
    te_nb1014 = np.load(DATA_PROCESSED / "te_nb1014.npy").astype(np.float64)
    in_r_nb1014 = in_rae(y_unblind, te_nb1014[unblind_idx])
    print(f"[ref] in_RAE nb1014 = {in_r_nb1014:.4f}")

    # ---- Save base outputs ----
    np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred.astype(np.float32))
    plain = SUBMISSIONS / f"{TAG}_avalon_fp_lgbm.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_pred,
    }).to_csv(plain, index=False)
    print(f"[save] oof_{TAG}.npy")
    print(f"[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    # ---- Conditional bag with chemprop_aux ----
    bag_path = None
    in_r_bag = None
    beats_nb1014 = False
    if pear < PEARSON_THRESHOLD:
        print(f"\n[bag] Pearson {pear:.4f} < {PEARSON_THRESHOLD} -- "
              f"bagging 50/50 with chemprop_aux")
        te_chemprop = np.load(
            DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
        te_bag = (BAG_WEIGHT * te_chemprop
                  + (1.0 - BAG_WEIGHT) * te_pred)
        in_r_bag = in_rae(y_unblind, te_bag[unblind_idx])
        beats_nb1014 = bool(in_r_bag < in_r_nb1014)
        print(f"   in_RAE(bag) = {in_r_bag:.4f}  "
              f"(vs nb1014 {in_r_nb1014:.4f}: "
              f"{'BEATS' if beats_nb1014 else 'WORSE'})")
        bag_path = SUBMISSIONS / f"{TAG}_avalon_bag_chemprop.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": te_bag,
        }).to_csv(bag_path, index=False)
        np.save(DATA_PROCESSED / f"te_{TAG}_bag.npy",
                te_bag.astype(np.float32))
        print(f"[save] {bag_path}")
    else:
        print(f"\n[bag] Pearson {pear:.4f} >= {PEARSON_THRESHOLD} -- "
              f"SKIP bag (too correlated to nb972)")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "feature_set": "avalon_512",
        "lgbm": {
            "objective": "huber", "alpha": 2.0,
            "n_estimators": 1500, "num_leaves": 64, "lr": 0.03,
        },
        "oof_rae": oof_rae,
        "in_rae_253": in_r,
        "pearson_with_nb972": pear,
        "pearson_threshold": PEARSON_THRESHOLD,
        "bag_done": bag_path is not None,
        "bag_in_rae": in_r_bag,
        "nb1014_in_rae": in_r_nb1014,
        "beats_nb1014": beats_nb1014,
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "plain_submission": str(plain),
        "bag_submission": str(bag_path) if bag_path else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   OOF RAE              = {oof_rae:.4f}")
    print(f"   in_RAE (253 unblind) = {in_r:.4f}")
    print(f"   Pearson vs nb972     = {pear:.4f}")
    if in_r_bag is not None:
        print(f"   bag in_RAE (50/50)   = {in_r_bag:.4f}")
        print(f"   nb1014 in_RAE        = {in_r_nb1014:.4f}")
        print(f"   beats nb1014         = {beats_nb1014}")
    print(f"   wall                 = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()
