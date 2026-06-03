"""nb1052 -- Non-linear meta-stacker (LGBM Huber on top of 5 OOFs + 5 physchem).

Pool (K=5 top PRE-unblind OOFs on 253 unblind):
    0. chemprop_aux       (PRIMARY-1, in_RAE 0.6216)
    1. nb972_long_train   (long-train Chemprop)
    2. nb914               (persistence-homology)
    3. nb960               (pseudo-label self-train)
    4. nb1030 Mordred LGBM

Plus 5 RDKit physchem features per compound: logp, mw, tpsa, fsp3,
formal_charge -- 10 total meta-features.

Hypothesis: a non-linear meta-stacker (shallow LGBM with Huber loss) may
extract interactions a linear SLSQP simplex cannot, e.g.
"trust chemprop_aux when logP < 3, else lean nb972". The risk is
overfitting at n=253 with 10 features; we mitigate with:
    - max_depth=4 (very shallow)
    - n_estimators=200
    - min_child_samples=15  (~6% of folds)
    - Huber loss alpha=2.0  (robust to leverage points)
    - Honest 5-fold cross-fit on the 253 ONLY -- the meta-features at
      deploy are the OOF predictions of each base on the 513 (te_*.npy),
      so there is no leak from the test-blind path.

Procedure:
    Build P_unb (253 x 5) from oof_*.npy of each base.
    Build X_phys_unb (253 x 5) from RDKit on unblind SMILES.
    Stack: X_meta_unb = [P_unb | X_phys_unb] -- (253, 10).
    5-fold KFold (seed=42) on 253:
        Fit LGBM(huber, a=2.0, depth=4, n_est=200, mcs=15) on 4 folds.
        Predict held-out fold; collect OOF.
    Pooled cross-fit RAE = honest meta-stack score.

Deploy:
    Fit one LGBM on the full 253 meta-frame.
    Build X_meta_513 = [preds_513 | X_phys_513] -- (513, 10).
    Predict 513.

Reference:
    nb1014 3-way SLSQP+stretch bag = 0.5930  (mean pooled CV).
    Strict beat tolerance = -0.005 (meta-stacker must hit <= 0.5880).

Outputs:
    data/processed/te_nb1052.npy
    data/processed/nb1052_summary.json
    submissions/nb1052_meta_stack_lgbm.csv
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

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.chem import compute_physchem
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1052"

# (display_name, oof_npy_stem, te_npy_stem, submission_csv_stem)
# oof_*.npy is on 4139 train OR 253 unblind; we'll handle both shapes.
CANDIDATES = [
    ("chemprop_aux",      "chemprop_aux",      "chemprop_aux",      "chemprop_aux"),
    ("nb972_long_train",  "nb972_long_train",  "nb972_long_train",  "nb972_long_train_optim"),
    ("nb914_persistence", "nb914",             "nb914",             "nb914_persistence_homology"),
    ("nb960_pseudo",      "nb960",             "nb960",             "nb960_pseudo_label_self_train"),
    ("nb1030_mordred",    "nb1030",            "nb1030",            "nb1030_mordred_lgbm"),
]
PHYS_KEYS = ["logp", "mw", "tpsa", "fsp3", "formal_charge"]

N_FOLDS = 5
SEED = 42
NB1014_REF = 0.5930  # mean pooled CV of nb1014 3-way bag

LGBM_PARAMS = dict(
    objective="huber",
    alpha=2.0,
    learning_rate=0.05,
    n_estimators=200,
    max_depth=4,
    num_leaves=15,           # cap leaves consistent with depth=4
    min_child_samples=15,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    verbosity=-1,
    random_state=SEED,
    n_jobs=2,
)


def load_te(stem: str, csv_stem: str, te_names: np.ndarray) -> np.ndarray:
    """Load the 513 deploy predictions of a base candidate."""
    npy = DATA_PROCESSED / f"te_{stem}.npy"
    if npy.exists():
        arr = np.load(npy).astype(np.float64)
        if arr.shape[0] == len(te_names):
            return arr
        print(f"   [warn] te_{stem}.npy shape {arr.shape}, expected "
              f"{len(te_names)}; falling back to csv")
    sub = pd.read_csv(SUBMISSIONS / f"{csv_stem}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{csv_stem}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def load_unb(oof_stem: str, te_arr_513: np.ndarray,
             unb_idx: np.ndarray, n_unb: int) -> np.ndarray:
    """Load 253-unblind predictions for a base candidate.

    Priority:
        1. oof_<stem>.npy with shape (n_unb,) -- direct unblind OOF.
        2. te_<stem>.npy[unb_idx]               -- deploy preds indexed.
    """
    oof = DATA_PROCESSED / f"oof_{oof_stem}.npy"
    if oof.exists():
        arr = np.load(oof).astype(np.float64)
        if arr.shape[0] == n_unb:
            print(f"   [load] oof_{oof_stem}.npy ({arr.shape}) used directly")
            return arr
        else:
            print(f"   [load] oof_{oof_stem}.npy shape {arr.shape}, "
                  f"!= n_unb={n_unb}; using te[unb_idx] instead")
    print(f"   [load] te_{oof_stem}.npy[unb_idx] used")
    return te_arr_513[unb_idx]


def build_physchem(smiles: np.ndarray) -> np.ndarray:
    """(N, 5) float array of [logp, mw, tpsa, fsp3, formal_charge]."""
    rows = []
    for smi in smiles:
        d = compute_physchem(smi)
        if d is None:
            rows.append([np.nan] * len(PHYS_KEYS))
        else:
            rows.append([float(d[k]) for k in PHYS_KEYS])
    X = np.asarray(rows, dtype=np.float64)
    # Median impute any NaN per-column
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.any(np.isnan(col)):
            med = float(np.nanmedian(col))
            col[np.isnan(col)] = med
            X[:, j] = col
    return X


def fit_predict_fold(X_tr, y_tr, X_va) -> np.ndarray:
    m = lgb.LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_tr, y_tr)
    return m.predict(X_va)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- non-linear meta-stack LGBM(Huber) on 5 OOFs + 5 physchem")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    preds_513 = np.column_stack([
        load_te(stem_te, csv_stem, te_names)
        for _, _, stem_te, csv_stem in CANDIDATES
    ])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    P_unb_cols = []
    for j, (name, oof_stem, te_stem, _) in enumerate(CANDIDATES):
        col = load_unb(oof_stem, preds_513[:, j], unb_idx, n_unb)
        P_unb_cols.append(col)
    P_unb = np.column_stack(P_unb_cols)
    print(f"[load] P_unb shape = {P_unb.shape}")

    # ---- Individual in_RAE on unblind ----
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, (name, *_rest) in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[name] = r
        print(f"   {name:24s}: {r:.4f}")

    # ---- Physchem features ----
    print("\n[phys] computing RDKit physchem on unblind + test...")
    unb_smiles = te_smiles[unb_idx]
    X_phys_unb = build_physchem(unb_smiles)
    X_phys_513 = build_physchem(te_smiles)
    print(f"   X_phys_unb shape = {X_phys_unb.shape}")
    print(f"   X_phys_513 shape = {X_phys_513.shape}")

    # ---- Stack meta-features ----
    X_meta_unb = np.column_stack([P_unb, X_phys_unb]).astype(np.float64)
    X_meta_513 = np.column_stack([preds_513, X_phys_513]).astype(np.float64)
    feat_names = [c[0] for c in CANDIDATES] + PHYS_KEYS
    print(f"\n[stack] X_meta_unb shape = {X_meta_unb.shape}  "
          f"(features: {feat_names})")

    # =================================================================
    # 5-fold honest cross-fit on 253
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD HONEST CROSS-FIT  (KFold seed={SEED}, "
          f"LGBM(huber alpha={LGBM_PARAMS['alpha']}, "
          f"depth={LGBM_PARAMS['max_depth']}, "
          f"n_est={LGBM_PARAMS['n_estimators']}, "
          f"mcs={LGBM_PARAMS['min_child_samples']}))")
    print("-" * 78)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = np.full(n_unb, np.nan)
    fold_records = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        X_tr, y_tr = X_meta_unb[tr_loc], y_unb[tr_loc]
        X_va, y_va = X_meta_unb[va_loc], y_unb[va_loc]
        pred_va = fit_predict_fold(X_tr, y_tr, X_va)
        oof_pred[va_loc] = pred_va
        rae_va = float(rae(y_va, pred_va))
        fold_records.append({
            "fold": k,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "val_rae": rae_va,
        })
        print(f"   fold {k}: n_tr={len(tr_loc)}, n_va={len(va_loc)}, "
              f"val_RAE={rae_va:.4f}")

    pooled_rae = float(rae(y_unb, oof_pred))
    fold_vals = [f["val_rae"] for f in fold_records]
    print(f"\n[cv] pooled cross-fit RAE = {pooled_rae:.4f}  "
          f"(fold mean {np.mean(fold_vals):.4f}, std {np.std(fold_vals):.4f})")
    print(f"[cv] vs nb1014 ref         = {NB1014_REF:.4f}")

    delta_vs_ref = pooled_rae - NB1014_REF
    if delta_vs_ref < -0.005:
        verdict = "BEATS_NB1014"
    elif abs(delta_vs_ref) <= 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"
    print(f"[verdict] delta = {delta_vs_ref:+.4f}  -> {verdict}")

    # =================================================================
    # Deploy: fit on full 253 meta-frame, predict 513
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (fit on all 253 meta-rows; predict 513)")
    print("-" * 78)
    deploy_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_model.fit(X_meta_unb, y_unb)
    deploy_513 = deploy_model.predict(X_meta_513).astype(np.float32)
    in_rae_train = float(rae(y_unb, deploy_model.predict(X_meta_unb)))
    print(f"   in-sample (train) RAE on 253 = {in_rae_train:.4f}  "
          "(overfit lower bound)")
    print(f"   te(513) mean / std            = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")

    # Importances
    importances = deploy_model.feature_importances_
    imp_pairs = sorted(zip(feat_names, importances.tolist()),
                       key=lambda x: -x[1])
    print("\n[importance] LGBM split-count importance (deploy fit):")
    for name, imp in imp_pairs:
        print(f"   {name:24s}: {int(imp)}")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_meta_stack_lgbm.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    summary = {
        "tag": TAG,
        "candidates": [c[0] for c in CANDIDATES],
        "phys_keys": PHYS_KEYS,
        "n_features": int(X_meta_unb.shape[1]),
        "n_unb": n_unb,
        "n_te": n_te,
        "lgbm_params": {k: (v if not isinstance(v, np.generic) else float(v))
                         for k, v in LGBM_PARAMS.items()},
        "n_folds": N_FOLDS,
        "seed": SEED,
        "indiv_in_rae": indiv_rae,
        "fold_records": fold_records,
        "pooled_cross_fit_rae": pooled_rae,
        "fold_mean_rae": float(np.mean(fold_vals)),
        "fold_std_rae": float(np.std(fold_vals)),
        "nb1014_reference": NB1014_REF,
        "delta_vs_nb1014": delta_vs_ref,
        "verdict": verdict,
        "in_sample_train_rae_overfit_bound": in_rae_train,
        "feature_importance": dict(imp_pairs),
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pool                    = {[c[0] for c in CANDIDATES]}")
    print(f"   physchem features       = {PHYS_KEYS}")
    print(f"   pooled cross-fit RAE    = {pooled_rae:.4f}")
    print(f"   nb1014 reference        = {NB1014_REF:.4f}")
    print(f"   delta                   = {delta_vs_ref:+.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   in-sample RAE (overfit) = {in_rae_train:.4f}")
    print(f"   wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("pooled_cross_fit_rae", "fold_mean_rae", "fold_std_rae",
              "delta_vs_nb1014", "verdict",
              "in_sample_train_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
