"""nb1023 -- XGBoost Huber orthogonal base for the nb1001 blend pool.

Hypothesis: nb972 is a slow-LR Huber LightGBM. Swapping the gradient
booster backend (LightGBM -> XGBoost) while keeping the loss shape
(Huber, alpha=2.0) and the slow-LR / deep / wide-patience recipe might
produce a base learner whose 513 predictions are *orthogonal* enough
(Pearson < 0.95) to nb972 to add real signal in the 3-way blend with
chemprop_aux.

Recipe (XGB equivalents of nb972):
  - XGBoost reg:pseudohubererror, huber_slope=2.0
  - max_depth=8, eta=0.005, n_estimators=2000,
    early_stopping_rounds=200
  - subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
    reg_lambda=0.2
  - Features: Morgan (2048) + RDKit desc (217) -> combined (2265)
  - Scaffold 5-fold CV on 4139 CRC; held-out fold serves as the
    early-stopping validation set per fold.
  - Predict 513 test; report in_RAE on 253 Phase-1 unblind index.
  - Compute Pearson( te_nb1023 , te_nb972 ) on the 513.

Decision rule:
  Pearson < 0.95 -> proceed to nb1001-style 3-way honest cross-fit
                    (chemprop_aux + nb972 + nb1023 + stretch).
  Pearson >= 0.95 -> redundant; skip blend.

Outputs:
  data/processed/oof_nb1023_xgb.npy
  data/processed/te_nb1023_xgb.npy
  data/processed/nb1023_summary.json
  submissions/nb1023_xgb_ortho_base.csv
  (conditional) data/processed/te_nb1023_blend.npy
  (conditional) submissions/nb1023_blend_crossfit.csv
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1023"
SEED = 42
N_FOLDS = 5

# XGB equivalents of nb972's LightGBM Huber recipe.
XGB_PARAMS = dict(
    objective="reg:pseudohubererror",
    huber_slope=2.0,
    max_depth=8,
    eta=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=0.2,
    tree_method="hist",
    nthread=4,
    seed=SEED,
    verbosity=0,
)
N_ESTIMATORS = 2000
EARLY_STOP = 200

PEARSON_THRESHOLD = 0.95
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
CV_SEED = 42  # for blend cross-fit


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit w on the simplex minimizing SSE of P @ w vs y."""
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    w0 = np.full(K, 1.0 / K)
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        w0,
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return np.asarray(res.x, dtype=float)


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- XGBoost Huber slow-LR orthogonal base")
    print("=" * 78)

    # ---- Truth / unblind for in_RAE ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    assert len(unb_idx) == len(y_unb) == 253

    # ---- Data ----
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    te_names = te["name"].values
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Computing combined features (Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

    # ---- Scaffold CV with long-train + early stop ----
    oof = np.full(n_tr, np.nan)
    best_iters = []
    fold_raes = []
    print(f"\nTraining {N_FOLDS} folds (eta={XGB_PARAMS['eta']}, "
          f"max_iter={N_ESTIMATORS}, patience={EARLY_STOP})...")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        base = float(np.mean(y_tr[tr_idx]))
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y_tr[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y_tr[va_idx])
        params_fold = dict(XGB_PARAMS, base_score=base)
        booster = xgb.train(
            params_fold,
            dtr,
            num_boost_round=N_ESTIMATORS,
            evals=[(dva, "va")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )
        best_it = int(booster.best_iteration) + 1  # 0-indexed -> count
        oof[va_idx] = booster.predict(dva, iteration_range=(0, best_it))
        fr = rae(y_tr[va_idx], oof[va_idx])
        fold_raes.append(fr)
        best_iters.append(best_it)
        elapsed = time.time() - t0
        print(f"  fold {fold+1}  best_iter={best_it:5d}  "
              f"RAE={fr:.4f}  elapsed={elapsed:6.1f}s", flush=True)

    oof_rae = float(rae(y_tr, oof))
    mean_best = int(np.mean(best_iters))
    print(f"\nOOF RAE = {oof_rae:.4f}")
    print(f"mean best_iteration across folds = {mean_best}")

    # ---- Final fit on full train, capped at mean best_iter ----
    print(f"\nFinal fit on full 4139 train, num_boost_round={mean_best}...")
    base_full = float(np.mean(y_tr))
    dall = xgb.DMatrix(X_tr, label=y_tr)
    dte = xgb.DMatrix(X_te)
    final_booster = xgb.train(
        dict(XGB_PARAMS, base_score=base_full),
        dall,
        num_boost_round=mean_best,
        verbose_eval=False,
    )
    te_preds = np.clip(
        final_booster.predict(dte),
        y_tr.min() - 0.5, y_tr.max() + 0.5,
    ).astype(np.float64)
    ratio = float(te_preds.std() / oof.std()) if oof.std() > 0 else 0.0
    in_r = in_rae(y_unb, te_preds[unb_idx])
    print(f"TEST  med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  "
          f"ratio(te/oof)={ratio:.2f}")
    print(f"in_RAE(253) = {in_r:.4f}")

    # ---- Persist base ----
    np.save(DATA_PROCESSED / f"oof_{TAG}_xgb.npy", oof)
    np.save(DATA_PROCESSED / f"te_{TAG}_xgb.npy", te_preds.astype(np.float32))
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": te_preds,
    }).to_csv(SUBMISSIONS / f"{TAG}_xgb_ortho_base.csv", index=False)

    # =================================================================
    # Pearson(te_nb1023, te_nb972) on the 513
    # =================================================================
    te_nb972 = load_te("nb972_long_train", te_names)
    pearson_513 = float(np.corrcoef(te_preds, te_nb972)[0, 1])
    print("\n" + "-" * 78)
    print(f"ORTHOGONALITY  Pearson(te_nb1023, te_nb972) on 513 = "
          f"{pearson_513:.4f}")
    print(f"threshold for ortho-add = {PEARSON_THRESHOLD}")
    print("-" * 78)

    summary = {
        "tag": TAG,
        "params": {k: v for k, v in XGB_PARAMS.items() if k != "verbosity"},
        "n_estimators_cap": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOP,
        "fold_best_iters": best_iters,
        "fold_raes": [float(x) for x in fold_raes],
        "mean_best_iter": mean_best,
        "oof_rae": oof_rae,
        "in_rae_253": float(in_r),
        "test_std": float(te_preds.std()),
        "te_oof_std_ratio": ratio,
        "pearson_with_nb972_513": pearson_513,
        "pearson_threshold": PEARSON_THRESHOLD,
    }

    # =================================================================
    # Conditional: 3-way blend honest cross-fit, nb1001-style
    # =================================================================
    if pearson_513 < PEARSON_THRESHOLD:
        print("\n*** ORTHOGONAL: running 3-way blend cross-fit ***")
        CANDIDATES = ["chemprop_aux", "nb972_long_train", f"{TAG}_xgb"]
        preds_513 = np.column_stack([
            load_te("chemprop_aux", te_names),
            te_nb972,
            te_preds,
        ])
        P_unb = preds_513[unb_idx]
        n_unb = len(y_unb)

        indiv = {}
        print("\n[indiv] in_RAE on 253 unblind:")
        for j, c in enumerate(CANDIDATES):
            r = float(rae(y_unb, P_unb[:, j]))
            indiv[c] = r
            print(f"   {c:30s}: {r:.4f}")

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        oof_blend = np.full(n_unb, np.nan)
        fold_rows = []
        for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
            w = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
            blend_tr = P_unb[tr_loc] @ w
            mu_tr = float(blend_tr.mean())
            s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
            blend_va = P_unb[va_loc] @ w
            oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
            rae_va = float(rae(y_unb[va_loc], oof_blend[va_loc]))
            fold_rows.append({
                "fold": k, "w": [float(x) for x in w],
                "s": s_f, "mu_tr": mu_tr,
                "train_rae": float(rae_tr), "val_rae": rae_va,
                "n_va": int(len(va_loc)),
            })
            print(f"   fold {k}: w={np.round(w,3).tolist()}  s={s_f:.2f}  "
                  f"mu_tr={mu_tr:.3f}  train_RAE={rae_tr:.4f}  "
                  f"val_RAE={rae_va:.4f}")

        pooled_rae = float(rae(y_unb, oof_blend))
        print(f"\n[cv] pooled honest 3-way cross-fit RAE on 253 = "
              f"{pooled_rae:.4f}")

        # Deploy: refit on all 253
        w_deploy = slsqp_simplex(P_unb, y_unb)
        blend_unb_all = P_unb @ w_deploy
        mu_deploy = float(blend_unb_all.mean())
        s_deploy, _ = best_stretch_on(blend_unb_all, y_unb, mu_deploy)
        in_rae_final = float(rae(
            y_unb,
            mu_deploy + s_deploy * (blend_unb_all - mu_deploy),
        ))
        print("\n[deploy] refit on all 253:")
        print(f"   w (chemprop_aux, nb972, nb1023) = "
              f"{np.round(w_deploy, 4).tolist()}")
        print(f"   mu = {mu_deploy:.4f}   s = {s_deploy:.2f}")
        print(f"   in-sample RAE (253) = {in_rae_final:.4f}")

        blend_513 = preds_513 @ w_deploy
        deploy_513 = (mu_deploy + s_deploy * (blend_513 - mu_deploy))
        deploy_513 = deploy_513.astype(np.float32)
        np.save(DATA_PROCESSED / f"te_{TAG}_blend.npy", deploy_513)
        plain = SUBMISSIONS / f"{TAG}_blend_crossfit.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te_names,
            "pEC50": deploy_513,
        }).to_csv(plain, index=False)
        print(f"\n[save] te_{TAG}_blend.npy")
        print(f"[save] {plain}")

        summary.update({
            "blend_candidates": CANDIDATES,
            "blend_indiv_in_rae": indiv,
            "blend_fold_results": fold_rows,
            "blend_crossfit_rae_253": pooled_rae,
            "blend_deploy_weights": [float(x) for x in w_deploy],
            "blend_deploy_mu": mu_deploy,
            "blend_deploy_s": float(s_deploy),
            "blend_in_sample_rae_overfit_bound": in_rae_final,
            "blend_deploy_te_mean": float(deploy_513.mean()),
            "blend_deploy_te_std": float(deploy_513.std()),
            "blend_plain_submission": str(plain),
        })
    else:
        print("\n*** REDUNDANT: Pearson >= threshold; skipping blend. ***")
        summary["blend_skipped_reason"] = (
            f"Pearson {pearson_513:.4f} >= {PEARSON_THRESHOLD}")

    summary["wall_sec"] = round(time.time() - t0, 2)
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    print(f"wall_time = {summary['wall_sec']:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    keys = [
        "oof_rae", "in_rae_253", "pearson_with_nb972_513",
        "blend_crossfit_rae_253", "blend_deploy_weights",
        "blend_deploy_s", "blend_in_sample_rae_overfit_bound",
    ]
    for k in keys:
        print(f"  {k}: {res.get(k)}")
