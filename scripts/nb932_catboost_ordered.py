"""nb932 -- CatBoost with ordered boosting (anti-leak gradient boosting).

Hypothesis: ordered boosting eliminates the prediction-shift target leakage that
affects standard boosting (LightGBM/XGBoost) on small datasets. With 4139 rows,
the bias correction CatBoost performs at every step should generalize better
than plain GBM, especially on the analog-expansion test set where scaffold-CV
already shows GBM optimism.

Pipeline:
  1) Combined features (Morgan 2048 + RDKit ~217) for train (4139) + test (513).
  2) Bemis-Murcko scaffold 5-fold CV.
  3) For each fold:
        CatBoostRegressor(iterations=2000, depth=8, lr=0.03,
                          loss=Huber:delta=2, bootstrap=Bernoulli,
                          subsample=0.85, random_strength=1.0,
                          boosting_type='Ordered',  # forced ordered boosting
                          allow_writing_files=False)
        fit on tr, predict val + test.
  4) OOF RAE; average test preds across 5 folds.
  5) in_RAE on 253 unblind rows.

Fallback: if catboost unavailable, use LightGBM with categorical target-encoded
features (NOT true ordered boosting; flagged in artifact).

Artifacts: C:/pxr_artifacts/nb932/
Wall-time budget: < 10 min.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

ART = Path("C:/pxr_artifacts/nb932")
ART.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
ITERATIONS = 2000
DEPTH = 8
LR = 0.03
WALL_BUDGET_S = 600  # 10 min


def try_catboost():
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor
    except Exception as e:
        print(f"catboost import failed: {e}")
        return None


def main():
    t0 = time.time()
    print("=== nb932: CatBoost ordered boosting ===")

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    print(f"train={len(tr)}  test={len(te)}")

    print("Featurizing (combined Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  "
          f"mem~{X_tr.nbytes/1e6:.1f}MB")

    # Scaffold-aware 5-fold CV indices
    print("Building scaffold 5-fold CV...")
    scaffolds = tr["smiles"].map(bemis_murcko).fillna("").tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=N_SPLITS, seed=SEED)
    print(f"  fold sizes: {[len(v) for _, v in folds]}")

    CatBoostRegressor = try_catboost()
    backend = "catboost" if CatBoostRegressor is not None else "lightgbm_fallback"
    print(f"backend = {backend}")

    oof = np.zeros(len(tr), dtype=np.float64)
    te_pred_acc = np.zeros(len(te), dtype=np.float64)
    fold_metrics = []

    boosting_used = None
    if backend == "catboost":
        # Try Ordered first; if it OOMs, fall back to Plain (still CatBoost, just no
        # per-permutation models). We try Ordered once at fold 0 to detect OOM
        # cheaply, then commit.
        boosting_used = "Ordered"
        for fi, (tr_idx, va_idx) in enumerate(folds):
            t_fold = time.time()
            elapsed = time.time() - t0
            if elapsed > WALL_BUDGET_S * 0.85:
                print(f"WARN: wall budget {elapsed:.0f}s; skipping remaining folds")
                # Average over folds we have
                break

            def make_model(boosting):
                return CatBoostRegressor(
                    iterations=ITERATIONS,
                    depth=DEPTH,
                    learning_rate=LR,
                    loss_function="Huber:delta=2",
                    bootstrap_type="Bernoulli",
                    subsample=0.85,
                    random_strength=1.0,
                    boosting_type=boosting,
                    max_ctr_complexity=1,
                    allow_writing_files=False,
                    random_seed=SEED + fi,
                    verbose=False,
                    thread_count=-1,
                )

            try:
                model = make_model(boosting_used)
                model.fit(
                    X_tr[tr_idx], y_tr[tr_idx],
                    eval_set=(X_tr[va_idx], y_tr[va_idx]),
                    early_stopping_rounds=100,
                    use_best_model=True,
                )
            except Exception as e:
                msg = str(e).lower()
                if "alloc" in msg or "memory" in msg:
                    print(f"  fold {fi}: {boosting_used} OOM ({e}); falling back to Plain")
                    boosting_used = "Plain"
                    model = make_model(boosting_used)
                    model.fit(
                        X_tr[tr_idx], y_tr[tr_idx],
                        eval_set=(X_tr[va_idx], y_tr[va_idx]),
                        early_stopping_rounds=100,
                        use_best_model=True,
                    )
                else:
                    raise

            va_pred = model.predict(X_tr[va_idx])
            oof[va_idx] = va_pred
            te_pred_acc += model.predict(X_te)

            f_rae = rae(y_tr[va_idx], va_pred)
            best_iter = model.get_best_iteration()
            fold_metrics.append({
                "fold": fi, "n_val": int(len(va_idx)),
                "val_rae": float(f_rae),
                "best_iter": int(best_iter) if best_iter is not None else -1,
                "fit_time_s": float(time.time() - t_fold),
            })
            print(f"  fold {fi}: val_RAE={f_rae:.4f}  best_iter={best_iter}  "
                  f"time={time.time()-t_fold:.1f}s")

    else:
        # Fallback: LightGBM with target-encoded categorical (Bemis-Murcko scaffold)
        # NOT true ordered boosting; documented in metrics.
        import lightgbm as lgb
        # Add scaffold target-encoding (smoothed mean) as a single extra feature
        scaff_ser = pd.Series(scaffolds, name="scaffold")
        glob_mean = y_tr.mean()
        for fi, (tr_idx, va_idx) in enumerate(folds):
            t_fold = time.time()
            elapsed = time.time() - t0
            if elapsed > WALL_BUDGET_S * 0.85:
                print(f"WARN: wall budget {elapsed:.0f}s; skipping remaining folds")
                break

            # Compute target-encoding within train fold only (no leak)
            df_tr = pd.DataFrame({"scaffold": scaff_ser.values[tr_idx],
                                  "y": y_tr[tr_idx]})
            agg = df_tr.groupby("scaffold")["y"].agg(["mean", "count"])
            # Smoothed mean (m=10): (n*mean + m*global) / (n + m)
            m = 10.0
            agg["te"] = (agg["count"] * agg["mean"] + m * glob_mean) / (agg["count"] + m)
            te_map = agg["te"].to_dict()

            te_feat_tr = np.array([te_map.get(s, glob_mean) for s in scaff_ser.values],
                                  dtype=np.float32).reshape(-1, 1)
            te_feat_te = np.full((len(te), 1), glob_mean, dtype=np.float32)
            # No test scaffold known here; column placeholder for consistency

            X_tr_f = np.hstack([X_tr, te_feat_tr])
            X_te_f = np.hstack([X_te, te_feat_te])

            model = lgb.LGBMRegressor(
                n_estimators=ITERATIONS,
                num_leaves=64,
                learning_rate=LR,
                objective="huber",
                alpha=2.0,
                bagging_fraction=0.85,
                bagging_freq=1,
                feature_fraction=0.85,
                random_state=SEED + fi,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(
                X_tr_f[tr_idx], y_tr[tr_idx],
                eval_set=[(X_tr_f[va_idx], y_tr[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            va_pred = model.predict(X_tr_f[va_idx])
            oof[va_idx] = va_pred
            te_pred_acc += model.predict(X_te_f)

            f_rae = rae(y_tr[va_idx], va_pred)
            fold_metrics.append({
                "fold": fi, "n_val": int(len(va_idx)),
                "val_rae": float(f_rae),
                "best_iter": int(model.best_iteration_ or -1),
                "fit_time_s": float(time.time() - t_fold),
            })
            print(f"  fold {fi}: val_RAE={f_rae:.4f}  best_iter={model.best_iteration_}  "
                  f"time={time.time()-t_fold:.1f}s")

    n_folds_done = len(fold_metrics)
    if n_folds_done == 0:
        raise RuntimeError("No folds completed within wall budget")
    te_pred = te_pred_acc / n_folds_done

    # OOF RAE only over rows actually scored
    scored_mask = np.zeros(len(tr), dtype=bool)
    for fi in range(n_folds_done):
        _, va_idx = folds[fi]
        scored_mask[va_idx] = True
    oof_rae = rae(y_tr[scored_mask], oof[scored_mask])
    print(f"\nOOF RAE (n={int(scored_mask.sum())}) = {oof_rae:.4f}")
    print(f"  te_pred mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    # Save artifacts
    np.save(ART / "oof.npy", oof)
    np.save(ART / "te_pred.npy", te_pred)
    np.save(DATA_PROCESSED / "te_nb932.npy", te_pred)
    np.save(DATA_PROCESSED / "oof_nb932.npy", oof)

    # In-RAE on 253 unblind
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te["name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb.loc[unb["Molecule Name"].isin(name_to_idx), "pEC50"].values
    in_rae = rae(unb_y, te_pred[unb_te_idx])
    print(f"Unblind n={len(unb_te_idx)}  in_RAE = {in_rae:.4f}")

    # Submission CSV
    sub = pd.DataFrame({
        "Molecule Name": te["name"].values,
        "SMILES": te["smiles"].values,
        "pEC50": te_pred,
    })
    out_csv = SUBMISSIONS / "nb932_catboost_ordered.csv"
    sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    metrics = {
        "backend": backend,
        "boosting_type": boosting_used,
        "ordered_boosting": backend == "catboost" and boosting_used == "Ordered",
        "n_folds_done": n_folds_done,
        "oof_rae": float(oof_rae),
        "in_rae": float(in_rae),
        "n_unblind": int(len(unb_te_idx)),
        "fold_metrics": fold_metrics,
        "wall_time_s": float(time.time() - t0),
    }
    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Done in {time.time()-t0:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
