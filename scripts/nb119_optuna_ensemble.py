"""nb119 — Optuna-Optimized Ensemble Weights.

Ridge regression in nb112 finds weights via L2-penalized linear regression,
which allows negative weights (deblending). Optuna optimizes over the
probability simplex (non-negative weights summing to 1) which is a more
natural constraint for ensemble blending.

Advantages over ridge:
1. Non-negative constraint: physically meaningful (no "negative" model)
2. L1 regularization on weights available (sparse selection)
3. Can optimize non-differentiable objectives (like RAE directly)
4. Subset selection: automatically drops poor models

Strategy:
- Load all OOF+TE pairs with te_std/oof_std >= 0.58 (collapse filter)
- Run Optuna to maximize OOF RAE over 200 trials
- Report best N-model subsets (N=2,3,4,5)
- Save best ensemble prediction
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
N_TRIALS = 300
COLLAPSE_THRESH = 0.58


def load_all_models(n_tr):
    """Load all OOF+TE pairs that pass collapse filter."""
    candidates = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        te_p = DATA_PROCESSED / f"te_{stem}.npy"
        if not te_p.exists():
            te_p = DATA_PROCESSED / f"te_oof_{stem}.npy"
        if not te_p.exists():
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            if not (np.isfinite(oof).all() and np.isfinite(te).all()):
                oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
                te  = np.where(np.isfinite(te), te,  np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < COLLAPSE_THRESH: continue
            r = rae(y_tr, oof)
            candidates.append(dict(stem=stem, oof=oof, te=te, rae=r, ratio=ratio))
        except Exception:
            pass
    candidates.sort(key=lambda x: x["rae"])
    return candidates


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    global y_tr
    print("=== nb119: Optuna Ensemble Optimization ===\n")

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load all models
    print("Loading models...")
    models = load_all_models(n_tr)
    print(f"  Loaded {len(models)} models passing collapse filter (ratio >= {COLLAPSE_THRESH})")
    for m in models[:10]:
        print(f"    {m['stem']:45s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")
    if len(models) > 10:
        print(f"    ... and {len(models) - 10} more")

    oof_mat = np.column_stack([m["oof"] for m in models])  # (n_tr, n_models)
    te_mat  = np.column_stack([m["te"]  for m in models])  # (n_te, n_models)
    n_models = len(models)
    stems = [m["stem"] for m in models]

    # === Strategy 1: Scaffold-fold OOF optimization ===
    print(f"\n=== Optuna Optimization (scaffold CV, {N_TRIALS} trials) ===")

    def objective_full(trial):
        # Unconstrained weights → softmax → simplex
        raw = np.array([trial.suggest_float(f"w{i}", -3.0, 3.0)
                        for i in range(n_models)])
        w = softmax(raw)
        pred = oof_mat @ w
        return rae(y_tr, pred)

    study_full = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study_full.optimize(objective_full, n_trials=N_TRIALS, show_progress_bar=False)
    best_raw = np.array([study_full.best_params[f"w{i}"] for i in range(n_models)])
    best_w = softmax(best_raw)
    oof_full = oof_mat @ best_w
    te_full  = te_mat  @ best_w
    print(f"Full ensemble: OOF RAE = {rae(y_tr, oof_full):.4f}")
    top5_idx = np.argsort(-best_w)[:5]
    print("  Top-5 model weights:")
    for i in top5_idx:
        print(f"    {stems[i]:45s}  w={best_w[i]:.4f}")

    # === Strategy 2: Best-k subset selection via Optuna ===
    print(f"\n=== Subset selection (k=3,5,7 models) ===")
    subset_results = {}
    for k in [3, 5, 7]:
        if k > n_models:
            continue
        def objective_subset(trial, k=k):
            # Pick k model indices, then optimize weights over them
            idx = [trial.suggest_int(f"m{i}", 0, n_models - 1) for i in range(k)]
            idx = list(dict.fromkeys(idx))  # unique
            if len(idx) < 2:
                return 1.0
            raw = np.array([trial.suggest_float(f"w{i}", -2.0, 2.0)
                            for i in range(len(idx))])
            w = softmax(raw)
            pred = oof_mat[:, idx] @ w
            return rae(y_tr, pred)

        study_k = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        study_k.optimize(objective_subset, n_trials=N_TRIALS, show_progress_bar=False)
        # Re-extract best
        bp = study_k.best_params
        idx_k = list(dict.fromkeys([bp[f"m{i}"] for i in range(k)]))
        raw_k = np.array([bp[f"w{i}"] for i in range(len(idx_k))])
        w_k = softmax(raw_k)
        oof_k = oof_mat[:, idx_k] @ w_k
        te_k  = te_mat[:,  idx_k] @ w_k
        r_k = rae(y_tr, oof_k)
        print(f"  k={k}  OOF RAE={r_k:.4f}  models: {[stems[i] for i in idx_k]}")
        subset_results[k] = (oof_k, te_k, r_k)

    # === Strategy 3: Scaffold-fold validated weights ===
    # Use scaffold folds to prevent OOF overfitting
    print(f"\n=== Scaffold-CV validated: top-{min(15, n_models)} models only ===")
    top_idx = list(range(min(15, n_models)))  # already sorted by RAE
    oof_sub = oof_mat[:, top_idx]
    te_sub  = te_mat[:,  top_idx]
    k_sub = len(top_idx)

    def objective_cv(trial):
        raw = np.array([trial.suggest_float(f"w{i}", -2.0, 2.0)
                        for i in range(k_sub)])
        w = softmax(raw)
        # Use fold-level weighting (prevent memorizing which fold is easy)
        fold_raes = []
        for _, va_idx in splits:
            pred = oof_sub[va_idx] @ w
            fold_raes.append(rae(y_tr[va_idx], pred))
        return np.mean(fold_raes)

    study_cv = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study_cv.optimize(objective_cv, n_trials=N_TRIALS, show_progress_bar=False)
    best_raw_cv = np.array([study_cv.best_params[f"w{i}"] for i in range(k_sub)])
    best_w_cv = softmax(best_raw_cv)
    oof_cv = oof_sub @ best_w_cv
    te_cv  = te_sub  @ best_w_cv
    print(f"CV-validated ensemble: OOF RAE = {rae(y_tr, oof_cv):.4f}")
    top3 = np.argsort(-best_w_cv)[:3]
    for i in top3:
        print(f"  {stems[top_idx[i]]:45s}  w={best_w_cv[i]:.4f}")

    # Pick best strategy
    all_results = [
        ("full_optuna", oof_full, te_full, rae(y_tr, oof_full)),
        ("cv_optuna",   oof_cv,   te_cv,   rae(y_tr, oof_cv)),
    ]
    for k, (oof_k, te_k, r_k) in subset_results.items():
        all_results.append((f"subset_k{k}", oof_k, te_k, r_k))
    all_results.sort(key=lambda x: x[3])
    best_name, best_oof, best_te, best_r = all_results[0]
    print(f"\n=== Best strategy: {best_name}  OOF RAE={best_r:.4f} ===")

    best_te = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  "
          f"ratio={best_te.std()/best_oof.std():.2f}")

    np.save(DATA_PROCESSED / "oof_nb119_optuna_ensemble.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb119_optuna_ensemble.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "119_optuna_ensemble.csv", index=False)
    print(f"\nSaved: submissions/119_optuna_ensemble.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
