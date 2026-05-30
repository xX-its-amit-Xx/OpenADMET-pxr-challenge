"""nb102 — Batch-1 Ensemble: blend nb97-101 OOF predictions.

Uses ElasticNetCV to find optimal weights across all batch-1 models
plus the prior best models (nb76 Delta-ML, nb62 grand v7).
Key question: does nb99 SC bio-fingerprint add signal on top of Delta-ML?
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, RidgeCV
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    r2    = 1 - np.sum((yt-yp)**2) / np.sum((yt-yt.mean())**2) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    if label:
        print(f"  [{label}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  R2={r2:.4f}  "
              f"r={pr:.4f}  rho={sp:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, R2=r2, Pearson=pr, Spearman=sp)


def load_oof_te(stem, label):
    oof_path = DATA_PROCESSED / f"oof_{stem}.npy"
    te_path  = DATA_PROCESSED / f"te_{stem}.npy"
    if not oof_path.exists():
        print(f"  MISSING OOF: {oof_path}")
        return None, None
    oof = np.load(oof_path)
    te  = np.load(te_path) if te_path.exists() else None
    print(f"  Loaded {label}: OOF shape={oof.shape}")
    return oof, te


def main():
    print("=== nb102: Batch-1 Grand Ensemble ===")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()

    # Load all batch-1 OOF predictions + prior best models
    models = [
        ("nb97_pxr_features",      "PXR Physics Features"),
        ("nb98_chembl_bayesian_fp", "ChEMBL Bayesian MACCS"),
        ("nb99_sc_bio_fp",          "SC Bio-Fingerprint"),
        ("nb100_emax_corrected",    "Emax Correction"),
        ("nb101_delta_base",        "Delta-ML Base"),
    ]

    # Also try to load prior grand ensemble OOF
    prior_models = []
    for stem in ["grand_v7", "nested_cv_ensemble"]:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        if oof_p.exists():
            prior_models.append((stem, stem))

    oofs, tes, names = [], [], []
    for stem, label in models + prior_models:
        oof, te_pred = load_oof_te(stem, label)
        if oof is not None and len(oof) == len(y_tr):
            valid = np.isfinite(oof)
            if valid.mean() > 0.9:
                r = rae(y_tr[valid], oof[valid])
                print(f"    -> OOF RAE = {r:.4f}")
                oofs.append(oof)
                tes.append(te_pred)
                names.append(label)

    if len(oofs) < 2:
        print("Not enough OOF predictions to blend. Exiting.")
        return

    # Stack OOF predictions
    O = np.column_stack(oofs)  # (N_train, n_models)
    print(f"\nBlending {len(names)} models: {names}")
    print(f"OOF matrix shape: {O.shape}")

    # Handle NaN by replacing with column mean
    for j in range(O.shape[1]):
        col = O[:, j]
        nan_mask = ~np.isfinite(col)
        if nan_mask.any():
            O[nan_mask, j] = np.nanmean(col)

    # ── Equal-weight average ──────────────────────────────────────────────────
    oof_equal = O.mean(axis=1)
    full_metrics(y_tr, oof_equal, "equal_weight_avg")

    # ── Best single model (Delta-ML) ─────────────────────────────────────────
    delta_idx = next((i for i, n in enumerate(names) if "Delta" in n), None)
    if delta_idx is not None:
        full_metrics(y_tr, O[:, delta_idx], f"delta_ml_standalone")

    # ── Ridge CV blending ────────────────────────────────────────────────────
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)
    oof_ridge = np.full(len(y_tr), np.nan)
    ridge_weights_list = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=3)
        ridge.fit(O[tr_idx], y_tr[tr_idx])
        oof_ridge[va_idx] = ridge.predict(O[va_idx])
        ridge_weights_list.append(ridge.coef_)
    full_metrics(y_tr, oof_ridge, "ridge_cv_blend")

    # ── ElasticNet CV blending ────────────────────────────────────────────────
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=[0.001, 0.01, 0.1, 1.0],
                      cv=5, max_iter=5000, random_state=SEED)
    en.fit(O, y_tr)
    oof_en = en.predict(O)
    full_metrics(y_tr, oof_en, "elasticnet_blend (in-sample)")

    # ── Delta-ML + SC Bio-FP two-model blend ─────────────────────────────────
    sc_idx = next((i for i, n in enumerate(names) if "SC" in n), None)
    if delta_idx is not None and sc_idx is not None:
        print("\nSweeping Delta-ML + SC Bio-FP blend weights:")
        best_alpha_2, best_rae_2 = 0.0, rae(y_tr, O[:, delta_idx])
        for alpha in np.arange(0.0, 1.01, 0.1):
            blend = (1 - alpha) * O[:, delta_idx] + alpha * O[:, sc_idx]
            r = rae(y_tr[np.isfinite(blend)], blend[np.isfinite(blend)])
            if r < best_rae_2:
                best_rae_2, best_alpha_2 = r, alpha
            print(f"  alpha={alpha:.1f}  RAE={r:.4f}")
        print(f"  Best alpha={best_alpha_2:.1f} (SC weight) RAE={best_rae_2:.4f}")
        oof_2blend = (1 - best_alpha_2) * O[:, delta_idx] + best_alpha_2 * O[:, sc_idx]
        full_metrics(y_tr, oof_2blend, f"delta+sc_blend(alpha={best_alpha_2:.1f})")

    # ── Final test predictions ────────────────────────────────────────────────
    print("\n=== Final test predictions ===")
    # Use the best strategy: Delta-ML dominant + small SC contribution if helpful
    Te = np.column_stack([t for t in tes if t is not None])

    # Strategy 1: Delta-ML alone (best single model)
    if delta_idx is not None and tes[delta_idx] is not None:
        te_delta = tes[delta_idx]
        print(f"  Delta-ML test: min={te_delta.min():.2f} med={np.median(te_delta):.2f} max={te_delta.max():.2f}")

    # Strategy 2: Best 2-model blend test predictions
    if delta_idx is not None and sc_idx is not None and tes[delta_idx] is not None and tes[sc_idx] is not None:
        te_2blend = (1-best_alpha_2)*tes[delta_idx] + best_alpha_2*tes[sc_idx]
        te_2blend = np.clip(te_2blend, y_tr.min()-0.5, y_tr.max()+0.5)
        sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_2blend})
        sub.to_csv(SUBMISSIONS / "102_delta_sc_blend.csv", index=False)
        print(f"  Saved 102_delta_sc_blend.csv")

    # Strategy 3: ElasticNet blend of all models
    if Te.shape[1] == len(names):
        te_en = en.predict(Te)
        te_en = np.clip(te_en, y_tr.min()-0.5, y_tr.max()+0.5)
        np.save(DATA_PROCESSED / "te_nb102_ensemble.npy", te_en)
        sub_en = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_en})
        sub_en.to_csv(SUBMISSIONS / "102_batch1_ensemble.csv", index=False)
        print(f"  Saved 102_batch1_ensemble.csv")
        print(f"  Test: min={te_en.min():.2f} med={np.median(te_en):.2f} max={te_en.max():.2f}")

    # Print model weights from ElasticNet
    print("\n=== ElasticNet model weights ===")
    for name, w in sorted(zip(names, en.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:35s}  weight={w:+.4f}")
    print(f"  intercept = {en.intercept_:.4f}")


if __name__ == "__main__":
    main()
