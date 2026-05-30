"""nb108 — Grand Ensemble v2: all batch-1 + batch-2 OOF models.

Combines:
  nb107_assay_decomp    (0.3785) — stacked Delta+SC+Emax+Selectivity
  nb101_delta_base      (0.4164) — Delta-ML template regression
  nb99_sc_bio_fp        (0.5249) — SC neighborhood biological fingerprint
  nb103_seed_propagation(0.4196) — seed-anchored MMP propagation
  nested_cv_ensemble    (0.4108) — prior nested CV grand ensemble
  grand_v7              (0.5189) — prior grand ensemble

Best strategy: Ridge/ElasticNet blend, then manual tuning.
nb107 dominates but others add diversity from orthogonal approaches.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, ElasticNetCV
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
        print(f"  [{label:40s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  "
              f"R2={r2:.4f}  r={pr:.4f}  rho={sp:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, R2=r2, Pearson=pr, Spearman=sp)


def load_pair(stem):
    oof = DATA_PROCESSED / f"oof_{stem}.npy"
    te  = DATA_PROCESSED / f"te_{stem}.npy"
    if oof.exists():
        return np.load(oof), (np.load(te) if te.exists() else None)
    return None, None


def main():
    print("=== nb108: Grand Ensemble v2 ===\n")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ── Load all OOF / test predictions ──────────────────────────────────────
    candidates = [
        ("nb107_assay_decomp",       "AssayDecomp(Delta+SC+Emax+Sel)"),
        ("nb101_delta_base",         "Delta-ML base"),
        ("nb99_sc_bio_fp",           "SC Bio-Fingerprint"),
        ("nb103_seed_propagation",   "Seed SAR Propagation"),
        ("nested_cv_ensemble",       "Nested CV Ensemble (prior)"),
        ("grand_v7",                 "Grand v7 (prior)"),
        ("nb97_pxr_features",        "PXR Physics Features"),
        ("nb98_chembl_bayesian_fp",  "ChEMBL Bayesian MACCS"),
        ("nb100_emax_corrected",     "Emax Correction"),
    ]

    oofs, tes, names = [], [], []
    print("Loading OOF predictions:")
    for stem, label in candidates:
        oof, te_pred = load_pair(stem)
        if oof is not None and len(oof) == len(y_tr):
            valid = np.isfinite(oof)
            if valid.mean() > 0.9:
                oof_filled = oof.copy()
                oof_filled[~valid] = np.nanmean(oof)
                r = rae(y_tr, oof_filled)
                print(f"  {label:42s}  OOF RAE={r:.4f}")
                oofs.append(oof_filled)
                tes.append(te_pred)
                names.append(label)

    O = np.column_stack(oofs)
    print(f"\nBlending {len(names)} models, OOF matrix: {O.shape}")

    # ── Correlation analysis ─────────────────────────────────────────────────
    print("\nPairwise OOF correlations (top 3 vs best):")
    best_idx = 0  # AssayDecomp is best
    for j in range(1, min(5, len(names))):
        r, _ = stats.pearsonr(O[:, best_idx], O[:, j])
        print(f"  {names[best_idx]} vs {names[j]}: r={r:.3f}")

    # ── Single-model baselines ────────────────────────────────────────────────
    print("\nSingle-model baselines:")
    for j, name in enumerate(names):
        full_metrics(y_tr, O[:, j], name)

    # ── Sweep nb107 + nested_cv blend ────────────────────────────────────────
    print("\nSweeping AssayDecomp + NestedCV blend:")
    nc_idx = next((i for i, n in enumerate(names) if "Nested" in n), None)
    if nc_idx is not None:
        best_alpha, best_r = 0.0, rae(y_tr, O[:, 0])
        for alpha in np.arange(0.0, 0.51, 0.05):
            blend = (1-alpha)*O[:, 0] + alpha*O[:, nc_idx]
            r = rae(y_tr, blend)
            if r < best_r:
                best_r, best_alpha = r, alpha
            print(f"  alpha(NestedCV)={alpha:.2f}  RAE={r:.4f}")
        print(f"  Best: alpha={best_alpha:.2f}  RAE={best_r:.4f}")

    # ── Sweep nb107 + Delta-ML blend ─────────────────────────────────────────
    print("\nSweeping AssayDecomp + DeltaML blend:")
    dm_idx = next((i for i, n in enumerate(names) if "Delta-ML base" in n), None)
    if dm_idx is not None:
        best_alpha_dm, best_r_dm = 0.0, rae(y_tr, O[:, 0])
        for alpha in np.arange(0.0, 0.51, 0.05):
            blend = (1-alpha)*O[:, 0] + alpha*O[:, dm_idx]
            r = rae(y_tr, blend)
            if r < best_r_dm:
                best_r_dm, best_alpha_dm = r, alpha
            print(f"  alpha(DeltaML)={alpha:.2f}  RAE={r:.4f}")
        print(f"  Best: alpha={best_alpha_dm:.2f}  RAE={best_r_dm:.4f}")

    # ── RidgeCV scaffold blend ────────────────────────────────────────────────
    print("\nRidgeCV scaffold blend:")
    oof_ridge = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3)
        ridge.fit(O[tr_idx], y_tr[tr_idx])
        oof_ridge[va_idx] = ridge.predict(O[va_idx])
    full_metrics(y_tr, oof_ridge, "RidgeCV blend")

    # ── ElasticNet blend (in-sample) ─────────────────────────────────────────
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=[0.001, 0.01, 0.1, 1.0],
                      cv=5, max_iter=5000, random_state=SEED)
    en.fit(O, y_tr)
    oof_en = en.predict(O)
    full_metrics(y_tr, oof_en, "ElasticNet blend (in-sample)")
    print("\nElasticNet weights:")
    for name, w in sorted(zip(names, en.coef_), key=lambda x: abs(x[1]), reverse=True):
        if abs(w) > 0.001:
            print(f"  {name:42s}  weight={w:+.4f}")
    print(f"  intercept={en.intercept_:.4f}")

    # ── Test predictions ─────────────────────────────────────────────────────
    print("\n=== Test predictions ===")
    Te = np.column_stack([t if t is not None else np.zeros(len(te)) for t in tes])

    # Strategy 1: AssayDecomp alone (best single OOF)
    te_best = tes[0]
    if te_best is not None:
        te_best = np.clip(te_best, y_tr.min()-0.5, y_tr.max()+0.5)
        sub1 = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_best})
        sub1.to_csv(SUBMISSIONS / "108_assay_decomp_alone.csv", index=False)
        print(f"  Saved 108_assay_decomp_alone.csv")

    # Strategy 2: Best 2-model blend (AssayDecomp + NestedCV or DeltaML)
    if nc_idx is not None and tes[nc_idx] is not None and tes[0] is not None:
        te_2b = np.clip(
            (1-best_alpha)*tes[0] + best_alpha*tes[nc_idx],
            y_tr.min()-0.5, y_tr.max()+0.5
        )
        sub2 = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_2b})
        sub2.to_csv(SUBMISSIONS / "108_decomp_nested_blend.csv", index=False)
        print(f"  Saved 108_decomp_nested_blend.csv  (alpha_NestedCV={best_alpha:.2f})")

    # Strategy 3: RidgeCV (fit on full OOF, predict test)
    ridge_final = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_final.fit(O, y_tr)
    te_ridge = np.clip(ridge_final.predict(Te), y_tr.min()-0.5, y_tr.max()+0.5)
    sub3 = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_ridge})
    sub3.to_csv(SUBMISSIONS / "108_ridge_grand_ensemble.csv", index=False)
    print(f"  Saved 108_ridge_grand_ensemble.csv")
    print(f"  Ridge test: min={te_ridge.min():.2f}  med={np.median(te_ridge):.2f}  max={te_ridge.max():.2f}")

    # Save best for downstream use
    np.save(DATA_PROCESSED / "oof_nb108_grand_v2.npy", oof_ridge)
    np.save(DATA_PROCESSED / "te_nb108_grand_v2.npy", te_ridge)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BATCH-1 + BATCH-2 RESULTS SUMMARY")
    print("="*60)
    rows = []
    for j, name in enumerate(names):
        r = rae(y_tr, O[:, j])
        rows.append({"Model": name, "OOF_RAE": r})
    rows.append({"Model": "RidgeCV blend", "OOF_RAE": rae(y_tr[np.isfinite(oof_ridge)], oof_ridge[np.isfinite(oof_ridge)])})
    df = pd.DataFrame(rows).sort_values("OOF_RAE")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
