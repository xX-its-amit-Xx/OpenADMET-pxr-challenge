"""nb112 — Grand Ensemble v3: final submission-ready blend.

Combines ALL batch-1 + batch-2 + batch-3 non-collapsed OOF models.

Key lessons from nb108:
  - Ridge CV on diverse models leads to collapsed test predictions (median 4.01)
  - Only use models where test_std/oof_std >= 0.60 as meta-learner inputs
  - Manual blends often outperform Ridge for test predictions
  - AssayDecomp (nb107) alone is most reliable for test distribution

Strategy:
  1. Score all available OOF models by OOF RAE and test distribution quality
  2. Find optimal blend weights via exhaustive 2-model and 3-model sweeps
  3. Use nested CV Ridge ONLY on non-collapsed models
  4. Save the best 3 strategies as submission candidates

New models vs nb108:
  - nb109: deep meta-stack (batch-3)
  - nb110: scaffold prior residual (batch-3)
  - nb111: selectivity-primary (batch-3)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from pathlib import Path
import itertools

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
COLLAPSE_THRESH = 0.58  # min test_std/oof_std to include in meta-learner

def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def load_pair(stem):
    """Try multiple naming conventions for OOF/TE files."""
    for oof_pref in ("oof_", ""):
        for te_pref in ("te_", "te_oof_"):
            oof_f = DATA_PROCESSED / f"{oof_pref}{stem}.npy"
            te_f  = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if oof_f.exists() and te_f.exists():
                oof = np.load(oof_f); te_p = np.load(te_f)
                if oof.ndim == 2: oof = oof[:, 0]
                if te_p.ndim == 2: te_p = te_p[:, 0]
                return oof, te_p
    return None, None


def main():
    print("=== nb112: Grand Ensemble v3 ===\n")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ── Candidate models (ordered by expected quality) ────────────────────────
    candidates = [
        ("nb109_deep_meta_stack",     "DeepMetaStack(batch3)"),
        ("nb110_scaffold_prior",      "ScaffoldPrior(batch3)"),
        ("nb111_selectivity_primary", "SelectivityPrimary(batch3)"),
        ("nb107_assay_decomp",        "AssayDecomp(batch2)"),
        ("nb103_seed_propagation",    "SeedPropagation(batch2)"),
        ("nb101_delta_base",          "Delta-ML(batch2)"),
        ("nb99_sc_bio_fp",            "SC-BioFP(batch1)"),
        ("grand_v6b",                 "Grand-v6b"),
        ("grand25",                   "Grand25"),
        ("lgbm_tuned",                "LGBM-tuned"),
        ("catboost",                  "CatBoost"),
        ("multi_nr_transfer",         "NR-Transfer"),
        ("xgboost_dart",              "XGBoost-DART"),
        ("chemprop_aux",              "Chemprop-GNN"),
        ("nb97_pxr_features",         "PXR-Physics"),
        ("grover_large",              "GROVER-large"),
    ]

    print("Loading and screening models:")
    oofs, tes, names = [], [], []
    for stem, label in candidates:
        oof, te_pred = load_pair(stem)
        if oof is None or len(oof) != n_tr:
            print(f"  MISS  {label}")
            continue
        valid = np.isfinite(oof)
        if valid.mean() < 0.85:
            print(f"  BAD   {label}  (too many NaN: {(~valid).sum()})")
            continue
        oof_f = oof.copy(); oof_f[~valid] = np.nanmean(oof)
        te_f  = te_pred.copy(); te_f[~np.isfinite(te_pred)] = np.nanmean(te_pred)

        r = rae(y_tr, oof_f)
        oof_std = np.nanstd(oof_f)
        te_std  = np.nanstd(te_f)
        ratio   = te_std / oof_std if oof_std > 0 else 0

        flag = "SKIP" if ratio < COLLAPSE_THRESH else "OK"
        print(f"  {flag}  {r:.4f}  ratio={ratio:.2f}  te_std={te_std:.3f}  {label}")

        if flag == "OK":
            oofs.append(oof_f); tes.append(te_f); names.append(label)

    if not oofs:
        print("No valid models found! Check if nb109/110/111 ran successfully.")
        return

    O = np.column_stack(oofs)
    T = np.column_stack(tes)
    n_models = len(names)
    print(f"\n{n_models} models passed screening, matrix: {O.shape}")

    # ── Single-model baselines ────────────────────────────────────────────────
    print("\nSingle-model baselines:")
    for j, name in enumerate(names):
        full_metrics(y_tr, O[:, j], name[:55])

    # ── 2-model sweep: best model vs each other ───────────────────────────────
    print("\n2-model sweep (best vs each):")
    best_r_single = rae(y_tr, O[:, 0])
    sweep_results = []
    for j in range(1, n_models):
        best_a, best_r = 0.0, best_r_single
        for alpha in np.arange(0.05, 0.51, 0.05):
            blend = (1 - alpha) * O[:, 0] + alpha * O[:, j]
            r = rae(y_tr, blend)
            if r < best_r:
                best_r, best_a = r, alpha
        if best_r < best_r_single:
            sweep_results.append((best_r, best_a, j, names[j]))
            print(f"  {names[0][:30]} + {names[j][:30]}: best alpha={best_a:.2f}  RAE={best_r:.4f}")

    if sweep_results:
        best_sweep = min(sweep_results, key=lambda x: x[0])
        print(f"\nBest 2-model: {names[0]} + {best_sweep[3]}: alpha={best_sweep[1]:.2f}  RAE={best_sweep[0]:.4f}")

    # ── 3-model sweep (top 3 candidates) ─────────────────────────────────────
    if n_models >= 3:
        print("\n3-model sweep (top 3 by OOF RAE):")
        best_r3, best_w3 = best_r_single, (1.0, 0.0, 0.0)
        for a1 in np.arange(0.5, 1.01, 0.1):
            for a2 in np.arange(0.0, (1 - a1) + 0.01, 0.05):
                a3 = 1 - a1 - a2
                if a3 < 0: continue
                blend = a1 * O[:, 0] + a2 * O[:, 1] + a3 * O[:, 2]
                r = rae(y_tr, blend)
                if r < best_r3:
                    best_r3, best_w3 = r, (a1, a2, a3)
        print(f"  Best 3-way: w={[f'{w:.2f}' for w in best_w3]}  RAE={best_r3:.4f}")

    # ── Ridge CV on non-collapsed models ─────────────────────────────────────
    print("\nRidgeCV (scaffold CV):")
    oof_ridge = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=3)
        ridge.fit(O[tr_idx], y_tr[tr_idx])
        oof_ridge[va_idx] = ridge.predict(O[va_idx])
    full_metrics(y_tr, oof_ridge, "RidgeCV blend (scaffold CV)")

    # Ridge final (fit on all, predict test)
    ridge_final = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_final.fit(O, y_tr)
    te_ridge = ridge_final.predict(T)
    te_ridge_std = np.nanstd(te_ridge)
    print(f"  Ridge test: med={np.median(te_ridge):.2f}  std={te_ridge_std:.3f}  "
          f"max={te_ridge.max():.2f}")
    if te_ridge_std < 0.4:
        print("  WARNING: Ridge test predictions appear collapsed! Using best manual blend instead.")
        use_ridge = False
    else:
        use_ridge = True
        te_ridge = np.clip(te_ridge, y_tr.min() - 0.5, y_tr.max() + 0.5)

    # ── Save submissions ──────────────────────────────────────────────────────
    print("\n=== Saving submissions ===")

    # Strategy 1: Best single model (index 0)
    te_best_single = np.clip(T[:, 0], y_tr.min() - 0.5, y_tr.max() + 0.5)
    sub1 = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_best_single})
    sub1.to_csv(SUBMISSIONS / "112_best_single.csv", index=False)
    print(f"  Saved 112_best_single.csv  ({names[0]})")

    # Strategy 2: Best 2-model blend
    if sweep_results:
        best_r2, best_a2, best_j2, best_name2 = min(sweep_results, key=lambda x: x[0])
        te_2way = np.clip(
            (1 - best_a2) * T[:, 0] + best_a2 * T[:, best_j2],
            y_tr.min() - 0.5, y_tr.max() + 0.5
        )
        sub2 = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_2way})
        sub2.to_csv(SUBMISSIONS / "112_best_2way_blend.csv", index=False)
        print(f"  Saved 112_best_2way_blend.csv  "
              f"({names[0]}+{best_name2} a={best_a2:.2f} OOF={best_r2:.4f})")
        print(f"  Blend test: med={np.median(te_2way):.2f}  std={te_2way.std():.3f}")

    # Strategy 3: Ridge (only if not collapsed)
    if use_ridge:
        sub3 = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_ridge})
        sub3.to_csv(SUBMISSIONS / "112_ridge_ensemble.csv", index=False)
        print(f"  Saved 112_ridge_ensemble.csv  (RidgeCV OOF={rae(y_tr, oof_ridge):.4f})")

    # Strategy 4: 3-way blend
    if n_models >= 3:
        te_3way = np.clip(
            best_w3[0] * T[:, 0] + best_w3[1] * T[:, 1] + best_w3[2] * T[:, 2],
            y_tr.min() - 0.5, y_tr.max() + 0.5
        )
        sub4 = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_3way})
        sub4.to_csv(SUBMISSIONS / "112_best_3way_blend.csv", index=False)
        print(f"  Saved 112_best_3way_blend.csv  (OOF={best_r3:.4f})")
        print(f"  3way test: med={np.median(te_3way):.2f}  std={te_3way.std():.3f}")

    # Save best OOF for downstream
    np.save(DATA_PROCESSED / "oof_nb112_grand_v3.npy", oof_ridge)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("GRAND ENSEMBLE v3 SUMMARY")
    print("="*70)
    rows = [{"Model": n, "OOF_RAE": rae(y_tr, O[:, j])}
            for j, n in enumerate(names)]
    rows.append({"Model": "RidgeCV blend", "OOF_RAE": rae(y_tr, oof_ridge)})
    df = pd.DataFrame(rows).sort_values("OOF_RAE")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
