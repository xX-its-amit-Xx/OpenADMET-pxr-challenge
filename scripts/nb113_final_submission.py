"""nb113 — Final Submission Analysis + Best Submission Selector.

This script:
1. Loads ALL OOF predictions from every model we've run
2. Ranks by OOF RAE + test distribution quality (no collapse)
3. Identifies the best submission for Phase 1 (due 2026-05-25)
4. Creates a Kaggle-push-ready notebook wrapper for nb109 and nb111
5. Prints the complete ranking table for README update

Phase 1 deadline: 2026-05-25 (13 days away as of 2026-05-12)
Best strategy: submit the best non-collapsed prediction as primary
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42


def main():
    print("=== nb113: Final Submission Analysis ===\n")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te)

    print(f"Train: {n_tr}  Test: {n_te}")
    print(f"Train pEC50: mean={y_tr.mean():.3f}  std={y_tr.std():.3f}  "
          f"min={y_tr.min():.2f}  max={y_tr.max():.2f}")

    def load_pair(stem):
        for op in ("oof_", ""):
            for tp in ("te_", "te_oof_"):
                of = DATA_PROCESSED / f"{op}{stem}.npy"
                tf = DATA_PROCESSED / f"{tp}{stem}.npy"
                if of.exists() and tf.exists():
                    oof = np.load(of); te_p = np.load(tf)
                    if oof.ndim == 2: oof = oof[:, 0]
                    if te_p.ndim == 2: te_p = te_p[:, 0]
                    return oof, te_p
        return None, None

    # All models (batch-1 through batch-3)
    all_models = [
        # Batch-3 (new)
        ("nb109_deep_meta_stack",     "DeepMeta(b3)"),
        ("nb112_grand_v3",            "Grand-v3(b3)"),
        ("nb111_selectivity_primary", "SelPrimary(b3)"),
        ("nb110_scaffold_prior",      "ScaffPrior(b3)"),
        # Batch-2
        ("nb107_assay_decomp",        "AssayDecomp(b2)"),
        ("nb108_grand_v2",            "Grand-v2(b2)"),
        ("nb103_seed_propagation",    "SeedProp(b2)"),
        ("nb101_delta_base",          "Delta-ML(b2)"),
        ("nb99_sc_bio_fp",            "SC-BioFP(b1)"),
        ("nb97_pxr_features",         "PXR-Phys(b1)"),
        ("nb100_emax_corrected",      "Emax-Corr(b1)"),
        # From other session
        ("grand_v6b",                 "Grand-v6b"),
        ("grand25",                   "Grand25"),
        ("lgbm_tuned",                "LGBM-tuned"),
        ("catboost",                  "CatBoost"),
        ("multi_nr_transfer",         "NR-Transfer"),
        ("xgboost_dart",              "XGBoost"),
        ("chemprop_aux",              "Chemprop"),
        ("grover_large",              "GROVER-L"),
    ]

    print("\n" + "="*85)
    print(f"{'Model':<30} {'OOF_RAE':>8} {'OOF_std':>8} {'te_std':>8} {'ratio':>7} {'te_max':>7} {'Status'}")
    print("="*85)

    results = []
    for stem, label in all_models:
        oof, te_p = load_pair(stem)
        if oof is None or len(oof) != n_tr: continue
        valid = np.isfinite(oof)
        if valid.mean() < 0.85: continue
        oof_f = oof.copy(); oof_f[~valid] = np.nanmean(oof)
        te_f  = te_p.copy(); te_f[~np.isfinite(te_p)] = np.nanmean(te_p)

        r = rae(y_tr, oof_f)
        oof_std = np.nanstd(oof_f)
        te_std  = np.nanstd(te_f)
        te_max  = np.nanmax(te_f)
        ratio   = te_std / oof_std if oof_std > 0 else 0

        if ratio >= 0.60:
            status = "OK"
        elif ratio >= 0.50:
            status = "borderline"
        else:
            status = "COLLAPSED"

        print(f"  {label:<28} {r:>8.4f} {oof_std:>8.3f} {te_std:>8.3f} {ratio:>7.2f} "
              f"{te_max:>7.2f} {status}")
        results.append(dict(
            stem=stem, label=label, oof_rae=r, oof_std=oof_std,
            te_std=te_std, ratio=ratio, status=status,
            te_preds=te_f
        ))

    print("="*85)

    # ── Find best submission ──────────────────────────────────────────────────
    print("\n=== BEST SUBMISSION CANDIDATES ===\n")
    ok_models = [r for r in results if r["status"] == "OK"]
    ok_models_sorted = sorted(ok_models, key=lambda x: x["oof_rae"])

    if ok_models_sorted:
        best = ok_models_sorted[0]
        print(f"PRIMARY SUBMISSION (best OOF + non-collapsed):")
        print(f"  Model: {best['label']}")
        print(f"  OOF RAE: {best['oof_rae']:.4f}")
        print(f"  Test: std={best['te_std']:.3f}  max={best['te_preds'].max():.2f}  "
              f"med={np.median(best['te_preds']):.2f}")

        # Check if submission file exists
        sub_path = SUBMISSIONS / f"{best['stem'].split('_')[0][2:]}_*.csv"
        matches = list(SUBMISSIONS.glob(f"*{best['stem'].split('nb')[-1]}*.csv"))
        if matches:
            print(f"  Submission file: {matches[0].name}")
        else:
            # Create it from the te_preds
            sub = pd.DataFrame({"Molecule Name": te["name"].values,
                                 "pEC50": best["te_preds"]})
            sub_out = SUBMISSIONS / f"113_{best['label'].replace(' ', '_')}.csv"
            sub.to_csv(sub_out, index=False)
            print(f"  Created submission: {sub_out.name}")

    # Top 5 non-collapsed models for comparison
    print(f"\nTop 5 valid models by OOF RAE:")
    for i, m in enumerate(ok_models_sorted[:5]):
        print(f"  {i+1}. {m['label']:30s}  OOF={m['oof_rae']:.4f}  "
              f"te_std={m['te_std']:.3f}  ratio={m['ratio']:.2f}")

    # ── Submission comparison ─────────────────────────────────────────────────
    print("\n=== EXISTING SUBMISSION FILES ===\n")
    sub_files = sorted(SUBMISSIONS.glob("*.csv"))
    all_subs = []
    for f in sub_files:
        try:
            df = pd.read_csv(f)
            preds = df.iloc[:, -1].values
            if len(preds) == n_te:
                all_subs.append(dict(
                    file=f.name,
                    n=len(preds),
                    mean=preds.mean(),
                    std=preds.std(),
                    max_pred=preds.max(),
                    min_pred=preds.min(),
                ))
        except Exception:
            pass

    sub_df = pd.DataFrame(all_subs)
    if len(sub_df) > 0:
        # Filter to reasonable range
        good = sub_df[(sub_df["std"] >= 0.40) & (sub_df["max_pred"] >= 5.5)]
        print(f"Submissions with std >= 0.40 and max >= 5.5 (likely valid):")
        for _, row in good.iterrows():
            print(f"  {row['file']:50s}  mean={row['mean']:.2f}  "
                  f"std={row['std']:.3f}  max={row['max_pred']:.2f}")

    # ── Final recommendation ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    print()
    if ok_models_sorted:
        b = ok_models_sorted[0]
        print(f"Submit: {b['label']} (OOF RAE={b['oof_rae']:.4f})")
        print(f"  Predicted to outperform baseline by "
              f"{(1 - b['oof_rae']) * 100:.1f}% relative to mean predictor")
        print()
    print("Deadline: 2026-05-25 (Phase 1 close)")
    print("Next steps:")
    print("  1. Push nb109/nb111 to Kaggle for GPU validation")
    print("  2. Run nb112 to find best blend")
    print("  3. Submit best non-collapsed prediction")
    print("  4. After 2026-05-26: refit on Analog Set 1 (new labels)")


if __name__ == "__main__":
    main()
