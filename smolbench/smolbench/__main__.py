"""CLI: python -m smolbench train.csv [test.csv] --smiles SMILES --target pEC50 --out out/"""
from __future__ import annotations
import argparse, sys
import pandas as pd
from . import benchmark, text_report, make_figures
from .featurizers import available_featurizers
from .models import available_models


def main(argv=None):
    ap = argparse.ArgumentParser(prog="smolbench",
        description="Auto model+data optimization for small molecular datasets (honest scaffold CV).")
    ap.add_argument("train", nargs="?", help="training CSV (SMILES + target columns)")
    ap.add_argument("test", nargs="?", help="optional test CSV (SMILES) to predict")
    ap.add_argument("--smiles", default="SMILES", help="SMILES column name")
    ap.add_argument("--target", default="y", help="target column name")
    ap.add_argument("--featurizers", default="morgan,rdkit_desc,maccs,erg",
                    help="comma list or 'all'")
    ap.add_argument("--models", default="ridge,rf,lgbm,knn,histgbm", help="comma list or 'all'")
    ap.add_argument("--cv", default="scaffold", choices=["scaffold", "butina", "random"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-hpo", action="store_true")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--truth", default=None, help="column in test CSV with true labels (for figures/metrics)")
    ap.add_argument("--out", default="smolbench_out")
    ap.add_argument("--list", action="store_true", help="list featurizers/models and exit")
    a = ap.parse_args(argv)

    if a.list:
        print("featurizers:", ", ".join(available_featurizers()))
        print("models:     ", ", ".join(available_models())); return 0
    if not a.train:
        ap.error("train CSV is required (or use --list)")

    fz = available_featurizers() if a.featurizers == "all" else a.featurizers.split(",")
    md = available_models() if a.models == "all" else a.models.split(",")
    test = pd.read_csv(a.test) if a.test else None
    res = benchmark(a.train, test[[a.smiles]] if test is not None else None,
                    smiles_col=a.smiles, target_col=a.target,
                    featurizers=fz, models=md, cv=a.cv, n_folds=a.folds,
                    hpo=not a.no_hpo, ensemble=not a.no_ensemble,
                    calibrate=not a.no_calibrate, out_dir=a.out)
    print(text_report(res))
    yt = test[a.truth].values if (test is not None and a.truth and a.truth in test.columns) else None
    make_figures(res, a.out, y_true=yt, target_name=a.target)
    if yt is not None and res.test_predictions is not None:
        from .metrics import all_metrics
        print("\nTEST metrics:", {k: round(v, 4) for k, v in all_metrics(yt, res.test_predictions).items()})
    print(f"\nWritten to {a.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
