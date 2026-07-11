"""smolbench — automated model + data optimization for SMALL small-molecule datasets.

Give it a train CSV (SMILES + numeric target) and it combinatorially searches
featurizers x data-prep x models x hyperparameters under *honest* (scaffold/cluster)
cross-validation, then reports out-of-fold metrics, fits the best config on the full
train, predicts your test set, and surfaces the small-data pitfalls we learned the hard
way (random-CV optimism, overfit-stack liability).

    import smolbench as sb
    res = sb.benchmark("train.csv", "test.csv", smiles_col="SMILES", target_col="pEC50",
                       featurizers=["morgan", "rdkit_desc", "maccs", "erg"],
                       models=["ridge", "rf", "lgbm", "knn", "svr"],
                       cv="scaffold", ensemble=True, calibrate=True, out_dir="out/")
    print(res.top(10))
    print(sb.text_report(res))
    sb.make_figures(res, "out/", y_true=test_truth)   # if you have test labels

Designed to drop into DeepChem workflows or stand alone. All core featurizers are
RDKit-only (no network/GPU); optional boosters (lightgbm/xgboost/catboost) are used if
installed.
"""
from .core import benchmark, BenchmarkResult, PREPS
from .report import make_figures, text_report
from .featurizers import featurize, available_featurizers, register_featurizer, FEATURIZERS
from .models import available_models, register_model, MODELS
from .cv import make_folds
from .metrics import all_metrics, rae, mae, r2
from .stability import stability_check, compare_top

__version__ = "0.1.0"
__all__ = ["benchmark", "BenchmarkResult", "make_figures", "text_report", "featurize",
           "available_featurizers", "register_featurizer", "available_models",
           "register_model", "make_folds", "all_metrics", "rae", "mae", "r2", "PREPS",
           "stability_check", "compare_top"]
