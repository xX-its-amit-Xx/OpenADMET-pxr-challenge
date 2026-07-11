"""Model registry — regressors with sensible small-data defaults + light HPO grids.

Each entry is (constructor, param_grid). Optional deps (xgboost, catboost) are skipped
gracefully. Register your own with ``register_model(name, ctor, grid)``.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

MODELS: dict[str, tuple] = {}


def register_model(name, ctor, grid=None):
    MODELS[name] = (ctor, grid or [{}])


register_model("ridge", lambda **k: Ridge(**k), [{"alpha": a} for a in (0.1, 1, 10, 100, 1000)])
register_model("elasticnet", lambda **k: ElasticNet(**k),
               [{"alpha": a, "l1_ratio": l} for a in (0.01, 0.1, 1) for l in (0.2, 0.5, 0.8)])
register_model("rf", lambda **k: RandomForestRegressor(n_jobs=-1, random_state=0, **k),
               [{"n_estimators": 400, "max_features": mf} for mf in ("sqrt", 0.3)])
register_model("extratrees", lambda **k: ExtraTreesRegressor(n_jobs=-1, random_state=0, **k),
               [{"n_estimators": 400, "max_features": mf} for mf in ("sqrt", 0.3)])
register_model("histgbm", lambda **k: HistGradientBoostingRegressor(random_state=0, **k),
               [{"learning_rate": lr, "max_iter": 400, "l2_regularization": l2}
                for lr in (0.03, 0.1) for l2 in (0.0, 1.0)])
register_model("knn", lambda **k: KNeighborsRegressor(n_jobs=-1, **k),
               [{"n_neighbors": n, "weights": "distance"} for n in (3, 5, 9)])
register_model("svr", lambda **k: SVR(**k),
               [{"C": c, "gamma": "scale"} for c in (1, 10)])
register_model("krr", lambda **k: KernelRidge(**k),
               [{"alpha": a, "kernel": "rbf"} for a in (0.1, 1, 10)])
register_model("gp", lambda **k: GaussianProcessRegressor(
    kernel=RBF() + WhiteKernel(), normalize_y=True, alpha=1e-6, random_state=0, **k), [{}])

# optional boosters
try:
    from lightgbm import LGBMRegressor
    register_model("lgbm", lambda **k: LGBMRegressor(n_jobs=-1, verbose=-1, random_state=0, **k),
                   [{"n_estimators": 500, "num_leaves": nl, "learning_rate": lr}
                    for nl in (31, 63) for lr in (0.03, 0.1)])
except Exception:
    pass
try:
    from xgboost import XGBRegressor
    register_model("xgboost", lambda **k: XGBRegressor(n_jobs=-1, verbosity=0, random_state=0, **k),
                   [{"n_estimators": 500, "max_depth": d, "learning_rate": 0.05} for d in (4, 7)])
except Exception:
    pass
try:
    from catboost import CatBoostRegressor
    register_model("catboost", lambda **k: CatBoostRegressor(verbose=0, random_state=0, **k),
                   [{"iterations": 600, "depth": d, "learning_rate": 0.04} for d in (6, 8)])
except Exception:
    pass


def available_models():
    return sorted(MODELS)


def build(name, params):
    ctor, _ = MODELS[name]
    return ctor(**params)


def grid(name):
    return MODELS[name][1]
