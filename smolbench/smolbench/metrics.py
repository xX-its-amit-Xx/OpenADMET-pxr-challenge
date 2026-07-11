"""Regression metrics for molecular property prediction."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, kendalltau


def rae(y, p):
    """Relative Absolute Error = sum|y-p| / sum|y-median(y)|  (<1 beats the median predictor)."""
    d = np.abs(y - np.median(y)).sum()
    return float(np.abs(y - p).sum() / d) if d > 0 else float("nan")


def mae(y, p): return float(np.abs(y - p).mean())
def rmse(y, p): return float(np.sqrt(((y - p) ** 2).mean()))


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def all_metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return {
        "RAE": rae(y, p), "MAE": mae(y, p), "RMSE": rmse(y, p), "R2": r2(y, p),
        "Spearman": float(spearmanr(y, p).correlation),
        "Kendall": float(kendalltau(y, p).correlation),
    }
