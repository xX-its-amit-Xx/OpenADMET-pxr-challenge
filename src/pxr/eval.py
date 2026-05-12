"""Evaluation metrics and CV splitting utilities."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Metrics (match the official scoring config exactly)
# ---------------------------------------------------------------------------

def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Absolute Error — the primary challenge metric (lower is better).

    RAE = MAE / MAE(mean predictor).  RAE = 1.0 means you're no better than
    predicting the training mean; RAE < 1.0 means you're adding signal.
    """
    denom = np.sum(np.abs(y_true - y_true.mean()))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All official activity metrics in one call."""
    return {
        "RAE":          rae(y_true, y_pred),
        "MAE":          mean_absolute_error(y_true, y_pred),
        "R2":           r2_score(y_true, y_pred),
        "Spearman":     spearmanr(y_true, y_pred).statistic,
        "Kendall_tau":  kendalltau(y_true, y_pred).statistic,
    }


# ---------------------------------------------------------------------------
# Scaffold-based GroupKFold
# ---------------------------------------------------------------------------

def scaffold_kfold_indices(
    scaffolds: pd.Series | list[str | None],
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, val_idx) pairs that keep scaffolds unsplit.

    Each scaffold appears entirely in one fold. This gives a realistic
    estimate of out-of-scaffold generalisation.  Singleton molecules
    (no scaffold / ring-free) are assigned to folds round-robin.
    """
    rng = np.random.default_rng(seed)
    n = len(scaffolds) if not isinstance(scaffolds, pd.Series) else len(scaffolds)

    # Group compound indices by scaffold
    scaffold_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        key = sc if (sc and pd.notna(sc)) else f"__singleton_{i}__"
        scaffold_to_idx[key].append(i)

    scaffold_groups = list(scaffold_to_idx.values())
    if shuffle:
        rng.shuffle(scaffold_groups)

    # Sort largest scaffolds first so big families don't skew one fold
    scaffold_groups.sort(key=len, reverse=True)

    # Assign scaffold groups to folds greedily (minimise fold size variance)
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_assignments: list[list[list[int]]] = [[] for _ in range(n_splits)]
    for group in scaffold_groups:
        smallest_fold = int(np.argmin(fold_sizes))
        fold_assignments[smallest_fold].append(group)
        fold_sizes[smallest_fold] += len(group)

    # Build (train_idx, val_idx) pairs
    all_idx = np.arange(n)
    splits = []
    for fold in range(n_splits):
        val_idx = np.array([i for group in fold_assignments[fold] for i in group])
        train_idx = np.setdiff1d(all_idx, val_idx)
        splits.append((train_idx, val_idx))
    return splits


def cv_score(
    model,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Run cross-validation and return per-fold metrics as a DataFrame."""
    rows = []
    for fold, (tr_idx, val_idx) in enumerate(splits):
        model.fit(X[tr_idx], y[tr_idx])
        preds = model.predict(X[val_idx])
        m = compute_metrics(y[val_idx], preds)
        m["fold"] = fold
        m["n_train"] = len(tr_idx)
        m["n_val"] = len(val_idx)
        rows.append(m)
    return pd.DataFrame(rows)