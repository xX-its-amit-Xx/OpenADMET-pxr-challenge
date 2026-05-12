"""Preprocessing utilities for the PXR activity model.

Pipeline steps (apply in order):
  1. remove_label_outliers  — drop high-SE / wide-CI training rows
  2. clip_feature_outliers  — winsorise extreme RDKit descriptor values
  3. FeatureSelector.fit_transform — drop near-zero-variance + correlated features
  4. upsample_by_category   — random oversample active / moderate compounds
  5. pseudo_inactive_augment — add external SMILES with noisy inactive labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# ---------------------------------------------------------------------------
# 1. Label outlier removal
# ---------------------------------------------------------------------------

def remove_label_outliers(
    df: pd.DataFrame,
    col: str = "pec50",
    se_col: str | None = "pec50_se",
    ci_lo_col: str | None = "pec50_lo",
    ci_hi_col: str | None = "pec50_hi",
    max_se: float = 0.5,
    max_ci_width: float | None = None,
) -> pd.DataFrame:
    """Remove rows with poorly-measured pEC50 values.

    Primary filter is standard error (max_se). An optional CI-width filter
    can catch additional outliers when SE is unavailable.
    No z-score filter is applied: the PXR training distribution has no
    extreme label outliers (verified: 0 compounds with |z| > 3).
    """
    mask = pd.Series(True, index=df.index)
    n_se = n_ci = 0

    if se_col and se_col in df.columns:
        se_mask = df[se_col].isna() | (df[se_col] <= max_se)
        n_se = (~se_mask).sum()
        mask &= se_mask

    if (max_ci_width is not None
            and ci_lo_col and ci_lo_col in df.columns
            and ci_hi_col and ci_hi_col in df.columns):
        width = df[ci_hi_col] - df[ci_lo_col]
        ci_mask = width.isna() | (width <= max_ci_width)
        n_ci = (~ci_mask & mask).sum()   # only count those not already removed
        mask &= ci_mask

    n_removed = (~mask).sum()
    print(
        f"remove_label_outliers: {len(df):,} -> {mask.sum():,} rows "
        f"(removed {n_removed}: SE>{max_se}={n_se}"
        + (f", CI_width>{max_ci_width}={n_ci}" if max_ci_width else "")
        + ")"
    )
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature outlier clipping (RDKit descriptors only — skip binary FP bits)
# ---------------------------------------------------------------------------

def clip_feature_outliers(
    X: np.ndarray,
    iqr_factor: float = 5.0,
    fp_cols: int = 2048,
) -> np.ndarray:
    """Winsorise descriptor features at Q1 − factor·IQR and Q3 + factor·IQR.

    The first ``fp_cols`` columns are treated as binary Morgan FP bits and
    are left unchanged.
    """
    X = X.copy()
    if fp_cols < X.shape[1]:
        desc = X[:, fp_cols:]
        q1 = np.nanpercentile(desc, 25, axis=0)
        q3 = np.nanpercentile(desc, 75, axis=0)
        iqr = q3 - q1
        lo = q1 - iqr_factor * iqr
        hi = q3 + iqr_factor * iqr
        X[:, fp_cols:] = np.clip(desc, lo, hi)
    return X


# ---------------------------------------------------------------------------
# 3. Feature selection (sklearn-compatible transformer)
# ---------------------------------------------------------------------------

class FeatureSelector:
    """Drop near-zero-variance features, then drop highly correlated descriptors.

    Fit on training data; apply the same selection to val / test splits.

    Parameters
    ----------
    var_thresh : float
        Remove features whose variance (across the training set) is below
        this threshold.  Default 0.01 catches near-constant features.
    corr_thresh : float
        Among RDKit descriptor features, drop one from each pair whose
        Pearson |r| exceeds this threshold.  Default 0.95.
    fp_cols : int
        Number of leading columns that are binary Morgan FP bits.  These
        skip the correlation filter (binary, sparse — rarely correlated).
    """

    def __init__(
        self,
        var_thresh: float = 0.01,
        corr_thresh: float = 0.95,
        fp_cols: int = 2048,
    ) -> None:
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh
        self.fp_cols = fp_cols
        self._kept_indices: np.ndarray | None = None   # original column indices

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "FeatureSelector":
        n_orig = X_train.shape[1]

        # Step 1: variance threshold
        vt = VarianceThreshold(threshold=self.var_thresh)
        vt.fit(X_train)
        vt_support = vt.get_support()
        kept = np.where(vt_support)[0]
        X_vt = X_train[:, kept]
        n_var_removed = n_orig - len(kept)
        print(f"[FeatureSelector] Variance filter: {n_orig} -> {len(kept)} "
              f"(removed {n_var_removed})")

        # Step 2: correlation filter on descriptor columns only
        desc_local = np.where(kept >= self.fp_cols)[0]  # local indices in X_vt
        n_corr_removed = 0

        if len(desc_local) > 1:
            C = np.corrcoef(X_vt[:, desc_local].T)
            np.fill_diagonal(C, 0.0)
            drop_set: set[int] = set()
            for i in range(len(desc_local)):
                if i in drop_set:
                    continue
                for j in np.where(np.abs(C[i]) > self.corr_thresh)[0]:
                    if j > i:
                        drop_set.add(j)

            fp_local = np.where(kept < self.fp_cols)[0]
            keep_desc = np.array(
                [i for i in range(len(desc_local)) if i not in drop_set]
            )
            keep_local = np.sort(
                np.concatenate([fp_local, desc_local[keep_desc]])
            )
            kept = kept[keep_local]
            n_corr_removed = len(drop_set)
            print(
                f"[FeatureSelector] Correlation filter: "
                f"{len(desc_local)} -> {len(keep_desc)} descriptors "
                f"(removed {n_corr_removed})"
            )

        print(
            f"[FeatureSelector] Final: {len(kept)} features "
            f"(from {n_orig}; total removed {n_orig - len(kept)})"
        )
        self._kept_indices = kept
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._kept_indices is None:
            raise RuntimeError("FeatureSelector.fit() must be called first.")
        return X[:, self._kept_indices]

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @property
    def kept_indices(self) -> np.ndarray:
        if self._kept_indices is None:
            raise RuntimeError("FeatureSelector not fitted yet.")
        return self._kept_indices

    @property
    def n_features_out(self) -> int:
        return len(self.kept_indices)


# ---------------------------------------------------------------------------
# 4. Upsampling by activity category
# ---------------------------------------------------------------------------

def upsample_by_category(
    X: np.ndarray,
    y: np.ndarray,
    inactive_thresh: float = 5.0,
    active_thresh: float = 6.0,
    target_active_frac: float = 0.10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random oversample moderate and active compounds to reduce class imbalance.

    Targets active compounds (pEC50 ≥ active_thresh) at ``target_active_frac``
    of the total training size.  Moderate compounds get half that boost.
    Returns ``(X_aug, y_aug)`` with duplicated rows appended.
    """
    rng = np.random.default_rng(random_state)
    inactive = np.where(y < inactive_thresh)[0]
    moderate = np.where((y >= inactive_thresh) & (y < active_thresh))[0]
    active   = np.where(y >= active_thresh)[0]

    n = len(y)
    n_extra_active   = max(0, int(n * target_active_frac) - len(active))
    n_extra_moderate = max(0, int(n * target_active_frac / 2) - len(moderate))

    extra_a = (rng.choice(active,   size=n_extra_active,   replace=True)
               if n_extra_active   > 0 else np.empty(0, int))
    extra_m = (rng.choice(moderate, size=n_extra_moderate, replace=True)
               if n_extra_moderate > 0 else np.empty(0, int))

    all_idx = np.concatenate([np.arange(n), extra_a, extra_m])
    print(
        f"upsample_by_category: "
        f"inactive={len(inactive):,} / moderate={len(moderate):,} / "
        f"active={len(active):,}  +{len(extra_a):,} active "
        f"+ {len(extra_m):,} moderate  -> {len(all_idx):,} total"
    )
    return X[all_idx], y[all_idx]


def category_sample_weights(
    y: np.ndarray,
    inactive_thresh: float = 5.0,
    active_thresh: float = 6.0,
    moderate_weight: float = 2.0,
    active_weight: float = 8.0,
) -> np.ndarray:
    """Per-sample weights for LightGBM (pass as sample_weight=...).

    Upweights moderate and active compounds without duplicating rows.
    """
    w = np.ones(len(y), dtype=np.float32)
    w[(y >= inactive_thresh) & (y < active_thresh)] = moderate_weight
    w[y >= active_thresh] = active_weight
    return w


# ---------------------------------------------------------------------------
# 5. Pseudo-inactive augmentation (external SMILES -> noisy inactive labels)
# ---------------------------------------------------------------------------

def pseudo_inactive_augment(
    smiles: list[str],
    inactive_mean: float = 4.30,
    inactive_std: float = 0.24,   # matches assay noise floor
    pec50_lo: float = 1.5,
    pec50_hi: float = 5.0,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Assign synthetic inactive pEC50 labels to SMILES with no assay data.

    Labels are clipped to [pec50_lo, pec50_hi] to prevent assigning active
    values by accident.  inactive_std defaults to the assay noise floor (0.24)
    so added noise matches measurement uncertainty.

    Intended use: add single_conc low-FC compounds or ZINC compounds as
    soft-negative training examples for the primary PXR pEC50 task.
    """
    rng = np.random.default_rng(seed)
    y = np.clip(
        rng.normal(inactive_mean, inactive_std, size=len(smiles)),
        pec50_lo, pec50_hi,
    )
    return smiles, y
