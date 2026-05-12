"""Feature computation for the activity track.

Three feature sets:
- ``rdkit_desc``  : ~200 RDKit 2D descriptors (same as tutorial baseline)
- ``morgan``      : ECFP4 Morgan fingerprint (2048-bit)
- ``combined``    : Morgan + rdkit_desc concatenated

All return ``(N, D)`` float32 arrays; failed SMILES become NaN rows (then
imputed to column median before training — see ``impute``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru

from .chem import morgan_fp_batch

# Instantiate the descriptor calculator once — it pre-computes the feature list.
_DESC_CALC = uru.RDKitDescriptors()


_D_RDKIT: int | None = None  # cached descriptor dimension


def rdkit_desc(smiles: list[str | None]) -> np.ndarray:
    """Compute ~200 RDKit 2D descriptors.  Returns (N, D) float32."""
    global _D_RDKIT
    if _D_RDKIT is None:
        _D_RDKIT = len(_DESC_CALC.calc_smiles("C"))

    if len(smiles) == 0:
        return np.empty((0, _D_RDKIT), dtype=np.float32)

    rows = []
    for smi in smiles:
        try:
            rows.append(_DESC_CALC.calc_smiles(smi) if smi else None)
        except Exception:
            rows.append(None)

    d = next((len(r) for r in rows if r is not None), _D_RDKIT)
    out = np.array([r if r is not None else [np.nan] * d for r in rows], dtype=np.float32)
    return out


def morgan(smiles: list[str | None], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP4 Morgan fingerprints.  Returns (N, n_bits) float32."""
    return morgan_fp_batch(smiles, radius=radius, n_bits=n_bits).astype(np.float32)


def combined(smiles: list[str | None]) -> np.ndarray:
    """Concatenate Morgan FP + RDKit 2D descriptors. Returns (N, 2048+D) float32."""
    return np.hstack([morgan(smiles), rdkit_desc(smiles)])


def impute(X: np.ndarray) -> np.ndarray:
    """Replace NaN columns with their column median (0 for all-NaN columns)."""
    X = X.copy()
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


def feature_names_rdkit() -> list[str]:
    """Return the descriptor names in the same order as rdkit_desc()."""
    return list(uru.RDKitDescriptors().desc_names)