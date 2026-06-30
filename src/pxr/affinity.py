"""Convert ANY affinity/potency measurement to the p-scale (p = -log10[M]), matching the PXR pEC50 target.

Core insight (cycle-310): IC50, EC50, AC50, Ki, Kd, (and Km as a rough binding proxy) are all concentrations ->
-log10(molar) puts them on the SAME p-scale as pEC50, so external potency data from ANY assay type is usable as
training / auxiliary-head / read-across signal once unit-normalized. Already-log values (pIC50/pKd/pKi) pass through.

NOT directly convertible: kcat (turnover, 1/s) is a RATE not an affinity -> do not p-convert. kcat/Km (catalytic
efficiency, M^-1 s^-1) IS binding-related; if only kcat is available, treat as a weak orthogonal feature, not a label.
"""
import numpy as np

UNIT_TO_M = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "micromolar": 1e-6,
             "nM": 1e-9, "nanomolar": 1e-9, "pM": 1e-12, "fM": 1e-15}


def to_p(value, unit="nM"):
    """Concentration (IC50/EC50/AC50/Ki/Kd/Km) in `unit` -> p-value (-log10 M). Vectorized."""
    v = np.asarray(value, dtype=float) * UNIT_TO_M.get(unit, 1e-9)
    return -np.log10(np.clip(v, 1e-15, None))


def already_p(value):
    """Pass-through for values already on a p-scale (pIC50/pEC50/pKd/pKi/pChEMBL)."""
    return np.asarray(value, dtype=float)


def normalize_potency(value, kind):
    """kind in {'conc_nM','conc_uM','conc_M','p','pchembl'} -> p-scale. Convenience dispatcher."""
    if kind in ("p", "pchembl", "pic50", "pec50", "pkd", "pki"):
        return already_p(value)
    if kind.startswith("conc_"):
        return to_p(value, unit=kind.split("_", 1)[1])
    raise ValueError(f"unknown potency kind {kind}")
