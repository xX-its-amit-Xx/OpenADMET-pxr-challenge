"""Loaders for the OpenADMET PXR Challenge dataset.

Columns in the raw CSVs use long names with unit suffixes
(e.g. ``pEC50_std.error (-log10(molarity))``). The loaders here rename
them to short, code-friendly snake_case while keeping the originals
accessible via ``raw=True``.
"""
from __future__ import annotations

import pandas as pd

from .paths import DATA_RAW

# Short-name map for the activity / counter-assay configs (same schema).
_DOSE_RESPONSE_RENAME = {
    "Molecule Name": "name",
    "SMILES": "smiles",
    "OCNT Batch": "batch",
    "OCNT_ID": "ocnt_id",
    "Split": "split",
    "source": "source",
    "pEC50": "pec50",
    "pEC50_std.error (-log10(molarity))": "pec50_se",
    "pEC50_ci.lower (-log10(molarity))": "pec50_lo",
    "pEC50_ci.upper (-log10(molarity))": "pec50_hi",
    "Emax_estimate (log2FC vs. baseline)": "emax",
    "Emax_std.error (log2FC vs. baseline)": "emax_se",
    "Emax_ci.lower (log2FC vs. baseline)": "emax_lo",
    "Emax_ci.upper (log2FC vs. baseline)": "emax_hi",
    "Emax.vs.pos.ctrl_estimate (dimensionless)": "emax_rel",
    "Emax.vs.pos.ctrl_std.error (dimensionless)": "emax_rel_se",
    "Emax.vs.pos.ctrl_ci.lower (dimensionless)": "emax_rel_lo",
    "Emax.vs.pos.ctrl_ci.upper (dimensionless)": "emax_rel_hi",
}

_TEST_RENAME = {
    "Molecule Name": "name",
    "SMILES": "smiles",
}

_SINGLE_CONC_RENAME = {
    "Molecule Name": "name",
    "OCNT Batch": "batch",
    "SMILES": "smiles",
    "OCNT_ID": "ocnt_id",
    "split": "split",
    "source": "source",
}

_STRUCTURE_RENAME = {
    "structure": "structure_code",
    "smiles": "smiles",
    "Molecule Name": "name",
    "OCNT_ID": "ocnt_id",
}

# Two CHANGELOG-2026-05-27 add-ons share a wide-format crude/semi-pure schema.
_SEMI_PURE_RENAME = {
    "SMILES": "smiles",
    "OCNT_ID": "ocnt_id",
    "Semi-Pure Batch ID": "batch",
    "Semi-Pure EC50s (µM)": "ec50_um",
    "Semi-Pure pEC50s (log)": "pec50_raw",
    "Semi-Pure Emax normalized": "emax_norm",
    "Semi-Pure Emax raw": "emax_raw",
    "Corrected Semi-Pure EC50 (µM)": "ec50_um_corr",
    "Corrected Semi-Pure pEC50 (log)": "pec50",
    "Semi-Pure DRC pEC50 SE (log)": "pec50_se_drc",
    "Corrected Semi-Pure pEC50 ±1 SE (log)": "pec50_se",
    "Volatility": "volatility",
    "Semi-Pure Product Yield (%)": "product_yield_pct",
    "Semi-Pure Correction Factor": "correction_factor",
}

_CRUDES_RENAME = {
    "SMILES": "smiles",
    "OCNT_ID": "ocnt_id",
    "Crude Batch ID": "batch",
    "Crude EC50s (µM)": "ec50_um",
    "Crude pEC50s (log)": "pec50_raw",
    "Crude Emax normalized": "emax_norm",
    "Crude Emax raw": "emax_raw",
    "Corrected Crude EC50 (µM)": "ec50_um_corr",
    "Corrected Crude pEC50 (log)": "pec50",
    "Crude DRC pEC50 SE (log)": "pec50_se_drc",
    "Corrected Crude pEC50 ±1 SE (log)": "pec50_se",
    "Volatility": "volatility",
    "Crude Product Yield (%)": "product_yield_pct",
    "Crude Correction Factor": "correction_factor",
}


def load_semi_pure(raw: bool = False) -> pd.DataFrame:
    """96-compound microscale semi-pure dose-response set (CHANGELOG 2026-05-27)."""
    df = pd.read_csv(DATA_RAW / "pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")
    return df if raw else _rename(df, _SEMI_PURE_RENAME)


def load_crudes(raw: bool = False) -> pd.DataFrame:
    """Crude / htchem direct-to-biology library dose-response (CHANGELOG 2026-05-27)."""
    df = pd.read_csv(DATA_RAW / "pxr-challenge_htchem-libraries_TRAIN.csv")
    return df if raw else _rename(df, _CRUDES_RENAME)


def load_phase1_unblinded(raw: bool = False) -> pd.DataFrame:
    """Phase-1 unblinded subset of the 513-test (CHANGELOG 2026-05-27): 253 rows w/ labels."""
    df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    return df if raw else _rename(df, _DOSE_RESPONSE_RENAME)


def _rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def load_train(raw: bool = False) -> pd.DataFrame:
    """4,139-row dose-response training set with pEC50/Emax + std errors and CIs."""
    df = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    return df if raw else _rename(df, _DOSE_RESPONSE_RENAME)


def load_test(raw: bool = False) -> pd.DataFrame:
    """513-compound blinded test set (only name + SMILES)."""
    df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    return df if raw else _rename(df, _TEST_RENAME)


def load_counter(raw: bool = False) -> pd.DataFrame:
    """2,859-row PXR-null counter-assay dose-response data (same schema as train)."""
    df = pd.read_csv(DATA_RAW / "pxr-challenge_counter-assay_TRAIN.csv")
    return df if raw else _rename(df, _DOSE_RESPONSE_RENAME)


def load_single_conc(raw: bool = False) -> pd.DataFrame:
    """21,003-row single-concentration primary screen (log2 fold-change + FDR)."""
    df = pd.read_csv(DATA_RAW / "pxr-challenge_single_concentration_TRAIN.csv")
    return df if raw else _rename(df, _SINGLE_CONC_RENAME)


def load_structure(raw: bool = False) -> pd.DataFrame:
    """184-row structure-track test set (crystal codes + SMILES)."""
    df = pd.read_csv(DATA_RAW / "pxr-challenge_structure_TEST_BLINDED.csv")
    return df if raw else _rename(df, _STRUCTURE_RENAME)


def load_all() -> dict[str, pd.DataFrame]:
    """Convenience: load every config into a dict keyed by short name."""
    return {
        "train": load_train(),
        "test": load_test(),
        "counter": load_counter(),
        "single_conc": load_single_conc(),
        "structure": load_structure(),
        "semi_pure": load_semi_pure(),
        "crudes": load_crudes(),
        "phase1_unblinded": load_phase1_unblinded(),
    }
