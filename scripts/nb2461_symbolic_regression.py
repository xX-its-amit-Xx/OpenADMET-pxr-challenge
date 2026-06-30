"""nb2461 — Symbolic regression (PySR) on chemprop-anchor residual.

Target: y_unb - nb2240_mean_bag_oof_K20
Features: 8 physchem descriptors (MW, LogP, TPSA, HBD, HBA, RotBonds, fsp3, charge)
Protocol: 5-fold scaffold CV, PySRRegressor genetic programming.
Gate: mean_rae < 0.4570 -> PROMOTE, else FAIL.
Substrate: nb2240 K=20 mean-bag (anchor=chemprop_aux, 28-feature SHAP LGBM).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"
SUMMARY_PATH = PROC / "nb2461_summary.json"

sys.path.insert(0, str(REPO / "src"))


def save_summary(status: str, payload: dict | None = None) -> None:
    out = {"status": status, "notebook": "nb2461", "method": "pysr_symbolic_regression"}
    if payload:
        out.update(payload)
    SUMMARY_PATH.write_text(json.dumps(out, indent=2, default=str))


# ---------- install / Julia check ----------
try:
    from pysr import PySRRegressor  # noqa: F401
except Exception as e:
    save_summary("INSTALL_FAILED", {"error": f"pysr import failed: {e}"})
    sys.exit(0)


# ---------- substrate load ----------
te_anchor = np.load(PROC / "te_nb2240_K20.npy")       # (513,)
oof_anchor = np.load(PROC / "nb2240_mean_bag_oof_K20.npy")  # (253,)
y_unb = np.load(PROC / "_audit_unblind_y.npy")        # (253,)
unb_idx = np.load(PROC / "_audit_unblind_idx.npy")    # (253,) into 0..512

assert te_anchor.shape == (513,)
assert oof_anchor.shape == y_unb.shape == unb_idx.shape == (253,)

# residual target
resid = y_unb - oof_anchor


# ---------- physchem features (8 cols) ----------
from pxr.chem import compute_physchem, bemis_murcko, standardize_smiles  # noqa: E402

test_df = pd.read_csv(RAW / "pxr-challenge_TEST_BLINDED.csv")
unb_smiles = test_df["SMILES"].iloc[unb_idx].tolist()

PHYS_KEYS = ["mw", "logp", "tpsa", "hbd", "hba", "rotbonds", "fsp3", "formal_charge"]


def phys_row(smi):
    p = compute_physchem(smi)
    if p is None:
        return [np.nan] * len(PHYS_KEYS)
    return [p.get(k, np.nan) for k in PHYS_KEYS]


X_unb = np.array([phys_row(s) for s in unb_smiles], dtype=np.float64)
col_med = np.nanmedian(X_unb, axis=0)
ix_nan = np.isnan(X_unb)
X_unb[ix_nan] = np.take(col_med, np.where(ix_nan)[1])
assert X_unb.shape == (253, 8)


# ---------- scaffolds for CV split ----------
scaffolds = [bemis_murcko(s) or f"__sing_{i}__" for i, s in enumerate(unb_smiles)]

from pxr.eval import scaffold_kfold_indices, rae  # noqa: E402

folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)


# ---------- 5-fold scaffold CV with PySR ----------
fold_rae = []
fold_models = []
t0 = time.time()
print(f"[nb2461] Starting 5-fold scaffold CV PySR (~5 min/fold timeout)...", flush=True)

for fi, (tr_idx, va_idx) in enumerate(folds):
    print(f"[fold {fi}] tr={len(tr_idx)} va={len(va_idx)}", flush=True)
    X_tr, X_va = X_unb[tr_idx], X_unb[va_idx]
    y_tr = resid[tr_idx]
    # anchor val pred = oof_anchor (LB-faithful 28-feat LGBM mean-bag K=20)
    anchor_va = oof_anchor[va_idx]
    y_va_truth = y_unb[va_idx]

    model = PySRRegressor(
        niterations=20,
        populations=15,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt"],
        timeout_in_seconds=300,
        deterministic=True,
        random_state=42,
        procs=0,             # deterministic requires single-thread
        multithreading=False,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
        warm_start=False,
    )
    try:
        model.fit(X_tr, y_tr, variable_names=PHYS_KEYS)
        resid_hat_va = np.asarray(model.predict(X_va), dtype=np.float64)
    except Exception as e:
        print(f"[fold {fi}] PySR fit failed: {e}", flush=True)
        resid_hat_va = np.zeros_like(y_va_truth)

    final_va = anchor_va + resid_hat_va
    r = float(rae(y_va_truth, final_va))
    fold_rae.append(r)
    try:
        best_eq = str(model.get_best()["equation"])
    except Exception:
        best_eq = "n/a"
    fold_models.append(best_eq)
    print(f"[fold {fi}] RAE={r:.4f}  best_eq={best_eq}", flush=True)

mean_rae = float(np.mean(fold_rae))
elapsed = time.time() - t0

gate = "PROMOTE" if mean_rae < 0.4570 else "FAIL"
print(f"[nb2461] mean_rae={mean_rae:.4f}  gate={gate}  elapsed={elapsed:.1f}s", flush=True)

save_summary(
    gate,
    {
        "anchor": "nb2240_K20_mean_bag",
        "gate_threshold": 0.4570,
        "mean_rae": mean_rae,
        "fold_rae": fold_rae,
        "fold_best_equations": fold_models,
        "features": PHYS_KEYS,
        "n_unb": 253,
        "n_folds": 5,
        "pysr_config": {
            "niterations": 20,
            "populations": 15,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["sin", "cos", "exp", "log", "sqrt"],
            "timeout_in_seconds": 300,
            "deterministic": True,
            "random_state": 42,
        },
        "elapsed_sec": elapsed,
    },
)
