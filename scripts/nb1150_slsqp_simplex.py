"""nb1150 -- Scaffold-CV SLSQP simplex blend on 4 LB-honest anchors.

ANCHORS (all OOFs on the 253 unblind; LB-honest = honest cross-fit, no truth leak):
    1. chemprop_aux           -> data/processed/nb1133_chemprop_aux_pred_oof.npy  (~0.5879)
    2. nb503                  -> data/processed/nb503_pred_oof.npy                (0.5116)
    3. nb1014                 -> data/processed/nb1133_nb1014_pred_oof.npy        (~0.5799)
    4. nb2112 (chemprop_aux + K28 LGBM resid) -> nb2103_mean_bag_oof_K28.npy      (0.4737)

PROTOCOL:
    - Build scaffold k-fold split (n=5) on Murcko scaffolds of the 253 unblind SMILES.
    - For each fold: SLSQP minimize fold-OOF RAE on the held-out indices, weights on the
      simplex (w>=0, sum(w)=1).
    - Concatenate per-fold validation predictions -> OOS-concat blended OOF.
    - Report scaffold-CV RAE, per-fold weight vectors, degeneracy flag (max(w) > 0.85),
      plus single-pool SLSQP weight vector for cross-check.
    - DEPLOY GATE: scaffold-CV RAE <= 0.5027 (nb2112-honest 0.5057 - 0.003).
    - If gate passes: build submissions/nb1150_deploy_slsqp4.csv using mean-of-per-fold
      simplex weights applied to te_*.npy 513-vectors.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize

from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

RDLogger.DisableLog("rdApp.*")

TAG = "nb1150"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

DECISION_GATE = 0.5027  # = 0.5057 (nb2112 honest scaffold-CV target) - 0.003
DEGEN_MAX_W = 0.85

# Anchor file layout
ANCHOR_OOF_PATHS = {
    "chemprop_aux": DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
    "nb503":        DATA_PROCESSED / "nb503_pred_oof.npy",
    "nb1014":       DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy",
    "nb2112":       DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",  # honest LB-faithful proxy
}
ANCHOR_TE_PATHS = {
    "chemprop_aux": DATA_PROCESSED / "te_chemprop_aux.npy",
    "nb503":        DATA_PROCESSED / "te_nb503.npy",
    "nb1014":       DATA_PROCESSED / "te_nb1014.npy",
    "nb2112":       DATA_PROCESSED / "te_nb2112.npy",
}
ANCHOR_NAMES = list(ANCHOR_OOF_PATHS.keys())  # fixed order

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y   = DATA_PROCESSED / "_audit_unblind_y.npy"


def _murcko_scaffold(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) or ""
    except Exception:
        return ""


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 5,
                   seed: int = 0) -> tuple[np.ndarray, float]:
    """Minimize RAE(y, P @ w) with w on the simplex (w>=0, sum w=1).

    Multi-start for robustness; returns best (w, rae).
    """
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w):
        return rae(y, P @ w)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(n_starts - 1):
        x = rng.dirichlet(np.ones(K))
        starts.append(x)

    best_w, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            w = np.clip(res.x, 0.0, 1.0)
            s = w.sum()
            if s <= 0:
                continue
            w = w / s
            r = rae(y, P @ w)
            if r < best_r:
                best_r, best_w = r, w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = rae(y, P @ best_w)
    return best_w, best_r


def main() -> None:
    t0 = time.time()
    out = {"tag": TAG, "decision_gate": DECISION_GATE,
           "anchors": ANCHOR_NAMES, "anchor_files": {k: str(v) for k, v in ANCHOR_OOF_PATHS.items()}}

    # ----- load -----
    unb_idx = np.load(UNBLIND_IDX)
    y = np.load(UNBLIND_Y).astype(np.float64)
    n = len(y)
    assert n == 253, f"expected 253 unblind, got {n}"

    P_list, indiv_rae = [], {}
    for name in ANCHOR_NAMES:
        oof = np.load(ANCHOR_OOF_PATHS[name]).astype(np.float64)
        assert oof.shape == (253,), f"{name} OOF wrong shape {oof.shape}"
        # Verify no truth leak: sha-style sanity (a clean OOF should NOT equal y on >5% of rows)
        exact_eq_frac = float(np.mean(np.isclose(oof, y, atol=1e-6)))
        indiv_rae[name] = round(rae(y, oof), 4)
        P_list.append(oof)
        if exact_eq_frac > 0.05:
            print(f"WARN {name}: {exact_eq_frac:.1%} rows exactly equal truth -- possible leak")
    P = np.stack(P_list, axis=1)  # (253, 4)
    out["indiv_rae_unb"] = indiv_rae

    # ----- scaffold-fold split on 253 unblind -----
    te = load_test()
    smi_unb = te.iloc[unb_idx]["smiles"].tolist()
    scaffolds = [_murcko_scaffold(s) for s in smi_unb]
    n_scaf = len(set([s for s in scaffolds if s]))
    out["n_unique_scaffolds"] = n_scaf

    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)

    # ----- per-fold SLSQP -----
    blended_oof = np.full(n, np.nan, dtype=np.float64)
    per_fold = []
    fold_weights = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        # SLSQP weights minimize RAE on TRAIN portion (held-out from val)
        w, r_train = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
        val_pred = P[va_idx] @ w
        blended_oof[va_idx] = val_pred
        r_val = rae(y[va_idx], val_pred)
        per_fold.append({
            "fold": fi,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "weights": {ANCHOR_NAMES[k]: round(float(w[k]), 4) for k in range(len(w))},
            "train_rae": round(float(r_train), 4),
            "val_rae": round(float(r_val), 4),
            "max_w": round(float(w.max()), 4),
            "degenerate": bool(w.max() > DEGEN_MAX_W),
        })
        fold_weights.append(w)
    fold_weights = np.stack(fold_weights, axis=0)  # (5, 4)

    assert not np.any(np.isnan(blended_oof)), "OOF coverage gap"
    scaffold_cv_rae = round(float(rae(y, blended_oof)), 4)

    out["per_fold"] = per_fold
    out["scaffold_cv_rae_oos_concat"] = scaffold_cv_rae

    # ----- single-pool SLSQP (sanity / deploy weights) -----
    w_full, r_full = _simplex_slsqp(P, y, n_starts=12, seed=0)
    out["full_pool_slsqp"] = {
        "weights": {ANCHOR_NAMES[k]: round(float(w_full[k]), 4) for k in range(len(w_full))},
        "rae_in_sample": round(float(r_full), 4),
        "max_w": round(float(w_full.max()), 4),
        "degenerate": bool(w_full.max() > DEGEN_MAX_W),
    }

    # Mean-of-fold weights (LB-honest deploy weights)
    w_mean = fold_weights.mean(axis=0)
    w_mean = w_mean / w_mean.sum()
    out["mean_of_fold_weights"] = {ANCHOR_NAMES[k]: round(float(w_mean[k]), 4)
                                   for k in range(len(w_mean))}
    out["any_fold_degenerate"] = bool(any(f["degenerate"] for f in per_fold))

    # Compare apples to apples: scaffold_cv RAE applying mean-of-fold weights to full P
    rae_mean_w = round(float(rae(y, P @ w_mean)), 4)
    out["in_sample_rae_mean_of_fold_w"] = rae_mean_w

    # ----- gate -----
    passes = scaffold_cv_rae <= DECISION_GATE
    out["passes_gate"] = bool(passes)

    # ----- deploy -----
    deploy_csv = None
    if passes:
        te_mat = np.stack([np.load(ANCHOR_TE_PATHS[name]).astype(np.float64)
                           for name in ANCHOR_NAMES], axis=1)  # (513, 4)
        deploy_pred_513 = te_mat @ w_mean
        te_df = load_test()
        sub = pd.DataFrame({
            "SMILES": te_df["smiles"].values,
            "Molecule Name": te_df["name"].values,
            "pEC50": deploy_pred_513,
        })
        deploy_csv = SUBMISSIONS_DIR / f"{TAG}_deploy_slsqp4.csv"
        sub.to_csv(deploy_csv, index=False)
        out["deploy_csv"] = str(deploy_csv)
        out["deploy_pred_mean"] = round(float(deploy_pred_513.mean()), 4)
        out["deploy_pred_std"] = round(float(deploy_pred_513.std()), 4)
        out["deploy_in_sample_unb_rae"] = round(float(rae(y, deploy_pred_513[unb_idx])), 4)
    else:
        out["deploy_csv"] = None
        out["skip_reason"] = (f"scaffold_cv_rae={scaffold_cv_rae:.4f} > "
                              f"gate={DECISION_GATE:.4f}; not deploying")

    out["wall_sec"] = round(time.time() - t0, 2)

    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Compact console echo
    print(json.dumps({
        "tag": TAG,
        "indiv_rae_unb": out["indiv_rae_unb"],
        "scaffold_cv_rae_oos_concat": scaffold_cv_rae,
        "gate": DECISION_GATE,
        "passes_gate": out["passes_gate"],
        "any_fold_degenerate": out["any_fold_degenerate"],
        "mean_of_fold_weights": out["mean_of_fold_weights"],
        "full_pool_weights": out["full_pool_slsqp"]["weights"],
        "deploy_csv": out["deploy_csv"],
        "wall_sec": out["wall_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
