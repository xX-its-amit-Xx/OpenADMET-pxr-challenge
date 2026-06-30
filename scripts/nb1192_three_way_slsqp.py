"""nb1192 -- 3-way SLSQP simplex on top-3 honest candidates from cycles 143-146.

Top-3 anchors (honest scaffold-CV cross-fit RAE on 253 unblind):
    A = nb1162  (RAE 0.4204) -- stacking pyramid: SLSQP({nb2103_K28, chemprop_aux,
                                nb730_honest, nb503, nb562}) + per-fold rank-stretch.
    B = nb1150  (RAE 0.4710) -- SLSQP({chemprop_aux, nb503, nb1014, nb2103_K28}).
    C = nb1158  (RAE 0.5012) -- K=32 LGBM(MSE) scaffold-bag on chemprop_aux residual.

Neither nb1162 nor nb1150 cached its concatenated OOS prediction vector;
both saved the per-fold simplex weights and constituent anchor OOF paths.
We RECONSTRUCT each meta-anchor's 253-OOF by applying the cached per-fold
weights (held-out portion only) to the constituent anchor OOFs -- this is
LB-honest because each fold's weight was fit on TRAIN only.
nb1158 cached its bag OOF directly.

PROTOCOL:
  1. Build 3x253 OOF matrix.
  2. For each kf_seed in {1001..1005}:
       scaffold-5-fold on Murcko scaffolds of the 253 unblind SMILES;
       per fold, SLSQP simplex (w >= 0, sum w = 1) on TRAIN, score VAL.
       Pool VAL preds -> scaffold-CV RAE.
       Track per-fold weight vectors.
  3. Report mean +/- std RAE across seeds.
  4. Degeneracy = any-fold max(w) > 0.85.
  5. Gate: mean_rae <= 0.43 AND mean-of-fold w[nb1162] in [0.40, 0.80].
  6. If gate passes -> deploy = mean-of-fold weights x stack(te_nb1162, te_nb1150, te_nb1158).

Outputs:
    scripts/nb1192_three_way_slsqp.py (this file)
    data/processed/nb1192_summary.json
    submissions/nb1192_three_way_slsqp.csv  (only on gate pass)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
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

TAG = "nb1192"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

DECISION_GATE_RAE = 0.43
DEGEN_MAX_W = 0.85
NB1162_W_LO, NB1162_W_HI = 0.40, 0.80
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y   = DATA_PROCESSED / "_audit_unblind_y.npy"

# Deploy te files for the three meta-anchors
TE_PATHS = {
    "nb1162": DATA_PROCESSED / "te_nb1162.npy",
    "nb1150": DATA_PROCESSED / "te_nb1150.npy",
    "nb1158": DATA_PROCESSED / "te_nb1158.npy",
}
ANCHOR_NAMES = ["nb1162", "nb1150", "nb1158"]


# ---------------------------------------------------------------------------
# Reconstruct nb1162 OOF: SLSQP({nb2103_K28, chemprop_aux, nb730_honest,
# nb503, nb562}) with per-fold weights cached in nb1162_summary.json.
# Each fold's weights were fit on TRAIN-only, so applying them to held-out
# VAL anchor preds is LB-honest.
# ---------------------------------------------------------------------------
def reconstruct_nb1162_oof(y: np.ndarray, smi_unb: list[str]) -> np.ndarray:
    summary = json.loads((DATA_PROCESSED / "nb1162_summary.json").read_text())
    anchor_paths = [DATA_PROCESSED / p for p in summary["anchor_oof_paths"]]
    A = np.stack([np.load(p).astype(np.float64) for p in anchor_paths], axis=1)  # (253, 5)
    assert A.shape == (253, 5)

    # nb1162 used scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)
    scaffolds = [_murcko_scaffold(s) for s in smi_unb]
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)

    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fi, (tr_idx, va_idx) in enumerate(folds):
        fr = summary["fold_results"][fi]
        w = np.asarray(fr["w"], dtype=np.float64)
        # nb1162 also has per-fold mu shift but s=1 in all folds so reduces to
        # blend = A @ w. We additionally honor the cached mu shift form
        # pred = mu + s * (A @ w - mu) which collapses to A @ w when s=1.
        s = float(fr.get("s", 1.0))
        mu_tr = float(fr.get("mu_tr", 0.0))
        blend = A[va_idx] @ w
        oof[va_idx] = mu_tr + s * (blend - mu_tr) if s != 1.0 else blend
    assert not np.any(np.isnan(oof)), "nb1162 OOF coverage gap"
    return oof


# ---------------------------------------------------------------------------
# Reconstruct nb1150 OOF: SLSQP({chemprop_aux, nb503, nb1014, nb2103_K28}).
# ---------------------------------------------------------------------------
def reconstruct_nb1150_oof(y: np.ndarray, smi_unb: list[str]) -> np.ndarray:
    summary = json.loads((DATA_PROCESSED / "nb1150_summary.json").read_text())
    name_to_path = {
        "chemprop_aux": DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
        "nb503":        DATA_PROCESSED / "nb503_pred_oof.npy",
        "nb1014":       DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy",
        "nb2112":       DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
    }
    anchor_names = summary["anchors"]  # ['chemprop_aux','nb503','nb1014','nb2112']
    A = np.stack([np.load(name_to_path[n]).astype(np.float64) for n in anchor_names], axis=1)
    assert A.shape == (253, 4)

    scaffolds = [_murcko_scaffold(s) for s in smi_unb]
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)

    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fi, (tr_idx, va_idx) in enumerate(folds):
        fr = summary["per_fold"][fi]
        wd = fr["weights"]
        w = np.asarray([wd[n] for n in anchor_names], dtype=np.float64)
        oof[va_idx] = A[va_idx] @ w
    assert not np.any(np.isnan(oof)), "nb1150 OOF coverage gap"
    return oof


def _murcko_scaffold(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) or ""
    except Exception:
        return ""


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 8,
                   seed: int = 0) -> tuple[np.ndarray, float]:
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w):
        return rae(y, P @ w)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    # corner starts for K=3 to escape interior local minima
    for k in range(K):
        x = np.zeros(K)
        x[k] = 1.0
        starts.append(x)
    for _ in range(max(0, n_starts - 1 - K)):
        starts.append(rng.dirichlet(np.ones(K)))

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

    # ---- load truth + unblind smiles ----
    unb_idx = np.load(UNBLIND_IDX)
    y = np.load(UNBLIND_Y).astype(np.float64)
    n = len(y)
    assert n == 253

    te = load_test()
    smi_unb = te.iloc[unb_idx]["smiles"].tolist()

    # ---- build OOF matrix (3, 253) ----
    oof_1162 = reconstruct_nb1162_oof(y, smi_unb)
    oof_1150 = reconstruct_nb1150_oof(y, smi_unb)
    oof_1158 = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    assert oof_1158.shape == (253,)
    P = np.stack([oof_1162, oof_1150, oof_1158], axis=1)

    indiv_rae = {
        "nb1162": round(float(rae(y, oof_1162)), 4),
        "nb1150": round(float(rae(y, oof_1150)), 4),
        "nb1158": round(float(rae(y, oof_1158)), 4),
    }

    # ---- multi-seed scaffold-CV SLSQP ----
    scaffolds = [_murcko_scaffold(s) for s in smi_unb]
    n_scaf = len(set(s for s in scaffolds if s))

    seed_results = []
    per_seed_rae = []
    per_seed_fold_w = []  # list of (5,3) arrays
    per_seed_pool_w = []  # list of (3,) arrays (single-pool SLSQP per seed)
    for seed in KF_SEEDS:
        folds = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed)
        blended = np.full(n, np.nan, dtype=np.float64)
        fw, fold_info = [], []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            w, r_train = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
            val_pred = P[va_idx] @ w
            blended[va_idx] = val_pred
            r_val = rae(y[va_idx], val_pred)
            fold_info.append({
                "fold": fi,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "weights": {ANCHOR_NAMES[k]: round(float(w[k]), 4) for k in range(3)},
                "train_rae": round(float(r_train), 4),
                "val_rae": round(float(r_val), 4),
                "max_w": round(float(w.max()), 4),
                "degenerate": bool(w.max() > DEGEN_MAX_W),
            })
            fw.append(w)
        fw_arr = np.stack(fw, axis=0)
        per_seed_fold_w.append(fw_arr)

        scv_rae = float(rae(y, blended))
        per_seed_rae.append(scv_rae)

        w_pool, r_pool = _simplex_slsqp(P, y, n_starts=12, seed=seed)
        per_seed_pool_w.append(w_pool)

        seed_results.append({
            "seed": seed,
            "scaffold_cv_rae": round(scv_rae, 4),
            "per_fold": fold_info,
            "mean_of_fold_w": {ANCHOR_NAMES[k]: round(float(fw_arr.mean(axis=0)[k]), 4)
                               for k in range(3)},
            "any_fold_degenerate": bool(any(f["degenerate"] for f in fold_info)),
            "pool_w": {ANCHOR_NAMES[k]: round(float(w_pool[k]), 4) for k in range(3)},
            "pool_rae_in_sample": round(float(r_pool), 4),
        })

    per_seed_rae_arr = np.asarray(per_seed_rae)
    rae_mean = float(per_seed_rae_arr.mean())
    rae_std  = float(per_seed_rae_arr.std(ddof=0))

    # ---- aggregate weights across seeds and folds (25 weight vectors) ----
    all_fold_w = np.concatenate(per_seed_fold_w, axis=0)  # (25, 3)
    w_mean_all = all_fold_w.mean(axis=0)
    w_mean_all = w_mean_all / w_mean_all.sum()
    any_degen = bool(any(sr["any_fold_degenerate"] for sr in seed_results))

    out = {
        "tag": TAG,
        "anchors": ANCHOR_NAMES,
        "anchor_te_paths": {k: str(v) for k, v in TE_PATHS.items()},
        "indiv_rae_unb": indiv_rae,
        "n_unique_scaffolds_unb": n_scaf,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "per_seed_scaffold_cv_rae": [round(r, 4) for r in per_seed_rae],
        "rae_mean": round(rae_mean, 4),
        "rae_std": round(rae_std, 4),
        "seed_results": seed_results,
        "aggregate_mean_of_fold_w": {ANCHOR_NAMES[k]: round(float(w_mean_all[k]), 4)
                                     for k in range(3)},
        "any_fold_degenerate": any_degen,
        "gate_rae_target": DECISION_GATE_RAE,
        "gate_nb1162_lo": NB1162_W_LO,
        "gate_nb1162_hi": NB1162_W_HI,
    }

    # ---- gate ----
    w_nb1162 = float(w_mean_all[0])
    gate_a = rae_mean <= DECISION_GATE_RAE
    gate_b = (NB1162_W_LO <= w_nb1162 <= NB1162_W_HI)
    passes = bool(gate_a and gate_b)
    out["gate_a_rae_le_target"] = bool(gate_a)
    out["gate_b_nb1162_in_band"] = bool(gate_b)
    out["passes_gate"] = passes

    # ---- deploy ----
    deploy_csv = None
    if passes:
        te_mat = np.stack([np.load(TE_PATHS[n]).astype(np.float64)
                           for n in ANCHOR_NAMES], axis=1)
        deploy_pred = te_mat @ w_mean_all
        sub = pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": deploy_pred,
        })
        deploy_csv = SUBMISSIONS_DIR / f"{TAG}_three_way_slsqp.csv"
        sub.to_csv(deploy_csv, index=False)
        out["deploy_csv"] = str(deploy_csv)
        out["deploy_pred_mean"] = round(float(deploy_pred.mean()), 4)
        out["deploy_pred_std"] = round(float(deploy_pred.std()), 4)
        out["deploy_in_sample_unb_rae"] = round(float(rae(y, deploy_pred[unb_idx])), 4)
    else:
        out["deploy_csv"] = None
        out["skip_reason"] = (
            f"rae_mean={rae_mean:.4f} (gate <= {DECISION_GATE_RAE}) "
            f"AND w[nb1162]={w_nb1162:.4f} (gate in [{NB1162_W_LO}, {NB1162_W_HI}]): "
            f"gate_a={gate_a}, gate_b={gate_b}"
        )

    out["wall_sec"] = round(time.time() - t0, 2)

    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "tag": TAG,
        "indiv_rae_unb": out["indiv_rae_unb"],
        "per_seed_scaffold_cv_rae": out["per_seed_scaffold_cv_rae"],
        "rae_mean": out["rae_mean"],
        "rae_std": out["rae_std"],
        "aggregate_mean_of_fold_w": out["aggregate_mean_of_fold_w"],
        "any_fold_degenerate": out["any_fold_degenerate"],
        "gate_a_rae_le_target": out["gate_a_rae_le_target"],
        "gate_b_nb1162_in_band": out["gate_b_nb1162_in_band"],
        "passes_gate": out["passes_gate"],
        "deploy_csv": out["deploy_csv"],
        "wall_sec": out["wall_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
