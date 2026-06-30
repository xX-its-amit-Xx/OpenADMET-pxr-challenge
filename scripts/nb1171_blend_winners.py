"""nb1171 -- Blend the two cycle-143 verified winners.

WINNERS:
    A. nb1150 (4-anchor SLSQP simplex; nested scaffold-CV RAE 0.4710)
       OOF is regenerated here (the script doesn't persist its blended_oof).
       Anchors: chemprop_aux, nb503, nb1014, nb2103_K28 mean-bag.
    B. nb1158 (LGBM-MSE residual on chemprop_aux, K=32 scaffold-fold mean-bag,
       diagnostic per-seed mean = 0.4902 fresh-seed cross-fit).
       OOF: data/processed/nb1158_mean_bag_oof_K32.npy

STRATEGIES tested under outer scaffold-5-fold:
    1. Simple 50/50 mean
    2. Convex SLSQP on the simplex (weights fit per outer fold on TRAIN portion)
    3. Rank-stretch on the SLSQP blend (scalar s in [0.95, 1.20], grid)

GATE: scaffold-CV RAE <= 0.4680 (= 0.4710 - 0.003).
If passes -> deploy CSV combining te_nb1150.npy and te_nb1158.npy with the
mean-of-fold SLSQP weights (+ optional rank-stretch if the third strategy wins).
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

TAG = "nb1171"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

DECISION_GATE = 0.4680  # = nb1150 nested cross-fit 0.4710 - 0.003
DEGEN_MAX_W = 0.85

# nb1150 anchor layout (must match nb1150_slsqp_simplex.py exactly)
ANCHOR_OOF_PATHS = {
    "chemprop_aux": DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
    "nb503":        DATA_PROCESSED / "nb503_pred_oof.npy",
    "nb1014":       DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy",
    "nb2112":       DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
}
ANCHOR_TE_PATHS = {
    "chemprop_aux": DATA_PROCESSED / "te_chemprop_aux.npy",
    "nb503":        DATA_PROCESSED / "te_nb503.npy",
    "nb1014":       DATA_PROCESSED / "te_nb1014.npy",
    "nb2112":       DATA_PROCESSED / "te_nb2112.npy",
}
ANCHOR_NAMES = list(ANCHOR_OOF_PATHS.keys())

NB1158_OOF_PATH = DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
NB1150_TE_PATH = DATA_PROCESSED / "te_nb1150.npy"
NB1158_TE_PATH = DATA_PROCESSED / "te_nb1158.npy"

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y   = DATA_PROCESSED / "_audit_unblind_y.npy"

STRETCH_GRID = np.round(np.arange(0.95, 1.205, 0.01), 3)


def _murcko_scaffold(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) or ""
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
    for _ in range(n_starts - 1):
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


def _reconstruct_nb1150_oof(y: np.ndarray, scaffolds: list[str]) -> np.ndarray:
    """Recompute nb1150's nested scaffold-CV blended OOF on the 253 unblind."""
    P_list = []
    for name in ANCHOR_NAMES:
        P_list.append(np.load(ANCHOR_OOF_PATHS[name]).astype(np.float64))
    P = np.stack(P_list, axis=1)  # (253, 4)
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)
    blended = np.full(len(y), np.nan, dtype=np.float64)
    for fi, (tr_idx, va_idx) in enumerate(folds):
        w, _ = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
        blended[va_idx] = P[va_idx] @ w
    assert not np.any(np.isnan(blended))
    return blended


def main() -> None:
    t0 = time.time()
    out: dict = {"tag": TAG, "decision_gate": DECISION_GATE,
                 "winners": {"A": "nb1150_SLSQP4_nested_scaffoldCV",
                             "B": "nb1158_LGBM_K32_residual_meanBag"}}

    # ----- load truth + scaffolds -----
    unb_idx = np.load(UNBLIND_IDX)
    y = np.load(UNBLIND_Y).astype(np.float64)
    n = len(y)
    assert n == 253
    te_df = load_test()
    smi_unb = te_df.iloc[unb_idx]["smiles"].tolist()
    scaffolds = [_murcko_scaffold(s) for s in smi_unb]
    out["n_unique_scaffolds"] = len(set([s for s in scaffolds if s]))

    # ----- load A (reconstruct nb1150 OOF) and B (nb1158 mean-bag K=32) -----
    oof_A = _reconstruct_nb1150_oof(y, scaffolds)
    oof_B = np.load(NB1158_OOF_PATH).astype(np.float64)
    assert oof_B.shape == (n,), f"nb1158 OOF wrong shape {oof_B.shape}"

    rae_A = float(rae(y, oof_A))
    rae_B = float(rae(y, oof_B))
    out["indiv_rae"] = {"nb1150": round(rae_A, 4), "nb1158": round(rae_B, 4)}

    # ----- correlation -----
    pearson = float(np.corrcoef(oof_A, oof_B)[0, 1])
    spearman = float(pd.Series(oof_A).corr(pd.Series(oof_B), method="spearman"))
    resid_A = y - oof_A
    resid_B = y - oof_B
    resid_corr = float(np.corrcoef(resid_A, resid_B)[0, 1])
    out["correlations"] = {"pearson_pred": round(pearson, 4),
                            "spearman_pred": round(spearman, 4),
                            "pearson_residual": round(resid_corr, 4)}

    P = np.stack([oof_A, oof_B], axis=1)  # (253, 2)

    # ----- outer scaffold-CV (seed 42, n=5) -----
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)

    # Strategy 1: simple 50/50 mean
    s1 = 0.5 * oof_A + 0.5 * oof_B
    rae_s1 = float(rae(y, s1))

    # Strategy 2: convex SLSQP per fold (fit on TRAIN portion, predict VAL)
    blended2 = np.full(n, np.nan)
    fold_w = []
    per_fold = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        w, r_tr = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
        val_pred = P[va_idx] @ w
        blended2[va_idx] = val_pred
        r_va = float(rae(y[va_idx], val_pred))
        per_fold.append({"fold": fi, "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
                         "w_nb1150": round(float(w[0]), 4), "w_nb1158": round(float(w[1]), 4),
                         "train_rae": round(float(r_tr), 4), "val_rae": r_va,
                         "max_w": round(float(w.max()), 4),
                         "degenerate": bool(w.max() > DEGEN_MAX_W)})
        fold_w.append(w)
    fold_w = np.stack(fold_w, axis=0)
    w_mean = fold_w.mean(axis=0)
    w_mean = w_mean / w_mean.sum()
    rae_s2 = float(rae(y, blended2))

    # Strategy 3: rank-stretch on the SLSQP blend (cross-fit s per fold)
    # blended2 is OOS-concat; we now grid-search s on TRAIN portion of each fold and
    # apply that s to VAL — producing a second OOS-concat vector for honest evaluation.
    blended3 = np.full(n, np.nan)
    per_fold_s = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        # In-fold blend prediction (refit weights to be consistent with strategy 2)
        w_f = fold_w[fi]
        mu_tr = float((P[tr_idx] @ w_f).mean())
        pred_tr = P[tr_idx] @ w_f
        # grid-search best s on this fold's TRAIN
        best_s, best_r = 1.0, np.inf
        for s_try in STRETCH_GRID:
            cand = mu_tr + s_try * (pred_tr - mu_tr)
            r_try = float(rae(y[tr_idx], cand))
            if r_try < best_r:
                best_r, best_s = r_try, float(s_try)
        # apply to VAL
        pred_va = P[va_idx] @ w_f
        blended3[va_idx] = mu_tr + best_s * (pred_va - mu_tr)
        per_fold_s.append({"fold": fi, "best_s": best_s, "train_rae_at_s": round(best_r, 4)})
    rae_s3 = float(rae(y, blended3))

    out["strategies"] = {
        "mean50_50":         {"rae": round(rae_s1, 4)},
        "slsqp_per_fold":    {"rae": round(rae_s2, 4),
                              "mean_of_fold_weights": {"nb1150": round(float(w_mean[0]), 4),
                                                       "nb1158": round(float(w_mean[1]), 4)},
                              "any_fold_degenerate": bool(any(f["degenerate"] for f in per_fold)),
                              "per_fold": per_fold},
        "slsqp_then_rank_stretch": {"rae": round(rae_s3, 4),
                                     "per_fold_s": per_fold_s,
                                     "mean_s": round(float(np.mean([f["best_s"] for f in per_fold_s])), 4)},
    }

    # Choose best strategy
    strat_rae = {"mean50_50": rae_s1, "slsqp_per_fold": rae_s2, "slsqp_then_rank_stretch": rae_s3}
    best_name = min(strat_rae, key=strat_rae.get)
    best_rae = strat_rae[best_name]
    out["best_strategy"] = best_name
    out["best_rae"] = round(best_rae, 4)
    out["delta_vs_nb1150"] = round(best_rae - rae_A, 4)
    out["delta_vs_nb1158"] = round(best_rae - rae_B, 4)
    out["passes_gate"] = bool(best_rae <= DECISION_GATE)

    # ----- deploy if gate passes -----
    deploy_csv = None
    if out["passes_gate"]:
        te_A = np.load(NB1150_TE_PATH).astype(np.float64)
        te_B = np.load(NB1158_TE_PATH).astype(np.float64)
        assert te_A.shape == (513,) and te_B.shape == (513,)
        if best_name == "mean50_50":
            deploy_pred = 0.5 * te_A + 0.5 * te_B
        else:  # slsqp_per_fold or slsqp_then_rank_stretch
            deploy_pred = w_mean[0] * te_A + w_mean[1] * te_B
            if best_name == "slsqp_then_rank_stretch":
                # Use mean of fold-optimal s as a conservative deploy stretch
                mu_deploy = float(deploy_pred.mean())
                s_deploy = float(np.mean([f["best_s"] for f in per_fold_s]))
                deploy_pred = mu_deploy + s_deploy * (deploy_pred - mu_deploy)
                out["deploy_stretch_s"] = round(s_deploy, 4)
        sub = pd.DataFrame({
            "SMILES": te_df["smiles"].values,
            "Molecule Name": te_df["name"].values,
            "pEC50": deploy_pred,
        })
        deploy_csv = SUBMISSIONS_DIR / f"{TAG}_blend_{best_name}.csv"
        sub.to_csv(deploy_csv, index=False)
        out["deploy_csv"] = str(deploy_csv)
        out["deploy_pred_mean"] = round(float(deploy_pred.mean()), 4)
        out["deploy_pred_std"] = round(float(deploy_pred.std()), 4)
        out["deploy_in_sample_unb_rae"] = round(float(rae(y, deploy_pred[unb_idx])), 4)
    else:
        out["deploy_csv"] = None
        out["skip_reason"] = (f"best_rae={best_rae:.4f} > gate={DECISION_GATE:.4f}; "
                              f"no improvement vs nb1150 (Delta={out['delta_vs_nb1150']:+.4f})")

    out["wall_sec"] = round(time.time() - t0, 2)

    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "tag": TAG,
        "indiv_rae": out["indiv_rae"],
        "correlations": out["correlations"],
        "strategies": {k: v["rae"] for k, v in out["strategies"].items()},
        "best_strategy": best_name,
        "best_rae": out["best_rae"],
        "gate": DECISION_GATE,
        "passes_gate": out["passes_gate"],
        "deploy_csv": out["deploy_csv"],
        "wall_sec": out["wall_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
