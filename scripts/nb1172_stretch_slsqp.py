"""nb1172 -- Per-fold rank-stretch on the nb1150 SLSQP-blend scaffold-CV OOF.

CONTEXT
-------
nb1150 produced a scaffold-CV OOF on the 253 unblind by running SLSQP simplex
weighting over 4 anchors {chemprop_aux, nb503, nb1014, nb2112} per fold and
concatenating fold-validation predictions.  Scaffold-CV pooled RAE = 0.4710.

Per memory `feedback_rank_stretch_universal`: scalar rank-stretch
    pred' = mu_train + s * (pred - mu_train)
with s in [1.05, 1.10] reliably picks up -0.003 to -0.005 RAE on every
variance-compressed predictor because pred_std ~ 0.75 vs truth_std ~ 1.03 on
novel-scaffold OOD.

PROTOCOL
--------
1. Reconstruct nb1150 SLSQP scaffold-CV OOF (load anchors, scaffold-fold split,
   per-fold SLSQP simplex, concat val preds) -- identical to nb1150.  Verify
   pooled RAE == 0.4710 +/- 1e-4.
2. PER-FOLD variant:  for each fold compute the LOCAL stretch
        s_k = std(y[fold_train]) / std(pred_oof[fold_train])
   apply  stretched_va = mu_tr + s_k * (pred_oof[fold_val] - mu_tr)
   concatenate -> scaffold-CV RAE for "per_fold_local_s".
3. UNIVERSAL sweep: for each s in {1.00, 1.025, 1.05, 1.075, 1.10}:
     for each fold:
        mu_tr        = mean(pred_oof[fold_train])
        stretched_va = mu_tr + s * (pred_oof[fold_val] - mu_tr)
     concat -> pooled scaffold-CV RAE.
4. Decision gate: best variant must beat 0.4680 (i.e., -0.003 vs nb1150 0.4710).
5. If passes: build deploy CSV
        te_blend = sum_k w_mean[k] * te_anchor[k]   (513-vec, nb1150 deploy)
        mu_full  = mean(pred_oof)                   (fit on the 253 set)
        te_final = mu_full + s_best * (te_blend - mu_full)
   Saves submissions/nb1172_stretch_slsqp_{variant}.csv.

OUTPUTS
-------
scripts/nb1172_stretch_slsqp.py
data/processed/nb1172_summary.json
submissions/nb1172_stretch_slsqp_{variant}.csv  (only if gate passes)
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
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1172"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# Anchors -- identical to nb1150
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

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y   = DATA_PROCESSED / "_audit_unblind_y.npy"

# nb1150 scaffold-fold config (must match exactly)
N_FOLDS = 5
SCAFFOLD_SEED = 42

# Stretch sweep
S_GRID = [1.000, 1.025, 1.050, 1.075, 1.100]

# Gates
NB1150_BASELINE = 0.4710     # nb1150 scaffold-CV pooled RAE
DEPLOY_GATE     = 0.4680     # spec: must beat 0.4680
RECON_TOL       = 1e-3       # reconstruction sanity tolerance


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


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def main() -> None:
    t0 = time.time()
    out: dict = {
        "tag": TAG,
        "method": "per_fold_and_universal_rank_stretch_on_nb1150_slsqp_oof",
        "anchors": ANCHOR_NAMES,
        "n_folds": N_FOLDS,
        "scaffold_seed": SCAFFOLD_SEED,
        "s_grid": S_GRID,
        "nb1150_baseline_rae": NB1150_BASELINE,
        "deploy_gate": DEPLOY_GATE,
    }

    # ---- load 253-row anchors + truth ----
    unb_idx = np.load(UNBLIND_IDX)
    y = np.load(UNBLIND_Y).astype(np.float64)
    n = len(y)
    assert n == 253, f"expected 253 unblind, got {n}"

    P_list, indiv_rae = [], {}
    for name in ANCHOR_NAMES:
        oof = np.load(ANCHOR_OOF_PATHS[name]).astype(np.float64)
        assert oof.shape == (253,), f"{name} OOF wrong shape {oof.shape}"
        indiv_rae[name] = round(float(rae(y, oof)), 4)
        P_list.append(oof)
    P = np.stack(P_list, axis=1)  # (253, 4)
    out["indiv_rae_unb"] = indiv_rae

    # ---- scaffold fold split (must match nb1150) ----
    te_df_full = load_test()
    smi_unb = te_df_full.iloc[unb_idx]["smiles"].tolist()
    scaffolds = [_murcko_scaffold(s) for s in smi_unb]
    out["n_unique_scaffolds"] = len(set([s for s in scaffolds if s]))
    folds = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, shuffle=True,
                                   seed=SCAFFOLD_SEED)

    # ---- reconstruct nb1150 OOF (per-fold SLSQP, concat val preds) ----
    pred_oof = np.full(n, np.nan, dtype=np.float64)
    fold_weights = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        w, _ = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
        pred_oof[va_idx] = P[va_idx] @ w
        fold_weights.append(w)
    fold_weights = np.stack(fold_weights, axis=0)
    assert not np.any(np.isnan(pred_oof)), "OOF coverage gap during reconstruction"

    recon_rae = float(rae(y, pred_oof))
    out["recon_scaffold_cv_rae"] = round(recon_rae, 4)
    out["recon_matches_nb1150"] = bool(abs(recon_rae - NB1150_BASELINE) < RECON_TOL)
    if not out["recon_matches_nb1150"]:
        print(f"WARN: recon RAE {recon_rae:.4f} != nb1150 {NB1150_BASELINE:.4f}")

    # Variance diagnostics
    out["pred_oof_mean"] = round(float(pred_oof.mean()), 4)
    out["pred_oof_std"]  = round(float(pred_oof.std()), 4)
    out["truth_mean"]    = round(float(y.mean()), 4)
    out["truth_std"]     = round(float(y.std()), 4)
    out["variance_match_ideal_s"] = round(float(y.std() / max(pred_oof.std(), 1e-9)), 4)

    # ---- VARIANT A: per-fold LOCAL stretch s_k = std(y_tr)/std(pred_tr) ----
    per_fold_local_oof = np.full(n, np.nan, dtype=np.float64)
    per_fold_local_info = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        mu_tr  = float(pred_oof[tr_idx].mean())
        std_p  = float(pred_oof[tr_idx].std())
        std_y  = float(y[tr_idx].std())
        s_k    = float(std_y / max(std_p, 1e-9))
        per_fold_local_oof[va_idx] = _stretch(pred_oof[va_idx], mu_tr, s_k)
        per_fold_local_info.append({
            "fold": fi,
            "n_val": int(len(va_idx)),
            "mu_tr": round(mu_tr, 4),
            "std_pred_tr": round(std_p, 4),
            "std_y_tr": round(std_y, 4),
            "s_k": round(s_k, 4),
            "val_rae": round(float(rae(y[va_idx], per_fold_local_oof[va_idx])), 4),
        })
    per_fold_local_rae = round(float(rae(y, per_fold_local_oof)), 4)
    out["per_fold_local"] = {
        "scaffold_cv_rae": per_fold_local_rae,
        "delta_vs_nb1150": round(per_fold_local_rae - NB1150_BASELINE, 4),
        "folds": per_fold_local_info,
    }

    # ---- VARIANT B: universal s sweep ----
    per_s = []
    for s in S_GRID:
        oof_s = np.full(n, np.nan, dtype=np.float64)
        fold_raes = []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            mu_tr = float(pred_oof[tr_idx].mean())
            oof_s[va_idx] = _stretch(pred_oof[va_idx], mu_tr, s)
            fold_raes.append(float(rae(y[va_idx], oof_s[va_idx])))
        pooled = round(float(rae(y, oof_s)), 4)
        per_s.append({
            "s": float(s),
            "pooled_scaffold_cv_rae": pooled,
            "delta_vs_nb1150": round(pooled - NB1150_BASELINE, 4),
            "fold_raes": [round(r, 4) for r in fold_raes],
            "pred_std_after": round(float(oof_s.std()), 4),
        })
    out["per_s"] = per_s

    # ---- pick best across both variants ----
    candidates = [
        {"variant": "per_fold_local", "rae": per_fold_local_rae, "s_label": "per_fold_local"},
    ] + [{"variant": f"universal_s{r['s']:.3f}", "rae": r["pooled_scaffold_cv_rae"],
          "s_label": r["s"]}
         for r in per_s]
    best = min(candidates, key=lambda r: r["rae"])
    out["best_variant"] = best["variant"]
    out["best_scaffold_cv_rae"] = best["rae"]
    out["best_delta_vs_nb1150"] = round(best["rae"] - NB1150_BASELINE, 4)
    out["passes_deploy_gate"] = bool(best["rae"] <= DEPLOY_GATE)

    # ---- deploy if gate passes ----
    deploy_csv = None
    if out["passes_deploy_gate"]:
        # mean-of-fold weights for the SLSQP blend (matches nb1150 deploy)
        w_mean = fold_weights.mean(axis=0)
        w_mean = w_mean / w_mean.sum()
        out["mean_of_fold_weights"] = {ANCHOR_NAMES[k]: round(float(w_mean[k]), 4)
                                       for k in range(len(w_mean))}

        te_mat = np.stack([np.load(ANCHOR_TE_PATHS[name]).astype(np.float64)
                           for name in ANCHOR_NAMES], axis=1)  # (513, 4)
        te_blend = te_mat @ w_mean  # nb1150 deploy 513-vec

        if best["variant"] == "per_fold_local":
            # Deploy with the FULL-pool local stretch (no fold available at deploy)
            mu_full = float(pred_oof.mean())
            s_full = float(y.std() / max(pred_oof.std(), 1e-9))
            te_final = _stretch(te_blend, mu_full, s_full)
            variant_tag = f"perfoldlocal_sFull{s_full:.3f}"
            out["deploy_s_full"] = round(s_full, 4)
        else:
            s_best = float(best["s_label"])
            mu_full = float(pred_oof.mean())
            te_final = _stretch(te_blend, mu_full, s_best)
            variant_tag = f"universal_s{s_best:.3f}"
            out["deploy_s"] = s_best

        te_df = load_test()
        sub = pd.DataFrame({
            "SMILES": te_df["smiles"].values,
            "Molecule Name": te_df["name"].values,
            "pEC50": te_final.astype(np.float32),
        })
        deploy_csv = SUBMISSIONS_DIR / f"{TAG}_stretch_slsqp_{variant_tag}.csv"
        sub.to_csv(deploy_csv, index=False)

        out["deploy_csv"] = str(deploy_csv)
        out["deploy_pred_mean"] = round(float(te_final.mean()), 4)
        out["deploy_pred_std"] = round(float(te_final.std()), 4)
        out["deploy_in_sample_unb_rae"] = round(float(rae(y, te_final[unb_idx])), 4)
    else:
        out["deploy_csv"] = None
        out["skip_reason"] = (f"best_rae={best['rae']:.4f} > "
                              f"gate={DEPLOY_GATE:.4f}; not deploying")

    out["wall_sec"] = round(time.time() - t0, 2)

    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "tag": TAG,
        "recon_rae": out["recon_scaffold_cv_rae"],
        "recon_matches_nb1150": out["recon_matches_nb1150"],
        "per_fold_local_rae": out["per_fold_local"]["scaffold_cv_rae"],
        "universal_sweep": {r["s"]: r["pooled_scaffold_cv_rae"] for r in per_s},
        "best_variant": out["best_variant"],
        "best_rae": out["best_scaffold_cv_rae"],
        "delta_vs_nb1150": out["best_delta_vs_nb1150"],
        "passes_gate": out["passes_deploy_gate"],
        "deploy_csv": out["deploy_csv"],
        "wall_sec": out["wall_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
