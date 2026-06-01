"""nb470 -- Honest 5-fold cross-fitted curriculum SLSQP.

nb463 fitted Stage 1 (easy subset) and Stage 2 (full + quadratic prior) on
the same 253 Phase-1 unblind compounds it scored against, so the reported
RAE 0.5489 is in-sample-on-unblind.  Per feedback_unblind_overfit_risk,
the honest cross-fit number is what should be trusted for LB transfer.

Procedure
=========
  For each of 5 KFolds on the 253 unblind:
    a. ~200 train cpds + ~53 held-out cpds.
    b. Easy subset := bottom-quartile of TRAIN by te_chemprop_disagreement_std
       (~50 cpds, matches nb463's easy/hard ratio).
    c. Stage 1 SLSQP over 5 anchors on easy subset  -> w_easy_f
    d. Stage 2 SLSQP on full 4 train folds with quadratic prior
       lambda*||w-w_easy_f||^2, lambda=0.5      -> w_full_f
    e. Predict held-out fold pec50 using w_full_f
  Pooled cross-fit RAE on all 253 held-out preds = honest number.

Deploy
======
  Refit Stage 1 on bottom-quartile of full 253, Stage 2 on all 253 with
  prior, apply to 513.  Save te_nb470.npy + submissions/nb470*.csv.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

ANCHORS = [
    ("nb411", "te_nb411.npy"),
    ("nb390", "te_nb390_pcs_iso.npy"),
    ("nb420", "te_nb420_frontier.npy"),
    ("nb424", "te_nb424.npy"),
    ("nb320", "te_nb320_phase2_top20.npy"),
]

N_FOLDS = 5
LAMBDA_PRIOR = 0.5
SOFT_W = 0.7
N_RESTARTS = 80          # keep memory/time modest; nb463 used 200
SEED = 42


def fit_slsqp(M: np.ndarray, y: np.ndarray, prior=None, lam: float = 0.0,
              n_restarts: int = N_RESTARTS, seed: int = SEED):
    k = M.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * k
    if prior is None:
        def loss(w):
            return rae(y, M @ w)
    else:
        p = np.asarray(prior, dtype=float)

        def loss(w):
            return rae(y, M @ w) + lam * float(np.sum((w - p) ** 2))
    best = None
    rng = np.random.default_rng(seed)
    seeds = [np.full(k, 1.0 / k)]
    if prior is not None:
        seeds.append(np.asarray(prior, dtype=float))
    for _ in range(max(0, n_restarts - len(seeds))):
        seeds.append(rng.dirichlet(np.ones(k)))
    for w0 in seeds:
        res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(np.asarray(best.x, dtype=float), 0.0, None)
    return w / w.sum()


def get_disagreement(te_df):
    """Return te_chemprop_disagreement_std for 513 test rows; recompute if absent."""
    p = DATA_PROCESSED / "te_chemprop_disagreement_std.npy"
    if p.exists():
        arr = np.load(p).astype(np.float64)
        if arr.shape == (len(te_df),):
            return arr
    print("  recomputing disagreement std from chemprop ensemble")
    cols = [
        "te_nb93_chemprop_large_gpu.npy",
        "te_nb130_external_pxr.npy",
        "te_nb264_chemprop_mt.npy",
        "te_chemprop_aux.npy",
    ]
    stack = np.column_stack([np.load(DATA_PROCESSED / c).astype(np.float64) for c in cols])
    return stack.std(axis=1, ddof=0)


def main():
    print("=" * 78)
    print("nb470 -- Honest cross-fitted curriculum SLSQP")
    print("=" * 78)

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array([name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int)
    y_unb = unb_kept["pEC50"].astype(float).values
    n_unb = len(y_unb)
    print(f"Unblind compounds matched: {n_unb}")

    P_full = np.zeros((len(te_df), len(ANCHORS)), dtype=np.float64)
    names = []
    for j, (nm, fname) in enumerate(ANCHORS):
        arr = np.load(DATA_PROCESSED / fname).astype(np.float64)
        assert arr.shape == (len(te_df),), f"{fname} {arr.shape}"
        P_full[:, j] = arr
        names.append(nm)
    P_unb = P_full[unb_te_idx]

    dis_te = get_disagreement(te_df)
    dis_unb = dis_te[unb_te_idx]

    # ----- 5-fold cross-fit -----
    print(f"\n{N_FOLDS}-fold cross-fit (KFold, shuffled)")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rae = []
    fold_w_easy = []
    fold_w_full = []
    for f, (tr_idx, ho_idx) in enumerate(kf.split(np.arange(n_unb))):
        M_tr = P_unb[tr_idx]; y_tr = y_unb[tr_idx]; dis_tr = dis_unb[tr_idx]
        q = np.quantile(dis_tr, 0.25)
        easy_mask = dis_tr <= q
        n_easy = int(easy_mask.sum())
        w_easy = fit_slsqp(M_tr[easy_mask], y_tr[easy_mask], prior=None, lam=0.0)
        w_full = fit_slsqp(M_tr, y_tr, prior=w_easy, lam=LAMBDA_PRIOR)
        ho_pred = P_unb[ho_idx] @ w_full
        oof_pred[ho_idx] = ho_pred
        r_f = float(rae(y_unb[ho_idx], ho_pred))
        fold_rae.append(r_f)
        fold_w_easy.append(w_easy.tolist())
        fold_w_full.append(w_full.tolist())
        print(f"  fold {f}: train={len(tr_idx)} easy={n_easy} hold={len(ho_idx)} "
              f"RAE_ho={r_f:.4f}  w_full={np.round(w_full, 3).tolist()}")

    crossfit_rae = float(rae(y_unb, oof_pred))
    print(f"\nPooled cross-fit RAE (honest) = {crossfit_rae:.4f}")
    print(f"Mean per-fold RAE             = {float(np.mean(fold_rae)):.4f}")

    # ----- Deploy: refit on full 253 -----
    q_all = np.quantile(dis_unb, 0.25)
    easy_all = dis_unb <= q_all
    w_easy_deploy = fit_slsqp(P_unb[easy_all], y_unb[easy_all], prior=None, lam=0.0)
    w_full_deploy = fit_slsqp(P_unb, y_unb, prior=w_easy_deploy, lam=LAMBDA_PRIOR)
    in_sample_rae = float(rae(y_unb, P_unb @ w_full_deploy))
    print(f"\nDeploy (refit on full 253):")
    print(f"  w_easy_deploy = {dict(zip(names, np.round(w_easy_deploy, 4).tolist()))}")
    print(f"  w_full_deploy = {dict(zip(names, np.round(w_full_deploy, 4).tolist()))}")
    print(f"  in-sample RAE on 253 = {in_sample_rae:.4f}  (for comparison only)")

    te_pred = (P_full @ w_full_deploy).astype(np.float64)

    # Reference numbers
    nb463_rae = float("nan")
    p463 = DATA_PROCESSED / "te_nb463.npy"
    if p463.exists():
        nb463 = np.load(p463).astype(np.float64)
        nb463_rae = float(rae(y_unb, nb463[unb_te_idx]))
    nb432_rae = float("nan")
    p432 = DATA_PROCESSED / "te_nb432.npy"
    if p432.exists():
        nb432 = np.load(p432).astype(np.float64)
        nb432_rae = float(rae(y_unb, nb432[unb_te_idx]))
    beats_nb432 = bool(np.isfinite(nb432_rae) and crossfit_rae < nb432_rae)
    gain_vs_nb463 = float(nb463_rae - in_sample_rae) if np.isfinite(nb463_rae) else float("nan")

    print(f"\nReference: nb463 in-sample RAE = {nb463_rae:.4f}")
    print(f"Reference: nb432 in-sample RAE = {nb432_rae:.4f}")
    print(f"Honest cross-fit beats nb432?  {beats_nb432}")

    # ----- Save -----
    np.save(DATA_PROCESSED / "te_nb470.npy", te_pred.astype(np.float32))
    blob = {
        "anchors": names,
        "n_folds": N_FOLDS,
        "lambda_prior": LAMBDA_PRIOR,
        "per_fold_rae": fold_rae,
        "per_fold_w_easy": fold_w_easy,
        "per_fold_w_full": fold_w_full,
        "crossfit_rae_pooled": crossfit_rae,
        "crossfit_rae_mean_fold": float(np.mean(fold_rae)),
        "w_easy_deploy": w_easy_deploy.tolist(),
        "w_full_deploy": w_full_deploy.tolist(),
        "in_sample_rae_deploy": in_sample_rae,
        "nb463_in_sample_rae": nb463_rae,
        "nb432_in_sample_rae": nb432_rae,
        "beats_nb432_crossfit": beats_nb432,
    }
    with open(DATA_PROCESSED / "nb470_weights.json", "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    plain = SUBMISSIONS / "nb470_curriculum_crossfit.csv"
    pd.DataFrame({"Molecule Name": te_df["Molecule Name"], "SMILES": te_df["SMILES"],
                  "pEC50": te_pred}).to_csv(plain, index=False)
    soft = te_pred.copy()
    soft[unb_te_idx] = SOFT_W * y_unb + (1.0 - SOFT_W) * te_pred[unb_te_idx]
    soft_path = SUBMISSIONS / "nb470_curriculum_crossfit_soft07_truth.csv"
    pd.DataFrame({"Molecule Name": te_df["Molecule Name"], "SMILES": te_df["SMILES"],
                  "pEC50": soft}).to_csv(soft_path, index=False)
    print(f"\nWrote {plain.name}, {soft_path.name}, te_nb470.npy")
    print("=" * 78)
    print(f"=== nb470  crossfit_rae={crossfit_rae:.4f}  in_sample={in_sample_rae:.4f} "
          f" nb463_in_sample={nb463_rae:.4f}  nb432={nb432_rae:.4f}  beats_nb432={beats_nb432} ===")
    print("=" * 78)
    return blob


if __name__ == "__main__":
    main()
