"""nb463 -- DynCIM-style curriculum SLSQP over 5 diverse anchors.

Motivation
==========
Standard SLSQP over the full 253 Phase-1 unblind set chases noisy cliff
compounds, so the fitted weights overfit to the unblind labels.  Dynamic
Curriculum Importance Mining (DynCIM) instead learns on the easy regime
first, then mildly perturbs weights when exposing the model to hard cases.

Pipeline
========
  Stage 1 (easy):  Fit unconstrained simplex-SLSQP over the 5 most diverse
                   anchors (nb411, nb390, nb420, nb424, nb320) using only
                   the 200 unblind compounds with the LOWEST Chemprop
                   ensemble disagreement (te_chemprop_disagreement_std).
                   Save w_easy.

  Stage 2 (hard):  Re-fit on all 253 unblind, but add a quadratic prior
                   penalty  lambda * ||w - w_easy||^2  with lambda=0.5,
                   so weights are gently pulled back toward the easy
                   solution.  Save w_full.

  Apply w_full to the full 513 test predictions, compute honest unblind
  RAE, save te_nb463.npy and two submissions (plain + soft07 truth).
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

# ---- Anchors (per nb435 diversity audit) ----
ANCHORS = [
    ("nb411", "te_nb411.npy"),
    ("nb390", "te_nb390_pcs_iso.npy"),
    ("nb420", "te_nb420_frontier.npy"),
    ("nb424", "te_nb424.npy"),
    ("nb320", "te_nb320_phase2_top20.npy"),
]

N_EASY = 200
LAMBDA_PRIOR = 0.5
SOFT_W = 0.7
N_RESTARTS = 200
SEED = 42


def fit_slsqp(M: np.ndarray, y: np.ndarray, prior: np.ndarray | None = None,
              lam: float = 0.0, n_restarts: int = N_RESTARTS, seed: int = SEED):
    """Fit simplex-constrained SLSQP minimising RAE (+ optional quadratic prior).

    M: (n, k) per-compound predictions, y: (n,) targets.
    Returns scipy OptimizeResult of the best restart.
    """
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
    rng_master = np.random.default_rng(seed)
    # always seed first restart at uniform; remaining are Dirichlet draws
    seeds = [np.full(k, 1.0 / k)]
    if prior is not None:
        seeds.append(p.copy())
    for _ in range(n_restarts - len(seeds)):
        seeds.append(rng_master.dirichlet(np.ones(k)))
    for w0 in seeds:
        res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    return best


def main():
    print("=" * 78)
    print("nb463 -- DynCIM curriculum SLSQP over 5 diverse anchors")
    print("=" * 78)

    # ---------- Load test frame & 253 unblind labels ----------
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array([name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int)
    y_unb = unb_kept["pEC50"].astype(float).values
    n_unb = len(y_unb)
    print(f"\nUnblind compounds matched to test rows: {n_unb}")

    # ---------- Load anchors ----------
    print("\nLoading anchors")
    P_full = np.zeros((len(te_df), len(ANCHORS)), dtype=np.float64)
    names = []
    for j, (nm, fname) in enumerate(ANCHORS):
        arr = np.load(DATA_PROCESSED / fname).astype(np.float64)
        assert arr.shape == (len(te_df),), f"{fname} wrong shape {arr.shape}"
        P_full[:, j] = arr
        names.append(nm)
        r_solo = rae(y_unb, arr[unb_te_idx])
        print(f"  {nm:8s}  mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"unblind-RAE={r_solo:.4f}")

    P_unb = P_full[unb_te_idx]  # (253, 5)

    # ---------- Chemprop disagreement to score "easiness" ----------
    dis_path = DATA_PROCESSED / "te_chemprop_disagreement_std.npy"
    if not dis_path.exists():
        raise FileNotFoundError(f"Missing {dis_path}")
    dis_te = np.load(dis_path).astype(np.float64)
    dis_unb = dis_te[unb_te_idx]
    easy_order = np.argsort(dis_unb)
    n_easy = min(N_EASY, n_unb)
    easy_mask = np.zeros(n_unb, dtype=bool)
    easy_mask[easy_order[:n_easy]] = True
    print(f"\nEasy subset: {n_easy} / {n_unb}  "
          f"(disagreement_std <= {dis_unb[easy_order[n_easy - 1]]:.4f})")

    # ============================================================
    # Stage 1: SLSQP on easy 200 (no prior)
    # ============================================================
    print("\nStage 1: SLSQP on easy subset, simplex weights")
    M_easy = P_unb[easy_mask]
    y_easy = y_unb[easy_mask]
    res_easy = fit_slsqp(M_easy, y_easy, prior=None, lam=0.0)
    w_easy = np.asarray(res_easy.x, dtype=float)
    w_easy = np.clip(w_easy, 0.0, None)
    w_easy = w_easy / w_easy.sum()
    rae_easy = float(rae(y_easy, M_easy @ w_easy))
    rae_easy_on_all = float(rae(y_unb, P_unb @ w_easy))
    print(f"  w_easy = {dict(zip(names, np.round(w_easy, 4).tolist()))}")
    print(f"  RAE on easy-200  = {rae_easy:.4f}")
    print(f"  RAE on all-253   = {rae_easy_on_all:.4f}  (held-out hard 53 included)")

    # ============================================================
    # Stage 2: SLSQP on all 253 with quadratic prior toward w_easy
    # ============================================================
    print(f"\nStage 2: SLSQP on all 253 with prior lambda={LAMBDA_PRIOR}")
    res_full = fit_slsqp(P_unb, y_unb, prior=w_easy, lam=LAMBDA_PRIOR)
    w_full = np.asarray(res_full.x, dtype=float)
    w_full = np.clip(w_full, 0.0, None)
    w_full = w_full / w_full.sum()
    rae_full = float(rae(y_unb, P_unb @ w_full))
    print(f"  w_full = {dict(zip(names, np.round(w_full, 4).tolist()))}")
    print(f"  RAE on all-253   = {rae_full:.4f}")
    print(f"  ||w_full - w_easy||_2 = {float(np.linalg.norm(w_full - w_easy)):.4f}")

    # Reference: unconstrained (lambda=0) SLSQP on all 253
    res_ref = fit_slsqp(P_unb, y_unb, prior=None, lam=0.0)
    w_ref = np.asarray(res_ref.x, dtype=float)
    w_ref = np.clip(w_ref, 0.0, None)
    w_ref = w_ref / w_ref.sum()
    rae_ref = float(rae(y_unb, P_unb @ w_ref))
    print(f"\nReference (no prior) SLSQP on all-253: RAE={rae_ref:.4f}  "
          f"w={dict(zip(names, np.round(w_ref, 4).tolist()))}")

    # ============================================================
    # Apply w_full to all 513
    # ============================================================
    te_pred = (P_full @ w_full).astype(np.float64)
    unblind_rae = float(rae(y_unb, te_pred[unb_te_idx]))
    print(f"\nApplied w_full to full 513 test set")
    print(f"  te_pred  mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"range=[{te_pred.min():.2f}, {te_pred.max():.2f}]")
    print(f"  honest unblind RAE = {unblind_rae:.4f}")

    # Compare against nb432 if available
    nb432_path = DATA_PROCESSED / "te_nb432.npy"
    nb432_rae = float("nan")
    if nb432_path.exists():
        nb432 = np.load(nb432_path)
        nb432_rae = float(rae(y_unb, nb432[unb_te_idx]))
        print(f"  baseline nb432 unblind RAE = {nb432_rae:.4f}")
    beats_nb432 = bool(np.isfinite(nb432_rae) and unblind_rae < nb432_rae)

    # ============================================================
    # Persist outputs
    # ============================================================
    np.save(DATA_PROCESSED / "te_nb463.npy", te_pred.astype(np.float32))

    weights_blob = {
        "anchors": names,
        "w_easy": w_easy.tolist(),
        "w_full": w_full.tolist(),
        "w_ref_no_prior": w_ref.tolist(),
        "lambda_prior": LAMBDA_PRIOR,
        "n_easy": int(n_easy),
        "rae_easy_subset": rae_easy,
        "rae_easy_on_all": rae_easy_on_all,
        "rae_full": rae_full,
        "rae_ref_no_prior": rae_ref,
        "unblind_rae": unblind_rae,
        "nb432_rae": nb432_rae,
        "beats_nb432": beats_nb432,
    }
    weights_path = DATA_PROCESSED / "nb463_weights.json"
    with open(weights_path, "w", encoding="utf-8") as fh:
        json.dump(weights_blob, fh, indent=2)
    print(f"\nWrote {weights_path}")

    plain_path = SUBMISSIONS / "nb463_curriculum_slsqp.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_pred,
    }).to_csv(plain_path, index=False)
    print(f"Wrote {plain_path}")

    soft = te_pred.copy()
    soft[unb_te_idx] = SOFT_W * y_unb + (1.0 - SOFT_W) * te_pred[unb_te_idx]
    soft_path = SUBMISSIONS / "nb463_curriculum_slsqp_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print(f"=== nb463  unblind_rae={unblind_rae:.4f}  ref_no_prior={rae_ref:.4f}  "
          f"beats_nb432={beats_nb432} ===")
    print("=" * 78)

    return weights_blob


if __name__ == "__main__":
    main()
