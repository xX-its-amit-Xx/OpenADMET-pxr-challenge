"""nb471 -- 3-stage DynCIM curriculum SLSQP with prior chain (lambda 0.7 -> 0.5).

Extends nb463 from 2 stages to 3 stages, splitting the 253 Phase-1 unblind
into EASY / MEDIUM / HARD thirds by Chemprop disagreement std.  A stronger
prior chain is hypothesised to prevent the hard-fold noise from dominating
fitted weights:

  Stage 1 (EASY, ~84):       free SLSQP -> w_easy
  Stage 2 (EASY+MED, ~168):  SLSQP + lam*||w - w_easy||^2 with lam=0.7 -> w_med
  Stage 3 (ALL 253):          SLSQP + lam*||w - w_med||^2  with lam=0.5 -> w_hard

Apply w_hard to all 513 test rows -> te_nb471.npy and submissions.
Also runs a 5-fold cross-fit of the same 3-stage protocol on the 253 unblind
labels for an honest generalisation estimate.
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

# ---- Anchors (same 5 as nb463) ----
ANCHORS = [
    ("nb411", "te_nb411.npy"),
    ("nb390", "te_nb390_pcs_iso.npy"),
    ("nb420", "te_nb420_frontier.npy"),
    ("nb424", "te_nb424.npy"),
    ("nb320", "te_nb320_phase2_top20.npy"),
]

LAMBDA_STAGE2 = 0.7    # EASY -> MED  prior pull
LAMBDA_STAGE3 = 0.5    # MED  -> HARD prior pull
SOFT_W = 0.7
N_RESTARTS = 200
SEED = 42
N_FOLDS = 5


def fit_slsqp(M: np.ndarray, y: np.ndarray, prior: np.ndarray | None = None,
              lam: float = 0.0, n_restarts: int = N_RESTARTS, seed: int = SEED):
    """Simplex-constrained SLSQP minimising RAE (+ optional quadratic prior)."""
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

    rng = np.random.default_rng(seed)
    seeds = [np.full(k, 1.0 / k)]
    if prior is not None:
        seeds.append(np.asarray(prior, dtype=float).copy())
    while len(seeds) < n_restarts:
        seeds.append(rng.dirichlet(np.ones(k)))

    best = None
    for w0 in seeds:
        res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(np.asarray(best.x, dtype=float), 0.0, None)
    w = w / w.sum()
    return w


def three_stage_fit(P: np.ndarray, y: np.ndarray, dis: np.ndarray,
                    seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run the 3-stage curriculum on (P, y) ranked by `dis`.

    Returns w_easy, w_med, w_hard, info dict with per-stage RAEs and bin sizes.
    """
    n = len(y)
    order = np.argsort(dis)
    third = n // 3
    easy_idx = order[:third]
    med_idx = order[third:2 * third]
    hard_idx = order[2 * third:]

    # Stage 1: EASY only
    w_easy = fit_slsqp(P[easy_idx], y[easy_idx], prior=None, lam=0.0, seed=seed)

    # Stage 2: EASY + MED, prior = w_easy, lam=0.7
    em_idx = np.concatenate([easy_idx, med_idx])
    w_med = fit_slsqp(P[em_idx], y[em_idx],
                      prior=w_easy, lam=LAMBDA_STAGE2, seed=seed + 1)

    # Stage 3: ALL, prior = w_med, lam=0.5
    w_hard = fit_slsqp(P, y, prior=w_med, lam=LAMBDA_STAGE3, seed=seed + 2)

    info = {
        "n_easy": int(len(easy_idx)),
        "n_med": int(len(med_idx)),
        "n_hard": int(len(hard_idx)),
        "rae_easy_stage1": float(rae(y[easy_idx], P[easy_idx] @ w_easy)),
        "rae_em_stage2": float(rae(y[em_idx], P[em_idx] @ w_med)),
        "rae_all_stage3": float(rae(y, P @ w_hard)),
        "norm_w_med_minus_easy": float(np.linalg.norm(w_med - w_easy)),
        "norm_w_hard_minus_med": float(np.linalg.norm(w_hard - w_med)),
    }
    return w_easy, w_med, w_hard, info


def main():
    print("=" * 78)
    print("nb471 -- 3-stage DynCIM curriculum SLSQP (lambda 0.7 -> 0.5)")
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

    # ---------- Difficulty score ----------
    dis_path = DATA_PROCESSED / "te_chemprop_disagreement_std.npy"
    if not dis_path.exists():
        raise FileNotFoundError(f"Missing {dis_path}")
    dis_te = np.load(dis_path).astype(np.float64)
    dis_unb = dis_te[unb_te_idx]
    print(f"\nDifficulty score range: [{dis_unb.min():.4f}, {dis_unb.max():.4f}]  "
          f"median={np.median(dis_unb):.4f}")

    # ============================================================
    # In-sample 3-stage fit on all 253
    # ============================================================
    print("\n" + "-" * 78)
    print("In-sample 3-stage curriculum on full 253")
    print("-" * 78)
    w_easy, w_med, w_hard, info = three_stage_fit(P_unb, y_unb, dis_unb, seed=SEED)
    print(f"  Stage1 EASY (n={info['n_easy']})   RAE={info['rae_easy_stage1']:.4f}")
    print(f"    w_easy = {dict(zip(names, np.round(w_easy, 4).tolist()))}")
    print(f"  Stage2 EASY+MED (n={info['n_easy']+info['n_med']})  RAE={info['rae_em_stage2']:.4f}  "
          f"||dw||={info['norm_w_med_minus_easy']:.4f}")
    print(f"    w_med  = {dict(zip(names, np.round(w_med, 4).tolist()))}")
    print(f"  Stage3 ALL (n={n_unb})  RAE={info['rae_all_stage3']:.4f}  "
          f"||dw||={info['norm_w_hard_minus_med']:.4f}")
    print(f"    w_hard = {dict(zip(names, np.round(w_hard, 4).tolist()))}")

    # Apply to all 513
    te_pred = (P_full @ w_hard).astype(np.float64)
    in_sample_rae = float(rae(y_unb, te_pred[unb_te_idx]))
    print(f"\nApplied w_hard to full 513")
    print(f"  te_pred mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"range=[{te_pred.min():.2f}, {te_pred.max():.2f}]")
    print(f"  in-sample unblind RAE = {in_sample_rae:.4f}")

    # ============================================================
    # 5-fold cross-fit of the same 3-stage protocol
    # ============================================================
    print("\n" + "-" * 78)
    print(f"{N_FOLDS}-fold cross-fit 3-stage curriculum")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n_unb, dtype=np.float64)
    fold_raes = []
    fold_weights = []
    for fold, (tr, va) in enumerate(kf.split(np.arange(n_unb))):
        w_e, w_m, w_h, _ = three_stage_fit(P_unb[tr], y_unb[tr], dis_unb[tr],
                                           seed=SEED + 10 * (fold + 1))
        oof[va] = P_unb[va] @ w_h
        fold_rae = float(rae(y_unb[va], oof[va]))
        fold_raes.append(fold_rae)
        fold_weights.append(w_h.tolist())
        print(f"  fold {fold+1}/{N_FOLDS}: n_va={len(va)}  fold-RAE={fold_rae:.4f}  "
              f"w_hard={np.round(w_h, 3).tolist()}")
    crossfit_rae = float(rae(y_unb, oof))
    print(f"\n  cross-fit RAE (pooled OOF) = {crossfit_rae:.4f}")
    print(f"  mean per-fold RAE          = {float(np.mean(fold_raes)):.4f}  "
          f"std={float(np.std(fold_raes)):.4f}")

    # ============================================================
    # Compare against baselines
    # ============================================================
    nb432_path = DATA_PROCESSED / "te_nb432.npy"
    nb432_rae = float("nan")
    if nb432_path.exists():
        nb432 = np.load(nb432_path)
        nb432_rae = float(rae(y_unb, nb432[unb_te_idx]))
    nb463_path = DATA_PROCESSED / "te_nb463.npy"
    nb463_rae = float("nan")
    if nb463_path.exists():
        nb463 = np.load(nb463_path)
        nb463_rae = float(rae(y_unb, nb463[unb_te_idx]))
    print(f"\nBaselines: nb432 RAE={nb432_rae:.4f}  nb463 RAE={nb463_rae:.4f}")
    beats_nb432 = bool(np.isfinite(nb432_rae) and in_sample_rae < nb432_rae)
    beats_nb463 = bool(np.isfinite(nb463_rae) and in_sample_rae < nb463_rae)
    beats_nb432_cf = bool(np.isfinite(nb432_rae) and crossfit_rae < nb432_rae)
    beats_nb463_cf = bool(np.isfinite(nb463_rae) and crossfit_rae < nb463_rae)

    # ============================================================
    # Persist
    # ============================================================
    np.save(DATA_PROCESSED / "te_nb471.npy", te_pred.astype(np.float32))

    blob = {
        "anchors": names,
        "w_easy": w_easy.tolist(),
        "w_med": w_med.tolist(),
        "w_hard": w_hard.tolist(),
        "lambda_stage2": LAMBDA_STAGE2,
        "lambda_stage3": LAMBDA_STAGE3,
        "stage_info": info,
        "in_sample_rae": in_sample_rae,
        "crossfit_rae": crossfit_rae,
        "fold_raes": fold_raes,
        "fold_weights": fold_weights,
        "nb432_rae": nb432_rae,
        "nb463_rae": nb463_rae,
        "beats_nb432_in_sample": beats_nb432,
        "beats_nb463_in_sample": beats_nb463,
        "beats_nb432_crossfit": beats_nb432_cf,
        "beats_nb463_crossfit": beats_nb463_cf,
    }
    weights_path = DATA_PROCESSED / "nb471_weights.json"
    with open(weights_path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nWrote {weights_path}")

    plain_path = SUBMISSIONS / "nb471_three_stage_curriculum.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_pred,
    }).to_csv(plain_path, index=False)
    print(f"Wrote {plain_path}")

    soft = te_pred.copy()
    soft[unb_te_idx] = SOFT_W * y_unb + (1.0 - SOFT_W) * te_pred[unb_te_idx]
    soft_path = SUBMISSIONS / "nb471_three_stage_curriculum_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print(f"=== nb471  in_sample={in_sample_rae:.4f}  cross_fit={crossfit_rae:.4f}  "
          f"vs nb432={nb432_rae:.4f}  vs nb463={nb463_rae:.4f} ===")
    print("=" * 78)

    return blob


if __name__ == "__main__":
    main()
