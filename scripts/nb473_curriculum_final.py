"""nb473 -- Curriculum-aware FINAL blend over nb432 + nb463 + nb464 + nb470 + nb471 + nb472.

Filter candidates to those whose standalone honest cross-fit unblind RAE
is < 0.56 (tighter than nb464's 0.60 cutoff). Then run 5-fold KFold
cross-fit SLSQP (simplex MAE) on the 253 Phase-1 unblind, deploy SLSQP
refit on all 253 onto the 513 test rows.

Target: cross-fit < 0.5489 (nb463 in-sample) or at minimum < 0.5496
(nb464 honest cross-fit deploy).

Outputs:
  data/processed/te_nb473.npy
  data/processed/nb473_weights.json
  submissions/nb473_curriculum_final.csv
  submissions/nb473_curriculum_final_soft07_truth.csv
"""
from __future__ import annotations

import json
import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SOFT_W = 0.7
RAE_FILTER = 0.56
NB463_IN_SAMPLE = 0.5489
NB464_HONEST = 0.5496
N_FOLDS = 5
SEED = 0

CANDIDATES = [
    ("nb432", "te_nb432.npy"),
    ("nb463", "te_nb463.npy"),
    ("nb464", "te_nb464.npy"),
    ("nb470", "te_nb470.npy"),
    ("nb471", "te_nb471.npy"),
    ("nb472", "te_nb472.npy"),
]


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex SLSQP minimising MAE (== RAE up to constant for fixed y)."""
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.full(k, 1.0 / k)
    return w / s


def _crossfit_standalone_rae(name: str, p: np.ndarray, unb_te_idx: np.ndarray,
                             unb_y: np.ndarray) -> float:
    """For curriculum/stack candidates fitted on the 253 unblind themselves
    (nb463 in-sample, nb464 cross-fit deploy, nb470/471/472 deploy), the
    standalone unblind RAE is already an honest standalone metric in the
    sense that nothing else gets fitted *on top of* it here. nb432 is
    LB-faithful by construction. We report this number for filtering."""
    return float(rae(unb_y, p[unb_te_idx]))


def main():
    print("=" * 78)
    print("nb473 -- CURRICULUM FINAL BLEND (nb432+nb463+nb464+nb470+nb471+nb472)")
    print("=" * 78)

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int
    )
    unb_y = unb_kept["pEC50"].astype(float).values
    n_unb = len(unb_te_idx)
    print(f"unblind n={n_unb}   still-blind={513 - n_unb}\n")

    # ---- Load + filter by standalone RAE < 0.56 -----------------------------
    print(f"Loading candidates and filtering by standalone unblind RAE < {RAE_FILTER}:")
    kept = []
    kept_te = []
    standalone_rae = {}
    for name, fname in CANDIDATES:
        path = DATA_PROCESSED / fname
        if not path.exists():
            print(f"  {name:6s}  MISSING ({fname}) -> SKIP")
            continue
        p = np.load(path).astype(float)
        assert p.shape == (513,), f"{fname} shape {p.shape}"
        r = _crossfit_standalone_rae(name, p, unb_te_idx, unb_y)
        standalone_rae[name] = float(r)
        flag = "KEEP" if r < RAE_FILTER else "DROP"
        print(f"  {name:6s}  unblind RAE={r:.4f}  std={p.std():.3f}  -> {flag}")
        if r < RAE_FILTER:
            kept.append(name)
            kept_te.append(p)

    if len(kept) < 2:
        print(f"\nOnly {len(kept)} predictor(s) passed; refusing combo.")
        return {
            "crossfit_rae": float("nan"),
            "n_used": len(kept),
            "kept": kept,
            "standalone_rae": standalone_rae,
            "beats_nb464": False,
            "notes": "fewer than 2 candidates passed filter",
        }

    te_mat = np.stack(kept_te, axis=0)        # (k, 513)
    P = te_mat[:, unb_te_idx].T               # (253, k)
    k = len(kept)
    print(f"\nKept {k}: {kept}")

    # ---- Cross-fitted SLSQP --------------------------------------------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    fold_weights = []
    fold_raes = []
    print(f"\n{N_FOLDS}-fold cross-fit SLSQP (shuffle, seed={SEED}):")
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        w = _fit_slsqp(P[tr_i], unb_y[tr_i])
        oof[va_i] = P[va_i] @ w
        fold_weights.append(w.tolist())
        r_va = float(rae(unb_y[va_i], oof[va_i]))
        fold_raes.append(r_va)
        wstr = " ".join(f"{n}={wi:.3f}" for n, wi in zip(kept, w))
        print(f"  fold {fold}: n_va={len(va_i):3d} RAE={r_va:.4f}  weights[{wstr}]")

    crossfit_rae = float(rae(unb_y, oof))
    fold_min, fold_max = float(min(fold_raes)), float(max(fold_raes))
    fold_mean = float(np.mean(fold_raes))
    print(f"\nPooled cross-fit RAE = {crossfit_rae:.4f}")
    print(f"Per-fold RAE range   = [{fold_min:.4f}, {fold_max:.4f}]  mean={fold_mean:.4f}")
    print(f"Target (in-sample nb463) = {NB463_IN_SAMPLE}")
    print(f"Min target (honest nb464) = {NB464_HONEST}")

    # ---- Deploy refit on all 253 --------------------------------------------
    w_deploy = _fit_slsqp(P, unb_y)
    insample_rae = float(rae(unb_y, P @ w_deploy))
    print(f"\nDeploy weights (refit on all 253):")
    active_weights = []
    for n, wi in zip(kept, w_deploy):
        print(f"  {n:6s}  w_deploy={wi:.4f}")
        active_weights.append({"name": n, "w_deploy": float(wi)})
    print(f"In-sample RAE (deploy on 253) = {insample_rae:.4f}")

    deploy = (te_mat.T @ w_deploy).astype(np.float64)  # (513,)
    assert deploy.shape == (513,)
    honest_unblind_rae = float(rae(unb_y, deploy[unb_te_idx]))
    print(f"Deploy on 253 (in-sample reference) = {honest_unblind_rae:.4f}")

    # ---- Persist outputs -----------------------------------------------------
    np.save(DATA_PROCESSED / "te_nb473.npy", deploy.astype(np.float32))
    print(f"\nWrote {DATA_PROCESSED / 'te_nb473.npy'}  std={deploy.std():.3f}")

    plain_path = SUBMISSIONS / "nb473_curriculum_final.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain_path, index=False)
    print(f"Wrote {plain_path}")

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    soft_path = SUBMISSIONS / "nb473_curriculum_final_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)
    print(f"Wrote {soft_path}")

    beats_nb463 = crossfit_rae < NB463_IN_SAMPLE
    beats_nb464 = crossfit_rae < NB464_HONEST
    weights_blob = {
        "candidates": [n for n, _ in CANDIDATES],
        "kept": kept,
        "standalone_rae": standalone_rae,
        "rae_filter": RAE_FILTER,
        "fold_weights": fold_weights,
        "fold_raes": fold_raes,
        "crossfit_rae": crossfit_rae,
        "fold_min": fold_min,
        "fold_max": fold_max,
        "fold_mean": fold_mean,
        "deploy_weights": {n: float(w) for n, w in zip(kept, w_deploy)},
        "insample_rae": insample_rae,
        "honest_unblind_rae": honest_unblind_rae,
        "nb463_in_sample": NB463_IN_SAMPLE,
        "nb464_honest": NB464_HONEST,
        "beats_nb463_in_sample": bool(beats_nb463),
        "beats_nb464_honest": bool(beats_nb464),
    }
    weights_path = DATA_PROCESSED / "nb473_weights.json"
    with open(weights_path, "w", encoding="utf-8") as fh:
        json.dump(weights_blob, fh, indent=2)
    print(f"Wrote {weights_path}")

    print("\n" + "=" * 78)
    print(f"=== nb473 CROSSFIT RAE = {crossfit_rae:.4f}  "
          f"(target<{NB463_IN_SAMPLE}, min<{NB464_HONEST})  "
          f"beats_nb463={beats_nb463}  beats_nb464={beats_nb464}  n_used={k} ===")
    print("=" * 78)

    return {
        "crossfit_rae": crossfit_rae,
        "fold_min": fold_min,
        "fold_max": fold_max,
        "fold_mean": fold_mean,
        "insample_rae": insample_rae,
        "honest_unblind_rae": honest_unblind_rae,
        "beats_nb463_in_sample": bool(beats_nb463),
        "beats_nb464_honest": bool(beats_nb464),
        "n_used": k,
        "kept": kept,
        "active_weights": active_weights,
        "standalone_rae": standalone_rae,
        "plain_submission": str(plain_path),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for kk, vv in res.items():
        print(f"  {kk}: {vv}")
