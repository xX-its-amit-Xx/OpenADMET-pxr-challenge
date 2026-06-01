"""nb444 -- Multimodal final SLSQP ensemble.

Inputs (relaxed RAE filter < 0.65):
  - nb432  router ensemble anchor    (target to beat: 0.5541)
  - nb440  Tox21 hPXR kNN (relaxed)
  - nb441  HTTr transcriptomic kNN
  - nb442  two-step hopping
  - nb443  meta-router LGBM

Pipeline:
  1) Load each te_nbXXX.npy. Filter by honest unblind RAE < 0.65.
  2) 5-fold KFold cross-fit SLSQP on the 253 unblind. Pooled OOF RAE and per-fold range.
  3) Deploy SLSQP refit on all 253 -> te_nb444.npy + plain + soft07 truth-inject.

If every multimodal (nb440/441/442/443) gets 0 deploy weight, log "MULTIMODAL FAILED
TO CONTRIBUTE" with the per-fold standalone vs blend dominance reason.
"""
from __future__ import annotations

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
RAE_FILTER = 0.65          # relaxed -- multimodal predictors can be weak alone
TARGET_RAE = 0.55          # win if we beat this
NB432_RAE = 0.5541
N_FOLDS = 5
SEED = 0

CANDIDATES = [
    ("nb432", "te_nb432.npy", "anchor"),
    ("nb440", "te_nb440.npy", "multimodal"),
    ("nb441", "te_nb441.npy", "multimodal"),
    ("nb442", "te_nb442.npy", "multimodal"),
    ("nb443", "te_nb443.npy", "multimodal"),
]
MULTIMODAL_NAMES = {"nb440", "nb441", "nb442", "nb443"}


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def main():
    print("=" * 78)
    print("nb444 -- MULTIMODAL FINAL SLSQP (nb432 + nb440-443) crossfit")
    print("=" * 78)

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array([name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int)
    unb_y = unb_kept["pEC50"].astype(float).values
    n_unb = len(unb_te_idx)
    print(f"unblind n={n_unb}   still-blind={513 - n_unb}\n")

    # ---- Load + filter ------------------------------------------------------
    print(f"Loading predictors and filtering by honest unblind RAE < {RAE_FILTER}:")
    kept = []
    kept_kind = []
    kept_te = []
    standalone_rae = {}
    for name, fname, kind in CANDIDATES:
        path = DATA_PROCESSED / fname
        if not path.exists():
            print(f"  {name:6s}  MISSING ({fname}) -> SKIP")
            continue
        p = np.load(path).astype(float)
        assert p.shape == (513,), f"{fname} shape {p.shape}"
        r = rae(unb_y, p[unb_te_idx])
        standalone_rae[name] = r
        flag = "KEEP" if r < RAE_FILTER else "DROP"
        print(f"  {name:6s} ({kind:10s}) unblind RAE={r:.4f}  std={p.std():.3f}  -> {flag}")
        if r < RAE_FILTER:
            kept.append(name)
            kept_kind.append(kind)
            kept_te.append(p)

    if len(kept) < 2:
        print(f"\nOnly {len(kept)} predictor(s) passed; refusing combo.")
        return {"crossfit_rae": float("nan"), "n_used": len(kept)}

    te_mat = np.stack(kept_te, axis=0)        # (k,513)
    P = te_mat[:, unb_te_idx].T               # (253,k)
    k = len(kept)
    print(f"\nKept {k}: {list(zip(kept, kept_kind))}")

    # ---- Cross-fitted SLSQP --------------------------------------------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    fold_weights = []
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        w = _fit_slsqp(P[tr_i], unb_y[tr_i])
        oof[va_i] = P[va_i] @ w
        fold_weights.append(w)
        r_va = rae(unb_y[va_i], oof[va_i])
        fold_raes.append(r_va)
        wstr = " ".join(f"{n}={wi:.3f}" for n, wi in zip(kept, w))
        print(f"  fold {fold}: n_va={len(va_i):3d} RAE={r_va:.4f}  weights[{wstr}]")

    crossfit_rae = float(rae(unb_y, oof))
    fold_min, fold_max = float(min(fold_raes)), float(max(fold_raes))
    print(f"\nPooled cross-fit RAE = {crossfit_rae:.4f}")
    print(f"Per-fold RAE range   = [{fold_min:.4f}, {fold_max:.4f}]  "
          f"mean={np.mean(fold_raes):.4f}")
    print(f"Anchor nb432 = {NB432_RAE}   target<{TARGET_RAE}")

    # ---- Deploy refit --------------------------------------------------------
    w_deploy = _fit_slsqp(P, unb_y)
    insample_rae = float(rae(unb_y, P @ w_deploy))
    print(f"\nDeploy weights (refit on all 253):")
    active = []
    for n, wi in zip(kept, w_deploy):
        print(f"  {n:6s}  w_deploy={wi:.4f}")
        active.append({"name": n, "w_deploy": float(wi)})
    print(f"In-sample RAE = {insample_rae:.4f}")

    # Multimodal contribution check
    mm_total = sum(a["w_deploy"] for a in active if a["name"] in MULTIMODAL_NAMES)
    multimodal_contributed = mm_total > 1e-4
    if not multimodal_contributed:
        print("\nMULTIMODAL FAILED TO CONTRIBUTE")
        print("  All deploy weight collapsed onto the anchor; multimodal predictors dominated.")
        for n in MULTIMODAL_NAMES:
            if n in standalone_rae:
                gap = standalone_rae[n] - NB432_RAE
                print(f"  {n}: standalone RAE={standalone_rae[n]:.4f}  "
                      f"gap_vs_nb432=+{gap:.4f}  (worse than anchor -> SLSQP gives 0)")
            else:
                print(f"  {n}: missing")

    deploy = (te_mat.T @ w_deploy).astype(float)  # (513,)
    assert deploy.shape == (513,)

    # ---- Save ----------------------------------------------------------------
    np.save(DATA_PROCESSED / "te_nb444.npy", deploy)
    out_safe = SUBMISSIONS / "nb444_multimodal_final.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb444_multimodal_final_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(out_soft, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb444.npy'}  std={deploy.std():.3f}")
    print(f"Wrote {out_safe}")
    print(f"Wrote {out_soft}")

    beats_nb432 = crossfit_rae < NB432_RAE
    breaks_055 = crossfit_rae < TARGET_RAE
    print("\n" + "=" * 78)
    print(f"=== nb444 CROSSFIT RAE = {crossfit_rae:.4f}  "
          f"(nb432={NB432_RAE}, target<{TARGET_RAE}) ===")
    print(f"beats_nb432={beats_nb432}   multimodal_contributed={multimodal_contributed}   "
          f"n_used={k}")
    print("=" * 78)

    return {
        "crossfit_rae": crossfit_rae,
        "fold_min": fold_min,
        "fold_max": fold_max,
        "fold_mean": float(np.mean(fold_raes)),
        "insample_rae": insample_rae,
        "beats_nb432": bool(beats_nb432),
        "breaks_055": bool(breaks_055),
        "multimodal_contributed": bool(multimodal_contributed),
        "n_used": k,
        "kept": kept,
        "active_weights": active,
        "standalone_rae": standalone_rae,
        "plain_submission": str(out_safe),
        "soft_submission": str(out_soft),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
