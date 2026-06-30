"""nb971 -- Stacked SLSQP over 6 honest PRE-unblind anchors (cycle 131 plan #5).

GOAL
    Beat nb2112 honest cross-fit RAE 0.4698 by SLSQP-blending 6 anchors with
    weights fit via 5-fold cross-fit on the 253 unblind labels.

CONTAMINATION POLICY (per memory feedback_te_vs_pred_oof_protocol +
                      feedback_anchor_contamination_chain)
    For each anchor we MUST use predictions on 253 from a model that was NEVER
    trained on those 253 labels.  Two sources:
      (a) `nbXXX_pred_oof.npy` produced by 5-fold cross-fit on the 253 (honest).
      (b) `te_X.npy[unb_idx]` IF the underlying model was trained on 4139 train
          only (PRE-unblind).  This is automatically true when te_mtime is
          BEFORE y_unb_mtime + a small buffer.

    Anchors whose te file post-dates y_unb mtime (e.g. nb2112 te emitted by a
    deploy refit on 4139+253) cannot use te[unb_idx] -- that path is in-sample
    truth-injection.  For those we MUST source from an honest cross-fit OOF.

ANCHOR -> HONEST 253-PREDICTION RESOLUTION
    a1 chemprop_aux  te_chemprop_aux.npy[unb_idx]                         (PRE-unb te, in_RAE=0.6216)
    a2 nb503         te_nb503.npy[unb_idx]                                (PRE-unb te, in_RAE=0.4249) -- low because nb503 is itself a curriculum+residual stack trained on 4139; te[unb_idx] = honest holdout
    a3 nb464         te_nb464.npy[unb_idx]                                (PRE-unb te, in_RAE=0.5489)
    a4 nb471         te_nb471.npy[unb_idx]                                (PRE-unb te, in_RAE=0.5506)
    a5 nb432         te_nb432.npy[unb_idx]                                (PRE-unb te, in_RAE=0.5519)
    a6 nb2112-honest nb2103_mean_bag_oof_K28.npy                          (HONEST cross-fit OOF, RAE=0.4737;
                                                                          nb2112 te[unb_idx]=0.1006 is contaminated -- DO NOT USE)

PROTOCOL
    1. Verify each anchor's PRE-unblind/honest-OOF source and emit contamination
       receipts (te_mtime vs y_mtime; sha8; pearson to truth).
    2. SLSQP solve over weights w (>=0, sum=1) MINIMIZING:
       a) MAE  (mean abs error)
       b) HUBER (delta=0.5) loss
       Both done via:
        - in-sample on full 253 (one-shot baseline)
        - 5-fold cross-fit (fit on 4 folds, predict on 1) -- this is the
          unblind-honest figure that gates deploy.
    3. Deploy only if 5-fold cross-fit RAE beats nb2112 floor by >= 0.005 mRAE.
       Deploy CSV uses 513-row anchors with in-sample weights -- BUT for the
       nb2112-honest column we use the deploy te_nb2112.npy for 513-row
       prediction (because deploy IS allowed to refit on all 4139+253 labels --
       only the BLEND WEIGHTS must be cross-fit on the 253).

OUTPUTS
    data/processed/nb971_summary.json
    submissions/nb971_deploy_slsqp6.csv      (only if cross-fit RAE < 0.4648)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SUB = ROOT / "submissions"
RAW = ROOT / "data" / "raw"

sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

TAG = "nb971"
SEED = 0
N_FOLDS = 5

NB2112_FLOOR = 0.4698          # current honest floor to beat
IMPROVEMENT_DELTA = 0.005      # cycle policy: deploy only if >= 5 mRAE better
HUBER_DELTA = 0.5


# ---------- 1. Anchor registry --------------------------------------------
# `unb_source`:
#   "te_unb_idx" -> use te[unb_idx] (only safe if te was emitted PRE-unblind)
#   path string  -> use a saved honest 5-fold cross-fit OOF file
ANCHORS = [
    {"name": "chemprop_aux", "te_path": "data/processed/te_chemprop_aux.npy", "unb_source": "te_unb_idx",                             "reported_rae": 0.6216},
    # nb503 has BOTH a PRE-unblind te (te_nb503[unb_idx]=0.4249) and a saved 5-fold
    # cross-fit OOF (nb503_pred_oof.npy=0.5116).  The OOF is the conservative,
    # protocol-faithful choice (LB-honest 0.5116 per ladder snapshot nb964).
    {"name": "nb503",        "te_path": "data/processed/te_nb503.npy",        "unb_source": "data/processed/nb503_pred_oof.npy",       "reported_rae": 0.5116},
    {"name": "nb464",        "te_path": "data/processed/te_nb464.npy",        "unb_source": "te_unb_idx",                             "reported_rae": 0.5489},
    {"name": "nb471",        "te_path": "data/processed/te_nb471.npy",        "unb_source": "te_unb_idx",                             "reported_rae": 0.5506},
    {"name": "nb432",        "te_path": "data/processed/te_nb432.npy",        "unb_source": "te_unb_idx",                             "reported_rae": 0.5519},
    {"name": "nb2112",       "te_path": "data/processed/te_nb2112.npy",       "unb_source": "data/processed/nb2103_mean_bag_oof_K28.npy", "reported_rae": 0.4737},
]


# ---------- helpers --------------------------------------------------------
def rae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / np.abs(y - y.mean()).sum())


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).mean())


def huber(y: np.ndarray, p: np.ndarray, delta: float = HUBER_DELTA) -> float:
    r = np.abs(y - p)
    quad = np.minimum(r, delta)
    lin = r - quad
    return float((0.5 * quad ** 2 + delta * lin).mean())


def slsqp_blend(X: np.ndarray, y: np.ndarray, loss_fn: Callable, w0: np.ndarray | None = None) -> np.ndarray:
    n_anchor = X.shape[1]
    if w0 is None:
        w0 = np.ones(n_anchor) / n_anchor
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n_anchor

    def objective(w):
        return loss_fn(y, X @ w)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"ftol": 1e-9, "maxiter": 500})
    w = res.x
    w = np.clip(w, 0.0, None)
    if w.sum() > 0:
        w = w / w.sum()
    return w


def cross_fit_blend(X: np.ndarray, y: np.ndarray, loss_fn: Callable, n_folds: int = N_FOLDS, seed: int = SEED):
    """5-fold cross-fit: fit SLSQP weights on 4 folds, predict on the 1 held-out fold."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred_oof = np.zeros_like(y)
    fold_weights = []
    for tr, va in kf.split(X):
        w = slsqp_blend(X[tr], y[tr], loss_fn)
        pred_oof[va] = X[va] @ w
        fold_weights.append(w)
    fold_weights = np.asarray(fold_weights)
    return pred_oof, fold_weights


def sha_first_n(arr: np.ndarray, n: int = 8) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:n]


# ---------- 2. Load + verify ------------------------------------------------
def main() -> None:
    y_unb = np.load(PROC / "_audit_unblind_y.npy")
    unb_idx = np.load(PROC / "_audit_unblind_idx.npy")
    y_unb_mtime = (PROC / "_audit_unblind_y.npy").stat().st_mtime
    print(f"y_unb: n={len(y_unb)} mean={y_unb.mean():.3f} std={y_unb.std():.3f}")
    print(f"y_unb_mtime={y_unb_mtime:.0f}")

    anchor_meta = []
    cols_unb, cols_513 = [], []
    for a in ANCHORS:
        te = np.load(ROOT / a["te_path"])
        assert te.shape == (513,), f"{a['name']} bad shape {te.shape}"
        te_mtime = (ROOT / a["te_path"]).stat().st_mtime

        # Resolve honest unblind prediction
        if a["unb_source"] == "te_unb_idx":
            # PRE-unblind te file -- te[unb_idx] IS the honest holdout
            pred_unb = te[unb_idx]
            unb_src_path = a["te_path"] + "[unb_idx]"
            unb_src_mtime = te_mtime
        else:
            # Honest 5-fold cross-fit OOF (separate file)
            oof = np.load(ROOT / a["unb_source"])
            assert oof.shape == (253,), f"{a['name']} OOF bad shape {oof.shape}"
            pred_unb = oof
            unb_src_path = a["unb_source"]
            unb_src_mtime = (ROOT / a["unb_source"]).stat().st_mtime

        a_rae = rae(y_unb, pred_unb)
        pre_unblind_te = te_mtime < y_unb_mtime + 86400
        sha = sha_first_n(pred_unb)
        pearson = float(np.corrcoef(pred_unb, y_unb)[0, 1]) if pred_unb.std() > 0 else 0.0
        truth_inj = pearson > 0.995
        anchor_meta.append({
            "name": a["name"],
            "te_path": a["te_path"],
            "te_mtime": te_mtime,
            "pre_unblind_te_by_mtime": pre_unblind_te,
            "unb_source": unb_src_path,
            "unb_source_mtime": unb_src_mtime,
            "honest_rae_on_unb": round(a_rae, 4),
            "reported_rae": a["reported_rae"],
            "pred_unb_mean": float(pred_unb.mean()),
            "pred_unb_std": float(pred_unb.std()),
            "pred_unb_sha8": sha,
            "pearson_to_y_unb": round(pearson, 4),
            "truth_injection_flag": truth_inj,
        })
        assert not truth_inj, f"{a['name']} flagged as truth-injection (pearson={pearson:.3f}) -- contamination"
        cols_unb.append(pred_unb)
        cols_513.append(te)
        print(f"  {a['name']:<14} unb_src={unb_src_path:<54}  honest_RAE={a_rae:.4f}  pearson={pearson:.3f}")

    X_unb = np.column_stack(cols_unb)        # (253, 6)
    X_513 = np.column_stack(cols_513)        # (513, 6)
    print(f"X_unb shape={X_unb.shape}, X_513 shape={X_513.shape}")

    # ---------- 3. SLSQP variants ------------------------------------------
    results = {}
    names = [a["name"] for a in ANCHORS]

    # 3a) In-sample SLSQP (one shot)
    for loss_name, loss_fn in [("mae", mae), ("huber", huber)]:
        w_in = slsqp_blend(X_unb, y_unb, loss_fn)
        pred_in = X_unb @ w_in
        rae_in = rae(y_unb, pred_in)
        results[f"slsqp_{loss_name}_insample"] = {
            "weights": w_in.tolist(),
            "rae_insample": round(rae_in, 4),
        }
        print(f"\n[insample {loss_name}] RAE={rae_in:.4f} weights={dict(zip(names, np.round(w_in, 3).tolist()))}")

    # 3b) 5-fold cross-fit SLSQP
    for loss_name, loss_fn in [("mae", mae), ("huber", huber)]:
        pred_cf, fold_w = cross_fit_blend(X_unb, y_unb, loss_fn)
        rae_cf = rae(y_unb, pred_cf)
        mean_w = fold_w.mean(axis=0)
        results[f"slsqp_{loss_name}_crossfit"] = {
            "mean_fold_weights": mean_w.tolist(),
            "fold_weights": fold_w.tolist(),
            "rae_crossfit": round(rae_cf, 4),
        }
        print(f"[crossfit {loss_name}] RAE={rae_cf:.4f} mean_weights={dict(zip(names, np.round(mean_w, 3).tolist()))}")

    # ---------- 4. Per-anchor baselines ------------------------------------
    per_anchor_rae = {a["name"]: round(rae(y_unb, X_unb[:, i]), 4) for i, a in enumerate(ANCHORS)}
    print(f"\nper-anchor honest_RAE on 253: {per_anchor_rae}")

    # ---------- 5. Decide deploy -------------------------------------------
    cf_mae = results["slsqp_mae_crossfit"]["rae_crossfit"]
    cf_huber = results["slsqp_huber_crossfit"]["rae_crossfit"]
    best_cf = min(cf_mae, cf_huber)
    best_loss = "mae" if cf_mae <= cf_huber else "huber"

    target_rae = NB2112_FLOOR - IMPROVEMENT_DELTA
    beats_floor = best_cf <= target_rae
    deploy_csv_path = None

    if beats_floor:
        w_deploy = np.asarray(results[f"slsqp_{best_loss}_insample"]["weights"])
        pred_513 = X_513 @ w_deploy
        pred_513 = np.clip(pred_513, 3.0, 11.0)
        test_blind = pd.read_csv(RAW / "pxr-challenge_TEST_BLINDED.csv")
        smiles_col = next((c for c in test_blind.columns if c.lower() == "smiles"), "SMILES")
        name_col = next((c for c in test_blind.columns if "name" in c.lower()), "Molecule Name")
        assert len(test_blind) == 513
        out = pd.DataFrame({
            "SMILES": test_blind[smiles_col].values,
            "Molecule Name": test_blind[name_col].values,
            "pEC50": pred_513,
        })
        deploy_csv_path = SUB / "nb971_deploy_slsqp6.csv"
        out.to_csv(deploy_csv_path, index=False)
        print(f"\n[DEPLOY] wrote {deploy_csv_path}  best_cf_RAE={best_cf:.4f} (loss={best_loss})")
    else:
        print(f"\n[NO DEPLOY] best_cf_RAE={best_cf:.4f} (loss={best_loss}) does NOT beat target {target_rae:.4f}  (floor {NB2112_FLOOR} - {IMPROVEMENT_DELTA})")

    # ---------- 6. Save summary --------------------------------------------
    summary = {
        "tag": TAG,
        "cycle": 131,
        "plan_idx": 5,
        "method": "stacked_slsqp_6anchor",
        "n_anchors": len(ANCHORS),
        "y_unb_n": int(len(y_unb)),
        "y_unb_mean": float(y_unb.mean()),
        "y_unb_std": float(y_unb.std()),
        "y_unb_mtime": y_unb_mtime,
        "anchor_meta": anchor_meta,
        "per_anchor_honest_rae": per_anchor_rae,
        "slsqp_results": results,
        "floor": {"id": "nb2112_chemprop_aux_K28_median_bag", "rae": NB2112_FLOOR},
        "improvement_delta_required": IMPROVEMENT_DELTA,
        "best_crossfit": {"loss": best_loss, "rae": best_cf, "beats_floor": beats_floor, "target_rae": target_rae},
        "deploy_csv": str(deploy_csv_path) if deploy_csv_path else None,
    }
    out_json = PROC / "nb971_summary.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
