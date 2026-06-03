"""nb1133 -- Residual GB across 3 alternative anchors.

Hypothesis
----------
nb1123 ran a shallow LGBM Huber on the residual of nb1070 (a heavily
post-processed stretch median-bag) and found the residual signal was
near-noise (most pred-power already absorbed by the upstream stretch).
The alternative anchors here are LESS post-processed -- in particular
chemprop_aux is a raw MPNN with no per-quantile stretch on top, so
its residual may retain MORE extractable chemistry-feature signal that
a shallow LGBM can recover.

Anchors tested
--------------
  1. chemprop_aux  -- raw multi-head MPNN (PRE-unblind, in_RAE 0.6216
                       on 253, predicted LB ~0.6246).
  2. nb1001        -- single-seed nb988 blend + scalar stretch
                       (per nb1001_pred_oof.npy honest cross-fit).
  3. nb1014        -- 5-seed bag (median across nb988 seeds).

Procedure (per anchor)
----------------------
  - anchor_oof on 253 = te_<anchor>.npy[unb_idx]  (PRE-unblind anchors
    only; chemprop_aux is verified PRE-unblind, nb1001/nb1014 may have
    POST-unblind components -- we report in_RAE on 253 as the anchor
    floor and let the residual cross-fit be honest by construction).
  - residual = y_unb - anchor_oof  (signed).
  - features = combined Morgan+RDKit (2265) on the 253 unblind SMILES.
  - shallow LGBM Huber (max_depth=3, n_est=80, lr=0.05, alpha=1.0),
    honest 5-fold KFold cross-fit -> residual_oof.
  - pred_corrected = anchor_oof + residual_oof; pooled cross-fit RAE.

Outputs
-------
  data/processed/nb1133_<anchor>_residual_oof.npy
  data/processed/nb1133_<anchor>_pred_oof.npy
  data/processed/nb1133_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1133"

# (anchor_label, te_filename)
ANCHORS = [
    ("chemprop_aux", "te_chemprop_aux.npy"),
    ("nb1001",       "te_nb1001.npy"),
    ("nb1014",       "te_nb1014.npy"),
]

RESID_SEED = 42
RESID_FOLDS = 5
LGBM_PARAMS = dict(
    objective="huber",
    alpha=1.0,
    learning_rate=0.05,
    n_estimators=80,
    max_depth=3,
    num_leaves=7,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    verbosity=-1,
    random_state=RESID_SEED,
    n_jobs=2,
)

NB1123_REF_DELTA = -0.0000   # nb1123 was near-neutral on nb1070 anchor


def _crossfit_residual(X: np.ndarray, residual: np.ndarray,
                       anchor_oof: np.ndarray, y_unb: np.ndarray,
                       label: str) -> tuple[np.ndarray, list]:
    n = len(y_unb)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=RESID_SEED)
    residual_oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        mdl = LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X[tr_loc], residual[tr_loc])
        r_pred = mdl.predict(X[va_loc])
        residual_oof[va_loc] = r_pred
        resid_va_true = residual[va_loc]
        denom = max(((resid_va_true - resid_va_true.mean()) ** 2).sum(), 1e-12)
        r2 = 1.0 - (((resid_va_true - r_pred) ** 2).sum() / denom)
        sp_fold = float(spearmanr(resid_va_true, r_pred).correlation)
        rae_base = float(rae(y_unb[va_loc], anchor_oof[va_loc]))
        rae_corr = float(rae(y_unb[va_loc], anchor_oof[va_loc] + r_pred))
        fold_records.append({
            "fold": k,
            "n_va": int(len(va_loc)),
            "rae_base": rae_base,
            "rae_corr": rae_corr,
            "delta": rae_corr - rae_base,
            "resid_r2_on_va": float(r2),
            "spearman_resid_pred_vs_true_va": sp_fold,
            "pred_resid_mean": float(r_pred.mean()),
            "pred_resid_std": float(r_pred.std()),
        })
        print(f"   [{label}] fold {k}: n_va={len(va_loc):3d}  "
              f"base={rae_base:.4f}  corr={rae_corr:.4f}  "
              f"(d={rae_corr - rae_base:+.4f})  "
              f"R2={r2:+.3f}  rho={sp_fold:+.3f}  "
              f"|pred|std={r_pred.std():.3f}")
    return residual_oof, fold_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- shallow LGBM Huber on residual across 3 anchors")
    print(f"          features = combined Morgan+RDKit (2265)")
    print(f"          LGBM: max_depth=3, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print(f"          anchors = {[a[0] for a in ANCHORS]}")
    print("=" * 78)

    # ---- Load 513 test + unblind idx/truth ----
    te = load_test()
    te_smiles = te["smiles"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unb_idx = {unb_idx.shape}, y_unb = {y_unb.shape}")

    # ---- Build features once (shared across anchors) ----
    smi_unb = te_smiles[unb_idx].tolist()
    print(f"[feat] computing combined(Morgan+RDKit) on n={len(smi_unb)} "
          f"unblind SMILES")
    X_unb = impute(combined(smi_unb))
    print(f"[feat] X_unb shape = {X_unb.shape}")

    per_anchor = []
    for label, fname in ANCHORS:
        print("\n" + "-" * 78)
        print(f"ANCHOR = {label}  (te file: {fname})")
        print("-" * 78)
        te_pred = np.load(DATA_PROCESSED / fname).astype(np.float64)
        if te_pred.shape[0] != 513:
            print(f"   [skip] {fname} shape {te_pred.shape} != 513")
            per_anchor.append({"anchor": label, "skipped": True})
            continue
        anchor_oof = te_pred[unb_idx].astype(np.float64)
        rae_anchor = float(rae(y_unb, anchor_oof))
        residual = y_unb - anchor_oof
        print(f"   [base] in-sample RAE(anchor[unb_idx], truth) = "
              f"{rae_anchor:.4f}")
        print(f"   [resid] mean={residual.mean():+.4f}  "
              f"std={residual.std():.4f}  "
              f"min={residual.min():+.4f}  max={residual.max():+.4f}")

        residual_oof, fold_records = _crossfit_residual(
            X_unb, residual, anchor_oof, y_unb, label)

        pred_corrected_oof = anchor_oof + residual_oof
        pooled_rae_base = float(rae(y_unb, anchor_oof))
        pooled_rae_corr = float(rae(y_unb, pred_corrected_oof))
        delta = pooled_rae_corr - pooled_rae_base
        rho_resid = float(spearmanr(residual_oof, residual).correlation)
        rho_corr = float(spearmanr(pred_corrected_oof, y_unb).correlation)
        rho_base = float(spearmanr(anchor_oof, y_unb).correlation)

        print(f"   [pooled] base={pooled_rae_base:.4f}  "
              f"corrected={pooled_rae_corr:.4f}  delta={delta:+.4f}  "
              f"rho_resid={rho_resid:+.3f}")

        if delta <= -0.003:
            verdict = "RESIDUAL_HELPS"
        elif abs(delta) < 0.003:
            verdict = "RESIDUAL_NEUTRAL"
        else:
            verdict = "RESIDUAL_HURTS_NOISE_DOMINATES"

        np.save(DATA_PROCESSED / f"{TAG}_{label}_residual_oof.npy",
                residual_oof.astype(np.float32))
        np.save(DATA_PROCESSED / f"{TAG}_{label}_pred_oof.npy",
                pred_corrected_oof.astype(np.float32))

        per_anchor.append({
            "anchor": label,
            "te_file": fname,
            "pooled_rae_base": pooled_rae_base,
            "pooled_rae_corrected": pooled_rae_corr,
            "delta": delta,
            "verdict": verdict,
            "spearman_resid_oof_vs_resid_truth": rho_resid,
            "spearman_pred_corrected_vs_truth": rho_corr,
            "spearman_anchor_vs_truth": rho_base,
            "residual_oof_std": float(residual_oof.std()),
            "residual_truth_std": float(residual.std()),
            "fold_records": fold_records,
        })

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("ALL ANCHORS POOLED CROSS-FIT SUMMARY")
    print("=" * 78)
    valid = [r for r in per_anchor if not r.get("skipped")]
    if valid:
        best = min(valid, key=lambda r: r["pooled_rae_corrected"])
    else:
        best = None
    print(f"{'anchor':<14s} {'base':>7s} {'corrected':>10s} "
          f"{'delta':>8s} {'rho_r':>7s} verdict")
    for r in per_anchor:
        if r.get("skipped"):
            print(f"{r['anchor']:<14s}  (skipped)")
            continue
        print(f"{r['anchor']:<14s} {r['pooled_rae_base']:>7.4f} "
              f"{r['pooled_rae_corrected']:>10.4f} {r['delta']:>+8.4f} "
              f"{r['spearman_resid_oof_vs_resid_truth']:>+7.3f} "
              f"{r['verdict']}")
    if best is not None:
        print(f"\n[best] {best['anchor']}  corrected pooled RAE = "
              f"{best['pooled_rae_corrected']:.4f}  "
              f"(delta {best['delta']:+.4f})")

    nb1123_corr = None
    nb1123_summary = DATA_PROCESSED / "nb1123_summary.json"
    if nb1123_summary.exists():
        try:
            with open(nb1123_summary) as f:
                nb1123_corr = float(json.load(f).get("pooled_rae_corrected"))
        except Exception:
            nb1123_corr = None
    beats_nb1123 = (best is not None and nb1123_corr is not None
                    and best["pooled_rae_corrected"]
                    < nb1123_corr - 0.003)

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "feature_dim": int(X_unb.shape[1]),
        "lgbm_params": LGBM_PARAMS,
        "resid_folds": RESID_FOLDS,
        "resid_seed": RESID_SEED,
        "anchors": per_anchor,
        "best_anchor": (best["anchor"] if best is not None else None),
        "best_pooled_rae_corrected": (best["pooled_rae_corrected"]
                                      if best is not None else None),
        "nb1123_pooled_rae_corrected": nb1123_corr,
        "beats_nb1123": bool(beats_nb1123),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  best_anchor: {res.get('best_anchor')}")
    print(f"  best_pooled_rae_corrected: "
          f"{res.get('best_pooled_rae_corrected')}")
    print(f"  nb1123_pooled_rae_corrected: "
          f"{res.get('nb1123_pooled_rae_corrected')}")
    print(f"  beats_nb1123: {res.get('beats_nb1123')}")
    for r in res.get("anchors", []):
        if r.get("skipped"):
            continue
        print(f"  - {r['anchor']:<14s} base={r['pooled_rae_base']:.4f} "
              f"corr={r['pooled_rae_corrected']:.4f} "
              f"delta={r['delta']:+.4f} verdict={r['verdict']}")
