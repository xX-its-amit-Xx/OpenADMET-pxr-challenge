"""nb1160 - Heterogeneous boosting family sweep (ExtraTrees + GBR + HistGB + RF).

Per cycle 142/143/144 findings, all single-paradigm methods fail under
scaffold-CV. The hypothesis here is that paradigm diversity across
4 sklearn ensembles (extra-randomized trees, gradient boosting, histogram-
binned GB, random forest) provides residual orthogonality that SLSQP can
exploit.

Protocol:
  1. Build combined Morgan + RDKit features (2265 cols) for train (4139)
     and test (513), median-imputed.
  2. Scaffold 5-fold CV (src.pxr.eval.scaffold_kfold_indices).
  3. Train 4 sklearn ensembles per fold, collect OOF + per-fold test preds.
  4. SLSQP-blend OOFs with simplex constraint (sum=1, w>=0).
  5. Gate: scaffold-CV RAE <= 0.5027 AND no single family weight 100% AND
     each family contributes >= 10%.
  6. If gate passes, build deploy CSV with mean test preds blended per
     SLSQP weights.

Outputs:
  data/processed/nb1160_per_family_oof.npy          (4, 4139) float32
  data/processed/nb1160_per_family_te.npy           (4, 513)  float32
  data/processed/nb1160_blend_oof.npy               (4139,)   float32
  data/processed/nb1160_blend_te.npy                (513,)    float32
  data/processed/nb1160_summary.json
  submissions/nb1160_hetero_boost.csv               (if gate passes)
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
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1160"
N_FOLDS = 5
SEED = 42
GATE_RAE = 0.5027
MIN_FAMILY_WEIGHT = 0.10  # each family must contribute >= 10%
MAX_FAMILY_WEIGHT = 0.999  # no single family allowed to be 100%

FAMILIES = ["et", "gbr", "histgb", "rf"]


def make_models(seed: int) -> dict:
    """Return one fresh instance per family."""
    return {
        "et": ExtraTreesRegressor(
            n_estimators=500,
            n_jobs=-1,
            random_state=seed,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=seed,
        ),
        "histgb": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=8,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        ),
        "rf": RandomForestRegressor(
            n_estimators=1000,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def slsqp_simplex_blend(
    oofs: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """SLSQP minimisation of RAE(sum w_i * oofs_i, y) under simplex constraint."""
    k = oofs.shape[0]

    def loss(w):
        blend = (w[:, None] * oofs).sum(axis=0)
        return rae(y, blend)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = [(0.0, 1.0)] * k
    w0 = np.ones(k) / k
    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-8, "maxiter": 400},
    )
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum() if w.sum() > 0 else np.ones(k) / k
    return w, float(loss(w))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} - Heterogeneous boosting family sweep "
          f"(ExtraTrees + GBR + HistGB + RF)")
    print(f"        gate: scaffold-CV RAE <= {GATE_RAE:.4f}  "
          f"each family >= {MIN_FAMILY_WEIGHT:.0%}  no single = 100%")
    print("=" * 78)

    # -------- Load data + scaffold splits --------
    tr = load_train()
    te = load_test()
    y = tr["pec50"].values.astype(np.float64)
    n_tr, n_te = len(y), len(te)
    print(f"[load] n_train = {n_tr}  n_test = {n_te}")

    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, seed=SEED)
    print(f"[split] scaffold {N_FOLDS}-fold  -> fold sizes: "
          f"{[len(v) for _, v in splits]}")

    # -------- Features --------
    print("[feat] computing combined Morgan + RDKit features ...")
    X_tr = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    X_te = impute(combined(te["smiles"].tolist())).astype(np.float32)
    print(f"[feat] X_tr {X_tr.shape}  X_te {X_te.shape}")

    # -------- Per-family OOF + test --------
    oofs = np.zeros((len(FAMILIES), n_tr), dtype=np.float64)
    tes_fold = np.zeros((len(FAMILIES), N_FOLDS, n_te), dtype=np.float64)

    for fi, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n--- fold {fi + 1}/{N_FOLDS}  "
              f"n_tr={len(tr_idx)}  n_va={len(va_idx)} ---")
        models = make_models(SEED + fi)
        for ki, fam in enumerate(FAMILIES):
            tt0 = time.time()
            mdl = models[fam]
            mdl.fit(X_tr[tr_idx], y[tr_idx])
            oofs[ki, va_idx] = mdl.predict(X_tr[va_idx])
            tes_fold[ki, fi] = mdl.predict(X_te)
            fold_rae = rae(y[va_idx], oofs[ki, va_idx])
            print(f"   {fam:6s}  fold_RAE = {fold_rae:.4f}  "
                  f"(t = {time.time() - tt0:.1f}s)")

    # Per-family aggregated OOF + mean test
    print("\n" + "-" * 78)
    print("PER-FAMILY POOLED SCAFFOLD-CV RAE")
    print("-" * 78)
    per_family_rae: dict[str, float] = {}
    for ki, fam in enumerate(FAMILIES):
        r = float(rae(y, oofs[ki]))
        per_family_rae[fam] = r
        print(f"   {fam:6s}  pooled RAE = {r:.4f}")
    tes_mean = tes_fold.mean(axis=1)  # (4, n_te)

    # -------- SLSQP blend --------
    print("\n" + "-" * 78)
    print("SLSQP SIMPLEX BLEND (sum=1, w>=0)")
    print("-" * 78)
    weights, blend_rae = slsqp_simplex_blend(oofs, y)
    weight_map = {fam: float(weights[i]) for i, fam in enumerate(FAMILIES)}
    print(f"   weights: " + "  ".join(
        f"{fam}={weight_map[fam]:.3f}" for fam in FAMILIES))
    print(f"   blend pooled RAE = {blend_rae:.4f}")

    blend_oof = (weights[:, None] * oofs).sum(axis=0)
    blend_te = (weights[:, None] * tes_mean).sum(axis=0)

    # -------- Gate evaluation --------
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    g_rae_pass = blend_rae <= GATE_RAE
    g_min_pass = all(weight_map[f] >= MIN_FAMILY_WEIGHT for f in FAMILIES)
    g_max_pass = max(weight_map.values()) <= MAX_FAMILY_WEIGHT
    print(f"   gate.rae        {blend_rae:.4f} <= {GATE_RAE:.4f} "
          f"=> {'PASS' if g_rae_pass else 'FAIL'}")
    print(f"   gate.min_w      min_w = {min(weight_map.values()):.3f} >= "
          f"{MIN_FAMILY_WEIGHT:.2f}  => {'PASS' if g_min_pass else 'FAIL'}")
    print(f"   gate.max_w      max_w = {max(weight_map.values()):.3f} <= "
          f"{MAX_FAMILY_WEIGHT:.3f}  => {'PASS' if g_max_pass else 'FAIL'}")
    gate_pass = bool(g_rae_pass and g_min_pass and g_max_pass)
    verdict = "GATE_PASS_DEPLOY" if gate_pass else "GATE_FAIL_NO_DEPLOY"
    print(f"   VERDICT: {verdict}")

    # -------- Save arrays --------
    np.save(DATA_PROCESSED / f"{TAG}_per_family_oof.npy",
            oofs.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_family_te.npy",
            tes_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_blend_oof.npy",
            blend_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_blend_te.npy",
            blend_te.astype(np.float32))
    print(f"\n[save] {TAG}_per_family_oof.npy  {oofs.shape}")
    print(f"[save] {TAG}_per_family_te.npy   {tes_mean.shape}")
    print(f"[save] {TAG}_blend_oof.npy       {blend_oof.shape}")
    print(f"[save] {TAG}_blend_te.npy        {blend_te.shape}")

    # -------- Deploy CSV (only if gate passes) --------
    sub_path = None
    if gate_pass:
        sub_path = SUBMISSIONS / f"{TAG}_hetero_boost.csv"
        sub_df = pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": blend_te,
        })
        sub_df.to_csv(sub_path, index=False)
        print(f"[save] {sub_path}  rows={len(sub_df)}")

    # -------- Summary --------
    summary = {
        "tag": TAG,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "families": FAMILIES,
        "per_family_rae": {k: float(v) for k, v in per_family_rae.items()},
        "weights": weight_map,
        "blend_scaffold_cv_rae": float(blend_rae),
        "gate_rae_threshold": GATE_RAE,
        "min_family_weight": MIN_FAMILY_WEIGHT,
        "max_family_weight": MAX_FAMILY_WEIGHT,
        "gate_rae_pass": bool(g_rae_pass),
        "gate_min_weight_pass": bool(g_min_pass),
        "gate_max_weight_pass": bool(g_max_pass),
        "gate_overall_pass": bool(gate_pass),
        "verdict": verdict,
        "submission_csv": str(sub_path) if sub_path else None,
        "wall_time_sec": round(time.time() - t0, 1),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {summary_path}")
    print(f"[done] wall = {summary['wall_time_sec']}s")
    return summary


if __name__ == "__main__":
    main()
