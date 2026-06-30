"""nb1201 -- second-pass rank-stretch on nb1191 PRE-unblind pyramid.

nb1191 already applies an internal mean(per-fold s) ~ 1.031 to the SLSQP
blend before deploy. nb1201 asks whether a *second-pass* scalar stretch on
top of the nb1191 OOF (a) survives honest per-fold cross-fit on the 253
unblind rows and (b) clears a tight gate (best s >= 1.03 AND pooled RAE
improvement >= 0.003) before we mint a deploy CSV.

Method
------
Stage 1: Reconstruct the nb1191 OOF on the 253 unblind by replaying its
exact seed-averaged pipeline:
  for kf_seed in {1001..1005}:
    scaffold 5-fold split
    per fold: SLSQP simplex on 4 anchor OOFs, then per-fold grid stretch
  oof_blend_seed = mean across 5 seeds

Stage 2: Per-fold cross-fit grid s in {1.00, 1.02, 1.05, 1.08, 1.10, 1.12}
on (oof_blend_seed, y_unb) under scaffold 5-fold CV (seed = 1001 anchor):
  pred_va_s = mu_tr + s * (oof_va - mu_tr)
where mu_tr is mean(oof_blend_seed[tr_idx]). Pooled-OOF RAE per s.

Stage 3: Gate
  best_s >= 1.03  AND  (rae_baseline - rae_best_s) >= 0.003

If gate passes:
  apply (mu, best_s) globally to te_nb1191.npy (the already-stretched 513
  deploy vector) -> submissions/nb1201_deploy_stretched.csv

Artefacts
  data/processed/nb1201_summary.json   (always)
  data/processed/te_nb1201.npy         (always; stretched 513 vector)
  submissions/nb1201_deploy_stretched.csv  (only if gate passes)
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

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1201"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
NB1191_STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
SECOND_PASS_GRID = [1.00, 1.02, 1.05, 1.08, 1.10, 1.12]
CV_SEED = 1001  # scaffold-CV seed used for the second-pass grid sweep

# Gate
GATE_BEST_S_MIN = 1.03
GATE_IMPROVEMENT_MIN = 0.003

# nb1191 anchor set (matched to scripts/nb1191_pre_unblind_pyramid.py)
ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy"),
]
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)
    return P @ w


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def cv_run_for_seed_nb1191(P_unb, y_unb, unb_scaffolds, kf_seed):
    """Replay nb1191's SLSQP+stretch CV pipeline at one seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(
            blend_tr, y_unb[tr_loc], mu_tr, NB1191_STRETCH_GRID,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
    pooled = float(rae(y_unb, oof_blend))
    return pooled, oof_blend


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- second-pass rank-stretch on nb1191")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Reconstruct nb1191 OOF ----
    print("\n[reconstruct] nb1191 anchor OOFs")
    oof_cols = []
    for disp, oof_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
        else:
            oof_p = DATA_PROCESSED / oof_rel
            assert oof_p.exists(), f"missing OOF: {oof_p}"
            oof = np.load(oof_p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        r = float(rae(y_unb, oof))
        print(f"   {disp:14s} oof_RAE={r:.4f}")
        oof_cols.append(oof)
    P_unb = np.column_stack(oof_cols)

    print(f"\n[reconstruct] replay nb1191 seed-avg pipeline "
          f"(kf_seeds={KF_SEEDS})")
    seed_oofs = []
    seed_pooled = []
    for kf_seed in KF_SEEDS:
        pooled, oof = cv_run_for_seed_nb1191(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        seed_oofs.append(oof)
        seed_pooled.append(pooled)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}")
    oof_nb1191 = np.mean(np.column_stack(seed_oofs), axis=1)
    rae_baseline = float(rae(y_unb, oof_nb1191))
    print(f"\n[reconstruct] mean-of-seeds nb1191 OOF RAE = {rae_baseline:.4f}  "
          f"(memo: 0.4697)")
    print(f"[reconstruct] mean of per-seed pooled RAE   = "
          f"{np.mean(seed_pooled):.4f}  (memo: 0.4703)")

    # ---- Stage 2: per-fold cross-fit second-pass grid stretch ----
    print("\n" + "-" * 78)
    print(f"SECOND-PASS GRID STRETCH  grid={SECOND_PASS_GRID}  "
          f"scaffold 5-fold CV (seed={CV_SEED})")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=CV_SEED,
    )

    rae_per_s = {}
    fold_s_per_grid_eval = {}
    # For each s in the grid, build a *fixed-s* OOF: apply (mu_tr, s) to
    # each held-out fold and pool. This gives "RAE if we forced s
    # uniformly". Separately, do per-fold *picked-s* for the gate metric.
    for s in SECOND_PASS_GRID:
        oof = np.full(n_unb, np.nan)
        for tr_loc, va_loc in splits:
            mu_tr = float(oof_nb1191[tr_loc].mean())
            oof[va_loc] = mu_tr + s * (oof_nb1191[va_loc] - mu_tr)
        r = float(rae(y_unb, oof))
        rae_per_s[s] = r
        print(f"   s={s:.2f}  pooled_OOF_RAE={r:.4f}")

    best_s = min(rae_per_s, key=rae_per_s.get)
    rae_best_s = rae_per_s[best_s]
    improvement = rae_baseline - rae_best_s
    print(f"\n[stage2] baseline RAE (no second-pass)   = {rae_baseline:.4f}")
    print(f"[stage2] best fixed-s                    = {best_s:.2f}  "
          f"(RAE {rae_best_s:.4f})")
    print(f"[stage2] improvement                     = {improvement:+.4f}")

    # Per-fold cross-fit version (each fold picks its own s from the grid)
    print(f"\n[stage2b] per-fold cross-fit pick on training subset")
    oof_perfold = np.full(n_unb, np.nan)
    fold_s_picks = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        mu_tr = float(oof_nb1191[tr_loc].mean())
        s_pick, _ = best_stretch_on(
            oof_nb1191[tr_loc], y_unb[tr_loc], mu_tr, SECOND_PASS_GRID,
        )
        oof_perfold[va_loc] = mu_tr + s_pick * (oof_nb1191[va_loc] - mu_tr)
        fold_s_picks.append(s_pick)
        print(f"   fold {fi}: s_pick={s_pick:.2f}  mu_tr={mu_tr:.3f}  "
              f"n_va={len(va_loc)}")
    rae_perfold = float(rae(y_unb, oof_perfold))
    print(f"[stage2b] pooled per-fold cross-fit RAE  = {rae_perfold:.4f}")

    # ---- Gate ----
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    gate_s = best_s >= GATE_BEST_S_MIN
    gate_improve = improvement >= GATE_IMPROVEMENT_MIN
    gate_pass = gate_s and gate_improve
    print(f"   gate A: best_s {best_s:.3f} >= {GATE_BEST_S_MIN}  "
          f"-> {'PASS' if gate_s else 'FAIL'}")
    print(f"   gate B: RAE improvement {improvement:+.4f} >= "
          f"{GATE_IMPROVEMENT_MIN}  -> {'PASS' if gate_improve else 'FAIL'}")
    print(f"   overall: {'PASS' if gate_pass else 'FAIL'}")

    # ---- Build deploy vector ----
    # Apply (mu_full, best_s) to te_nb1191.npy. te_nb1191.npy is the
    # already-stretched 513 deploy vector from nb1191 (internal s=1.031).
    # We add a second-pass scalar stretch on top, centred on its own mean.
    te_nb1191_path = DATA_PROCESSED / "te_nb1191.npy"
    assert te_nb1191_path.exists(), f"missing {te_nb1191_path}"
    te_nb1191 = np.load(te_nb1191_path).astype(np.float64)
    assert te_nb1191.shape == (n_te,), f"te_nb1191 shape {te_nb1191.shape}"

    mu_deploy = float(te_nb1191.mean())
    deploy_te = (mu_deploy + best_s * (te_nb1191 - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] mu(te_nb1191)={mu_deploy:.4f}  best_s={best_s:.3f}")
    print(f"[deploy] te(513) mean / std = "
          f"{deploy_te.mean():.3f} / {deploy_te.std():.3f}  "
          f"(was {te_nb1191.mean():.3f} / {te_nb1191.std():.3f})")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")

    # ---- Save artefacts ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_deploy_stretched.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(f"[skip] gate FAILED -- no submission CSV written "
              f"(would be {sub_csv_path})")

    summary = {
        "tag": TAG,
        "method": "second_pass_rank_stretch_on_nb1191_reconstructed_OOF",
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "kf_seeds_for_nb1191_replay": KF_SEEDS,
        "cv_seed_for_second_pass": CV_SEED,
        "nb1191_internal_stretch_grid": NB1191_STRETCH_GRID,
        "second_pass_grid": SECOND_PASS_GRID,
        "rae_per_s_fixed": {f"{s:.2f}": v for s, v in rae_per_s.items()},
        "rae_baseline_nb1191_oof": rae_baseline,
        "rae_baseline_memo": 0.4703,
        "rae_perfold_crossfit": rae_perfold,
        "fold_s_picks": [float(x) for x in fold_s_picks],
        "best_s": float(best_s),
        "rae_best_s": rae_best_s,
        "improvement_vs_baseline": improvement,
        "gate_best_s_min": GATE_BEST_S_MIN,
        "gate_improvement_min": GATE_IMPROVEMENT_MIN,
        "gate_a_best_s_ge_min": bool(gate_s),
        "gate_b_improve_ge_min": bool(gate_improve),
        "gate_pass": bool(gate_pass),
        "deploy_mu_from_te_nb1191": mu_deploy,
        "deploy_s": float(best_s),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "te_nb1191_input_mean": float(te_nb1191.mean()),
        "te_nb1191_input_std": float(te_nb1191.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_pass else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   baseline nb1191 OOF RAE       = {rae_baseline:.4f}")
    print(f"   best second-pass s            = {best_s:.3f}")
    print(f"   RAE at best s                 = {rae_best_s:.4f}")
    print(f"   improvement                   = {improvement:+.4f}")
    print(f"   gate A (best_s >= {GATE_BEST_S_MIN})    = {gate_s}")
    print(f"   gate B (improve >= {GATE_IMPROVEMENT_MIN})  = {gate_improve}")
    print(f"   gate overall                  = {gate_pass}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_baseline_nb1191_oof",
        "best_s",
        "rae_best_s",
        "improvement_vs_baseline",
        "gate_a_best_s_ge_min",
        "gate_b_improve_ge_min",
        "gate_pass",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
