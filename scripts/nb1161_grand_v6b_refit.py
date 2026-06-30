"""nb1161 -- Genuine scaffold-CV grand_v6b SLSQP refit.

Plan (cycle 144):
  grand_v6b_calib.csv currently sits at LB 0.6234 but its random-CV OOF RAE is
  0.5250 -- a ~+0.12 random-CV inflation. Refit the SLSQP convex blend over the
  same 12-component pool that fed grand_v6b_calib (per nb392.PREDICTORS minus
  grand_v6b_calib itself) but under scaffold_kfold_indices(seed=42) so the blend
  weights are honest under analog-expansion test shift.

Steps:
  1. Load 12 component (oof, te) pairs aligned to 4139 train / 513 test.
  2. Sha256 the scaffold splits + each oof tail so we know the anchors match
     what nb2103 / nb562 / nb730 use.
  3. Re-run SLSQP simplex blend under BOTH random KFold (baseline that produced
     the inflated v6b weights) and scaffold KFold (honest). Cross-fit per fold,
     then refit on all 4139 for deploy weights.
  4. Compare new scaffold weights vs the random baseline. Compute L1 weight
     change %.
  5. Gates:
       (a) scaffold-CV pooled RAE <= 0.5027 (5% below grand_v6b_calib OOF 0.5250)
       (b) weights changed >= 5% L1 from random-CV version
       (c) no single component carries > 80% mass
  6. If ALL gates pass: build deploy CSV at
     submissions/nb1161_deploy_grand_v6b_scaffold.csv (test predictions =
     T @ w_scaffold) AND save te_nb1161 / oof_nb1161 arrays for ladder use.
  7. Save data/processed/nb1161_summary.json with the full audit trail.

Outputs:
  scripts/nb1161_grand_v6b_refit.py        (this file)
  data/processed/nb1161_summary.json
  data/processed/te_nb1161.npy             (only if gate passes)
  data/processed/oof_nb1161.npy            (only if gate passes)
  submissions/nb1161_deploy_grand_v6b_scaffold.csv  (only if gate passes)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from time import time

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1161"
SEED = 42
N_FOLDS = 5

# Base-model pool that fed grand_v6b_calib (per nb392 PREDICTORS minus the
# meta-stacks-of-stacks nb239/nb224/nb179 and grand_v6b_calib itself; those
# meta-stacks include 4139-row OOFs that are already-blended and dominate any
# downstream SLSQP, which masks the random-vs-scaffold inflation we want to
# audit). Keeping only base learners gives the refit room to actually move.
POOL = [
    "nb93_chemprop_large_gpu",
    "nb130_external_pxr",
    "nb264_chemprop_mt",
    "nb303_dann",
    "chemprop_aux",
    "nb305_mope",
    "nb306_cepsmim",
    "catboost",
    "deep_ensemble",
]

# Gates
RAE_GATE = 0.5027            # 5% below grand_v6b_calib OOF (0.5250)
WEIGHT_DELTA_GATE = 0.05     # L1/2 >= 5% change
MAX_SINGLE_WEIGHT = 0.80     # no single component > 80%

# Reference numbers (legacy)
GRAND_V6B_CALIB_OOF = 0.5250


# ---------------------------------------------------------------------------
def sha12(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def load_pair(stem: str, n_tr: int, n_te: int):
    """Try oof_/te_ then plain stems; return (oof, te) or (None, None)."""
    for op in ("oof_", ""):
        for tp in ("te_", "te_oof_"):
            of = DATA_PROCESSED / f"{op}{stem}.npy"
            tf = DATA_PROCESSED / f"{tp}{stem}.npy"
            if of.exists() and tf.exists():
                try:
                    a = np.load(of).astype(np.float64)
                    b = np.load(tf).astype(np.float64)
                    if a.ndim == 2: a = a[:, 0]
                    if b.ndim == 2: b = b[:, 0]
                    if a.shape == (n_tr,) and b.shape == (n_te,):
                        return a, b
                except Exception:
                    continue
    return None, None


def fit_slsqp_mae(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    if k == 1:
        return np.array([1.0])
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 500, "disp": False})
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return np.full(k, 1.0 / k) if s <= 0 else w / s


def crossfit_splits(splits, P: np.ndarray, y: np.ndarray, label: str):
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_raes, fold_w = [], []
    for f, (tr_i, va_i) in enumerate(splits):
        w = fit_slsqp_mae(P[tr_i], y[tr_i])
        oof[va_i] = P[va_i] @ w
        r = float(rae(y[va_i], oof[va_i]))
        fold_raes.append(r)
        fold_w.append(w)
    pooled = float(rae(y, oof))
    return oof, pooled, fold_raes, fold_w


def main() -> dict:
    t0 = time()
    print("=" * 78)
    print(f"{TAG} -- scaffold-CV grand_v6b SLSQP refit")
    print("=" * 78)

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(tr); n_te = len(te)
    print(f"train={n_tr}  test={n_te}")

    # ---- splits ----
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    scaff_splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)
    rand_splits = list(KFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=SEED).split(np.arange(n_tr)))

    print("\nScaffold split anchors:")
    scaf_anchors = []
    for f, (tridx, vaidx) in enumerate(scaff_splits):
        a = {"fold": f, "n_tr": int(len(tridx)), "n_va": int(len(vaidx)),
             "tr_sha": sha12(tridx), "va_sha": sha12(vaidx)}
        scaf_anchors.append(a)
        print(f"  fold {f}: n_va={len(vaidx)}  tr_sha={a['tr_sha']} "
              f"va_sha={a['va_sha']}")

    # ---- load components ----
    print(f"\nLoading {len(POOL)} components:")
    oofs, tes, names, comp_meta = [], [], [], []
    for stem in POOL:
        oof, te_p = load_pair(stem, n_tr, n_te)
        if oof is None:
            print(f"  MISS  {stem}")
            comp_meta.append({"stem": stem, "loaded": False})
            continue
        oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
        te_p = np.where(np.isfinite(te_p), te_p, np.nanmean(te_p))
        r_alone = float(rae(y_tr, oof))
        oof_sha = sha12(oof.astype(np.float32))
        te_sha = sha12(te_p.astype(np.float32))
        ratio = float(te_p.std() / oof.std()) if oof.std() > 0 else 0.0
        print(f"  OK    {stem:30s} RAE={r_alone:.4f}  ratio={ratio:.2f}  "
              f"oof_sha={oof_sha}  te_sha={te_sha}")
        oofs.append(oof); tes.append(te_p); names.append(stem)
        comp_meta.append({
            "stem": stem, "loaded": True,
            "rae_alone_train": r_alone,
            "te_oof_std_ratio": ratio,
            "oof_sha": oof_sha, "te_sha": te_sha,
        })

    if len(oofs) < 2:
        print("ERROR: too few components loaded; abort")
        return {"success": False, "n_loaded": len(oofs)}

    P = np.column_stack(oofs)
    T = np.column_stack(tes)
    print(f"\nstacked: P={P.shape}  T={T.shape}")

    # ---- baseline grand_v6b_calib ----
    g_oof, g_te = load_pair("grand_v6b_calib", n_tr, n_te)
    if g_oof is None:
        print("WARN: grand_v6b_calib OOF/TE not found; baseline disabled")
        baseline_rae = float("nan")
    else:
        baseline_rae = float(rae(y_tr, g_oof))
        print(f"\nbaseline grand_v6b_calib OOF RAE = {baseline_rae:.4f}  "
              f"(reference {GRAND_V6B_CALIB_OOF:.4f})")

    # ---- random KFold SLSQP (proxy for original v6b inflated weights) ----
    # Deploy weight = MEAN of per-fold weights (cross-fit average), NOT a
    # refit on all 4139. This reproduces how the original grand_v6b_calib
    # weights were chosen on random splits.
    print("\n--- random KFold SLSQP cross-fit (baseline) ---")
    _, rand_pooled, rand_raes, rand_w_list = crossfit_splits(
        rand_splits, P, y_tr, "random KFold")
    w_rand = np.mean(np.stack(rand_w_list, axis=0), axis=0)
    w_rand = w_rand / w_rand.sum() if w_rand.sum() > 0 else w_rand
    for n, w in zip(names, w_rand):
        print(f"  {n:30s} w_rand={w:.4f}")
    print(f"random KFold pooled RAE = {rand_pooled:.4f}  per-fold "
          f"{min(rand_raes):.4f}..{max(rand_raes):.4f}")

    # ---- scaffold KFold SLSQP (HONEST) ----
    print("\n--- scaffold KFold SLSQP cross-fit (honest) ---")
    oof_scaf, scaf_pooled, scaf_raes, scaf_w_list = crossfit_splits(
        scaff_splits, P, y_tr, "scaffold KFold")
    w_scaf = np.mean(np.stack(scaf_w_list, axis=0), axis=0)
    w_scaf = w_scaf / w_scaf.sum() if w_scaf.sum() > 0 else w_scaf
    for n, w in zip(names, w_scaf):
        print(f"  {n:30s} w_scaf={w:.4f}")
    print(f"scaffold KFold pooled RAE = {scaf_pooled:.4f}  per-fold "
          f"{min(scaf_raes):.4f}..{max(scaf_raes):.4f}")

    # ---- comparison ----
    weight_l1 = float(np.sum(np.abs(w_scaf - w_rand)))
    weight_l2 = float(np.sqrt(np.sum((w_scaf - w_rand) ** 2)))
    print(f"\nweight L1 = {weight_l1:.4f}   L2 = {weight_l2:.4f}")
    print("per-component (rand -> scaf, delta):")
    for n, wr, ws in zip(names, w_rand, w_scaf):
        print(f"  {n:30s} {wr:.4f} -> {ws:.4f}   d={ws - wr:+.4f}")
    max_w = float(w_scaf.max())

    # ---- gates ----
    rae_pass = scaf_pooled <= RAE_GATE
    delta_pass = weight_l1 >= WEIGHT_DELTA_GATE
    cap_pass = max_w < MAX_SINGLE_WEIGHT
    all_pass = bool(rae_pass and delta_pass and cap_pass)

    print("\n" + "=" * 78)
    print("GATES")
    print("=" * 78)
    print(f"  (a) scaffold RAE <= {RAE_GATE:.4f}    : "
          f"{scaf_pooled:.4f}  {'PASS' if rae_pass else 'FAIL'}")
    print(f"  (b) weight L1 >= {WEIGHT_DELTA_GATE:.2f}        : "
          f"{weight_l1:.4f}  {'PASS' if delta_pass else 'FAIL'}")
    print(f"  (c) max single weight < {MAX_SINGLE_WEIGHT:.2f}  : "
          f"{max_w:.4f}  {'PASS' if cap_pass else 'FAIL'}")
    print(f"  -> deploy = {all_pass}")

    deploy_csv = None
    te_deploy = T @ w_scaf
    deploy_clip = np.clip(te_deploy, y_tr.min() - 0.5, y_tr.max() + 0.5)
    if all_pass:
        deploy_csv = SUBMISSIONS / "nb1161_deploy_grand_v6b_scaffold.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": deploy_clip.astype(np.float64),
        }).to_csv(deploy_csv, index=False)
        np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_clip.astype(np.float32))
        np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof_scaf.astype(np.float32))
        print(f"\nWrote {deploy_csv}")
        print(f"Wrote te_{TAG}.npy, oof_{TAG}.npy")
    else:
        print("\nGates failed; no deploy CSV written.")

    summary = {
        "tag": TAG,
        "method": "scaffold_cv_slsqp_refit_of_grand_v6b_pool",
        "wall_sec": round(time() - t0, 2),
        "n_train": n_tr, "n_test": n_te,
        "pool": names,
        "pool_meta": comp_meta,
        "scaffold_split": {
            "seed": SEED, "n_folds": N_FOLDS,
            "anchors": scaf_anchors,
        },
        "baseline_grand_v6b_calib_oof_rae": baseline_rae,
        "random_kfold": {
            "pooled_rae": rand_pooled,
            "fold_raes": [float(x) for x in rand_raes],
            "deploy_weights": {n: float(w) for n, w in zip(names, w_rand)},
        },
        "scaffold_kfold": {
            "pooled_rae": scaf_pooled,
            "fold_raes": [float(x) for x in scaf_raes],
            "deploy_weights": {n: float(w) for n, w in zip(names, w_scaf)},
        },
        "weight_delta": {
            "l1": weight_l1, "l2": weight_l2,
            "per_component_delta": {n: float(ws - wr) for n, wr, ws in
                                    zip(names, w_rand, w_scaf)},
            "max_single_weight": max_w,
        },
        "gates": {
            "rae_pass": bool(rae_pass), "rae_gate": RAE_GATE,
            "delta_pass": bool(delta_pass), "delta_gate": WEIGHT_DELTA_GATE,
            "cap_pass": bool(cap_pass), "cap_gate": MAX_SINGLE_WEIGHT,
            "all_pass": all_pass,
        },
        "deploy": {
            "built": all_pass,
            "csv": str(deploy_csv) if deploy_csv else None,
            "te_path": str(DATA_PROCESSED / f"te_{TAG}.npy") if all_pass else None,
            "oof_path": str(DATA_PROCESSED / f"oof_{TAG}.npy") if all_pass else None,
        },
    }
    out = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(json.dumps({k: v for k, v in res.items() if k in
                      ("tag", "baseline_grand_v6b_calib_oof_rae",
                       "random_kfold", "scaffold_kfold", "weight_delta",
                       "gates", "deploy")}, indent=2))
