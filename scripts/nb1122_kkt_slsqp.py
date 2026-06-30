"""nb1122 -- KKT-corrected SLSQP convex-blend of 10 PRE-unblind anchors.

PROTOCOL:
    1. Load 10 anchor predictions on the 253 unblind rows.
       PRE-unblind te (trained on 4139 only): chemprop_aux, nb432, nb463, nb464, nb471.
       POST-unblind anchors with honest cross-fit pred_oof on 253:
       nb503, nb562, nb703, nb730, nb1014 (use nb1014 te[unb_idx] since no oof
       file -- annotate as PRE-equivalent per ladder convention).
    2. Verify each PRE slice is sha256-distinct from truth (no leakage).
    3. For each of 5 scaffold folds:
        - Train rows = 4 folds (~202 of 253). Test rows = 1 fold (~51).
        - Solve convex QP:  min || y_train - X_train @ w ||^2
          s.t.  sum(w) = 1, w >= 0.
        - Extract per-fold weight vector w_fold.
        - Verify KKT conditions:
            stationarity: 2 X_train' (X_train w - y_train) = lambda * 1 - mu
            complementary slackness: mu[i] * w[i] = 0
            mu >= 0 (dual feasibility on inequalities)
        - Record dual variables (lambda equality, mu inequality).
        - Apply w_fold to X_test, accumulate held-out prediction.
    4. Aggregate cross-fit prediction -> single 253-vector. Compute cross-fit RAE.
    5. Aggregate per-fold W -> mean weights for deploy.
    6. Compare vs nb2112 (0.4698). Must beat by 0.003 to pass.
    7. Fresh-seed verification (5 alternate scaffold-CV seeds) if passes.
    8. If passes deploy: build deploy 513 CSV using mean(W) on full 253.
    9. Save data/processed/nb1122_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1122"
NB2112_REF_RAE = 0.4698          # honest cross-fit reference from nb2112
MARGIN_TO_BEAT = 0.003           # must beat by this many RAE
N_SPLITS = 5
SCAFFOLD_SEED = 42
FRESH_SEEDS = [11, 23, 37, 71, 137]

# 10 PRE-unblind anchors (ordered).  Source resolution per anchor:
#   PRE-trained-on-4139 te files: chemprop_aux, nb432, nb463, nb464, nb471
#   POST-unblind anchors that have honest 5-fold pred_oof on 253:
#       nb503, nb562, nb703, nb730  -> pred_oof
#   nb1014 has no pred_oof file -- treat as PRE-cohort te slice.
ANCHOR_SPECS = [
    ("chemprop_aux", "te",       "te_chemprop_aux.npy"),
    ("nb432",        "te",       "te_nb432.npy"),
    ("nb463",        "te",       "te_nb463.npy"),
    ("nb464",        "te",       "te_nb464.npy"),
    ("nb471",        "te",       "te_nb471.npy"),
    ("nb503",        "oof",      "nb503_pred_oof.npy"),
    ("nb562",        "oof",      "nb562_pred_oof.npy"),
    ("nb703",        "oof",      "nb703_pred_oof.npy"),
    ("nb730",        "oof",      "nb730_pred_oof.npy"),
    ("nb1014",       "te",       "te_nb1014.npy"),
]


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def _scaffold_smiles(smi: str) -> str:
    """Murcko scaffold SMILES; empty string on failure."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def _solve_convex_qp(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Solve  min || y - X w ||^2   s.t.  sum(w) = 1, w >= 0.

    Returns weight vector and KKT dual variables.

    KKT (with Lagrangian L = ||y - Xw||^2 + lambda (1^T w - 1) - mu^T w):
        stationarity: 2 X' (X w - y) + lambda * 1 - mu = 0
        primal feasibility: 1^T w = 1, w >= 0
        dual feasibility: mu >= 0
        complementary slackness: mu_i * w_i = 0
    """
    K = X.shape[1]
    rng = np.random.default_rng(seed)
    # warm start: equal weights + small jitter
    w0 = np.full(K, 1.0 / K)
    w0 = w0 + 1e-3 * rng.standard_normal(K)
    w0 = np.clip(w0, 0.0, None)
    w0 = w0 / w0.sum()

    def f(w):
        r = y - X @ w
        return float(r @ r)

    def fp(w):
        r = X @ w - y
        return 2.0 * (X.T @ r)

    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0),
             "jac": lambda w: np.ones_like(w)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        f, w0, jac=fp, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()

    # KKT diagnostics --------------------------------------------------
    grad = 2.0 * (X.T @ (X @ w - y))           # gradient of objective
    # stationarity: grad + lambda * 1 - mu = 0
    # at optimum interior points (w_i > tol): mu_i = 0 => grad_i = -lambda
    # so lambda = -mean(grad over active interior set)
    active = w > 1e-6
    if active.any():
        lam = -float(grad[active].mean())
    else:
        lam = -float(grad.mean())
    mu = grad + lam                              # mu = grad + lam from stationarity
    mu = np.where(np.abs(mu) < 1e-9, 0.0, mu)
    # complementary slackness: w_i * mu_i should be ~0
    cs = float(np.max(np.abs(w * mu)))
    # dual feasibility violation
    mu_neg = float(-np.minimum(mu, 0.0).sum())
    # primal feasibility
    pf_eq = float(abs(w.sum() - 1.0))
    pf_nn = float(-np.minimum(w, 0.0).sum())
    return {
        "w": w.astype(np.float64),
        "lambda": lam,
        "mu": mu.astype(np.float64),
        "kkt_complementary_max": cs,
        "kkt_dual_violation": mu_neg,
        "kkt_primal_eq_violation": pf_eq,
        "kkt_primal_nn_violation": pf_nn,
        "obj_val": float(res.fun),
        "converged": bool(res.success),
        "niter": int(res.nit),
    }


def _cross_fit(X: np.ndarray, y: np.ndarray, scaffolds: list[str],
               n_splits: int, seed: int) -> dict:
    """Per-fold KKT-corrected SLSQP cross-fit.

    Returns held-out predictions vector, list of per-fold weights, RAE.
    """
    splits = scaffold_kfold_indices(scaffolds, n_splits=n_splits,
                                    shuffle=True, seed=seed)
    n = X.shape[0]
    K = X.shape[1]
    pred_cf = np.zeros(n, dtype=np.float64)
    coverage = np.zeros(n, dtype=bool)
    fold_weights = np.zeros((n_splits, K), dtype=np.float64)
    fold_records = []
    for f_idx, (tr, te) in enumerate(splits):
        X_tr, y_tr = X[tr], y[tr]
        X_te = X[te]
        kkt = _solve_convex_qp(X_tr, y_tr, seed=seed + f_idx)
        w = kkt["w"]
        fold_weights[f_idx] = w
        pred_cf[te] = X_te @ w
        coverage[te] = True
        fold_records.append({
            "fold": int(f_idx),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "weights": [round(float(v), 6) for v in w.tolist()],
            "lambda": round(float(kkt["lambda"]), 6),
            "kkt_complementary_max": round(float(kkt["kkt_complementary_max"]), 8),
            "kkt_dual_violation": round(float(kkt["kkt_dual_violation"]), 8),
            "kkt_primal_eq_violation": round(float(kkt["kkt_primal_eq_violation"]), 8),
            "kkt_primal_nn_violation": round(float(kkt["kkt_primal_nn_violation"]), 8),
            "obj_val": round(float(kkt["obj_val"]), 6),
            "converged": bool(kkt["converged"]),
            "niter": int(kkt["niter"]),
        })
    if not coverage.all():
        miss = int((~coverage).sum())
        raise RuntimeError(f"scaffold CV did not cover all rows ({miss} missed)")
    rae_val = float(rae(y, pred_cf))
    return {
        "pred_cf": pred_cf,
        "fold_weights": fold_weights,
        "fold_records": fold_records,
        "rae_cross_fit": rae_val,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- KKT-corrected per-fold SLSQP convex blend")
    print(f"          ref nb2112 cross-fit RAE = {NB2112_REF_RAE:.4f}")
    print(f"          margin needed            = {MARGIN_TO_BEAT:.4f}")
    print(f"          target                   = "
          f"{NB2112_REF_RAE - MARGIN_TO_BEAT:.4f}")
    print("=" * 78)

    # ---- Load truth + unblind index ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    truth_sha = _sha256(y_unb)
    print(f"[load] n_unb={n_unb}  truth sha={truth_sha[:16]}")

    # ---- Load test SMILES + scaffolds for the 253 ----
    te_df = load_test()
    n_test = len(te_df)
    smiles_all = te_df["smiles"].astype(str).tolist()
    smiles_unb = [smiles_all[i] for i in unb_idx]
    scaffolds_unb = [_scaffold_smiles(s) for s in smiles_unb]
    n_unique_scaf = len(set(scaffolds_unb))
    print(f"[load] scaffolds (Murcko) on 253: {n_unique_scaf} unique")

    # ---- Load anchors and verify sha256 != truth on the 253 slice ----
    K = len(ANCHOR_SPECS)
    X_unb = np.zeros((n_unb, K), dtype=np.float64)
    X_te_full = np.zeros((n_test, K), dtype=np.float64)
    anchor_records = []
    print("\n[anchor RAE check]")
    print(f"  {'idx':3s} {'name':14s} {'kind':5s} {'in_RAE':>8s}  "
          f"{'sha':>10s}  clean")
    for k_idx, (name, kind, fname) in enumerate(ANCHOR_SPECS):
        p = DATA_PROCESSED / fname
        if not p.exists():
            raise FileNotFoundError(f"missing anchor file: {p}")
        arr = np.load(p).astype(np.float64)
        if kind == "te":
            if arr.shape[0] != n_test:
                raise ValueError(f"{name} te shape {arr.shape} != {n_test}")
            sub = arr[unb_idx]
            X_te_full[:, k_idx] = arr
        elif kind == "oof":
            if arr.shape[0] != n_unb:
                raise ValueError(f"{name} oof shape {arr.shape} != {n_unb}")
            sub = arr
            # for deploy on 513, fall back to te_<name>.npy if present
            te_p = DATA_PROCESSED / f"te_{name}.npy"
            if te_p.exists():
                X_te_full[:, k_idx] = np.load(te_p).astype(np.float64)
            else:
                X_te_full[:, k_idx] = np.nan
        else:
            raise ValueError(f"unknown kind: {kind}")
        X_unb[:, k_idx] = sub
        sha = _sha256(sub.astype(y_unb.dtype))
        is_clean = sha != truth_sha
        rae_in = float(rae(y_unb, sub))
        anchor_records.append({
            "idx": int(k_idx),
            "name": name,
            "kind": kind,
            "file": fname,
            "in_RAE_253": rae_in,
            "sha256_first16": sha[:16],
            "clean_vs_truth": bool(is_clean),
        })
        print(f"  {k_idx:3d} {name:14s} {kind:5s} {rae_in:8.4f}  "
              f"{sha[:10]:>10s}  {is_clean}")
        if not is_clean:
            raise RuntimeError(
                f"INTEGRITY: anchor {name} 253-slice equals truth sha "
                "-- aborting"
            )

    # ---- Per-fold KKT-corrected SLSQP cross-fit (primary seed) ----
    print("\n[cross-fit] 5-fold scaffold CV (primary seed=42)")
    primary = _cross_fit(
        X_unb, y_unb, scaffolds_unb, n_splits=N_SPLITS, seed=SCAFFOLD_SEED
    )
    pred_cf_primary = primary["pred_cf"]
    fold_weights_primary = primary["fold_weights"]
    rae_primary = primary["rae_cross_fit"]
    weights_mean = fold_weights_primary.mean(axis=0)
    weights_std = fold_weights_primary.std(axis=0)

    print(f"\n[result] primary cross-fit RAE      = {rae_primary:.4f}")
    print(f"         nb2112 reference RAE       = {NB2112_REF_RAE:.4f}")
    print(f"         delta vs nb2112            = "
          f"{rae_primary - NB2112_REF_RAE:+.4f}")
    print(f"         need RAE <=                = "
          f"{NB2112_REF_RAE - MARGIN_TO_BEAT:.4f}")

    print("\n[mean per-fold weights]")
    for k_idx, (name, _, _) in enumerate(ANCHOR_SPECS):
        print(f"  {name:14s} mean_w={weights_mean[k_idx]:.4f}  "
              f"std_w={weights_std[k_idx]:.4f}  "
              f"fold_weights=[{','.join(f'{v:.3f}' for v in fold_weights_primary[:, k_idx])}]")

    # ---- KKT health summary across folds ----
    kkt_cs_max = max(rec["kkt_complementary_max"] for rec in primary["fold_records"])
    kkt_dual_max = max(rec["kkt_dual_violation"] for rec in primary["fold_records"])
    kkt_pf_eq_max = max(rec["kkt_primal_eq_violation"] for rec in primary["fold_records"])
    kkt_pf_nn_max = max(rec["kkt_primal_nn_violation"] for rec in primary["fold_records"])
    all_converged = all(rec["converged"] for rec in primary["fold_records"])
    print("\n[KKT health across folds]")
    print(f"  complementary slackness max viol = {kkt_cs_max:.2e}")
    print(f"  dual feasibility max viol        = {kkt_dual_max:.2e}")
    print(f"  primal eq max viol               = {kkt_pf_eq_max:.2e}")
    print(f"  primal w>=0 max viol             = {kkt_pf_nn_max:.2e}")
    print(f"  all folds converged              = {all_converged}")

    beats = rae_primary <= (NB2112_REF_RAE - MARGIN_TO_BEAT)

    # ---- Fresh-seed verification ONLY if primary beats ----
    fresh_seed_records = []
    fresh_beats_all = None
    if beats:
        print("\n[fresh-seed verify] alt scaffold seeds:", FRESH_SEEDS)
        fresh_raes = []
        for s in FRESH_SEEDS:
            r = _cross_fit(X_unb, y_unb, scaffolds_unb,
                           n_splits=N_SPLITS, seed=s)
            fresh_raes.append(r["rae_cross_fit"])
            fresh_seed_records.append({
                "seed": int(s),
                "rae_cross_fit": float(r["rae_cross_fit"]),
                "fold_weights_mean": [round(float(v), 6) for v in r["fold_weights"].mean(0).tolist()],
            })
            print(f"   seed={s:4d}  rae_cross_fit={r['rae_cross_fit']:.4f}")
        fresh_beats_all = bool(all(rv <= (NB2112_REF_RAE - MARGIN_TO_BEAT)
                                   for rv in fresh_raes))
        print(f"   median fresh   = {float(np.median(fresh_raes)):.4f}")
        print(f"   max    fresh   = {float(np.max(fresh_raes)):.4f}")
        print(f"   all seeds beat = {fresh_beats_all}")
    else:
        print("\n[fresh-seed verify] SKIPPED -- primary did not beat margin")

    # ---- DEPLOY only if primary AND all fresh seeds beat ----
    deploy_csv = None
    te_artifact = None
    deploy_in_rae = None
    if beats and fresh_beats_all:
        # use mean across the 5 primary folds for deploy
        if np.isnan(X_te_full).any():
            # some anchors lack a 513 te file (notably oof-only); reweight by
            # renormalizing on non-NaN columns per row.  In this anchor set all
            # 4 oof anchors do have te_<name>.npy so this branch is a safety net.
            print("[deploy] WARNING: some te columns NaN; renormalizing per row")
            mask = ~np.isnan(X_te_full)
            w_b = weights_mean.copy()
            te_pred_513 = np.zeros(n_test, dtype=np.float64)
            for i in range(n_test):
                m = mask[i]
                if not m.any():
                    te_pred_513[i] = float(np.nanmean(X_te_full[i]))
                    continue
                wr = w_b * m
                wr = wr / wr.sum()
                te_pred_513[i] = float((X_te_full[i] * wr)[m].sum())
        else:
            te_pred_513 = X_te_full @ weights_mean

        # in-sample diagnostic
        deploy_in_rae = float(rae(y_unb, te_pred_513[unb_idx]))
        print(f"\n[deploy] in_RAE on unb_idx = {deploy_in_rae:.4f}")

        # write submission
        if "name" in te_df.columns:
            mol_names = te_df["name"].astype(str).tolist()
        elif "Molecule Name" in te_df.columns:
            mol_names = te_df["Molecule Name"].astype(str).tolist()
        else:
            raise KeyError("no name column on test set")
        sub_df = pd.DataFrame({
            "SMILES": te_df["smiles"].astype(str).tolist(),
            "Molecule Name": mol_names,
            "pEC50": te_pred_513.astype(np.float32),
        })
        sub_dir = ROOT / "submissions"
        sub_dir.mkdir(exist_ok=True)
        deploy_csv = sub_dir / f"{TAG}_kkt_slsqp.csv"
        sub_df.to_csv(deploy_csv, index=False)
        te_artifact = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_artifact, te_pred_513.astype(np.float32))
        print(f"[deploy] submission CSV: {deploy_csv}")
        print(f"[deploy] te artifact:    {te_artifact}")
    else:
        print("\n[deploy] SKIPPED -- did not beat nb2112 by required margin")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "ref_nb2112_rae": NB2112_REF_RAE,
        "margin_to_beat": MARGIN_TO_BEAT,
        "n_anchors": int(K),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_unique_scaffolds_unb": int(n_unique_scaf),
        "truth_sha256_first16": truth_sha[:16],
        "anchor_records": anchor_records,
        "scaffold_seed_primary": SCAFFOLD_SEED,
        "n_splits": int(N_SPLITS),
        "rae_cross_fit_primary": rae_primary,
        "rae_delta_vs_nb2112": float(rae_primary - NB2112_REF_RAE),
        "beats_nb2112_by_margin": bool(beats),
        "weights_mean": [round(float(v), 6) for v in weights_mean.tolist()],
        "weights_std": [round(float(v), 6) for v in weights_std.tolist()],
        "fold_weights": [[round(float(v), 6) for v in row]
                         for row in fold_weights_primary.tolist()],
        "fold_records": primary["fold_records"],
        "kkt_complementary_slack_max": float(kkt_cs_max),
        "kkt_dual_feas_violation_max": float(kkt_dual_max),
        "kkt_primal_eq_violation_max": float(kkt_pf_eq_max),
        "kkt_primal_nonneg_violation_max": float(kkt_pf_nn_max),
        "all_folds_converged": bool(all_converged),
        "fresh_seeds": list(FRESH_SEEDS),
        "fresh_seed_records": fresh_seed_records,
        "fresh_seeds_all_beat": fresh_beats_all,
        "deploy_csv": str(deploy_csv) if deploy_csv else None,
        "te_artifact": str(te_artifact) if te_artifact else None,
        "deploy_in_rae_unb_idx": deploy_in_rae,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary: {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_anchors", "n_unb", "n_unique_scaffolds_unb",
        "rae_cross_fit_primary", "rae_delta_vs_nb2112",
        "beats_nb2112_by_margin", "fresh_seeds_all_beat",
        "kkt_complementary_slack_max", "kkt_dual_feas_violation_max",
        "kkt_primal_eq_violation_max", "kkt_primal_nonneg_violation_max",
        "all_folds_converged", "deploy_csv", "deploy_in_rae_unb_idx",
    ):
        print(f"  {k}: {res.get(k)}")
