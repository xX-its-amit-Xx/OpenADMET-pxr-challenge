"""nb2820 -- Platt-scaling per anchor then linear SLSQP blend.

NEW PARADIGM (vs cycle-134 paradigm exhaustion):
    Prior meta-stack attacks on the (chemprop_aux + nb2240_K20 + counter_clean,
    n=253) tuple were either tree splits (LGBM/XGBoost/CatBoost), sparse
    linear (LASSO), ridge/MLP, or a tiny ReLU NN -- ALL collapsed to the
    0.4718-0.4720 deep-30 ceiling. They all kept the raw anchor scale.

    Platt scaling is sigmoid (Bradley/Niculescu-Mizil 2005) calibration:
    fit logistic-regression A,B on (anchor_pred, binary_y > 5.5) per fold,
    then squash each anchor through sigma(A*pred + B). This is a monotone
    non-linear transform with a single decision boundary (the 5.5 hit
    threshold). After Platt, the anchors live on a *common probabilistic
    scale* aligned to hit/miss orientation -- distinct inductive bias from
    every prior stacker, because the input scale itself moves.

    Final blend is linear SLSQP simplex (w >= 0, sum=1) over the
    Platt-transformed anchors. Linear pre-Platt; non-linear via the
    per-anchor sigmoid; back to linear post-Platt. This is the smallest
    capacity-add over the existing linear blends (each anchor adds 2 params
    A, B -- 6 free params total vs SLSQP-simplex K-1=2 weights, so 8
    total << 253 OOF labels).

SUBSTRATE (PRE-clean only, 3 anchors):
    - nb2240_K20      (K=20 residual stack on chemprop_aux)
    - chemprop_aux    (nb1133, 4139 PRE-unblind only)
    - counter_clean   (nb2490 counter-assay residual, nb730-free)

PROTOCOL:
    - 5-fold scaffold CV on 253 unblind, 5 kf_seeds {42, 1, 7, 137, 1009}
    - Per anchor per fold:
        sklearn LogisticRegression on (anchor_pred_train, binary_y > 5.5)
        -> coefficients A, B; transform = sigmoid(A*pred + B)
    - Linear blend via SLSQP simplex on Platt-transformed anchors
    - Since sigmoid output is in [0,1] and y is pEC50 in ~[3,9], we
      rescale the blended sigmoid back to pEC50 via per-fold linear
      regression on train (slope, intercept) -- this avoids hard-coding
      a [3,9] mapping and tracks fold-train distribution.
    - Deploy: refit Platt-params + SLSQP-weights + rescale on all 253,
      predict te 513

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT
    else                -> FAIL

Outputs:
    scripts/nb2820_platt_stacking.py
    data/processed/nb2820_summary.json
    data/processed/nb2820_pred_oof.npy   (253,) float32
    data/processed/te_nb2820.npy         (513,) float32
    submissions/nb2820_platt_stacking.csv
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
import pandas as pd
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2820"
N_FOLDS = 5
KF_SEEDS = [42, 1, 7, 137, 1009]
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
HIT_THRESHOLD = 5.5  # pEC50 >= 5.5 -> binary 1 for Platt fit

# ---- PRE-clean anchors only (3 required) ----
CANDIDATE_ANCHORS = [
    ("nb2240_K20",   DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
                     DATA_PROCESSED / "te_nb2240_K20.npy",
                     "PRE-clean (K=20 residual stack on chemprop_aux)"),
    ("chemprop_aux", DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
                     DATA_PROCESSED / "te_chemprop_aux.npy",
                     "PRE-clean (4139 PRE-unblind only)"),
    ("counter_clean",DATA_PROCESSED / "nb2490_pred_oof.npy",
                     DATA_PROCESSED / "te_nb2490.npy",
                     "PRE-clean counter-assay residual on chemprop_aux (nb730-free)"),
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def fit_platt_per_anchor(P_tr: np.ndarray, y_tr: np.ndarray):
    """Fit sigmoid(A*p + B) per anchor on train data via LogisticRegression."""
    K = P_tr.shape[1]
    y_bin = (y_tr >= HIT_THRESHOLD).astype(int)
    coefs = np.zeros((K, 2), dtype=np.float64)  # (A, B) per anchor
    # Edge: if y_bin is constant, Platt degenerates -> fallback identity
    if y_bin.min() == y_bin.max():
        coefs[:, 0] = 1.0
        coefs[:, 1] = 0.0
        return coefs, True
    for j in range(K):
        x = P_tr[:, j].reshape(-1, 1).astype(np.float64)
        lr = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=500,
        )
        lr.fit(x, y_bin)
        coefs[j, 0] = float(lr.coef_.ravel()[0])
        coefs[j, 1] = float(lr.intercept_.ravel()[0])
    return coefs, False


def apply_platt(P: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    out = np.zeros_like(P, dtype=np.float64)
    for j in range(K):
        A, B = coefs[j]
        out[:, j] = _sigmoid(A * P[:, j] + B)
    return out


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


def fit_linear_rescale(blend_tr: np.ndarray, y_tr: np.ndarray):
    """Linear rescale of sigmoid blend (in [0,1]) back to pEC50 scale."""
    # y_tr = slope * blend_tr + intercept; closed-form OLS
    x = blend_tr.astype(np.float64)
    y = y_tr.astype(np.float64)
    xm = x.mean()
    ym = y.mean()
    denom = float(np.sum((x - xm) ** 2))
    if denom < 1e-12:
        return 0.0, float(ym)
    slope = float(np.sum((x - xm) * (y - ym)) / denom)
    intercept = float(ym - slope * xm)
    return slope, intercept


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Platt scaling per anchor + SLSQP linear blend")
    print("=" * 78)

    # ---- Load ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test} n_unb={n_unb}")

    # ---- Resolve anchors (strict: need all 3) ----
    anchor_names = []
    anchor_provenance = {}
    oof_cols, te_cols = [], []
    anchor_skipped = {}
    for name, oof_path, te_path, prov in CANDIDATE_ANCHORS:
        if not oof_path.exists():
            anchor_skipped[name] = f"pred_oof missing at {oof_path}"
            print(f"   SKIP {name}: pred_oof missing")
            continue
        if not te_path.exists():
            anchor_skipped[name] = f"te missing at {te_path}"
            print(f"   SKIP {name}: te missing")
            continue
        oof = np.load(oof_path).astype(np.float64)
        te_v = np.load(te_path).astype(np.float64)
        if oof.shape[0] != n_unb:
            anchor_skipped[name] = f"shape mismatch oof={oof.shape} expected ({n_unb},)"
            print(f"   SKIP {name}: shape mismatch")
            continue
        if te_v.shape[0] != n_test:
            anchor_skipped[name] = f"shape mismatch te={te_v.shape} expected ({n_test},)"
            print(f"   SKIP {name}: te shape mismatch")
            continue
        anchor_names.append(name)
        anchor_provenance[name] = prov
        oof_cols.append(oof)
        te_cols.append(te_v)
    K = len(anchor_names)
    if K < 3:
        raise RuntimeError(f"Need 3 PRE-clean anchors, got {K}: {anchor_names}")

    P_unb = np.column_stack(oof_cols)  # (253, K)
    P_te = np.column_stack(te_cols)    # (513, K)
    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    print(f"[anchors] K={K}  hit_thresh={HIT_THRESHOLD}")
    for k in anchor_names:
        print(f"   {k:14s}  unb_RAE={rae_anchors[k]:.4f}  [{anchor_provenance[k]}]")
    n_hits = int((y_unb >= HIT_THRESHOLD).sum())
    print(f"[binary] n_hits(y>={HIT_THRESHOLD}) = {n_hits}/{n_unb} "
          f"({100.0*n_hits/n_unb:.1f}%)")
    if anchor_skipped:
        print(f"[skipped] {anchor_skipped}")

    # ---- Scaffold folds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}  n_folds={N_FOLDS}  kf_seeds={KF_SEEDS}")

    # ---- Platt + SLSQP + linear-rescale CV ----
    print("\n" + "-" * 78)
    print("PLATT + SLSQP CV (5 kf_seeds x 5-fold scaffold)")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_mean_fold = []
    per_seed_w_mean = []
    per_seed_platt_A_mean = np.zeros((len(KF_SEEDS), K), dtype=np.float64)
    per_seed_platt_B_mean = np.zeros((len(KF_SEEDS), K), dtype=np.float64)
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        shuffle=True, seed=kf_seed)
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae, fold_w, fold_A, fold_B = [], [], [], []
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            # Per-anchor Platt fit on fold-train
            coefs, degen = fit_platt_per_anchor(P_unb[tr_loc], y_unb[tr_loc])
            P_tr_plt = apply_platt(P_unb[tr_loc], coefs)
            P_va_plt = apply_platt(P_unb[va_loc], coefs)
            # SLSQP simplex blend on Platt-space against y
            # (We blend Platt outputs and then linear-rescale to pEC50;
            #  the rescale absorbs the [0,1]->[3,9] mapping.)
            # But SLSQP needs a target -- we first rescale Platt blend.
            # Strategy: SLSQP on Platt directly minimising MSE against
            # standardised y (mean-center y on train) -- but a simpler &
            # exact path: fit SLSQP weights w on Platt, then linear-rescale
            # blend_tr to y_tr. To get a proper SLSQP, we target the
            # linearly-rescaled-and-then-rescaled-back y. Use a two-step
            # iteration: (1) find initial slope/intercept from mean Platt
            # blend, (2) fit SLSQP minimising || slope*Pw + intercept - y ||
            # which reduces to standard MSE on rescaled space. Since
            # rescale is scalar affine, the optimal w is invariant to it
            # AS LONG AS the rescale is the same in both train/val. We
            # therefore (a) fit w via SLSQP on raw Platt vs y, then
            # (b) fit slope/intercept of y vs (P_tr_plt @ w) -- this is
            # the standard Platt-blend-then-calibrate recipe.
            w_f = slsqp_simplex(P_tr_plt, y_unb[tr_loc])
            blend_tr = P_tr_plt @ w_f
            slope, intercept = fit_linear_rescale(blend_tr, y_unb[tr_loc])
            blend_va = P_va_plt @ w_f
            pred_va = slope * blend_va + intercept
            oof[va_loc] = pred_va
            r = float(rae(y_unb[va_loc], pred_va))
            fold_rae.append(r)
            fold_w.append(w_f)
            fold_A.append(coefs[:, 0])
            fold_B.append(coefs[:, 1])
        pooled = float(rae(y_unb, oof))
        mean_fold = float(np.mean(fold_rae))
        per_seed_pooled.append(pooled)
        per_seed_mean_fold.append(mean_fold)
        w_mean = np.mean(np.column_stack(fold_w), axis=1)
        per_seed_w_mean.append(w_mean.tolist())
        per_seed_platt_A_mean[s_idx] = np.mean(np.column_stack(fold_A), axis=1)
        per_seed_platt_B_mean[s_idx] = np.mean(np.column_stack(fold_B), axis=1)
        oof_seed_stack[s_idx] = oof
        print(f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  "
              f"mean_fold={mean_fold:.4f}  w_mean={np.round(w_mean, 3).tolist()}")

    mean_pooled = float(np.mean(per_seed_pooled))
    std_pooled = float(np.std(per_seed_pooled))
    oof_final = oof_seed_stack.mean(axis=0)
    final_pooled_on_seed_mean = float(rae(y_unb, oof_final))

    print(f"\n[wide-seed] mean pooled = {mean_pooled:.4f} +/- {std_pooled:.4f} "
          f"(n_seeds={len(KF_SEEDS)})")
    print(f"[wide-seed] pooled on seed-mean OOF = {final_pooled_on_seed_mean:.4f}")

    # ---- Gate ----
    if mean_pooled < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_pooled < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_pooled {mean_pooled:.4f}  "
          f"(< {GATE_PROMOTE} PROMOTE / < {GATE_MARGINAL} MARGINAL)  ->  {verdict}")

    # ---- Deploy: refit Platt+SLSQP+rescale on all 253, predict 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit Platt+SLSQP+rescale on all 253, predict 513")
    print("-" * 78)
    coefs_full, degen_full = fit_platt_per_anchor(P_unb, y_unb)
    P_unb_plt = apply_platt(P_unb, coefs_full)
    P_te_plt = apply_platt(P_te, coefs_full)
    w_deploy = slsqp_simplex(P_unb_plt, y_unb)
    blend_unb = P_unb_plt @ w_deploy
    slope_deploy, intercept_deploy = fit_linear_rescale(blend_unb, y_unb)
    te_pred = (slope_deploy * (P_te_plt @ w_deploy) + intercept_deploy).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy A         = {[float(v) for v in coefs_full[:, 0]]}")
    print(f"   deploy B         = {[float(v) for v in coefs_full[:, 1]]}")
    print(f"   deploy w         = {[float(v) for v in w_deploy]}")
    print(f"   deploy slope/int = {slope_deploy:.4f}/{intercept_deploy:.4f}")
    print(f"   te mean={te_pred.mean():.3f} std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << pooled)")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_final.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_platt_stacking.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "platt_scaling_per_anchor_then_slsqp_linear_blend",
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_provenance,
        "anchor_skipped": anchor_skipped,
        "anchor_in_rae": rae_anchors,
        "hit_threshold": HIT_THRESHOLD,
        "n_hits": n_hits,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_mean_fold_rae": per_seed_mean_fold,
        "per_seed_w_mean": per_seed_w_mean,
        "per_seed_platt_A_mean": per_seed_platt_A_mean.tolist(),
        "per_seed_platt_B_mean": per_seed_platt_B_mean.tolist(),
        "mean_pooled_rae": mean_pooled,
        "std_pooled_rae": std_pooled,
        "pooled_on_seed_mean_oof": final_pooled_on_seed_mean,
        "mean_rae": mean_pooled,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_platt_A": [float(v) for v in coefs_full[:, 0]],
        "deploy_platt_B": [float(v) for v in coefs_full[:, 1]],
        "deploy_platt_degenerate": bool(degen_full),
        "deploy_w": [float(v) for v in w_deploy],
        "deploy_slope": float(slope_deploy),
        "deploy_intercept": float(intercept_deploy),
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean pooled RAE     = {mean_pooled:.4f} +/- {std_pooled:.4f}  ({verdict})")
    print(f"   K anchors used      = {K}  ({anchor_names})")
    print(f"   deploy w            = {[round(float(v), 4) for v in w_deploy]}")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("mean_pooled_rae", "std_pooled_rae", "verdict",
              "K_anchors", "te_unb_in_sample_rae", "submission_csv"):
        print(f"  {k}: {res.get(k)}")
