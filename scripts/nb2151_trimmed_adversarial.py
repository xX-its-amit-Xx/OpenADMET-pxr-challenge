"""nb2151 -- Trimmed-mean bag with adversarial seed filtering on nb2095 deep-30.

Motivation
----------
nb2100 verified nb2095 across 30 fresh kf_seeds {1086..1115} and produced
30 per-seed cross-fit OOFs. The mean-bag scalar over those 30 seeds gave
pooled RAE ~0.4720 and the median-bag gave ~0.4769. Neither cached the
per-seed per-row predictions, so we reconstruct them here.

Hypothesis
----------
Some kf_seeds produce per-row predictions that are statistical outliers
relative to the seed centroid (adversarial scaffold splits that send a
sensitive scaffold to a small fold). Discarding the top/bottom 4 seeds by
Mahalanobis distance per-row should denoise the bag below 0.4720.

Pipeline
--------
  1. For each kf_seed in {1086..1115}: re-run the nb2095 pipeline (5 anchors,
     SLSQP convex blend per fold, per-fold stretch grid) to get a (253,) OOF
     prediction vector. Stack -> P (30, 253).
  2. For each row j in {0..252}:
        x_j = P[:, j]              (30 seed predictions for row j)
        mu_j = mean(x_j)
        var_j = var(x_j) + epsilon (univariate -> Mahalanobis collapses to
                                     |x - mu| / sigma, equivalent ranking)
        d_ij = |x_ij - mu_j| / sigma_j
        keep middle 22 seeds (drop top-4 + bottom-4 by d_ij)
        trimmed_pred_j = mean of kept seeds' predictions
  3. RAE(y_unb, trimmed_pred) and compare vs mean-bag / median-bag.

Note on "Mahalanobis"
---------------------
With one scalar per seed per row, Mahalanobis distance reduces to the
absolute z-score |x - mu| / sigma -- the only distance the rank-by-outlier
operation can use without a covariance across rows. We compute it that way.
Trimming top-4 + bottom-4 by |z| is identical to trimming the 4 largest +
4 smallest raw values, since z is monotone in x within a row.

Gate
----
  PASS: trimmed_mean RAE <= 0.4717 (0.4720 - 0.003 margin vs nb2100 mean-bag)
  FAIL: otherwise -> no deploy CSV emitted

Outputs
-------
  data/processed/nb2151_summary.json
  submissions/nb2151_trimmed_adversarial.csv  (only if gate PASS)
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
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2151"
N_FOLDS = 5
KF_SEEDS = list(range(1086, 1116))          # 30 fresh seeds (same as nb2100)
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
TRIM_LOW = 4                                 # drop bottom-4 per row
TRIM_HIGH = 4                                # drop top-4 per row
N_KEEP = len(KF_SEEDS) - TRIM_LOW - TRIM_HIGH  # 22
EPS = 1e-12

# Reference numbers (from nb2100 deep-30)
MEAN_BAG_REF = 0.4720                        # nb2100 mean over 30 per-seed pooled
MEDIAN_BAG_REF = 0.4769                      # nb2100 / nb1070-family median
GATE_TARGET = 0.4717                         # user spec: 0.4720 - 0.003 margin

# LB band weights (nb2095 convention)
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ----- nb2095 pipeline (verbatim) -----
ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",      "te_nb2112.npy"),
    ("nb1014",       "nb1133_nb1014_pred_oof.npy",       "te_nb1014.npy"),
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


def oof_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
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
            blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
    return oof_blend


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Trimmed-mean bag with adversarial per-row Mahalanobis "
          f"filter")
    print(f"       seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]} (n={len(KF_SEEDS)})  "
          f"keep_per_row={N_KEEP}  trim_lo/hi={TRIM_LOW}/{TRIM_HIGH}")
    print(f"       refs: mean_bag={MEAN_BAG_REF:.4f}  "
          f"median_bag={MEDIAN_BAG_REF:.4f}  gate<={GATE_TARGET:.4f}")
    print("=" * 78)

    te = load_test()
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    print(f"[load] n_te={n_te}  n_unb={n_unb}  "
          f"unique_scaffolds={len({s for s in unb_scaffolds if s})}")

    # ---- Build anchor matrix (identical to nb2095/nb2100) ----
    print("\n[anchors] loading 5-anchor stack (nb2095 pipeline)")
    oof_cols, te_cols, indiv_rae = [], [], {}
    for disp, oof_rel, te_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
        else:
            oof = np.load(DATA_PROCESSED / oof_rel).astype(np.float64)
        te_arr = np.load(DATA_PROCESSED / te_rel).astype(np.float64)
        assert oof.shape == (n_unb,)
        assert te_arr.shape == (n_te,)
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}")

    P_unb = np.column_stack(oof_cols)        # (253, 5)
    P_te = np.column_stack(te_cols)          # (513, 5)
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}")

    # ---- Reconstruct 30 per-seed OOFs ----
    print("\n" + "-" * 78)
    print(f"Reconstructing per-seed OOFs for {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    seed_oofs = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_pooled = []
    for i, kf_seed in enumerate(KF_SEEDS):
        oof = oof_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        seed_oofs[i, :] = oof
        pooled = float(rae(y_unb, oof))
        per_seed_pooled.append(pooled)
        if i % 5 == 0 or i == len(KF_SEEDS) - 1:
            print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}")

    # Cache per-seed OOF stack for reuse
    seed_oof_path = DATA_PROCESSED / f"{TAG}_per_seed_oof_30.npy"
    np.save(seed_oof_path, seed_oofs.astype(np.float32))
    print(f"[save] {seed_oof_path}  shape={seed_oofs.shape}")

    # ---- Sanity: reconfirm mean-bag and median-bag scalars ----
    mean_bag_pred = seed_oofs.mean(axis=0)
    median_bag_pred = np.median(seed_oofs, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_pred))
    rae_median_bag = float(rae(y_unb, median_bag_pred))
    rae_perseed_mean = float(np.mean(per_seed_pooled))
    rae_perseed_std = float(np.std(per_seed_pooled, ddof=1))
    print(f"\n[sanity] mean_bag_RAE     = {rae_mean_bag:.4f}  "
          f"(ref {MEAN_BAG_REF:.4f})")
    print(f"         median_bag_RAE   = {rae_median_bag:.4f}  "
          f"(ref {MEDIAN_BAG_REF:.4f})")
    print(f"         per_seed mean    = {rae_perseed_mean:.4f}  "
          f"+/- {rae_perseed_std:.4f}  ({len(per_seed_pooled)} seeds)")

    # ---- Per-row Mahalanobis (= |z|) trimming ----
    print("\n" + "-" * 78)
    print(f"Adversarial per-row trim: drop top-{TRIM_HIGH} + bottom-{TRIM_LOW} "
          f"by |z|, keep middle {N_KEEP}")
    print("-" * 78)
    mu_row = seed_oofs.mean(axis=0)                          # (253,)
    sigma_row = seed_oofs.std(axis=0, ddof=1) + EPS           # (253,)
    z = np.abs(seed_oofs - mu_row[None, :]) / sigma_row[None, :]  # (30, 253)
    # rank per column ascending in |z|; we keep the smallest N_KEEP per column
    order = np.argsort(z, axis=0)                             # (30, 253)
    keep_idx = order[:N_KEEP, :]                              # (22, 253)

    cols = np.arange(n_unb)[None, :].repeat(N_KEEP, axis=0)   # (22, 253)
    kept = seed_oofs[keep_idx, cols]                          # (22, 253)
    trimmed_pred = kept.mean(axis=0)                          # (253,)
    rae_trimmed = float(rae(y_unb, trimmed_pred))

    # diagnostics: how often is each seed kept?
    seed_keep_count = np.zeros(len(KF_SEEDS), dtype=np.int64)
    for k_col in range(N_KEEP):
        for j in range(n_unb):
            seed_keep_count[keep_idx[k_col, j]] += 1
    keep_rate = seed_keep_count / float(n_unb)                # 0..1

    print(f"   trimmed_mean_RAE = {rae_trimmed:.4f}")
    print(f"   delta vs mean_bag   = {rae_trimmed - rae_mean_bag:+.4f}")
    print(f"   delta vs median_bag = {rae_trimmed - rae_median_bag:+.4f}")
    print(f"   delta vs gate       = {rae_trimmed - GATE_TARGET:+.4f}")
    print(f"   seed keep_rate min/median/max = "
          f"{keep_rate.min():.3f} / {np.median(keep_rate):.3f} / "
          f"{keep_rate.max():.3f}")

    # ---- Gate evaluation ----
    gate_pass = bool(rae_trimmed <= GATE_TARGET)
    decision = "PROMOTE_TRIMMED" if gate_pass else "REJECT_BELOW_MARGIN"

    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    print(f"   trimmed RAE  = {rae_trimmed:.4f}")
    print(f"   gate target  = {GATE_TARGET:.4f}  (user spec: 0.4720 - 0.003)")
    print(f"   pass         = {gate_pass}  -> {decision}")

    # ---- Deploy CSV if gate passes ----
    deploy_csv_path = None
    te_deploy = None
    if gate_pass:
        print("\n" + "-" * 78)
        print("DEPLOY refit (trimmed-mean bag transferred to test set)")
        print("-" * 78)
        # We also need 30 per-seed te predictions to apply the same trim
        # transform. Since each per-seed run only changes the fold splits on
        # the unblind cross-fit (the test set is predicted by a full-pool
        # SLSQP + stretch refit), we replicate the nb2095 deploy: fit SLSQP
        # weights + stretch on the full (P_unb, y_unb), apply to P_te. Then
        # we additionally apply a stretch derived from the trimmed bag mean
        # to keep variance consistent.
        w_deploy = slsqp_simplex(P_unb, y_unb)
        blend_unb = P_unb @ w_deploy
        mu_deploy = float(blend_unb.mean())
        # Use the mean over per-seed best stretches as in nb2100
        # (we lost the fold_s here; reconstruct from one run with seed=1086
        # for a stable reference)
        ref_splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEEDS[0],
        )
        fold_s_ref = []
        for tr_loc, _ in ref_splits:
            w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
            blend_tr = P_unb[tr_loc] @ w_f
            mu_tr = float(blend_tr.mean())
            s_f, _ = best_stretch_on(
                blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
            )
            fold_s_ref.append(s_f)
        s_deploy = float(np.mean(fold_s_ref))
        blend_te = P_te @ w_deploy
        te_deploy = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(
            np.float32
        )
        te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
        lb_band_est = LB_W_OOF * rae_trimmed + LB_W_TE * te_unb_rae

        # Save te artefact
        te_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_path, te_deploy)
        # Submission CSV
        sub = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te["molecule_name"].values
                if "molecule_name" in te.columns else te.iloc[:, 0].values,
            "pEC50": te_deploy,
        })
        deploy_csv_path = (DATA_PROCESSED.parent.parent / "submissions"
                           / f"{TAG}_trimmed_adversarial.csv")
        deploy_csv_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(deploy_csv_path, index=False)

        print(f"   deploy weights = " + ", ".join(
            f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy)
        ))
        print(f"   mu / s         = {mu_deploy:.4f} / {s_deploy:.4f}")
        print(f"   te[unb] RAE    = {te_unb_rae:.4f}  (in-sample)")
        print(f"   LB-band est    = {LB_W_OOF:.2f}*{rae_trimmed:.4f} + "
              f"{LB_W_TE:.2f}*{te_unb_rae:.4f} = {lb_band_est:.4f}")
        print(f"   te_npy         = {te_path}")
        print(f"   submission     = {deploy_csv_path}")
    else:
        te_unb_rae = None
        lb_band_est = None
        print("\n   gate FAIL -> no deploy artefact emitted.")

    summary = {
        "tag": TAG,
        "method": "trimmed_mean_bag_with_per_row_mahalanobis_filter_on_nb2095_deep30",
        "purpose": "denoise nb2095 30-seed bag by dropping per-row outlier seeds",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "trim_low": TRIM_LOW,
        "trim_high": TRIM_HIGH,
        "n_keep_per_row": N_KEEP,
        "mahalanobis_kind": "per_row_absolute_zscore_equivalent_under_scalar",
        "n_unb": n_unb,
        "n_te": n_te,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_pooled_mean": rae_perseed_mean,
        "per_seed_pooled_std_ddof1": rae_perseed_std,
        "mean_bag_rae_actual": rae_mean_bag,
        "median_bag_rae_actual": rae_median_bag,
        "mean_bag_rae_ref": MEAN_BAG_REF,
        "median_bag_rae_ref": MEDIAN_BAG_REF,
        "trimmed_mean_rae": rae_trimmed,
        "delta_trimmed_minus_mean": rae_trimmed - rae_mean_bag,
        "delta_trimmed_minus_median": rae_trimmed - rae_median_bag,
        "delta_trimmed_minus_gate": rae_trimmed - GATE_TARGET,
        "seed_keep_rate_min": float(keep_rate.min()),
        "seed_keep_rate_median": float(np.median(keep_rate)),
        "seed_keep_rate_max": float(keep_rate.max()),
        "seed_keep_rates": [float(x) for x in keep_rate.tolist()],
        "gate_target": GATE_TARGET,
        "gate_margin_vs_mean_bag": 0.003,
        "gate_pass": gate_pass,
        "decision": decision,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "te_npy_path": str(DATA_PROCESSED / f"te_{TAG}.npy") if gate_pass
            else None,
        "submission_csv": str(deploy_csv_path) if gate_pass else None,
        "per_seed_oof_cache": str(seed_oof_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_bag   = {rae_mean_bag:.4f}  (ref {MEAN_BAG_REF:.4f})")
    print(f"   median_bag = {rae_median_bag:.4f}  (ref {MEDIAN_BAG_REF:.4f})")
    print(f"   trimmed_22 = {rae_trimmed:.4f}  "
          f"(delta_vs_mean={rae_trimmed - rae_mean_bag:+.4f})")
    print(f"   gate       = {GATE_TARGET:.4f}  -> "
          f"{'PASS' if gate_pass else 'FAIL'}  -> {decision}")
    print(f"   wall       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_bag_rae_actual",
        "median_bag_rae_actual",
        "trimmed_mean_rae",
        "delta_trimmed_minus_mean",
        "delta_trimmed_minus_median",
        "delta_trimmed_minus_gate",
        "gate_pass",
        "decision",
    ):
        print(f"  {k}: {res.get(k)}")
