"""nb2191 -- Trimmed-mean adversarial on nb2171 deep-30.

Mirror of nb2151 (which was a no-op on nb2095). Per cycle 166 + nb2180:
nb2171 5-anchor PRE-clean swap (nb2103_K28, chemprop_aux, reconstructed
nb1191, nb503, nb562) deep-verified across kf_seeds {1116..1145} at pooled
RAE 0.4682 +/- 0.0024 (n=30). We reconstruct the same 30 per-seed OOFs and
ask: does dropping per-row adversarial seeds (top-4 + bottom-4 by |z|, keep
middle 22) tighten the bag below 0.4682?

Pipeline (identical structure to nb2151 / nb2100):
  1. Re-run nb2171 SLSQP+stretch pipeline for each kf_seed in {1116..1145}
     to recover the (30, 253) per-seed OOF matrix.
  2. Per row j: rank seeds by |x - mu_j| / sigma_j, keep middle 22.
  3. Trimmed-mean per row -> RAE(y_unb, trimmed_pred).
  4. Compare vs nb2171 deep-30 mean-bag 0.4682; decision margin 0.003.

Mahalanobis -> |z|: with one scalar per (seed, row), Mahalanobis distance
to the row centroid collapses to |x_ij - mu_j| / sigma_j. Trim by |z|
ranks seeds identically to trimming the 4 largest + 4 smallest raw values.

Gate:
  PASS (PROMOTE_TRIMMED): trimmed_RAE <= 0.4682 - 0.003 = 0.4652
  FAIL (REJECT_BELOW_MARGIN): otherwise; no deploy CSV emitted
    -- mirrors nb2151's neutral verdict on nb2095 if the trim has no effect.

Outputs:
  data/processed/nb2191_summary.json
  data/processed/nb2191_per_seed_oof_30.npy
  submissions/nb2191_trimmed_nb2171.csv (only on gate pass)
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2191"
N_FOLDS = 5
KF_SEEDS = list(range(1116, 1146))               # 30 deep seeds (= nb2180 set)
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
TRIM_LOW = 4
TRIM_HIGH = 4
N_KEEP = len(KF_SEEDS) - TRIM_LOW - TRIM_HIGH    # 22
EPS = 1e-12

# Reference: nb2180 deep-30 mean-bag of nb2171 (data/processed/nb2180_summary.json)
NB2171_DEEP30_MEAN = 0.4682
NB2171_DEEP30_STD = 0.0024
MEAN_BAG_REF = NB2171_DEEP30_MEAN
GATE_MARGIN = 0.003
GATE_TARGET = MEAN_BAG_REF - GATE_MARGIN          # 0.4652

LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---- Anchors (identical to nb2171 / nb2180) ----
ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("nb1191",       "_RECONSTRUCT_nb1191_oof",           "te_nb1191.npy"),
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]
NB1191_ANCHOR_IDX = 2

# nb1191 reconstruction params (identical to nb2171 / nb2180)
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031

NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)
    return P @ w


def reconstruct_nb1191_oof(n_unb: int) -> np.ndarray:
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    nb2112_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    assert chemprop_oof.shape == (n_unb,)
    assert nb1150_oof.shape == (n_unb,)
    assert nb1158_oof.shape == (n_unb,)
    assert nb2112_oof.shape == (n_unb,)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


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
    fold_s_list = []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(
            blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_s_list.append(s_f)
    return oof_blend, fold_s_list


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Trimmed-mean adversarial on nb2171 deep-30 (mirror of nb2151)")
    print(f"       seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]} (n={len(KF_SEEDS)})  "
          f"keep_per_row={N_KEEP}  trim_lo/hi={TRIM_LOW}/{TRIM_HIGH}")
    print(f"       nb2171 deep-30 mean-bag ref = {MEAN_BAG_REF:.4f}  "
          f"gate <= {GATE_TARGET:.4f}  (margin {GATE_MARGIN:.3f})")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    print(f"[load] n_te={n_te}  n_unb={n_unb}  "
          f"unique_scaffolds={len({s for s in unb_scaffolds if s})}")

    # ---- Build anchor matrix (identical to nb2171 / nb2180) ----
    print("\n[anchors] loading 5-anchor stack (nb2171 PRE-clean pipeline)")
    oof_cols, te_cols, indiv_rae = [], [], {}
    for disp, oof_rel, te_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1191_oof":
            oof = reconstruct_nb1191_oof(n_unb)
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

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}")

    # ---- Reconstruct 30 per-seed OOFs ----
    print("\n" + "-" * 78)
    print(f"Reconstructing per-seed OOFs for {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    seed_oofs = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_pooled = []
    per_seed_fold_s = []
    for i, kf_seed in enumerate(KF_SEEDS):
        oof, fold_s = oof_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        seed_oofs[i, :] = oof
        pooled = float(rae(y_unb, oof))
        per_seed_pooled.append(pooled)
        per_seed_fold_s.append(fold_s)
        if i % 5 == 0 or i == len(KF_SEEDS) - 1:
            print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
                  f"mean_s={np.mean(fold_s):.3f}")

    # Cache per-seed OOF stack
    seed_oof_path = DATA_PROCESSED / f"{TAG}_per_seed_oof_30.npy"
    np.save(seed_oof_path, seed_oofs.astype(np.float32))
    print(f"[save] {seed_oof_path}  shape={seed_oofs.shape}")

    # ---- Sanity: mean-bag and median-bag scalars ----
    mean_bag_pred = seed_oofs.mean(axis=0)
    median_bag_pred = np.median(seed_oofs, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_pred))
    rae_median_bag = float(rae(y_unb, median_bag_pred))
    rae_perseed_mean = float(np.mean(per_seed_pooled))
    rae_perseed_std = float(np.std(per_seed_pooled, ddof=1))
    print(f"\n[sanity] mean_bag_RAE     = {rae_mean_bag:.4f}  "
          f"(nb2180 deep-30 ref {MEAN_BAG_REF:.4f})")
    print(f"         median_bag_RAE   = {rae_median_bag:.4f}")
    print(f"         per_seed mean    = {rae_perseed_mean:.4f}  "
          f"+/- {rae_perseed_std:.4f}  ({len(per_seed_pooled)} seeds)")

    # ---- Per-row Mahalanobis (= |z|) trimming ----
    print("\n" + "-" * 78)
    print(f"Adversarial per-row trim: drop top-{TRIM_HIGH} + bottom-{TRIM_LOW} "
          f"by |z|, keep middle {N_KEEP}")
    print("-" * 78)
    mu_row = seed_oofs.mean(axis=0)                              # (253,)
    sigma_row = seed_oofs.std(axis=0, ddof=1) + EPS               # (253,)
    z = np.abs(seed_oofs - mu_row[None, :]) / sigma_row[None, :]  # (30, 253)
    order = np.argsort(z, axis=0)                                 # ascending |z|
    keep_idx = order[:N_KEEP, :]                                  # (22, 253)
    cols = np.arange(n_unb)[None, :].repeat(N_KEEP, axis=0)       # (22, 253)
    kept = seed_oofs[keep_idx, cols]                              # (22, 253)
    trimmed_pred = kept.mean(axis=0)                              # (253,)
    rae_trimmed = float(rae(y_unb, trimmed_pred))

    # seed keep-rate diagnostic
    seed_keep_count = np.zeros(len(KF_SEEDS), dtype=np.int64)
    for k_col in range(N_KEEP):
        for j in range(n_unb):
            seed_keep_count[keep_idx[k_col, j]] += 1
    keep_rate = seed_keep_count / float(n_unb)

    print(f"   trimmed_mean_RAE  = {rae_trimmed:.4f}")
    print(f"   delta vs mean_bag = {rae_trimmed - rae_mean_bag:+.4f}")
    print(f"   delta vs median   = {rae_trimmed - rae_median_bag:+.4f}")
    print(f"   delta vs gate     = {rae_trimmed - GATE_TARGET:+.4f}")
    print(f"   seed keep_rate min/median/max = "
          f"{keep_rate.min():.3f} / {np.median(keep_rate):.3f} / "
          f"{keep_rate.max():.3f}")

    # ---- Gate ----
    gate_pass = bool(rae_trimmed <= GATE_TARGET)
    decision = "PROMOTE_TRIMMED" if gate_pass else "REJECT_BELOW_MARGIN"
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    print(f"   trimmed RAE  = {rae_trimmed:.4f}")
    print(f"   gate target  = {GATE_TARGET:.4f}  "
          f"(nb2171 deep-30 mean {MEAN_BAG_REF:.4f} - margin {GATE_MARGIN:.3f})")
    print(f"   pass         = {gate_pass}  -> {decision}")

    # ---- Deploy CSV on gate pass ----
    deploy_csv_path = None
    te_unb_rae = None
    lb_band_est = None
    s_deploy = None
    w_deploy_arr = None
    if gate_pass:
        print("\n" + "-" * 78)
        print("DEPLOY refit (trimmed-bag transfer to 513)")
        print("-" * 78)
        w_deploy_arr = slsqp_simplex(P_unb, y_unb)
        blend_unb = P_unb @ w_deploy_arr
        mu_deploy = float(blend_unb.mean())
        # Use mean over all 30 seeds' fold_s (matches nb2180 deploy convention)
        s_deploy = float(np.mean(
            [s for fold_s in per_seed_fold_s for s in fold_s]
        ))
        blend_te = P_te @ w_deploy_arr
        deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(
            np.float32
        )
        te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
        lb_band_est = LB_W_OOF * rae_trimmed + LB_W_TE * te_unb_rae

        te_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_path, deploy_te)
        sub = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        })
        deploy_csv_path = SUBMISSIONS / f"{TAG}_trimmed_nb2171.csv"
        deploy_csv_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(deploy_csv_path, index=False)
        print(f"   deploy weights = " + ", ".join(
            f"{disp}={w:.4f}"
            for (disp, _, _), w in zip(ANCHORS, w_deploy_arr)
        ))
        print(f"   mu / s         = {mu_deploy:.4f} / {s_deploy:.4f}")
        print(f"   te[unb] RAE    = {te_unb_rae:.4f}  (in-sample)")
        print(f"   LB-band est    = {LB_W_OOF:.2f}*{rae_trimmed:.4f} + "
              f"{LB_W_TE:.2f}*{te_unb_rae:.4f} = {lb_band_est:.4f}")
        print(f"   te_npy         = {te_path}")
        print(f"   submission     = {deploy_csv_path}")
    else:
        print("\n   gate FAIL -> no deploy artefact emitted.")

    summary = {
        "tag": TAG,
        "method": (
            "trimmed_mean_bag_with_per_row_mahalanobis_filter_on_nb2171_deep30"
        ),
        "mirror_of": "nb2151",
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
        "mahalanobis_kind": (
            "per_row_absolute_zscore_equivalent_under_scalar"
        ),
        "n_unb": n_unb,
        "n_te": n_te,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_pooled_mean": rae_perseed_mean,
        "per_seed_pooled_std_ddof1": rae_perseed_std,
        "mean_bag_rae_actual": rae_mean_bag,
        "median_bag_rae_actual": rae_median_bag,
        "mean_bag_rae_ref_nb2180_deep30": MEAN_BAG_REF,
        "trimmed_mean_rae": rae_trimmed,
        "delta_trimmed_minus_mean": rae_trimmed - rae_mean_bag,
        "delta_trimmed_minus_median": rae_trimmed - rae_median_bag,
        "delta_trimmed_minus_gate": rae_trimmed - GATE_TARGET,
        "seed_keep_rate_min": float(keep_rate.min()),
        "seed_keep_rate_median": float(np.median(keep_rate)),
        "seed_keep_rate_max": float(keep_rate.max()),
        "seed_keep_rates": [float(x) for x in keep_rate.tolist()],
        "gate_target": GATE_TARGET,
        "gate_margin_vs_mean_bag": GATE_MARGIN,
        "gate_pass": gate_pass,
        "decision": decision,
        "deploy_weights": (
            [
                {"name": disp, "w": float(w)}
                for (disp, _, _), w in zip(ANCHORS, w_deploy_arr)
            ]
            if gate_pass else None
        ),
        "deploy_s": s_deploy,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "te_npy_path": (
            str(DATA_PROCESSED / f"te_{TAG}.npy") if gate_pass else None
        ),
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
    print(f"   mean_bag      = {rae_mean_bag:.4f}  "
          f"(nb2180 deep-30 ref {MEAN_BAG_REF:.4f})")
    print(f"   median_bag    = {rae_median_bag:.4f}")
    print(f"   trimmed_22    = {rae_trimmed:.4f}  "
          f"(delta_vs_mean={rae_trimmed - rae_mean_bag:+.4f})")
    print(f"   gate          = {GATE_TARGET:.4f}  -> "
          f"{'PASS' if gate_pass else 'FAIL'}  -> {decision}")
    print(f"   wall          = {time.time() - t0:.1f}s")
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
