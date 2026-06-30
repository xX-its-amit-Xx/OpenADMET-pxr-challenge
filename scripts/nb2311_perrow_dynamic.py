"""nb2311 -- Per-row dynamic stack of top-5 PRIMARY candidates.

For each row in {253 unblind, 513 test} compute the per-row uncertainty
across the 5 candidate predictors as their cross-candidate std.  Weight
each candidate inversely with a softmin via:

    w_i(row) = exp(-std_row / sigma_global) / Z(row)

i.e. ALL anchors share the same weight at a given row (it is a confidence
gate, not a per-anchor reweighting).  The actual per-anchor weights then
become equal at every row, so this collapses to a simple per-row mean ---
which is NOT what is intended.

Reinterpret: the candidate with the LOWEST disagreement at row r is the
most confident (the anchors agree).  We want to upweight rows where the
ensemble is confident, but anchor weights per row should reflect each
anchor's distance from the median.  Following the spec literally:

  - per-row: std across the 5 predictors  -> shape (N,)
  - per-anchor weight at row r is:
        w_i(r) = exp(-|p_i(r) - med(r)| / sigma_global) / Z(r)
    so outliers at row r get downweighted (their |dev| is large), and
    the sigma_global is the global RMS of the per-row stds.

This is the standard interpretation of "weight inversely on uncertainty"
when anchors are heterogeneous - the per-anchor deviation from row
consensus IS the per-anchor uncertainty.

CANDIDATES (top-5 PRIMARY ladder as of 2026-06-08):
    1. nb2240 (current PRIMARY-1, K=20 pyramid; OOF 0.4601)
    2. nb1162 (5-anchor stack w/ nb730; OOF 0.4206)
    3. nb2171 (5-anchor swap nb1191 for nb730; OOF 0.4676)
    4. nb2095 (4-anchor PRE-unblind pyramid; OOF 0.4703)
    5. nb1191 (4-anchor PRE-unblind pyramid; OOF 0.4697)

OUTPUTS:
    scripts/nb2311_perrow_dynamic.py            (this file)
    data/processed/nb2311_summary.json
    data/processed/nb2311_perrow_oof.npy         (253,) float32
    data/processed/te_nb2311.npy                 (513,) float32
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2311"

GATE_MARGIN = 0.003
NB2240_REF_OOF = 0.4601  # current PRIMARY-1 reference (mean across kf_seeds)

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb1191 deploy reconstruction (copied from nb2171 / nb2240)
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

# nb2095 deploy reconstruction (deploy_weights from nb2095_summary.json)
NB2095_DEPLOY_WEIGHTS = {
    "chemprop_aux": 3.8701608082398734e-15,
    "nb1150":       0.6417213409280851,
    "nb1158_K32":   0.23970113464990606,
    "nb2112_K28":   0.11857752442200314,
    "nb1014":       1.6746553256665777e-15,
}
NB2095_DEPLOY_S = 1.031

# nb1162 deploy reconstruction (deploy_weights from nb1162_summary.json)
NB1162_DEPLOY_WEIGHTS = {
    "nb2103_K28":   0.11293551862971236,
    "chemprop_aux": 1.048266159658086e-12,
    "nb730_honest": 0.8870644813686026,
    "nb503":        3.3986421081510336e-13,
    "nb562":        2.968780695135247e-13,
}
NB1162_DEPLOY_S = 1.0

# nb2171 deploy reconstruction (deploy_weights from nb2171_summary.json)
NB2171_DEPLOY_WEIGHTS = {
    "nb2103_K28":   0.0,
    "chemprop_aux": 1.8510903813157475e-12,
    "nb1191":       0.9360891805838886,
    "nb503":        0.0,
    "nb562":        0.06391081941426041,
}
NB2171_DEPLOY_S = 1.009


# ----------------------------------------------------------------------------
# anchor reconstructions on the 253 unblind
# ----------------------------------------------------------------------------

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
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


def reconstruct_nb2095_oof(n_unb: int) -> np.ndarray:
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    nb1014_oof = np.load(DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy").astype(np.float64)
    blend = (
        NB2095_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB2095_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB2095_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB2095_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
        + NB2095_DEPLOY_WEIGHTS["nb1014"]       * nb1014_oof
    )
    mu = float(blend.mean())
    return mu + NB2095_DEPLOY_S * (blend - mu)


def reconstruct_nb1162_oof(n_unb: int) -> np.ndarray:
    nb2103_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb730_oof = np.load(DATA_PROCESSED / "nb730_honest_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    blend = (
        NB1162_DEPLOY_WEIGHTS["nb2103_K28"]   * nb2103_oof
        + NB1162_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1162_DEPLOY_WEIGHTS["nb730_honest"] * nb730_oof
        + NB1162_DEPLOY_WEIGHTS["nb503"]        * nb503_oof
        + NB1162_DEPLOY_WEIGHTS["nb562"]        * nb562_oof
    )
    mu = float(blend.mean())
    return mu + NB1162_DEPLOY_S * (blend - mu)


def reconstruct_nb2171_oof(n_unb: int) -> np.ndarray:
    nb2103_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    blend = (
        NB2171_DEPLOY_WEIGHTS["nb2103_K28"]   * nb2103_oof
        + NB2171_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB2171_DEPLOY_WEIGHTS["nb1191"]       * nb1191_oof
        + NB2171_DEPLOY_WEIGHTS["nb503"]        * nb503_oof
        + NB2171_DEPLOY_WEIGHTS["nb562"]        * nb562_oof
    )
    mu = float(blend.mean())
    return mu + NB2171_DEPLOY_S * (blend - mu)


# ----------------------------------------------------------------------------
# per-row dynamic blender
# ----------------------------------------------------------------------------

def per_row_softmin_weights(P: np.ndarray, sigma_global: float) -> np.ndarray:
    """Return (N, K) weight matrix.

    For each row r, weight anchor i by exp(-|p_i(r) - med(r)| / sigma).
    The median is robust to outliers in the consensus.
    """
    med = np.median(P, axis=1, keepdims=True)   # (N, 1)
    dev = np.abs(P - med)                        # (N, K)
    logits = -dev / max(sigma_global, 1e-9)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    w = np.exp(logits)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum > 0, w_sum, 1.0)
    return w / w_sum


def per_row_blend(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (P * W).sum(axis=1)


def scaffold_cv_perrow(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds,
    kf_seed: int,
) -> tuple[float, np.ndarray, float]:
    """For each scaffold fold: estimate sigma_global on training rows
    (cross-row std of the K predictors), then apply per-row dynamic
    blend on the validation rows. Returns (pooled_rae, oof_blend, mean_sigma).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed
    )
    n_unb = P_unb.shape[0]
    oof = np.full(n_unb, np.nan)
    sigmas = []
    for tr_loc, va_loc in splits:
        # sigma_global is RMS of per-row std across the 5 anchors on the train fold
        per_row_std_tr = P_unb[tr_loc].std(axis=1)
        sigma = float(np.sqrt((per_row_std_tr ** 2).mean()))
        sigmas.append(sigma)
        W_va = per_row_softmin_weights(P_unb[va_loc], sigma)
        oof[va_loc] = per_row_blend(P_unb[va_loc], W_va)
    return float(rae(y_unb, oof)), oof, float(np.mean(sigmas))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-row dynamic stack of top-5 PRIMARY candidates")
    print("=" * 78)

    # ---- truth + scaffolds ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_unique_scaf}")

    # ---- 5 candidate OOFs on 253 ----
    print("\n[anchors -- OOFs on 253]")
    oof_nb2240 = np.load(DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy").astype(np.float64)
    oof_nb1162 = reconstruct_nb1162_oof(n_unb)
    oof_nb2171 = reconstruct_nb2171_oof(n_unb)
    oof_nb2095 = reconstruct_nb2095_oof(n_unb)
    oof_nb1191 = reconstruct_nb1191_oof(n_unb)

    anchors = [
        ("nb2240", oof_nb2240, "te_nb2240.npy"),
        ("nb1162", oof_nb1162, "te_nb1162.npy"),
        ("nb2171", oof_nb2171, "te_nb2171.npy"),
        ("nb2095", oof_nb2095, "te_nb2095.npy"),
        ("nb1191", oof_nb1191, "te_nb1191.npy"),
    ]
    indiv_oof_rae = {}
    oof_cols = []
    te_cols = []
    for name, oof, te_fname in anchors:
        assert oof.shape == (n_unb,), f"{name} oof shape {oof.shape}"
        r = float(rae(y_unb, oof))
        indiv_oof_rae[name] = r
        oof_cols.append(oof)
        te_path = DATA_PROCESSED / te_fname
        te_arr = np.load(te_path).astype(np.float64)
        assert te_arr.shape == (n_test,), f"{name} te shape {te_arr.shape}"
        te_cols.append(te_arr)
        print(f"   {name:8s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)   # (253, 5)
    P_te = np.column_stack(te_cols)     # (513, 5)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

    # ---- pairwise correlation diagnostic ----
    corr = np.corrcoef(P_unb.T)
    print("\n[corr] anchor-anchor correlation on 253:")
    names = [a[0] for a in anchors]
    header = "        " + "  ".join(f"{n:>7s}" for n in names)
    print(header)
    for i, n_i in enumerate(names):
        row = "  ".join(f"{corr[i, j]:>7.3f}" for j in range(K_anch))
        print(f"  {n_i:>5s}  {row}")

    # ---- per-row uncertainty (std across 5 predictors) on 253 + 513 ----
    per_row_std_unb = P_unb.std(axis=1)
    per_row_std_te = P_te.std(axis=1)
    sigma_global_unb = float(np.sqrt((per_row_std_unb ** 2).mean()))
    sigma_global_te = float(np.sqrt((per_row_std_te ** 2).mean()))
    print(f"\n[uncertainty] per-row std on 253:  mean={per_row_std_unb.mean():.4f}  "
          f"median={np.median(per_row_std_unb):.4f}  "
          f"min={per_row_std_unb.min():.4f}  max={per_row_std_unb.max():.4f}")
    print(f"[uncertainty] per-row std on 513:  mean={per_row_std_te.mean():.4f}  "
          f"median={np.median(per_row_std_te):.4f}  "
          f"min={per_row_std_te.min():.4f}  max={per_row_std_te.max():.4f}")
    print(f"[uncertainty] sigma_global (RMS)   unb={sigma_global_unb:.4f}  te={sigma_global_te:.4f}")

    # ---- scaffold 5-fold CV across 5 seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  (sigma per fold-train)")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, mean_sigma = scaffold_cv_perrow(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "mean_sigma": float(mean_sigma),
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  mean_sigma={mean_sigma:.4f}")
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_mean_of_seed_oofs = float(rae(y_unb, mean_oof))
    print(f"\n[cv] mean across {len(KF_SEEDS)} seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs            = {rae_mean_of_seed_oofs:.4f}")

    # ---- deploy on 513: refit sigma on all 253 train predictors ----
    print("\n" + "-" * 78)
    print("DEPLOY (sigma fit on all 253)")
    print("-" * 78)
    sigma_deploy = sigma_global_unb
    W_te_deploy = per_row_softmin_weights(P_te, sigma_deploy)
    deploy_te = per_row_blend(P_te, W_te_deploy).astype(np.float32)
    W_unb_deploy = per_row_softmin_weights(P_unb, sigma_deploy)
    deploy_unb = per_row_blend(P_unb, W_unb_deploy)
    in_rae_final = float(rae(y_unb, deploy_unb))
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    mean_anchor_weights_unb = W_unb_deploy.mean(axis=0)
    mean_anchor_weights_te = W_te_deploy.mean(axis=0)
    print(f"   sigma_deploy = {sigma_deploy:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std    = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")
    print(f"   mean anchor weight  (unb): " + ", ".join(
        f"{names[i]}={mean_anchor_weights_unb[i]:.3f}" for i in range(K_anch)))
    print(f"   mean anchor weight  (te) : " + ", ".join(
        f"{names[i]}={mean_anchor_weights_te[i]:.3f}" for i in range(K_anch)))

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    lb_low = lb_band_est - 0.05
    lb_high = lb_band_est + 0.05
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = {lb_band_est:.4f}  "
          f"[{lb_low:.4f}, {lb_high:.4f}]")

    # ---- gate vs nb2240 ----
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    print("\n" + "-" * 78)
    print(f"GATE EVALUATION  (vs nb2240 ref pooled_rae {NB2240_REF_OOF:.4f}, margin {GATE_MARGIN})")
    print("-" * 78)
    print(f"   nb2311 OOF (5-seed mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   nb2240 reference         = {NB2240_REF_OOF:.4f}")
    print(f"   delta                    = {delta_vs_nb2240:+.4f}")
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print(f"   verdict                  = {verdict}")

    # ---- save artefacts ----
    oof_npy_path = DATA_PROCESSED / f"{TAG}_perrow_oof.npy"
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_npy_path, deploy_unb.astype(np.float32))
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {oof_npy_path}")
    print(f"[save] {te_npy_path}")

    summary = {
        "tag": TAG,
        "method": "per_row_dynamic_softmin_top5_PRIMARY",
        "anchors": [a[0] for a in anchors],
        "anchor_te_paths": [a[2] for a in anchors],
        "anchor_oof_rae_unb": indiv_oof_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "anchor_corr_unb": [[float(corr[i, j]) for j in range(K_anch)] for i in range(K_anch)],
        "per_row_std_unb_mean": float(per_row_std_unb.mean()),
        "per_row_std_unb_median": float(np.median(per_row_std_unb)),
        "per_row_std_unb_min": float(per_row_std_unb.min()),
        "per_row_std_unb_max": float(per_row_std_unb.max()),
        "per_row_std_te_mean": float(per_row_std_te.mean()),
        "per_row_std_te_median": float(np.median(per_row_std_te)),
        "per_row_std_te_min": float(per_row_std_te.min()),
        "per_row_std_te_max": float(per_row_std_te.max()),
        "sigma_global_unb": sigma_global_unb,
        "sigma_global_te": sigma_global_te,
        "sigma_deploy": sigma_deploy,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": rae_mean_of_seed_oofs,
        "mean_anchor_weights_unb": [
            {"name": names[i], "w": float(mean_anchor_weights_unb[i])}
            for i in range(K_anch)
        ],
        "mean_anchor_weights_te": [
            {"name": names[i], "w": float(mean_anchor_weights_te[i])}
            for i in range(K_anch)
        ],
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_low,
        "lb_band_high": lb_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "compare_nb2240_oof": NB2240_REF_OOF,
        "delta_vs_nb2240": delta_vs_nb2240,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat),
        "gate_flat_vs_nb2240": bool(gate_flat),
        "verdict_vs_nb2240": verdict,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "oof_npy_path": str(oof_npy_path),
        "te_npy_path": str(te_npy_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled_rae_mean_seeds = {pooled_rae_mean_seeds:.4f}  (+/- {pooled_rae_std_seeds:.4f})")
    print(f"   delta_vs_nb2240       = {delta_vs_nb2240:+.4f}  ({verdict})")
    print(f"   te_unb_rae_in_sample  = {te_unb_rae:.4f}")
    print(f"   LB-band               = {lb_band_est:.4f}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "delta_vs_nb2240",
        "verdict_vs_nb2240",
        "gate_beat_nb2240",
        "sigma_deploy",
        "mean_anchor_weights_unb",
        "lb_band_estimate",
    ):
        print(f"  {k}: {res.get(k)}")
