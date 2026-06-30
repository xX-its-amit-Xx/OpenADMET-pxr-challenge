"""nb2321 -- Chemprop_aux DUPLICATE test: SLSQP orthogonality breakdown.

HYPOTHESIS:
    Adding chemprop_aux TWICE to a SLSQP pyramid (once raw, once
    rank-stretched at s=1.05) tests whether the convex blender
    cleanly zeros out one duplicate due to Pearson ~1.0 correlation,
    or whether numerical drift allows non-trivial weight split.

PYRAMID INPUTS:
    0. nb2240_K20             (K=20 RFE residual mean-bag anchor)
    1. chemprop_aux_raw       (nb1133_chemprop_aux_pred_oof.npy)
    2. chemprop_aux_stretched (raw with mu + 1.05*(p-mu), s=1.05 fixed)
    3. nb503
    4. nb562

PROTOCOL:
    - 5 seeds x 5 scaffold-folds SLSQP + rank-stretch
    - Compare pooled OOF RAE vs nb2240 ref 0.4601 (mean-bag, K=20)
    - Gate margin 0.003
    - Expected: SLSQP should collapse one chemprop_aux copy to ~0
      weight because cor(raw, stretched) ~ 1.0 (rank-monotone affine).

OUTPUTS:
    scripts/nb2321_chemprop_dup.py
    data/processed/nb2321_summary.json
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
from rdkit import RDLogger
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2321"

# ---- gates / config ----
NB2240_REF_OOF = 0.4601  # nb2240 pooled mean (chemprop_aux + K=20 + 5 anchors)
GATE_MARGIN = 0.003
DUP_STRETCH_S = 1.05      # fixed stretch on the duplicate column

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]


# ---- SLSQP + stretch ----
def slsqp_simplex(P, y):
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
            best_r, best_s = r, float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_s


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- chemprop_aux DUPLICATE-ANCHOR SLSQP orthogonality test")
    print("=" * 78)

    # ---- Load truth + scaffolds ----
    te = load_test()
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")

    # ---- Load OOFs ----
    nb2240_oof = np.load(DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy").astype(np.float64)
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)

    # ---- Build stretched copy of chemprop_aux ----
    mu_cp = float(chemprop_oof.mean())
    chemprop_stretched_oof = mu_cp + DUP_STRETCH_S * (chemprop_oof - mu_cp)

    # ---- Pearson between dupes (sanity) ----
    pear_raw_stretched = float(np.corrcoef(chemprop_oof, chemprop_stretched_oof)[0, 1])
    print(f"\n[dupe] Pearson(raw, stretched s={DUP_STRETCH_S}) = {pear_raw_stretched:.6f}")
    print(f"[dupe] (rank-monotone affine -> expect 1.000000)")

    # ---- Verify shapes + report per-anchor RAE ----
    anchors = [
        ("nb2240_K20",            nb2240_oof),
        ("chemprop_aux_raw",      chemprop_oof),
        ("chemprop_aux_stretch",  chemprop_stretched_oof),
        ("nb503",                 nb503_oof),
        ("nb562",                 nb562_oof),
    ]
    indiv_rae = {}
    cols = []
    print("\n[anchors]")
    for disp, oof in anchors:
        assert oof.shape == (n_unb,), f"{disp} shape {oof.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        cols.append(oof)
        print(f"   {disp:24s} oof_RAE={r:.4f}  mean={oof.mean():.3f}  std={oof.std():.3f}")
    P_unb = np.column_stack(cols)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  K={K_anch}")

    # ---- Pearson matrix ----
    pcorr = np.corrcoef(P_unb.T)
    print("\n[corr matrix]")
    names = [a[0] for a in anchors]
    print("              " + "  ".join(f"{n[:10]:>10s}" for n in names))
    for i, ni in enumerate(names):
        row = "  ".join(f"{pcorr[i, j]:10.4f}" for j in range(K_anch))
        print(f"   {ni[:12]:12s}  {row}")

    # ---- 5-fold scaffold CV x 5 seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fs],
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        wm = np.mean(fw, axis=0)
        w_str = "  ".join(f"{disp[:14]:>14s}={wm[i]:.3f}" for i, (disp, _) in enumerate(anchors))
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  s_mean={np.mean(fs):.3f}")
        print(f"      w_mean: {w_str}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))

    # ---- Mean weights across all 25 folds ----
    all_fold_w = np.array([per_seed[i]["fold_w_mean"] for i in range(len(per_seed))])
    overall_w_mean = all_fold_w.mean(axis=0)
    overall_w_std = all_fold_w.std(axis=0)
    raw_idx = 1   # chemprop_aux_raw
    str_idx = 2   # chemprop_aux_stretch
    cp_total = float(overall_w_mean[raw_idx] + overall_w_mean[str_idx])
    cp_raw_share = float(overall_w_mean[raw_idx] / max(cp_total, 1e-12))
    cp_str_share = float(overall_w_mean[str_idx] / max(cp_total, 1e-12))
    one_zeroed = bool(overall_w_mean[raw_idx] < 1e-3 or overall_w_mean[str_idx] < 1e-3)

    print("\n" + "-" * 78)
    print("ORTHOGONALITY DIAGNOSTIC")
    print("-" * 78)
    print(f"   chemprop_aux_raw     w_mean={overall_w_mean[raw_idx]:.4f}  std={overall_w_std[raw_idx]:.4f}")
    print(f"   chemprop_aux_stretch w_mean={overall_w_mean[str_idx]:.4f}  std={overall_w_std[str_idx]:.4f}")
    print(f"   combined_cp_weight   {cp_total:.4f}")
    print(f"   raw_share / str_share = {cp_raw_share:.3f} / {cp_str_share:.3f}")
    print(f"   one_dupe_zeroed (<1e-3) = {one_zeroed}")
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # ---- Gate ----
    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print("\n" + "-" * 78)
    print(f"GATE  (vs nb2240 ref {NB2240_REF_OOF:.4f}, margin {GATE_MARGIN})")
    print("-" * 78)
    print(f"   nb2321 OOF (5-seed mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta                    = {delta_vs_nb2240:+.4f}")
    print(f"   verdict                  = {verdict}")

    summary = {
        "tag": TAG,
        "method": "chemprop_aux_duplicate_anchor_slsqp_test",
        "anchors": [a[0] for a in anchors],
        "anchor_oof_rae": indiv_rae,
        "dupe_stretch_s": DUP_STRETCH_S,
        "pearson_raw_stretched": pear_raw_stretched,
        "pearson_matrix": pcorr.tolist(),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "overall_w_mean": [float(x) for x in overall_w_mean],
        "overall_w_std": [float(x) for x in overall_w_std],
        "cp_raw_w_mean": float(overall_w_mean[raw_idx]),
        "cp_str_w_mean": float(overall_w_mean[str_idx]),
        "cp_combined_w_mean": cp_total,
        "cp_raw_share": cp_raw_share,
        "cp_str_share": cp_str_share,
        "one_dupe_zeroed": one_zeroed,
        "nb2240_ref_oof": NB2240_REF_OOF,
        "delta_vs_nb2240": delta_vs_nb2240,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat),
        "gate_flat_vs_nb2240": bool(gate_flat),
        "verdict_vs_nb2240": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled OOF (5 seeds)       = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2240 (0.4601)   = {delta_vs_nb2240:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   cp_raw / cp_stretch w_mean = {overall_w_mean[raw_idx]:.3f} / {overall_w_mean[str_idx]:.3f}")
    print(f"   one_dupe_zeroed            = {one_zeroed}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "delta_vs_nb2240",
        "verdict_vs_nb2240",
        "cp_raw_w_mean",
        "cp_str_w_mean",
        "one_dupe_zeroed",
        "pearson_raw_stretched",
    ):
        print(f"  {k}: {res.get(k)}")
