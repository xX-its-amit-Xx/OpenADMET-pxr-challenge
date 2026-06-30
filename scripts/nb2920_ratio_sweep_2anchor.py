"""nb2920 -- Ratio sweep over 2-anchor {nb2240_K20, nb1191} blend.

NEW PARADIGM:
    nb2902 used equal-weight 0.5/0.5 and reached pooled RAE = 0.4599.
    nb2240_K20 is the stronger standalone anchor (oof_RAE 0.4630 vs 0.4647
    for nb1191), so a w > 0.5 ratio in favour of nb2240_K20 may unlock a
    better blend. This script sweeps the mixing weight across a coarse grid
    and gates on the best mean RAE.

ANCHORS:
    nb2240_K20 -- standalone oof_RAE ~0.4630 (PRE-clean, K=20 RFE LGBM on
                  chemprop_aux residual). Weighted by w.
    nb1191     -- standalone oof_RAE ~0.4647 (PRE-clean post-hoc pyramid
                  blend, deep-30 verified ceiling 0.4718). Weighted by (1-w).
    pred(w) = w * nb2240_oof + (1 - w) * nb1191_oof

PROTOCOL:
    * w-grid: {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.
    * 5-fold scaffold CV on the 253 unblind compounds, single deterministic
      seed kf_seed=1001.
    * For each w report fold-wise RAE on the blended pred plus the pooled
      RAE on the 253 unblind labels.
    * Mirror the best w on the 513 deploy te arrays (best by mean_rae).

GATES (applied on best w):
    best mean_rae < 0.4570 -> "PROMOTE"
    best mean_rae < 0.4598 -> "MARGINAL_BEAT"
    otherwise              -> "FAIL"

Outputs:
    scripts/nb2920_ratio_sweep_2anchor.py
    data/processed/nb2920_summary.json
    data/processed/nb2920_pred_oof.npy   (253-vector, best-w blend on unb)
    data/processed/te_nb2920.npy         (513-vector, best-w blend on te)
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2920"

# ---- Anchors ----
ANCHOR_A_NAME = "nb2240_K20"
ANCHOR_A_OOF = "nb2240_mean_bag_oof_K20.npy"
ANCHOR_A_TE = "te_nb2240_K20.npy"

ANCHOR_B_NAME = "nb1191"
ANCHOR_B_OOF = "nb1191_pred_oof.npy"
ANCHOR_B_TE = "te_nb1191.npy"

# ---- w grid (weight on ANCHOR_A = nb2240_K20) ----
W_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ---- CV protocol ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Gates ----
PROMOTE_THR = 0.4570
MARGINAL_THR = 0.4598


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ratio sweep over 2-anchor blend "
          f"{{{ANCHOR_A_NAME}, {ANCHOR_B_NAME}}}")
    print("=" * 78)
    print(f"   w-grid  : {W_GRID}  (weight on {ANCHOR_A_NAME})")
    print(f"   protocol: {N_FOLDS}-fold scaffold CV, kf_seed={KF_SEED}")
    print(f"   gates   : <{PROMOTE_THR} PROMOTE  <{MARGINAL_THR} MARGINAL_BEAT")

    # ---- Load test set names/smiles and unblind labels ----
    te = load_test()
    te_names = (
        te["name"].values if "name" in te.columns else te["Molecule Name"].values
    )
    te_smiles = (
        te["smiles"].values if "smiles" in te.columns else te["SMILES"].values
    )
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[load] n_te={n_te}  n_unb={n_unb}  "
          f"unique_scaffolds={n_unique_scaf}")

    # ---- Load anchor OOFs/te ----
    p_a_oof = DATA_PROCESSED / ANCHOR_A_OOF
    p_b_oof = DATA_PROCESSED / ANCHOR_B_OOF
    p_a_te = DATA_PROCESSED / ANCHOR_A_TE
    p_b_te = DATA_PROCESSED / ANCHOR_B_TE
    for p in (p_a_oof, p_b_oof, p_a_te, p_b_te):
        assert p.exists(), f"missing anchor artefact: {p}"

    oof_a = np.load(p_a_oof).astype(np.float64)
    oof_b = np.load(p_b_oof).astype(np.float64)
    te_a = np.load(p_a_te).astype(np.float64)
    te_b = np.load(p_b_te).astype(np.float64)
    assert oof_a.shape == (n_unb,), f"{ANCHOR_A_NAME} oof {oof_a.shape}"
    assert oof_b.shape == (n_unb,), f"{ANCHOR_B_NAME} oof {oof_b.shape}"
    assert te_a.shape == (n_te,), f"{ANCHOR_A_NAME} te {te_a.shape}"
    assert te_b.shape == (n_te,), f"{ANCHOR_B_NAME} te {te_b.shape}"

    indiv_oof_rae = {
        ANCHOR_A_NAME: float(rae(y_unb, oof_a)),
        ANCHOR_B_NAME: float(rae(y_unb, oof_b)),
    }
    print(f"\n[anchor] {ANCHOR_A_NAME:12s} oof_RAE={indiv_oof_rae[ANCHOR_A_NAME]:.4f}")
    print(f"[anchor] {ANCHOR_B_NAME:12s} oof_RAE={indiv_oof_rae[ANCHOR_B_NAME]:.4f}")

    # ---- Scaffold-CV splits (deterministic, shared across all w) ----
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    # ---- Sweep ----
    sweep_results = []
    print(f"\n[sweep] running w-grid over {len(W_GRID)} weights ...")
    for w in W_GRID:
        blend_oof = w * oof_a + (1.0 - w) * oof_b   # (253,)
        pooled = float(rae(y_unb, blend_oof))
        fold_raes = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            r_va = float(rae(y_unb[va_loc], blend_oof[va_loc]))
            fold_raes.append(r_va)
        fold_arr = np.asarray(fold_raes)
        fold_mean = float(fold_arr.mean())
        fold_std = float(fold_arr.std())
        sweep_results.append({
            "w": float(w),
            "pooled_rae": pooled,
            "fold_rae_mean": fold_mean,
            "fold_rae_std": fold_std,
            "fold_raes": fold_raes,
        })
        print(f"   w={w:.2f}  pooled={pooled:.4f}  "
              f"fold_mean={fold_mean:.4f}  fold_std={fold_std:.4f}")

    # ---- Best w (by pooled RAE) ----
    best_idx = int(np.argmin([r["pooled_rae"] for r in sweep_results]))
    best = sweep_results[best_idx]
    best_w = best["w"]
    mean_rae = best["pooled_rae"]
    print(f"\n[best] w={best_w:.2f}  pooled_rae={mean_rae:.4f}  "
          f"fold_mean={best['fold_rae_mean']:.4f} +/- {best['fold_rae_std']:.4f}")

    # ---- Gate ----
    if mean_rae < PROMOTE_THR:
        verdict = "PROMOTE"
    elif mean_rae < MARGINAL_THR:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae={mean_rae:.4f}  ->  {verdict}")

    # ---- Best-w blend (oof + te) ----
    blend_oof_best = best_w * oof_a + (1.0 - best_w) * oof_b
    blend_te_best = best_w * te_a + (1.0 - best_w) * te_b
    te_unb_rae = float(rae(y_unb, blend_te_best[unb_idx]))
    print(f"[deploy] te[unb_idx]_RAE (in-sample) = {te_unb_rae:.4f}")
    print(f"[deploy] te mean = {blend_te_best.mean():.4f}  "
          f"std = {blend_te_best.std():.4f}")

    # ---- Save artefacts ----
    out_oof = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    out_te = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(out_oof, blend_oof_best.astype(np.float32))
    np.save(out_te, blend_te_best.astype(np.float32))
    print(f"\n[save] {out_oof}")
    print(f"[save] {out_te}")

    summary = {
        "tag": TAG,
        "method": "2_anchor_ratio_sweep_nb2240K20_nb1191",
        "anchors": [
            {"name": ANCHOR_A_NAME, "oof_path": ANCHOR_A_OOF,
             "te_path": ANCHOR_A_TE,
             "oof_rae_unb": indiv_oof_rae[ANCHOR_A_NAME]},
            {"name": ANCHOR_B_NAME, "oof_path": ANCHOR_B_OOF,
             "te_path": ANCHOR_B_TE,
             "oof_rae_unb": indiv_oof_rae[ANCHOR_B_NAME]},
        ],
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "w_grid": W_GRID,
        "sweep_results": sweep_results,
        "best_w": float(best_w),
        "best_pooled_rae": mean_rae,
        "best_fold_rae_mean": best["fold_rae_mean"],
        "best_fold_rae_std": best["fold_rae_std"],
        "mean_rae": mean_rae,
        "indiv_oof_rae_unb": indiv_oof_rae,
        "promote_thr": PROMOTE_THR,
        "marginal_thr": MARGINAL_THR,
        "verdict": verdict,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(blend_te_best.mean()),
        "deploy_te_std": float(blend_te_best.std()),
        "pred_oof_path": str(out_oof),
        "te_npy_path": str(out_te),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchors            = {ANCHOR_A_NAME} + {ANCHOR_B_NAME}")
    print(f"   w-grid             = {W_GRID}")
    for r in sweep_results:
        marker = " <-- BEST" if r["w"] == best_w else ""
        print(f"   w={r['w']:.2f}  pooled={r['pooled_rae']:.4f}  "
              f"fold_mean={r['fold_rae_mean']:.4f}{marker}")
    print(f"   best w             = {best_w}")
    print(f"   best mean_rae      = {mean_rae:.4f}")
    print(f"   gate verdict       = {verdict}")
    print(f"   wall               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "verdict",
        "best_w",
        "best_pooled_rae",
        "best_fold_rae_mean",
        "best_fold_rae_std",
        "indiv_oof_rae_unb",
    ):
        print(f"  {k}: {res.get(k)}")
