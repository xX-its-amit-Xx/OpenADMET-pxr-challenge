"""nb2902 -- 2-anchor equal-weight {nb2240_K20, nb1191} mean blend.

NEW PARADIGM:
    Simplest possible 2-anchor mean blend of the two best PRE-clean anchors.
    No SLSQP, no rank-stretch, no per-fold weights -- just an equal-weight
    arithmetic mean of nb2240_K20 and nb1191, evaluated under deterministic
    5-fold scaffold CV on the 253 unblind labels (kf_seed=1001).

ANCHORS:
    nb2240_K20 -- standalone oof_RAE 0.4630 (PRE-clean, K=20 RFE LGBM on
                  chemprop_aux residual; the strongest known single anchor)
    nb1191     -- standalone oof_RAE 0.4647 (PRE-clean post-hoc pyramid
                  blend, deep-30 verified ceiling 0.4718)
    pred = 0.5 * nb2240_oof + 0.5 * nb1191_oof

PROTOCOL:
    * 5-fold scaffold CV on the 253 unblind compounds (single deterministic
      seed kf_seed=1001).
    * Compute fold-wise RAE and pooled RAE on the equal-weight mean (no fit
      step -- weights are fixed at 0.5 / 0.5).
    * Mirror the same equal-weight mean on the 513 deploy te arrays.

GATES:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    otherwise         -> "FAIL"

Outputs:
    scripts/nb2902_k_ensemble_nb2240_nb1191.py
    data/processed/nb2902_summary.json
    data/processed/nb2902_pred_oof.npy   (253-vector, equal-weight on unb)
    data/processed/te_nb2902.npy         (513-vector, equal-weight on te)
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

TAG = "nb2902"

# ---- Anchors ----
ANCHOR_A_NAME = "nb2240_K20"
ANCHOR_A_OOF = "nb2240_mean_bag_oof_K20.npy"
ANCHOR_A_TE = "te_nb2240_K20.npy"

ANCHOR_B_NAME = "nb1191"
ANCHOR_B_OOF = "nb1191_pred_oof.npy"
ANCHOR_B_TE = "te_nb1191.npy"

# Equal weights, fixed (no fit)
W_A = 0.5
W_B = 0.5

# ---- CV protocol ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Gates ----
PROMOTE_THR = 0.4570
MARGINAL_THR = 0.4598


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 2-anchor equal-weight mean blend  "
          f"{{{ANCHOR_A_NAME}, {ANCHOR_B_NAME}}}")
    print("=" * 78)
    print(f"   weights : {ANCHOR_A_NAME}={W_A}  {ANCHOR_B_NAME}={W_B}")
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

    # ---- Equal-weight mean (no fit) ----
    blend_oof = W_A * oof_a + W_B * oof_b   # (253,)
    blend_te = W_A * te_a + W_B * te_b      # (513,)

    pooled_rae = float(rae(y_unb, blend_oof))
    print(f"\n[blend] pooled equal-weight RAE on 253 = {pooled_rae:.4f}")

    # ---- Scaffold-CV fold-wise diagnostics (no fit step; just per-fold RAE) ----
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    fold_results = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        r_va = float(rae(y_unb[va_loc], blend_oof[va_loc]))
        r_a_va = float(rae(y_unb[va_loc], oof_a[va_loc]))
        r_b_va = float(rae(y_unb[va_loc], oof_b[va_loc]))
        fold_results.append({
            "fold": int(fi),
            "n_val": int(len(va_loc)),
            "fold_rae_blend": r_va,
            "fold_rae_anchor_A": r_a_va,
            "fold_rae_anchor_B": r_b_va,
        })
        print(f"   fold {fi}  n_val={len(va_loc):3d}  "
              f"blend={r_va:.4f}  {ANCHOR_A_NAME}={r_a_va:.4f}  "
              f"{ANCHOR_B_NAME}={r_b_va:.4f}")

    fold_arr = np.asarray([r["fold_rae_blend"] for r in fold_results])
    fold_mean = float(fold_arr.mean())
    fold_std = float(fold_arr.std())
    print(f"\n[CV] fold RAE mean +/- std = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"[CV] pooled RAE (single-pass) = {pooled_rae:.4f}")

    # Primary verdict number = pooled RAE on the 253 (deterministic;
    # equal-weight mean is fold-independent for the pooled metric).
    mean_rae = pooled_rae

    # ---- Gate ----
    if mean_rae < PROMOTE_THR:
        verdict = "PROMOTE"
    elif mean_rae < MARGINAL_THR:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  ->  {verdict}")

    # ---- te diagnostic (in-sample on unb_idx slice) ----
    te_unb_rae = float(rae(y_unb, blend_te[unb_idx]))
    print(f"[deploy] te[unb_idx]_RAE (in-sample) = {te_unb_rae:.4f}")
    print(f"[deploy] te mean = {blend_te.mean():.4f}  "
          f"std = {blend_te.std():.4f}")

    # ---- Save artefacts ----
    out_oof = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    out_te = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(out_oof, blend_oof.astype(np.float32))
    np.save(out_te, blend_te.astype(np.float32))
    print(f"\n[save] {out_oof}")
    print(f"[save] {out_te}")

    summary = {
        "tag": TAG,
        "method": "2_anchor_equal_weight_mean_nb2240K20_nb1191",
        "anchors": [
            {"name": ANCHOR_A_NAME, "oof_path": ANCHOR_A_OOF,
             "te_path": ANCHOR_A_TE, "weight": W_A,
             "oof_rae_unb": indiv_oof_rae[ANCHOR_A_NAME]},
            {"name": ANCHOR_B_NAME, "oof_path": ANCHOR_B_OOF,
             "te_path": ANCHOR_B_TE, "weight": W_B,
             "oof_rae_unb": indiv_oof_rae[ANCHOR_B_NAME]},
        ],
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "indiv_oof_rae_unb": indiv_oof_rae,
        "pooled_rae": pooled_rae,
        "fold_results": fold_results,
        "fold_rae_mean": fold_mean,
        "fold_rae_std": fold_std,
        "mean_rae": mean_rae,
        "promote_thr": PROMOTE_THR,
        "marginal_thr": MARGINAL_THR,
        "verdict": verdict,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(blend_te.mean()),
        "deploy_te_std": float(blend_te.std()),
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
    print(f"   weights            = {W_A} / {W_B} (fixed equal mean)")
    print(f"   {ANCHOR_A_NAME} oof_RAE   = {indiv_oof_rae[ANCHOR_A_NAME]:.4f}")
    print(f"   {ANCHOR_B_NAME}      oof_RAE   = {indiv_oof_rae[ANCHOR_B_NAME]:.4f}")
    print(f"   pooled RAE (253)   = {pooled_rae:.4f}")
    print(f"   fold mean +/- std  = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"   gate verdict       = {verdict}")
    print(f"   wall               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "verdict",
        "mean_rae",
        "pooled_rae",
        "fold_rae_mean",
        "fold_rae_std",
        "indiv_oof_rae_unb",
    ):
        print(f"  {k}: {res.get(k)}")
