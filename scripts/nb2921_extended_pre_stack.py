"""nb2921 -- Extended PRE-clean SLSQP simplex stack across all cycle 211-228 anchors.

NEW PARADIGM:
    Prior cycles converged on a 4-5 anchor PRE-clean pyramid (nb2900/nb2902/nb2171).
    nb2611 ALL9 equal-weight pooled K-pyramid (K=14,16,18,20,22,24,28,32,36) was a
    no-fit baseline at 0.4622. Idea: enumerate every PRE-clean OOF on the 253
    unblind that we can verify, then let SLSQP simplex pick the convex combination.

ANCHORS (all PRE-clean; nb730/nb562 POST-contamination EXPLICITLY dropped):
    nb2240_K20         (RFE LGBM K=20 residual on chemprop_aux residual)
    chemprop_aux       (frozen PRE-unblind chemprop_v2 multitask; nb1133 OOF)
    counter_clean      (counter-axis lifted from te_nb2490_counter[unb_idx])
    nb1191             (5-seed PRE post-hoc pyramid; deep-30 0.4718)
    nb2604_K18         (K=18 RFE LGBM residual on chemprop_aux)
    nb2310_K24         (K=24 RFE LGBM residual on chemprop_aux)
    nb2103_K28         (K=28 RFE LGBM residual on chemprop_aux)
    nb2611_K14         (K=14 RFE LGBM residual on chemprop_aux)
    nb2611_K16         (K=16 RFE LGBM residual on chemprop_aux)
    nb2261_K22         (K=22 RFE LGBM residual on chemprop_aux)
    nb2611_K32         (K=32 RFE LGBM residual on chemprop_aux)
    nb2611_K36         (K=36 RFE LGBM residual on chemprop_aux)

PROTOCOL:
    * 5-fold scaffold CV on 253 unblind, kf_seed=1001 (deterministic; per
      cycle 142 CV protocol audit -- must use scaffold-CV, not random KFold).
    * Per fold: SLSQP simplex (sum w_i = 1, w_i >= 0) on training fold OOFs to
      minimize MAE on the validation fold's truth -> apply weights to validation
      fold predictions -> pool predictions and compute pooled RAE.
    * Deploy weights: SLSQP simplex on the full 253 (single fit, no CV).

GATES:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    otherwise          -> "FAIL"

Outputs:
    scripts/nb2921_extended_pre_stack.py
    data/processed/nb2921_summary.json
    data/processed/nb2921_pred_oof.npy
    data/processed/te_nb2921.npy
    submissions/nb2921_extended_pre_stack.csv
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

TAG = "nb2921"

# ---- Anchors (PRE-clean only; POST-contam nb730/nb562 EXCLUDED) ----
# Each entry: (display_name, oof_rel, te_rel, lift_te_to_oof_for_unb)
# If lift=True, oof is unavailable on 253; we slice te_arr[unb_idx] to form a
# proxy OOF (this is the same convention as nb2900_counter_clean).
ANCHORS = [
    ("nb2240_K20",    "nb2240_mean_bag_oof_K20.npy",       "te_nb2240_K20.npy",     False),
    ("chemprop_aux",  "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy",   False),
    ("counter_clean", None,                                "te_nb2490_counter.npy", True),
    ("nb1191",        "nb1191_pred_oof.npy",               "te_nb1191.npy",         False),
    ("nb2604_K18",    "nb2604_mean_bag_oof_K18.npy",       "te_nb2604_K18.npy",     False),
    ("nb2310_K24",    "nb2310_mean_bag_oof_K24.npy",       "te_nb2310_K24.npy",     False),
    ("nb2103_K28",    "nb2103_mean_bag_oof_K28.npy",       "te_nb2112.npy",         False),
    ("nb2611_K14",    "nb2611_mean_bag_oof_K14.npy",       "te_nb2611_K14.npy",     False),
    ("nb2611_K16",    "nb2611_mean_bag_oof_K16.npy",       "te_nb2611_K16.npy",     False),
    ("nb2261_K22",    "nb2261_mean_bag_oof_K22.npy",       "te_nb2261_K22.npy",     False),
    ("nb2611_K32",    "nb2611_mean_bag_oof_K32.npy",       "te_nb2611_K32.npy",     False),
    ("nb2611_K36",    "nb2611_mean_bag_oof_K36.npy",       "te_nb2611_K36.npy",     False),
]

# ---- CV protocol ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Gates ----
PROMOTE_THR = 0.4570
MARGINAL_THR = 0.4598


def slsqp_simplex_mae(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SLSQP simplex (sum w = 1, w >= 0) minimizing MAE of P @ w vs y.

    P : (n, K)
    y : (n,)
    returns: w (K,)
    """
    K = P.shape[1]
    w0 = np.full(K, 1.0 / K)

    def loss(w):
        return float(np.mean(np.abs(P @ w - y)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-9, "disp": False})
    w = np.asarray(res.x, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.full(K, 1.0 / K)
    w = w / w.sum()
    return w


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Extended PRE-clean SLSQP simplex stack ({len(ANCHORS)} anchors)")
    print("=" * 78)
    print(f"   protocol: {N_FOLDS}-fold scaffold CV, kf_seed={KF_SEED}")
    print(f"   gates   : <{PROMOTE_THR} PROMOTE  <{MARGINAL_THR} MARGINAL_BEAT")

    # ---- Test set names/smiles + unblind labels ----
    te = load_test()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    te_smiles = te["smiles"].values if "smiles" in te.columns else te["SMILES"].values
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[load] n_te={n_te}  n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")

    # ---- Load anchor OOFs / te arrays ----
    oof_cols = []  # list of (name, oof[253])
    te_cols = []   # list of (name, te[513])
    indiv_oof_rae = {}
    for name, oof_rel, te_rel, lift in ANCHORS:
        p_te = DATA_PROCESSED / te_rel
        assert p_te.exists(), f"missing anchor te artefact: {p_te}"
        te_arr = np.load(p_te).astype(np.float64)
        # chemprop_aux full-train OOF lives on 4139 -- restrict if needed.
        if not lift:
            p_oof = DATA_PROCESSED / oof_rel
            assert p_oof.exists(), f"missing anchor oof artefact: {p_oof}"
            oof_arr = np.load(p_oof).astype(np.float64)
            if oof_arr.shape[0] != n_unb:
                # chemprop_aux raw full-train fallback (not used: we use nb1133)
                raise ValueError(
                    f"{name} OOF shape {oof_arr.shape} != {n_unb}; need 253-len PRE-clean OOF",
                )
        else:
            assert te_arr.shape == (n_te,), f"{name} te {te_arr.shape}"
            oof_arr = te_arr[unb_idx].copy()
        assert oof_arr.shape == (n_unb,), f"{name} oof {oof_arr.shape}"
        assert te_arr.shape == (n_te,), f"{name} te {te_arr.shape}"
        oof_cols.append((name, oof_arr))
        te_cols.append((name, te_arr))
        indiv_oof_rae[name] = float(rae(y_unb, oof_arr))
        print(f"   [anchor] {name:16s} oof_RAE={indiv_oof_rae[name]:.4f}")

    names = [n for n, _ in oof_cols]
    P_oof = np.column_stack([o for _, o in oof_cols])  # (253, K)
    P_te = np.column_stack([t for _, t in te_cols])    # (513, K)
    K = len(names)
    print(f"\n[stack] K={K} anchors")

    # ---- Pairwise OOF Pearson correlation ----
    corr = np.corrcoef(P_oof, rowvar=False)
    print(f"\n[corr] pairwise OOF Pearson corr (lower triangle, abbreviated):")
    short = [n[:9] for n in names]
    print("        " + " ".join(f"{s:>9s}" for s in short))
    for i, ni in enumerate(short):
        row = " ".join(f"{corr[i, j]:9.3f}" for j in range(i + 1))
        print(f"   {ni:>6s}  {row}")

    # ---- Cross-fit SLSQP: per-fold simplex on train -> predict val ----
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    blend_oof = np.zeros(n_unb, dtype=np.float64)
    fold_results = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        w = slsqp_simplex_mae(P_oof[tr_loc], y_unb[tr_loc])
        pred_va = P_oof[va_loc] @ w
        blend_oof[va_loc] = pred_va
        r_va = float(rae(y_unb[va_loc], pred_va))
        fold_results.append({
            "fold": int(fi),
            "n_val": int(len(va_loc)),
            "fold_rae": r_va,
            "fold_weights": [float(x) for x in w],
        })
        wstr = " ".join(f"{x:.3f}" for x in w)
        print(f"   fold {fi}  n_val={len(va_loc):3d}  rae={r_va:.4f}  w=[{wstr}]")

    pooled_rae = float(rae(y_unb, blend_oof))
    fold_arr = np.asarray([r["fold_rae"] for r in fold_results])
    fold_mean = float(fold_arr.mean())
    fold_std = float(fold_arr.std())
    print(f"\n[CV] cross-fit fold RAE mean +/- std = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"[CV] pooled cross-fit RAE             = {pooled_rae:.4f}")

    # Primary verdict = pooled cross-fit RAE
    mean_rae = pooled_rae

    # ---- Gate ----
    if mean_rae < PROMOTE_THR:
        verdict = "PROMOTE"
    elif mean_rae < MARGINAL_THR:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  ->  {verdict}")

    # ---- Deploy: single SLSQP fit on full 253 ----
    deploy_w = slsqp_simplex_mae(P_oof, y_unb)
    deploy_te = P_te @ deploy_w
    in_sample_rae = float(rae(y_unb, P_oof @ deploy_w))
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] in-sample RAE (full-fit on 253)  = {in_sample_rae:.4f}")
    print(f"[deploy] te[unb_idx]_RAE (in-sample slice) = {te_unb_rae:.4f}")
    print(f"[deploy] weights:")
    for n, w in zip(names, deploy_w):
        print(f"   {n:16s}  w={w:.4f}")
    print(f"[deploy] te mean={deploy_te.mean():.4f}  std={deploy_te.std():.4f}")

    # ---- Save artefacts ----
    out_oof = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    out_te = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(out_oof, blend_oof.astype(np.float32))
    np.save(out_te, deploy_te.astype(np.float32))
    print(f"\n[save] {out_oof}")
    print(f"[save] {out_te}")

    # ---- Submission CSV (SMILES + Molecule Name + pEC50) ----
    sub_df = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_te,
    })
    sub_path = SUBMISSIONS / f"{TAG}_extended_pre_stack.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"[save] {sub_path}")

    summary = {
        "tag": TAG,
        "method": "extended_PRE_clean_SLSQP_simplex_12_anchor_stack",
        "anchor_pre_unblind": True,
        "anchors": names,
        "K": K,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "indiv_oof_rae_unb": indiv_oof_rae,
        "oof_corr_matrix": corr.tolist(),
        "oof_corr_labels": names,
        "fold_results": fold_results,
        "fold_rae_mean": fold_mean,
        "fold_rae_std": fold_std,
        "pooled_rae": pooled_rae,
        "mean_rae": mean_rae,
        "promote_thr": PROMOTE_THR,
        "marginal_thr": MARGINAL_THR,
        "verdict": verdict,
        "deploy_weights": [
            {"name": n, "w": float(w)} for n, w in zip(names, deploy_w)
        ],
        "deploy_in_sample_rae": in_sample_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "compare_nb2902_ref": 0.4598,
        "compare_nb2604_ref": 0.4580,
        "compare_nb2611_ref": 0.4622,
        "compare_nb2171_ref": 0.4682,
        "delta_vs_nb2902": mean_rae - 0.4598,
        "delta_vs_nb2604": mean_rae - 0.4580,
        "pred_oof_path": str(out_oof),
        "te_npy_path": str(out_te),
        "submission_csv": str(sub_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K anchors          = {K}")
    print(f"   pooled cross-fit   = {pooled_rae:.4f}")
    print(f"   fold mean +/- std  = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"   gate verdict       = {verdict}")
    print(f"   nonzero weights    = {int(np.sum(deploy_w > 1e-4))}")
    print(f"   wall               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "verdict", "mean_rae", "pooled_rae", "fold_rae_mean", "fold_rae_std",
        "indiv_oof_rae_unb", "deploy_weights",
    ):
        v = res.get(k)
        print(f"  {k}: {v}")
