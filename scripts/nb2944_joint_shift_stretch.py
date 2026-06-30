"""nb2944 -- Joint shift+stretch optimization on nb2934.

NEW PARADIGM:
    pred = (mu + shift) + s * (nb2934_pred - mu)
    Sweep (shift, s) jointly via golden-section over a small 2D grid.

    shift  in [-0.2, 0.2]
    s      in [ 0.9, 1.2]

    df = 2 free parameters per fold (shift, s).  Cross-fit per scaffold-fold
    on 253 unblind to keep the OOF prediction honest.

HYPOTHESIS:
    nb2934 5-component equal-weight stack lives at the post-hoc-blend
    ceiling band (cf. cycle-163 deep-30 cluster 0.4718-0.4720, cycle-167
    nb2171 0.4682).  Plain rank-stretch alone has been closed at the
    deep-ensemble level (nb2200 selected s=1.000), but a *joint* (shift,s)
    optimization may extract residual calibration: 5-component equal-weight
    can have residual bias even when ensemble-averaging fixes pure variance
    compression.

PROTOCOL:
    1. Load nb2934 OOF (253,) + te (513,) + truth y_unb (253,).
    2. 5-fold scaffold CV on 253, kf_seed=1001 (matches nb2934).
    3. Per fold:
         * On TRAIN portion of the 252 (n-1 held-out fold), grid-search
           (shift, s) over a coarse mesh then refine with golden-section.
         * Apply best (shift_k, s_k) to VAL portion of pred_base to fill OOF.
    4. Pool OOF across folds, compute pooled RAE.
    5. For te (513): refit (shift, s) on the FULL 253 (single fit) and
       apply to te_base.  (Same single-shot deploy pattern as nb2200/nb562.)

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4585  -> "BETTER_THAN_NB2934"
    else                -> "FAIL"

References:
    nb2934 5-component equal-weight RAE = 0.4585 (incumbent)
    nb2604 4-K equal-weight RAE         = 0.4580
    nb2171 deep-30 PRIMARY-1            = 0.4682
    nb1191 deep-30 PRE-clean            = 0.4718
    chemprop_aux SAFETY-FLOOR           = 0.6216

Outputs:
    scripts/nb2944_joint_shift_stretch.py
    data/processed/nb2944_summary.json
    data/processed/nb2944_pred_oof.npy    (253,) float32
    data/processed/te_nb2944.npy          (513,) float32
    submissions/nb2944_joint_shift_stretch.csv  (on any non-FAIL)
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2944"

# ---- Base anchor ----
BASE_OOF_PATH = DATA_PROCESSED / "nb2934_pred_oof.npy"
BASE_TE_PATH = DATA_PROCESSED / "te_nb2934.npy"

# ---- Eval protocol ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Sweep bounds ----
SHIFT_LO, SHIFT_HI = -0.2, 0.2
S_LO, S_HI = 0.9, 1.2

# ---- Golden-section settings ----
GR = (np.sqrt(5.0) - 1.0) / 2.0   # ~0.618
GS_TOL = 1e-4
GS_MAX_ITER = 60

# ---- Coarse grid for joint warm-start (before golden-section refine) ----
SHIFT_GRID = np.linspace(SHIFT_LO, SHIFT_HI, 9)   # step 0.05
S_GRID = np.linspace(S_LO, S_HI, 7)               # step 0.05

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_BETTER = 0.4585

# ---- References ----
NB2934_REF = 0.4585
NB2604_REF = 0.4580
NB2171_REF = 0.4682
NB1191_REF = 0.4718
CHEMPROP_AUX_REF = 0.6216


def apply_ss(pred: np.ndarray, mu: float, shift: float, s: float) -> np.ndarray:
    """pred = (mu + shift) + s * (pred - mu)."""
    return (mu + shift) + s * (pred - mu)


def fit_ss(y: np.ndarray, pred: np.ndarray, mu: float) -> tuple[float, float, float]:
    """Joint (shift, s) fit via coarse grid + 1D golden-section refines.

    Returns (best_shift, best_s, best_rae).
    """
    # Coarse 2D grid scan
    best_rae = np.inf
    best_shift = 0.0
    best_s = 1.0
    for sh in SHIFT_GRID:
        for ss in S_GRID:
            r = float(rae(y, apply_ss(pred, mu, sh, ss)))
            if r < best_rae:
                best_rae = r
                best_shift = float(sh)
                best_s = float(ss)

    # Alternating golden-section refines (axis at a time, 3 passes)
    for _ in range(3):
        # Refine shift (s fixed)
        a, b = SHIFT_LO, SHIFT_HI
        for _ in range(GS_MAX_ITER):
            if (b - a) < GS_TOL:
                break
            c = b - GR * (b - a)
            d = a + GR * (b - a)
            rc = float(rae(y, apply_ss(pred, mu, c, best_s)))
            rd = float(rae(y, apply_ss(pred, mu, d, best_s)))
            if rc < rd:
                b = d
            else:
                a = c
        new_shift = 0.5 * (a + b)
        new_rae = float(rae(y, apply_ss(pred, mu, new_shift, best_s)))
        if new_rae < best_rae:
            best_rae = new_rae
            best_shift = new_shift

        # Refine s (shift fixed)
        a, b = S_LO, S_HI
        for _ in range(GS_MAX_ITER):
            if (b - a) < GS_TOL:
                break
            c = b - GR * (b - a)
            d = a + GR * (b - a)
            rc = float(rae(y, apply_ss(pred, mu, best_shift, c)))
            rd = float(rae(y, apply_ss(pred, mu, best_shift, d)))
            if rc < rd:
                b = d
            else:
                a = c
        new_s = 0.5 * (a + b)
        new_rae = float(rae(y, apply_ss(pred, mu, best_shift, new_s)))
        if new_rae < best_rae:
            best_rae = new_rae
            best_s = new_s

    return best_shift, best_s, best_rae


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- joint shift+stretch on nb2934")
    print(f"          paradigm: pred = (mu+shift) + s*(base-mu), df=2 per fold")
    print(f"          shift in [{SHIFT_LO},{SHIFT_HI}]   "
          f"s in [{S_LO},{S_HI}]")
    print(f"          refs nb2934={NB2934_REF}  nb2604={NB2604_REF}  "
          f"nb2171={NB2171_REF}")
    print("=" * 78)

    # ---- Load truth + scaffolds ----
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (te_df["smiles"].astype(str).tolist()
                 if "smiles" in te_df.columns
                 else te_df["SMILES"].astype(str).tolist())
    te_names = (te_df["name"].values
                if "name" in te_df.columns
                else te_df["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load nb2934 base ----
    if not BASE_OOF_PATH.exists():
        raise FileNotFoundError(f"missing base OOF: {BASE_OOF_PATH}")
    if not BASE_TE_PATH.exists():
        raise FileNotFoundError(f"missing base te: {BASE_TE_PATH}")
    pred_base_oof = np.load(BASE_OOF_PATH).astype(np.float64)
    pred_base_te = np.load(BASE_TE_PATH).astype(np.float64)
    if pred_base_oof.shape != (n_unb,):
        raise ValueError(f"base oof shape {pred_base_oof.shape} != ({n_unb},)")
    if pred_base_te.shape != (n_test,):
        raise ValueError(f"base te shape {pred_base_te.shape} != ({n_test},)")
    base_rae = float(rae(y_unb, pred_base_oof))
    print(f"[base]  nb2934  oof_RAE={base_rae:.4f}  "
          f"oof_mean={pred_base_oof.mean():.3f}  oof_std={pred_base_oof.std():.3f}")
    print(f"[base]  nb2934  te_mean={pred_base_te.mean():.3f}  "
          f"te_std={pred_base_te.std():.3f}  (truth_std={y_unb.std():.3f})")

    # ---- 5-fold scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-fold scaffold CV  kf_seed={KF_SEED}  (cross-fit (shift,s) per fold)")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        # mu computed on TRAIN portion only (per-fold honest)
        mu_tr = float(pred_base_oof[tr_loc].mean())
        shift_k, s_k, train_rae = fit_ss(
            y_unb[tr_loc], pred_base_oof[tr_loc], mu_tr,
        )
        pred_va = apply_ss(pred_base_oof[va_loc], mu_tr, shift_k, s_k)
        oof_pooled[va_loc] = pred_va
        r_va = float(rae(y_unb[va_loc], pred_va))
        per_fold.append({
            "fold": fi,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "mu_tr": mu_tr,
            "shift": shift_k,
            "s": s_k,
            "train_rae": train_rae,
            "val_rae": r_va,
        })
        print(f"  fold {fi}  n_va={len(va_loc):4d}  mu_tr={mu_tr:.3f}  "
              f"shift={shift_k:+.4f}  s={s_k:.4f}  "
              f"train_RAE={train_rae:.4f}  val_RAE={r_va:.4f}")

    if np.isnan(oof_pooled).any():
        raise RuntimeError("scaffold splits did not cover all rows")
    pooled_rae = float(rae(y_unb, oof_pooled))
    mean_rae = pooled_rae
    per_fold_val_mean = float(np.mean([f["val_rae"] for f in per_fold]))
    per_fold_val_std = float(np.std([f["val_rae"] for f in per_fold]))

    print(f"\n  pooled RAE             = {pooled_rae:.4f}")
    print(f"  per-fold val mean      = {per_fold_val_mean:.4f}")
    print(f"  per-fold val std       = {per_fold_val_std:.4f}")
    print(f"  base RAE (no transform)= {base_rae:.4f}")
    print(f"  delta vs base          = {pooled_rae - base_rae:+.4f}")

    # ---- Deploy fit on full 253 -> apply to te (513) ----
    print("\n" + "-" * 78)
    print("Deploy fit (full 253) -> apply to te (513)")
    print("-" * 78)
    mu_full = float(pred_base_oof.mean())
    shift_d, s_d, train_rae_d = fit_ss(y_unb, pred_base_oof, mu_full)
    pred_oof_unb = apply_ss(pred_base_oof, mu_full, shift_d, s_d)
    pred_te_513 = apply_ss(pred_base_te, mu_full, shift_d, s_d)
    in_sample_rae = float(rae(y_unb, pred_oof_unb))
    te_unb_in = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"  mu_full={mu_full:.4f}  shift={shift_d:+.4f}  s={s_d:.4f}  "
          f"train_RAE={train_rae_d:.4f}")
    print(f"  in_sample_RAE (full)  = {in_sample_rae:.4f}")
    print(f"  te[unb_idx] in_sample = {te_unb_in:.4f}")
    print(f"  te (513) mean={pred_te_513.mean():.3f}  "
          f"std={pred_te_513.std():.3f}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_BETTER:
        verdict = "BETTER_THAN_NB2934"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"(<{GATE_PROMOTE} PROMOTE / <{GATE_BETTER} BETTER) -> {verdict}")
    print(f"  delta vs nb2934 ({NB2934_REF}) = {mean_rae - NB2934_REF:+.4f}")
    print(f"  delta vs nb2604 ({NB2604_REF}) = {mean_rae - NB2604_REF:+.4f}")
    print(f"  delta vs nb2171 ({NB2171_REF}) = {mean_rae - NB2171_REF:+.4f}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_pooled.astype(np.float32))   # honest cross-fit OOF
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"\n[save] {oof_path}  (honest cross-fit OOF)")
    print(f"[save] {te_path}    (deploy-refit te 513)")

    sub_csv = SUBMISSIONS / f"{TAG}_joint_shift_stretch.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te_513.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] verdict=FAIL; no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "joint_shift_stretch_on_nb2934",
        "paradigm": "pred=(mu+shift)+s*(base-mu); df=2 per fold, cross-fit",
        "base": "nb2934",
        "base_oof_path": str(BASE_OOF_PATH),
        "base_te_path": str(BASE_TE_PATH),
        "base_rae": base_rae,
        "shift_bounds": [SHIFT_LO, SHIFT_HI],
        "s_bounds": [S_LO, S_HI],
        "shift_grid_n": int(len(SHIFT_GRID)),
        "s_grid_n": int(len(S_GRID)),
        "gs_tol": GS_TOL,
        "gs_max_iter": GS_MAX_ITER,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_fold": per_fold,
        "per_fold_val_mean_rae": per_fold_val_mean,
        "per_fold_val_std_rae": per_fold_val_std,
        "pooled_rae": pooled_rae,
        "mean_rae": mean_rae,
        "deploy_mu_full": mu_full,
        "deploy_shift": shift_d,
        "deploy_s": s_d,
        "deploy_train_rae": train_rae_d,
        "deploy_in_sample_rae": in_sample_rae,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "delta_vs_base": pooled_rae - base_rae,
        "delta_vs_nb2934": mean_rae - NB2934_REF,
        "delta_vs_nb2604": mean_rae - NB2604_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "delta_vs_nb1191": mean_rae - NB1191_REF,
        "nb2934_ref": NB2934_REF,
        "nb2604_ref": NB2604_REF,
        "nb2171_ref": NB2171_REF,
        "nb1191_ref": NB1191_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   base nb2934 RAE         = {base_rae:.4f}")
    print(f"   pooled RAE (cross-fit)  = {pooled_rae:.4f}")
    print(f"   delta vs nb2934 ref     = {mean_rae - NB2934_REF:+.4f}")
    print(f"   deploy (shift, s)       = ({shift_d:+.4f}, {s_d:.4f})")
    print(f"   in-sample full RAE      = {in_sample_rae:.4f}")
    print(f"   te[unb_idx] in_sample   = {te_unb_in:.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae", "mean_rae", "verdict",
        "delta_vs_nb2934", "delta_vs_nb2604", "delta_vs_nb2171",
        "deploy_shift", "deploy_s", "deploy_in_sample_rae",
        "te_unb_in_sample_rae", "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
