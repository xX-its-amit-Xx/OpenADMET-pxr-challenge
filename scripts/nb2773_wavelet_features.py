"""nb2773 -- Wavelet (DWT) features on Morgan FP-2048 + LGBM residual on chemprop_aux.

NEW PARADIGM (vs Mordred / SHAP-K / RFE-K20 substrates):
    Treat each molecule's Morgan FP-2048 as a 1D binary signal, then apply
    a discrete wavelet transform (DWT, Daubechies db2, level=4) to extract
    multi-scale coefficient bands:

        cA4  -- coarse approximation         (~130 coefs)
        cD4  -- coarsest detail              (~130 coefs)
        cD3  -- mid-coarse detail            (~258 coefs)
        cD2  -- mid-fine detail              (~514 coefs)
        cD1  -- finest detail                (~1025 coefs)

    Each band summarises FP bit-pattern structure at a different
    spatial scale.  Approximation coefs capture aggregate bit density
    in coarse windows; detail coefs capture local bit-flip transitions
    (which correspond to specific substructure adjacencies in ECFP4
    bit ordering).  Pick the TOP-50 coefficient indices ranked by
    cross-corpus variance (high variance => most discriminative) to
    form a compact 50-D wavelet feature substrate.

    Hypothesis: multi-scale frequency-domain decomposition of the FP
    bit-vector exposes substructure regularities that axis-aligned LGBM
    splits on the raw 2048-bit space cannot easily reach, providing a
    SUBSTRATE-CHANGE move (cycle-134 sense) on the fixed chemprop_aux
    anchor.  Only model class identical to nb2700 (LGBM); the
    preprocessing pipeline = Morgan -> wavedec -> top-var-50 is the
    new axis being tested.

PROTOCOL:
    1. Compute Morgan FP-2048 per molecule for the unb 253 + test 513.
    2. pywt.wavedec(fp.astype(float), 'db2', level=4) per row ->
       concatenate (cA4, cD4, cD3, cD2, cD1) into a single ~2057-D vector.
    3. Variance ranking on the FIT corpus (unb 253) of the 2057 coefs;
       take TOP-50 indices.  Apply the SAME indices to te 513.
    4. residual = y_unb - chemprop_aux_te[unb_idx]  (only PRE-clean anchor).
    5. LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03) on 50-D
       wavelet features, residual target.
    6. 5-fold scaffold CV (`scaffold_kfold_indices`), 5 kf_seeds
       {1001..1005}.  Per-fold variance ranking is recomputed on the
       fold-train slice to avoid val-set leakage into the index pick.
    7. Deploy: full-unb-fit variance index -> refit LGBM on 253 per seed,
       predict 513 wavelet features; mean-bag aggregate.

GATE (mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2773_wavelet_features.py
    data/processed/nb2773_summary.json
    data/processed/nb2773_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2773.npy         (513,) float32 deploy refit
    submissions/nb2773_wavelet_features.csv
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
import pywt
import lightgbm as lgb

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2773"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Wavelet hyperparameters
WAVELET = "db2"
DWT_LEVEL = 4
FP_BITS = 2048
TOP_K = 50

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630


def _lgbm_params(seed: int) -> dict:
    """LGBM hyperparams identical to nb2700 (only substrate changes)."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _wavedec_row(fp_row: np.ndarray) -> np.ndarray:
    """One molecule's Morgan FP -> concatenated DWT coefficient vector.

    Returns 1D float32 array of length ~2057 (db2 level=4 on 2048 input).
    Order: [cA4, cD4, cD3, cD2, cD1] (coarse -> fine).
    """
    coeffs = pywt.wavedec(fp_row.astype(np.float64), WAVELET, level=DWT_LEVEL)
    return np.concatenate(coeffs).astype(np.float32)


def _wavedec_batch(fp_mat: np.ndarray) -> np.ndarray:
    """Vectorize wavedec over rows of an (N, 2048) FP matrix.

    Returns (N, D) where D ~= 2057 for db2 level=4.
    """
    n = fp_mat.shape[0]
    # one probe to learn D
    probe = _wavedec_row(fp_mat[0])
    D = probe.shape[0]
    out = np.empty((n, D), dtype=np.float32)
    out[0] = probe
    for i in range(1, n):
        out[i] = _wavedec_row(fp_mat[i])
    return out


def _scaffold_cv_one_seed(
    W_unb: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass with PER-FOLD variance ranking on train slice.

    Returns (oof_residual_pred, per_fold_diagnostics).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # Per-fold variance ranking on TRAIN slice only (no val leakage)
        var_tr = W_unb[tr_loc].var(axis=0)
        top_idx = np.argsort(-var_tr)[:TOP_K]
        X_tr = W_unb[tr_loc][:, top_idx]
        X_va = W_unb[va_loc][:, top_idx]
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X_tr, residual[tr_loc])
        oof[va_loc] = mdl.predict(X_va)
        fold_diags.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "top_var_min": float(var_tr[top_idx].min()),
            "top_var_max": float(var_tr[top_idx].max()),
            "top_var_median": float(np.median(var_tr[top_idx])),
        })
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof, fold_diags


def _deploy_te_one_seed(
    W_unb: np.ndarray,
    residual: np.ndarray,
    W_te: np.ndarray,
    top_idx_full: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit LGBM on full 253 (top_idx from full-unb variance); predict 513."""
    X_unb = W_unb[:, top_idx_full]
    X_te = W_te[:, top_idx_full]
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DWT wavelet features on Morgan FP-2048 "
          f"(chemprop_aux residual)")
    print(f"        wavelet={WAVELET}  level={DWT_LEVEL}  FP={FP_BITS}  "
          f"top_K={TOP_K}")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        ref nb2240 K=20 substrate = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load test 513 (for SMILES) + truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"anchor te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Morgan FP-2048 for test 513 + extract unb 253 ----
    print("\n[fp] computing Morgan FP-2048 for test 513 ...")
    fp_te = morgan_fp_batch(test_smiles, radius=2, n_bits=FP_BITS).astype(np.float32)
    assert fp_te.shape == (n_test, FP_BITS), f"fp_te shape {fp_te.shape}"
    fp_unb = fp_te[unb_idx]
    assert fp_unb.shape == (n_unb, FP_BITS)
    print(f"[fp] fp_te={fp_te.shape}  fp_unb={fp_unb.shape}  "
          f"mean_bits_set_unb={fp_unb.sum(axis=1).mean():.1f}")

    # ---- DWT wavelet decomposition ----
    print(f"\n[dwt] wavedec wavelet={WAVELET} level={DWT_LEVEL} ...")
    t1 = time.time()
    W_te = _wavedec_batch(fp_te)
    W_unb = _wavedec_batch(fp_unb)
    print(f"[dwt] W_te={W_te.shape}  W_unb={W_unb.shape}  "
          f"wall={time.time() - t1:.1f}s")
    D_total = W_te.shape[1]
    assert W_te.shape == (n_test, D_total)
    assert W_unb.shape == (n_unb, D_total)
    # Document band sizes
    probe_bands = pywt.wavedec(fp_te[0].astype(np.float64), WAVELET, level=DWT_LEVEL)
    band_sizes = [int(b.shape[0]) for b in probe_bands]
    band_labels = ["cA4", "cD4", "cD3", "cD2", "cD1"]
    print(f"[dwt] band_sizes = {dict(zip(band_labels, band_sizes))}  "
          f"total={D_total}")

    # Top-K variance ranking on FULL unb (for deploy)
    var_full = W_unb.var(axis=0)
    top_idx_full = np.argsort(-var_full)[:TOP_K]
    # Where do the top-K live across bands?
    band_offsets = np.cumsum([0] + band_sizes)
    band_membership_full = {}
    for j, lbl in enumerate(band_labels):
        lo, hi = band_offsets[j], band_offsets[j + 1]
        band_membership_full[lbl] = int(((top_idx_full >= lo) & (top_idx_full < hi)).sum())
    print(f"[dwt] full-unb top-{TOP_K} band membership: {band_membership_full}")

    # ---- Scaffolds ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  K={TOP_K}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_fold_diags: list[list[dict]] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof, fold_diags = _scaffold_cv_one_seed(
            W_unb, residual, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        per_seed_fold_diags.append(fold_diags)
        te_resid = _deploy_te_one_seed(
            W_unb, residual, W_te, top_idx_full, seed,
        )
        per_seed_te_resid[i] = te_resid
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"wall={time.time() - ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_resid = per_seed_oof_resid.mean(axis=0)
    median_bag_resid = np.median(per_seed_oof_resid, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag   RAE = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE = {rae_median_bag:.4f}")
    print(f"[cv] anchor     RAE = {rae_anchor:.4f}  "
          f"(d_mean_bag = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"[cv] reference  nb2240 K=20 = {NB2240_K20_REF:.4f}  "
          f"(d = {rae_mean_bag - NB2240_K20_REF:+.4f})")

    # ---- Deploy te (mean-bag corrected) ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_resid).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_wavelet_features.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_bag_rae        = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "wavelet_DWT_db2_L4_top50_var_LGBM_residual_on_chemprop_aux",
        "rationale": (
            "DWT (db2, level=4) on Morgan FP-2048 -> top-50 variance "
            "coefficients -> LGBM residual on chemprop_aux anchor; "
            "substrate-change on fixed model class"
        ),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "wavelet": WAVELET,
        "dwt_level": DWT_LEVEL,
        "fp_bits": FP_BITS,
        "top_K": TOP_K,
        "band_labels": band_labels,
        "band_sizes": band_sizes,
        "wavedec_dim_total": int(D_total),
        "top_K_band_membership_full_unb": band_membership_full,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "feat_dim": int(TOP_K),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0]),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "delta_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_vs_nb2240_K20": rae_mean_bag - NB2240_K20_REF,
        "nb2240_K20_ref": NB2240_K20_REF,
        "per_seed_fold_diagnostics": per_seed_fold_diags,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20",
        "te_unb_in_sample_rae",
        "top_K_band_membership_full_unb",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
