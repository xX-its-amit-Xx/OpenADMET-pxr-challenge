"""nb2704 -- LightGBM with categorical TARGET ENCODING on Morgan FP-2048 bits.

NEW PARADIGM:
    Convert Morgan FP-2048 BINARY bits to TARGET-ENCODED CONTINUOUS features:
      encoded_bit_i = mean(y_train | bit_i=1) - mean(y_train | bit_i=0)
    Per fold: target encoding fit on fold-train, applied to fold-val (NO leak).
    Top-K=20 bits selected by |encoded_magnitude| -- the 20 bits whose
    presence/absence most shifts the residual mean.

    Hypothesis: dense per-bit continuous signal may split better than the
    raw 0/1 bit when downstream is LGBM tree splits, even though SHAP+LGBM
    on the same bits typically prefers the raw bits.  This is a non-LGBM-
    native FEATURE TRANSFORM axis (not a non-LGBM ranker), so it's
    orthogonal to the nb1162/nb1191/nb2171 K=28-SHAP feature paradigm.

PROTOCOL:
    1. Standardize 513 test SMILES, compute Morgan FP-2048 (radius=2).
    2. Slice X_unb_2048 = X_te_2048[unb_idx] (n=253).
    3. Anchor = te_chemprop_aux[unb_idx]; residual = y_unb - anchor.
    4. Outer 5-fold SCAFFOLD CV on 253 with 5 kf_seeds {1001..1005}:
       per fold:
         (a) target-encode each bit i:
               m1 = mean(residual_tr | bit_i=1) (fall back to 0 if empty)
               m0 = mean(residual_tr | bit_i=0) (fall back to 0 if empty)
               enc_i = m1 - m0
             score_i = |enc_i| * sqrt(min(n_pos, n_neg))  -- magnitude
             weighted by support so single-row bits do not dominate.
         (b) pick top-K=20 bits by score_i.
         (c) X_tr_TE = (bit_i==1)*m1_i + (bit_i==0)*m0_i across top-20
         (d) same applied to fold-val.
         (e) LGBM(seed=0) fit on residual; pred_va = anchor[va] + lgbm(X_va_TE).
       pool pred_oof across 5 folds -> single kf_seed RAE.
    5. Average across 5 kf_seeds.
    6. Deploy te: fit target encoding + bit selection on ALL 253;
       transform X_te_513; refit LGBM on 253 + predict 513.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4598 -> MARGINAL_BEAT
    else              -> FAIL

Outputs:
    data/processed/nb2704_summary.json
    data/processed/nb2704_pred_oof.npy   (253,) float32  (last kf_seed run)
    data/processed/te_nb2704.npy         (513,) float32
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
import lightgbm as lgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2704"

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5
K_TOP = 20
N_BITS = 2048
MORGAN_RADIUS = 2

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2171_REF = 0.4682
NB2604_REF = 0.4580


def _lgbm_params(seed: int):
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


def fit_target_encoding(X_bits: np.ndarray, residual: np.ndarray):
    """Fit per-bit target encoding on residual.

    Returns:
        m1: (n_bits,) per-bit mean residual when bit=1 (NaN -> 0 fallback)
        m0: (n_bits,) per-bit mean residual when bit=0 (NaN -> 0 fallback)
        score: (n_bits,) support-weighted |m1 - m0|
    """
    X = X_bits.astype(np.float32)
    n, n_bits = X.shape
    bit_sum = X.sum(axis=0).astype(np.float64)         # n_pos per bit
    n_neg = (n - bit_sum).astype(np.float64)
    resid = residual.astype(np.float64)
    sum_pos = (X * resid[:, None]).sum(axis=0)
    sum_all = float(resid.sum())
    sum_neg = sum_all - sum_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = np.where(bit_sum > 0, sum_pos / np.maximum(bit_sum, 1e-9), 0.0)
        m0 = np.where(n_neg > 0, sum_neg / np.maximum(n_neg, 1e-9), 0.0)
    enc = m1 - m0
    # support weight: sqrt(min(n_pos, n_neg)) to suppress single-row bits
    support = np.sqrt(np.minimum(bit_sum, n_neg)).astype(np.float64)
    score = np.abs(enc) * support
    return m1.astype(np.float32), m0.astype(np.float32), score.astype(np.float64)


def apply_target_encoding(X_bits: np.ndarray, m1: np.ndarray, m0: np.ndarray,
                          top_idx: np.ndarray) -> np.ndarray:
    """Transform raw 0/1 bits at top_idx to continuous per-bit-encoded values."""
    X = X_bits[:, top_idx].astype(np.float32)
    m1_sel = m1[top_idx][None, :]
    m0_sel = m0[top_idx][None, :]
    return X * m1_sel + (1.0 - X) * m0_sel


def run_one_kf_seed(X_unb_bits, residual, anchor, unb_scaffolds, kf_seed,
                    X_te_bits, te_anchor_513):
    """One outer scaffold-CV pass at a given kf_seed.

    Returns:
        pred_oof: (n_unb,) float64 -- mean-bag pred (anchor + lgbm) per fold
        per_fold_rae: list[float]
    """
    splits = list(scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    ))
    n_unb = X_unb_bits.shape[0]
    pred_oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    truth = anchor + residual
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        m1, m0, score = fit_target_encoding(X_unb_bits[tr_loc], residual[tr_loc])
        top_idx = np.argsort(-score)[:K_TOP]
        X_tr_TE = apply_target_encoding(X_unb_bits[tr_loc], m1, m0, top_idx)
        X_va_TE = apply_target_encoding(X_unb_bits[va_loc], m1, m0, top_idx)
        mdl = lgb.LGBMRegressor(**_lgbm_params(0))
        mdl.fit(X_tr_TE, residual[tr_loc])
        pred_va = anchor[va_loc] + mdl.predict(X_va_TE)
        pred_oof[va_loc] = pred_va
        per_fold_rae.append(float(rae(truth[va_loc], pred_va)))
    return pred_oof, per_fold_rae


def deploy_te(X_unb_bits, residual, X_te_bits, te_anchor_513):
    """Fit TE+top-K+LGBM on full 253; transform+predict 513."""
    m1, m0, score = fit_target_encoding(X_unb_bits, residual)
    top_idx = np.argsort(-score)[:K_TOP]
    X_unb_TE = apply_target_encoding(X_unb_bits, m1, m0, top_idx)
    X_te_TE = apply_target_encoding(X_te_bits, m1, m0, top_idx)
    mdl = lgb.LGBMRegressor(**_lgbm_params(0))
    mdl.fit(X_unb_TE, residual)
    te_resid_513 = mdl.predict(X_te_TE)
    return (te_anchor_513 + te_resid_513).astype(np.float32), top_idx.tolist()


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- target-encoded Morgan FP-2048 -> top-K={K_TOP} LGBM")
    print(f"        anchor=chemprop_aux  residual fit  scaffold {N_FOLDS}-fold CV")
    print(f"        kf_seeds={KF_SEEDS}")
    print(f"        gate: <{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL_BEAT")
    print("=" * 78)

    # Load truth + anchor
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] chemprop_aux te[unb] in_RAE = {rae_anchor:.4f}")

    # Standardize SMILES + compute Morgan FP 2048
    print("\n[fp] computing Morgan FP-2048 (radius=2) for 513 test SMILES ...")
    from rdkit import Chem
    std_smiles = []
    for s in te_smiles:
        m = standardize(s)
        std_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    X_te_bits = morgan_fp_batch(std_smiles, radius=MORGAN_RADIUS, n_bits=N_BITS)
    X_unb_bits = X_te_bits[unb_idx]
    print(f"[fp] X_te_bits={X_te_bits.shape}  X_unb_bits={X_unb_bits.shape}  "
          f"density={X_te_bits.mean():.4f}")

    # Scaffolds for unblind rows
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # Run scaffold CV across kf_seeds
    print("\n" + "-" * 78)
    print(f"running {len(KF_SEEDS)} kf_seeds x {N_FOLDS} folds scaffold-CV")
    print("-" * 78)
    per_kf_rae = []
    per_kf_fold_rae = []
    last_pred_oof = None
    for kf_seed in KF_SEEDS:
        ts = time.time()
        pred_oof, per_fold_rae = run_one_kf_seed(
            X_unb_bits, residual, anchor, unb_scaffolds, kf_seed,
            X_te_bits, te_anchor_513,
        )
        pooled = float(rae(y_unb, pred_oof))
        per_kf_rae.append(pooled)
        per_kf_fold_rae.append(per_fold_rae)
        last_pred_oof = pred_oof
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"per_fold=[{', '.join(f'{r:.4f}' for r in per_fold_rae)}]  "
              f"wall={time.time()-ts:.1f}s")

    mean_rae = float(np.mean(per_kf_rae))
    std_rae = float(np.std(per_kf_rae))
    print(f"\n[summary] mean_RAE (across {len(KF_SEEDS)} kf_seeds) = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")

    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] {verdict}")

    # Deploy te (refit on ALL 253)
    print("\n[deploy] refitting on ALL 253 + transforming 513 ...")
    deploy_te_pred, top_idx_full = deploy_te(
        X_unb_bits, residual, X_te_bits, te_anchor_513,
    )
    te_unb_in = float(rae(y_unb, deploy_te_pred[unb_idx]))
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")
    print(f"   top_idx_full (first 5) = {top_idx_full[:5]}")

    # Save artifacts
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, last_pred_oof.astype(np.float32))
    np.save(te_path, deploy_te_pred.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_target_encoding.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_te_pred.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "target_encoded_morgan_fp2048_topK20_lgbm_residual",
        "paradigm": "feature_transform_continuous_per_bit_target_encoding",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "morgan_radius": MORGAN_RADIUS,
        "n_bits": N_BITS,
        "k_top": K_TOP,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_kf_rae": per_kf_rae,
        "per_kf_fold_rae": per_kf_fold_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "delta_vs_chemprop_aux": mean_rae - CHEMPROP_AUX_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "delta_vs_nb2604": mean_rae - NB2604_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2171_ref": NB2171_REF,
        "nb2604_ref": NB2604_REF,
        "te_mean": float(deploy_te_pred.mean()),
        "te_std": float(deploy_te_pred.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "deploy_top_idx_2048": top_idx_full,
        "pred_oof_path": str(oof_path),
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
    print(f"  mean_RAE = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"  verdict  = {verdict}")
    print(f"  delta vs nb2171 = {mean_rae - NB2171_REF:+.4f}")
    print(f"  delta vs nb2604 = {mean_rae - NB2604_REF:+.4f}")
    print(f"  te[unb_idx] in-RAE = {te_unb_in:.4f}")
    print(f"  wall  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "verdict",
        "delta_vs_nb2171",
        "delta_vs_nb2604",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
