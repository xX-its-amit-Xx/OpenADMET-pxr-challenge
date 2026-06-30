"""nb2870 -- James-Stein shrinkage with CORRECTED polarity.

NEW PARADIGM:
    nb2840 used  shrinkage = sigma^2 / (sigma^2 + tau^2)
    which OVER-SHRINKS: it pulls the prediction TOWARD the prior mean
    proportional to noise, but the prior mean is fold-train marginal
    (a bad central estimate when the per-row anchor signal is strong).

    The Stein-admissible posterior-mean estimator under the conjugate
    Gaussian model with prior centred at the chemprop_aux baseline
    and likelihood signal nb2240 is

        pred = chemprop_aux + shrinkage * (nb2240 - chemprop_aux)
        shrinkage = tau^2 / (sigma^2 + tau^2)

    where tau^2 is the SIGNAL variance (how strong nb2240 disagrees
    with chemprop_aux on the fold-train, i.e. signal between the two
    anchors) and sigma^2 is the NOISE variance (residual variance of
    the fold-train).  This shrinks LESS when signal-to-noise is high,
    which is the correct James-Stein polarity.

PROTOCOL (exact spec):
    1. Per fold:
         tau^2   = var(nb2240[tr] - chemprop_aux[tr])
         sigma^2 = var(y[tr] - chemprop_aux[tr])
                   (fold-train residual variance vs chemprop_aux)
         shrink  = tau^2 / (sigma^2 + tau^2)
       Apply on va:
         pred_va = chemprop_aux[va] + shrink * (nb2240[va] - chemprop_aux[va])
    2. 5-fold scaffold CV on 253, kf_seed 1001 (single seed).
    3. Deploy on 513:
         tau^2_full   = var(nb2240[oof] - chemprop_aux[oof])
         sigma^2_full = var(y[oof] - chemprop_aux[oof])
         shrink_full  = tau^2_full / (sigma^2_full + tau^2_full)
         te_pred = chemprop_aux_te + shrink_full
                   * (nb2240_te - chemprop_aux_te)

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else               -> "FAIL"

Outputs:
    data/processed/nb2870_summary.json
    data/processed/nb2870_pred_oof.npy   (253,) float32
    data/processed/te_nb2870.npy         (513,) float32
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2870"

# ---- paths ----
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
ANCHOR_TE_FALLBACK = DATA_PROCESSED / "te_nb2240.npy"

CHEMPROP_OOF = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
CHEMPROP_TE = DATA_PROCESSED / "te_chemprop_aux.npy"

# ---- knobs ----
KF_SEED = 1001
N_FOLDS = 5
EPS = 1e-12

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


def cv_correct_js(
    anchor: np.ndarray,
    chemprop: np.ndarray,
    y: np.ndarray,
    scaffolds: list[str],
    kf_seed: int,
    n_folds: int,
):
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    records = []
    for fold_i, (tr, va) in enumerate(splits):
        # CORRECT polarity per spec
        tau2 = float(np.var(anchor[tr] - chemprop[tr], ddof=0))
        sigma2 = float(np.var(y[tr] - chemprop[tr], ddof=0))
        denom = sigma2 + tau2
        shrink = tau2 / denom if denom > EPS else 0.0
        pred_va = chemprop[va] + shrink * (anchor[va] - chemprop[va])
        oof[va] = pred_va
        records.append({
            "fold": int(fold_i),
            "n_train": int(tr.size),
            "n_val": int(va.size),
            "tau2_tr": tau2,
            "sigma2_tr": sigma2,
            "shrink": shrink,
            "rae_va": float(rae(y[va], pred_va)),
        })
    if np.isnan(oof).any():
        raise RuntimeError("OOF has NaN -- scaffold splits did not cover all rows")
    pooled = float(rae(y, oof))
    return pooled, oof, records


def deploy_correct_js(
    anchor_oof: np.ndarray,
    chemprop_oof: np.ndarray,
    y: np.ndarray,
    anchor_te: np.ndarray,
    chemprop_te: np.ndarray,
):
    tau2 = float(np.var(anchor_oof - chemprop_oof, ddof=0))
    sigma2 = float(np.var(y - chemprop_oof, ddof=0))
    denom = sigma2 + tau2
    shrink = tau2 / denom if denom > EPS else 0.0
    te_pred = chemprop_te + shrink * (anchor_te - chemprop_te)
    diag = {
        "tau2_full": tau2,
        "sigma2_full": sigma2,
        "shrink_full": shrink,
    }
    return te_pred.astype(np.float32), diag


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- James-Stein with CORRECT polarity (fix nb2840)")
    print("=" * 78)

    # ---- truth ----
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- anchors ----
    if not ANCHOR_OOF_PATH.exists():
        raise FileNotFoundError(ANCHOR_OOF_PATH)
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"anchor OOF shape {anchor_oof.shape}, expected ({n_unb},)")
    rae_anchor = float(rae(y_unb, anchor_oof))

    if ANCHOR_TE_PATH.exists():
        anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
        anchor_te_src = str(ANCHOR_TE_PATH)
    elif ANCHOR_TE_FALLBACK.exists():
        anchor_te = np.load(ANCHOR_TE_FALLBACK).astype(np.float64)
        anchor_te_src = str(ANCHOR_TE_FALLBACK)
        print(f"[warn] te_nb2240_K20.npy missing, fallback {ANCHOR_TE_FALLBACK.name}")
    else:
        raise FileNotFoundError(f"No K=20 te found")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"anchor te shape {anchor_te.shape}, expected ({n_test},)")
    print(f"[anchor] nb2240 K=20  oof_RAE={rae_anchor:.4f}  "
          f"oof_std={anchor_oof.std():.3f}  te_std={anchor_te.std():.3f}")

    chemprop_oof = np.load(CHEMPROP_OOF).astype(np.float64)
    chemprop_te = np.load(CHEMPROP_TE).astype(np.float64)
    if chemprop_oof.shape != (n_unb,):
        raise ValueError(f"chemprop oof shape {chemprop_oof.shape}")
    if chemprop_te.shape != (n_test,):
        raise ValueError(f"chemprop te shape {chemprop_te.shape}")
    rae_chemprop = float(rae(y_unb, chemprop_oof))
    print(f"[anchor] chemprop_aux oof_RAE={rae_chemprop:.4f}  "
          f"oof_std={chemprop_oof.std():.3f}  te_std={chemprop_te.std():.3f}")

    # ---- scaffold CV correct-polarity JS ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  kf_seed={KF_SEED}  n_folds={N_FOLDS}")
    print(f"  shrinkage = tau^2 / (sigma^2 + tau^2)  [CORRECT polarity]")
    print(f"  pred = chemprop_aux + shrink * (nb2240 - chemprop_aux)")
    print("-" * 78)
    pooled_rae, mean_oof, records = cv_correct_js(
        anchor_oof, chemprop_oof, y_unb, unb_scaffolds, KF_SEED, N_FOLDS,
    )
    for r in records:
        print(f"   fold={r['fold']}  n_va={r['n_val']:3d}  "
              f"tau2={r['tau2_tr']:.4f}  sigma2={r['sigma2_tr']:.4f}  "
              f"shrink={r['shrink']:.3f}  rae_va={r['rae_va']:.4f}")
    mean_rae = pooled_rae
    std_rae = 0.0
    print(f"\n[cv] pooled_RAE (kf_seed {KF_SEED}) = {mean_rae:.4f}")
    print(f"[cv] mean_oof std = {mean_oof.std():.3f}  "
          f"(anchor {anchor_oof.std():.3f}, truth {y_unb.std():.3f})")

    # ---- gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE}/<{GATE_MARGINAL})  verdict={verdict}")

    # ---- deploy 513 ----
    print("\n[deploy] full-pool shrinkage on 253 -> 513...")
    te_pred, deploy_diag = deploy_correct_js(
        anchor_oof, chemprop_oof, y_unb, anchor_te, chemprop_te,
    )
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"[deploy] tau2={deploy_diag['tau2_full']:.4f}  "
          f"sigma2={deploy_diag['sigma2_full']:.4f}  "
          f"shrink={deploy_diag['shrink_full']:.3f}")
    print(f"[deploy] te_unb_rae(in-sample)={te_unb_rae:.4f}  "
          f"te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "james_stein_correct_polarity_tau2_over_sum",
        "fix_of": "nb2840",
        "polarity_formula": "shrink = tau^2 / (sigma^2 + tau^2)",
        "pred_formula": "chemprop_aux + shrink * (nb2240 - chemprop_aux)",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": anchor_te_src,
        "chemprop_oof_path": str(CHEMPROP_OOF),
        "chemprop_te_path": str(CHEMPROP_TE),
        "anchor_pre_unblind": False,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "rae_anchor_K20_oof": rae_anchor,
        "rae_chemprop_oof": rae_chemprop,
        "anchor_oof_std": float(anchor_oof.std()),
        "anchor_te_std": float(anchor_te.std()),
        "chemprop_oof_std": float(chemprop_oof.std()),
        "chemprop_te_std": float(chemprop_te.std()),
        "truth_mean": float(y_unb.mean()),
        "truth_std": float(y_unb.std()),
        "pooled_rae_outer": mean_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "mean_oof_std": float(mean_oof.std()),
        "fold_records": records,
        "deploy_diag": deploy_diag,
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "gate_promote_below": GATE_PROMOTE,
        "gate_marginal_below": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K=20 anchor in_RAE         = {rae_anchor:.4f}")
    print(f"   chemprop_aux in_RAE        = {rae_chemprop:.4f}")
    print(f"   MEAN RAE (kf_seed {KF_SEED}) = {mean_rae:.4f}")
    print(f"   deploy shrink_full         = {deploy_diag['shrink_full']:.3f}")
    print(f"   te_unb_rae(in-sample)      = {te_unb_rae:.4f}")
    print(f"   gate thresholds            = <{GATE_PROMOTE} PROMOTE | "
          f"<{GATE_MARGINAL} MARGINAL")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_K20_oof",
        "rae_chemprop_oof",
        "mean_rae",
        "std_rae",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
