"""nb2880 -- James-Stein shrinkage anchored to nb1191 instead of chemprop_aux.

NEW PARADIGM (per cycle-170 prescription):
    nb2870 used chemprop_aux (deep-30 ~0.55-0.62) as the JS prior anchor and
    failed -- chemprop_aux is too WEAK a prior, so JS shrinkage toward it
    pulls predictions in the wrong direction.  nb1191 (PRE-unblind pyramid,
    deep-30 0.4718) is a much stronger central estimate.  Re-anchor:

        tau^2   = var(nb2240[tr] - nb1191[tr])      (signal variance)
        sigma^2 = var(y[tr]      - nb1191[tr])      (fold-train residual)
        shrink  = tau^2 / (sigma^2 + tau^2)
        pred    = nb1191 + shrink * (nb2240 - nb1191)

PROTOCOL:
    1. Load nb1191_pred_oof if exists; if not, abort gracefully with
       verdict "SKIPPED_NO_ANCHOR_OOF" and write summary stub so the
       cron ladder records the attempt.
    2. 5-fold scaffold CV on 253, kf_seed 1001 (single seed).
    3. Deploy on 513: tau/sigma re-estimated on the full 253 pool.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else               -> "FAIL"

Outputs:
    data/processed/nb2880_summary.json   (always)
    data/processed/nb2880_pred_oof.npy   (253,) float32   (only on success)
    data/processed/te_nb2880.npy         (513,) float32   (only on success)
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

TAG = "nb2880"

# ---- paths ----
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
ANCHOR_TE_FALLBACK = DATA_PROCESSED / "te_nb2240.npy"

PRIOR_OOF = DATA_PROCESSED / "nb1191_pred_oof.npy"
PRIOR_TE = DATA_PROCESSED / "te_nb1191.npy"

# ---- knobs ----
KF_SEED = 1001
N_FOLDS = 5
EPS = 1e-12

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


def cv_js_vs_prior(
    anchor: np.ndarray,
    prior: np.ndarray,
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
        tau2 = float(np.var(anchor[tr] - prior[tr], ddof=0))
        sigma2 = float(np.var(y[tr] - prior[tr], ddof=0))
        denom = sigma2 + tau2
        shrink = tau2 / denom if denom > EPS else 0.0
        pred_va = prior[va] + shrink * (anchor[va] - prior[va])
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


def deploy_js_vs_prior(
    anchor_oof: np.ndarray,
    prior_oof: np.ndarray,
    y: np.ndarray,
    anchor_te: np.ndarray,
    prior_te: np.ndarray,
):
    tau2 = float(np.var(anchor_oof - prior_oof, ddof=0))
    sigma2 = float(np.var(y - prior_oof, ddof=0))
    denom = sigma2 + tau2
    shrink = tau2 / denom if denom > EPS else 0.0
    te_pred = prior_te + shrink * (anchor_te - prior_te)
    diag = {
        "tau2_full": tau2,
        "sigma2_full": sigma2,
        "shrink_full": shrink,
    }
    return te_pred.astype(np.float32), diag


def write_skip_summary(reason: str, missing_path: str, n_unb: int, n_test: int,
                      t0: float):
    summary = {
        "tag": TAG,
        "method": "james_stein_anchored_to_nb1191_instead_of_chemprop_aux",
        "skipped": True,
        "skip_reason": reason,
        "missing_path": missing_path,
        "n_unb": n_unb,
        "n_te": n_test,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "verdict": "SKIPPED_NO_ANCHOR_OOF",
        "gate_promote_below": GATE_PROMOTE,
        "gate_marginal_below": GATE_MARGINAL,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[skip] {reason}")
    print(f"[save] {out_path}")
    return summary


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- James-Stein shrinkage anchored to nb1191 (was chemprop_aux)")
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

    # ---- prior (nb1191) availability gate ----
    if not PRIOR_OOF.exists():
        print(f"[gate] nb1191_pred_oof.npy NOT FOUND at {PRIOR_OOF}")
        print(f"[gate] nb1191 only published te_nb1191.npy (513) -- no 253 OOF.")
        print(f"[gate] cannot fit JS shrinkage without per-fold prior.")
        return write_skip_summary(
            reason="nb1191_pred_oof.npy_does_not_exist",
            missing_path=str(PRIOR_OOF),
            n_unb=n_unb,
            n_test=n_test,
            t0=t0,
        )
    if not PRIOR_TE.exists():
        return write_skip_summary(
            reason="te_nb1191.npy_does_not_exist",
            missing_path=str(PRIOR_TE),
            n_unb=n_unb,
            n_test=n_test,
            t0=t0,
        )

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
        raise FileNotFoundError("No K=20 te found")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"anchor te shape {anchor_te.shape}, expected ({n_test},)")
    print(f"[anchor] nb2240 K=20  oof_RAE={rae_anchor:.4f}  "
          f"oof_std={anchor_oof.std():.3f}  te_std={anchor_te.std():.3f}")

    prior_oof = np.load(PRIOR_OOF).astype(np.float64)
    prior_te = np.load(PRIOR_TE).astype(np.float64)
    if prior_oof.shape != (n_unb,):
        raise ValueError(f"prior oof shape {prior_oof.shape}, expected ({n_unb},)")
    if prior_te.shape != (n_test,):
        raise ValueError(f"prior te shape {prior_te.shape}, expected ({n_test},)")
    rae_prior = float(rae(y_unb, prior_oof))
    print(f"[prior] nb1191 oof_RAE={rae_prior:.4f}  "
          f"oof_std={prior_oof.std():.3f}  te_std={prior_te.std():.3f}")

    # ---- scaffold CV JS vs nb1191 prior ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  kf_seed={KF_SEED}  n_folds={N_FOLDS}")
    print(f"  shrinkage = tau^2 / (sigma^2 + tau^2)")
    print(f"  pred = nb1191 + shrink * (nb2240 - nb1191)")
    print("-" * 78)
    pooled_rae, mean_oof, records = cv_js_vs_prior(
        anchor_oof, prior_oof, y_unb, unb_scaffolds, KF_SEED, N_FOLDS,
    )
    for r in records:
        print(f"   fold={r['fold']}  n_va={r['n_val']:3d}  "
              f"tau2={r['tau2_tr']:.4f}  sigma2={r['sigma2_tr']:.4f}  "
              f"shrink={r['shrink']:.3f}  rae_va={r['rae_va']:.4f}")
    mean_rae = pooled_rae
    std_rae = 0.0
    print(f"\n[cv] pooled_RAE (kf_seed {KF_SEED}) = {mean_rae:.4f}")
    print(f"[cv] mean_oof std = {mean_oof.std():.3f}  "
          f"(anchor {anchor_oof.std():.3f}, prior {prior_oof.std():.3f}, "
          f"truth {y_unb.std():.3f})")

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
    te_pred, deploy_diag = deploy_js_vs_prior(
        anchor_oof, prior_oof, y_unb, anchor_te, prior_te,
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
        "method": "james_stein_anchored_to_nb1191_instead_of_chemprop_aux",
        "fix_of": "nb2870",
        "polarity_formula": "shrink = tau^2 / (sigma^2 + tau^2)",
        "pred_formula": "nb1191 + shrink * (nb2240 - nb1191)",
        "prior_anchor_tag": "nb1191",
        "signal_anchor_tag": "nb2240_K20",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": anchor_te_src,
        "prior_oof_path": str(PRIOR_OOF),
        "prior_te_path": str(PRIOR_TE),
        "anchor_pre_unblind": False,
        "prior_pre_unblind_note": "nb1191_PRE_unblind_pyramid_on_chemprop_aux",
        "skipped": False,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "rae_anchor_K20_oof": rae_anchor,
        "rae_prior_oof": rae_prior,
        "anchor_oof_std": float(anchor_oof.std()),
        "anchor_te_std": float(anchor_te.std()),
        "prior_oof_std": float(prior_oof.std()),
        "prior_te_std": float(prior_te.std()),
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
    print(f"   nb1191 prior in_RAE        = {rae_prior:.4f}")
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
        "skipped",
        "skip_reason",
        "rae_anchor_K20_oof",
        "rae_prior_oof",
        "mean_rae",
        "std_rae",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        if k in res:
            print(f"  {k}: {res.get(k)}")
