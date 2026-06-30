"""nb2212 -- Counter-assay-gated F2 abstention on nb2171 anchor.

Per pm06: F2 = greasy-novel-inactive over-prediction tail; targets the
-0.11 RAE remaining prize on the F2 cohort.

ABSTENTION RULE (per row r):
    fire_r := (scaf_train_freq[r] == 0)              # novel scaffold
              AND (pec50_null_hat[r] > 5.5)          # counter-assay activity ~promiscuous
              AND (chemprop_aux_te[r] > 6.0)         # main predictor calls it a hit
    if fire_r:  shrunk_r = w * nb2171_pred_r + (1 - w) * fallback
    else:       shrunk_r = nb2171_pred_r

We try TWO fallback flavours per the spec:
    A) fallback = chemprop_aux_te                (per-row, predictor disagreement)
    B) fallback = TRAIN_MEDIAN_PEC50 (= 5.2)    (corpus shrinkage)

Sweep w in {0.3, 0.5, 0.7}.

GATE: 0.003 absolute margin against nb2171 baseline (re-measured here).

INPUTS (all cached; no model refit needed):
    data/processed/te_nb2171.npy                       (513) deploy preds
    data/processed/te_chemprop_aux.npy                 (513) per-row chemprop_aux preds
    data/processed/nb2152_null_hat_te.npy              (513) PRE-unblind counter-assay
    data/processed/nb2171_summary.json                 per-seed fold weights / stretches
    data/processed/_audit_unblind_idx.npy              (253,) test-row indices labeled
    data/processed/_audit_unblind_y.npy                (253,) truth on unblind subset
    data/processed/nb1133_chemprop_aux_pred_oof.npy    (253,) anchor OOFs
    data/processed/nb1158_mean_bag_oof_K32.npy         (253,)
    data/processed/nb2103_mean_bag_oof_K28.npy         (253,)
    data/processed/nb503_pred_oof.npy                  (253,)
    data/processed/nb562_pred_oof.npy                  (253,)
    data/processed/nb1133_nb1014_pred_oof.npy          (253,) (for nb1191 reconstruction)

The nb2171 OOF on 253 is reconstructed from its cached per-seed fold_w_mean +
fold_s (already in nb2171_summary.json) applied to the same 5-anchor OOF
stack. This gives the exact scaffold-CV OOF vector that produced
pooled_rae_mean_seeds = 0.4676 (re-validated here).

OUTPUTS:
    scripts/nb2212_counter_gate.py
    data/processed/nb2212_summary.json
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

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2212"

# Thresholds (per task spec)
SCAF_FREQ_THR = 0           # scaf_train_freq == 0  (novel scaffold)
NULL_THR = 5.5              # pec50_null > 5.5
ANCHOR_THR = 6.0            # chemprop_aux > 6.0
W_GRID = [0.3, 0.5, 0.7]

# Compared baseline (per task brief). Re-measured below from the reconstructed
# nb2171 OOF for honesty.
NB2171_BASELINE_BRIEF = 0.4682
DECISION_MARGIN = 0.003

# Hardcoded train median fallback (recomputed from data; spec hints ~5.2)
TRAIN_MEDIAN_FALLBACK_HINT = 5.2

# ----------------------------------------------------------------------------
# nb2171 reconstruction constants (mirror nb2171_nb1162_anchor_swap.py)
# ----------------------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

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
ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb1191",       "_RECONSTRUCT_nb1191_oof"),
    ("nb503",        "nb503_pred_oof.npy"),
    ("nb562",        "nb562_pred_oof.npy"),
]


# ----------------------------------------------------------------------------
def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        v = np.load(DATA_PROCESSED / rel).astype(np.float64)
        assert v.shape == (n_unb,), f"{rel} shape {v.shape}"
        cols.append(v)
    return np.column_stack(cols) @ np.asarray(NB1150_SLSQP4_WEIGHTS, np.float64)


def reconstruct_nb1191_oof(n_unb: int) -> np.ndarray:
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    nb2112_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


def build_anchor_matrix(n_unb: int) -> np.ndarray:
    cols = []
    for disp, rel in ANCHORS:
        if rel == "_RECONSTRUCT_nb1191_oof":
            cols.append(reconstruct_nb1191_oof(n_unb))
        else:
            v = np.load(DATA_PROCESSED / rel).astype(np.float64)
            assert v.shape == (n_unb,)
            cols.append(v)
    return np.column_stack(cols)


def reconstruct_nb2171_oof(
    P_unb: np.ndarray, unb_scaffolds: list[str], per_seed: list[dict]
) -> tuple[np.ndarray, float]:
    """Replay nb2171's CV: for each (seed, fold), apply the cached fold_w_mean +
    fold_s on the val rows. Mean across seeds -> per-row OOF prediction.
    The pooled RAE matches nb2171_summary.json::rae_of_mean_of_seed_oofs.
    """
    n_unb = P_unb.shape[0]
    all_oofs = []
    for seed_rec in per_seed:
        kf_seed = int(seed_rec["kf_seed"])
        fold_s = seed_rec["fold_s"]            # list[float] length 5
        fold_w = seed_rec["fold_w_mean"]       # list[float] length K=5
        w = np.asarray(fold_w, np.float64)
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan)
        for (tr_loc, va_loc), s in zip(splits, fold_s):
            blend_tr = P_unb[tr_loc] @ w
            mu_tr = float(blend_tr.mean())
            blend_va = P_unb[va_loc] @ w
            oof[va_loc] = mu_tr + float(s) * (blend_va - mu_tr)
        all_oofs.append(oof)
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    return mean_oof, float(np.std([
        rae(np.zeros_like(o), o) for o in all_oofs
    ]))  # std isn't important; we use mean RAE


def apply_abstention(
    base_pred: np.ndarray,
    fire: np.ndarray,
    fallback: np.ndarray,
    w: float,
) -> np.ndarray:
    out = base_pred.copy()
    out[fire] = w * base_pred[fire] + (1.0 - w) * fallback[fire]
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- counter-assay-gated F2 abstention on nb2171 anchor")
    print(f"           rule: scaf_freq==0 AND null>{NULL_THR} AND aux>{ANCHOR_THR}")
    print(f"           sweep w in {W_GRID}, margin {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Inputs on 513 + 253 ----
    nb2171_te = np.load(DATA_PROCESSED / "te_nb2171.npy").astype(np.float64)
    aux_te = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    null_te_path = DATA_PROCESSED / "nb2152_null_hat_te.npy"
    assert null_te_path.exists(), f"missing {null_te_path}"
    null_te = np.load(null_te_path).astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_te = nb2171_te.shape[0]
    n_unb = y_unb.shape[0]
    assert n_te == 513 and n_unb == 253
    assert aux_te.shape == (n_te,)
    assert null_te.shape == (n_te,)
    print(f"[load] te(513)  nb2171:m={nb2171_te.mean():.3f}s={nb2171_te.std():.3f}  "
          f"aux:m={aux_te.mean():.3f}  null:m={null_te.mean():.3f}")

    # ---- Test SMILES / scaffolds, train scaffolds, train median ----
    te = load_test()
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    test_smiles = te[smi_col].astype(str).tolist()
    assert len(test_smiles) == n_te
    unb_smiles = [test_smiles[i] for i in unb_idx]

    tr = load_train()
    pec50_col = "pec50" if "pec50" in tr.columns else "pEC50"
    smi_tr_col = "smiles" if "smiles" in tr.columns else "SMILES"
    tr = tr.dropna(subset=[pec50_col, smi_tr_col]).reset_index(drop=True)
    train_median_pec50 = float(tr[pec50_col].astype(float).median())
    print(f"[train] n={len(tr)}  train_median_pec50 = {train_median_pec50:.4f}  "
          f"(hint was {TRAIN_MEDIAN_FALLBACK_HINT})")

    print(f"[scaf] computing train Murcko scaffolds ({len(tr)}) ...")
    tr_scaffolds = set()
    for s in tr[smi_tr_col].astype(str).tolist():
        sc = bemis_murcko(s)
        if sc:
            tr_scaffolds.add(sc)
    print(f"[scaf] unique train scaffolds = {len(tr_scaffolds)}")

    print(f"[scaf] computing unblind scaffolds ({n_unb}) and test scaffolds ({n_te}) ...")
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    te_scaffolds = [bemis_murcko(s) for s in test_smiles]

    scaf_freq_unb = np.array(
        [0 if (sc is None or sc == "" or sc not in tr_scaffolds) else 1
         for sc in unb_scaffolds], dtype=np.int32,
    )
    scaf_freq_te = np.array(
        [0 if (sc is None or sc == "" or sc not in tr_scaffolds) else 1
         for sc in te_scaffolds], dtype=np.int32,
    )
    n_novel_unb = int((scaf_freq_unb == 0).sum())
    n_novel_te = int((scaf_freq_te == 0).sum())
    print(f"[scaf] novel_scaffold rows  unb={n_novel_unb}/{n_unb} ({n_novel_unb/n_unb:.1%})  "
          f"te={n_novel_te}/{n_te} ({n_novel_te/n_te:.1%})")

    # ---- Reconstruct nb2171 OOF on 253 ----
    with open(DATA_PROCESSED / "nb2171_summary.json") as f:
        nb2171_meta = json.load(f)
    per_seed = nb2171_meta["per_seed_results"]
    P_unb = build_anchor_matrix(n_unb)
    nb2171_oof, _ = reconstruct_nb2171_oof(P_unb, unb_scaffolds, per_seed)
    rae_baseline_oof = float(rae(y_unb, nb2171_oof))
    print(f"\n[base] nb2171 reconstructed OOF RAE = {rae_baseline_oof:.4f}  "
          f"(reference {nb2171_meta['rae_of_mean_of_seed_oofs']:.4f}, "
          f"brief {NB2171_BASELINE_BRIEF})")
    # nb2171 te[unb_idx] in-sample, kept for diagnostics
    rae_baseline_te_unb = float(rae(y_unb, nb2171_te[unb_idx]))
    print(f"[base] nb2171 te[unb_idx] in-sample RAE = {rae_baseline_te_unb:.4f}")

    # ---- F2 cohort on 253 (unblind) and on 513 (full test) ----
    aux_unb = aux_te[unb_idx]
    null_unb = null_te[unb_idx]
    fire_unb = (
        (scaf_freq_unb == SCAF_FREQ_THR)
        & (null_unb > NULL_THR)
        & (aux_unb > ANCHOR_THR)
    )
    fire_te = (
        (scaf_freq_te == SCAF_FREQ_THR)
        & (null_te > NULL_THR)
        & (aux_te > ANCHOR_THR)
    )
    n_fire_unb = int(fire_unb.sum())
    n_fire_te = int(fire_te.sum())
    print(f"\n[F2 cohort] fire_rate  unb={n_fire_unb}/{n_unb} ({n_fire_unb/n_unb:.1%})  "
          f"te={n_fire_te}/{n_te} ({n_fire_te/n_te:.1%})")
    if n_fire_unb > 0:
        truths_fire = y_unb[fire_unb]
        preds_fire = nb2171_oof[fire_unb]
        print(f"[F2 cohort] on unb: truth mean={truths_fire.mean():.3f}  "
              f"pred mean={preds_fire.mean():.3f}  "
              f"bias = pred - truth = {preds_fire.mean() - truths_fire.mean():+.3f}")
        # RAE attributable to fire-rows (MAE share)
        mae_total = float(np.mean(np.abs(y_unb - nb2171_oof)))
        mae_fire  = float(np.mean(np.abs(y_unb[fire_unb] - nb2171_oof[fire_unb])))
        print(f"[F2 cohort] MAE   total={mae_total:.4f}  fire={mae_fire:.4f}  "
              f"share={(fire_unb.sum() * mae_fire) / (n_unb * mae_total):.1%}")

    # ---- Sweep w x fallback ----
    print("\n" + "-" * 78)
    print("SWEEP")
    print("-" * 78)
    fallbacks = {
        "chemprop_aux":   {"unb": aux_unb,                  "te": aux_te},
        "train_median":   {"unb": np.full(n_unb, train_median_pec50),
                            "te": np.full(n_te,  train_median_pec50)},
    }
    print(f"  {'fallback':<14s}  {'w':>5s}  {'n_fire':>6s}  {'RAE_oof':>8s}  "
          f"{'dRAE':>8s}  {'beats?':>7s}")
    results = []
    for fb_name, fb_arrs in fallbacks.items():
        for w in W_GRID:
            pred_oof = apply_abstention(nb2171_oof, fire_unb, fb_arrs["unb"], w)
            r_oof = float(rae(y_unb, pred_oof))
            d = r_oof - rae_baseline_oof
            beats = (rae_baseline_oof - r_oof) >= DECISION_MARGIN
            results.append({
                "fallback": fb_name,
                "w": float(w),
                "n_fire_unb": n_fire_unb,
                "fire_rate_unb": n_fire_unb / n_unb,
                "rae_oof": r_oof,
                "delta_vs_nb2171_oof": d,
                "delta_vs_brief_baseline": r_oof - NB2171_BASELINE_BRIEF,
                "beats_baseline_by_margin": bool(beats),
            })
            print(f"  {fb_name:<14s}  {w:>5.2f}  {n_fire_unb:>6d}  {r_oof:>8.4f}  "
                  f"{d:>+8.4f}  {'YES' if beats else 'no':>7s}")

    # ---- Diagnostic: at empirical null-percentile thresholds ----
    # The spec NULL_THR=5.5 fires 0 rows on this data: nb2152 outputs are
    # calibrated to a distribution centered at 3.0 (max 4.0). Report a
    # scale-appropriate diagnostic sweep at p75/p90/p95 of null_te.
    print("\n" + "-" * 78)
    print("DIAGNOSTIC: empirical null thresholds (p75/p90/p95)")
    print("-" * 78)
    diag_thrs = {
        "p75":  float(np.percentile(null_te, 75)),
        "p90":  float(np.percentile(null_te, 90)),
        "p95":  float(np.percentile(null_te, 95)),
    }
    print(f"  null_te percentiles: " + "  ".join(f"{k}={v:.3f}" for k, v in diag_thrs.items()))
    diag_results = []
    for thr_name, thr_val in diag_thrs.items():
        fire_d_unb = (
            (scaf_freq_unb == SCAF_FREQ_THR)
            & (null_unb > thr_val)
            & (aux_unb > ANCHOR_THR)
        )
        fire_d_te = (
            (scaf_freq_te == SCAF_FREQ_THR)
            & (null_te > thr_val)
            & (aux_te > ANCHOR_THR)
        )
        n_d_unb = int(fire_d_unb.sum())
        n_d_te = int(fire_d_te.sum())
        for fb_name, fb_arrs in fallbacks.items():
            for w in W_GRID:
                pred = apply_abstention(nb2171_oof, fire_d_unb, fb_arrs["unb"], w)
                r = float(rae(y_unb, pred))
                d = r - rae_baseline_oof
                beats = (rae_baseline_oof - r) >= DECISION_MARGIN
                diag_results.append({
                    "null_thr_name": thr_name,
                    "null_thr_value": thr_val,
                    "fallback": fb_name,
                    "w": float(w),
                    "n_fire_unb": n_d_unb,
                    "fire_rate_unb": n_d_unb / n_unb,
                    "n_fire_te": n_d_te,
                    "fire_rate_te": n_d_te / n_te,
                    "rae_oof": r,
                    "delta_vs_nb2171_oof": d,
                    "beats_baseline_by_margin": bool(beats),
                })
        # one-line print per threshold's best
        thr_rows = [r for r in diag_results if r["null_thr_name"] == thr_name]
        thr_best = sorted(thr_rows, key=lambda x: x["rae_oof"])[0]
        print(f"  {thr_name} thr={thr_val:.3f}  fire_unb={n_d_unb}/{n_unb} "
              f"({n_d_unb/n_unb:.1%})  best: fb={thr_best['fallback']} "
              f"w={thr_best['w']:.2f}  RAE={thr_best['rae_oof']:.4f}  "
              f"d={thr_best['delta_vs_nb2171_oof']:+.4f}  "
              f"{'BEATS' if thr_best['beats_baseline_by_margin'] else 'no'}")

    # ---- Best variant (across spec rule only) ----
    best = sorted(results, key=lambda x: x["rae_oof"])[0]
    diag_best = sorted(diag_results, key=lambda x: x["rae_oof"])[0]
    print("\n" + "=" * 78)
    print("BEST VARIANT")
    print("=" * 78)
    print(f"  fallback     = {best['fallback']}")
    print(f"  w            = {best['w']:.2f}")
    print(f"  RAE_oof      = {best['rae_oof']:.4f}")
    print(f"  baseline     = {rae_baseline_oof:.4f}  (nb2171 OOF, recon)")
    print(f"  delta        = {best['delta_vs_nb2171_oof']:+.4f}  "
          f"(margin {DECISION_MARGIN})")
    print(f"  verdict      = {'BEATS_BY_MARGIN' if best['beats_baseline_by_margin'] else 'BELOW_MARGIN'}")

    # ---- Summary JSON ----
    summary = {
        "tag": TAG,
        "method": "counter_assay_gated_F2_abstention_on_nb2171_anchor",
        "rule": {
            "scaf_train_freq_eq": SCAF_FREQ_THR,
            "null_threshold_gt": NULL_THR,
            "anchor_threshold_gt": ANCHOR_THR,
            "anchor": "chemprop_aux_te",
            "null_predictor": "nb2152_null_hat_te (PRE-unblind)",
        },
        "w_grid": W_GRID,
        "fallbacks": ["chemprop_aux", "train_median"],
        "n_te": n_te,
        "n_unb": n_unb,
        "n_novel_scaffold_unb": n_novel_unb,
        "n_novel_scaffold_te": n_novel_te,
        "n_fire_unb": n_fire_unb,
        "n_fire_te": n_fire_te,
        "fire_rate_unb": n_fire_unb / n_unb,
        "fire_rate_te":  n_fire_te / n_te,
        "train_median_pec50": train_median_pec50,
        "nb2171_oof_baseline_rae": rae_baseline_oof,
        "nb2171_brief_baseline_rae": NB2171_BASELINE_BRIEF,
        "nb2171_te_unb_in_sample_rae": rae_baseline_te_unb,
        "decision_margin": DECISION_MARGIN,
        "results": results,
        "best": best,
        "verdict": "BEATS_BY_MARGIN" if best["beats_baseline_by_margin"] else "BELOW_MARGIN",
        "diagnostic_null_percentile_thresholds": diag_thrs,
        "diagnostic_results": diag_results,
        "diagnostic_best": diag_best,
        "diagnostic_verdict": (
            "BEATS_BY_MARGIN" if diag_best["beats_baseline_by_margin"] else "BELOW_MARGIN"
        ),
        "note": (
            "Spec NULL_THR=5.5 fires 0/253 rows because nb2152 null_hat output "
            "is calibrated on a counter-assay pEC50 distribution centered at "
            "3.0 (max ~4.0). The diagnostic_* block scales NULL_THR to the "
            "empirical p75/p90/p95 of null_te to surface the rule's potential "
            "if calibrated to the underlying null distribution."
        ),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_json}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   baseline RAE (nb2171 OOF recon) = {rae_baseline_oof:.4f}")
    print(f"   F2 fire rate unb / te          = {n_fire_unb/n_unb:.1%} / {n_fire_te/n_te:.1%}")
    print(f"   best fallback                   = {best['fallback']}")
    print(f"   best w                          = {best['w']:.2f}")
    print(f"   best RAE                        = {best['rae_oof']:.4f}")
    print(f"   delta                           = {best['delta_vs_nb2171_oof']:+.4f}")
    print(f"   verdict                         = {summary['verdict']}")
    print(f"   diag best (emp thr) RAE         = {diag_best['rae_oof']:.4f}  "
          f"({diag_best['null_thr_name']}={diag_best['null_thr_value']:.3f}  "
          f"fb={diag_best['fallback']} w={diag_best['w']:.2f})  "
          f"d={diag_best['delta_vs_nb2171_oof']:+.4f}  "
          f"{summary['diagnostic_verdict']}")
    print(f"   wall                            = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "nb2171_oof_baseline_rae",
        "n_fire_unb",
        "fire_rate_unb",
        "n_fire_te",
        "fire_rate_te",
        "best",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
