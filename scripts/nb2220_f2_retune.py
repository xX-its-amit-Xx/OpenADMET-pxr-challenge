"""nb2220 -- F2 gate RETUNE with realistic thresholds matching actual data.

Cycle 170 finding (nb2212): The spec thresholds (pec50_null > 5.5,
chemprop_aux > 6.0) NEVER match the data:
  - nb2152_null_hat_te is calibrated on counter-assay distribution centered
    near 3.0, max ~4.0 (so > 5.5 fires 0/513 rows).
  - te_chemprop_aux distribution is right-shifted but rarely exceeds 6.0 on
    the test set; the joint AND of all three conditions is ~0/513.

This script RETUNES thresholds to be empirically grounded:
  - pec50_null  : use the p75 of nb2152_null_hat_te (~3.32)
  - chemprop_aux: use the p80 of te_chemprop_aux  (~5.0 expected)
  - Fire rule (per row r):
        scaf_train_freq[r] == 0  (novel scaffold)
        AND  pec50_null[r] > pct75_null
        AND  chemprop_aux[r] > pct80_aux
  - This should fire on 10-50 rows (vs 0 in nb2212).

For F2 rows, apply:
    shrunk_r = w * nb2171_pred_r + (1 - w) * 4.32   (train_median)
Sweep w in {0.3, 0.5, 0.7, 0.85}.

GATE: 0.003 absolute margin against nb2171 OOF baseline (re-measured on 253).

If a config BEATS the margin, build a deploy CSV applying the same gate to
the full 513 (using te_nb2171, te_chemprop_aux, nb2152_null_hat_te,
scaf_freq_te).

OUTPUTS:
    scripts/nb2220_f2_retune.py
    data/processed/nb2220_summary.json
    submissions/nb2220_f2_retune_<best>.csv   (only if beats margin)
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2220"

# Empirical-percentile thresholds (computed on full 513 te-vectors).
NULL_PCT = 75    # pec50_null > p75 -> ~3.32 vs absurd spec 5.5
ANCHOR_PCT = 80  # chemprop_aux > p80 -> "main predictor calls it a hit"
SCAF_FREQ_THR = 0

W_GRID = [0.30, 0.50, 0.70, 0.85]

NB2171_BASELINE_BRIEF = 0.4682
DECISION_MARGIN = 0.003

# ----------------------------------------------------------------------------
# nb2171 OOF reconstruction (mirrors nb2212 — already validated)
# ----------------------------------------------------------------------------
N_FOLDS = 5
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


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        v = np.load(DATA_PROCESSED / rel).astype(np.float64)
        assert v.shape == (n_unb,), f"{rel} shape {v.shape}"
        cols.append(v)
    return np.column_stack(cols) @ np.asarray(NB1150_SLSQP4_WEIGHTS, np.float64)


def reconstruct_nb1191_oof(n_unb: int) -> np.ndarray:
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
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


def reconstruct_nb2171_oof(P_unb, unb_scaffolds, per_seed):
    n_unb = P_unb.shape[0]
    all_oofs = []
    for seed_rec in per_seed:
        kf_seed = int(seed_rec["kf_seed"])
        fold_s = seed_rec["fold_s"]
        fold_w = seed_rec["fold_w_mean"]
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
    return np.mean(np.column_stack(all_oofs), axis=1)


def describe(name, x):
    return {
        "name": name,
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p80": float(np.percentile(x, 80)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RETUNED F2 gate (realistic empirical thresholds)")
    print(f"           null_pct={NULL_PCT}, anchor_pct={ANCHOR_PCT}, w_grid={W_GRID}")
    print(f"           gate margin {DECISION_MARGIN} vs nb2171 OOF")
    print("=" * 78)

    # ---- Load 513 te-vectors and 253 unblind labels ----
    nb2171_te = np.load(DATA_PROCESSED / "te_nb2171.npy").astype(np.float64)
    aux_te    = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    null_te   = np.load(DATA_PROCESSED / "nb2152_null_hat_te.npy").astype(np.float64)
    unb_idx   = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb     = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_te = nb2171_te.shape[0]
    n_unb = y_unb.shape[0]
    assert n_te == 513 and n_unb == 253
    assert aux_te.shape == (n_te,)
    assert null_te.shape == (n_te,)

    # ---- 1) Distribution stats (on 513 te-vectors) ----
    print("\n[1] DISTRIBUTION STATS (n=513)")
    print("-" * 78)
    null_stats = describe("pec50_null (nb2152_null_hat_te)", null_te)
    aux_stats  = describe("chemprop_aux (te_chemprop_aux)",  aux_te)
    nb2171_stats = describe("nb2171 (te_nb2171)", nb2171_te)
    for d in (null_stats, aux_stats, nb2171_stats):
        print(f"  {d['name']:<40s}  "
              f"mean={d['mean']:.3f}  p75={d['p75']:.3f}  "
              f"p80={d['p80']:.3f}  p90={d['p90']:.3f}  "
              f"p95={d['p95']:.3f}  max={d['max']:.3f}")

    # ---- 2) Realistic empirical thresholds ----
    NULL_THR = null_stats[f"p{NULL_PCT}"]
    ANCHOR_THR = aux_stats[f"p{ANCHOR_PCT}"]
    print(f"\n[2] CHOSEN THRESHOLDS")
    print(f"  NULL_THR   (p{NULL_PCT} of null_te) = {NULL_THR:.4f}")
    print(f"  ANCHOR_THR (p{ANCHOR_PCT} of aux_te)  = {ANCHOR_THR:.4f}")
    print(f"  (vs spec's absurd 5.5 / 6.0)")

    # ---- 3) Test SMILES, scaffolds, train_median ----
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
    # We hard-anchor to 4.32 per spec
    TRAIN_MEDIAN_FALLBACK = 4.32
    print(f"\n[3] train_median observed = {train_median_pec50:.4f}  "
          f"(using spec fallback = {TRAIN_MEDIAN_FALLBACK})")

    print(f"[scaf] computing train Murcko scaffolds (n={len(tr)}) ...")
    tr_scaffolds = set()
    for s in tr[smi_tr_col].astype(str).tolist():
        sc = bemis_murcko(s)
        if sc:
            tr_scaffolds.add(sc)
    print(f"[scaf] unique train scaffolds = {len(tr_scaffolds)}")

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
    print(f"[scaf] novel  unb={n_novel_unb}/{n_unb}  te={n_novel_te}/{n_te}")

    # ---- 4) Reconstruct nb2171 OOF baseline ----
    with open(DATA_PROCESSED / "nb2171_summary.json") as f:
        nb2171_meta = json.load(f)
    per_seed = nb2171_meta["per_seed_results"]
    P_unb = build_anchor_matrix(n_unb)
    nb2171_oof = reconstruct_nb2171_oof(P_unb, unb_scaffolds, per_seed)
    rae_baseline = float(rae(y_unb, nb2171_oof))
    print(f"\n[4] nb2171 OOF baseline (reconstructed) = {rae_baseline:.4f}  "
          f"(reference {nb2171_meta['rae_of_mean_of_seed_oofs']:.4f}, "
          f"brief {NB2171_BASELINE_BRIEF})")

    # ---- 5) F2 fire mask (retuned) ----
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
    print(f"\n[5] F2 fire (retuned)  unb={n_fire_unb}/{n_unb} ({n_fire_unb/n_unb:.1%})  "
          f"te={n_fire_te}/{n_te} ({n_fire_te/n_te:.1%})")
    if n_fire_unb > 0:
        tf = y_unb[fire_unb]
        pf = nb2171_oof[fire_unb]
        print(f"    fire_unb truth mean={tf.mean():.3f}  pred mean={pf.mean():.3f}  "
              f"bias={(pf.mean()-tf.mean()):+.3f}")
        mae_total = float(np.mean(np.abs(y_unb - nb2171_oof)))
        mae_fire = float(np.mean(np.abs(y_unb[fire_unb] - nb2171_oof[fire_unb])))
        print(f"    MAE total={mae_total:.4f}  fire={mae_fire:.4f}")

    # Verify fire-count is in the expected 10-50 band
    in_band = 10 <= n_fire_unb <= 50
    print(f"    fire-count band 10-50: {'YES' if in_band else 'NO'}  "
          f"(actual {n_fire_unb})")

    # ---- 6) Sweep w with train_median fallback (per spec) ----
    print("\n[6] SWEEP train_median=4.32  fire shrink rule")
    print("-" * 78)
    print(f"  {'w':>5s}  {'RAE_oof':>8s}  {'dRAE':>8s}  {'beats?':>7s}")
    results = []
    for w in W_GRID:
        pred = nb2171_oof.copy()
        pred[fire_unb] = (
            w * nb2171_oof[fire_unb] + (1.0 - w) * TRAIN_MEDIAN_FALLBACK
        )
        r = float(rae(y_unb, pred))
        d = r - rae_baseline
        beats = (rae_baseline - r) >= DECISION_MARGIN
        results.append({
            "w": float(w),
            "fallback_value": TRAIN_MEDIAN_FALLBACK,
            "n_fire_unb": n_fire_unb,
            "rae_oof": r,
            "delta_vs_nb2171_oof": d,
            "delta_vs_brief_baseline": r - NB2171_BASELINE_BRIEF,
            "beats_baseline_by_margin": bool(beats),
        })
        print(f"  {w:>5.2f}  {r:>8.4f}  {d:>+8.4f}  {'YES' if beats else 'no':>7s}")

    best = sorted(results, key=lambda x: x["rae_oof"])[0]
    print("\n" + "=" * 78)
    print(f"BEST: w={best['w']:.2f}  RAE={best['rae_oof']:.4f}  "
          f"d={best['delta_vs_nb2171_oof']:+.4f}  "
          f"{'BEATS' if best['beats_baseline_by_margin'] else 'BELOW MARGIN'}")
    print("=" * 78)

    # ---- 7) If beats: build deploy CSV ----
    deploy_csv = None
    if best["beats_baseline_by_margin"]:
        w = best["w"]
        te_pred = nb2171_te.copy()
        te_pred[fire_te] = (
            w * nb2171_te[fire_te] + (1.0 - w) * TRAIN_MEDIAN_FALLBACK
        )
        # CSV: SMILES, Molecule Name, pEC50
        te_df = load_test()
        smi_col2 = "smiles" if "smiles" in te_df.columns else "SMILES"
        mol_col = (
            "Molecule Name" if "Molecule Name" in te_df.columns
            else ("molecule_name" if "molecule_name" in te_df.columns else None)
        )
        assert mol_col is not None, f"no molecule-name col in {te_df.columns.tolist()}"
        out = pd.DataFrame({
            "SMILES": te_df[smi_col2].astype(str).values,
            "Molecule Name": te_df[mol_col].astype(str).values,
            "pEC50": te_pred.astype(np.float64),
        })
        assert len(out) == n_te
        wlabel = f"w{int(round(w*100)):03d}"
        deploy_csv = SUBMISSIONS / f"nb2220_f2_retune_{wlabel}.csv"
        out.to_csv(deploy_csv, index=False)
        print(f"\n[7] deploy CSV written: {deploy_csv}")
        print(f"    fire_te rows shrunk = {n_fire_te}/{n_te}  w={w:.2f}  "
              f"fallback={TRAIN_MEDIAN_FALLBACK}")
    else:
        print(f"\n[7] no CSV: best is below {DECISION_MARGIN} margin")

    # ---- 8) Summary JSON ----
    summary = {
        "tag": TAG,
        "method": (
            "f2_gate_retune_empirical_thresholds_train_median_shrink_on_nb2171_anchor"
        ),
        "rule": {
            "scaf_train_freq_eq": SCAF_FREQ_THR,
            "null_percentile": NULL_PCT,
            "null_threshold_value": float(NULL_THR),
            "anchor_percentile": ANCHOR_PCT,
            "anchor_threshold_value": float(ANCHOR_THR),
            "anchor": "te_chemprop_aux",
            "null_predictor": "nb2152_null_hat_te (PRE-unblind)",
        },
        "fallback_value": TRAIN_MEDIAN_FALLBACK,
        "train_median_observed": train_median_pec50,
        "w_grid": W_GRID,
        "n_te": n_te,
        "n_unb": n_unb,
        "n_novel_scaffold_unb": n_novel_unb,
        "n_novel_scaffold_te": n_novel_te,
        "n_fire_unb": n_fire_unb,
        "n_fire_te": n_fire_te,
        "fire_rate_unb": n_fire_unb / n_unb,
        "fire_rate_te":  n_fire_te / n_te,
        "fire_count_in_expected_band_10_50": bool(in_band),
        "distribution_stats": {
            "null_te": null_stats,
            "aux_te":  aux_stats,
            "nb2171_te": nb2171_stats,
        },
        "nb2171_oof_baseline_rae": rae_baseline,
        "nb2171_brief_baseline_rae": NB2171_BASELINE_BRIEF,
        "decision_margin": DECISION_MARGIN,
        "results": results,
        "best": best,
        "verdict": "BEATS_BY_MARGIN" if best["beats_baseline_by_margin"] else "BELOW_MARGIN",
        "deploy_csv": str(deploy_csv) if deploy_csv else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_json}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  baseline RAE             = {rae_baseline:.4f}")
    print(f"  retuned thr (null/aux)   = {NULL_THR:.3f} / {ANCHOR_THR:.3f}")
    print(f"  fire unb / te            = {n_fire_unb}/{n_unb}  /  {n_fire_te}/{n_te}")
    print(f"  band 10-50 hit           = {in_band}")
    print(f"  best w / RAE / dRAE      = {best['w']:.2f}  {best['rae_oof']:.4f}  "
          f"{best['delta_vs_nb2171_oof']:+.4f}")
    print(f"  verdict                  = {summary['verdict']}")
    print(f"  deploy_csv               = {deploy_csv}")
    print(f"  wall                     = {time.time() - t0:.1f}s")
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
        "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
