"""nb550 -- FAILURE-TAIL DIAGNOSTIC

Load nb503 honest cross-fit OOF (253 unblind), compute |truth - pred|,
take top-50 worst rows, and characterise them on:
  - truth/pred distributions (hits vs misses; under/over)
  - Murcko scaffold + scaffold frequency in train
  - physchem (MW, logP, TPSA, fsp3, rotbonds, charge)
  - z-scores vs train mean
  - Tanimoto top-1 + n_train_neighbors at sim>=0.40
  - chemprop disagreement (std across {nb492, nb500, nb501, nb502, chemprop_aux})

Saves table -> data/processed/nb550_failure_tail.csv
Prints top-3 insights about WHY the model fails on these rows.

Memory-safe: 253 unblind x 4139 train = ~1M Tanimoto entries (uint8 ops).
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from pxr.chem import (
    bemis_murcko,
    compute_physchem,
    morgan_fp_batch,
    standardize_smiles,
)
from pxr.paths import DATA_PROCESSED, DATA_RAW

TOP_N = 50
SIM_THR = 0.40


def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Tanimoto for two uint8 bit matrices, returns (nA, nB) float32."""
    A = A.astype(np.uint8)
    B = B.astype(np.uint8)
    # popcounts
    a_cnt = A.sum(axis=1, dtype=np.int32)  # (nA,)
    b_cnt = B.sum(axis=1, dtype=np.int32)  # (nB,)
    inter = A @ B.T  # (nA, nB) int32
    union = a_cnt[:, None] + b_cnt[None, :] - inter
    out = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return out.astype(np.float32)


def main() -> dict:
    print("=" * 78)
    print("nb550 -- FAILURE-TAIL DIAGNOSTIC  (nb503 cross-fit, top-50 worst)")
    print("=" * 78)

    # ---- Required artefacts ----
    needed = {
        "TEST":      DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "TRAIN":     DATA_RAW / "pxr-challenge_TRAIN.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb503_oof": DATA_PROCESSED / "nb503_pred_oof.npy",
    }
    # Optional disagreement inputs
    disagreement_tags = {
        "nb492": DATA_PROCESSED / "nb492_pred_oof.npy",
        "nb500": DATA_PROCESSED / "nb500_pred_oof.npy",
        "nb501": DATA_PROCESSED / "nb501_pred_oof.npy",
        "nb502": DATA_PROCESSED / "nb502_pred_oof.npy",
        # chemprop_aux is on full 513 test rows; we'll index by unb_idx
        "chemprop_aux_te": DATA_PROCESSED / "te_chemprop_aux.npy",
    }
    miss = [k for k, p in needed.items() if not Path(p).exists()]
    if miss:
        print("MISSING required:", miss)
        return {"success": False, "missing": miss}

    te_df = pd.read_csv(needed["TEST"])
    tr_df = pd.read_csv(needed["TRAIN"])
    unb_df = pd.read_csv(needed["UNBLINDED"])

    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"].tolist())}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int
    )
    truth = unb_df["pEC50"].astype(float).values
    pred = np.load(needed["nb503_oof"]).astype(np.float64)
    assert pred.shape == truth.shape, f"shape mismatch {pred.shape} vs {truth.shape}"
    n_unb = len(truth)
    print(f"Loaded {n_unb} unblind rows; nb503 pred shape {pred.shape}")

    err = np.abs(truth - pred)
    order = np.argsort(-err)
    top = order[:TOP_N]
    print(f"\nTop-{TOP_N} worst rows: error range "
          f"[{err[top].min():.3f}, {err[top].max():.3f}]  "
          f"mean={err[top].mean():.3f}")

    # ---- Names / SMILES for the top-50 ----
    top_names = unb_df["Molecule Name"].iloc[top].tolist()
    # Look up SMILES from TEST_BLINDED (authoritative); fall back to UNBLINDED
    smi_te = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))
    top_smiles = [smi_te[n] for n in top_names]

    # ---- Standardise SMILES for top-50 + all training ----
    print("\nStandardising SMILES...")
    top_std = [standardize_smiles(s) or s for s in top_smiles]
    tr_smiles = tr_df["SMILES"].astype(str).tolist()
    tr_std = [standardize_smiles(s) or s for s in tr_smiles]

    # ---- Scaffolds ----
    print("Computing scaffolds...")
    top_scaf = [bemis_murcko(s) or "" for s in top_std]
    tr_scaf = [bemis_murcko(s) or "" for s in tr_std]
    from collections import Counter
    tr_scaf_freq = Counter(tr_scaf)
    top_scaf_freq = [int(tr_scaf_freq.get(sc, 0)) for sc in top_scaf]

    # ---- Physchem ----
    print("Computing physchem...")
    PC_KEYS = ["mw", "logp", "tpsa", "fsp3", "rotbonds", "formal_charge"]
    top_pc = []
    for s in top_std:
        d = compute_physchem(s) or {k: np.nan for k in PC_KEYS}
        top_pc.append([d.get(k, np.nan) for k in PC_KEYS])
    top_pc = np.array(top_pc, dtype=float)  # (50, 6)

    tr_pc = []
    for s in tr_std:
        d = compute_physchem(s) or {k: np.nan for k in PC_KEYS}
        tr_pc.append([d.get(k, np.nan) for k in PC_KEYS])
    tr_pc = np.array(tr_pc, dtype=float)  # (4139, 6)
    tr_mean = np.nanmean(tr_pc, axis=0)
    tr_std_pc = np.nanstd(tr_pc, axis=0) + 1e-9
    top_pc_z = (top_pc - tr_mean) / tr_std_pc  # (50, 6)
    max_abs_z = np.nanmax(np.abs(top_pc_z), axis=1)  # extremity score

    # ---- Tanimoto top-1 + n_neighbors >= 0.40 (memory-safe chunks) ----
    print("Computing Tanimoto (50 x 4139)...")
    top_fp = morgan_fp_batch(top_std)        # (50, 2048) uint8
    tr_fp = morgan_fp_batch(tr_std)          # (4139, 2048) uint8
    sim = tanimoto_matrix(top_fp, tr_fp)     # (50, 4139) float32
    top1_sim = sim.max(axis=1)
    n_neigh_40 = (sim >= SIM_THR).sum(axis=1)

    # ---- Chemprop disagreement (std across candidate predictors) ----
    candidate_oofs: list[np.ndarray] = []
    cand_used: list[str] = []
    for tag, p in disagreement_tags.items():
        if not Path(p).exists():
            continue
        arr = np.load(p).astype(np.float64)
        if arr.shape == (n_unb,):
            candidate_oofs.append(arr)
            cand_used.append(tag)
        elif arr.shape == (len(te_df),):
            candidate_oofs.append(arr[unb_idx])
            cand_used.append(tag)
        else:
            pass  # skip unusable shapes
    if candidate_oofs:
        stack = np.stack(candidate_oofs, axis=0)  # (k, n_unb)
        disagreement_all = stack.std(axis=0)      # (n_unb,)
    else:
        disagreement_all = np.zeros(n_unb)
    disagreement_top = disagreement_all[top]
    print(f"Disagreement components used: {cand_used}")

    # ---- Build output table ----
    out = pd.DataFrame({
        "Molecule Name": top_names,
        "SMILES": top_smiles,
        "truth_pec50": truth[top],
        "nb503_pred":  pred[top],
        "error":       err[top],
        "scaffold":    top_scaf,
        "scaf_train_freq": top_scaf_freq,
        "mw":         top_pc[:, 0],
        "logp":       top_pc[:, 1],
        "tpsa":       top_pc[:, 2],
        "fsp3":       top_pc[:, 3],
        "rotbonds":   top_pc[:, 4],
        "formal_charge": top_pc[:, 5],
        "max_abs_z":  max_abs_z,
        "top1_sim_train":  top1_sim,
        "n_neighbors_sim40": n_neigh_40,
        "predictor_disagreement": disagreement_top,
    })
    out_path = DATA_PROCESSED / "nb550_failure_tail.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # =================================================================
    # Reporting
    # =================================================================
    print("\n" + "-" * 78)
    print("DISTRIBUTIONS  (top-50 worst rows)")
    print("-" * 78)
    t = truth[top]
    p = pred[top]
    resid = t - p  # +ve => under-prediction
    print(f"truth pec50:  mean={t.mean():.3f}  std={t.std():.3f}  "
          f"min={t.min():.3f}  max={t.max():.3f}")
    print(f"  truth quartiles: {np.percentile(t, [25, 50, 75]).round(3).tolist()}")
    n_hits = int((t >= 6.0).sum())
    n_lo = int((t < 5.0).sum())
    print(f"  truth >= 6.0 (hits):     {n_hits}/{TOP_N}  ({n_hits / TOP_N:.0%})")
    print(f"  truth <  5.0 (low):      {n_lo}/{TOP_N}  ({n_lo / TOP_N:.0%})")
    print(f"\nnb503 pred:   mean={p.mean():.3f}  std={p.std():.3f}  "
          f"min={p.min():.3f}  max={p.max():.3f}")
    print(f"residual (truth-pred):  mean={resid.mean():+.3f}  "
          f"std={resid.std():.3f}")
    n_under = int((resid > 0).sum())
    n_over  = int((resid < 0).sum())
    print(f"  under-predicted (resid>0): {n_under}/{TOP_N}  ({n_under / TOP_N:.0%})")
    print(f"  over-predicted  (resid<0): {n_over}/{TOP_N}  ({n_over  / TOP_N:.0%})")

    # truth distribution of full 253 for reference
    t_all = truth
    print("\nFor comparison, FULL 253 unblind:")
    print(f"  truth: mean={t_all.mean():.3f}  std={t_all.std():.3f}  "
          f"hits>=6: {(t_all >= 6.0).sum()}/{n_unb}")

    print("\n" + "-" * 78)
    print("SCAFFOLD DIVERSITY  (top-50)")
    print("-" * 78)
    sc_cnt = Counter([s for s in top_scaf if s])
    n_unique = len(sc_cnt)
    print(f"unique scaffolds among 50: {n_unique}")
    print(f"top scaffolds (count, scaffold):")
    for sc, c in sc_cnt.most_common(5):
        print(f"  {c:3d}  {sc[:80]}")
    print(f"top-50 with scaf_train_freq == 0: "
          f"{(np.array(top_scaf_freq) == 0).sum()}/{TOP_N}")
    print(f"top-50 with scaf_train_freq <= 2: "
          f"{(np.array(top_scaf_freq) <= 2).sum()}/{TOP_N}")
    median_scaf_freq = float(np.median(top_scaf_freq))
    print(f"median scaf_train_freq: {median_scaf_freq:.1f}")
    scaffold_concentration = float(sc_cnt.most_common(1)[0][1] / TOP_N) if sc_cnt else 0.0

    print("\n" + "-" * 78)
    print("PHYSCHEM EXTREMITY  (|z| from train mean)")
    print("-" * 78)
    for i, k in enumerate(PC_KEYS):
        print(f"  {k:>14s}: mean|z|={np.nanmean(np.abs(top_pc_z[:, i])):.2f}  "
              f"max|z|={np.nanmax(np.abs(top_pc_z[:, i])):.2f}")
    print(f"  rows with max|z| > 2:  "
          f"{int((max_abs_z > 2).sum())}/{TOP_N}")
    print(f"  rows with max|z| > 3:  "
          f"{int((max_abs_z > 3).sum())}/{TOP_N}")

    print("\n" + "-" * 78)
    print("kNN SIMILARITY TO TRAIN")
    print("-" * 78)
    print(f"top1_sim:       mean={top1_sim.mean():.3f}  "
          f"median={np.median(top1_sim):.3f}  "
          f"min={top1_sim.min():.3f}  max={top1_sim.max():.3f}")
    print(f"n_neighbors>=0.40:  mean={n_neigh_40.mean():.1f}  "
          f"median={int(np.median(n_neigh_40))}  "
          f"rows with 0: {int((n_neigh_40 == 0).sum())}/{TOP_N}")
    # FYI: full 253 reference
    print(f"top1_sim < 0.40:   {int((top1_sim < 0.40).sum())}/{TOP_N}")
    print(f"top1_sim < 0.30:   {int((top1_sim < 0.30).sum())}/{TOP_N}")

    print("\n" + "-" * 78)
    print("PREDICTOR DISAGREEMENT  (std across {nb492,nb500,nb501,nb502,chemprop_aux})")
    print("-" * 78)
    print(f"top-50 disagreement: mean={disagreement_top.mean():.3f}  "
          f"median={np.median(disagreement_top):.3f}")
    print(f"all 253 disagreement: mean={disagreement_all.mean():.3f}  "
          f"median={np.median(disagreement_all):.3f}")
    print(f"top-50 disagreement >= 0.50:  "
          f"{int((disagreement_top >= 0.50).sum())}/{TOP_N}")

    # =================================================================
    # Top-3 insights
    # =================================================================
    print("\n" + "=" * 78)
    print("TOP-3 INSIGHTS")
    print("=" * 78)
    # Decide failure mode by simple rules
    frac_hits_missed = ((t >= 6.0) & (p < 5.5)).sum() / TOP_N
    frac_low_overpred = ((t < 5.0) & (p > 5.5)).sum() / TOP_N
    frac_under = n_under / TOP_N
    frac_low_sim = (top1_sim < 0.40).sum() / TOP_N
    frac_extreme = (max_abs_z > 2).sum() / TOP_N
    frac_rare_scaffold = (np.array(top_scaf_freq) <= 2).sum() / TOP_N
    frac_no_neigh = (n_neigh_40 == 0).sum() / TOP_N

    insights = []
    if frac_hits_missed >= 0.20:
        insights.append(
            f"(a) HITS MISSED: {frac_hits_missed:.0%} of top-50 are true hits "
            f"(truth>=6) under-called as <5.5 -- the model floor-clips actives."
        )
    if frac_low_overpred >= 0.15:
        insights.append(
            f"(b) LOW-ACTIVE OVER-CALLED: {frac_low_overpred:.0%} are "
            f"truth<5 but predicted >5.5 -- false positives near cliffs."
        )
    if frac_rare_scaffold >= 0.40 or frac_no_neigh >= 0.20:
        insights.append(
            f"(c) NO SCAFFOLD/NEIGHBOR SUPPORT: "
            f"{frac_rare_scaffold:.0%} have scaffold appearing <=2x in train; "
            f"{frac_no_neigh:.0%} have zero neighbors at Tanimoto>=0.40."
        )
    if frac_extreme >= 0.30:
        insights.append(
            f"(d) PHYSCHEM EXTREMITY: {frac_extreme:.0%} have at least one "
            f"physchem |z|>2 vs train mean (out-of-distribution)."
        )
    if frac_under >= 0.65:
        insights.append(
            f"(e) SYSTEMATIC UNDER-PREDICTION: "
            f"{frac_under:.0%} are under-called -- regression-to-mean on tail actives."
        )
    if not insights:
        insights.append(
            "(e) ALL-OF-THE-ABOVE / mixed -- no single dominant failure mode "
            "with the chosen thresholds."
        )
    for ins in insights[:3]:
        print("  ", ins)

    # Decide top-level direction label for caller
    if frac_hits_missed >= max(frac_low_overpred, frac_extreme, frac_rare_scaffold):
        direction = "under-prediction of hits (regression-to-mean on actives)"
    elif frac_low_overpred >= max(frac_extreme, frac_rare_scaffold):
        direction = "over-prediction of low-active cliff partners"
    elif frac_rare_scaffold >= 0.40 or frac_no_neigh >= 0.20:
        direction = "scaffold/neighbor support gap (OOD chemistry)"
    elif frac_extreme >= 0.30:
        direction = "physchem outliers"
    else:
        direction = "mixed / all-of-the-above"

    summary = {
        "success": True,
        "failure_csv_path": str(out_path),
        "n_worst": TOP_N,
        "mean_truth_worst": float(t.mean()),
        "mean_pred_worst":  float(p.mean()),
        "mean_resid_worst": float(resid.mean()),
        "frac_hits_in_worst":         float((t >= 6.0).mean()),
        "frac_under_predicted":       float(frac_under),
        "frac_hits_missed_strong":    float(frac_hits_missed),
        "frac_low_overpred":          float(frac_low_overpred),
        "frac_rare_scaffold":         float(frac_rare_scaffold),
        "frac_no_neighbor_at_40":     float(frac_no_neigh),
        "frac_physchem_extreme":      float(frac_extreme),
        "median_top1_sim":            float(np.median(top1_sim)),
        "scaffold_concentration":     float(scaffold_concentration),
        "direction":                  direction,
        "insights":                   insights[:3],
        "disagreement_inputs":        cand_used,
    }

    print("\n" + "=" * 78)
    print("SUMMARY DICT")
    print("=" * 78)
    for k, v in summary.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for x in v:
                print(f"    - {x}")
        else:
            print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== DONE ====")
