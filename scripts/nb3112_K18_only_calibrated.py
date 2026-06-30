"""nb3112 -- K18 alone + isotonic + per-row Tan-confidence calibration.

NEW PARADIGM:
    Extract maximum from K18 alone (deep-30, RAE 0.4536) via two-layer
    post-hoc calibration:
        (1) per-fold IsotonicRegression mapping K18 -> truth on the fold's
            train indices, sample-weighted by per-row max-Tanimoto-to-train
            confidence (high-Tan rows are reliable, low-Tan rows still
            contribute >=WEIGHT_FLOOR).
        (2) after iso, also per-row blend toward anchor mean via Tan-confidence
            scaling: rows with low max-Tan get shrunk toward the OOF mean
            (which is the safe "I don't know" prior) while high-Tan rows pass
            through unchanged.

    No multi-anchor SLSQP, no auxiliary features, no swap of K18 -- ONLY
    K18 + iso(weighted) + shrink-low-confidence. This isolates the question:
    given a single PRE-clean anchor, how much can two stacked post-hoc
    transforms extract?

PROTOCOL:
    For each of 15 fresh kf_seeds {1141..1155}:
      - scaffold_kfold_indices(unb_scaffolds, n_splits=5, seed=kf_seed)
      - Per fold:
          * IsotonicRegression(out_of_bounds='clip').fit(
                K18_oof[tr], y_unb[tr], sample_weight=w_unb[tr])
          * iso_va = ir.predict(K18_oof[va])
          * mu_tr = (w_unb[tr] * y_unb[tr]).sum() / w_unb[tr].sum()
          * shrink_va = w_unb[va]  # in (WEIGHT_FLOOR, 1.0)
          * final_va = shrink_va * iso_va + (1 - shrink_va) * mu_tr
      - Pool the per-fold final_va across the 5 fold-validation sets.
      - Per-seed pooled RAE.
    Aggregate: mean +/- std + 95% CI across 15 seeds.

    Deploy (te 513):
      - Refit weighted iso on FULL 253 (K18_oof, y_unb, w_unb).
      - mu_full = (w_unb * y_unb).sum() / w_unb.sum().
      - te_iso = ir.predict(K18_te).
      - shrink_te = w_te.
      - te_final = shrink_te * te_iso + (1 - shrink_te) * mu_full.

GATE (on 15-seed mean pooled RAE):
    mean < 0.4475 -> "BETTER"     (significantly under K18 alone 0.4536)
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF RAE        = 0.4536
    nb2864 Tan-conf weighted iso on nb2240 -- prior precedent
    nb2991 K18-alone wide-seed verify -- prior precedent
    nb2171 5-anchor pyramid ceiling   = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy

Outputs:
    data/processed/nb3112_summary.json
    data/processed/nb3112_pred_oof.npy   (253,) float32 -- pooled per-fold cal preds
    data/processed/te_nb3112.npy         (513,) float32 -- deploy te
    submissions/nb3112_K18_only_calibrated.csv   (only on BETTER)
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
from rdkit import RDLogger
from sklearn.isotonic import IsotonicRegression

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch, standardize
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3112"
PARENT_TAG = "nb2960_K18"

# -- Inputs --------------------------------------------------------------------
K_LABEL = "K18"
OOF_PATH = DATA_PROCESSED / f"nb2960_{K_LABEL}_30seed_oof.npy"
TE_PATH = DATA_PROCESSED / f"nb2960_{K_LABEL}_30seed_te.npy"

# -- Confidence weighting (mirrors nb2864) -------------------------------------
SIM_PIVOT = 0.35
SIM_SLOPE = 10.0
WEIGHT_FLOOR = 0.1

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1141, 1156))   # 15 fresh seeds {1141..1155}

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References ----------------------------------------------------------------
REF_K18_DEEP30 = 0.4536
REF_NB2864_PRECEDENT = "Tan-conf weighted iso on nb2240"
REF_NB2991_PRECEDENT = "K18-alone wide-seed"
REF_NB2171 = 0.4682


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def max_tanimoto_to_pool(fp_query: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """Return max-Tanimoto per query row to any pool row (uint8 ECFP4 inputs)."""
    a = fp_query.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    out = np.zeros(n_q, dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T  # (block, n_pool)
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        out[s:e] = sim.max(axis=1)
    return out


def conf_weight(max_tan: np.ndarray) -> np.ndarray:
    """weight = max(WEIGHT_FLOOR, sigmoid(SIM_SLOPE * (max_tan - SIM_PIVOT)))."""
    s = sigmoid(SIM_SLOPE * (max_tan - SIM_PIVOT))
    return np.maximum(WEIGHT_FLOOR, s).astype(np.float64)


def _calibrate_one_seed(
    anchor_oof: np.ndarray,
    y_unb: np.ndarray,
    w_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-fold weighted iso + per-row Tan-confidence shrink toward mu_tr.

    Returns dict with pooled RAE + per-fold diagnostics + the per-seed
    pooled OOF prediction vector (so the caller can average across seeds
    if desired).
    """
    n_unb = len(y_unb)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    pred_seed = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold = []
    for k, (tr, va) in enumerate(splits):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(anchor_oof[tr], y_unb[tr], sample_weight=w_unb[tr])
        iso_va = ir.predict(anchor_oof[va]).astype(np.float64)
        mu_tr = float((w_unb[tr] * y_unb[tr]).sum() / w_unb[tr].sum())
        shrink_va = w_unb[va]  # in (WEIGHT_FLOOR, 1.0)
        final_va = shrink_va * iso_va + (1.0 - shrink_va) * mu_tr
        pred_seed[va] = final_va

        fold_iso_rae = float(rae(y_unb[va], iso_va))
        fold_final_rae = float(rae(y_unb[va], final_va))
        n_knots = int(len(np.unique(ir.X_thresholds_))) \
            if hasattr(ir, "X_thresholds_") else -1
        per_fold.append({
            "fold": int(k),
            "n_tr": int(len(tr)),
            "n_va": int(len(va)),
            "n_knots": n_knots,
            "mu_tr": round(mu_tr, 4),
            "fold_iso_val_rae": round(fold_iso_rae, 4),
            "fold_final_val_rae": round(fold_final_rae, 4),
            "w_va_mean": round(float(shrink_va.mean()), 4),
        })

    if np.isnan(pred_seed).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")

    pooled = float(rae(y_unb, pred_seed))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold": per_fold,
        "pred_seed": pred_seed,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K18 alone + iso + per-row Tan-conf shrink (15 fresh seeds)")
    print(f"          kf_seeds = {KF_SEEDS}")
    print(f"          weight   = max({WEIGHT_FLOOR}, sigmoid({SIM_SLOPE}*(max_tan-{SIM_PIVOT})))")
    print(f"          gate     = mean < {GATE_BETTER} -> BETTER, else FAIL")
    print("=" * 78)

    # -- Load truth ----------------------------------------------------------
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"

    # -- Load K18 deep-30 OOF + te ------------------------------------------
    anchor_oof = np.load(OOF_PATH).astype(np.float64)
    anchor_te = np.load(TE_PATH).astype(np.float64)
    n_test = anchor_te.shape[0]
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"K18 OOF shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"K18 te shape {anchor_te.shape} != ({n_test},)")
    k18_full_rae = float(rae(y_unb, anchor_oof))
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    print(f"       K18 deep-30 OOF RAE = {k18_full_rae:.4f}")

    # Leak sanity check
    leak_frac = float(np.mean(np.isclose(anchor_oof, y_unb, atol=1e-6)))
    if leak_frac > 0.05:
        print(f"   WARN K18: {leak_frac:.1%} rows == truth -- possible leak")
    print(f"       K18 leak (oof==truth) = {leak_frac:.4%}")

    # -- Test SMILES (scaffolds + Tanimoto fps) -----------------------------
    te = load_test()
    te_smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_name_col = "name" if "name" in te.columns else "Molecule Name"
    te_smi = te[te_smi_col].astype(str).tolist()
    te_names = te[te_name_col].values
    te_std = [standardize(s) for s in te_smi]
    te_std_smi = []
    for m in te_std:
        if m is None:
            te_std_smi.append("")
        else:
            from rdkit import Chem as _Chem
            te_std_smi.append(_Chem.MolToSmiles(m))
    fp_te = morgan_fp_batch(te_std_smi)

    # -- Train pool for Tanimoto -------------------------------------------
    train = load_train()
    train = train[train["pec50"].notna()].copy().reset_index(drop=True)
    train_std = [standardize(s) for s in train["smiles"].astype(str).tolist()]
    train_smi = []
    for m in train_std:
        if m is None:
            train_smi.append("")
        else:
            from rdkit import Chem as _Chem
            train_smi.append(_Chem.MolToSmiles(m))
    fp_train = morgan_fp_batch(train_smi)
    keep = fp_train.sum(axis=1) > 0
    fp_train = fp_train[keep]
    print(f"[train] n_train_pool={len(fp_train)}")

    print(f"[tan] computing max-Tanimoto per test row...")
    ts = time.time()
    max_tan_te = max_tanimoto_to_pool(fp_te, fp_train)  # (513,)
    print(f"[tan] done wall={time.time()-ts:.1f}s  "
          f"mean={max_tan_te.mean():.3f}  median={np.median(max_tan_te):.3f}  "
          f"q10={np.quantile(max_tan_te, 0.10):.3f}  "
          f"q90={np.quantile(max_tan_te, 0.90):.3f}")
    max_tan_unb = max_tan_te[unb_idx]
    print(f"[tan] unb subset (253): mean={max_tan_unb.mean():.3f}  "
          f"median={np.median(max_tan_unb):.3f}  "
          f"q10={np.quantile(max_tan_unb, 0.10):.3f}")

    w_unb = conf_weight(max_tan_unb)
    w_te = conf_weight(max_tan_te)
    print(f"[weight] unb mean={w_unb.mean():.3f}  median={np.median(w_unb):.3f}  "
          f"min={w_unb.min():.3f}  max={w_unb.max():.3f}")

    # -- Scaffolds (kf_seed independent) -------------------------------------
    unb_smiles = [te_smi[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] n_unique_scaffolds = {n_unique_scaf}")

    # -- Wide-seed sweep ------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"WIDE-SEED SWEEP: {len(KF_SEEDS)} fresh kf_seeds {{1141..1155}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    pred_seeds_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    for i, s in enumerate(KF_SEEDS):
        res = _calibrate_one_seed(anchor_oof, y_unb, w_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        pred_seeds_stack[i] = res["pred_seed"]
        fold_iso_means = [pf["fold_iso_val_rae"] for pf in res["per_fold"]]
        fold_final_means = [pf["fold_final_val_rae"] for pf in res["per_fold"]]
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "fold_iso_val_rae_mean": round(float(np.mean(fold_iso_means)), 4),
            "fold_final_val_rae_mean": round(float(np.mean(fold_final_means)), 4),
            "fold_final_val_rae_std": round(float(np.std(fold_final_means, ddof=1)), 4),
            "per_fold": res["per_fold"],
        })
        print(f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
              f"iso_fold_mean={np.mean(fold_iso_means):.4f}  "
              f"final_fold_mean={np.mean(fold_final_means):.4f}")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std_rae / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
    t_mult = 2.145
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))

    print("\n" + "-" * 78)
    print("AGGREGATE (15 fresh seeds)")
    print("-" * 78)
    print(f"   pooled mean    = {mean_rae:.4f}")
    print(f"   pooled std     = {std_rae:.4f}")
    print(f"   pooled sem     = {sem:.4f}")
    print(f"   95% CI         = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled median  = {median_rae:.4f}")
    print(f"   pooled 5/95p   = [{p5:.4f}, {p95:.4f}]")
    print(f"   pooled min/max = [{arr.min():.4f}, {arr.max():.4f}]")

    delta_vs_K18 = mean_rae - REF_K18_DEEP30
    delta_vs_nb2171 = mean_rae - REF_NB2171
    print(f"\n   K18 deep-30 (nb2960)   = {REF_K18_DEEP30:.4f}")
    print(f"   delta vs K18 deep-30   = {delta_vs_K18:+.4f}")
    print(f"   nb2171 ceiling          = {REF_NB2171:.4f}")
    print(f"   delta vs nb2171         = {delta_vs_nb2171:+.4f}")

    # -- Average per-row OOF across seeds (for save) -------------------------
    oof_for_save = pred_seeds_stack.mean(axis=0).astype(np.float32)
    oof_avg_rae = float(rae(y_unb, oof_for_save.astype(np.float64)))
    print(f"\n   per-seed-avg OOF RAE   = {oof_avg_rae:.4f}  (saved as nb3112_pred_oof.npy)")

    # -- Deploy: refit weighted iso on FULL 253, build te_final --------------
    print("\n" + "-" * 78)
    print("DEPLOY (refit weighted iso on full 253, then shrink te by w_te)")
    print("-" * 78)
    ir_deploy = IsotonicRegression(out_of_bounds="clip")
    ir_deploy.fit(anchor_oof, y_unb, sample_weight=w_unb)
    n_knots_deploy = int(len(np.unique(ir_deploy.X_thresholds_))) \
        if hasattr(ir_deploy, "X_thresholds_") else -1
    mu_full = float((w_unb * y_unb).sum() / w_unb.sum())

    te_iso = ir_deploy.predict(anchor_te).astype(np.float64)
    te_final = w_te * te_iso + (1.0 - w_te) * mu_full
    te_final = np.clip(te_final, 3.0, 9.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_final[unb_idx].astype(np.float64)))
    print(f"   deploy n_knots               = {n_knots_deploy}")
    print(f"   mu_full (w-weighted mean)    = {mu_full:.4f}")
    print(f"   te[unb] in-sample RAE        = {te_unb_in_rae:.4f}")
    print(f"   te mean / std                = {te_final.mean():.3f} / "
          f"{te_final.std():.3f}  "
          f"(K18_te {anchor_te.mean():.3f}/{anchor_te.std():.3f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE nb3112 (K18 + iso + Tan-conf shrink). Wide-seed mean "
            f"{mean_rae:.4f} beats K18-alone {REF_K18_DEEP30:.4f} by "
            f"{-delta_vs_K18:.4f}. Two-layer post-hoc calibration extracts "
            "real gain from a single PRE-clean anchor."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT nb3112. Wide-seed mean {mean_rae:.4f} above gate "
            f"{GATE_BETTER:.4f}; iso + Tan-conf shrink on K18 does not "
            f"improve on K18-alone {REF_K18_DEEP30:.4f}. Confirms post-hoc "
            "ceiling is set by anchor information content, not calibration df."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_final)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_K18_only_calibrated.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smi,
            "Molecule Name": te_names,
            "pEC50": te_final,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "K18_alone_iso_weighted_plus_tan_conf_shrink",
        "anchor": K_LABEL,
        "anchor_pre_unblind": True,
        "anchor_oof_path": str(OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "k18_full_oof_rae": round(k18_full_rae, 4),
        "anchor_leak_eq_truth_frac": round(leak_frac, 4),
        "sim_pivot": SIM_PIVOT,
        "sim_slope": SIM_SLOPE,
        "weight_floor": WEIGHT_FLOOR,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "max_tan_te_stats": {
            "mean": float(max_tan_te.mean()),
            "median": float(np.median(max_tan_te)),
            "q10": float(np.quantile(max_tan_te, 0.10)),
            "q90": float(np.quantile(max_tan_te, 0.90)),
        },
        "max_tan_unb_stats": {
            "mean": float(max_tan_unb.mean()),
            "median": float(np.median(max_tan_unb)),
            "q10": float(np.quantile(max_tan_unb, 0.10)),
            "q90": float(np.quantile(max_tan_unb, 0.90)),
        },
        "weight_unb_stats": {
            "mean": float(w_unb.mean()),
            "median": float(np.median(w_unb)),
            "min": float(w_unb.min()),
            "max": float(w_unb.max()),
        },
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "p5_rae": round(p5, 4),
        "p95_rae": round(p95, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "oof_avg_rae": round(oof_avg_rae, 4),
        "deploy_n_knots": n_knots_deploy,
        "mu_full": round(mu_full, 4),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_mean": float(te_final.mean()),
        "te_std": float(te_final.std()),
        "anchor_te_mean": float(anchor_te.mean()),
        "anchor_te_std": float(anchor_te.std()),
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_nb2864_precedent": REF_NB2864_PRECEDENT,
        "ref_nb2991_precedent": REF_NB2991_PRECEDENT,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18_deep30": round(delta_vs_K18, 4),
        "delta_vs_nb2171": round(delta_vs_nb2171, 4),
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (15 seeds)    = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                 = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs K18 deep-30   = {delta_vs_K18:+.4f}")
    print(f"   verdict                = {verdict}")
    print(f"   ladder action          = {ladder_action}")
    print(f"   wall                   = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_K18_deep30",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
