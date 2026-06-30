"""nb2961 -- Per-fold subset selection from {K18, K20, K24, K28} (vs global grid).

NEW PARADIGM:
    Prior cycles (nb2943, nb2960) pick a SINGLE global blend weight vector via
    grid search on the full 253-row training set, then validate it on the same
    253 rows via 5-fold scaffold CV. That couples weight-selection to all rows.

    Here we move the subset-selection INSIDE the CV loop:
      - For each outer scaffold fold:
          1. Train-fold (~80%) is used to greedy-forward-select a subset of the
             4 K-anchors {K18, K20, K24, K28} that minimizes train-fold RAE.
          2. The selected subset is equal-weighted (no per-weight grid).
          3. The held-out outer-val fold is predicted with the same subset
             (the K-anchor OOFs are already honest 5-fold cross-fit predictions
             generated in nb2960 with fresh seeds 3001..3030, so re-using them
             on the outer-val rows is NOT a leak: the outer-val rows are
             genuine OOF predictions, NOT in-sample training fits).
      - Each outer fold thus contributes its own per-fold-chosen subset.
      - Pooled RAE over the 5 outer folds = honest per-fold-subset CV.

    Repeated across kf_seed=1001 (single seed per task spec).

    DEPLOY te:
      - Greedy forward selection on FULL 253 -> select 1 global subset
      - Equal-weight that subset across deep-30 te arrays -> te_nb2961.

PROTOCOL:
    - K-anchors: nb2960 deep-30 fresh-seed bag means for K=18, 20, 24, 28
      (OOF on 253; te on 513)
    - Greedy forward: start empty, at each step add the K-anchor that most
      reduces equal-weight RAE on the held-in set; stop when no addition
      reduces RAE.
    - Outer CV: 5-fold scaffold split on the 253 unblind rows, kf_seed=1001
    - Equal-weight ensemble of the selected subset
    - Pooled RAE on the 5 outer-val folds = mean_rae for the gate.

GATE:
    mean_rae < 0.4570   -> "PROMOTE"
    mean_rae < 0.4576   -> "BETTER"        (beats nb2943 cached-5-seed 0.4576)
    else                -> "FAIL"

References:
    nb2960 K18 deep-30 OOF  = 0.4536  (best single)
    nb2960 K20 deep-30 OOF  = 0.4625
    nb2960 K24 deep-30 OOF  = 0.4687
    nb2960 K28 deep-30 OOF  = 0.4740
    nb2960 equal_K (K18,K24,K28) = 0.4567
    nb2960 blend 0.5*K20 + 0.5*equal_K (mean over 5 kf_seeds) = 0.4580
    chemprop_aux anchor (te[unb_idx]) RAE = 0.6216

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K20_30seed_oof.npy
    data/processed/nb2960_K24_30seed_oof.npy
    data/processed/nb2960_K28_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb2960_K20_30seed_te.npy
    data/processed/nb2960_K24_30seed_te.npy
    data/processed/nb2960_K28_30seed_te.npy

Outputs:
    data/processed/nb2961_summary.json
    data/processed/nb2961_pred_oof.npy        (253,) float32  -- per-fold subset OOF
    data/processed/te_nb2961.npy              (513,) float32  -- deploy te
    submissions/nb2961_per_fold_subset_search.csv  (only if verdict != "FAIL")
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

TAG = "nb2961"
PARENT_TAG = "nb2960"

# -- Inputs --------------------------------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

K_LABELS = ["K18", "K20", "K24", "K28"]
OOF_PATHS = {k: DATA_PROCESSED / f"nb2960_{k}_30seed_oof.npy" for k in K_LABELS}
TE_PATHS = {k: DATA_PROCESSED / f"nb2960_{k}_30seed_te.npy" for k in K_LABELS}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEED = 1001

# -- Gates ---------------------------------------------------------------------
GATE_PROMOTE = 0.4570
GATE_BETTER = 0.4576

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536
REF_K20 = 0.4625
REF_K24 = 0.4687
REF_K28 = 0.4740
REF_EQUAL_K_18_24_28 = 0.4567
REF_NB2960_BLEND = 0.4580
REF_NB2943 = 0.4576
REF_NB2171 = 0.4682
REF_NB1191 = 0.4718
REF_CHEMPROP_AUX = 0.6216


def _equal_weight_subset(oof_dict, subset, idx=None):
    """Equal-weight average of K-anchor OOFs on indices `idx` (None = all rows)."""
    if not subset:
        raise ValueError("subset cannot be empty")
    stack = np.stack(
        [oof_dict[k] if idx is None else oof_dict[k][idx] for k in subset],
        axis=0,
    )
    return stack.mean(axis=0)


def greedy_forward_subset(oof_dict, y, idx, candidates):
    """Greedy forward selection of K-anchors on rows `idx` minimizing RAE.

    Args:
        oof_dict : dict[label] -> (n_unb,) OOF array
        y        : (n_unb,) truth
        idx      : indices into y (training-fold rows)
        candidates : list of K labels to consider

    Returns:
        selected : list[str] -- order in which anchors were added
        rae_path : list[float] -- RAE after each step
    """
    selected = []
    remaining = list(candidates)
    rae_path = []
    best_rae = float("inf")
    while remaining:
        scores = {}
        for k in remaining:
            trial = selected + [k]
            pred = _equal_weight_subset(oof_dict, trial, idx=idx)
            scores[k] = float(rae(y[idx], pred))
        # pick best candidate to add
        k_best = min(scores, key=scores.get)
        if scores[k_best] < best_rae - 1e-8:
            selected.append(k_best)
            remaining.remove(k_best)
            best_rae = scores[k_best]
            rae_path.append(best_rae)
        else:
            break
    return selected, rae_path


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold subset selection from {K_LABELS}")
    print(f"          parent={PARENT_TAG} (deep-30 fresh-seed K-anchors)")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, kf_seed={KF_SEED}")
    print(f"          gate: <{GATE_PROMOTE} PROMOTE / <{GATE_BETTER} BETTER")
    print("=" * 78)

    # -- Load test, truth, anchor --------------------------------------------
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
    rae_anchor = float(rae(y_unb, te_anchor_513[unb_idx]))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {REF_CHEMPROP_AUX:.4f})")

    # -- Load deep-30 K-anchor OOFs + te arrays -------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load nb2960 deep-30 K-anchor OOFs and te arrays")
    print("-" * 78)
    oof_dict = {}
    te_dict = {}
    per_K_full_rae = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_arr = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_arr.shape} != ({n_test},)")
        oof_dict[k] = oof
        te_dict[k] = te_arr
        r = float(rae(y_unb, oof))
        per_K_full_rae[k] = r
        print(f"   {k}: oof_RAE = {r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")

    # -- Build scaffolds ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Per-fold greedy forward subset selection -----------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: outer CV with per-fold greedy forward subset selection")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        selected, rae_path = greedy_forward_subset(
            oof_dict, y_unb, tr_loc, K_LABELS,
        )
        # equal-weight predict outer-val
        pred_va = _equal_weight_subset(oof_dict, selected, idx=va_loc)
        oof_pooled[va_loc] = pred_va
        fold_rae = float(rae(y_unb[va_loc], pred_va))
        train_rae = float(rae(y_unb[tr_loc],
                               _equal_weight_subset(oof_dict, selected, idx=tr_loc)))
        fold_records.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "selected": selected,
            "greedy_rae_path": rae_path,
            "train_subset_rae": train_rae,
            "val_subset_rae": fold_rae,
        })
        print(f"   fold {fold_i}: tr={len(tr_loc):3d} va={len(va_loc):3d}  "
              f"selected={selected}  train_RAE={train_rae:.4f}  "
              f"val_RAE={fold_rae:.4f}")

    if np.isnan(oof_pooled).any():
        raise RuntimeError("scaffold splits did not cover all rows")
    pooled_rae = float(rae(y_unb, oof_pooled))
    per_fold_val = [r["val_subset_rae"] for r in fold_records]
    print(f"\n   pooled outer-val RAE = {pooled_rae:.4f}")
    print(f"   per-fold val RAE mean = {np.mean(per_fold_val):.4f}  "
          f"std = {np.std(per_fold_val, ddof=1):.4f}")

    # -- Subset frequency summary --------------------------------------------
    subset_freq = {k: 0 for k in K_LABELS}
    for r in fold_records:
        for k in r["selected"]:
            subset_freq[k] += 1
    print(f"\n   subset frequency across {N_FOLDS} folds: {subset_freq}")

    # -- Deploy: greedy forward on FULL 253 -> 1 global subset for te ---------
    print("\n" + "-" * 78)
    print("STEP 4: deploy subset selection (greedy forward on FULL 253)")
    print("-" * 78)
    deploy_selected, deploy_path = greedy_forward_subset(
        oof_dict, y_unb, np.arange(n_unb), K_LABELS,
    )
    deploy_oof_rae = float(rae(y_unb, _equal_weight_subset(oof_dict, deploy_selected)))
    print(f"   deploy_selected = {deploy_selected}")
    print(f"   greedy RAE path = {[f'{r:.4f}' for r in deploy_path]}")
    print(f"   deploy in-sample RAE = {deploy_oof_rae:.4f}")
    pred_te = np.stack([te_dict[k] for k in deploy_selected], axis=0).mean(axis=0)
    print(f"   te_mean={pred_te.mean():.3f}  te_std={pred_te.std():.3f}")
    te_unb_in_rae = float(rae(y_unb, pred_te[unb_idx]))
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in_rae:.4f}")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE")
    print("-" * 78)
    if pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    delta_vs_nb2960 = pooled_rae - REF_NB2960_BLEND
    delta_vs_nb2943 = pooled_rae - REF_NB2943
    delta_vs_K18 = pooled_rae - REF_K18
    delta_vs_equal_K = pooled_rae - REF_EQUAL_K_18_24_28
    print(f"   pooled_rae               = {pooled_rae:.4f}")
    print(f"   delta vs K18 (0.4536)    = {delta_vs_K18:+.4f}")
    print(f"   delta vs equal_K (0.4567)= {delta_vs_equal_K:+.4f}")
    print(f"   delta vs nb2943 (0.4576) = {delta_vs_nb2943:+.4f}")
    print(f"   delta vs nb2960 (0.4580) = {delta_vs_nb2960:+.4f}")
    print(f"   verdict                  = {verdict}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_pooled.astype(np.float32))
    np.save(te_path, pred_te.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_per_fold_subset_search.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "per_fold_greedy_forward_subset_of_K_anchors",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_candidates": K_LABELS,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_K_full_oof_rae": per_K_full_rae,
        "fold_records": fold_records,
        "subset_freq": subset_freq,
        "pooled_outer_val_rae": pooled_rae,
        "per_fold_val_rae_mean": float(np.mean(per_fold_val)),
        "per_fold_val_rae_std": float(np.std(per_fold_val, ddof=1)),
        "deploy_selected": deploy_selected,
        "deploy_greedy_path": deploy_path,
        "deploy_oof_in_sample_rae": deploy_oof_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te.mean()),
        "te_std": float(pred_te.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "mean_rae": pooled_rae,
        "ref_K18_deep30": REF_K18,
        "ref_K20_deep30": REF_K20,
        "ref_K24_deep30": REF_K24,
        "ref_K28_deep30": REF_K28,
        "ref_equal_K_18_24_28": REF_EQUAL_K_18_24_28,
        "ref_nb2960_blend": REF_NB2960_BLEND,
        "ref_nb2943": REF_NB2943,
        "ref_nb2171": REF_NB2171,
        "ref_nb1191": REF_NB1191,
        "ref_chemprop_aux": REF_CHEMPROP_AUX,
        "delta_vs_K18": delta_vs_K18,
        "delta_vs_equal_K": delta_vs_equal_K,
        "delta_vs_nb2943": delta_vs_nb2943,
        "delta_vs_nb2960": delta_vs_nb2960,
        "gate_promote": GATE_PROMOTE,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-K full-OOF RAE       = "
          + ", ".join([f"{k}={v:.4f}" for k, v in per_K_full_rae.items()]))
    print(f"   subset frequency         = {subset_freq}")
    print(f"   pooled outer-val RAE     = {pooled_rae:.4f}")
    print(f"   deploy selected          = {deploy_selected}")
    print(f"   deploy in-sample RAE     = {deploy_oof_rae:.4f}")
    print(f"   te[unb_idx] in-sample    = {te_unb_in_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_outer_val_rae",
        "per_fold_val_rae_mean",
        "per_fold_val_rae_std",
        "deploy_selected",
        "deploy_oof_in_sample_rae",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  subset_freq: {res.get('subset_freq')}")
