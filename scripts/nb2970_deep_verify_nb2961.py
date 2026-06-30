"""nb2970 -- Wide-seed deep verification of nb2961 per-fold subset search.

CONTEXT:
    nb2961 reported pooled_outer_val_rae = 0.4567 at kf_seed=1001 with
    per-fold greedy forward selection on {K18, K20, K24, K28}. df per fold
    is 2-4 (subset selection in {empty, K18, K18+K20, ...}); on a 5-fold
    n=253 cut this is small df but selection variance is real. Past lessons:
    - nb1086 lucky-seed trap: 5-seed bag missed regression on fresh seeds
    - nb1191 5-seed gate-A fail, 15-seed verified mean
    - nb2060/nb2095 deep-30 cycle-160 rule: 5-seed std under-disperses ~4x
    - cycle-149 rule: any 5-seed result that crosses a promote gate must be
      re-verified with >=15 NEW seeds before promotion.

PROTOCOL:
    - Re-run nb2961 pipeline (per-fold greedy forward subset selection +
      pooled outer-val RAE) across 15 fresh kf_seeds: 1006..1020
      (originally tested seeds were 1001-1005-era; these are NEW seeds).
    - For each kf_seed, build scaffold 5-fold splits, do per-fold greedy
      forward subset selection on K-anchors, equal-weight predict outer-val,
      compute pooled RAE.
    - Aggregate mean + std + 95% CI across 15 fresh seeds.
    - Compare to nb2961's 0.4567 and the PROMOTE gate 0.4570.

GATES:
    15-seed mean < 0.4570  -> "VERIFIED_PROMOTE_PRIMARY1"
    15-seed mean < 0.4598  -> "VERIFIED_MARGINAL"
    shift  > +0.005 vs 0.4567 -> "LUCKY_SEED_TRAP"
    else                   -> "MIXED_NEEDS_INVESTIGATION"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2960_K{18,20,24,28}_30seed_oof.npy
    data/processed/nb2960_K{18,20,24,28}_30seed_te.npy
    data/processed/nb2961_summary.json   (for reference of original result)

Outputs:
    data/processed/nb2970_summary.json
    data/processed/nb2970_pred_oof.npy   (253,) float32 -- mean-of-seeds OOF
    data/processed/te_nb2970.npy         (513,) float32 -- mean-of-seeds te
    submissions/nb2970_nb2961_verified.csv  (only if VERIFIED_PROMOTE_PRIMARY1)
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

TAG = "nb2970"
PARENT_TAG = "nb2961"

# -- Inputs --------------------------------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
K_LABELS = ["K18", "K20", "K24", "K28"]
OOF_PATHS = {k: DATA_PROCESSED / f"nb2960_{k}_30seed_oof.npy" for k in K_LABELS}
TE_PATHS = {k: DATA_PROCESSED / f"nb2960_{k}_30seed_te.npy" for k in K_LABELS}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1006, 1021))  # 1006..1020 inclusive = 15 NEW seeds

# -- Reference / gates ---------------------------------------------------------
ORIGINAL_RAE = 0.4567            # nb2961 kf_seed=1001
GATE_PROMOTE = 0.4570            # primary gate
GATE_MARGINAL = 0.4598           # secondary gate (mid-tier)
SHIFT_TRAP = 0.005               # mean > 0.4567 + 0.005 = lucky-seed trap


def _equal_weight_subset(oof_dict, subset, idx=None):
    if not subset:
        raise ValueError("subset cannot be empty")
    stack = np.stack(
        [oof_dict[k] if idx is None else oof_dict[k][idx] for k in subset],
        axis=0,
    )
    return stack.mean(axis=0)


def greedy_forward_subset(oof_dict, y, idx, candidates):
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
        k_best = min(scores, key=scores.get)
        if scores[k_best] < best_rae - 1e-8:
            selected.append(k_best)
            remaining.remove(k_best)
            best_rae = scores[k_best]
            rae_path.append(best_rae)
        else:
            break
    return selected, rae_path


def run_one_seed(oof_dict, y_unb, unb_scaffolds, kf_seed):
    """Per-fold greedy subset selection at one kf_seed -> pooled RAE + records."""
    n_unb = len(y_unb)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    subset_freq = {k: 0 for k in K_LABELS}
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        selected, rae_path = greedy_forward_subset(
            oof_dict, y_unb, tr_loc, K_LABELS,
        )
        pred_va = _equal_weight_subset(oof_dict, selected, idx=va_loc)
        oof_pooled[va_loc] = pred_va
        fold_rae = float(rae(y_unb[va_loc], pred_va))
        fold_records.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "selected": selected,
            "val_rae": fold_rae,
        })
        for k in selected:
            subset_freq[k] += 1
    if np.isnan(oof_pooled).any():
        raise RuntimeError(f"seed={kf_seed} did not cover all rows")
    pooled_rae = float(rae(y_unb, oof_pooled))
    return pooled_rae, oof_pooled, fold_records, subset_freq


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 15-seed deep verification of {PARENT_TAG}")
    print(f"          kf_seeds = {KF_SEEDS[0]}..{KF_SEEDS[-1]} ({len(KF_SEEDS)} NEW seeds)")
    print(f"          original {PARENT_TAG} (kf_seed=1001) RAE = {ORIGINAL_RAE:.4f}")
    print(f"          gate: <{GATE_PROMOTE} VERIFIED_PROMOTE_PRIMARY1")
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
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f}")

    # -- Load K-anchor OOF + te arrays ---------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load deep-30 K-anchor OOFs and te arrays")
    print("-" * 78)
    oof_dict = {}
    te_dict = {}
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
        print(f"   {k}: full_oof_RAE = {r:.4f}")

    # -- Build scaffolds (constant across seeds, only kf shuffling changes) ---
    print("\n" + "-" * 78)
    print("STEP 2: build scaffolds")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Loop kf_seeds --------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: re-run per-fold subset search at {len(KF_SEEDS)} fresh kf_seeds")
    print("-" * 78)
    per_seed = []
    pooled_oof_stack = []   # (15, 253)
    pooled_te_stack = []    # (15, 513)
    deploy_selected_freq = {k: 0 for k in K_LABELS}
    for kf_seed in KF_SEEDS:
        pooled_rae, pooled_oof, fold_records, subset_freq = run_one_seed(
            oof_dict, y_unb, unb_scaffolds, kf_seed,
        )
        # also compute deploy subset selection on FULL 253 (no kf dependency)
        # but the deploy te depends on K-anchor te arrays only; for kf
        # robustness we average kf-specific deploy choices.
        # NOTE: deploy_selected on full 253 is kf-seed independent (single
        # greedy forward), so we compute it once outside the loop below.
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled_rae),
            "per_fold": fold_records,
            "subset_freq": subset_freq,
        })
        pooled_oof_stack.append(pooled_oof)
        print(f"   kf_seed={kf_seed}: pooled_RAE={pooled_rae:.4f}  "
              f"subset_freq={subset_freq}")

    pooled_oof_stack = np.stack(pooled_oof_stack, axis=0)  # (15, 253)

    # -- Aggregate ------------------------------------------------------------
    rae_arr = np.array([r["pooled_rae"] for r in per_seed], dtype=np.float64)
    seed_mean = float(rae_arr.mean())
    seed_std = float(rae_arr.std(ddof=1))
    seed_min = float(rae_arr.min())
    seed_max = float(rae_arr.max())
    # 95% CI on mean (normal approx, t-table ~2.145 for df=14)
    t_975_df14 = 2.145
    ci_half = t_975_df14 * seed_std / np.sqrt(len(rae_arr))
    ci_lo = seed_mean - ci_half
    ci_hi = seed_mean + ci_half
    shift_vs_original = seed_mean - ORIGINAL_RAE
    n_better_than_original = int((rae_arr < ORIGINAL_RAE).sum())
    n_below_gate = int((rae_arr < GATE_PROMOTE).sum())

    print("\n" + "-" * 78)
    print("STEP 4: aggregate")
    print("-" * 78)
    print(f"   15-seed mean        = {seed_mean:.4f}")
    print(f"   15-seed std         = {seed_std:.4f}")
    print(f"   15-seed min         = {seed_min:.4f}")
    print(f"   15-seed max         = {seed_max:.4f}")
    print(f"   95% CI              = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"   shift vs original   = {shift_vs_original:+.4f} (gate {ORIGINAL_RAE:.4f})")
    print(f"   n seeds < original  = {n_better_than_original} / {len(rae_arr)}")
    print(f"   n seeds < gate(0.4570) = {n_below_gate} / {len(rae_arr)}")

    # -- Deploy: greedy forward on FULL 253 -> 1 global subset for te ---------
    # This is kf-seed independent (no scaffold splits used)
    print("\n" + "-" * 78)
    print("STEP 5: deploy subset selection (greedy forward on FULL 253)")
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
    print("STEP 6: GATE")
    print("-" * 78)
    if shift_vs_original > SHIFT_TRAP:
        verdict = "LUCKY_SEED_TRAP"
    elif seed_mean < GATE_PROMOTE:
        verdict = "VERIFIED_PROMOTE_PRIMARY1"
    elif seed_mean < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
    else:
        verdict = "MIXED_NEEDS_INVESTIGATION"

    print(f"   verdict = {verdict}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    # OOF: mean of 15 per-seed pooled OOFs (each is itself a pooled outer-val
    # prediction; averaging reduces kf-shuffle variance).
    oof_mean = pooled_oof_stack.mean(axis=0).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_mean)
    np.save(te_path, pred_te.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_nb2961_verified.csv"
    if verdict == "VERIFIED_PROMOTE_PRIMARY1":
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
        "method": "wide_seed_verification_of_per_fold_greedy_subset_search",
        "anchor_base": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_in_rae": rae_anchor,
        "K_candidates": K_LABELS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_seed": per_seed,
        "rae_array": rae_arr.tolist(),
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "seed_min": seed_min,
        "seed_max": seed_max,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "original_rae": ORIGINAL_RAE,
        "shift_vs_original": shift_vs_original,
        "n_better_than_original": n_better_than_original,
        "n_below_gate_promote": n_below_gate,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "shift_trap_threshold": SHIFT_TRAP,
        "deploy_selected": deploy_selected,
        "deploy_greedy_path": deploy_path,
        "deploy_oof_in_sample_rae": deploy_oof_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te.mean()),
        "te_std": float(pred_te.std()),
        "mean_rae": seed_mean,                 # canonical 'mean_rae' for tooling
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv)
                           if verdict == "VERIFIED_PROMOTE_PRIMARY1" else None),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   original (kf=1001)        = {ORIGINAL_RAE:.4f}")
    print(f"   15-seed mean (1006..1020) = {seed_mean:.4f} +/- {seed_std:.4f}")
    print(f"   95% CI                    = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"   shift                     = {shift_vs_original:+.4f}")
    print(f"   under-dispersion (orig vs 15-seed) = "
          f"{abs(seed_mean - ORIGINAL_RAE) / max(seed_std, 1e-9):.2f}-sigma")
    print(f"   n<{GATE_PROMOTE:.4f}                 = {n_below_gate}/{len(rae_arr)}")
    print(f"   deploy selected           = {deploy_selected}")
    print(f"   verdict                   = {verdict}")
    print(f"   wall                      = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "seed_mean",
        "seed_std",
        "ci95_lo",
        "ci95_hi",
        "shift_vs_original",
        "n_below_gate_promote",
        "n_seeds",
        "deploy_selected",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
