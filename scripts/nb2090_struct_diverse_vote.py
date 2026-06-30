"""nb2090 -- Structurally diverse tri-vote: {nb1158, chemprop_aux, nb1014}.

CONTEXT (per nb2081):
    nb2081 used {nb1158, nb503, nb562} but rho(nb503, nb562) = 0.9999 because
    both are rank-stretches over the same base predictor -- they are siblings.
    Diversity gate FAILED. Vote was effectively nb1158 + 2 echoes.

HYPOTHESIS:
    Build a tri-vote over 3 STRUCTURALLY DISTINCT model classes:
        nb1158        -- LGBM K=32 mean bag (ECFP + RDKit features, gradient boost)
        chemprop_aux  -- MPNN GNN multi-task (learned message-passing embeddings)
        nb1014        -- multi-seed bag stack (chemprop_aux + nb972 SLSQP blend +
                         rank-stretch); shares chemprop_aux as one component
    Each is a distinct ML class: tabular-boost vs neural-graph vs stacked-bag.

GATES:
    - Diversity: max pairwise Spearman among the 3 candidates must be <= 0.95.
      NOTE: rho(chemprop_aux, nb1014) measured at 0.9816 -- nb1014 contains
      chemprop_aux as a component (76% weight per nb1014_summary.json) so they
      are NOT structurally distinct in OOF rank space. Expect FAIL.
    - Distinctness: rho(vote, anchor=nb1158) must be <= 0.98.
    - Beat: pooled_RAE (mean across 5 KF seeds) must be <= 0.4697 - 0.003
      = 0.4667 to promote past nb2060.

MECHANISM (identical to nb2076/nb2081):
    1. Per-row rank for each of the 3 vectors (rankdata, method=average).
    2. Element-wise MEDIAN of the 3 rank vectors -> rank_vote in [1..N].
    3. Map rank_vote back to pec50 via empirical CDF of nb1158 (strongest indiv).

EVALUATION:
    Scaffold 5-fold CV across 5 seeds (1001..1005), tri_vote per fold val slice.
    CDF anchor inside each fold = nb1158 validation slice (no label leakage).

DEPLOY:
    If promote==True, tri-vote on full 513-row te, anchor = nb1158 te(513).
    Saved to data/processed/te_nb2090.npy and submissions/nb2090_struct_diverse_vote.csv.

Inputs (all verified to exist):
    OOFs on 253 (unblind):
        data/processed/nb1158_mean_bag_oof_K32.npy            (LGBM K=32 bag)
        data/processed/nb1133_chemprop_aux_pred_oof.npy       (MPNN GNN)
        data/processed/nb1133_nb1014_pred_oof.npy             (bag stack)
    te on 513:
        data/processed/te_nb1158.npy
        data/processed/te_chemprop_aux.npy
        data/processed/te_nb1014.npy
    Audit:
        data/processed/_audit_unblind_idx.npy
        data/processed/_audit_unblind_y.npy

Outputs:
    scripts/nb2090_struct_diverse_vote.py  (this file)
    data/processed/nb2090_summary.json
    data/processed/te_nb2090.npy
    submissions/nb2090_struct_diverse_vote.csv   (only if promote=True)
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
from scipy.stats import rankdata, spearmanr

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2090"
GATE_MARGIN = 0.003
NB2060_REF_RAE = 0.4697
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
SPEARMAN_MAX_ANCHOR = 0.98          # vote must not echo anchor
SPEARMAN_MAX_PAIRWISE = 0.95        # candidates must be structurally distinct

# Individual baselines (memo'd honest cross-fit numbers)
INDIV_REF = {
    "nb1158":       0.4902,
    "chemprop_aux": 0.6216,
    "nb1014":       0.5871,         # in_sample_rae_overfit_bound per summary
}


def empirical_cdf_map(anchor_values: np.ndarray, target_ranks: np.ndarray,
                      n_anchor: int) -> np.ndarray:
    """Map rank vector (1..n_anchor) back to pec50 via sorted anchor values."""
    anchor_sorted = np.sort(anchor_values)
    pos = np.clip(target_ranks - 1.0, 0.0, n_anchor - 1.0)
    lo = np.floor(pos).astype(int)
    hi = np.clip(lo + 1, 0, n_anchor - 1)
    frac = pos - lo
    return (1.0 - frac) * anchor_sorted[lo] + frac * anchor_sorted[hi]


def tri_vote(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
             anchor_values: np.ndarray):
    """Per-row median rank of 3 vectors, mapped back via anchor CDF."""
    assert p1.shape == p2.shape == p3.shape == anchor_values.shape, \
        f"shape mismatch: {p1.shape} {p2.shape} {p3.shape} {anchor_values.shape}"
    n = len(p1)
    r1 = rankdata(p1, method="average")
    r2 = rankdata(p2, method="average")
    r3 = rankdata(p3, method="average")
    rank_vote = np.median(np.column_stack([r1, r2, r3]), axis=1)
    return empirical_cdf_map(anchor_values, rank_vote, n), rank_vote


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- structurally diverse tri-vote {{nb1158, chemprop_aux, nb1014}}")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]

    # --- Load the 3 candidates (all cached -- no reconstruction) ---
    print("\n[load] candidates")
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    chem_oof   = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1014_oof = np.load(DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy").astype(np.float64)
    nb1158_te  = np.load(DATA_PROCESSED / "te_nb1158.npy").astype(np.float64)
    chem_te    = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    nb1014_te  = np.load(DATA_PROCESSED / "te_nb1014.npy").astype(np.float64)
    assert nb1158_oof.shape == chem_oof.shape == nb1014_oof.shape == (n_unb,)
    assert nb1158_te.shape  == chem_te.shape  == nb1014_te.shape  == (n_te,)

    indiv = {
        "nb1158":       float(rae(y_unb, nb1158_oof)),
        "chemprop_aux": float(rae(y_unb, chem_oof)),
        "nb1014":       float(rae(y_unb, nb1014_oof)),
    }
    te_arr = {"nb1158": nb1158_te, "chemprop_aux": chem_te, "nb1014": nb1014_te}
    for name, val in indiv.items():
        ref = INDIV_REF[name]
        gap = val - ref
        print(f"   {name:<14}  oof_RAE={val:.4f}  (memo={ref:.4f}, gap={gap:+.4f})  "
              f"te(513) mean={te_arr[name].mean():.3f} std={te_arr[name].std():.3f}")

    # --- Pairwise Spearman among candidates (structural diversity check) ---
    print("\n[diversity] pairwise Spearman among candidate OOFs")
    rho_15_chem  = float(spearmanr(nb1158_oof, chem_oof).correlation)
    rho_15_14    = float(spearmanr(nb1158_oof, nb1014_oof).correlation)
    rho_chem_14  = float(spearmanr(chem_oof,   nb1014_oof).correlation)
    pairwise = {
        "nb1158_vs_chemprop_aux":      rho_15_chem,
        "nb1158_vs_nb1014":            rho_15_14,
        "chemprop_aux_vs_nb1014":      rho_chem_14,
    }
    for k, v in pairwise.items():
        print(f"   rho({k:<30}) = {v:.4f}")
    max_pairwise = max(pairwise.values())
    diversity_pass = max_pairwise <= SPEARMAN_MAX_PAIRWISE
    print(f"[diversity] max pairwise = {max_pairwise:.4f}  "
          f"(<= {SPEARMAN_MAX_PAIRWISE} ? {'PASS' if diversity_pass else 'FAIL'})")
    if not diversity_pass:
        print(f"[diversity] NOTE: chemprop_aux is a component of nb1014 (76% weight "
              f"per nb1014_summary.json) -- they are NOT structurally distinct.")

    # --- Tri-vote on full 253 (global diagnostic; anchor CDF = nb1158 unb) ---
    vote_oof_global, rank_vote_unb = tri_vote(
        nb1158_oof, chem_oof, nb1014_oof, nb1158_oof,
    )
    rae_global = float(rae(y_unb, vote_oof_global))
    rho_anchor_nb1158 = float(spearmanr(rank_vote_unb, rankdata(nb1158_oof)).correlation)
    rho_anchor_chem   = float(spearmanr(rank_vote_unb, rankdata(chem_oof)).correlation)
    rho_anchor_nb1014 = float(spearmanr(rank_vote_unb, rankdata(nb1014_oof)).correlation)
    print(f"\n[global] tri-vote OOF RAE (anchor=nb1158)   = {rae_global:.4f}")
    print(f"[global] Spearman(rank_vote, nb1158)         = {rho_anchor_nb1158:.4f}")
    print(f"[global] Spearman(rank_vote, chemprop_aux)   = {rho_anchor_chem:.4f}")
    print(f"[global] Spearman(rank_vote, nb1014)         = {rho_anchor_nb1014:.4f}")
    distinct_pass = rho_anchor_nb1158 <= SPEARMAN_MAX_ANCHOR
    print(f"[distinct] rho(vote, nb1158) {rho_anchor_nb1158:.4f} <= "
          f"{SPEARMAN_MAX_ANCHOR} -> {'PASS' if distinct_pass else 'FAIL'}")

    # --- Scaffold 5-fold CV across 5 seeds (apply tri_vote per fold) ---
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_blend = np.full(n_unb, np.nan)
        for tr_loc, va_loc in splits:
            vote_va, _ = tri_vote(
                nb1158_oof[va_loc],
                chem_oof[va_loc],
                nb1014_oof[va_loc],
                nb1158_oof[va_loc],     # CDF anchor = nb1158 va slice
            )
            oof_blend[va_loc] = vote_va
        pooled = float(rae(y_unb, oof_blend))
        per_seed.append({"kf_seed": int(kf_seed), "pooled_rae": pooled})
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}")
    pooled_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_std  = float(np.std([r["pooled_rae"]  for r in per_seed]))
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean = float(rae(y_unb, mean_oof))
    print(f"\n[cv] pooled_RAE (mean of seeds) = {pooled_mean:.4f} "
          f"(+/- {pooled_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs   = {rae_of_mean:.4f}")

    # --- Deploy: tri-vote on 513, anchor = nb1158 te(513) ---
    vote_te, rank_vote_te = tri_vote(nb1158_te, chem_te, nb1014_te, nb1158_te)
    deploy_te = vote_te.astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] te(513) tri-vote: mean={deploy_te.mean():.3f} "
          f"std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")

    # --- Decision gate ---
    print("\n" + "-" * 78)
    print("DECISION GATE")
    print("-" * 78)
    delta_vs_nb2060      = pooled_mean - NB2060_REF_RAE
    delta_vs_nb1158      = pooled_mean - INDIV_REF["nb1158"]
    delta_vs_chemprop    = pooled_mean - INDIV_REF["chemprop_aux"]
    delta_vs_nb1014      = pooled_mean - INDIV_REF["nb1014"]
    print(f"   tri-vote OOF (pooled mean of seeds)  = {pooled_mean:.4f}")
    print(f"   nb2060 ref OOF                       = {NB2060_REF_RAE:.4f}")
    print(f"   delta vs nb2060 (negative = better)  = {delta_vs_nb2060:+.4f}")
    print(f"   delta vs nb1158                      = {delta_vs_nb1158:+.4f}")
    print(f"   delta vs chemprop_aux                = {delta_vs_chemprop:+.4f}")
    print(f"   delta vs nb1014                      = {delta_vs_nb1014:+.4f}")
    print(f"   gate margin                          = -{GATE_MARGIN:.4f}")
    beats_nb2060 = delta_vs_nb2060 <= -GATE_MARGIN
    beats_nb1158 = delta_vs_nb1158 <= -GATE_MARGIN
    promote = bool(diversity_pass and distinct_pass and beats_nb2060)
    print(f"   beats nb2060 by margin               = {'YES' if beats_nb2060 else 'NO'}")
    print(f"   beats nb1158 (best indiv) by margin  = {'YES' if beats_nb1158 else 'NO'}")
    print(f"   diversity gate                       = {'PASS' if diversity_pass else 'FAIL'}")
    print(f"   distinctness gate                    = {'PASS' if distinct_pass else 'FAIL'}")
    print(f"   promote                              = {'YES' if promote else 'NO'}")

    # --- Save artefacts ---
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_struct_diverse_vote.csv"
    if promote:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (promote PASSED)")
    else:
        print(f"[skip] promote=NO -- no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "tri_vote_median_rank_anchor_nb1158_cdf_STRUCTURALLY_DIVERSE",
        "candidates": ["nb1158", "chemprop_aux", "nb1014"],
        "candidate_classes": {
            "nb1158":       "LGBM_K32_mean_bag_ECFP_RDKit",
            "chemprop_aux": "MPNN_GNN_multitask",
            "nb1014":       "multi_seed_bag_stack_SLSQP_rankstretch",
        },
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "indiv_oof_rae_unb": indiv,
        "indiv_ref_memo": INDIV_REF,
        "pairwise_spearman": pairwise,
        "max_pairwise_spearman": max_pairwise,
        "spearman_max_pairwise_gate": SPEARMAN_MAX_PAIRWISE,
        "diversity_pass": bool(diversity_pass),
        "diversity_note": ("chemprop_aux is a 76%-weight component of nb1014; "
                           "they share rank space" if not diversity_pass else "all three distinct"),
        "global_oof_rae_anchor_nb1158": rae_global,
        "spearman_vote_vs_nb1158":      rho_anchor_nb1158,
        "spearman_vote_vs_chemprop_aux": rho_anchor_chem,
        "spearman_vote_vs_nb1014":      rho_anchor_nb1014,
        "spearman_max_anchor_gate": SPEARMAN_MAX_ANCHOR,
        "distinct_pass": bool(distinct_pass),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_mean,
        "pooled_rae_std_seeds": pooled_std,
        "rae_of_mean_of_seed_oofs": rae_of_mean,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "nb2060_ref_oof_rae": NB2060_REF_RAE,
        "delta_vs_nb2060":      delta_vs_nb2060,
        "delta_vs_nb1158":      delta_vs_nb1158,
        "delta_vs_chemprop_aux": delta_vs_chemprop,
        "delta_vs_nb1014":      delta_vs_nb1014,
        "gate_margin": GATE_MARGIN,
        "beats_nb2060_by_margin": bool(beats_nb2060),
        "beats_nb1158_by_margin": bool(beats_nb1158),
        "promote": promote,
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if promote else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled OOF RAE (cross-fit)        = {pooled_mean:.4f}")
    print(f"   max pairwise Spearman             = {max_pairwise:.4f}")
    print(f"   rho(vote, nb1158)                 = {rho_anchor_nb1158:.4f}")
    print(f"   delta vs nb2060                   = {delta_vs_nb2060:+.4f}")
    print(f"   delta vs nb1158                   = {delta_vs_nb1158:+.4f}")
    print(f"   diversity / distinct / beat       = "
          f"{diversity_pass} / {distinct_pass} / {beats_nb2060}")
    print(f"   promote                           = {promote}")
    print(f"   wall                              = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "max_pairwise_spearman",
        "spearman_vote_vs_nb1158",
        "delta_vs_nb2060",
        "delta_vs_nb1158",
        "diversity_pass",
        "distinct_pass",
        "beats_nb2060_by_margin",
        "promote",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
