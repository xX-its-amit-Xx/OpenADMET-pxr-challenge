"""nb2076 -- Tri-vote majority-rank blend of {nb1191, nb2060, nb1158}.

Mechanism (per-row, applied independently on the 253 unblind OOF set and the
513 deploy set):
    1. For each of the 3 candidates, compute the rank of each prediction
       within that candidate's prediction set.
    2. Element-wise MEDIAN of the 3 rank vectors (rank_vote in [1..N]).
    3. Map rank_vote back to pec50 via the empirical CDF of nb1191's
       prediction set (anchor CDF). This produces a rank-stable, anchor-
       calibrated pec50 vector that is monotone in the median rank.

Why this is distinct from any of the 3 inputs: median-of-ranks is a robust
order statistic; a row only sits at rank r in the vote if it is at or near
rank r in at least 2 of the 3 candidates. Disagreements get pulled toward
the consensus rank, not the consensus value.

Spearman gate: rho(rank_vote_unb, nb1191_unb) must be <= 0.98, otherwise
the vote is just nb1191 in disguise.

Honest evaluation: scaffold 5-fold CV on the 253 unblind. Since the vote
mechanism is non-parametric (no trained weights), we apply it identically
inside every fold; the only fold-dependent piece is whether we want a
fold-specific anchor CDF. We use the GLOBAL nb1191 unb CDF for anchor
(no fold leakage -- the CDF is built from nb1191 PREDICTIONS, not labels),
then evaluate RAE on the unb labels per-fold-pooled.

Compare vs nb2060 honest cross-fit RAE = 0.4697 (memo). Decision margin
0.003 -- promote only if tri_vote OOF RAE <= 0.4667.

Inputs (must exist):
    OOFs (n_unb=253):
        nb1158 -> nb1158_mean_bag_oof_K32.npy
        nb1191 -> reconstructed from cycle-147 deploy weights
                  (chemprop_aux/nb1150/nb1158_K32/nb2112_K28) +
                  per-seed mean stretch
        nb2060 -> reconstructed from cycle-158 deploy weights
                  (6 anchors, nb730 EXCLUDED) + per-seed mean stretch

    te (n_te=513):
        te_nb1158.npy, te_nb1191.npy, te_nb2060.npy (all cached)

Outputs:
    scripts/nb2076_tri_vote.py (this file)
    data/processed/nb2076_summary.json
    data/processed/te_nb2076.npy
    submissions/nb2076_tri_vote.csv  (ONLY if beats nb2060 by >= 0.003)
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

TAG = "nb2076"
GATE_MARGIN = 0.003
NB2060_REF_RAE = 0.4697
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
SPEARMAN_MAX = 0.98


def empirical_cdf_map(anchor_values: np.ndarray, target_ranks: np.ndarray,
                      n_anchor: int) -> np.ndarray:
    """Map rank vector (1..n_anchor) back to pec50 via sorted anchor values.

    Linear interpolation between anchor-rank positions; rank is interpreted
    as a continuous quantile position in the anchor sorted array.
    """
    anchor_sorted = np.sort(anchor_values)
    # target_ranks in [1, n_anchor]; convert to 0-indexed continuous positions
    pos = np.clip(target_ranks - 1.0, 0.0, n_anchor - 1.0)
    lo = np.floor(pos).astype(int)
    hi = np.clip(lo + 1, 0, n_anchor - 1)
    frac = pos - lo
    return (1.0 - frac) * anchor_sorted[lo] + frac * anchor_sorted[hi]


def tri_vote(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
             anchor_values: np.ndarray) -> np.ndarray:
    """Per-row median rank of 3 vectors, mapped back via anchor CDF.

    All four vectors must be the same length; anchor_values defines the
    empirical CDF used to convert the median rank to pec50.
    """
    assert p1.shape == p2.shape == p3.shape == anchor_values.shape, \
        f"shape mismatch: {p1.shape} {p2.shape} {p3.shape} {anchor_values.shape}"
    n = len(p1)
    r1 = rankdata(p1, method="average")
    r2 = rankdata(p2, method="average")
    r3 = rankdata(p3, method="average")
    rank_vote = np.median(np.column_stack([r1, r2, r3]), axis=1)
    return empirical_cdf_map(anchor_values, rank_vote, n), rank_vote


# ---------------------------------------------------------------------------
# Reconstruction of nb1191 and nb2060 OOF/te using their cached deploy weights
# ---------------------------------------------------------------------------

NB1191_DEPLOY_W = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_MU = 4.665121133673035
NB1191_S = 1.031

NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_TES = [
    "te_chemprop_aux.npy",
    "te_nb503.npy",
    "te_nb1014.npy",
    "te_nb2112.npy",
]
NB1150_SLSQP4_W = [0.0, 0.2942, 0.0, 0.7058]

NB2060_DEPLOY_W = {
    "chemprop_aux": 6.979568781588102e-13,
    "nb1150":       0.0,
    "nb1158_K32":   0.2175344331249161,
    "nb2112_K28":   0.558837768814273,
    "nb503":        0.0,
    "nb562":        0.22362779806011301,
}
NB2060_MU = 4.666365275994977
NB2060_S = 1.015


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        if not p.exists():
            # fallback: use deploy te[unb_idx] (less ideal -- in-sample)
            raise FileNotFoundError(f"missing nb1150 anchor OOF: {p}")
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_W, dtype=np.float64)
    return P @ w


def reconstruct_nb1150_te(n_te: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_TES:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 anchor te: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_te,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_W, dtype=np.float64)
    return P @ w


def reconstruct_nb1191(n_unb: int, n_te: int):
    """Returns (oof_253, te_513) for nb1191 via cached deploy weights."""
    # OOF anchors
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    cpa_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    cols_oof = [cpa_oof, nb1150_oof, nb1158_oof, nb2112_oof]
    w = np.array([
        NB1191_DEPLOY_W["chemprop_aux"],
        NB1191_DEPLOY_W["nb1150"],
        NB1191_DEPLOY_W["nb1158_K32"],
        NB1191_DEPLOY_W["nb2112_K28"],
    ])
    blend_oof = np.column_stack(cols_oof) @ w
    oof_253 = NB1191_MU + NB1191_S * (blend_oof - NB1191_MU)
    # te
    cpa_te = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    nb1150_te = reconstruct_nb1150_te(n_te)
    nb1158_te = np.load(DATA_PROCESSED / "te_nb1158.npy").astype(np.float64)
    nb2112_te = np.load(DATA_PROCESSED / "te_nb2112.npy").astype(np.float64)
    blend_te = np.column_stack([cpa_te, nb1150_te, nb1158_te, nb2112_te]) @ w
    te_513 = NB1191_MU + NB1191_S * (blend_te - NB1191_MU)
    return oof_253, te_513


def reconstruct_nb2060(n_unb: int, n_te: int):
    """Returns (oof_253, te_513) for nb2060 via cached deploy weights."""
    cpa_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    cols_oof = [cpa_oof, nb1150_oof, nb1158_oof, nb2112_oof, nb503_oof, nb562_oof]
    w = np.array([
        NB2060_DEPLOY_W["chemprop_aux"],
        NB2060_DEPLOY_W["nb1150"],
        NB2060_DEPLOY_W["nb1158_K32"],
        NB2060_DEPLOY_W["nb2112_K28"],
        NB2060_DEPLOY_W["nb503"],
        NB2060_DEPLOY_W["nb562"],
    ])
    blend_oof = np.column_stack(cols_oof) @ w
    oof_253 = NB2060_MU + NB2060_S * (blend_oof - NB2060_MU)
    # te (use cached deploy)
    te_513 = np.load(DATA_PROCESSED / "te_nb2060.npy").astype(np.float64)
    return oof_253, te_513


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- tri-vote majority-rank of {{nb1191, nb2060, nb1158}}")
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

    # --- Load/reconstruct the 3 candidates ---
    print("\n[reconstruct] candidates")
    nb1191_oof, nb1191_te = reconstruct_nb1191(n_unb, n_te)
    nb2060_oof, nb2060_te = reconstruct_nb2060(n_unb, n_te)
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb1158_te = np.load(DATA_PROCESSED / "te_nb1158.npy").astype(np.float64)

    indiv = {
        "nb1191": float(rae(y_unb, nb1191_oof)),
        "nb2060": float(rae(y_unb, nb2060_oof)),
        "nb1158": float(rae(y_unb, nb1158_oof)),
    }
    print(f"   nb1191 oof_RAE={indiv['nb1191']:.4f}   "
          f"te(513) mean={nb1191_te.mean():.3f} std={nb1191_te.std():.3f}")
    print(f"   nb2060 oof_RAE={indiv['nb2060']:.4f}   "
          f"te(513) mean={nb2060_te.mean():.3f} std={nb2060_te.std():.3f}")
    print(f"   nb1158 oof_RAE={indiv['nb1158']:.4f}   "
          f"te(513) mean={nb1158_te.mean():.3f} std={nb1158_te.std():.3f}")

    # Sanity: reconstructed nb1191/nb2060 should roughly match the memo'd
    # rae_of_mean_of_seed_oofs (single-seed deploy weights -> single OOF
    # vector; the memo number is across 5 fold-seeds so they will not match
    # exactly, but should be in the ballpark)
    print(f"   [audit] nb1191 reconstructed OOF RAE = {indiv['nb1191']:.4f} "
          f"(memo rae_of_mean_of_seed_oofs = 0.4697)")
    print(f"   [audit] nb2060 reconstructed OOF RAE = {indiv['nb2060']:.4f} "
          f"(memo rae_of_mean_of_seed_oofs = 0.4693)")

    # --- Tri-vote on full 253 (deploy diagnostic; anchor CDF = nb1191 unb) ---
    vote_oof_global, rank_vote_unb = tri_vote(
        nb1191_oof, nb2060_oof, nb1158_oof, nb1191_oof,
    )
    rae_global = float(rae(y_unb, vote_oof_global))
    rho_nb1191 = float(spearmanr(rank_vote_unb,
                                 rankdata(nb1191_oof)).correlation)
    rho_nb2060 = float(spearmanr(rank_vote_unb,
                                 rankdata(nb2060_oof)).correlation)
    rho_nb1158 = float(spearmanr(rank_vote_unb,
                                 rankdata(nb1158_oof)).correlation)
    print(f"\n[global] tri-vote OOF RAE (anchor=nb1191)   = {rae_global:.4f}")
    print(f"[global] Spearman(rank_vote, nb1191)         = {rho_nb1191:.4f}")
    print(f"[global] Spearman(rank_vote, nb2060)         = {rho_nb2060:.4f}")
    print(f"[global] Spearman(rank_vote, nb1158)         = {rho_nb1158:.4f}")

    # Sanity gate -- must be distinct from nb1191
    sanity_pass = rho_nb1191 <= SPEARMAN_MAX
    print(f"[sanity] rho(vote, nb1191) {rho_nb1191:.4f} <= "
          f"{SPEARMAN_MAX} -> {'PASS' if sanity_pass else 'FAIL'}")

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
            # On the validation rows only -- compute ranks WITHIN va, vote
            # using nb1191's va values as anchor CDF. This avoids any
            # leakage from training labels into the va prediction.
            vote_va, _ = tri_vote(
                nb1191_oof[va_loc],
                nb2060_oof[va_loc],
                nb1158_oof[va_loc],
                nb1191_oof[va_loc],
            )
            oof_blend[va_loc] = vote_va
        pooled = float(rae(y_unb, oof_blend))
        per_seed.append({"kf_seed": int(kf_seed), "pooled_rae": pooled})
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}")
    pooled_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean = float(rae(y_unb, mean_oof))
    print(f"\n[cv] pooled_RAE (mean of seeds) = {pooled_mean:.4f} "
          f"(+/- {pooled_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs   = {rae_of_mean:.4f}")

    # --- Deploy: tri-vote on 513, anchor = nb1191 te(513) ---
    vote_te, rank_vote_te = tri_vote(
        nb1191_te, nb2060_te, nb1158_te, nb1191_te,
    )
    deploy_te = vote_te.astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] te(513) tri-vote: mean={deploy_te.mean():.3f} "
          f"std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")

    # --- Decision gate ---
    print("\n" + "-" * 78)
    print("DECISION GATE")
    print("-" * 78)
    delta_vs_nb2060 = pooled_mean - NB2060_REF_RAE
    print(f"   tri-vote OOF (pooled mean of seeds) = {pooled_mean:.4f}")
    print(f"   nb2060 ref OOF                      = {NB2060_REF_RAE:.4f}")
    print(f"   delta (negative = better)           = {delta_vs_nb2060:+.4f}")
    print(f"   gate margin                         = -{GATE_MARGIN:.4f}")
    beats = delta_vs_nb2060 <= -GATE_MARGIN
    promote = bool(sanity_pass and beats)
    print(f"   beats nb2060 by margin              = "
          f"{'YES' if beats else 'NO'}")
    print(f"   sanity gate (rho <= {SPEARMAN_MAX})           = "
          f"{'PASS' if sanity_pass else 'FAIL'}")
    print(f"   promote                             = "
          f"{'YES' if promote else 'NO'}")

    # --- Save artefacts ---
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_tri_vote.csv"
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
        "method": "tri_vote_median_rank_anchor_nb1191_cdf",
        "candidates": ["nb1191", "nb2060", "nb1158"],
        "candidate_cycles": {"nb1191": 147, "nb2060": 158, "nb1158": 143},
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "indiv_oof_rae_unb": indiv,
        "global_oof_rae_anchor_nb1191": rae_global,
        "spearman_vote_vs_nb1191": rho_nb1191,
        "spearman_vote_vs_nb2060": rho_nb2060,
        "spearman_vote_vs_nb1158": rho_nb1158,
        "sanity_max_spearman": SPEARMAN_MAX,
        "sanity_pass": bool(sanity_pass),
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
        "delta_vs_nb2060": delta_vs_nb2060,
        "gate_margin": GATE_MARGIN,
        "beats_nb2060_by_margin": bool(beats),
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
    print(f"   pooled OOF RAE (cross-fit) = {pooled_mean:.4f}")
    print(f"   delta vs nb2060            = {delta_vs_nb2060:+.4f}")
    print(f"   sanity rho_nb1191          = {rho_nb1191:.4f}")
    print(f"   promote                    = {promote}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "delta_vs_nb2060",
        "spearman_vote_vs_nb1191",
        "sanity_pass",
        "beats_nb2060_by_margin",
        "promote",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
