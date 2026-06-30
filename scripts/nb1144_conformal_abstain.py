"""nb1144 -- Per-tier Conformal-prediction abstention on scaffold-CV residuals.

Hypothesis
----------
nb1141 cross-fit shallow LGBM Huber residual on chemprop_aux on the 253 unblind
under 5-fold scaffold-shuffled KFold. The 5-seed mean_bag yields RAE 0.5753
(beats chemprop_aux 0.6216 by -0.046). The remaining error tail concentrates
on rare/novel scaffolds where the residual model itself has wide partition-wise
disagreement (per-fold OOF spread is the conformal score).

Per-tier conformal split quantiles (Romano CQR, symmetric form):
  - Tier 1 (rare scaffolds, scaf_train_freq <= q33)
  - Tier 2 (mid)
  - Tier 3 (common, scaf_train_freq >= q66)
For each tier, compute 90% PI width as the (1 - 0.10) * (1 + 1/n) quantile
of |residual_oof - residual_oof_seed_median| across the 5 seeds -- i.e. the
inductive cross-seed conformal score (split conformal at fold level for the
residual model). Tier-wise width then ranks all 253 unblind by PI width.

Abstention rule
---------------
Drop the top 20% widest PI rows (51/253); on those rows substitute the
nb730 anchor (te_nb730.npy[unb_idx]) for nb1141 mean_bag. The remaining
80% keep the nb1141 mean_bag (fine-residual). This is the standard
selective-prediction wrapper: trust the residual where it's tight, fall
back to the robust anchor where it's loose.

Anchor
------
  nb730 -- multi-seed null-ensemble (Phase-3 P3 winner, honest cross-fit
           RAE 0.4603 reference but here we use te_nb730.npy[unb_idx]
           which on the 253 yields an in-sample estimate).
  pred  -- nb1141 mean_bag_oof (chemprop_aux + shallow LGBM Huber residual,
           honest 5-fold cross-fit on 253, 5 seeds, RAE 0.5753).

Procedure
---------
1. Load y_unb (253), nb1141 per_seed_corrected (5,253) and mean_bag (253),
   nb730 te slice (253), and 253-row test metadata with scaf_train_freq.
2. Compute per-row conformal score s_i = mean over seeds of
   |corrected_seed_i - median_seeds|  (cross-seed dispersion).
3. Tier by scaf_train_freq: q33 / q66 splits; per tier compute the
   (1-alpha)*(1 + 1/n_tier) empirical quantile q_tier with alpha=0.10.
4. PI_width_i = 2 * q_tier(i)   (symmetric Romano-CQR interval).
5. Abstain top 20% by PI_width: on those rows replace nb1141 mean_bag with
   nb730 anchor; keep nb1141 elsewhere.
6. Report pooled RAE of combined predictor; compare to nb1141 mean_bag,
   nb730 anchor, and the nb2103 scaffold-CV reference 0.5057.
7. Sweep abstention fractions {0.05, 0.10, 0.20, 0.30, 0.40} for ablation.

Verdict against margin
----------------------
- WINS_VS_NB2103:        combined RAE < 0.5057 - 0.003
- TIES_VS_NB2103:        |combined RAE - 0.5057| <= 0.003
- LOSES_VS_NB2103:       combined RAE > 0.5057 + 0.003

Outputs
-------
  data/processed/nb1144_conformal_combined_oof.npy   (253,) float32
  data/processed/nb1144_pi_widths.npy                (253,) float32
  data/processed/nb1144_summary.json
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1144"
ALPHA = 0.10            # 90% PI
ABSTAIN_FRAC = 0.20     # top 20% widest -> anchor fallback
NB2103_REF = 0.5057
DECISION_MARGIN = 0.003

ANCHOR_TE_FILE = "te_nb730.npy"
ANCHOR_LABEL = "nb730"
RESID_LABEL = "nb1141_mean_bag"

ABSTAIN_SWEEP = [0.05, 0.10, 0.20, 0.30, 0.40]


def _conformal_q(scores: np.ndarray, alpha: float) -> float:
    """Inductive split-conformal quantile: ceil((n+1)*(1-alpha))/n empirical
    quantile of |scores|.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    # CQR-style finite-sample correction
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return float(np.quantile(np.abs(scores), level, method="higher"))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-tier conformal abstention "
          f"(alpha={ALPHA:.2f}, abstain top {ABSTAIN_FRAC:.0%})")
    print(f"          anchor fallback = {ANCHOR_LABEL} ({ANCHOR_TE_FILE})")
    print(f"          residual predictor = {RESID_LABEL} "
          f"(nb1141_mean_bag_oof.npy on 253)")
    print("=" * 78)

    # ---- Load 253 unblind truth and indices ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = {y_unb.shape}")

    # ---- Load nb1141 per-seed and mean_bag (residual predictor on 253) ----
    per_seed_path = DATA_PROCESSED / "nb1141_per_seed_corrected_oof.npy"
    mean_bag_path = DATA_PROCESSED / "nb1141_mean_bag_oof.npy"
    if not per_seed_path.exists() or not mean_bag_path.exists():
        raise FileNotFoundError(
            f"nb1141 outputs missing: {per_seed_path}, {mean_bag_path}"
        )
    per_seed = np.load(per_seed_path).astype(np.float64)   # (5, 253)
    nb1141_oof = np.load(mean_bag_path).astype(np.float64)  # (253,)
    if per_seed.shape != (5, n_unb):
        raise ValueError(f"per_seed shape {per_seed.shape} != (5, {n_unb})")
    rae_nb1141 = float(rae(y_unb, nb1141_oof))
    print(f"[load] nb1141_mean_bag RAE = {rae_nb1141:.4f}  (ref 0.5753)")

    # ---- Load nb730 anchor fallback (te file on 513; slice to 253) ----
    anchor_te_path = DATA_PROCESSED / ANCHOR_TE_FILE
    if not anchor_te_path.exists():
        raise FileNotFoundError(f"{anchor_te_path} missing")
    anchor_te = np.load(anchor_te_path).astype(np.float64)
    if anchor_te.shape[0] != 513:
        raise ValueError(f"{anchor_te_path} shape {anchor_te.shape} != 513")
    anchor_oof = anchor_te[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR_LABEL} (in-sample on 253 from te) RAE = "
          f"{rae_anchor:.4f}")

    # ---- Load scaffold support for the 253 ----
    test513_path = Path("data/processed/postmortem/pm_test_chem_all513.parquet")
    if not test513_path.exists():
        raise FileNotFoundError(f"{test513_path} missing")
    test513 = pd.read_parquet(test513_path)
    if "is_unblind" in test513.columns:
        unb_meta = test513[test513["is_unblind"]].reset_index(drop=True)
    else:
        unb_meta = test513.iloc[unb_idx].reset_index(drop=True)
    if len(unb_meta) != n_unb:
        raise ValueError(
            f"unb_meta n={len(unb_meta)} != y_unb n={n_unb}"
        )
    # scaf_train_freq is 197/253 zero -> degenerate terciles. Use continuous
    # nn_sim_train (top-1 train Tanimoto) as the OOD tiering feature, which
    # is the standard similarity-to-training-set conformalization axis.
    scaf_freq = unb_meta["scaf_train_freq"].astype(float).values
    nn_sim = unb_meta["nn_sim_train"].astype(float).values
    print(f"[meta] scaf_train_freq zero count = "
          f"{int((scaf_freq == 0).sum())}/{n_unb} -> degenerate; "
          f"tiering by nn_sim_train (top-1 train Tanimoto) instead")
    print(f"[meta] nn_sim_train: "
          f"min={nn_sim.min():.3f}  max={nn_sim.max():.3f}  "
          f"median={np.median(nn_sim):.3f}")

    # ---- Define 3 tiers by nn_sim_train terciles ----
    # (lower similarity == more OOD == rare-relative-to-train)
    q33, q66 = np.quantile(nn_sim, [1/3, 2/3])
    tier = np.zeros(n_unb, dtype=int)
    tier[nn_sim <= q33] = 0                              # rare (low sim)
    tier[(nn_sim > q33) & (nn_sim <= q66)] = 1           # mid
    tier[nn_sim > q66] = 2                                # common (high sim)
    tier_names = {0: "rare", 1: "mid", 2: "common"}
    for t in (0, 1, 2):
        n_t = int((tier == t).sum())
        if n_t == 0:
            print(f"[tier] {tier_names[t]:6s}: n=0  (empty)")
            continue
        print(f"[tier] {tier_names[t]:6s}: n={n_t:3d}  "
              f"nn_sim range "
              f"[{nn_sim[tier == t].min():.3f}, "
              f"{nn_sim[tier == t].max():.3f}]  "
              f"scaf_freq_mean={scaf_freq[tier == t].mean():.2f}")

    # ---- Conformal score: cross-seed dispersion of corrected prediction
    # i.e. |seed_pred_i - median_over_seeds| pooled over 5 seeds, then per-tier
    # quantile -> PI half-width. ----
    seed_median = np.median(per_seed, axis=0)  # (253,)
    cross_seed_dev = np.abs(per_seed - seed_median[None, :])  # (5, 253)
    # per-row mean absolute deviation (could also use max)
    nonconf_score_per_row = cross_seed_dev.mean(axis=0)  # (253,)

    # ---- Per-tier 90% conformal quantile (Romano CQR symmetric form) ----
    pi_halfwidth = np.zeros(n_unb, dtype=np.float64)
    tier_q: dict[int, float] = {}
    for t in (0, 1, 2):
        mask = (tier == t)
        # Use the pooled cross-seed deviations within the tier as the
        # conformal calibration set. This is the analog of split-CQR with
        # the residual cross-fit folds providing the calibration distribution.
        cal_scores = cross_seed_dev[:, mask].reshape(-1)
        q_t = _conformal_q(cal_scores, ALPHA)
        tier_q[t] = q_t
        pi_halfwidth[mask] = q_t
        print(f"[conf] tier={tier_names[t]:6s}  "
              f"cal_n={cal_scores.size}  q_{int((1-ALPHA)*100)}={q_t:.4f}")

    pi_width = 2.0 * pi_halfwidth
    np.save(DATA_PROCESSED / f"{TAG}_pi_widths.npy",
            pi_width.astype(np.float32))

    # ---- Abstention: top abstain_frac widest PI -> anchor fallback ----
    def _abstain(frac: float) -> tuple[np.ndarray, dict]:
        """Replace top frac widest rows of nb1141_oof with anchor_oof."""
        k = int(round(frac * n_unb))
        if k == 0:
            return nb1141_oof.copy(), dict(
                abstain_frac=frac, n_abstain=0, rae=rae_nb1141,
                delta_vs_nb1141=0.0, tier_abstain_counts={}
            )
        # widest first (highest PI_width); ties broken by index
        order = np.argsort(-pi_width, kind="mergesort")
        abst_idx = order[:k]
        combined = nb1141_oof.copy()
        combined[abst_idx] = anchor_oof[abst_idx]
        r = float(rae(y_unb, combined))
        # per-tier abstain counts
        tcnt = {tier_names[t]: int((tier[abst_idx] == t).sum())
                for t in (0, 1, 2)}
        return combined, dict(
            abstain_frac=frac, n_abstain=k, rae=r,
            delta_vs_nb1141=r - rae_nb1141,
            delta_vs_anchor=r - rae_anchor,
            delta_vs_nb2103=r - NB2103_REF,
            tier_abstain_counts=tcnt,
        )

    # primary 20% abstain
    combined_oof, primary_rec = _abstain(ABSTAIN_FRAC)
    np.save(DATA_PROCESSED / f"{TAG}_conformal_combined_oof.npy",
            combined_oof.astype(np.float32))
    rae_combined = primary_rec["rae"]
    print("\n" + "-" * 78)
    print("PRIMARY: top 20% widest PI -> nb730 fallback")
    print("-" * 78)
    print(f"   n_abstain      = {primary_rec['n_abstain']}/{n_unb}")
    print(f"   tier counts    = {primary_rec['tier_abstain_counts']}")
    print(f"   RAE(combined)  = {rae_combined:.4f}")
    print(f"   d_vs_nb1141    = {primary_rec['delta_vs_nb1141']:+.4f}")
    print(f"   d_vs_nb730     = {primary_rec['delta_vs_anchor']:+.4f}")
    print(f"   d_vs_nb2103    = {primary_rec['delta_vs_nb2103']:+.4f}  "
          f"(ref {NB2103_REF:.4f})")

    # sweep
    print("\n" + "-" * 78)
    print("SWEEP: abstention fraction grid")
    print("-" * 78)
    sweep_records = []
    for f in ABSTAIN_SWEEP:
        _, rec = _abstain(f)
        sweep_records.append(rec)
        print(f"   frac={f:.2f}  n={rec['n_abstain']:3d}  "
              f"RAE={rec['rae']:.4f}  "
              f"d_vs_nb1141={rec['delta_vs_nb1141']:+.4f}  "
              f"d_vs_nb2103={rec['delta_vs_nb2103']:+.4f}")
    best = min(sweep_records, key=lambda r: r["rae"])
    print(f"\n   best in sweep: frac={best['abstain_frac']:.2f}  "
          f"RAE={best['rae']:.4f}")

    # ---- Verdict vs nb2103 reference ----
    diff = rae_combined - NB2103_REF
    if diff < -DECISION_MARGIN:
        verdict = "WINS_VS_NB2103"
    elif diff <= DECISION_MARGIN:
        verdict = "TIES_VS_NB2103"
    else:
        verdict = "LOSES_VS_NB2103"
    print(f"\n   verdict (margin {DECISION_MARGIN:.3f}): {verdict}  "
          f"(combined {rae_combined:.4f} vs nb2103 {NB2103_REF:.4f})")

    summary = {
        "tag": TAG,
        "method": "per_tier_conformal_abstain_Romano_CQR_symmetric",
        "anchor_fallback": ANCHOR_LABEL,
        "anchor_te_file": ANCHOR_TE_FILE,
        "residual_predictor": RESID_LABEL,
        "residual_oof_file": "nb1141_mean_bag_oof.npy",
        "residual_per_seed_file": "nb1141_per_seed_corrected_oof.npy",
        "n_unb": n_unb,
        "alpha_pi": ALPHA,
        "tier_definitions": {
            "split_by": "nn_sim_train",
            "split_by_note": "scaf_train_freq is 197/253 zero (degenerate); "
                             "using top-1 train Tanimoto instead",
            "q33": float(q33),
            "q66": float(q66),
            "tier_counts": {
                tier_names[t]: int((tier == t).sum()) for t in (0, 1, 2)
            },
        },
        "tier_conformal_q90": {tier_names[t]: tier_q[t] for t in (0, 1, 2)},
        "rae_nb1141_mean_bag": rae_nb1141,
        "rae_nb730_anchor_in_sample": rae_anchor,
        "abstain_frac_primary": ABSTAIN_FRAC,
        "primary": primary_rec,
        "rae_combined_primary": rae_combined,
        "delta_combined_vs_nb1141": rae_combined - rae_nb1141,
        "delta_combined_vs_nb730": rae_combined - rae_anchor,
        "delta_combined_vs_nb2103": rae_combined - NB2103_REF,
        "nb2103_ref_rae": NB2103_REF,
        "decision_margin": DECISION_MARGIN,
        "verdict": verdict,
        "sweep_records": sweep_records,
        "sweep_best_frac": best["abstain_frac"],
        "sweep_best_rae": best["rae"],
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_conformal_combined_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_pi_widths.npy'}")
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_nb1141_mean_bag", "rae_nb730_anchor_in_sample",
        "rae_combined_primary", "delta_combined_vs_nb1141",
        "delta_combined_vs_nb730", "delta_combined_vs_nb2103",
        "verdict", "sweep_best_frac", "sweep_best_rae",
    ):
        print(f"  {k}: {res.get(k)}")
