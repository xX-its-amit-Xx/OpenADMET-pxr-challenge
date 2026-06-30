"""nb3210 -- DEEP-30 verification of nb3204 (equal-weight ensemble of clip winners).

CONTEXT:
    nb3204 reported pooled 15-seed RAE = 0.4418 with std = 0.0000 across all
    seeds. The pooled_rae is split-invariant for FIXED predictions (clipping
    is deterministic on the full 253), so std=0 is a methodology artifact,
    NOT a real dispersion measurement. nb3204 's only honest variance signal
    sits in `per_fold_val_rae_mean` (split-sensitive: 0.4548 +/- 0.0109).

    Per cycle-160 deep-30 rule: PRIMARY decisions must use 30-seed bag, and
    the metric must vary with kf_seed for std to mean anything. For an
    equal-weight ensemble with no fitted parameters, the LB-faithful honest
    metric is the per-fold-val-RAE mean (averaged across the 5 folds) -- it
    captures the deploy-time variance across scaffold-CV splits.

PROTOCOL:
    Components: nb3190 + nb3173 + nb3174 pred_oofs (1/3 equal-weight mean)
    Equal-weight (1/3) mean -> single pred_ens (253,) fixed
    For each kf_seed in {1216..1245} (30 FRESH seeds, disjoint from nb3204):
        scaffold_kfold_indices(n_splits=5, seed=kf_seed)
        For each fold:
            per_fold_val_rae = rae(y_unb[va_loc], pred_ens[va_loc])
        Record per_fold_val_rae_mean (5-fold mean, split-sensitive)
        Also record pooled_rae for cross-reference (will be ~constant)
    Aggregate per-fold-mean -> mean +/- std over 30 seeds.

GATE (per task, on per-fold-mean):
    per-fold-mean mean < 0.4426 -> "VERIFIED_PROMOTE_PRIMARY1"
    per-fold-mean mean < 0.4437 -> "VERIFIED_MARGINAL"
    shift > +0.01 vs 15-seed pf_mean -> "LUCKY_SEED_TRAP"
    else                       -> "REJECT"

References:
    nb3204 15-seed pooled = 0.4418 (std=0, artifact)
    nb3204 15-seed pf_mean = 0.4548 +/- 0.0109 (real variance)
    nb2171 prior PRIMARY-1 = 0.4682 (post-hoc-blend ceiling on nb730 anchor)

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3190_pred_oof.npy   (253,) float32
    data/processed/nb3173_pred_oof.npy   (253,) float32
    data/processed/nb3174_pred_oof.npy   (253,) float32
    data/processed/te_nb3190.npy         (513,) float32
    data/processed/te_nb3173.npy         (513,) float32
    data/processed/te_nb3174.npy         (513,) float32

Outputs:
    data/processed/nb3210_summary.json
    data/processed/nb3210_pred_oof.npy   (253,) float32 -- equal-weight mean
    data/processed/te_nb3210.npy         (513,) float32 -- equal-weight te mean
    submissions/nb3210_deep_verify_nb3204.csv  (only on VERIFIED_*)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3210"
VERIFY_TAG = "nb3204"
COMPONENTS = ["nb3190", "nb3173", "nb3174"]

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1246))  # 30 FRESH seeds, disjoint from nb3204 {1186..1200}

# -- Gates (per task, on per-fold-mean) ----------------------------------------
GATE_PROMOTE_PRIMARY1 = 0.4426
GATE_MARGINAL = 0.4437
GATE_LUCKY_SHIFT = 0.01

# -- References ----------------------------------------------------------------
REF_NB3204_POOLED = 0.4418       # split-invariant artifact (std=0)
REF_NB3204_PF_MEAN = 0.4548      # split-sensitive honest metric
REF_NB3204_PF_STD = 0.0109
REF_NB3190 = 0.4426
REF_NB3173 = 0.4422
REF_NB3174 = 0.4433
REF_NB2171 = 0.4682


def _eval_one_seed(
    pred_ens: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Eval equal-weight ensemble at a single kf_seed (no fitting).

    Returns BOTH pooled_rae (~constant) AND per-fold-mean (split-sensitive).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    fold_val_raes = []
    n_unb = len(y_unb)
    covered = np.zeros(n_unb, dtype=bool)
    fold_sizes = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        covered[va_loc] = True
        val_pred = pred_ens[va_loc]
        fold_sizes.append(int(len(va_loc)))
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
    if not covered.all():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, pred_ens))  # split-invariant for fixed predictions
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_val_raes": fold_val_raes,
        "fold_sizes": fold_sizes,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- DEEP-30 VERIFICATION of {VERIFY_TAG} "
        f"(equal-weight ensemble of clip winners)"
    )
    print(f"          components = {COMPONENTS}, weights = (1/3, 1/3, 1/3)")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}} (disjoint from nb3204 1186..1200)"
    )
    print(
        f"          primary metric: per-fold-val-RAE mean (split-sensitive)"
    )
    print(
        f"          gate: pf_mean < {GATE_PROMOTE_PRIMARY1:.4f} -> PROMOTE_PRIMARY1, "
        f"< {GATE_MARGINAL:.4f} -> MARGINAL, "
        f"shift > +{GATE_LUCKY_SHIFT:.3f} vs ref pf_mean -> LUCKY_SEED_TRAP"
    )
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load component pred_oofs + te files ----------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {len(COMPONENTS)} component pred_oof + te files")
    print("-" * 78)
    comp_oofs = []
    comp_tes = []
    comp_oof_raes = []
    for tag in COMPONENTS:
        oof_path = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        te_path = DATA_PROCESSED / f"te_{tag}.npy"
        oof_arr = np.load(oof_path).astype(np.float64)
        te_arr = np.load(te_path).astype(np.float64)
        if oof_arr.shape != (n_unb,):
            raise ValueError(
                f"{tag} pred_oof shape {oof_arr.shape} != ({n_unb},)"
            )
        if te_arr.shape != (n_test,):
            raise ValueError(f"{tag} te shape {te_arr.shape} != ({n_test},)")
        r = float(rae(y_unb, oof_arr))
        comp_oofs.append(oof_arr)
        comp_tes.append(te_arr)
        comp_oof_raes.append(r)
        print(
            f"   {tag}: oof_RAE={r:.4f}  "
            f"oof mean={oof_arr.mean():.3f} std={oof_arr.std():.3f}  "
            f"te mean={te_arr.mean():.3f} std={te_arr.std():.3f}"
        )

    # Pairwise correlations
    print("\n   pairwise OOF correlations:")
    for i in range(len(COMPONENTS)):
        for j in range(i + 1, len(COMPONENTS)):
            c = float(np.corrcoef(comp_oofs[i], comp_oofs[j])[0, 1])
            print(f"     {COMPONENTS[i]} <-> {COMPONENTS[j]}: {c:.4f}")

    # -- Build equal-weight ensemble (must match nb3204 byte-for-byte) --------
    print("\n" + "-" * 78)
    print("STEP 2: equal-weight (1/3) mean of pred_oof and te")
    print("-" * 78)
    pred_ens = np.mean(np.stack(comp_oofs, axis=0), axis=0)
    te_ens = np.mean(np.stack(comp_tes, axis=0), axis=0).astype(np.float32)
    ens_full_oof_rae = float(rae(y_unb, pred_ens))
    print(
        f"   pred_ens: oof_RAE={ens_full_oof_rae:.4f}  "
        f"mean={pred_ens.mean():.3f}  std={pred_ens.std():.3f}  "
        f"min={pred_ens.min():.3f}  max={pred_ens.max():.3f}"
    )
    print(
        f"   te_ens:   mean={te_ens.mean():.3f}  std={te_ens.std():.3f}  "
        f"min={te_ens.min():.3f}  max={te_ens.max():.3f}"
    )

    # Sanity: ensemble should byte-match nb3204_pred_oof if it exists
    nb3204_oof_path = DATA_PROCESSED / "nb3204_pred_oof.npy"
    if nb3204_oof_path.exists():
        nb3204_oof = np.load(nb3204_oof_path).astype(np.float64)
        max_diff = float(np.max(np.abs(pred_ens - nb3204_oof)))
        print(
            f"   sanity: max|pred_ens - nb3204_pred_oof| = {max_diff:.2e}"
        )
        if max_diff > 1e-5:
            print(
                f"   WARN: rebuild diverges from nb3204_pred_oof "
                f"(max diff {max_diff:.2e}); deep-verify is on REBUILD, not stored file"
            )

    # Leak sanity on ensemble
    leak_eq = float(np.mean(np.isclose(pred_ens, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Scaffolds for outer CV -----------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(
        f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    per_fold_stds = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _eval_one_seed(pred_ens, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_val_raes": [round(v, 4) for v in res["fold_val_raes"]],
            "fold_sizes": res["fold_sizes"],
        })
        print(
            f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"pf_std={res['per_fold_val_rae_std']:.4f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    # -- Aggregate ------------------------------------------------------------
    arr_pooled = np.asarray(pooled_raes, dtype=np.float64)
    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(arr_pf)

    pooled_mean = float(arr_pooled.mean())
    pooled_std = float(arr_pooled.std(ddof=1)) if n_s > 1 else 0.0
    pf_mean = float(arr_pf.mean())
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=29, two-sided 95% t_mult = 2.0452
    t_mult = 2.0452
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem
    pf_median = float(np.median(arr_pf))

    # Dispersion ratio vs nb3204's 15-seed pf_std
    dispersion_ratio_pf = pf_std / REF_NB3204_PF_STD if REF_NB3204_PF_STD > 0 else float("nan")
    shift_pf = pf_mean - REF_NB3204_PF_MEAN
    shift_pooled = pooled_mean - REF_NB3204_POOLED

    # one-sample t-test on pf_mean vs reference
    t_stat_pf = (
        shift_pf / pf_sem if pf_sem > 0 else float("nan")
    )

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   POOLED (split-invariant, should be ~constant):")
    print(f"     mean = {pooled_mean:.4f}   std = {pooled_std:.6f}")
    print(f"     min/max = [{arr_pooled.min():.4f}, {arr_pooled.max():.4f}]")
    print(f"     shift vs nb3204 pooled ({REF_NB3204_POOLED:.4f}) = {shift_pooled:+.6f}")
    print(f"\n   PER-FOLD-MEAN (split-sensitive, gate metric):")
    print(f"     mean              = {pf_mean:.4f}")
    print(f"     std               = {pf_std:.4f}")
    print(f"     sem               = {pf_sem:.4f}")
    print(f"     95% CI (df=29)    = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median            = {pf_median:.4f}")
    print(f"     min/max           = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")
    print(
        f"\n   ref nb3204 15-seed pf_mean = {REF_NB3204_PF_MEAN:.4f} +/- "
        f"{REF_NB3204_PF_STD:.4f}"
    )
    print(f"   shift vs nb3204 pf_mean    = {shift_pf:+.4f}")
    print(f"   dispersion ratio (deep/15) = {dispersion_ratio_pf:.2f}x")
    print(f"   one-sample t (pf)          = {t_stat_pf:.3f}")
    print(
        f"\n   ref components: nb3190 {REF_NB3190:.4f}  "
        f"nb3173 {REF_NB3173:.4f}  nb3174 {REF_NB3174:.4f}"
    )
    print(f"   ref nb2171 PRIMARY-1   = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf)    = {REF_NB2171 - pf_mean:+.4f}")

    # -- Gate (on per-fold-mean) ---------------------------------------------
    print("\n" + "-" * 78)
    print("GATE  (metric = per-fold-val-RAE mean over 30 seeds)")
    print("-" * 78)
    if shift_pf > GATE_LUCKY_SHIFT:
        verdict = "LUCKY_SEED_TRAP"
        ladder_action = (
            f"REJECT - LUCKY-SEED TRAP. nb3210 deep-30 per-fold-mean "
            f"{pf_mean:.4f} shifted +{shift_pf:.4f} vs nb3204 15-seed "
            f"pf_mean {REF_NB3204_PF_MEAN:.4f} (> +{GATE_LUCKY_SHIFT:.3f} "
            f"threshold). nb3204's variance picture was under-dispersed by "
            f"{dispersion_ratio_pf:.2f}x. Same root cause as nb1086 / nb2060 "
            f"/ nb2095. The pooled_rae=0.4418 with std=0 was a methodology "
            f"artifact (predictions fixed -> pooled is split-invariant); the "
            f"only honest variance signal lives in per-fold-mean, which "
            f"shifts up materially on a deeper bag. DROP nb3204 from BETTER "
            f"band; keep current ladder (nb2171 0.4682 PRIMARY-1)."
        )
    elif pf_mean < GATE_PROMOTE_PRIMARY1:
        verdict = "VERIFIED_PROMOTE_PRIMARY1"
        ladder_action = (
            f"PROMOTE-PRIMARY1. nb3210 deep-30 per-fold-mean {pf_mean:.4f} "
            f"clears strict gate {GATE_PROMOTE_PRIMARY1:.4f} AND is within "
            f"+{GATE_LUCKY_SHIFT:.3f} of nb3204 15-seed reference "
            f"(shift {shift_pf:+.4f}). Dispersion ratio {dispersion_ratio_pf:.2f}x. "
            f"Equal-weight ensemble of clip winners verified -- variance "
            f"reduction on near-correlated (>0.999) components carries to "
            f"deeper bag. PROMOTE nb3204/nb3210 to PRIMARY-1 candidate, "
            f"displaces nb2171 (0.4682 -> {pf_mean:.4f}, "
            f"-{REF_NB2171 - pf_mean:.4f} RAE)."
        )
    elif pf_mean < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
        ladder_action = (
            f"VERIFIED-MARGINAL. nb3210 deep-30 per-fold-mean {pf_mean:.4f} "
            f"> strict gate {GATE_PROMOTE_PRIMARY1:.4f} but < MARGINAL gate "
            f"{GATE_MARGINAL:.4f}. Shift vs nb3204 = {shift_pf:+.4f} (within "
            f"lucky band). Dispersion ratio {dispersion_ratio_pf:.2f}x. "
            f"Operator carries to deep-30 at MARGINAL band but doesn't reach "
            f"the BETTER threshold. Keep as ALTERNATE, prefer nb2171 "
            f"PRIMARY-1 until LB re-grade confirms."
        )
    else:
        verdict = "REJECT"
        ladder_action = (
            f"REJECT. nb3210 deep-30 per-fold-mean {pf_mean:.4f} fails both "
            f"gates (PROMOTE {GATE_PROMOTE_PRIMARY1:.4f}, MARGINAL "
            f"{GATE_MARGINAL:.4f}). Shift vs nb3204 = {shift_pf:+.4f}. "
            f"Even though not flagged LUCKY-SEED-TRAP (< +{GATE_LUCKY_SHIFT:.3f}), "
            f"the deep-30 per-fold-mean lands above MARGINAL ceiling. "
            f"Equal-weight averaging of near-identical clip winners "
            f"(corr > 0.999) does not produce enough variance reduction. "
            f"Keep nb2171 PRIMARY-1; drop nb3204/nb3210 from BETTER band."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_ens.astype(np.float32))
    np.save(te_path, te_ens)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_deep_verify_nb3204.csv"
    if verdict.startswith("VERIFIED"):
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_ens,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "verify_tag": VERIFY_TAG,
        "method": "deep_30_verify_of_nb3204_equal_weight_ensemble",
        "components": COMPONENTS,
        "weights": [round(1.0 / len(COMPONENTS), 6)] * len(COMPONENTS),
        "component_oof_rae": {
            COMPONENTS[i]: round(comp_oof_raes[i], 4)
            for i in range(len(COMPONENTS))
        },
        "pairwise_corr": {
            f"{COMPONENTS[i]}_{COMPONENTS[j]}": round(
                float(np.corrcoef(comp_oofs[i], comp_oofs[j])[0, 1]), 4
            )
            for i in range(len(COMPONENTS))
            for j in range(i + 1, len(COMPONENTS))
        },
        "anchor_pre_unblind": True,
        "ens_full_oof_rae": round(ens_full_oof_rae, 4),
        "ens_leak_eq_truth_frac": round(leak_eq, 4),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_val_rae_means_array": [
            round(float(v), 4) for v in per_fold_means
        ],
        "per_fold_val_rae_stds_array": [
            round(float(v), 4) for v in per_fold_stds
        ],
        # Primary gate metric: per-fold-mean across 30 seeds
        "pf_mean": round(pf_mean, 4),
        "pf_std": round(pf_std, 4),
        "pf_sem": round(pf_sem, 4),
        "pf_ci95_low": round(pf_ci_low, 4),
        "pf_ci95_high": round(pf_ci_high, 4),
        "pf_median": round(pf_median, 4),
        "pf_min": round(float(arr_pf.min()), 4),
        "pf_max": round(float(arr_pf.max()), 4),
        # Pooled (artifact, ~constant)
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 6),
        "pooled_min": round(float(arr_pooled.min()), 4),
        "pooled_max": round(float(arr_pooled.max()), 4),
        "ref_nb3204_pooled": REF_NB3204_POOLED,
        "ref_nb3204_pf_mean": REF_NB3204_PF_MEAN,
        "ref_nb3204_pf_std": REF_NB3204_PF_STD,
        "shift_pf_vs_nb3204": round(shift_pf, 4),
        "shift_pooled_vs_nb3204": round(shift_pooled, 6),
        "dispersion_ratio_pf": round(dispersion_ratio_pf, 3),
        "one_sample_t_pf": round(float(t_stat_pf), 3) if not np.isnan(t_stat_pf) else None,
        "ref_nb3190": REF_NB3190,
        "ref_nb3173": REF_NB3173,
        "ref_nb3174": REF_NB3174,
        "ref_nb2171": REF_NB2171,
        "gain_pf_vs_nb2171": round(REF_NB2171 - pf_mean, 4),
        "te_mean": float(te_ens.mean()),
        "te_std": float(te_ens.std()),
        "te_min": float(te_ens.min()),
        "te_max": float(te_ens.max()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict.startswith("VERIFIED") else None
        ),
        "gate_promote_primary1": GATE_PROMOTE_PRIMARY1,
        "gate_marginal": GATE_MARGINAL,
        "gate_lucky_shift": GATE_LUCKY_SHIFT,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pf_mean ({n_s} seeds) = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   95% CI               = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   shift vs nb3204 pf   = {shift_pf:+.4f}")
    print(f"   dispersion ratio     = {dispersion_ratio_pf:.2f}x")
    print(f"   pooled (constant)    = {pooled_mean:.4f}")
    print(f"   verdict              = {verdict}")
    print(f"   wall                 = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pf_mean", "pf_std", "pf_ci95_low", "pf_ci95_high",
        "shift_pf_vs_nb3204", "dispersion_ratio_pf",
        "pooled_mean", "pooled_std",
        "gain_pf_vs_nb2171",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
