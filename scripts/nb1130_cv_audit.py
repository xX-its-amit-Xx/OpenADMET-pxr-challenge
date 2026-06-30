"""nb1130 -- CV PROTOCOL AUDIT.

URGENT: nb2103's 0.4737 mean-bag / 0.4698 median-bag at K=28 was measured
under sklearn KFold(shuffle=True), NOT under scaffold-aware CV. CLAUDE.md
and src/pxr/eval.py mandate scaffold_kfold_indices as the honest project
protocol. Random KFold is ~0.05 RAE optimistic on this analog-expansion
test set (memory: feedback_scaffold_cv_misleads.md).

PROTOCOL:
    1. Reload the EXACT 117-col 5-way K-tuned matrix slice nb2103 used,
       restricted to top-K=28 SHAP indices (cached at
       data/processed/X_unb_28_nb2103.npy).
    2. Run identical 5-seed bag (seeds 0,1,7,42,137) of LGBM(MSE) with
       the IDENTICAL hyperparams nb2103 used, but swap KFold(shuffle=True)
       for scaffold_kfold_indices keyed on Bemis-Murcko scaffolds of the
       253 unblind SMILES.
    3. Report mean-bag and median-bag RAE under scaffold CV.
    4. Quantify optimism gap = random_KFold_rae - scaffold_CV_rae.
    5. Re-audit recent candidate scores (cycle 134/137/138/139) by
       grepping CV protocol in their source scripts; produce CV-protocol
       label per candidate.
    6. Compute revised LB transfer estimate accounting for scaffold-CV
       baseline.

Outputs:
    scripts/nb1130_cv_audit.py
    data/processed/nb1130_summary.json
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
from sklearn.model_selection import KFold
import lightgbm as lgb

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1130"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# nb2103 K=28 cached feature matrix on 253 unblind (top-28 SHAP slice)
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

# nb2103 reported numbers (random KFold)
NB2103_K28_MEAN_BAG_RANDOM = 0.4737
NB2103_K28_MEDIAN_BAG_RANDOM = 0.4698

# LB-transfer empirical correction (PRE-unblind)
LB_TRANSFER_PRE_UNBLIND_OFFSET = 0.003  # in_RAE -> LB calibration
# Random-KFold optimism (memory feedback_scaffold_cv_misleads.md): ~+0.05
RANDOM_KFOLD_OPTIMISM_ESTIMATE = 0.05


def _lgbm_params(seed: int) -> dict:
    """Match nb2103 / nb2081 / nb2063 hyperparams exactly."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_scaffold(X: np.ndarray, residual: np.ndarray,
                                  splits, seed: int) -> np.ndarray:
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError(f"NaN in OOF (seed={seed})")
    return oof


def _residual_cross_fit_random(X: np.ndarray, residual: np.ndarray,
                                seed: int) -> np.ndarray:
    """Verify nb2103 reproduction under random KFold."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _scan_cv_protocol(script_path: Path) -> dict:
    """Detect CV protocol used in a candidate script via static grep."""
    if not script_path.exists():
        return {"path": str(script_path), "exists": False}
    txt = script_path.read_text(errors="ignore")
    has_random_kfold = "KFold(n_splits" in txt and "shuffle=True" in txt
    has_scaffold = "scaffold_kfold_indices" in txt
    has_groupkfold = "GroupKFold" in txt
    if has_scaffold:
        protocol = "scaffold_CV"
    elif has_groupkfold:
        protocol = "GroupKFold"
    elif has_random_kfold:
        protocol = "random_KFold"
    else:
        protocol = "unknown"
    return {
        "path": str(script_path),
        "exists": True,
        "protocol": protocol,
        "has_random_kfold": has_random_kfold,
        "has_scaffold_kfold": has_scaffold,
        "has_groupkfold": has_groupkfold,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CV PROTOCOL AUDIT (scaffold vs random KFold)")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag(random) = "
          f"{NB2103_K28_MEAN_BAG_RANDOM:.4f}  "
          f"median_bag(random) = {NB2103_K28_MEDIAN_BAG_RANDOM:.4f}")
    print("=" * 78)

    # ---- Verify nb2103 protocol from source ----
    nb2103_path = Path(__file__).resolve().parent / "nb2103_shap_K_fine.py"
    nb2103_scan = _scan_cv_protocol(nb2103_path)
    print(f"\n[scan] nb2103 protocol: {nb2103_scan['protocol']}  "
          f"(random_kfold={nb2103_scan['has_random_kfold']}, "
          f"scaffold={nb2103_scan['has_scaffold_kfold']})")
    if nb2103_scan["protocol"] != "random_KFold":
        print(f"   [warn] expected random_KFold; got {nb2103_scan['protocol']}")

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor

    # ---- Load cached nb2103 K=28 feature matrix ----
    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(
            f"Need cached K=28 feature matrix: {X_UNB_28_PATH}  "
            "(written by nb2103 / nb2140)"
        )
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    if X_unb_28.shape != (n_unb, 28):
        raise ValueError(f"X_unb_28 shape mismatch: {X_unb_28.shape}")
    print(f"[load] X_unb_28 = {X_unb_28.shape}")

    # ---- Build scaffold splits on 253 unblind ----
    te_unb_smiles = [test_smiles[i] for i in unb_idx]
    scaffs = [bemis_murcko(s) for s in te_unb_smiles]
    n_unique_scaff = len(set(s for s in scaffs if s))
    n_none = sum(1 for s in scaffs if not s)
    print(f"[scaff] unique scaffolds = {n_unique_scaff}  none/empty = {n_none}")

    # 5 scaffold seeds (different random shuffles of scaffold groups)
    scaffold_splits_per_seed = {
        s: scaffold_kfold_indices(scaffs, n_splits=RESID_FOLDS, seed=s)
        for s in RESID_SEEDS
    }
    fold_sizes_seed0 = [len(va) for _, va in scaffold_splits_per_seed[0]]
    print(f"[scaff] fold sizes (seed=0): {fold_sizes_seed0}")

    # ---- Run scaffold-CV 5-seed bag ----
    print("\n" + "-" * 78)
    print("SCAFFOLD-CV 5-SEED BAG (HONEST PROTOCOL)")
    print("-" * 78)
    per_seed_corrected_scaffold = np.zeros((len(RESID_SEEDS), n_unb),
                                            dtype=np.float64)
    per_seed_rae_scaffold = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        splits = scaffold_splits_per_seed[s]
        resid_oof_s = _residual_cross_fit_scaffold(
            X_unb_28, residual, splits, s
        )
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected_scaffold[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae_scaffold.append(rae_s)
        print(f"   seed={s:3d}  scaffold_RAE = {rae_s:.4f}  "
              f"(d_vs_anchor = {rae_s - rae_anchor:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_scaffold = per_seed_corrected_scaffold.mean(axis=0)
    median_bag_scaffold = np.median(per_seed_corrected_scaffold, axis=0)
    rae_mean_bag_scaffold = float(rae(y_unb, mean_bag_scaffold))
    rae_median_bag_scaffold = float(rae(y_unb, median_bag_scaffold))
    print(f"\n   pooled mean_bag scaffold_RAE   = {rae_mean_bag_scaffold:.4f}")
    print(f"   pooled median_bag scaffold_RAE = {rae_median_bag_scaffold:.4f}")

    # ---- Reproduce nb2103 random-KFold for verification ----
    print("\n" + "-" * 78)
    print("RANDOM-KFOLD 5-SEED BAG (REPRODUCE nb2103)")
    print("-" * 78)
    per_seed_corrected_random = np.zeros((len(RESID_SEEDS), n_unb),
                                          dtype=np.float64)
    per_seed_rae_random = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_random(X_unb_28, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected_random[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae_random.append(rae_s)
        print(f"   seed={s:3d}  random_RAE = {rae_s:.4f}  "
              f"(d_vs_anchor = {rae_s - rae_anchor:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_random = per_seed_corrected_random.mean(axis=0)
    median_bag_random = np.median(per_seed_corrected_random, axis=0)
    rae_mean_bag_random = float(rae(y_unb, mean_bag_random))
    rae_median_bag_random = float(rae(y_unb, median_bag_random))
    print(f"\n   pooled mean_bag random_RAE    = {rae_mean_bag_random:.4f}  "
          f"(nb2103 reported {NB2103_K28_MEAN_BAG_RANDOM:.4f})")
    print(f"   pooled median_bag random_RAE  = {rae_median_bag_random:.4f}  "
          f"(nb2103 reported {NB2103_K28_MEDIAN_BAG_RANDOM:.4f})")

    # ---- Optimism gap ----
    optimism_mean = rae_mean_bag_scaffold - rae_mean_bag_random
    optimism_median = rae_median_bag_scaffold - rae_median_bag_random
    print("\n" + "-" * 78)
    print("OPTIMISM GAP (scaffold - random)")
    print("-" * 78)
    print(f"   mean_bag    optimism = {optimism_mean:+.4f}  "
          f"(scaffold {rae_mean_bag_scaffold:.4f} vs random "
          f"{rae_mean_bag_random:.4f})")
    print(f"   median_bag  optimism = {optimism_median:+.4f}  "
          f"(scaffold {rae_median_bag_scaffold:.4f} vs random "
          f"{rae_median_bag_random:.4f})")

    # ---- Audit recent candidate scripts ----
    print("\n" + "-" * 78)
    print("RECENT CANDIDATE CV-PROTOCOL AUDIT (cycles 134/137/138/139)")
    print("-" * 78)
    script_dir = Path(__file__).resolve().parent
    audit_scripts = [
        # cycle 134-ish (nb213x, nb214x, nb215x)
        "nb2131_pooled_25bag_K28L14.py",
        "nb2132_pooled_25bag_K28L12.py",
        "nb2133_lr_grid_K28.py",
        "nb2141_catboost_K28.py",
        "nb2142_lgbm_huber_K28.py",
        "nb2151_minchild_K28.py",
        "nb2153_featfrac_K28.py",
        # cycle 137-ish (nb216x)
        "nb2158_monotone_K28.py",
        "nb2160_dart_K28.py",
        "nb2163_pooled_aggregate.py",
        "nb2165_xgb_shap_K28.py",
        "nb2166_row_bootstrap_K28.py",
        "nb2167_sklearn_stack_K28.py",
        # cycle 138-ish (nb217x)
        "nb2170_anchor_nb730.py",
        "nb2172_family_ablation.py",
        "nb2173_residual_cascade.py",
        "nb2174_tanh_target.py",
        "nb2178_k_sweep_nb730.py",
        "nb2179_anchor_blend.py",
        # cycle 139-ish (nb218x / nb219x)
        "nb2183_nb730_honest.py",
        "nb2184_reeval_honest_anchor.py",
        "nb2185_alt_anchors.py",
        "nb2189_truly_honest.py",
        "nb2190_conformal_shrink.py",
        "nb2191_mlp_head.py",
    ]
    audits = []
    by_proto = {"scaffold_CV": 0, "GroupKFold": 0,
                "random_KFold": 0, "unknown": 0, "missing": 0}
    for s in audit_scripts:
        rec = _scan_cv_protocol(script_dir / s)
        if not rec["exists"]:
            by_proto["missing"] += 1
            audits.append({"script": s, "exists": False, "protocol": "missing"})
            print(f"   {s:42s}  MISSING")
            continue
        proto = rec["protocol"]
        by_proto[proto] = by_proto.get(proto, 0) + 1
        audits.append({"script": s, "exists": True, "protocol": proto})
        print(f"   {s:42s}  {proto}")
    print(f"\n   protocol counts: {by_proto}")

    # ---- Honest floor + revised LB transfer ----
    print("\n" + "-" * 78)
    print("REVISED LB TRANSFER ESTIMATE")
    print("-" * 78)
    # LB transfer model: LB ~ in_RAE + 0.003 for PRE-unblind random-KFold
    # If scaffold-CV is the honest baseline, revise:
    lb_est_from_random = rae_mean_bag_random + LB_TRANSFER_PRE_UNBLIND_OFFSET
    lb_est_from_scaffold = rae_mean_bag_scaffold + LB_TRANSFER_PRE_UNBLIND_OFFSET
    lb_est_pessimistic = rae_mean_bag_random + RANDOM_KFOLD_OPTIMISM_ESTIMATE
    print(f"   prior  LB est (random + 0.003)        = {lb_est_from_random:.4f}")
    print(f"   honest LB est (scaffold + 0.003)      = {lb_est_from_scaffold:.4f}")
    print(f"   pessim LB est (random + 0.05)         = {lb_est_pessimistic:.4f}")
    print(f"   observed optimism gap                 = {optimism_mean:+.4f}")

    # ---- Verdict ----
    if optimism_mean > 0.04:
        gap_verdict = "RANDOM_KFOLD_HIGHLY_OPTIMISTIC"
    elif optimism_mean > 0.015:
        gap_verdict = "RANDOM_KFOLD_MODERATELY_OPTIMISTIC"
    elif optimism_mean > 0.005:
        gap_verdict = "RANDOM_KFOLD_MARGINALLY_OPTIMISTIC"
    elif abs(optimism_mean) <= 0.005:
        gap_verdict = "RANDOM_KFOLD_AND_SCAFFOLD_AGREE"
    else:
        gap_verdict = "RANDOM_KFOLD_CONSERVATIVE"  # rare
    print(f"\n   gap verdict = {gap_verdict}")

    if rae_mean_bag_scaffold > 0.50:
        floor_verdict = "HONEST_FLOOR_ABOVE_0_50_PRIOR_FLOOR_UNDERESTIMATED"
    elif rae_mean_bag_scaffold > 0.4737 + 0.01:
        floor_verdict = "HONEST_FLOOR_ABOVE_NB2103_BY_>0_01"
    elif rae_mean_bag_scaffold > 0.4737 + 0.003:
        floor_verdict = "HONEST_FLOOR_ABOVE_NB2103_BY_3_TO_10MILLI"
    elif abs(rae_mean_bag_scaffold - 0.4737) <= 0.003:
        floor_verdict = "HONEST_FLOOR_MATCHES_NB2103_RANDOM_KFOLD"
    else:
        floor_verdict = "SCAFFOLD_CV_BEATS_RANDOM_KFOLD_UNUSUAL"
    print(f"   floor verdict = {floor_verdict}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "scaffold_CV_vs_random_KFold_audit_on_nb2103_K28",
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "K_audited": 28,
        "rae_anchor_chemprop_aux": rae_anchor,
        # nb2103 reproducibility
        "nb2103_protocol_scan": nb2103_scan,
        "nb2103_K28_mean_bag_reported_random": NB2103_K28_MEAN_BAG_RANDOM,
        "nb2103_K28_median_bag_reported_random": NB2103_K28_MEDIAN_BAG_RANDOM,
        "rae_mean_bag_random_reproduced": rae_mean_bag_random,
        "rae_median_bag_random_reproduced": rae_median_bag_random,
        "random_kfold_reproduction_delta_mean": (
            rae_mean_bag_random - NB2103_K28_MEAN_BAG_RANDOM
        ),
        # scaffold-CV honest numbers
        "rae_mean_bag_scaffold": rae_mean_bag_scaffold,
        "rae_median_bag_scaffold": rae_median_bag_scaffold,
        "per_seed_rae_scaffold": per_seed_rae_scaffold,
        "per_seed_rae_random": per_seed_rae_random,
        # gap
        "optimism_mean_bag": optimism_mean,
        "optimism_median_bag": optimism_median,
        "gap_verdict": gap_verdict,
        "floor_verdict": floor_verdict,
        # scaffold split topology
        "n_unique_scaffolds_in_253": int(n_unique_scaff),
        "fold_sizes_seed0": fold_sizes_seed0,
        # LB transfer
        "lb_transfer_pre_unblind_offset": LB_TRANSFER_PRE_UNBLIND_OFFSET,
        "lb_est_from_random_kfold": lb_est_from_random,
        "lb_est_from_scaffold_cv": lb_est_from_scaffold,
        "lb_est_pessimistic_random_plus_0_05": lb_est_pessimistic,
        # recent candidate audit
        "recent_candidate_audits": audits,
        "recent_candidate_protocol_counts": by_proto,
        # pre/post unblind taxonomy
        "lb_faithfulness_taxonomy": {
            "PRE_unblind_random_KFold": (
                "optimistic ~+0.05; in_RAE < honest cross-fit"
            ),
            "PRE_unblind_scaffold_CV": (
                "LB-faithful; honest project protocol per CLAUDE.md"
            ),
            "POST_unblind_any": (
                "untrustworthy per feedback_lb_two_regime_calibration"
            ),
        },
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K_audited",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_reported_random",
        "rae_mean_bag_random_reproduced",
        "rae_mean_bag_scaffold",
        "rae_median_bag_scaffold",
        "optimism_mean_bag",
        "optimism_median_bag",
        "gap_verdict",
        "floor_verdict",
        "lb_est_from_random_kfold",
        "lb_est_from_scaffold_cv",
        "recent_candidate_protocol_counts",
    ):
        print(f"  {k}: {res.get(k)}")
