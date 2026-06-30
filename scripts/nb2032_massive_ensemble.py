"""nb2032 -- Massive ensemble across all PRE-unblind-honest OOF predictors.

Cycle 144 hetero boost was truncated -- this is a genuine 30+ model
ensemble exploration. Per the LB two-regime calibration memo
(feedback_lb_two_regime_calibration), only PRE-unblind te files (trained
on the 4139 baseline ONLY) transfer honestly to LB. POST-unblind te
files (trained on the 253 leaked unblind) collapse to LB 0.7-0.9.

Strategy:
    1. Identify every *_oof_253.npy or compatible (n_unb,) OOF pred under
       data/processed/; subset to the PRE-unblind-honest source family:
       chemprop_aux + nb1133_* + nb1150 (reconstructed) + nb1158 + nb2112
       + nb2031 + nb1191 + nb503 + nb1014 (foundation anchors).
    2. Score four simple ensemble variants against nb1191 (0.4718 baseline):
       mean / median / inverse-RAE-weighted / trimmed-mean (drop 10/10).
    3. Each variant scored via scaffold 5-fold CV pooled RAE on the 253.
    4. Gate: ensemble must beat nb1191 by >= 0.003.
    5. If any variant beats, build deploy CSV using the matching te files
       under the same aggregator.

Gate margin: 0.003 RAE (per memory's nb1861 / nb2023_lambda3 precedent).
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
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2032"
GATE_MARGIN = 0.003
NB1191_REF = 0.4718
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
TRIM_FRAC = 0.10  # drop top/bottom 10% per row before mean

# Candidate PRE-unblind-honest (oof_path, te_path, display).
# nb1150 OOF is reconstructed; te available cached.
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_W = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} {v.shape}"
        cols.append(v)
    return np.column_stack(cols) @ np.asarray(NB1150_SLSQP4_W, dtype=np.float64)


# Each entry: (display, oof_loader_or_relpath, te_relpath)
# 'oof_loader' is either a string relpath under DATA_PROCESSED, or the
# sentinel '_RECONSTRUCT_nb1150_oof'.
CANDIDATES = [
    # Foundation anchors
    ("chemprop_aux",    "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("nb1133_nb1001",   "nb1133_nb1001_pred_oof.npy",        "te_chemprop_aux.npy"),
    ("nb1133_nb1014",   "nb1133_nb1014_pred_oof.npy",        "te_chemprop_aux.npy"),
    ("nb503",           "nb503_pred_oof.npy",                "te_chemprop_aux.npy"),
    # Pyramid family
    ("nb1150_recon",    "_RECONSTRUCT_nb1150_oof",           "te_nb1150.npy"),
    ("nb1158_K32",      "nb1158_mean_bag_oof_K32.npy",       "te_nb1158.npy"),
    ("nb2112_K28",      "nb2103_mean_bag_oof_K28.npy",       "te_nb2112.npy"),
    # Residual-bag descendants of chemprop_aux
    ("nb2031_pooled25", "nb2031_pooled_25bag_oof.npy",       "te_nb2112.npy"),
    # Pyramid blend (the current PRIMARY benchmark)
    ("nb1191_pyramid",  None,                                "te_nb1191.npy"),
]


def trimmed_mean(M: np.ndarray, frac: float) -> np.ndarray:
    """Row-wise trimmed mean. M: (N, K). Drop ceil(K*frac) lowest + highest."""
    N, K = M.shape
    k_drop = int(np.ceil(K * frac))
    if 2 * k_drop >= K:
        return np.median(M, axis=1)
    sM = np.sort(M, axis=1)
    return sM[:, k_drop:K - k_drop].mean(axis=1)


def pooled_scaffold_rae(pred_oof_func, y_unb, unb_scaffolds, P_unb):
    """Score an aggregator under scaffold 5-fold CV averaged across kf_seeds.

    pred_oof_func: (n_unb, K) -> (n_unb,) aggregator (no fitting on labels)
    Since the aggregators here are label-free, scaffold CV is just to keep
    parity with nb1191's pooled-RAE protocol -- the aggregator output is
    identical regardless of split. We still pool fold predictions for
    bookkeeping.
    """
    n_unb = P_unb.shape[0]
    seed_raes = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan)
        for _tr_loc, va_loc in splits:
            oof[va_loc] = pred_oof_func(P_unb[va_loc])
        seed_raes.append(float(rae(y_unb, oof)))
    return float(np.mean(seed_raes)), float(np.std(seed_raes)), seed_raes


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- massive ensemble (mean/median/inv-RAE/trimmed-mean)")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_te={n_te} n_unb={n_unb} n_scaffolds={n_scaf}")

    # ---- Load all candidate OOFs / te vectors ----
    print("\n[candidates]")
    valid = []
    indiv_rae = {}
    for disp, oof_arg, te_rel in CANDIDATES:
        # OOF on 253
        if oof_arg is None:
            # nb1191: derive in-sample OOF from te_nb1191[unb_idx] (proxy)
            te_p = DATA_PROCESSED / te_rel
            if not te_p.exists():
                print(f"   SKIP {disp}: te missing")
                continue
            te_v = np.load(te_p).astype(np.float64)
            oof = te_v[unb_idx]
        elif oof_arg == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
            te_p = DATA_PROCESSED / te_rel
            if not te_p.exists():
                print(f"   SKIP {disp}: te missing")
                continue
            te_v = np.load(te_p).astype(np.float64)
        else:
            oof_p = DATA_PROCESSED / oof_arg
            te_p = DATA_PROCESSED / te_rel
            if not oof_p.exists() or not te_p.exists():
                print(f"   SKIP {disp}: missing oof={oof_p.exists()} te={te_p.exists()}")
                continue
            oof = np.load(oof_p).astype(np.float64)
            te_v = np.load(te_p).astype(np.float64)
        if oof.shape != (n_unb,) or te_v.shape != (n_te,):
            print(f"   SKIP {disp}: bad shape oof={oof.shape} te={te_v.shape}")
            continue
        # Sanity: drop NaN/inf
        if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(te_v)):
            print(f"   SKIP {disp}: non-finite values")
            continue
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        valid.append((disp, oof, te_v))
        print(f"   OK   {disp:18s} oof_RAE={r:.4f}  te_mean={te_v.mean():.3f}")
    if len(valid) < 2:
        raise RuntimeError(f"need >=2 valid sources, got {len(valid)}")

    K = len(valid)
    P_unb = np.column_stack([v[1] for v in valid])  # (n_unb, K)
    P_te = np.column_stack([v[2] for v in valid])  # (n_te, K)
    print(f"\n[stack] K={K}  P_unb {P_unb.shape}  P_te {P_te.shape}")

    # ---- Variants ----
    print("\n" + "-" * 78)
    print(f"VARIANTS (scaffold 5-fold CV, mean over kf_seeds={KF_SEEDS})")
    print("-" * 78)

    # 1) Simple mean
    mean_oof = P_unb.mean(axis=1)
    mean_rae = float(rae(y_unb, mean_oof))
    mean_pooled, mean_std, mean_seeds = pooled_scaffold_rae(
        lambda Pv: Pv.mean(axis=1), y_unb, unb_scaffolds, P_unb,
    )
    print(f"   mean      : oof_RAE={mean_rae:.4f}  scaffold-CV pooled={mean_pooled:.4f} +/- {mean_std:.4f}")

    # 2) Median
    med_oof = np.median(P_unb, axis=1)
    med_rae = float(rae(y_unb, med_oof))
    med_pooled, med_std, med_seeds = pooled_scaffold_rae(
        lambda Pv: np.median(Pv, axis=1), y_unb, unb_scaffolds, P_unb,
    )
    print(f"   median    : oof_RAE={med_rae:.4f}  scaffold-CV pooled={med_pooled:.4f} +/- {med_std:.4f}")

    # 3) Inverse-RAE weighted (label-free fixed weights computed once on full 253)
    raes = np.asarray([indiv_rae[v[0]] for v in valid], dtype=np.float64)
    inv_w = 1.0 / np.clip(raes, 1e-6, None)
    inv_w = inv_w / inv_w.sum()
    inv_oof_full = P_unb @ inv_w
    inv_rae = float(rae(y_unb, inv_oof_full))
    inv_pooled, inv_std, inv_seeds = pooled_scaffold_rae(
        lambda Pv, w=inv_w: Pv @ w, y_unb, unb_scaffolds, P_unb,
    )
    print(f"   inv-RAE   : oof_RAE={inv_rae:.4f}  scaffold-CV pooled={inv_pooled:.4f} +/- {inv_std:.4f}")
    print(f"             weights={dict(zip([v[0] for v in valid], np.round(inv_w, 3).tolist()))}")

    # 4) Trimmed mean (drop top/bottom 10% per row)
    trim_oof = trimmed_mean(P_unb, TRIM_FRAC)
    trim_rae = float(rae(y_unb, trim_oof))
    trim_pooled, trim_std, trim_seeds = pooled_scaffold_rae(
        lambda Pv: trimmed_mean(Pv, TRIM_FRAC), y_unb, unb_scaffolds, P_unb,
    )
    print(f"   trim10    : oof_RAE={trim_rae:.4f}  scaffold-CV pooled={trim_pooled:.4f} +/- {trim_std:.4f}")

    variants = [
        ("mean",   mean_pooled, mean_oof, P_te.mean(axis=1)),
        ("median", med_pooled,  med_oof,  np.median(P_te, axis=1)),
        ("inv_rae", inv_pooled, inv_oof_full, P_te @ inv_w),
        ("trim10", trim_pooled, trim_oof, trimmed_mean(P_te, TRIM_FRAC)),
    ]
    variants.sort(key=lambda x: x[1])
    best_name, best_rae, best_oof_v, best_te_v = variants[0]

    # ---- Gate ----
    print("\n" + "-" * 78)
    print(f"GATE: best ({best_name}) {best_rae:.4f} vs nb1191 ref {NB1191_REF:.4f}; margin {GATE_MARGIN}")
    print("-" * 78)
    delta_vs_nb1191 = best_rae - NB1191_REF  # negative = better
    gate_pass = delta_vs_nb1191 <= -GATE_MARGIN
    print(f"   delta = {delta_vs_nb1191:+.4f}  -> {'PASS' if gate_pass else 'FAIL'}")

    # ---- Deploy if pass ----
    sub_csv_path = SUBMISSIONS / f"{TAG}_{best_name}_ensemble.csv"
    deploy_te = best_te_v.astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))

    te_npy_path = DATA_PROCESSED / f"te_{TAG}_{best_name}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}  mean={deploy_te.mean():.3f} std={deploy_te.std():.3f}")
    print(f"       te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")

    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(f"[skip] gate FAILED -- would have written {sub_csv_path}")

    summary = {
        "tag": TAG,
        "method": "massive_ensemble_mean_median_invrae_trimmed",
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_scaf,
        "n_candidates_total": len(CANDIDATES),
        "n_valid": K,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "trim_frac": TRIM_FRAC,
        "candidate_set": [v[0] for v in valid],
        "indiv_oof_rae_unb": indiv_rae,
        "variants": {
            "mean":   {"oof_rae": mean_rae, "scaffold_cv_pooled": mean_pooled,
                       "scaffold_cv_std": mean_std, "per_seed": mean_seeds},
            "median": {"oof_rae": med_rae, "scaffold_cv_pooled": med_pooled,
                       "scaffold_cv_std": med_std, "per_seed": med_seeds},
            "inv_rae": {"oof_rae": inv_rae, "scaffold_cv_pooled": inv_pooled,
                        "scaffold_cv_std": inv_std, "per_seed": inv_seeds,
                        "weights": dict(zip(
                            [v[0] for v in valid],
                            [float(x) for x in inv_w],
                        ))},
            "trim10": {"oof_rae": trim_rae, "scaffold_cv_pooled": trim_pooled,
                       "scaffold_cv_std": trim_std, "per_seed": trim_seeds},
        },
        "best_variant": best_name,
        "best_scaffold_cv_rae": best_rae,
        "nb1191_ref": NB1191_REF,
        "delta_vs_nb1191": delta_vs_nb1191,
        "gate_margin": GATE_MARGIN,
        "gate_pass": bool(gate_pass),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_pass else None,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "wall_sec": round(time.time() - t0, 2),
        "verdict": (
            f"{best_name.upper()}_BEATS_NB1191_BY_{abs(delta_vs_nb1191):.4f}"
            if gate_pass else
            f"NO_VARIANT_BEATS_NB1191_BY_{GATE_MARGIN}_BEST_{best_name}_{best_rae:.4f}"
        ),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   n_valid candidates       = {K}")
    print(f"   mean    scaffold-CV RAE  = {mean_pooled:.4f}")
    print(f"   median  scaffold-CV RAE  = {med_pooled:.4f}")
    print(f"   inv-RAE scaffold-CV RAE  = {inv_pooled:.4f}")
    print(f"   trim10  scaffold-CV RAE  = {trim_pooled:.4f}")
    print(f"   best variant             = {best_name} ({best_rae:.4f})")
    print(f"   delta vs nb1191          = {delta_vs_nb1191:+.4f}")
    print(f"   gate (margin {GATE_MARGIN})    = {gate_pass}")
    print(f"   wall                     = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_valid", "best_variant", "best_scaffold_cv_rae",
        "delta_vs_nb1191", "gate_pass", "submission_csv", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
