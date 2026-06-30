"""nb3033 -- K18+K19 LOG-SPACE simplex (geometric-mean blend) vs the linear-space
nb3002/nb2992 paradigm.

NEW PARADIGM:
    Instead of   pred = w * K18 + (1-w) * K19           (linear/arithmetic mean),
    use          pred = exp(w * log(K18) + (1-w) * log(K19))   (geometric mean).

    Same 2-anchor pool {K18, K19} (both deep-30), same per-fold protocol,
    but the blend lives in log-pEC50 space.

    Rationale: pEC50 is itself a log-transformed potency. Linear blends of
    two log-quantities are arithmetic in log space; an exponential-of-mean-log
    blend (= geometric mean in pEC50 space, double-log in raw EC50 space) may
    capture a different (multiplicative) error structure that linear simplex
    cannot.

PROTOCOL:
    Anchors:
        K18: nb2960 deep-30 fresh-seed OOFs + te arrays (RAE 0.4536)
        K19: nb3000 deep-30 fresh-seed OOFs + te arrays (RAE 0.4607)
    For each (kf_seed, fold):
        - Golden-section search over w in [0, 1] on fold-train slice
        - loss(w) = RAE(y_tr, exp(w * log(K18_tr) + (1-w) * log(K19_tr)))
        - Apply blended prediction to held-out fold-val slice
    Outer CV: 5-fold scaffold split, 5 fresh kf_seeds {1051..1055}
    Per seed: pooled RAE on the 5 outer-val folds (covering all 253)
    Reported gate metric = MEAN pooled RAE across the 5 seeds.

    Deploy:
        - Golden-section over w on FULL 253 -> single global w*
        - Apply to (513, 2) anchor te arrays -> te_nb3033

GATE:
    mean_pooled_rae < 0.4511  -> "BETTER"
    else                      -> "FAIL"

References (linear-space baselines):
    nb2960 K18 deep-30 OOF                              = 0.4536
    nb3000 K19 deep-30 OOF                              = 0.4607
    nb2982 per-fold simplex K18,K20 (linear, deep-30)   = 0.4505
    nb2992 per-fold simplex K18,K19,K20 (linear)        = 0.4479
    nb3002 per-fold simplex K18,K19 (linear, deep-30)   = unset target
    nb2171 prior post-hoc-blend ceiling                 = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3033_summary.json
    data/processed/nb3033_pred_oof.npy  (253,) float32 -- per-fold log-blend OOF
                                         (first-seed slice; representative)
    data/processed/te_nb3033.npy        (513,) float32 -- deploy te
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

TAG = "nb3033"
PARENT_TAG = "nb2960+nb3000"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K19"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1051, 1056))  # 5 fresh seeds {1051..1055}

# -- Golden-section tuning -----------------------------------------------------
GS_TOL = 1e-6
GS_MAX_ITER = 200
PHI = (np.sqrt(5.0) - 1.0) / 2.0  # 0.6180339...

# Floor for log() to avoid log(0); pEC50 typically lives in [3, 9] anyway.
LOG_FLOOR = 1e-6

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4511

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2982 = 0.4505
REF_NB2992 = 0.4479
REF_NB2171 = 0.4682


def _log_blend(w: float, P_log: np.ndarray) -> np.ndarray:
    """Geometric mean blend: pred = exp(w*log(K18) + (1-w)*log(K19))."""
    return np.exp(w * P_log[:, 0] + (1.0 - w) * P_log[:, 1])


def _golden_section(P_log: np.ndarray, y: np.ndarray,
                    lo: float = 0.0, hi: float = 1.0,
                    tol: float = GS_TOL,
                    max_iter: int = GS_MAX_ITER) -> tuple[float, float]:
    """Minimize RAE(y, exp(w*log(K18)+(1-w)*log(K19))) over w in [lo, hi]."""
    a, b = float(lo), float(hi)

    def loss(w: float) -> float:
        return float(rae(y, _log_blend(w, P_log)))

    # Endpoint values (so we never miss boundary minima at w=0 or w=1)
    la = loss(a)
    lb = loss(b)

    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    lc = loss(c)
    ld = loss(d)
    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if lc < ld:
            b, lb = d, ld
            d, ld = c, lc
            c = b - PHI * (b - a)
            lc = loss(c)
        else:
            a, la = c, lc
            c, lc = d, ld
            d = a + PHI * (b - a)
            ld = loss(d)
    # candidate minima: interior bracket midpoint + both endpoints
    interior = 0.5 * (a + b)
    li = loss(interior)
    cands = [(a, la), (b, lb), (interior, li)]
    w_best, l_best = min(cands, key=lambda t: t[1])
    return float(w_best), float(l_best)


def _run_one_seed(kf_seed: int, P_log_unb: np.ndarray, y_unb: np.ndarray,
                  unb_scaffolds: list[str]) -> tuple[float, list[dict], np.ndarray, list[float]]:
    """Per-fold golden-section log-blend with one kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    fold_w_list = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, r_train = _golden_section(P_log_unb[tr_loc], y_unb[tr_loc])
        val_pred = _log_blend(w, P_log_unb[va_loc])
        oof_blend[va_loc] = val_pred
        r_val = float(rae(y_unb[va_loc], val_pred))
        fold_w_list.append(w)
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
            "w_K18": round(float(w), 4),
            "w_K19": round(float(1.0 - w), 4),
            "train_rae": round(float(r_train), 4),
            "val_rae": round(r_val, 4),
        })
    if np.isnan(oof_blend).any():
        raise RuntimeError(f"scaffold splits did not cover all 253 rows (kf_seed={kf_seed})")
    pooled_rae = float(rae(y_unb, oof_blend))
    return pooled_rae, fold_records, oof_blend, fold_w_list


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LOG-SPACE per-fold simplex on 2 deep-30 K-anchors {K_LABELS}")
    print(f"          parents: nb2960 (K18 deep-30) + nb3000 (K19 deep-30)")
    print(f"          paradigm: pred = exp(w*log(K18) + (1-w)*log(K19))")
    print(f"          per fold: 1D golden-section over w in [0,1]")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, {len(KF_SEEDS)} seeds "
          f"{KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          gate: mean_pooled < {GATE_BETTER} -> BETTER")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load K-anchor OOFs + te arrays --------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K-anchor OOFs and te arrays (both deep-30)")
    print("-" * 78)
    oof_cols, te_cols = [], []
    per_K_full_rae = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_arr = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_arr.shape} != ({n_test},)")
        oof_cols.append(oof)
        te_cols.append(te_arr)
        r = float(rae(y_unb, oof))
        per_K_full_rae[k] = round(r, 4)
        print(f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE={r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}  "
              f"oof_min={oof.min():.3f}")

    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    # Log-transform (with floor for safety). pEC50 ~ [3,9], well > 0.
    n_floor_unb = int((P_unb < LOG_FLOOR).sum())
    n_floor_te = int((P_te < LOG_FLOOR).sum())
    if n_floor_unb or n_floor_te:
        print(f"   WARN floored vals: unb={n_floor_unb}, te={n_floor_te}")
    P_log_unb = np.log(np.maximum(P_unb, LOG_FLOOR))
    P_log_te = np.log(np.maximum(P_te, LOG_FLOOR))

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Correlation in linear and log space
    corr_lin = float(np.corrcoef(P_unb.T)[0, 1])
    corr_log = float(np.corrcoef(P_log_unb.T)[0, 1])
    print(f"\n   OOF correlation (linear)  = {corr_lin:.4f}")
    print(f"   OOF correlation (log)     = {corr_log:.4f}")

    # Reference: pure-log endpoints (w=1 -> K18 alone in log space then exp;
    # this is identity since y_pred=exp(log K18)=K18 itself).
    rae_w1 = float(rae(y_unb, _log_blend(1.0, P_log_unb)))
    rae_w0 = float(rae(y_unb, _log_blend(0.0, P_log_unb)))
    rae_w_half = float(rae(y_unb, _log_blend(0.5, P_log_unb)))
    print(f"   ref RAE @ w=1.0 (=K18)    = {rae_w1:.4f}")
    print(f"   ref RAE @ w=0.0 (=K19)    = {rae_w0:.4f}")
    print(f"   ref RAE @ w=0.5 (geomean) = {rae_w_half:.4f}")

    # -- Build scaffolds ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Per-seed per-fold golden-section log-blend --------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: outer CV with per-fold golden-section, {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_records = {}
    per_seed_fold_weights = {}
    first_seed_oof_blend = None
    for seed in KF_SEEDS:
        p_rae, fold_recs, oof_blend, fold_ws = _run_one_seed(
            seed, P_log_unb, y_unb, unb_scaffolds,
        )
        per_seed_pooled.append(p_rae)
        per_seed_fold_records[str(seed)] = fold_recs
        per_seed_fold_weights[str(seed)] = [round(float(w), 4) for w in fold_ws]
        if first_seed_oof_blend is None:
            first_seed_oof_blend = oof_blend
        mean_val = float(np.mean([r["val_rae"] for r in fold_recs]))
        w_mean = float(np.mean(fold_ws))
        print(f"   seed={seed}  pooled={p_rae:.4f}  per-fold mean={mean_val:.4f}  "
              f"mean(w_K18)={w_mean:.3f}")

    arr_pooled = np.asarray(per_seed_pooled)
    mean_pooled = float(arr_pooled.mean())
    std_pooled = float(arr_pooled.std(ddof=1)) if len(arr_pooled) > 1 else 0.0
    min_pooled = float(arr_pooled.min())
    max_pooled = float(arr_pooled.max())
    print(f"\n   POOLED-OUTER-VAL RAE over {len(KF_SEEDS)} seeds:")
    print(f"     mean = {mean_pooled:.4f}")
    print(f"     std  = {std_pooled:.4f}")
    print(f"     min  = {min_pooled:.4f}")
    print(f"     max  = {max_pooled:.4f}")

    # Aggregate fold weights
    all_ws = [w for seed_key, wlist in per_seed_fold_weights.items() for w in wlist]
    mean_w_K18 = float(np.mean(all_ws))
    print(f"\n   mean w_K18 across {len(all_ws)} (seed,fold) cells = {mean_w_K18:.4f}")
    print(f"   mean w_K19                                       = {1.0 - mean_w_K18:.4f}")

    # -- Deploy: golden-section on FULL 253 -> single global w ---------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy golden-section on FULL 253")
    print("-" * 78)
    w_full, r_full = _golden_section(P_log_unb, y_unb)
    print(f"   in-sample RAE = {r_full:.4f}  w_K18={w_full:.4f}  w_K19={1.0-w_full:.4f}")

    te_pred = _log_blend(w_full, P_log_te).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te(full) mean={te_pred.mean():.3f} std={te_pred.std():.3f}  "
          f"in-sample unb RAE={te_unb_in_rae:.4f}")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE")
    print("-" * 78)
    verdict = "BETTER" if mean_pooled < GATE_BETTER else "FAIL"
    delta_vs_K18 = mean_pooled - REF_K18
    delta_vs_K19 = mean_pooled - REF_K19
    delta_vs_nb2982 = mean_pooled - REF_NB2982
    delta_vs_nb2992 = mean_pooled - REF_NB2992
    delta_vs_nb2171 = mean_pooled - REF_NB2171
    print(f"   mean_pooled_rae          = {mean_pooled:.4f} (std {std_pooled:.4f})")
    print(f"   delta vs K18 (0.4536)    = {delta_vs_K18:+.4f}")
    print(f"   delta vs K19 (0.4607)    = {delta_vs_K19:+.4f}")
    print(f"   delta vs nb2982 (0.4505) = {delta_vs_nb2982:+.4f}")
    print(f"   delta vs nb2992 (0.4479) = {delta_vs_nb2992:+.4f}")
    print(f"   delta vs nb2171 (0.4682) = {delta_vs_nb2171:+.4f}")
    print(f"   gate    BETTER if < {GATE_BETTER}")
    print(f"   verdict = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, first_seed_oof_blend.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}  (single-seed OOF, kf_seed={KF_SEEDS[0]})")
    print(f"   [save] {te_path}   (deploy from FULL-253 golden-section w)")

    sub_csv = SUBMISSIONS / f"{TAG}_K18_K19_log_simplex.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "per_fold_golden_section_LOG_SPACE_blend_K18_K19_deep30_5seed",
        "paradigm": "log_space_simplex_geometric_mean_vs_linear",
        "blend_formula": "exp(w*log(K18) + (1-w)*log(K19))",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_linear": round(corr_lin, 4),
        "oof_corr_log": round(corr_log, 4),
        "ref_rae_w1_K18_only": round(rae_w1, 4),
        "ref_rae_w0_K19_only": round(rae_w0, 4),
        "ref_rae_w_half_geomean": round(rae_w_half, 4),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "gs_tol": GS_TOL,
        "gs_max_iter": GS_MAX_ITER,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_seed_pooled_rae": [round(r, 5) for r in per_seed_pooled],
        "per_seed_fold_records": per_seed_fold_records,
        "per_seed_fold_weights_w_K18": per_seed_fold_weights,
        "pooled_rae_mean": round(mean_pooled, 5),
        "pooled_rae_std": round(std_pooled, 5),
        "pooled_rae_min": round(min_pooled, 5),
        "pooled_rae_max": round(max_pooled, 5),
        "mean_w_K18_across_all_cells": round(mean_w_K18, 4),
        "mean_w_K19_across_all_cells": round(1.0 - mean_w_K18, 4),
        "full_pool_golden_section": {
            "w_K18": round(float(w_full), 4),
            "w_K19": round(float(1.0 - w_full), 4),
            "rae_in_sample": round(float(r_full), 4),
        },
        "te_unb_in_sample_rae_full_pool": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "mean_rae": mean_pooled,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2982": REF_NB2982,
        "ref_nb2992": REF_NB2992,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18": delta_vs_K18,
        "delta_vs_K19": delta_vs_K19,
        "delta_vs_nb2982": delta_vs_nb2982,
        "delta_vs_nb2992": delta_vs_nb2992,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-K full-OOF RAE       = "
          + ", ".join([f"{k}={v:.4f}" for k, v in per_K_full_rae.items()]))
    print(f"   pooled outer-val RAE     = {mean_pooled:.4f} +/- {std_pooled:.4f} "
          f"({len(KF_SEEDS)} seeds)")
    print(f"   min/max pooled RAE       = {min_pooled:.4f} / {max_pooled:.4f}")
    print(f"   mean w_K18               = {mean_w_K18:.4f}  (w_K19={1.0-mean_w_K18:.4f})")
    print(f"   full-pool w_K18          = {w_full:.4f}")
    print(f"   te[unb_idx] in-sample    = {te_unb_in_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean",
        "pooled_rae_std",
        "pooled_rae_min",
        "pooled_rae_max",
        "mean_w_K18_across_all_cells",
        "full_pool_golden_section",
        "te_unb_in_sample_rae_full_pool",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
