"""nb3203 -- Specific (q05, q97) clip on nb3090 best quantile-conditional combo.

NEW PARADIGM:
    Targeted single combo at mid-aggressive on high tail. Fix the nb3090 winning
    combo (q_cut=0.35, w_K18_low=0.95, w_K18_high=0.40) and apply per-fold
    (q05, q97) clip to the OOF blend predictions. Idea: nb3090 ceiling is
    n_unb=253 with K18+K19 deep-30 anchors; tail-clipping the blend at (q05, q97)
    per fold may compress the extreme-tail noise that the post-hoc blend cannot
    correct from the anchor alone.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    1. For each fold:
        a. Compute fold-train K18 q_cut quantile threshold q (q_cut=0.35).
        b. For fold-val rows: apply nb3090 quantile-conditional blend
           (low: 0.95*K18 + 0.05*K19; high: 0.40*K18 + 0.60*K19).
        c. Compute fold-train clip bounds: q05 = quantile(blend_train, 0.05),
           q97 = quantile(blend_train, 0.97). Clip val_pred to [q05, q97].
    2. Stitch into oof_blend (253,); pooled_rae across the 5 outer folds.
    Repeat for 15 fresh kf_seeds {1186..1200}; report mean.

GATE (on 15-seed mean RAE):
    mean < 0.4426 -> "BETTER"   (beats nb3090 by 0.0046 = 1.0%)
    mean < 0.4437 -> "MARGINAL" (beats nb3090 by 0.0035 = 0.78%)
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF              = 0.4536
    nb3000 K19 deep-30 OOF              = 0.4607
    nb3080 wide-seed verify nb3073      = 0.4475 (15 seeds)
    nb3090 finer-grid best (q=0.35)     = 0.4472 (15 seeds) <- ANCHOR
    nb2171 prior post-hoc ceiling       = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3203_summary.json
    data/processed/nb3203_pred_oof.npy  (253,) float32 -- median-seed OOF
    data/processed/te_nb3203.npy        (513,) float32 -- deploy te
    submissions/nb3203_specific_q05_q97.csv (only on BETTER verdict)
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

TAG = "nb3203"
PARENT_TAG = "nb3090"

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
KF_SEEDS = list(range(1186, 1201))  # 15 fresh seeds {1186..1200}

# -- Fixed nb3090-winning combo + clip bounds ----------------------------------
Q_CUT = 0.35
W_K18_LOW = 0.95
W_K18_HIGH = 0.40
Q_CLIP_LOW = 0.05   # q05
Q_CLIP_HIGH = 0.97  # q97

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4426
GATE_MARGINAL = 0.4437

# -- References ----------------------------------------------------------------
REF_NB3090 = 0.4472
REF_NB3080 = 0.4475
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q_thr: float,
    w_low: float,
    w_high: float,
) -> np.ndarray:
    """Per-row hard-split blend.

    rows with p_k18 <= q_thr -> (w_low * p_k18 + (1-w_low) * p_k19)
    rows with p_k18 >  q_thr -> (w_high * p_k18 + (1-w_high) * p_k19)
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = w_low * p_k18[low_mask] + (1.0 - w_low) * p_k19[low_mask]
    out[~low_mask] = w_high * p_k18[~low_mask] + (1.0 - w_high) * p_k19[~low_mask]
    return out


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run nb3090 blend + per-fold (q05, q97) clip at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_clip_bounds = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # Step 1: K18 quantile threshold from fold-train
        q_thr = float(np.quantile(P_unb[tr_loc, 0], Q_CUT))
        # Step 2a: compute fold-train blend (for clip-bound estimation)
        tr_p_k18 = P_unb[tr_loc, 0]
        tr_p_k19 = P_unb[tr_loc, 1]
        tr_blend = _blend_quantile_conditional(
            tr_p_k18, tr_p_k19, q_thr, W_K18_LOW, W_K18_HIGH,
        )
        # Step 2b: compute fold-val blend
        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        val_pred = _blend_quantile_conditional(
            val_p_k18, val_p_k19, q_thr, W_K18_LOW, W_K18_HIGH,
        )
        # Step 3: (q05, q97) clip bounds from fold-train blend, apply to val
        q_low = float(np.quantile(tr_blend, Q_CLIP_LOW))
        q_high = float(np.quantile(tr_blend, Q_CLIP_HIGH))
        val_clipped = np.clip(val_pred, q_low, q_high)
        oof_blend[va_loc] = val_clipped
        fold_clip_bounds.append((q_low, q_high))

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "oof": oof_blend,
        "fold_clip_bounds": fold_clip_bounds,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- specific (q05, q97) clip on nb3090 best combo")
    print(f"          combo    = (q_cut={Q_CUT}, w_low={W_K18_LOW}, "
          f"w_high={W_K18_HIGH})")
    print(f"          clip     = (q{int(Q_CLIP_LOW*100):02d}, "
          f"q{int(Q_CLIP_HIGH*100):02d}) per fold")
    print(f"          kf_seeds = {KF_SEEDS}")
    print(f"          gate: mean < {GATE_BETTER} BETTER, "
          f"< {GATE_MARGINAL} MARGINAL, else FAIL")
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

    # -- Load K18, K19 deep-30 anchor OOFs + te arrays ------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K19 deep-30 OOFs and te arrays")
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
        print(f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    corr = float(np.corrcoef(P_unb.T)[0, 1])
    print(f"   pairwise corr({K_LABELS[0]}, {K_LABELS[1]}) = {corr:.4f}")

    # -- Scaffolds (kf_seed independent) -------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Run 15 seeds --------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"RUN: 15 seeds x {N_FOLDS} folds, nb3090 blend + (q05, q97) clip")
    print("-" * 78)
    seed_results = []
    pooled_raes = []
    oof_stack = []
    all_clip_bounds = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        seed_results.append({
            "kf_seed": s,
            "pooled_rae": round(res["pooled_rae"], 4),
            "fold_clip_bounds": [(round(lo, 4), round(hi, 4))
                                 for (lo, hi) in res["fold_clip_bounds"]],
        })
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        all_clip_bounds.extend(res["fold_clip_bounds"])
        print(f"   seed={s}  rae={res['pooled_rae']:.4f}  "
              f"wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1))
    min_rae = float(arr.min())
    max_rae = float(arr.max())

    print("\n" + "-" * 78)
    print(f"   15-seed mean = {mean_rae:.4f}")
    print(f"   15-seed std  = {std_rae:.4f}")
    print(f"   min / max    = [{min_rae:.4f}, {max_rae:.4f}]")
    print(f"   ref nb3090   = {REF_NB3090:.4f}")
    print(f"   delta        = {mean_rae - REF_NB3090:+.4f}")

    # Mean clip bounds summary
    clip_arr = np.asarray(all_clip_bounds, dtype=np.float64)
    mean_q_low = float(clip_arr[:, 0].mean())
    mean_q_high = float(clip_arr[:, 1].mean())
    print(f"   mean clip bounds: q05={mean_q_low:.3f}  q97={mean_q_high:.3f}")

    # -- Deploy te (full-pool clip) ------------------------------------------
    deploy_q_thr = float(np.quantile(P_unb[:, 0], Q_CUT))
    te_pred = _blend_quantile_conditional(
        P_te[:, 0], P_te[:, 1], deploy_q_thr,
        W_K18_LOW, W_K18_HIGH,
    )
    # Deploy clip: use ALL unblind blend (no fold-leave-out) as proxy
    unb_blend = _blend_quantile_conditional(
        P_unb[:, 0], P_unb[:, 1], deploy_q_thr,
        W_K18_LOW, W_K18_HIGH,
    )
    deploy_q_low = float(np.quantile(unb_blend, Q_CLIP_LOW))
    deploy_q_high = float(np.quantile(unb_blend, Q_CLIP_HIGH))
    te_pred = np.clip(te_pred, deploy_q_low, deploy_q_high)
    te_pred = np.clip(te_pred, 3.0, 9.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   deploy q_thr     = {deploy_q_thr:.4f}")
    print(f"   deploy clip      = [{deploy_q_low:.3f}, {deploy_q_high:.3f}]")
    print(f"   te(513) mean     = {te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    med_seed_idx = int(np.argsort(pooled_arr)[len(pooled_arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={pooled_arr[med_seed_idx]:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE candidate. nb3203 (q05, q97) clip on nb3090 best combo "
            f"15-seed mean {mean_rae:.4f} beats GATE_BETTER {GATE_BETTER} "
            f"(delta vs nb3090: {mean_rae - REF_NB3090:+.4f}). "
            "Tail compression on quantile-conditional blend opens new ceiling. "
            "Consider deploy with deep-30 confirmation."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL"
        ladder_action = (
            f"MARGINAL. nb3203 15-seed mean {mean_rae:.4f} clears GATE_MARGINAL "
            f"{GATE_MARGINAL} but not GATE_BETTER {GATE_BETTER}. "
            f"Delta vs nb3090: {mean_rae - REF_NB3090:+.4f}. "
            "Requires deep-30 verification before promote."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3203 15-seed mean {mean_rae:.4f} fails both gates "
            f"(BETTER<{GATE_BETTER}, MARGINAL<{GATE_MARGINAL}). "
            f"Delta vs nb3090: {mean_rae - REF_NB3090:+.4f}. "
            "(q05, q97) tail clip does not break nb3090 quantile-conditional "
            "ceiling. Keep nb3090 / prior PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_specific_q05_q97.csv"
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
        "method": "specific_q05_q97_clip_on_nb3090_quantile_conditional_blend",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "q_cut": Q_CUT,
        "w_K18_low": W_K18_LOW,
        "w_K18_high": W_K18_HIGH,
        "q_clip_low": Q_CLIP_LOW,
        "q_clip_high": Q_CLIP_HIGH,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_results": seed_results,
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "min_rae": round(min_rae, 4),
        "max_rae": round(max_rae, 4),
        "mean_clip_bound_low": round(mean_q_low, 4),
        "mean_clip_bound_high": round(mean_q_high, 4),
        "ref_nb3090": REF_NB3090,
        "ref_nb3080": REF_NB3080,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3090": round(mean_rae - REF_NB3090, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "deploy_clip_low": round(deploy_q_low, 4),
        "deploy_clip_high": round(deploy_q_high, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "gate_better": GATE_BETTER,
        "gate_marginal": GATE_MARGINAL,
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
    print(f"   combo (q, w_low, w_high)      = "
          f"({Q_CUT}, {W_K18_LOW}, {W_K18_HIGH})")
    print(f"   clip (q_low, q_high)          = "
          f"({Q_CLIP_LOW}, {Q_CLIP_HIGH})")
    print(f"   mean_rae (15 seeds)           = {mean_rae:.4f}")
    print(f"   std_rae                       = {std_rae:.4f}")
    print(f"   delta vs nb3090               = {mean_rae - REF_NB3090:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "delta_vs_nb3090",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
