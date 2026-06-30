"""nb2264 -- Counter-assay-routed per-row blending on nb2240 K=20 deploy.

PROTOCOL:
    For each test row r, compute disagreement
        d_r = |chembl_pec50_r - main_anchor_r|
    where main_anchor_r is the nb2240 K=20 deploy prediction on row r.
    pred_chembl_pec50 is feature idx 115 of the 117-col K=20 matrix
    (also cached as pred_chembl_pec50_513.npy on the 513 test set).

    For each disagreement threshold thr in {0.5, 1.0, 1.5}:
        if d_r > thr:
            route to counter-aware variant
                = w * nb2240 + (1 - w) * chemprop_aux
                  (chemprop_aux is the counter-head dual-task GNN; it carries
                   the PXR vs counter-assay separation signal)
        else:
            keep nb2240 unchanged.

    Sweep w in {0.3, 0.5, 0.7} per threshold.

EVALUATION:
    Scaffold 5-fold CV RAE on 253 unblind across kf_seeds {1001..1005}.
    GATE: best variant must be <= 0.4571 (nb2240 OOF 0.4598 - 0.0030 margin).

OUTPUTS:
    scripts/nb2264_counter_route.py
    data/processed/nb2264_summary.json
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
from pxr.paths import DATA_PROCESSED

TAG = "nb2264"

# Sweep grid
THR_GRID = [0.5, 1.0, 1.5]
W_GRID = [0.3, 0.5, 0.7]

# Reference gate
NB2240_REF_OOF = 0.4598      # nb2240 pooled_rae_mean_seeds
GATE_MARGIN = 0.003
GATE_TARGET = NB2240_REF_OOF - GATE_MARGIN  # 0.4568

# Scaffold CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]


# ----------------------------------------------------------------------------
def load_inputs():
    te = load_test()
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    test_smiles = te[smi_col].astype(str).tolist()
    n_te = len(test_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    # Main anchor: nb2240 deploy preds on 513 (used for 513-side disagreement
    # diagnostic only; never sliced to unb_idx for evaluation -- that would
    # be in-sample on a model trained on the unblind labels).
    te_nb2240 = np.load(DATA_PROCESSED / "te_nb2240.npy").astype(np.float64)
    assert te_nb2240.shape == (n_te,), f"te_nb2240 {te_nb2240.shape}"

    # HONEST main anchor on 253: nb2240's mean-bag OOF (chemprop_aux + K=20
    # residual LGBM, 5-seed mean, 5-fold cross-fit). Pooled RAE 0.4630.
    main_oof_unb = np.load(
        DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
    ).astype(np.float64)
    assert main_oof_unb.shape == (n_unb,), f"main_oof_unb {main_oof_unb.shape}"

    # Counter-aware route target on 513 (te) and 253 (OOF). chemprop_aux
    # is the dual-task GNN (PXR + counter-assay heads).
    te_chemprop_aux = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    assert te_chemprop_aux.shape == (n_te,), f"te_chemprop_aux {te_chemprop_aux.shape}"
    counter_oof_unb = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    assert counter_oof_unb.shape == (n_unb,), f"counter_oof_unb {counter_oof_unb.shape}"

    # chembl_pec50 feature on 513 (this is idx 115 in the 117-col K=20 matrix)
    pred_chembl_pec50 = np.load(
        DATA_PROCESSED / "pred_chembl_pec50_513.npy"
    ).astype(np.float64)
    assert pred_chembl_pec50.shape == (n_te,), f"pred_chembl_pec50 {pred_chembl_pec50.shape}"

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]

    return {
        "test_smiles": test_smiles,
        "n_te": n_te,
        "unb_idx": unb_idx,
        "y_unb": y_unb,
        "n_unb": n_unb,
        "te_nb2240": te_nb2240,
        "main_oof_unb": main_oof_unb,
        "te_chemprop_aux": te_chemprop_aux,
        "counter_oof_unb": counter_oof_unb,
        "pred_chembl_pec50": pred_chembl_pec50,
        "unb_smiles": unb_smiles,
        "unb_scaffolds": unb_scaffolds,
    }


def apply_route(
    main_pred: np.ndarray,
    counter_pred: np.ndarray,
    disagreement: np.ndarray,
    thr: float,
    w: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (routed_pred, fire_mask)."""
    fire = disagreement > thr
    out = main_pred.copy()
    out[fire] = w * main_pred[fire] + (1.0 - w) * counter_pred[fire]
    return out, fire


def scaffold_cv_rae(
    main_pred_unb: np.ndarray,
    counter_pred_unb: np.ndarray,
    disagreement_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    thr: float,
    w: float,
) -> dict:
    """Apply route then evaluate scaffold-CV pooled RAE across kf seeds.

    Routing has no learned parameters, so the CV splits only stratify
    the evaluation; the routed prediction is computed once and the same
    OOF vector is scored per seed (only the fold partition cycles
    through). Reported RAE is pooled across all rows per seed; the
    pooled-mean-seeds equals the simple pooled RAE because every row
    is in exactly one validation fold for each seed.
    """
    routed, fire_mask = apply_route(
        main_pred_unb, counter_pred_unb, disagreement_unb, thr, w,
    )
    per_seed = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full_like(y_unb, np.nan)
        for tr_loc, va_loc in splits:
            oof[va_loc] = routed[va_loc]
        per_seed.append(float(rae(y_unb, oof)))
    return {
        "thr": float(thr),
        "w": float(w),
        "n_fire_unb": int(fire_mask.sum()),
        "fire_rate_unb": float(fire_mask.mean()),
        "rae_per_seed": per_seed,
        "rae_mean_seeds": float(np.mean(per_seed)),
        "rae_std_seeds": float(np.std(per_seed)),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- counter-assay route on nb2240 K=20 deploy")
    print(f"       gate = nb2240 OOF {NB2240_REF_OOF:.4f} - {GATE_MARGIN} "
          f"= {GATE_TARGET:.4f}")
    print("=" * 78)

    d = load_inputs()
    n_te = d["n_te"]
    n_unb = d["n_unb"]
    print(f"[load] n_te={n_te}  n_unb={n_unb}")
    print(f"[load] te_nb2240         mean={d['te_nb2240'].mean():.3f}  "
          f"std={d['te_nb2240'].std():.3f}")
    print(f"[load] te_chemprop_aux   mean={d['te_chemprop_aux'].mean():.3f}  "
          f"std={d['te_chemprop_aux'].std():.3f}")
    print(f"[load] pred_chembl_pec50 mean={d['pred_chembl_pec50'].mean():.3f}  "
          f"std={d['pred_chembl_pec50'].std():.3f}")

    # te-side (513) anchors: only used for disagreement diagnostic + the
    # 513 disagreement percentile report. Evaluation on 253 uses OOF.
    main_te = d["te_nb2240"]
    counter_te = d["te_chemprop_aux"]
    chembl_te = d["pred_chembl_pec50"]

    # 253-side HONEST anchors (cross-fit OOFs)
    main_unb = d["main_oof_unb"]
    counter_unb = d["counter_oof_unb"]
    chembl_unb = chembl_te[d["unb_idx"]]   # feature, not a label-fit model
    y_unb = d["y_unb"]
    unb_scaffolds = d["unb_scaffolds"]

    rae_main_unb = float(rae(y_unb, main_unb))
    rae_counter_unb = float(rae(y_unb, counter_unb))
    rae_main_te_unb_in_sample = float(rae(y_unb, main_te[d["unb_idx"]]))
    print(f"\n[base] nb2240 OOF (cross-fit) RAE        = {rae_main_unb:.4f}")
    print(f"[base] chemprop_aux OOF (cross-fit) RAE  = {rae_counter_unb:.4f}")
    print(f"[diag] nb2240 te[unb_idx] in-sample RAE  = {rae_main_te_unb_in_sample:.4f}  "
          f"(leaky -- not used for evaluation)")
    print(f"[base] nb2240 reference OOF (5-seed mean) = {NB2240_REF_OOF:.4f}")

    # Disagreement on 253 (gate variable) uses HONEST main_unb anchor
    disagree_unb = np.abs(chembl_unb - main_unb)
    disagree_te = np.abs(chembl_te - main_te)
    print(f"\n[disagree] |chembl - nb2240|  unb mean={disagree_unb.mean():.3f}  "
          f"max={disagree_unb.max():.3f}  std={disagree_unb.std():.3f}")
    for q in (50, 75, 90, 95):
        print(f"   p{q:>3}: unb={np.percentile(disagree_unb, q):.3f}   "
              f"te={np.percentile(disagree_te, q):.3f}")

    # Sweep thr x w
    print("\n" + "-" * 78)
    print("SWEEP  thr in {0.5, 1.0, 1.5}  x  w in {0.3, 0.5, 0.7}")
    print("-" * 78)
    print(f"  {'thr':>5}  {'w':>5}  {'n_fire_unb':>10}  {'fire%':>6}  "
          f"{'RAE_mean':>8}  {'dRAE':>8}  {'beats?':>7}")
    results = []
    for thr in THR_GRID:
        for w in W_GRID:
            res = scaffold_cv_rae(
                main_unb, counter_unb, disagree_unb,
                y_unb, unb_scaffolds, thr, w,
            )
            # Also record te-side fire rate for diagnostic
            res["n_fire_te"] = int((disagree_te > thr).sum())
            res["fire_rate_te"] = float((disagree_te > thr).mean())
            res["delta_vs_nb2240_oof"] = res["rae_mean_seeds"] - NB2240_REF_OOF
            res["beats_gate"] = bool(res["rae_mean_seeds"] <= GATE_TARGET)
            results.append(res)
            print(f"  {thr:>5.2f}  {w:>5.2f}  {res['n_fire_unb']:>10d}  "
                  f"{res['fire_rate_unb']*100:>5.1f}%  "
                  f"{res['rae_mean_seeds']:>8.4f}  "
                  f"{res['delta_vs_nb2240_oof']:>+8.4f}  "
                  f"{'YES' if res['beats_gate'] else 'no':>7s}")

    # Best variant
    best = sorted(results, key=lambda x: x["rae_mean_seeds"])[0]
    print("\n" + "=" * 78)
    print("BEST VARIANT")
    print("=" * 78)
    print(f"  thr            = {best['thr']:.2f}")
    print(f"  w              = {best['w']:.2f}")
    print(f"  n_fire_unb     = {best['n_fire_unb']}/{n_unb} "
          f"({best['fire_rate_unb']*100:.1f}%)")
    print(f"  n_fire_te      = {best['n_fire_te']}/{n_te} "
          f"({best['fire_rate_te']*100:.1f}%)")
    print(f"  RAE_mean_seeds = {best['rae_mean_seeds']:.4f}")
    print(f"  reference      = {NB2240_REF_OOF:.4f}  (nb2240 OOF)")
    print(f"  delta          = {best['delta_vs_nb2240_oof']:+.4f}")
    print(f"  gate target    = {GATE_TARGET:.4f}  (margin {GATE_MARGIN})")
    verdict = "BEATS_GATE" if best["beats_gate"] else "BELOW_GATE"
    print(f"  verdict        = {verdict}")

    summary = {
        "tag": TAG,
        "method": "counter_assay_routed_per_row_blend_on_nb2240_K20",
        "rule": {
            "disagreement": "abs(pred_chembl_pec50 - nb2240_main_anchor)",
            "chembl_feature_idx_in_117": 115,
            "main_anchor": "te_nb2240.npy (K=20 deploy)",
            "counter_aware_variant": "w*nb2240 + (1-w)*chemprop_aux (dual-task counter-head)",
        },
        "thr_grid": THR_GRID,
        "w_grid": W_GRID,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_te": n_te,
        "n_unb": n_unb,
        "disagree_unb_mean": float(disagree_unb.mean()),
        "disagree_unb_std": float(disagree_unb.std()),
        "disagree_unb_max": float(disagree_unb.max()),
        "disagree_unb_percentiles": {
            "50": float(np.percentile(disagree_unb, 50)),
            "75": float(np.percentile(disagree_unb, 75)),
            "90": float(np.percentile(disagree_unb, 90)),
            "95": float(np.percentile(disagree_unb, 95)),
        },
        "disagree_te_percentiles": {
            "50": float(np.percentile(disagree_te, 50)),
            "75": float(np.percentile(disagree_te, 75)),
            "90": float(np.percentile(disagree_te, 90)),
            "95": float(np.percentile(disagree_te, 95)),
        },
        "main_oof_rae_unb": rae_main_unb,
        "counter_oof_rae_unb": rae_counter_unb,
        "main_te_unb_in_sample_rae_diagnostic": rae_main_te_unb_in_sample,
        "nb2240_ref_oof": NB2240_REF_OOF,
        "gate_margin": GATE_MARGIN,
        "gate_target": GATE_TARGET,
        "results": results,
        "best": best,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_json}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best (thr={best['thr']:.2f}, w={best['w']:.2f})  "
          f"RAE={best['rae_mean_seeds']:.4f}  "
          f"delta={best['delta_vs_nb2240_oof']:+.4f}  {verdict}")
    print(f"   wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("best", "verdict", "wall_sec"):
        print(f"  {k}: {res.get(k)}")
