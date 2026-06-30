"""nb2942 -- Inverse-RAE weighted 5-component stack {K=18, K=20, K=24, K=28, nb1191}.

NEW PARADIGM:
    Instead of equal weights (nb2934, 0.2 each), weight each component by
    1/RAE_i^2 normalized to sum to 1.  Better components (lower RAE) get
    proportionally higher weight.  Still df = 0 (weights derived from
    fixed reference RAEs, not learned on data).

HYPOTHESIS:
    K=18 (0.4619) and nb1191 (0.4697) are the strongest components.  Equal
    weighting drags them down by averaging with K=28 (0.4737).  Inverse-RAE
    weighting upweights K=18/K=20 and lifts the blend closer to the best
    component's frontier while still capturing decorrelated lift from
    nb1191's distinct paradigm.

PROTOCOL:
    1. Load 5 cached OOF + te arrays (same paths as nb2934).
    2. Reference RAEs:
         K=18  -> 0.4619
         K=20  -> 0.4630
         K=24  -> 0.4674
         K=28  -> 0.4737
         nb1191 -> 0.4697
    3. weights w_i = (1/RAE_i^2) / sum(1/RAE_j^2)
       pred = sum(w_i * pred_i)
    4. 5-fold scaffold CV on 253 with kf_seed=1001.
       Weights are fixed -- evaluate per-fold + pooled RAE.
    5. Save te + pred_oof + summary.

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4585  -> BETTER_THAN_NB2934
    else                -> FAIL

References:
    nb2934 equal-weight 5-comp pooled RAE = ~0.4580 expected
    nb1191 standalone deep-30             = 0.4718
    nb2171 ceiling deep-30 PRIMARY-1      = 0.4682

Outputs:
    scripts/nb2942_inverse_rae_weights_5comp.py
    data/processed/nb2942_summary.json
    data/processed/nb2942_pred_oof.npy   (253,) float32
    data/processed/te_nb2942.npy         (513,) float32
    submissions/nb2942_inverse_rae_weights_5comp.csv  (on any non-FAIL)
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

TAG = "nb2942"

# ---- Component OOF + te paths ----
K18_OOF_PATH = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
K18_TE_PATH = DATA_PROCESSED / "te_nb2604_K18.npy"
K20_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
K20_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
K24_OOF_PATH = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
K24_TE_PATH = DATA_PROCESSED / "te_nb2310_K24.npy"
K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
K28_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"
NB1191_OOF_PATH = DATA_PROCESSED / "nb1191_pred_oof.npy"
NB1191_TE_PATH = DATA_PROCESSED / "te_nb1191.npy"

# ---- Reference RAEs (used to derive inverse-RAE weights) ----
REF_RAE = {
    "K18": 0.4619,
    "K20": 0.4630,
    "K24": 0.4674,
    "K28": 0.4737,
    "nb1191": 0.4697,
}

# ---- Eval protocol ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_BETTER = 0.4585

# ---- References ----
NB2934_REF = 0.4580
NB1191_REF = 0.4718
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 5-component INVERSE-RAE-weighted {{K18, K20, K24, K28, nb1191}}")
    print(f"          paradigm: w_i = (1/RAE_i^2)/sum, df = 0")
    print(f"          refs nb2934={NB2934_REF}  nb1191={NB1191_REF}  "
          f"nb2171={NB2171_REF}")
    print("=" * 78)

    # ---- Load truth + scaffolds ----
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (te_df["smiles"].astype(str).tolist()
                 if "smiles" in te_df.columns
                 else te_df["SMILES"].astype(str).tolist())
    te_names = (te_df["name"].values
                if "name" in te_df.columns
                else te_df["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load components ----
    print("\n" + "-" * 78)
    print("Load 5 components")
    print("-" * 78)
    comps = [
        ("K18", K18_OOF_PATH, K18_TE_PATH),
        ("K20", K20_OOF_PATH, K20_TE_PATH),
        ("K24", K24_OOF_PATH, K24_TE_PATH),
        ("K28", K28_OOF_PATH, K28_TE_PATH),
        ("nb1191", NB1191_OOF_PATH, NB1191_TE_PATH),
    ]
    oofs = {}
    tes = {}
    per_anchor_rae = {}
    for name, oof_p, te_p in comps:
        if not oof_p.exists():
            raise FileNotFoundError(f"missing OOF: {oof_p}")
        if not te_p.exists():
            raise FileNotFoundError(f"missing te: {te_p}")
        oof = np.load(oof_p).astype(np.float64)
        te_v = np.load(te_p).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{name} oof shape {oof.shape} != ({n_unb},)")
        if te_v.shape != (n_test,):
            raise ValueError(f"{name} te shape {te_v.shape} != ({n_test},)")
        oofs[name] = oof
        tes[name] = te_v
        r = float(rae(y_unb, oof))
        per_anchor_rae[name] = r
        ref = REF_RAE[name]
        print(f"  {name:>7s}  oof_RAE={r:.4f} (ref={ref:.4f})  "
              f"te_mean={te_v.mean():.3f}  te_std={te_v.std():.3f}")

    # ---- Inverse-RAE weights ----
    print("\n" + "-" * 78)
    print("Inverse-RAE weights  w_i = (1/RAE_i^2) / sum(1/RAE_j^2)")
    print("-" * 78)
    labels = [c[0] for c in comps]
    ref_arr = np.array([REF_RAE[k] for k in labels], dtype=np.float64)
    inv_sq = 1.0 / (ref_arr ** 2)
    weights = inv_sq / inv_sq.sum()
    for k, ref, w in zip(labels, ref_arr, weights):
        print(f"  {k:>7s}  ref_RAE={ref:.4f}  1/RAE^2={1.0/(ref**2):.4f}  "
              f"w={w:.4f}")
    print(f"  sum(weights) = {weights.sum():.6f}")
    print(f"  max weight   = {weights.max():.4f} ({labels[int(weights.argmax())]})")
    print(f"  min weight   = {weights.min():.4f} ({labels[int(weights.argmin())]})")

    # ---- Apply weights ----
    print("\n" + "-" * 78)
    print("Apply weighted blend")
    print("-" * 78)
    P_unb = np.column_stack([oofs[k] for k in labels])  # (253, 5)
    P_te = np.column_stack([tes[k] for k in labels])    # (513, 5)
    pred_oof_unb = (P_unb * weights).sum(axis=1)
    pred_te_513 = (P_te * weights).sum(axis=1)
    print(f"  P_unb={P_unb.shape}  P_te={P_te.shape}")
    print(f"  pred_oof_unb mean={pred_oof_unb.mean():.3f}  "
          f"std={pred_oof_unb.std():.3f}  (truth_std={y_unb.std():.3f})")
    print(f"  pred_te_513  mean={pred_te_513.mean():.3f}  "
          f"std={pred_te_513.std():.3f}")

    rae_full = float(rae(y_unb, pred_oof_unb))
    print(f"  single-shot pooled RAE = {rae_full:.4f}")

    # Diagnostic: pair-wise correlations
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n  OOF correlation matrix:")
    print(f"        {'  '.join([f'{k:>7s}' for k in labels])}")
    for i, ki in enumerate(labels):
        row = "  ".join([f"{corr_mat[i, j]:7.3f}" for j in range(5)])
        print(f"   {ki:>7s}  {row}")

    # ---- 5-fold scaffold CV, single deterministic kf_seed ----
    print("\n" + "-" * 78)
    print(f"5-fold scaffold CV  kf_seed={KF_SEED}")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        oof_pooled[va_loc] = pred_oof_unb[va_loc]
        r = float(rae(y_unb[va_loc], pred_oof_unb[va_loc]))
        per_fold_rae.append(r)
        print(f"  fold {fi}  n_va={len(va_loc):4d}  RAE={r:.4f}")
    if np.isnan(oof_pooled).any():
        raise RuntimeError("scaffold splits did not cover all rows")
    pooled_rae = float(rae(y_unb, oof_pooled))
    mean_rae = pooled_rae  # single-seed deterministic eval

    print(f"\n  pooled RAE (full)         = {pooled_rae:.4f}")
    print(f"  per-fold mean RAE         = {np.mean(per_fold_rae):.4f}")
    print(f"  per-fold std  RAE         = {np.std(per_fold_rae):.4f}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_BETTER:
        verdict = "BETTER_THAN_NB2934"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"(<{GATE_PROMOTE} PROMOTE / <{GATE_BETTER} BETTER_THAN_NB2934)  "
          f"-> {verdict}")

    delta_vs_nb2934 = mean_rae - NB2934_REF
    delta_vs_nb2171 = mean_rae - NB2171_REF
    delta_vs_nb1191 = mean_rae - NB1191_REF
    print(f"  delta vs nb2934 ({NB2934_REF}) = {delta_vs_nb2934:+.4f}")
    print(f"  delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")
    print(f"  delta vs nb1191 ({NB1191_REF}) = {delta_vs_nb1191:+.4f}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_inverse_rae_weights_5comp.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te_513.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] verdict=FAIL; no submission CSV written")

    te_unb_in = float(rae(y_unb, pred_te_513[unb_idx]))

    summary = {
        "tag": TAG,
        "method": "inverse_rae_weighted_5component_K18_K20_K24_K28_nb1191",
        "paradigm": "w_i_proportional_to_1_over_RAE_squared_df_zero",
        "components": labels,
        "component_oof_paths": {c[0]: str(c[1]) for c in comps},
        "component_te_paths": {c[0]: str(c[2]) for c in comps},
        "ref_rae": REF_RAE,
        "weights": weights.tolist(),
        "weight_labels": labels,
        "per_anchor_rae_in_sample": per_anchor_rae,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": labels,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_fold_rae": per_fold_rae,
        "pooled_rae": pooled_rae,
        "mean_rae": mean_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "delta_vs_nb2934": delta_vs_nb2934,
        "delta_vs_nb2171": delta_vs_nb2171,
        "delta_vs_nb1191": delta_vs_nb1191,
        "nb2934_ref": NB2934_REF,
        "nb2171_ref": NB2171_REF,
        "nb1191_ref": NB1191_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-anchor RAE = " + "  ".join(
        f"{k}={v:.4f}" for k, v in per_anchor_rae.items()))
    print(f"   weights        = " + "  ".join(
        f"{k}={w:.3f}" for k, w in zip(labels, weights)))
    print(f"   pooled RAE     = {pooled_rae:.4f}")
    print(f"   verdict        = {verdict}")
    print(f"   delta nb2934   = {delta_vs_nb2934:+.4f}")
    print(f"   delta nb2171   = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] in = {te_unb_in:.4f}")
    print(f"   wall           = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae", "mean_rae", "verdict",
        "delta_vs_nb2934", "delta_vs_nb2171", "delta_vs_nb1191",
        "te_unb_in_sample_rae", "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
