"""nb2952 -- Add K=21, K=22 to extend the equal-K ratio grid.

NEW PARADIGM:
    nb2943 best used equal_K = mean(K18, K24, K28).  Add K=21 and K=22 RFE
    pyramids to the average so equal_K_extended = mean(K18, K21, K22, K24, K28).
    Blend at 0.5 * nb2240 + 0.5 * equal_K_extended.

ANCHORS (all cached):
    K18    -> nb2604_mean_bag_oof_K18.npy + te_nb2604_K18.npy
    K21    -> nb2631_mean_bag_oof_K21.npy + te_nb2631_K21.npy
    K22    -> nb2261_mean_bag_oof_K22.npy + te_nb2261_K22.npy
    K24    -> nb2310_mean_bag_oof_K24.npy + te_nb2310_K24.npy
    K28    -> nb2103_mean_bag_oof_K28.npy + te_nb2112.npy
    nb2240 -> K=20 (nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy)

BLEND:
    equal_K_extended = (K18 + K21 + K22 + K24 + K28) / 5
    pred = 0.5 * nb2240 + 0.5 * equal_K_extended

PROTOCOL:
    1. Load 6 cached OOF + te arrays.
    2. Build equal_K_extended over 5 K-anchors.
    3. Compute 0.5 * nb2240 + 0.5 * equal_K_extended on OOF + te.
    4. 5-fold scaffold CV on 253, kf_seed=1001, pooled RAE.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4576 -> BETTER_THAN_NB2943
    else              -> FAIL

References:
    nb2943 best (w2240=0.5, w_K=0.5)             = 0.4576 (equal_K = K18+K24+K28)
    nb2934 5-anchor equal weight pooled RAE      = 0.4580
    nb2604 4-K equal weight pooled RAE           = 0.4580
    nb2171 ceiling deep-30 PRIMARY-1             = 0.4682
    nb1191 PRE-pyramid deep-30                   = 0.4718
    chemprop_aux                                  = 0.6216

Outputs:
    scripts/nb2952_explicit_K21_K22.py
    data/processed/nb2952_summary.json
    data/processed/nb2952_pred_oof.npy   (253,) float32
    data/processed/te_nb2952.npy         (513,) float32
    submissions/nb2952_explicit_K21_K22.csv  (on any non-FAIL)
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

TAG = "nb2952"

# ---- Component OOF + te paths (all cached) ----
K18_OOF_PATH = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
K18_TE_PATH = DATA_PROCESSED / "te_nb2604_K18.npy"
K20_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"   # == nb2240 anchor
K20_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
K21_OOF_PATH = DATA_PROCESSED / "nb2631_mean_bag_oof_K21.npy"
K21_TE_PATH = DATA_PROCESSED / "te_nb2631_K21.npy"
K22_OOF_PATH = DATA_PROCESSED / "nb2261_mean_bag_oof_K22.npy"
K22_TE_PATH = DATA_PROCESSED / "te_nb2261_K22.npy"
K24_OOF_PATH = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
K24_TE_PATH = DATA_PROCESSED / "te_nb2310_K24.npy"
K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
K28_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"

# ---- Eval protocol ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Blend weights ----
W_NB2240 = 0.5
W_EQUAL_K = 0.5

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_BETTER_NB2943 = 0.4576

# ---- References ----
NB2943_REF = 0.4576
NB2934_REF = 0.4580
NB2604_REF = 0.4580
NB2171_REF = 0.4682
NB1191_REF = 0.4718
CHEMPROP_AUX_REF = 0.6216


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- extend equal_K with K=21, K=22 -> 5-anchor equal_K_extended")
    print(f"          paradigm: 0.5 * nb2240 + 0.5 * mean(K18,K21,K22,K24,K28)")
    print(f"          refs nb2943={NB2943_REF}  nb2934={NB2934_REF}  "
          f"nb2171={NB2171_REF}  nb1191={NB1191_REF}")
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
    print("Load 6 components (K18, K20=nb2240, K21, K22, K24, K28)")
    print("-" * 78)
    comps = [
        ("K18", K18_OOF_PATH, K18_TE_PATH),
        ("K20", K20_OOF_PATH, K20_TE_PATH),   # == nb2240
        ("K21", K21_OOF_PATH, K21_TE_PATH),
        ("K22", K22_OOF_PATH, K22_TE_PATH),
        ("K24", K24_OOF_PATH, K24_TE_PATH),
        ("K28", K28_OOF_PATH, K28_TE_PATH),
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
        print(f"  {name:>5s}  oof_RAE={r:.4f}  te_mean={te_v.mean():.3f}  "
              f"te_std={te_v.std():.3f}  [{oof_p.name}]")

    # ---- Build equal_K_extended from {K18, K21, K22, K24, K28} (5 anchors, K20 carried by nb2240) ----
    equal_K_oof = (
        oofs["K18"] + oofs["K21"] + oofs["K22"] + oofs["K24"] + oofs["K28"]
    ) / 5.0
    equal_K_te = (
        tes["K18"] + tes["K21"] + tes["K22"] + tes["K24"] + tes["K28"]
    ) / 5.0
    equal_K_rae = float(rae(y_unb, equal_K_oof))
    print(f"\n  equal_K_extended = mean(K18, K21, K22, K24, K28)  "
          f"oof_RAE={equal_K_rae:.4f}  te_mean={equal_K_te.mean():.3f}  "
          f"te_std={equal_K_te.std():.3f}")

    # Also compute baseline equal_K (3 anchors, nb2943 recipe) for delta visibility
    equal_K3_oof = (oofs["K18"] + oofs["K24"] + oofs["K28"]) / 3.0
    equal_K3_te = (tes["K18"] + tes["K24"] + tes["K28"]) / 3.0
    equal_K3_rae = float(rae(y_unb, equal_K3_oof))
    print(f"  equal_K_3anchor  = mean(K18, K24, K28)            "
          f"oof_RAE={equal_K3_rae:.4f}  te_mean={equal_K3_te.mean():.3f}  "
          f"te_std={equal_K3_te.std():.3f}   (nb2943 recipe)")

    # ---- Build scaffold splits once (deterministic) ----
    print("\n" + "-" * 78)
    print(f"5-fold scaffold CV  kf_seed={KF_SEED}")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    # ---- Build blended pred (0.5*nb2240 + 0.5*equal_K_extended) ----
    pred_oof = W_NB2240 * oofs["K20"] + W_EQUAL_K * equal_K_oof
    pred_te = W_NB2240 * tes["K20"] + W_EQUAL_K * equal_K_te

    # 5-fold scaffold CV pooled RAE
    oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        oof_pooled[va_loc] = pred_oof[va_loc]
        per_fold.append(float(rae(y_unb[va_loc], pred_oof[va_loc])))
    if np.isnan(oof_pooled).any():
        raise RuntimeError("scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_pooled))
    fold_mean = float(np.mean(per_fold))
    fold_std = float(np.std(per_fold))

    print(f"\n  blend  w_nb2240={W_NB2240}  w_equal_K_extended={W_EQUAL_K}")
    print(f"  pooled_RAE = {pooled:.4f}")
    print(f"  fold_mean  = {fold_mean:.4f}  fold_std={fold_std:.4f}")
    print(f"  per-fold   = " + "  ".join(f"f{i}={v:.4f}"
                                          for i, v in enumerate(per_fold)))

    # ---- Sanity comparison: 0.5*nb2240 + 0.5*equal_K_3anchor (nb2943 best) ----
    pred_oof_3 = W_NB2240 * oofs["K20"] + W_EQUAL_K * equal_K3_oof
    oof_pooled_3 = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_3 = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        oof_pooled_3[va_loc] = pred_oof_3[va_loc]
        per_fold_3.append(float(rae(y_unb[va_loc], pred_oof_3[va_loc])))
    pooled_3 = float(rae(y_unb, oof_pooled_3))
    print(f"\n  [sanity] 0.5*nb2240 + 0.5*equal_K_3anchor (nb2943 recipe) "
          f"pooled_RAE = {pooled_3:.4f}")

    # ---- Gate ----
    if pooled < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled < GATE_BETTER_NB2943:
        verdict = "BETTER_THAN_NB2943"
    else:
        verdict = "FAIL"
    print(f"\n[gate] pooled_RAE={pooled:.4f}  "
          f"(<{GATE_PROMOTE} PROMOTE / <{GATE_BETTER_NB2943} BETTER_THAN_NB2943)  "
          f"-> {verdict}")

    delta_vs_nb2943 = pooled - NB2943_REF
    delta_vs_nb2934 = pooled - NB2934_REF
    delta_vs_nb2171 = pooled - NB2171_REF
    delta_vs_nb1191 = pooled - NB1191_REF
    delta_vs_3anchor = pooled - pooled_3
    print(f"  delta vs nb2943 ({NB2943_REF}) = {delta_vs_nb2943:+.4f}")
    print(f"  delta vs nb2934 ({NB2934_REF}) = {delta_vs_nb2934:+.4f}")
    print(f"  delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")
    print(f"  delta vs nb1191 ({NB1191_REF}) = {delta_vs_nb1191:+.4f}")
    print(f"  delta vs 3anchor ({pooled_3:.4f}) = {delta_vs_3anchor:+.4f}  "
          f"(K=21,22 contribution)")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof.astype(np.float32))
    np.save(te_path, pred_te.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_explicit_K21_K22.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] verdict=FAIL; no submission CSV written")

    te_unb_in = float(rae(y_unb, pred_te[unb_idx]))

    summary = {
        "tag": TAG,
        "method": "explicit_K21_K22_extended_equal_K",
        "paradigm": "0.5_nb2240_plus_0.5_equal_K_extended_5anchor",
        "anchors": {
            "nb2240": "K=20 (nb2240_mean_bag_oof_K20.npy)",
            "K18": "nb2604_mean_bag_oof_K18.npy",
            "K21": "nb2631_mean_bag_oof_K21.npy",
            "K22": "nb2261_mean_bag_oof_K22.npy",
            "K24": "nb2310_mean_bag_oof_K24.npy",
            "K28": "nb2103_mean_bag_oof_K28.npy",
        },
        "w_nb2240": W_NB2240,
        "w_equal_K_extended": W_EQUAL_K,
        "per_anchor_rae_in_sample": per_anchor_rae,
        "equal_K_extended_oof_rae": equal_K_rae,
        "equal_K_3anchor_oof_rae": equal_K3_rae,
        "sanity_3anchor_pooled_rae": pooled_3,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "pooled_rae": pooled,
        "mean_rae": pooled,
        "per_fold_rae": per_fold,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "gate_promote": GATE_PROMOTE,
        "gate_better_nb2943": GATE_BETTER_NB2943,
        "verdict": verdict,
        "delta_vs_nb2943": delta_vs_nb2943,
        "delta_vs_nb2934": delta_vs_nb2934,
        "delta_vs_nb2171": delta_vs_nb2171,
        "delta_vs_nb1191": delta_vs_nb1191,
        "delta_vs_3anchor": delta_vs_3anchor,
        "nb2943_ref": NB2943_REF,
        "nb2934_ref": NB2934_REF,
        "nb2604_ref": NB2604_REF,
        "nb2171_ref": NB2171_REF,
        "nb1191_ref": NB1191_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "te_mean": float(pred_te.mean()),
        "te_std": float(pred_te.std()),
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
    print(f"   per-anchor RAE  = " + "  ".join(
        f"{k}={v:.4f}" for k, v in per_anchor_rae.items()))
    print(f"   equal_K_ext RAE = {equal_K_rae:.4f}")
    print(f"   3anchor RAE     = {equal_K3_rae:.4f}  (K18,K24,K28 only)")
    print(f"   blend           = {W_NB2240}*nb2240 + {W_EQUAL_K}*equal_K_extended")
    print(f"   pooled RAE      = {pooled:.4f}")
    print(f"   fold_mean+/-std = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"   verdict         = {verdict}")
    print(f"   delta nb2943    = {delta_vs_nb2943:+.4f}")
    print(f"   delta nb2171    = {delta_vs_nb2171:+.4f}")
    print(f"   delta 3anchor   = {delta_vs_3anchor:+.4f}  (K=21,22 contribution)")
    print(f"   te[unb_idx] in  = {te_unb_in:.4f}")
    print(f"   wall            = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae", "mean_rae", "verdict",
        "delta_vs_nb2943", "delta_vs_nb2934", "delta_vs_nb2171",
        "delta_vs_nb1191", "delta_vs_3anchor",
        "te_unb_in_sample_rae", "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
