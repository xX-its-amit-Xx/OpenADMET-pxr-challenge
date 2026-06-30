"""nb2864 -- Tanimoto-confidence weighted isotonic calibration on nb2240.

NEW PARADIGM:
    Standard isotonic calibration (nb2632) gives every fold-train (anchor, y)
    pair equal weight when fitting the monotone map.  But the anchor's
    reliability is NOT uniform: rows with HIGH max-Tanimoto-to-train are on
    a chemistry the chemprop_aux + nb2240 LGBM stack has seen before; rows
    with LOW max-Tanimoto are off-manifold and the anchor prediction is
    noisier.  Weighting the isotonic fit by confidence:

        weight_i = max(0.1, sigmoid(10*(max_tan_i - 0.35)))

    lets the high-confidence rows dominate the monotone map shape (so the
    map is tuned to the predictable regime) while the low-confidence rows
    still contribute a small (>=0.1) signal so they don't collapse to a
    single knot.  sklearn IsotonicRegression supports sample_weight natively.

PROTOCOL:
    - Anchor = nb2240_mean_bag_oof_K20.npy (253 honest cross-fit) +
      te_nb2240.npy (513 deploy).
    - Confidence per row: max-Tanimoto to the 4139 train Morgan FPs.
      For the 253 unblind: max_tan_unb (253-vec).
      For the 513 test:    max_tan_te  (513-vec).
    - weight_i = max(0.1, sigmoid(10*(max_tan_i - 0.35))).
    - 5-fold scaffold CV on 253, single kf_seed=1001.
      Per fold:
        IsotonicRegression(out_of_bounds='clip').fit(
            anchor_oof[tr], y_unb[tr], sample_weight=w_unb[tr]
        )
        pred_va = ir.predict(anchor_oof[va])
    - Pooled cross-fit RAE on 253 = mean_rae.
    - Deploy: refit weighted iso on FULL 253 anchor (with w_unb full vec);
      transform te_nb2240 (513).

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT
    else               -> FAIL

Outputs:
    scripts/nb2864_tan_conf_isotonic.py
    data/processed/nb2864_summary.json
    data/processed/nb2864_pred_oof.npy   (253,) float32
    data/processed/te_nb2864.npy         (513,) float32
    submissions/nb2864_tan_conf_isotonic.csv  (on non-FAIL)
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
from sklearn.isotonic import IsotonicRegression

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch, standardize
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2864"

# ---- Anchor: nb2240 K=20 mean-bag ----
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"

# ---- Confidence weighting ----
SIM_PIVOT = 0.35
SIM_SLOPE = 10.0
WEIGHT_FLOOR = 0.1

# ---- CV eval ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- Reference numbers ----
NB2171_REF = 0.4682
NB2240_REF = 0.4790  # approximate K=20 mean-bag on 253


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def max_tanimoto_to_pool(fp_query: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """Return max-Tanimoto per query row to any pool row (uint8 ECFP4 inputs)."""
    a = fp_query.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    out = np.zeros(n_q, dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T  # (block, n_pool)
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        out[s:e] = sim.max(axis=1)
    return out


def conf_weight(max_tan: np.ndarray) -> np.ndarray:
    """weight = max(WEIGHT_FLOOR, sigmoid(SIM_SLOPE * (max_tan - SIM_PIVOT)))."""
    s = sigmoid(SIM_SLOPE * (max_tan - SIM_PIVOT))
    return np.maximum(WEIGHT_FLOOR, s).astype(np.float64)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Tanimoto-confidence weighted isotonic on nb2240")
    print(f"          weight = max({WEIGHT_FLOOR}, sigmoid({SIM_SLOPE}*(max_tan - {SIM_PIVOT})))")
    print(f"          scaffold CV n_folds={N_FOLDS}  kf_seed={KF_SEED}")
    print(f"          gate PROMOTE < {GATE_PROMOTE}  MARGINAL_BEAT < {GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load truth + audit indices ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    # ---- Load anchor (nb2240) ----
    if not ANCHOR_OOF_PATH.exists():
        raise FileNotFoundError(f"missing anchor OOF: {ANCHOR_OOF_PATH}")
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing anchor te: {ANCHOR_TE_PATH}")
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    n_test = anchor_te.shape[0]
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"anchor_oof shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"anchor_te shape {anchor_te.shape} != ({n_test},)")
    rae_anchor_oof = float(rae(y_unb, anchor_oof))
    print(f"[load] nb2240 anchor_oof RAE = {rae_anchor_oof:.4f}")
    print(f"       anchor_oof mean={anchor_oof.mean():.3f}  std={anchor_oof.std():.3f}  "
          f"(truth_std {y_unb.std():.3f})")

    # ---- Load test SMILES (for scaffolds + Tanimoto vs train) ----
    te = load_test()
    te_smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_name_col = "name" if "name" in te.columns else "Molecule Name"
    te_smi = te[te_smi_col].astype(str).tolist()
    te_names = te[te_name_col].values

    te_std = [standardize(s) for s in te_smi]
    te_std_smi = []
    for m in te_std:
        if m is None:
            te_std_smi.append("")
        else:
            from rdkit import Chem as _Chem
            te_std_smi.append(_Chem.MolToSmiles(m))
    fp_te = morgan_fp_batch(te_std_smi)
    print(f"[test] n={n_test}  fp shape={fp_te.shape}")

    # ---- Load train for the pool ----
    train = load_train()
    train = train[train["pec50"].notna()].copy().reset_index(drop=True)
    train_std = [standardize(s) for s in train["smiles"].astype(str).tolist()]
    train_smi = []
    for m in train_std:
        if m is None:
            train_smi.append("")
        else:
            from rdkit import Chem as _Chem
            train_smi.append(_Chem.MolToSmiles(m))
    fp_train = morgan_fp_batch(train_smi)
    keep = fp_train.sum(axis=1) > 0
    fp_train = fp_train[keep]
    print(f"[train] n_train_pool={len(fp_train)}")

    # ---- Max-Tanimoto per test row to train pool ----
    print(f"[tan] computing max-Tanimoto per test row...")
    ts = time.time()
    max_tan_te = max_tanimoto_to_pool(fp_te, fp_train)  # (513,)
    print(f"[tan] done wall={time.time()-ts:.1f}s  "
          f"mean={max_tan_te.mean():.3f}  median={np.median(max_tan_te):.3f}  "
          f"q10={np.quantile(max_tan_te, 0.10):.3f}  "
          f"q90={np.quantile(max_tan_te, 0.90):.3f}")

    max_tan_unb = max_tan_te[unb_idx]
    print(f"[tan] unblind subset (253): "
          f"mean={max_tan_unb.mean():.3f}  median={np.median(max_tan_unb):.3f}  "
          f"q10={np.quantile(max_tan_unb, 0.10):.3f}")

    w_unb = conf_weight(max_tan_unb)
    w_te = conf_weight(max_tan_te)
    print(f"[weight] unb mean={w_unb.mean():.3f}  median={np.median(w_unb):.3f}  "
          f"min={w_unb.min():.3f}  max={w_unb.max():.3f}")
    print(f"[weight] te  mean={w_te.mean():.3f}  median={np.median(w_te):.3f}")

    # ---- Scaffold info ----
    unb_smiles = [te_smi[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] n_unique={n_unique_scaf}")

    # ============================================================
    # STEP 1: weighted-isotonic scaffold CV  (single kf_seed=1001)
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STEP 1: scaffold {N_FOLDS}-fold CV  weighted isotonic  kf_seed={KF_SEED}")
    print("-" * 78)

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    iso_oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold = []
    for k, (tr, va) in enumerate(splits):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(anchor_oof[tr], y_unb[tr], sample_weight=w_unb[tr])
        pred_va = ir.predict(anchor_oof[va]).astype(np.float64)
        iso_oof[va] = pred_va

        n_knots = int(len(np.unique(ir.X_thresholds_))) \
            if hasattr(ir, "X_thresholds_") else -1
        fold_rae = float(rae(y_unb[va], pred_va))
        per_fold.append({
            "fold": int(k),
            "n_tr": int(len(tr)),
            "n_va": int(len(va)),
            "n_knots": n_knots,
            "fold_val_rae": fold_rae,
            "w_tr_mean": float(w_unb[tr].mean()),
            "w_tr_min": float(w_unb[tr].min()),
        })
        print(f"   fold={k}  n_tr={len(tr):3d}  n_va={len(va):3d}  "
              f"knots={n_knots:3d}  w_tr_mean={w_unb[tr].mean():.3f}  "
              f"fold_val_rae={fold_rae:.4f}")

    if np.isnan(iso_oof).any():
        raise RuntimeError("scaffold splits did not cover all rows")

    mean_rae = float(rae(y_unb, iso_oof))
    print(f"\n[eval] pooled cross-fit RAE on 253 = {mean_rae:.4f}")

    # ============================================================
    # STEP 2: gate
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 2: gate")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae={mean_rae:.4f}  "
          f"PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}  -> {verdict}")
    print(f"[gate] delta vs nb2171 ceiling ({NB2171_REF:.4f}) = "
          f"{mean_rae - NB2171_REF:+.4f}")
    print(f"[gate] delta vs nb2240 anchor ({rae_anchor_oof:.4f}) = "
          f"{mean_rae - rae_anchor_oof:+.4f}")

    # ============================================================
    # STEP 3: deploy
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 3: deploy (weighted iso on full 253)")
    print("-" * 78)
    ir_deploy = IsotonicRegression(out_of_bounds="clip")
    ir_deploy.fit(anchor_oof, y_unb, sample_weight=w_unb)
    n_knots_deploy = int(len(np.unique(ir_deploy.X_thresholds_))) \
        if hasattr(ir_deploy, "X_thresholds_") else -1
    in_rae_deploy = float(rae(y_unb, ir_deploy.predict(anchor_oof)))
    te_final = ir_deploy.predict(anchor_te).astype(np.float64)
    te_unb_in = float(rae(y_unb, te_final[unb_idx]))
    print(f"   deploy n_knots               = {n_knots_deploy}")
    print(f"   deploy in-sample RAE (253)   = {in_rae_deploy:.4f}")
    print(f"   te[unb_idx] in-sample RAE    = {te_unb_in:.4f}")
    print(f"   te_final mean={te_final.mean():.3f}  std={te_final.std():.3f}  "
          f"(anchor_te {anchor_te.mean():.3f}/{anchor_te.std():.3f})")

    # ============================================================
    # STEP 4: save
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 4: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, iso_oof.astype(np.float32))
    np.save(te_path, te_final.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_tan_conf_isotonic.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smi,
            "Molecule Name": te_names,
            "pEC50": te_final.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip submission] verdict=FAIL")

    # ---- summary JSON ----
    summary = {
        "tag": TAG,
        "method": "tanimoto_confidence_weighted_isotonic_on_nb2240",
        "paradigm": "weighted_post_hoc_isotonic_df_~knots",
        "anchor": "nb2240_mean_bag_oof_K20",
        "anchor_pre_unblind": True,
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_oof_rae": rae_anchor_oof,
        "sim_pivot": SIM_PIVOT,
        "sim_slope": SIM_SLOPE,
        "weight_floor": WEIGHT_FLOOR,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "max_tan_te_stats": {
            "mean": float(max_tan_te.mean()),
            "median": float(np.median(max_tan_te)),
            "q10": float(np.quantile(max_tan_te, 0.10)),
            "q90": float(np.quantile(max_tan_te, 0.90)),
        },
        "max_tan_unb_stats": {
            "mean": float(max_tan_unb.mean()),
            "median": float(np.median(max_tan_unb)),
            "q10": float(np.quantile(max_tan_unb, 0.10)),
            "q90": float(np.quantile(max_tan_unb, 0.90)),
        },
        "weight_unb_stats": {
            "mean": float(w_unb.mean()),
            "median": float(np.median(w_unb)),
            "min": float(w_unb.min()),
            "max": float(w_unb.max()),
        },
        "n_unique_scaffolds": n_unique_scaf,
        "per_fold": per_fold,
        "mean_rae": mean_rae,
        "deploy_n_knots": n_knots_deploy,
        "in_sample_rae": in_rae_deploy,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_final.mean()),
        "te_std": float(te_final.std()),
        "anchor_te_mean": float(anchor_te.mean()),
        "anchor_te_std": float(anchor_te.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "delta_vs_anchor": mean_rae - rae_anchor_oof,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict != "FAIL" else None),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                  = nb2240 K=20 (RAE {rae_anchor_oof:.4f})")
    print(f"   weight rule             = max({WEIGHT_FLOOR}, sigmoid({SIM_SLOPE}*(s-{SIM_PIVOT})))")
    print(f"   mean_rae (cross-fit)    = {mean_rae:.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   delta vs nb2171 ceiling = {mean_rae - NB2171_REF:+.4f}")
    print(f"   te[unb_idx] in-RAE      = {te_unb_in:.4f}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "verdict",
        "delta_vs_nb2171",
        "delta_vs_anchor",
        "te_unb_in_sample_rae",
        "deploy_n_knots",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
