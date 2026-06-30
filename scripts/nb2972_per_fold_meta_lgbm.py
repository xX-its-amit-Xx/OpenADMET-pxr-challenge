"""nb2972 -- Per-fold LGBM meta-stack on 4 K-anchor deep-30 OOFs.

NEW PARADIGM:
    Instead of SLSQP simplex (linear convex combination, 3 free params on the
    simplex of 4) or greedy weight search, train a TINY LGBM per fold on
    (4 K-anchor OOFs, y) to learn FOLD-SPECIFIC, NON-LINEAR weights.

    LGBM(max_depth=2, num_leaves=3, n_estimators=50, learning_rate=0.05) is
    capacity-matched to n=~200 per train-fold: ~50 splits over 4 features =
    a small constellation of (feature, threshold) interactions.  This is the
    minimal non-linear step beyond the simplex.

ANCHOR POOL (n=4, all PRE-clean deep-30 K-RFE pyramids on chemprop_aux):
    1. K=18  -> data/processed/nb2960_K18_30seed_oof.npy + _te.npy
    2. K=20  -> data/processed/nb2960_K20_30seed_oof.npy + _te.npy
    3. K=24  -> data/processed/nb2960_K24_30seed_oof.npy + _te.npy
    4. K=28  -> data/processed/nb2960_K28_30seed_oof.npy + _te.npy
    nb1191, chemprop_aux EXCLUDED (the 4 K-anchors already carry the
    chemprop_aux anchor inside their residual stack).

PROTOCOL:
    1. Stack 4 OOFs into P_unb (253, 4) and 4 te into P_te (513, 4).
    2. Scaffold 5-fold CV with single kf_seed=1001 (per task spec).
    3. Per fold: fit LGBM(max_depth=2, num_leaves=3, n_est=50, lr=0.05)
       on (P_unb[tr], y[tr]); predict on val.
    4. Pool val preds -> oof; report fold-mean RAE and pooled RAE.
    5. Deploy: refit one LGBM on all 253 (P_unb, y) -> predict on (513, 4)
       te stack -> te_nb2972.npy.

GATE:
    mean_rae < 0.4567 -> "BETTER"
    mean_rae < 0.4570 -> "PROMOTE"
    else              -> "FAIL"

References:
    nb2960 cached blend (5-seed)        = 0.4576
    nb2962 4-K simplex SLSQP            = competing baseline
    nb2171 deep-30 ceiling PRIMARY-1    = 0.4682
    chemprop_aux PRE-clean anchor       = 0.6216

ARTIFACTS:
    data/processed/nb2972_summary.json
    data/processed/nb2972_pred_oof.npy   (253,) float32
    data/processed/te_nb2972.npy         (513,) float32
    submissions/nb2972_per_fold_meta_lgbm.csv  (non-FAIL only)
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
import lightgbm as lgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2972"

# -- Per-fold LGBM hyperparameters (tiny meta-stack capacity) ----------------
META_MAX_DEPTH = 2
META_NUM_LEAVES = 3
META_N_EST = 50
META_LR = 0.05
META_SEED = 1001  # fixed deploy seed (also used as kf_seed)

# -- 5-fold scaffold CV with single seed ------------------------------------
N_FOLDS = 5
KF_SEED = 1001

# -- Gates -------------------------------------------------------------------
GATE_BETTER = 0.4567
GATE_PROMOTE = 0.4570

# -- References --------------------------------------------------------------
NB2960_REF = 0.4576
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216

# -- Anchor pool (deep-30 K-RFE pyramids on chemprop_aux residual stack) ----
ANCHORS = [
    ("K18", DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
            DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
            "PRE-clean deep-30 K=18 RFE residual stack on chemprop_aux anchor"),
    ("K20", DATA_PROCESSED / "nb2960_K20_30seed_oof.npy",
            DATA_PROCESSED / "nb2960_K20_30seed_te.npy",
            "PRE-clean deep-30 K=20 RFE residual stack on chemprop_aux anchor"),
    ("K24", DATA_PROCESSED / "nb2960_K24_30seed_oof.npy",
            DATA_PROCESSED / "nb2960_K24_30seed_te.npy",
            "PRE-clean deep-30 K=24 RFE residual stack on chemprop_aux anchor"),
    ("K28", DATA_PROCESSED / "nb2960_K28_30seed_oof.npy",
            DATA_PROCESSED / "nb2960_K28_30seed_te.npy",
            "PRE-clean deep-30 K=28 RFE residual stack on chemprop_aux anchor"),
]


def _meta_lgbm(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        max_depth=META_MAX_DEPTH,
        num_leaves=META_NUM_LEAVES,
        n_estimators=META_N_EST,
        learning_rate=META_LR,
        min_child_samples=5,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold LGBM meta-stack on 4 K-anchor deep-30 OOFs")
    print(f"          meta-LGBM(max_depth={META_MAX_DEPTH}, num_leaves={META_NUM_LEAVES}, "
          f"n_est={META_N_EST}, lr={META_LR})")
    print(f"          5-fold scaffold CV, kf_seed={KF_SEED}")
    print(f"          gate: <{GATE_BETTER} BETTER / <{GATE_PROMOTE} PROMOTE")
    print("=" * 78)

    # -- Load truth + scaffolds ----------------------------------------------
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

    # -- Load 4 K-anchor OOFs + te -------------------------------------------
    anchor_names: list[str] = []
    anchor_provenance: dict[str, str] = {}
    oof_cols: list[np.ndarray] = []
    te_cols: list[np.ndarray] = []
    for name, oof_p, te_p, prov in ANCHORS:
        if not oof_p.exists():
            raise FileNotFoundError(f"{name}: oof missing at {oof_p}")
        if not te_p.exists():
            raise FileNotFoundError(f"{name}: te missing at {te_p}")
        oof = np.load(oof_p).astype(np.float64)
        te_v = np.load(te_p).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{name}: oof shape {oof.shape} != ({n_unb},)")
        if te_v.shape[0] != n_test:
            raise ValueError(f"{name}: te shape {te_v.shape[0]} != {n_test}")
        anchor_names.append(name)
        anchor_provenance[name] = prov
        oof_cols.append(oof)
        te_cols.append(te_v)
    K = len(anchor_names)
    P_unb = np.column_stack(oof_cols).astype(np.float32)  # (253, 4)
    P_te = np.column_stack(te_cols).astype(np.float32)    # (513, 4)

    anchor_in_rae = {k: round(float(rae(y_unb, P_unb[:, i])), 4)
                     for i, k in enumerate(anchor_names)}
    print(f"\n[anchors] K={K}")
    for k in anchor_names:
        print(f"   {k:6s}  unb_RAE={anchor_in_rae[k]:.4f}  [{anchor_provenance[k]}]")

    # Leak sanity (anchors should NOT be bit-equal to truth on >5% of rows)
    leak_flags = {}
    for i, k in enumerate(anchor_names):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # OOF correlation matrix (sanity)
    corr_mat = np.corrcoef(P_unb.T.astype(np.float64))
    print(f"\n  OOF correlation matrix:")
    print(f"        {'  '.join([f'{k:>6s}' for k in anchor_names])}")
    for i, ki in enumerate(anchor_names):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(K)])
        print(f"   {ki:>6s}  {row}")

    # -- Scaffolds ------------------------------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[scaffold] unique={n_unique_scaf}  n_folds={N_FOLDS}  kf_seed={KF_SEED}")

    # -- 5-fold scaffold CV per-fold meta-LGBM --------------------------------
    print("\n" + "-" * 78)
    print(f"per-fold meta-LGBM (5-fold scaffold CV, kf_seed={KF_SEED})")
    print("-" * 78)
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                    shuffle=True, seed=KF_SEED)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rows: list[dict] = []
    per_fold_rae: list[float] = []
    per_fold_imp: list[list[float]] = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        mdl = _meta_lgbm(seed=KF_SEED * 11 + f_idx)
        mdl.fit(P_unb[tr_loc], y_unb[tr_loc])
        val_pred = mdl.predict(P_unb[va_loc])
        oof_blend[va_loc] = val_pred
        r_val = float(rae(y_unb[va_loc], val_pred))
        per_fold_rae.append(r_val)
        try:
            imp = mdl.booster_.feature_importance(importance_type="gain")
        except Exception:
            imp = np.zeros(K, dtype=float)
        imp = np.asarray(imp, dtype=float)
        if imp.sum() > 0:
            imp_norm = imp / imp.sum()
        else:
            imp_norm = np.full(K, 1.0 / K)
        per_fold_imp.append(imp_norm.tolist())
        per_fold_rows.append({
            "fold": int(f_idx),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
            "val_rae": round(r_val, 4),
            "gain_importance": {anchor_names[k]: round(float(imp_norm[k]), 4)
                                for k in range(K)},
        })
        print(f"   fold={f_idx}  n_tr={len(tr_loc):>3d}  n_va={len(va_loc):>3d}  "
              f"val_RAE={r_val:.4f}  "
              f"imp={{ " + ", ".join(
                  f"{anchor_names[k]}={imp_norm[k]:.3f}" for k in range(K)) + " }}")

    assert not np.any(np.isnan(oof_blend)), "OOF coverage gap (scaffold splits)"
    pooled_rae = float(rae(y_unb, oof_blend))
    mean_fold_rae = float(np.mean(per_fold_rae))
    print(f"\n  pooled_RAE  = {pooled_rae:.4f}")
    print(f"  mean_fold   = {mean_fold_rae:.4f}  "
          f"min={min(per_fold_rae):.4f}  max={max(per_fold_rae):.4f}")

    # -- Gate -----------------------------------------------------------------
    if pooled_rae < GATE_BETTER:
        verdict = "BETTER"
    elif pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    else:
        verdict = "FAIL"
    delta_vs_nb2960 = pooled_rae - NB2960_REF
    delta_vs_nb2171 = pooled_rae - NB2171_REF
    print(f"\n[gate] pooled_RAE={pooled_rae:.4f}  "
          f"(< {GATE_BETTER} BETTER / < {GATE_PROMOTE} PROMOTE)  -> {verdict}")
    print(f"  delta vs nb2960 ({NB2960_REF}) = {delta_vs_nb2960:+.4f}")
    print(f"  delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")

    # -- Deploy: refit one LGBM on all 253 -> predict (513, 4) te ------------
    print("\n" + "-" * 78)
    print(f"DEPLOY: refit meta-LGBM on full 253 -> predict on (513, 4) te")
    print("-" * 78)
    mdl_full = _meta_lgbm(seed=META_SEED)
    mdl_full.fit(P_unb, y_unb)
    te_pred = mdl_full.predict(P_te).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    try:
        full_imp = mdl_full.booster_.feature_importance(importance_type="gain")
    except Exception:
        full_imp = np.zeros(K, dtype=float)
    full_imp = np.asarray(full_imp, dtype=float)
    if full_imp.sum() > 0:
        full_imp_norm = full_imp / full_imp.sum()
    else:
        full_imp_norm = np.full(K, 1.0 / K)
    full_imp_dict = {anchor_names[k]: round(float(full_imp_norm[k]), 4)
                     for k in range(K)}
    print(f"   full_pool gain_importance = {full_imp_dict}")
    print(f"   te(deploy) mean={te_pred.mean():.3f} std={te_pred.std():.3f}  "
          f"in-sample unb_RAE={te_unb_in:.4f}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_blend.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_per_fold_meta_lgbm.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict=FAIL; no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "per_fold_meta_lgbm_on_4K_deep30_anchors",
        "paradigm": "non_linear_per_fold_meta_stack",
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_provenance,
        "anchor_in_rae": anchor_in_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "anchor_pre_unblind": True,
        "nb1191_excluded": True,
        "chemprop_aux_excluded": True,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": anchor_names,
        "meta_lgbm_params": {
            "max_depth": META_MAX_DEPTH,
            "num_leaves": META_NUM_LEAVES,
            "n_estimators": META_N_EST,
            "learning_rate": META_LR,
            "min_child_samples": 5,
            "reg_lambda": 1.0,
        },
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "per_fold_rows": per_fold_rows,
        "per_fold_val_rae": per_fold_rae,
        "per_fold_gain_importance": per_fold_imp,
        "pooled_rae": pooled_rae,
        "mean_fold_rae": mean_fold_rae,
        "mean_rae": pooled_rae,
        "deploy_full_pool_gain_importance": full_imp_dict,
        "gate_better": GATE_BETTER,
        "gate_promote": GATE_PROMOTE,
        "verdict": verdict,
        "delta_vs_nb2960": delta_vs_nb2960,
        "delta_vs_nb2171": delta_vs_nb2171,
        "nb2960_ref": NB2960_REF,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "te_unb_in_sample_rae": round(te_unb_in, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    # Compact final echo
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor_in_rae        = {anchor_in_rae}")
    print(f"   pooled_rae           = {pooled_rae:.4f}")
    print(f"   mean_fold_rae        = {mean_fold_rae:.4f}")
    print(f"   delta vs nb2960      = {delta_vs_nb2960:+.4f}")
    print(f"   delta vs nb2171      = {delta_vs_nb2171:+.4f}")
    print(f"   deploy gain_imp      = {full_imp_dict}")
    print(f"   te_unb_in_sample_rae = {te_unb_in:.4f}")
    print(f"   verdict              = {verdict}")
    print(f"   wall                 = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
