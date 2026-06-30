"""nb1063 -- Nonconvex LGBM-meta stack across 3 most-distinct base predictors.

Cycle 135 method. Hypothesis: the 3 most-distinct honest cross-fit base
predictors -- nb2112 (chemprop_aux + K=28 LGBM, honest 0.4698-0.4737),
nb1014 (multi-seed bag, honest residual 0.5799 via nb1133), and
chemprop_aux alone (PRE-unblind in_RAE 0.6216, honest residual 0.5879
via nb1133) -- exhibit region-conditional disagreement that a LGBM
meta-learner can exploit, since the residual structure across rows
depends on both predictor agreement and scaffold/similarity context.

Per memory feedback_stack_overfitting + feedback_train_oof_blend_transfer:
nonconvex stack on n=253 with 8 meta-features is borderline. The 5-fold
scaffold cross-fit is the only honest measurement; predicted LB carries a
+0.10 conservative shift estimate. Decision margin 0.003.

Honest cross-fit anchors used (n=253):
  - nb2103_mean_bag_oof_K28.npy   (nb2112 honest mean-bag OOF, 0.4737)
  - nb1133_nb1014_pred_oof.npy    (nb1014 residual-stacked OOF, 0.5799)
  - nb1133_chemprop_aux_pred_oof.npy (chemprop_aux residual-stacked, 0.5879)

NOTE: te_nb2112.npy[unb_idx] = 0.1006 is the DEPLOY REFIT (trained on
the 253 labels -- POST-unblind leakage). For honest cross-fit meta-stack
we MUST use the OOF files above; deploy te_*.npy may only be applied at
deploy time AFTER the meta-LGBM weights are fit on honest OOF.

Meta-features (8 per row):
  0: pred_nb2112   -- honest cross-fit anchor on 253
  1: pred_nb1014   -- honest cross-fit anchor on 253
  2: pred_chemprop -- honest cross-fit anchor on 253
  3: spread        -- max(p1,p2,p3) - min(p1,p2,p3)
  4: mean          -- mean(p1,p2,p3)
  5: std           -- std(p1,p2,p3)
  6: scaf_train_freq -- bemis_murcko scaffold freq in train (4139)
  7: max_train_sim   -- top-1 Tanimoto to train (513 test_difficulty.top1_sim)

Cross-fit: 5-fold scaffold (pxr.eval.scaffold_kfold_indices), seed 42.

Meta-LGBM (per spec):
  num_leaves=15, n_estimators=300, learning_rate=0.03,
  reg_lambda=1.0, max_depth=4, objective='regression' (mse)
  min_child_samples=5 (conservative, smallest fold ~50)

Outputs
-------
  data/processed/nb1063_pred_oof.npy        (n=253, meta cross-fit)
  data/processed/te_nb1063_meta.npy         (n=513, deploy meta preds)
  submissions/nb1063_lgbm_meta_stack.csv    (deploy)
  data/processed/nb1063_summary.json
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
from lightgbm import LGBMRegressor

from pxr.chem import add_standard_columns
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, PROJECT_ROOT

TAG = "nb1063"
SUBMISSIONS = PROJECT_ROOT / "submissions"
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

# Honest cross-fit OOF anchors on the 253 unblind set
OOF_FILES = {
    "nb2112":       "nb2103_mean_bag_oof_K28.npy",
    "nb1014":       "nb1133_nb1014_pred_oof.npy",
    "chemprop_aux": "nb1133_chemprop_aux_pred_oof.npy",
}

# Deploy-time te_* preds on 513 (used only AFTER meta-LGBM is fit on OOF)
TE_FILES = {
    "nb2112":       "te_nb2112.npy",
    "nb1014":       "te_nb1014.npy",
    "chemprop_aux": "te_chemprop_aux.npy",
}

N_FOLDS = 5
SEED = 42
DECISION_MARGIN = 0.003
NB2112_HONEST_REF = 0.4698     # nb2103 K=28 median-bag
NB2112_HONEST_MEAN = 0.4737    # nb2103 K=28 mean-bag (matches the OOF file used here)

META_LGBM = dict(
    objective="regression",
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    reg_lambda=1.0,
    max_depth=4,
    min_child_samples=5,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    verbosity=-1,
    random_state=SEED,
    n_jobs=2,
)


def _build_meta_features(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                         scaf_freq: np.ndarray,
                         max_train_sim: np.ndarray) -> np.ndarray:
    """Stack 8 meta-features per row.

    Order: [pred_nb2112, pred_nb1014, pred_chemprop,
            spread, mean, std, scaf_train_freq, max_train_sim].
    """
    P = np.column_stack([p1, p2, p3]).astype(np.float64)
    spread = P.max(axis=1) - P.min(axis=1)
    mean   = P.mean(axis=1)
    std    = P.std(axis=1)
    X = np.column_stack([
        p1, p2, p3,
        spread, mean, std,
        scaf_freq.astype(np.float64),
        max_train_sim.astype(np.float64),
    ])
    return X


def _scaffold_crossfit(X: np.ndarray, y: np.ndarray,
                       scaffolds: list[str]) -> tuple[np.ndarray, list[dict]]:
    """5-fold scaffold cross-fit meta-LGBM; return OOF preds + per-fold."""
    splits = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, seed=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(splits):
        mdl = LGBMRegressor(**META_LGBM)
        mdl.fit(X[tr_loc], y[tr_loc])
        p = mdl.predict(X[va_loc])
        oof[va_loc] = p
        r_fold = float(rae(y[va_loc], p))
        folds.append({
            "fold": k,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "rae_val": r_fold,
            "pred_mean": float(p.mean()),
            "pred_std": float(p.std()),
            "truth_mean": float(y[va_loc].mean()),
            "truth_std": float(y[va_loc].std()),
        })
        print(f"   fold {k}: n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  "
              f"RAE={r_fold:.4f}  pred_std={p.std():.3f}  "
              f"truth_std={y[va_loc].std():.3f}")
    return oof, folds


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Nonconvex LGBM-meta stack (cycle 135)")
    print(f"          3 honest cross-fit base predictors -> 8 meta-features -> LGBM")
    print(f"          meta-LGBM: num_leaves=15 n_est=300 lr=0.03 lambda=1.0 depth=4")
    print(f"          5-fold scaffold cross-fit; decision margin {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth + unblind index ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unb_idx {unb_idx.shape}, y_unb {y_unb.shape}  "
          f"(mean {y_unb.mean():.3f} std {y_unb.std():.3f})")

    # ---- Load 3 honest cross-fit base OOF preds on 253 ----
    base_oofs = {}
    base_oof_rae = {}
    for label, fname in OOF_FILES.items():
        a = np.load(DATA_PROCESSED / fname).astype(np.float64)
        assert a.shape[0] == 253, f"{fname} expected n=253 OOF, got {a.shape}"
        base_oofs[label] = a
        r = float(rae(y_unb, a))
        base_oof_rae[label] = r
        print(f"[anchor-oof] {label:14s} <- {fname:42s} RAE={r:.4f}  "
              f"mean={a.mean():.3f}  std={a.std():.3f}")

    # ---- Per-row meta-features beyond base preds ----
    te = add_standard_columns(load_test())
    tr = add_standard_columns(load_train())
    scaf_tr_counts = tr["scaffold"].value_counts().to_dict()
    te_scaffold_all = te["scaffold"].values
    te_scaffold_unb = te_scaffold_all[unb_idx]
    scaf_freq_unb = np.array(
        [scaf_tr_counts.get(s, 0) for s in te_scaffold_unb], dtype=np.int64)
    print(f"[meta] scaf_train_freq on 253: mean={scaf_freq_unb.mean():.2f}  "
          f"frac_zero={(scaf_freq_unb==0).mean():.3f}  "
          f"max={scaf_freq_unb.max()}")

    td = pd.read_parquet(DATA_PROCESSED / "test_difficulty.parquet")
    max_train_sim_all = td["top1_sim"].values
    max_train_sim_unb = max_train_sim_all[unb_idx]
    print(f"[meta] max_train_sim on 253: mean={max_train_sim_unb.mean():.3f}  "
          f"std={max_train_sim_unb.std():.3f}")

    # ---- Build meta features (n=253, 8 cols) ----
    X_meta = _build_meta_features(
        base_oofs["nb2112"], base_oofs["nb1014"], base_oofs["chemprop_aux"],
        scaf_freq_unb, max_train_sim_unb,
    )
    print(f"[X] meta feature matrix on 253: shape={X_meta.shape}")
    print(f"    cols = [p_nb2112, p_nb1014, p_chemprop, spread, mean, std, "
          f"scaf_freq, max_train_sim]")

    # ---- 5-fold scaffold cross-fit ----
    print(f"\n[cv] 5-fold scaffold cross-fit meta-LGBM ...")
    pred_oof, fold_records = _scaffold_crossfit(
        X_meta, y_unb, list(te_scaffold_unb))
    pooled_rae = float(rae(y_unb, pred_oof))
    print(f"\n[pooled] meta cross-fit RAE = {pooled_rae:.4f}")

    # ---- Compare vs anchor floor ----
    anchor_floor = base_oof_rae["nb2112"]
    delta_vs_anchor = pooled_rae - anchor_floor
    if delta_vs_anchor <= -DECISION_MARGIN:
        verdict = "BEATS_ANCHOR"
    elif abs(delta_vs_anchor) < DECISION_MARGIN:
        verdict = "TIES_ANCHOR_WITHIN_MARGIN"
    else:
        verdict = "LOSES_TO_ANCHOR"
    print(f"\n[anchor floor] nb2112 OOF mean-bag = {anchor_floor:.4f}  "
          f"(nb2112 median-bag ref = {NB2112_HONEST_REF:.4f})")
    print(f"[delta vs anchor]  meta - anchor = {delta_vs_anchor:+.4f}  "
          f"({verdict})")

    # ---- Save honest OOF artifact regardless of verdict ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / (TAG + '_pred_oof.npy')}")

    # ---- Build deploy meta predictions on 513 (only if beats anchor) ----
    deploy_csv_path = None
    te_meta_path = None
    pred_deploy_513 = None
    if verdict == "BEATS_ANCHOR":
        # Apply meta-LGBM at deploy time to 513-row te_* predictions.
        # Re-fit meta on ALL 253 (no held-out split) -- standard deploy
        # convention (mirrors nb2112 deploy refit pattern).
        deploy_mdl = LGBMRegressor(**META_LGBM)
        deploy_mdl.fit(X_meta, y_unb)
        in_sample_pred = deploy_mdl.predict(X_meta)
        in_sample_rae = float(rae(y_unb, in_sample_pred))
        print(f"[deploy] in-sample RAE on 253 (refit-on-all) = "
              f"{in_sample_rae:.4f}  (lower bound, optimistic by definition)")

        # Build 513-row meta features from te_* on full test set + per-row meta.
        te_preds = {label: np.load(DATA_PROCESSED / fname).astype(np.float64)
                    for label, fname in TE_FILES.items()}
        for label, a in te_preds.items():
            assert a.shape[0] == 513, f"{label} te shape {a.shape} != 513"
        scaf_freq_all = np.array(
            [scaf_tr_counts.get(s, 0) for s in te_scaffold_all], dtype=np.int64)
        X_meta_513 = _build_meta_features(
            te_preds["nb2112"], te_preds["nb1014"], te_preds["chemprop_aux"],
            scaf_freq_all, max_train_sim_all,
        )
        pred_deploy_513 = deploy_mdl.predict(X_meta_513).astype(np.float64)
        te_meta_path = DATA_PROCESSED / f"te_{TAG}_meta.npy"
        np.save(te_meta_path, pred_deploy_513.astype(np.float32))
        print(f"[save] {te_meta_path}  shape={pred_deploy_513.shape}  "
              f"mean={pred_deploy_513.mean():.3f}  std={pred_deploy_513.std():.3f}")

        sub = pd.DataFrame({
            "SMILES":        te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50":         pred_deploy_513,
        })
        deploy_csv_path = SUBMISSIONS / f"{TAG}_lgbm_meta_stack.csv"
        sub.to_csv(deploy_csv_path, index=False)
        print(f"[save] {deploy_csv_path}  rows={len(sub)}  cols={list(sub.columns)}")
    else:
        print(f"[deploy] SKIPPED -- verdict={verdict} (margin {DECISION_MARGIN})")

    # ---- Conservative LB shift estimate per memory ----
    predicted_lb_low = pooled_rae + 0.10
    print(f"\n[lb-estimate] conservative shift +0.10 -> predicted LB ~{predicted_lb_low:.4f}")
    print(f"               (per feedback_train_oof_blend_transfer + "
          f"feedback_stack_overfitting; n=253, 8 meta-features is borderline)")

    summary = {
        "tag": TAG,
        "method": "lgbm_meta_stack_nonconvex_3_base",
        "cycle": 135,
        "n_unb": int(n_unb),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "decision_margin": DECISION_MARGIN,
        "meta_lgbm_params": META_LGBM,
        "feature_dim": int(X_meta.shape[1]),
        "feature_names": [
            "pred_nb2112", "pred_nb1014", "pred_chemprop_aux",
            "spread_max_minus_min", "mean_3preds", "std_3preds",
            "scaf_train_freq", "max_train_sim_top1",
        ],
        "base_anchors_oof_files": OOF_FILES,
        "base_anchors_oof_rae_on_253": base_oof_rae,
        "anchor_floor_label": "nb2112",
        "anchor_floor_rae": float(anchor_floor),
        "nb2112_mean_bag_ref": NB2112_HONEST_MEAN,
        "nb2112_median_bag_ref": NB2112_HONEST_REF,
        "pooled_rae_meta_cross_fit": pooled_rae,
        "delta_vs_anchor": float(delta_vs_anchor),
        "verdict": verdict,
        "fold_records": fold_records,
        "predicted_lb_shifted_plus_0_10": float(predicted_lb_low),
        "predicted_lb_shift_note": (
            "n=253 nonconvex stack + 8 meta-features is borderline per "
            "feedback_stack_overfitting + feedback_train_oof_blend_transfer; "
            "add +0.10 conservative shift to honest cross-fit when "
            "estimating LB"
        ),
        "deploy_csv": str(deploy_csv_path) if deploy_csv_path else None,
        "te_meta_path": str(te_meta_path) if te_meta_path else None,
        "pred_oof_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
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
    print(f"  pooled_rae_meta_cross_fit: {res['pooled_rae_meta_cross_fit']:.4f}")
    print(f"  anchor_floor (nb2112):     {res['anchor_floor_rae']:.4f}")
    print(f"  delta vs anchor:           {res['delta_vs_anchor']:+.4f}")
    print(f"  verdict:                   {res['verdict']}")
    print(f"  predicted LB (+0.10):      {res['predicted_lb_shifted_plus_0_10']:.4f}")
    print(f"  deploy_csv:                {res['deploy_csv']}")
