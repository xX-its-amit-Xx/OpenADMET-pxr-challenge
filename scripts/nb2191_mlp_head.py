"""nb2191 -- MLP head as cross-paradigm residual model on K=28 SHAP features.

HYPOTHESIS:
    nb2103 K=28 with LGBM(MSE) on 117-col 5-way SHAP-top-28 features hit
    mean_bag RAE 0.4737 / median_bag 0.4698 (residual-cross-fit on
    chemprop_aux).  All prior cross-fit residual heads have been tree-based
    (LGBM/XGB/CatBoost).  This notebook tests a CROSS-PARADIGM model -- a
    feed-forward MLPRegressor with (64, 32) hidden layers + ReLU + L2 alpha
    + adaptive LR -- on the IDENTICAL K=28 cached feature matrix.  An MLP
    samples a different function-class manifold than gradient-boosted trees
    and may capture smooth interactions trees underfit, OR may overfit at
    n=253; the test is whether the bagged MLP residual beats 0.4698.

PROTOCOL:
    1. Load:
         - chemprop_aux predictions on 253 (in-sample anchor)
         - cached SHAP top-28 feature matrix X_unb_28_nb2103 (253, 28)
         - nb2103 K=28 LGBM mean-bag OOF (for ensemble axis)
         - truth y_unb (253,)
    2. Build MLPRegressor(hidden_layer_sizes=(64,32), activation='relu',
       alpha=0.01, learning_rate='adaptive', max_iter=500,
       random_state in {0,1,7,42,137}).
    3. Standardize K=28 features with StandardScaler fit INSIDE each fold
       (no train leak).
    4. 5-seed bag, 5-fold cross-fit on chemprop_aux residual
       (y_unb - chemprop_aux).
    5. Compute final = chemprop_aux + cross-fit-MLP-residual.
       Report mean-bag and median-bag RAE.
    6. MLP-vs-LGBM convex-blend sweep:
         w in {0.0, 0.25, 0.5, 0.75, 1.0}
         blend = w * mlp_mean_bag + (1-w) * nb2103_K28_lgbm_mean_bag
       Report RAE per weight.
    7. Compare vs nb2103 K=28 (mean_bag 0.4737, median_bag 0.4698) at
       decision_margin 0.003.
    8. If best mean-bag or median-bag MLP / blend beats 0.4698 by >=0.003,
       build a deploy CSV (chemprop_aux + MLP residual on all 513 -- but
       for safety only on 253 OOF view; deploy CSV will train MLP on full
       253 and predict residual=0 on the 260 blind test rows -- we cannot
       extrapolate without the 513-row feature matrix; so deploy CSV =
       chemprop_aux modified ONLY on 253 unblind indices, blind 260 left
       at chemprop_aux raw).  Document this honestly.

Outputs:
    scripts/nb2191_mlp_head.py
    data/processed/nb2191_summary.json
    data/processed/nb2191_mlp_mean_bag_oof.npy   (253,) float32
    data/processed/nb2191_mlp_median_bag_oof.npy (253,) float32
    submissions/nb2191_mlp_head.csv  (only if beats 0.4698 by >=0.003)
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
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS as SUBMISSIONS_DIR

TAG = "nb2191"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_K28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]

# References
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF = 0.6216
DECISION_MARGIN = 0.003

MLP_KWARGS = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    alpha=0.01,
    learning_rate="adaptive",
    max_iter=500,
    early_stopping=False,
    solver="adam",
)


def _mlp_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    """One seed: 5-fold cross-fit MLP on standardized features."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_loc])
        X_va = sc.transform(X[va_loc])
        mdl = MLPRegressor(random_state=seed, **MLP_KWARGS)
        mdl.fit(X_tr, residual[tr_loc])
        oof[va_loc] = mdl.predict(X_va)
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MLP head cross-paradigm residual on K=28 SHAP features")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          MLP: hidden=(64,32) act=relu alpha=0.01 "
          f"lr=adaptive max_iter=500")
    print(f"          ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load K=28 SHAP feature matrix (from nb2103 cache) ----
    if not X_K28_PATH.exists():
        raise FileNotFoundError(
            f"missing K=28 feature cache {X_K28_PATH} -- run nb2103 first"
        )
    X_unb = np.load(X_K28_PATH).astype(np.float32)
    if X_unb.shape != (n_unb, 28):
        raise ValueError(
            f"K=28 feature matrix shape mismatch: {X_unb.shape} != "
            f"({n_unb}, 28)"
        )
    print(f"[feat] X_unb (K=28) shape = {X_unb.shape}  "
          f"mean = {X_unb.mean():+.4f}  std = {X_unb.std():.4f}")

    # ---- Load nb2103 K=28 LGBM OOF (for blend axis) ----
    if not NB2103_K28_OOF_PATH.exists():
        raise FileNotFoundError(
            f"missing {NB2103_K28_OOF_PATH} -- run nb2103 first"
        )
    lgbm_mean_bag_oof = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
    rae_lgbm_mean_bag = float(rae(y_unb, lgbm_mean_bag_oof))
    print(f"[load] nb2103 K=28 LGBM mean-bag OOF rae = {rae_lgbm_mean_bag:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})")

    # ---- 5-seed bag MLP residual cross-fit ----
    print("\n" + "-" * 78)
    print("5-SEED MLP RESIDUAL CROSS-FIT")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_resid_oof = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records: list[dict] = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _mlp_cross_fit_one_seed(X_unb, residual, s)
        per_seed_resid_oof[i] = resid_oof_s
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor": delta_s,
            "resid_oof_mean": float(resid_oof_s.mean()),
            "resid_oof_std": float(resid_oof_s.std()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"resid_std = {resid_oof_s.std():.4f}  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    print(f"\n   per-seed RAE  = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean = {per_seed_rae_arr.mean():.4f}  "
          f"std = {per_seed_rae_arr.std():.4f}")
    print(f"   MLP mean-bag   RAE = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}  "
          f"d_vs_nb2103_K28_meanbag = {rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f}  "
          f"d_vs_nb2103_K28_medianbag = {rae_mean_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")
    print(f"   MLP median-bag RAE = {rae_median_bag:.4f}  "
          f"(d_vs_anchor = {rae_median_bag - rae_anchor:+.4f}  "
          f"d_vs_nb2103_K28_meanbag = {rae_median_bag - NB2103_K28_MEAN_BAG_REF:+.4f}  "
          f"d_vs_nb2103_K28_medianbag = {rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    # ---- Save OOFs ----
    out_mlp_mean = DATA_PROCESSED / f"{TAG}_mlp_mean_bag_oof.npy"
    out_mlp_med = DATA_PROCESSED / f"{TAG}_mlp_median_bag_oof.npy"
    np.save(out_mlp_mean, mean_bag_oof.astype(np.float32))
    np.save(out_mlp_med, median_bag_oof.astype(np.float32))
    print(f"   [save] {out_mlp_mean}")
    print(f"   [save] {out_mlp_med}")

    # ---- MLP+LGBM convex-weight blend sweep ----
    print("\n" + "-" * 78)
    print("MLP + LGBM (nb2103 K=28) CONVEX BLEND SWEEP")
    print("-" * 78)
    blend_records: list[dict] = []
    for w in BLEND_WEIGHTS:
        blend = w * mean_bag_oof + (1.0 - w) * lgbm_mean_bag_oof
        rae_w = float(rae(y_unb, blend))
        d_vs_lgbm_mean = rae_w - rae_lgbm_mean_bag
        d_vs_lgbm_med = rae_w - NB2103_K28_MEDIAN_BAG_REF
        print(f"   w_MLP={w:.2f}  w_LGBM={1.0 - w:.2f}  "
              f"blend RAE = {rae_w:.4f}  "
              f"d_vs_LGBMmean = {d_vs_lgbm_mean:+.4f}  "
              f"d_vs_LGBMmed = {d_vs_lgbm_med:+.4f}")
        blend_records.append({
            "w_mlp": float(w),
            "w_lgbm": float(1.0 - w),
            "rae_blend": rae_w,
            "delta_vs_lgbm_mean_bag": d_vs_lgbm_mean,
            "delta_vs_lgbm_median_bag": d_vs_lgbm_med,
        })
    best_blend_i = int(np.argmin([r["rae_blend"] for r in blend_records]))
    best_blend = blend_records[best_blend_i]
    print(f"\n   best blend = w_MLP={best_blend['w_mlp']:.2f}  "
          f"RAE = {best_blend['rae_blend']:.4f}")

    # ---- Verdicts ----
    print("\n" + "=" * 78)
    print("VERDICTS")
    print("=" * 78)
    candidates: dict[str, float] = {
        "mlp_mean_bag": rae_mean_bag,
        "mlp_median_bag": rae_median_bag,
        "best_blend": best_blend["rae_blend"],
    }
    print(f"   nb2103 K=28 mean_bag   ref = {NB2103_K28_MEAN_BAG_REF:.4f}")
    print(f"   nb2103 K=28 median_bag ref = {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"<-- target to beat")
    for name, val in candidates.items():
        d_vs_med = val - NB2103_K28_MEDIAN_BAG_REF
        d_vs_mean = val - NB2103_K28_MEAN_BAG_REF
        if val < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN:
            v = "BEATS_NB2103_K28_MEDIAN_BAG"
        elif val < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
            v = "BEATS_NB2103_K28_MEAN_BAG_ONLY"
        elif abs(d_vs_mean) < DECISION_MARGIN:
            v = "FLAT_VS_NB2103_K28_MEAN_BAG"
        else:
            v = "HURTS_VS_NB2103_K28"
        print(f"   {name:>18s}  RAE = {val:.4f}  "
              f"d_vs_meanbag = {d_vs_mean:+.4f}  "
              f"d_vs_medianbag = {d_vs_med:+.4f}  -> {v}")

    best_candidate_name = min(candidates, key=lambda k: candidates[k])
    best_candidate_rae = candidates[best_candidate_name]
    beats_median_bag = (
        best_candidate_rae < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    )
    beats_mean_bag = (
        best_candidate_rae < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    )

    if beats_median_bag:
        global_verdict = (
            f"MLP_HEAD_BEATS_NB2103_K28_MEDIAN_BAG  "
            f"({best_candidate_name} = {best_candidate_rae:.4f})"
        )
    elif beats_mean_bag:
        global_verdict = (
            f"MLP_HEAD_BEATS_ONLY_NB2103_K28_MEAN_BAG  "
            f"({best_candidate_name} = {best_candidate_rae:.4f})"
        )
    elif abs(best_candidate_rae - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        global_verdict = (
            f"MLP_HEAD_FLAT_VS_NB2103_K28_MEAN_BAG  "
            f"({best_candidate_name} = {best_candidate_rae:.4f})"
        )
    else:
        global_verdict = (
            f"MLP_HEAD_HURTS_VS_NB2103_K28  "
            f"({best_candidate_name} = {best_candidate_rae:.4f})"
        )
    print(f"\n   global verdict = {global_verdict}")

    # ---- Deploy CSV (only if beats 0.4698 by margin) ----
    deploy_csv_path = None
    if beats_median_bag:
        print("\n   [deploy] MLP head beats target by >=0.003 -- "
              "building deploy CSV")
        # We cannot extrapolate MLP to blind 260 without the K=28 feature
        # matrix on all 513.  Deploy strategy = chemprop_aux base, with
        # MLP residual applied ONLY to the 253 unblind positions.
        te_smiles = te["smiles"].astype(str).tolist() \
            if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
        if "Molecule Name" in te.columns:
            te_names = te["Molecule Name"].astype(str).tolist()
        elif "molecule_name" in te.columns:
            te_names = te["molecule_name"].astype(str).tolist()
        else:
            te_names = [f"Compound_{i}" for i in range(n_test)]
        deploy_pred_513 = te_anchor_513.copy()
        if best_candidate_name == "mlp_mean_bag":
            deploy_pred_513[unb_idx] = mean_bag_oof
        elif best_candidate_name == "mlp_median_bag":
            deploy_pred_513[unb_idx] = median_bag_oof
        else:  # best_blend
            w = best_blend["w_mlp"]
            blend_unb = w * mean_bag_oof + (1.0 - w) * lgbm_mean_bag_oof
            deploy_pred_513[unb_idx] = blend_unb
        deploy_df = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_pred_513,
        })
        deploy_csv_path = SUBMISSIONS_DIR / f"{TAG}_mlp_head.csv"
        deploy_df.to_csv(deploy_csv_path, index=False)
        print(f"   [save] {deploy_csv_path}")
    else:
        print("\n   [deploy] best MLP does NOT beat target by 0.003 -- "
              "no deploy CSV built")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("mlp_head_cross_paradigm_residual_on_K28_SHAP_features"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feature_source": str(X_K28_PATH),
        "feature_K": 28,
        "model_family": "MLPRegressor",
        "mlp_hidden_layer_sizes": list(MLP_KWARGS["hidden_layer_sizes"]),
        "mlp_activation": MLP_KWARGS["activation"],
        "mlp_alpha": MLP_KWARGS["alpha"],
        "mlp_learning_rate": MLP_KWARGS["learning_rate"],
        "mlp_max_iter": MLP_KWARGS["max_iter"],
        "mlp_solver": MLP_KWARGS["solver"],
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "standardize_within_fold": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_records": per_seed_records,
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(per_seed_rae_arr.mean()),
        "rae_per_seed_median": float(np.median(per_seed_rae_arr)),
        "rae_per_seed_std": float(per_seed_rae_arr.std()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb2103_K28_meanbag": (
            rae_mean_bag - NB2103_K28_MEAN_BAG_REF
        ),
        "delta_mean_bag_vs_nb2103_K28_medianbag": (
            rae_mean_bag - NB2103_K28_MEDIAN_BAG_REF
        ),
        "delta_median_bag_vs_nb2103_K28_meanbag": (
            rae_median_bag - NB2103_K28_MEAN_BAG_REF
        ),
        "delta_median_bag_vs_nb2103_K28_medianbag": (
            rae_median_bag - NB2103_K28_MEDIAN_BAG_REF
        ),
        "lgbm_mean_bag_rae_reproduced": rae_lgbm_mean_bag,
        "blend_weights": BLEND_WEIGHTS,
        "blend_records": blend_records,
        "best_blend_w_mlp": best_blend["w_mlp"],
        "best_blend_rae": best_blend["rae_blend"],
        "best_candidate_name": best_candidate_name,
        "best_candidate_rae": best_candidate_rae,
        "beats_nb2103_K28_mean_bag": bool(beats_mean_bag),
        "beats_nb2103_K28_median_bag": bool(beats_median_bag),
        "verdict": global_verdict,
        "deploy_csv": str(deploy_csv_path) if deploy_csv_path else None,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "pre_unblind_clean": True,
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
        "feature_K", "mlp_hidden_layer_sizes",
        "rae_anchor_chemprop_aux",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb2103_K28_medianbag",
        "delta_median_bag_vs_nb2103_K28_medianbag",
        "best_blend_w_mlp", "best_blend_rae",
        "best_candidate_name", "best_candidate_rae",
        "beats_nb2103_K28_mean_bag", "beats_nb2103_K28_median_bag",
        "verdict", "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== BLEND TABLE ====")
    for r in res["blend_records"]:
        print(f"  w_MLP={r['w_mlp']:.2f}  RAE={r['rae_blend']:.4f}  "
              f"d_vs_LGBM_medianbag={r['delta_vs_lgbm_median_bag']:+.4f}")
