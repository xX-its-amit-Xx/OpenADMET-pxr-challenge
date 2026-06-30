"""nb2730 -- Quantile Forest (modern, maintained) on K=20 chemprop_aux residual.

NEW PARADIGM:
    `quantile-forest` is the actively maintained, numpy 2.x compatible
    replacement for the unmaintained scikit-garden.  It implements the same
    Meinshausen 2006 algorithm: fit ONE random forest under squared loss,
    cache each leaf's training-row labels, and at predict time recover the
    conditional y|x distribution by aggregating training labels from the
    leaves that the test row lands in across the bag.  Any quantile can
    then be read off post-hoc -- bagging variance reduction is shared
    across all quantiles.

    Distinction from quantile-LGBM (loss-driven, one model per quantile)
    and from scikit-garden (numpy<2 dead-end): quantile-forest is the
    cleanest non-boosting / non-loss-driven path to a per-row conditional
    distribution available on the current python/numpy stack.

PROTOCOL:
    1. Try `import quantile_forest`; if it fails, save INSTALL_FAILED
       summary and exit clean (no NPY artefacts written).
    2. Slice X_K20 = first 20 cols of X_117_unb / X_117_te.
    3. Anchor: chemprop_aux (PRE-unblind, verified clean).
       Residual target = y_unb - anchor_unb.
    4. Model: RandomForestQuantileRegressor(
                n_estimators=300, max_depth=8,
                min_samples_leaf=2, random_state=42).
       Fit on residual; predict 0.5 quantile (median) per row -> resid_hat.
       Final per-row pred = anchor + resid_hat.
    5. 5-fold scaffold CV on 253 unblind, kf_seeds {1001..1005}.
    6. Deploy: refit on all 253 -> predict 0.5 quantile on 513.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else            -> "FAIL"

Outputs:
    scripts/nb2730_quantile_forest_modern.py
    data/processed/nb2730_summary.json
    data/processed/nb2730_pred_oof.npy   (253,) float32   [only if no INSTALL_FAILED]
    data/processed/te_nb2730.npy         (513,) float32   [only if no INSTALL_FAILED]
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

# quantile-forest import is wrapped so a missing/uninstallable package
# exits clean with verdict INSTALL_FAILED rather than crashing the cron.
QF_IMPORT_OK = True
QF_IMPORT_ERR = None
RandomForestQuantileRegressor = None
try:
    from quantile_forest import RandomForestQuantileRegressor  # type: ignore
except Exception as _e:  # pragma: no cover
    QF_IMPORT_OK = False
    QF_IMPORT_ERR = repr(_e)

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2730"

# --------------------------------------------------------------------------
# Quantile Forest hyperparameters (spec)
# --------------------------------------------------------------------------
QF_N_ESTIMATORS = 300
QF_MAX_DEPTH = 8
QF_MIN_SAMPLES_LEAF = 2
QF_RANDOM_STATE = 42
QF_QUANTILE = 0.5  # median

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Number of K=20 cols sliced from the 117-col block (first-20 contract).
K_SLICE = 20


def _new_qf(seed: int = QF_RANDOM_STATE):
    return RandomForestQuantileRegressor(
        n_estimators=QF_N_ESTIMATORS,
        max_depth=QF_MAX_DEPTH,
        min_samples_leaf=QF_MIN_SAMPLES_LEAF,
        random_state=int(seed),
        default_quantiles=QF_QUANTILE,
        n_jobs=2,
    )


def _save_install_failed_summary(reason: str, t0: float) -> dict:
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    summary = {
        "tag": TAG,
        "method": "quantile_forest_modern_k20_chemprop_aux_residual",
        "paradigm": (
            "conditional_quantile_distribution_per_row_meinshausen_2006_"
            "quantile_forest_RandomForestQuantileRegressor_modern"
        ),
        "install_failed": True,
        "install_error": reason,
        "verdict": "INSTALL_FAILED",
        "promote": False,
        "marginal_beat": False,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "qf_n_estimators": QF_N_ESTIMATORS,
        "qf_max_depth": QF_MAX_DEPTH,
        "qf_min_samples_leaf": QF_MIN_SAMPLES_LEAF,
        "qf_random_state": QF_RANDOM_STATE,
        "qf_quantile": QF_QUANTILE,
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}  (INSTALL_FAILED)")
    return summary


def _qf_predict_median(mdl, X):
    """Predict the 0.5 quantile (median) of the conditional y|x distribution.

    `quantile_forest.RandomForestQuantileRegressor.predict(X, quantiles=0.5)`
    returns a 1-D array of medians (one per row).  We coerce to float64 and
    flatten defensively in case a future minor-release returns (N,1).
    """
    out = mdl.predict(X, quantiles=QF_QUANTILE)
    return np.asarray(out, dtype=np.float64).reshape(-1)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Quantile Forest (modern) on K=20 chemprop_aux residual")
    print("=" * 78)

    if not QF_IMPORT_OK:
        print(f"[FATAL] quantile_forest import failed: {QF_IMPORT_ERR}")
        print(
            "[FATAL] `pip install quantile-forest` did not produce an "
            "importable module on this python/numpy combo. Exiting clean "
            "with INSTALL_FAILED."
        )
        return _save_install_failed_summary(QF_IMPORT_ERR, t0)

    # ---- Load test set + scaffolds + truth ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_smiles = te[smi_col].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_uniq_scaf}")

    # ---- Load X_117 then slice to first K=20 cols ----
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    assert X_unb_117.shape == (n_unb, 117), f"X_unb shape {X_unb_117.shape}"
    assert X_te_117.shape == (n_test, 117), f"X_te shape {X_te_117.shape}"
    X_unb = X_unb_117[:, :K_SLICE].astype(np.float32)
    X_te = X_te_117[:, :K_SLICE].astype(np.float32)
    print(f"[feat] X_unb_K20={X_unb.shape}  X_te_K20={X_te.shape}  "
          f"slice=first-{K_SLICE}-cols")

    # ---- Anchor (chemprop_aux, PRE-unblind verified-clean) ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f} "
          f"(PRE-clean PRIMARY-1 baseline)")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"QF: n_estimators={QF_N_ESTIMATORS}  max_depth={QF_MAX_DEPTH}  "
          f"min_samples_leaf={QF_MIN_SAMPLES_LEAF}  seed={QF_RANDOM_STATE}  "
          f"quantile={QF_QUANTILE}")
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        fold_info = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            mdl = _new_qf(seed=QF_RANDOM_STATE + kf_seed + fi)
            mdl.fit(X_unb[tr_loc], resid_unb[tr_loc])
            oof_resid[va_loc] = _qf_predict_median(mdl, X_unb[va_loc])
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(oof_resid[va_loc].mean()),
                "resid_pred_std": float(oof_resid[va_loc].std()),
            })
        assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
        oof_pred = anchor_unb + oof_resid
        # gentle clip to a sane pEC50 range
        oof_pred = np.clip(oof_pred, 3.0, 8.0)
        pooled = float(rae(y_unb, oof_pred))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "folds": fold_info,
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(oof_pred)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  wall={time.time()-ts:.1f}s")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] pooled RAE mean across seeds = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs      = {final_oof_rae:.4f}")
    print(f"[cv] delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")

    # ---- Deploy: refit on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit on all 253 unblind -> apply to 513")
    print("-" * 78)
    deploy_mdl = _new_qf(seed=QF_RANDOM_STATE)
    deploy_mdl.fit(X_unb, resid_unb)
    deploy_resid_te = _qf_predict_median(deploy_mdl, X_te)
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  "
          f"(in-sample, deploy refit on all 253)")

    # ---- Gate ----
    if pooled_rae_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {pooled_rae_mean:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "Quantile Forest (Meinshausen 2006) via quantile-forest "
            "RandomForestQuantileRegressor; predict median (0.5 quantile) "
            "of conditional y|x distribution on chemprop_aux residual over "
            "first-K=20 cols of X_117. Modern maintained replacement for "
            "scikit-garden (numpy 2.x compatible)."
        ),
        "paradigm": (
            "conditional_quantile_distribution_per_row_meinshausen_2006_"
            "quantile_forest_RandomForestQuantileRegressor_modern"
        ),
        "install_failed": False,
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "qf_n_estimators": QF_N_ESTIMATORS,
        "qf_max_depth": QF_MAX_DEPTH,
        "qf_min_samples_leaf": QF_MIN_SAMPLES_LEAF,
        "qf_random_state": QF_RANDOM_STATE,
        "qf_quantile": QF_QUANTILE,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "per_seed": per_seed,
        "mean_rae": pooled_rae_mean,
        "pooled_rae_std_seeds": pooled_rae_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "delta_vs_anchor": pooled_rae_mean - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 seeds)            = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"   delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "rae_anchor_unb",
        "delta_vs_anchor",
        "verdict",
        "te_unb_rae_in_sample",
        "install_failed",
    ):
        print(f"  {k}: {res.get(k)}")
