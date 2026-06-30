"""nb2852 -- IsolationForest pre-filter (remove anomalous train rows) on K=20.

NEW PARADIGM (anomaly-based row pruning, vs row reweighting / row bootstrap):
    Detect rows in the unblind-train fold that lie far from the bulk of the
    feature distribution and DROP them entirely before fitting the LGBM
    residual head. Hypothesis: a small slice of high-leverage anomalous rows
    distort tree splits enough that removing them yields a cleaner fit on the
    typical-density manifold.

    Differs from:
      - bagging  (resample with replacement, no rows truly removed)
      - reweighting (rows weighted, still present)
      - subspace (drop features, not rows)
      - novel-scaffold reject (chemistry prior, NOT density-based)

PROTOCOL:
    For each contamination c in {0.02, 0.05, 0.10}:
        For each kf_seed in {1001..1005}:
            Scaffold 5-fold CV on 253 unblind:
                On TR fold:
                    fit IsolationForest(contamination=c, random_state=42)
                    on X_unb_K20[TR] only (NO target leakage)
                    pred labels: -1 = anomalous (drop), +1 = inlier (keep)
                    -> kept_idx subset of TR
                Fit LGBM(K=20) on (X[kept_idx], resid[kept_idx])
                Predict resid on VA -> blend with anchor_unb[VA] -> RAE
            Pool VA preds across folds, compute pooled RAE
        Wide-seed mean over 5 kf_seeds.
    Pick best contamination by mean_pooled_rae.

    Deploy: refit IsolationForest on ALL 253 X_unb_K20 (best c), drop
    anomalous rows, refit LGBM on kept rows, predict 513 -> blend with
    anchor_te. Save pred_oof from best-contamination kf_seed=1001
    (single-seed reference).

GATE:
    best_mean_rae < 0.4570  -> "PROMOTE"
    best_mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else                    -> "FAIL"

OUTPUTS:
    data/processed/nb2852_summary.json
    data/processed/nb2852_pred_oof.npy   (253,) float32 (best c, seed=1001 OOF)
    data/processed/te_nb2852.npy         (513,) float32 deploy
    submissions/nb2852_isolation_forest_filter.csv
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
import lightgbm as lgb
from sklearn.ensemble import IsolationForest

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2852"

# --------------------------------------------------------------------------
# Substrate
# --------------------------------------------------------------------------
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
K_SLICE = 20             # use first K=20 cols of 117-col substrate

# Outer scaffold CV
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Contamination sweep
CONTAM_GRID = [0.02, 0.05, 0.10]
IFOREST_RANDOM_STATE = 42
IFOREST_N_ESTIMATORS = 100  # sklearn default

# Output clip range
CLIP_LO = 3.0
CLIP_HI = 8.0

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


def _new_lgbm(seed: int) -> lgb.LGBMRegressor:
    """LGBM K=20 head: same family as nb2832 random-subspace bag."""
    return lgb.LGBMRegressor(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


def _iforest_keep_mask(X_tr: np.ndarray, contamination: float) -> np.ndarray:
    """Fit IsolationForest on X_tr only (no labels). Return bool mask of
    INLIER rows (True = keep)."""
    iso = IsolationForest(
        contamination=contamination,
        random_state=IFOREST_RANDOM_STATE,
        n_estimators=IFOREST_N_ESTIMATORS,
        n_jobs=2,
    )
    iso.fit(X_tr)
    labels = iso.predict(X_tr)  # +1 = inlier, -1 = outlier
    return labels == 1


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- IsolationForest pre-filter on K={K_SLICE}, "
          f"sweep contamination={CONTAM_GRID}")
    print("=" * 78)

    # ---- Load test set + scaffolds + truth ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    name_col = "name" if "name" in te.columns else "Molecule Name"
    te_smiles = te[smi_col].astype(str).tolist()
    te_names = te[name_col].values
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
    print(f"[feat] X_unb_K20={X_unb.shape}  X_te_K20={X_te.shape}")

    # ---- Anchor (chemprop_aux, PRE-unblind verified-clean) ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f}")

    # ---- Residual target ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}")

    # ---- Sweep contamination ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}  contamination={CONTAM_GRID}")
    print("-" * 78)

    per_c_results = {}
    oof_per_c_seed1001 = {}  # store OOF preds at seed=1001 for each c

    for c in CONTAM_GRID:
        ts_c = time.time()
        per_seed_pooled = []
        per_seed_mean_fold = []
        per_seed_drop_counts = []
        oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
        per_seed_summary = []

        for s_idx, kf_seed in enumerate(KF_SEEDS):
            ts = time.time()
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
            )
            mean_oof_pred = np.full(n_unb, np.nan, dtype=np.float64)
            fold_rae = []
            fold_drops = []
            for fi, (tr_loc, va_loc) in enumerate(splits):
                X_tr = X_unb[tr_loc]
                X_va = X_unb[va_loc]
                r_tr = resid_unb[tr_loc]

                # IsolationForest on X_tr only, drop anomalous
                keep_mask = _iforest_keep_mask(X_tr, contamination=c)
                n_drop = int((~keep_mask).sum())
                fold_drops.append(n_drop)
                X_tr_kept = X_tr[keep_mask]
                r_tr_kept = r_tr[keep_mask]

                # Guard rail: if too few kept rows, fall back to all TR
                if len(r_tr_kept) < 20:
                    X_tr_kept = X_tr
                    r_tr_kept = r_tr

                seed_fit = int(kf_seed * 1000 + fi)
                m = _new_lgbm(seed_fit)
                m.fit(X_tr_kept, r_tr_kept)
                resid_va = m.predict(X_va)
                blended = np.clip(anchor_unb[va_loc] + resid_va,
                                  CLIP_LO, CLIP_HI)
                mean_oof_pred[va_loc] = blended
                fold_rae.append(float(rae(y_unb[va_loc], blended)))

            assert not np.isnan(mean_oof_pred).any(), "OOF has NaN"
            pooled = float(rae(y_unb, mean_oof_pred))
            mean_fold = float(np.mean(fold_rae))
            per_seed_pooled.append(pooled)
            per_seed_mean_fold.append(mean_fold)
            per_seed_drop_counts.append(int(sum(fold_drops)))
            oof_seed_stack[s_idx] = mean_oof_pred
            per_seed_summary.append({
                "kf_seed": int(kf_seed),
                "pooled_rae": pooled,
                "mean_fold_rae": mean_fold,
                "total_dropped_rows": int(sum(fold_drops)),
                "per_fold_drops": [int(d) for d in fold_drops],
                "wall_sec": round(time.time() - ts, 2),
            })
            print(f"   c={c:.2f}  seed={kf_seed}  pooled={pooled:.4f}  "
                  f"mean_fold={mean_fold:.4f}  dropped={sum(fold_drops)}/"
                  f"{n_unb*(N_FOLDS-1)//N_FOLDS:d}  "
                  f"wall={time.time() - ts:.1f}s")

        mean_pooled = float(np.mean(per_seed_pooled))
        std_pooled = float(np.std(per_seed_pooled))
        oof_seed1001 = oof_seed_stack[0].copy()
        oof_per_c_seed1001[c] = oof_seed1001

        per_c_results[c] = {
            "contamination": float(c),
            "mean_pooled_rae": mean_pooled,
            "std_pooled_rae": std_pooled,
            "per_seed_pooled": per_seed_pooled,
            "per_seed_mean_fold": per_seed_mean_fold,
            "per_seed_drop_counts": per_seed_drop_counts,
            "per_seed_summary": per_seed_summary,
            "wall_sec": round(time.time() - ts_c, 2),
        }
        print(f"   [c={c:.2f}] wide-seed mean_pooled = {mean_pooled:.4f} +/- "
              f"{std_pooled:.4f}  (wall {time.time() - ts_c:.1f}s)")

    # ---- Pick best contamination ----
    best_c = min(per_c_results.keys(),
                 key=lambda c: per_c_results[c]["mean_pooled_rae"])
    best_mean = per_c_results[best_c]["mean_pooled_rae"]
    best_std = per_c_results[best_c]["std_pooled_rae"]
    oof_best = oof_per_c_seed1001[best_c]  # seed=1001 OOF at best c

    print("\n" + "-" * 78)
    print("CONTAMINATION SWEEP RESULT")
    print("-" * 78)
    for c in CONTAM_GRID:
        r = per_c_results[c]
        marker = "  <-- best" if c == best_c else ""
        print(f"   c={c:.2f}  mean={r['mean_pooled_rae']:.4f} +/- "
              f"{r['std_pooled_rae']:.4f}{marker}")

    # ---- Gate ----
    if best_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   best_c              = {best_c}")
    print(f"   best_mean_pooled    = {best_mean:.4f}")
    print(f"   gate PROMOTE        = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL       = < {GATE_MARGINAL}")
    print(f"   verdict             = {verdict}")

    # ---- Deploy: refit IForest on all 253, drop anomalous, refit LGBM, predict 513 ----
    print("\n" + "-" * 78)
    print(f"DEPLOY: best contamination c={best_c} on all 253, then LGBM K=20")
    print("-" * 78)
    deploy_keep = _iforest_keep_mask(X_unb, contamination=best_c)
    n_deploy_drop = int((~deploy_keep).sum())
    print(f"[deploy] IForest dropped {n_deploy_drop}/{n_unb} rows "
          f"(contamination={best_c})")
    X_unb_kept = X_unb[deploy_keep]
    resid_kept = resid_unb[deploy_keep]
    deploy_seed = 424242
    m_deploy = _new_lgbm(deploy_seed)
    m_deploy.fit(X_unb_kept, resid_kept)
    resid_te = m_deploy.predict(X_te)
    deploy_te = np.clip(anchor_te + resid_te, CLIP_LO, CLIP_HI).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  "
          f"std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f} (in-sample)")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_best.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}  (best c={best_c}, kf_seed=1001 OOF)")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_isolation_forest_filter.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_te,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": (
            "IsolationForest pre-filter on K=20: per fold/seed, fit IForest "
            "on TR features only (no labels), drop anomalous rows, then fit "
            "LGBM(K=20) on kept rows for chemprop_aux residual. Sweep "
            f"contamination in {CONTAM_GRID}, pick best by wide-seed mean."
        ),
        "paradigm": "isolation_forest_row_filter_lgbm_residual_K20_chemprop_aux",
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "contamination_grid": CONTAM_GRID,
        "best_contamination": float(best_c),
        "best_mean_pooled_rae": best_mean,
        "best_std_pooled_rae": best_std,
        "iforest_random_state": IFOREST_RANDOM_STATE,
        "iforest_n_estimators": IFOREST_N_ESTIMATORS,
        "lgbm_hparams": dict(
            objective="regression",
            max_depth=4,
            num_leaves=15,
            n_estimators=300,
            learning_rate=0.03,
            min_child_samples=5,
            reg_lambda=2.0,
        ),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim_substrate": int(X_unb.shape[1]),
        "k_slice_first_n_of_117": K_SLICE,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "per_contamination_results": {
            str(c): per_c_results[c] for c in CONTAM_GRID
        },
        "deploy_n_dropped": n_deploy_drop,
        "deploy_n_kept": int(n_unb - n_deploy_drop),
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(deploy_te.mean()),
        "te_std": float(deploy_te.std()),
        "mean_rae": best_mean,
        "delta_vs_anchor": best_mean - rae_anchor_unb,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best contamination              = {best_c}")
    print(f"   best wide-seed mean pooled RAE  = {best_mean:.4f} +/- "
          f"{best_std:.4f}  ({verdict})")
    print(f"   delta vs anchor                 = "
          f"{best_mean - rae_anchor_unb:+.4f}")
    print(f"   te[unb_idx] in_sample RAE       = {te_unb_in_rae:.4f}")
    print(f"   wall                            = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_contamination",
        "best_mean_pooled_rae",
        "best_std_pooled_rae",
        "delta_vs_anchor",
        "verdict",
        "te_unb_in_sample_rae",
        "deploy_n_dropped",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
