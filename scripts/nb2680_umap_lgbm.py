"""nb2680 -- UMAP(20-d) non-linear feature projection + LGBM residual.

NEW PARADIGM (cycle 169 prescription = substrate change):
    Post-hoc-blend axes on chemprop_aux are exhausted; nb2171 0.4682 is
    the deep-30 PRE-clean ceiling on the K=18 / 117-col tuple.  Cycle
    167-169 closed feature-ranker, anchor-swap, deep-ensemble, rank-
    stretch, and spectral-Laplacian axes.

    UMAP differs from PCA / Laplacian by preserving LOCAL manifold
    structure under a non-linear projection (k-NN graph + cross-entropy
    fuzzy simplicial set).  LGBM on a UMAP-20 projection sees the same
    n=253 chemprop_aux residual signal but routed through 20 features
    that are non-linear summaries of the 117-col K=18 substrate.  If
    UMAP captures a manifold geometry that axis-aligned LGBM splits on
    the raw 117-col matrix do not, the residual cross-fit RAE breaks
    the 0.4682 ceiling.

    This is a substrate change in the cycle-134 sense: anchor unchanged
    (chemprop_aux), feature set CHANGED (raw 117-col -> UMAP-20).

PROTOCOL:
    1.  Load X_117 from data/processed/pyramid/X_117_unb.npy (253, 117)
        and X_117_te.npy (513, 117).
    2.  Concat (X_unb || X_te) -> StandardScaler -> umap.UMAP(
            n_components=20, n_neighbors=15, min_dist=0.1, random_state=42)
        Fit on the combined matrix (UMAP is transductive; this is the
        standard recipe for fixed test set with known SMILES).
    3.  Slice back into Z_unb (253, 20) and Z_te (513, 20).
    4.  LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03) on Z_unb
        predicting chemprop_aux residual = y_unb - anchor[unb_idx].
        For each kf_seed in {1001,1002,1003,1004,1005}:
          - 5-fold scaffold split via `scaffold_kfold_indices(seed=kf)`
          - bag of 5 LGBM lgbm_seeds {0..4} per fold
          - resid_oof[kf] = mean over 5 lgbm_seeds
          - te_resid[kf]  = mean over 5 full-fit LGBM predictions on Z_te
        per_kf_rae[kf]  = rae(y_unb, anchor + resid_oof[kf])
    5.  mean_rae = mean over 5 kf_seeds
    6.  Deploy:
          pred_oof_unb = mean over (5 kf x 5 bag) of (anchor + resid_oof)
          pred_te_513  = mean over 5 bag of (anchor_te + te_resid)
    7.  Gate:
          mean_rae < 0.4570 -> PROMOTE
          mean_rae < 0.4598 -> MARGINAL_BEAT
          else              -> FAIL

Inputs:
    data/processed/pyramid/X_117_unb.npy
    data/processed/pyramid/X_117_te.npy
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy

Outputs:
    data/processed/nb2680_summary.json
    data/processed/nb2680_pred_oof.npy   (253,) float32
    data/processed/te_nb2680.npy         (513,) float32
    submissions/nb2680_umap_lgbm.csv
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
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# umap-learn install gate (per task spec)
try:
    import umap  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"UMAP import failed: {e}")
    print("INSTALL_FAILED")
    sys.exit(0)

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2680"
PARENT_TAG = "nb2171"

# Protocol params
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
BAG_SEEDS = [0, 1, 2, 3, 4]
RESID_FOLDS = 5
UMAP_N_COMP = 20
UMAP_NEIGH = 15
UMAP_MIN_DIST = 0.1
UMAP_RND = 42

# Reference numbers
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216

# Gates (per task spec)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Input artifact paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_scaffold(X, residual, scaffolds, kf_seed, lgbm_seed):
    """5-fold scaffold-CV residual prediction (one LGBM bag entry)."""
    n = len(residual)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=kf_seed,
    )
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(lgbm_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def main():
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- UMAP(20-d) projection + LGBM residual on chemprop_aux")
    print(f"        UMAP n_comp={UMAP_N_COMP} n_neigh={UMAP_NEIGH} "
          f"min_dist={UMAP_MIN_DIST} seed={UMAP_RND}")
    print(f"        ref nb2171 deep-30 ceiling = {NB2171_REF:.4f}")
    print(f"        gate PROMOTE < {GATE_PROMOTE} / MARGINAL_BEAT < {GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load truth, anchor, scaffolds, X_117 substrate ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in 253 = {n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor_unb

    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"missing pyramid substrate: {X117_UNB_PATH} / {X117_TE_PATH}"
        )
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    print(f"[load] X_117_unb = {X_unb_117.shape}  X_117_te = {X_te_117.shape}")
    if X_unb_117.shape[1] != 117 or X_te_117.shape[1] != 117:
        raise ValueError("X_117 substrate is not 117-d")
    if X_unb_117.shape[0] != n_unb or X_te_117.shape[0] != n_test:
        raise ValueError("X_117 row counts don't match unb/test sizes")

    # ---- StandardScaler then transductive UMAP on (X_unb || X_te) ----
    print("\n" + "-" * 78)
    print(f"STEP 1: StandardScaler -> UMAP({UMAP_N_COMP}-d)  "
          f"transductive on (unb+te) = {n_unb + n_test} rows")
    print("-" * 78)
    X_all = np.vstack([X_unb_117, X_te_117]).astype(np.float32)
    scaler = StandardScaler()
    X_all_std = scaler.fit_transform(X_all).astype(np.float32)
    print(f"   X_all_std = {X_all_std.shape}  "
          f"mean={X_all_std.mean():.3e}  std={X_all_std.std():.3f}")

    import umap as umap_mod
    t_umap_0 = time.time()
    reducer = umap_mod.UMAP(
        n_components=UMAP_N_COMP,
        n_neighbors=UMAP_NEIGH,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RND,
        metric="euclidean",
        verbose=False,
    )
    Z_all = reducer.fit_transform(X_all_std).astype(np.float32)
    print(f"   UMAP fit done in {time.time()-t_umap_0:.1f}s  Z_all={Z_all.shape}")

    Z_unb = Z_all[:n_unb]
    Z_te = Z_all[n_unb:]
    print(f"   Z_unb = {Z_unb.shape}  Z_te = {Z_te.shape}")
    print(f"   Z_unb mean/std = {Z_unb.mean():.3f}/{Z_unb.std():.3f}  "
          f"Z_te mean/std = {Z_te.mean():.3f}/{Z_te.std():.3f}")

    # ---- LGBM residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"STEP 2: LGBM residual  5 kf_seeds x 5 bag_seeds = "
          f"{len(KF_SEEDS) * len(BAG_SEEDS)} cross-fits")
    print("-" * 78)

    pred_unb_all = np.zeros(
        (len(KF_SEEDS), len(BAG_SEEDS), n_unb), dtype=np.float64
    )
    pred_te_per_bag = np.zeros((len(BAG_SEEDS), n_test), dtype=np.float64)
    per_seed_te_done = np.zeros(len(BAG_SEEDS), dtype=bool)
    per_kf_bagmean_rae = []
    total = len(KF_SEEDS) * len(BAG_SEEDS)
    done = 0
    for k_i, kf_seed in enumerate(KF_SEEDS):
        print(f"\n  --- kf_seed={kf_seed} ----------")
        per_seed_corr_rae = []
        for b_i, b_seed in enumerate(BAG_SEEDS):
            ts = time.time()
            resid_oof = _residual_cross_fit_scaffold(
                Z_unb, residual, unb_scaffolds, kf_seed=kf_seed, lgbm_seed=b_seed,
            )
            pred_unb_all[k_i, b_i] = anchor_unb + resid_oof
            per_seed_corr_rae.append(
                float(rae(y_unb, anchor_unb + resid_oof))
            )
            if not per_seed_te_done[b_i]:
                te_resid = _train_full_then_predict_te(
                    Z_unb, residual, Z_te, seed=b_seed,
                )
                pred_te_per_bag[b_i] = te_anchor_513 + te_resid
                per_seed_te_done[b_i] = True
            done += 1
            print(f"      kf={kf_seed} bag={b_seed}  "
                  f"rae={per_seed_corr_rae[-1]:.4f}  "
                  f"wall={time.time()-ts:.2f}s  "
                  f"({done}/{total})")
        bagmean_pred = pred_unb_all[k_i].mean(axis=0)
        bagmean_rae = float(rae(y_unb, bagmean_pred))
        per_kf_bagmean_rae.append(bagmean_rae)
        print(f"   kf_seed={kf_seed}  5-bag-mean POOLED RAE = {bagmean_rae:.4f}")

    per_kf_arr = np.array(per_kf_bagmean_rae, dtype=np.float64)
    mean_rae = float(per_kf_arr.mean())
    std_rae = float(per_kf_arr.std(ddof=1))
    median_rae = float(np.median(per_kf_arr))
    min_rae = float(per_kf_arr.min())
    max_rae = float(per_kf_arr.max())

    print("\n" + "-" * 78)
    print("STEP 3: multi-kf statistics")
    print("-" * 78)
    print("   per-kf 5-bag-mean RAE:")
    for kf, r in zip(KF_SEEDS, per_kf_bagmean_rae):
        print(f"     kf={kf}  RAE={r:.4f}")
    print(f"\n   mean +/- std  = {mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"   median        = {median_rae:.4f}")
    print(f"   min / max     = {min_rae:.4f} / {max_rae:.4f}")
    print(f"   anchor (chemprop_aux) RAE = {rae_anchor:.4f}")
    print(f"   delta vs anchor           = {mean_rae - rae_anchor:+.4f}")
    print(f"   delta vs nb2171 (0.4682)  = {mean_rae - NB2171_REF:+.4f}")

    # ---- Deploy artifacts ----
    print("\n" + "-" * 78)
    print("STEP 4: deploy artifacts")
    print("-" * 78)
    pred_oof_unb = pred_unb_all.reshape(-1, n_unb).mean(axis=0)
    pred_te_513 = pred_te_per_bag.mean(axis=0)
    deploy_pooled_rae = float(rae(y_unb, pred_oof_unb))
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"   deploy bag-mean (5kf x 5bag) OOF pooled RAE = "
          f"{deploy_pooled_rae:.4f}")
    print(f"   te[unb_idx] in-sample RAE                   = "
          f"{te_unb_in_rae:.4f}")
    print(f"   pred_oof_unb std = {pred_oof_unb.std():.3f} "
          f"(truth_std {y_unb.std():.3f})")
    print(f"   pred_te_513  mean/std = {pred_te_513.mean():.3f}/"
          f"{pred_te_513.std():.3f}")

    # ---- Gate ----
    print("\n" + "-" * 78)
    print("STEP 5: GATE")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   mean_rae = {mean_rae:.4f}")
    print(f"     <{GATE_PROMOTE} -> PROMOTE")
    print(f"     <{GATE_MARGINAL} -> MARGINAL_BEAT")
    print(f"     else            -> FAIL")
    print(f"   -> {verdict}")

    # ---- Save ----
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_umap_lgbm.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "umap20_standardscaler_lgbm_residual",
        "rationale": "non-linear manifold projection vs nb2171 raw 117-col ceiling",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "umap_n_components": UMAP_N_COMP,
        "umap_n_neighbors": UMAP_NEIGH,
        "umap_min_dist": UMAP_MIN_DIST,
        "umap_random_state": UMAP_RND,
        "kf_seeds": KF_SEEDS,
        "bag_seeds": BAG_SEEDS,
        "n_kf": len(KF_SEEDS),
        "n_bag": len(BAG_SEEDS),
        "n_total_fits": len(KF_SEEDS) * len(BAG_SEEDS),
        "resid_folds": RESID_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_kf_bagmean_rae": per_kf_bagmean_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "median_rae": median_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "deploy_pooled_rae": deploy_pooled_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-kf 5-bag-mean = "
          + ", ".join(f"{r:.4f}" for r in per_kf_bagmean_rae))
    print(f"   mean +/- std      = {mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"   delta vs nb2171   = {mean_rae - NB2171_REF:+.4f}")
    print(f"   verdict           = {verdict}")
    print(f"   wall              = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("mean_rae", "std_rae", "min_rae", "max_rae",
              "deploy_pooled_rae", "te_unb_in_sample_rae",
              "delta_vs_nb2171", "verdict"):
        print(f"  {k}: {res.get(k)}")
