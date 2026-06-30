"""nb2741 -- Per-test-row LOCAL LGBM fit on top-K Tanimoto neighbors.

NEW PARADIGM (vs global LGBM):
    Every prior LGBM script in the ladder fits ONE model on ALL training
    rows, then predicts each test row with the same global decision tree
    ensemble. Tree splits get chosen to minimize global residual variance
    and are necessarily compromises across diverse scaffolds. For a
    novel-scaffold test row, the global splits are dominated by far-away
    training chemistry and may not align with the local pEC50 manifold.

    This script flips that around: for EACH test row, find its top-100
    Tanimoto-nearest train rows (on Morgan ECFP4 FP-2048), then fit a
    SMALL LGBM (max_depth=4, n_est=100, lr=0.05) using ONLY those 100 rows
    on K=20 features. Each test row gets its OWN local model. The tree
    splits get chosen to fit the LOCAL residual surface around that one
    test row, which should better track the pEC50 landscape in the
    immediate chemical neighborhood. Cost: 513 LGBM fits per deploy
    + 253 LGBM fits per CV fold = ~766 small LGBM fits.

    Hypothesis: local-LGBM should outperform global on novel-scaffold rows
    where the global model averages across too-diverse chemistry, and
    should at least match global on easy rows where the local 100 NN are
    well-represented in the global fit. Net effect on RAE depends on
    whether 100 rows are enough to fit a useful local tree (LGBM is
    designed to be data-hungry; 100 rows + 20 features is the regime
    where overfitting starts to matter, hence max_depth=4 and only
    n_est=100).

PROTOCOL:
    1. Load 4139 train + 253 unblind + 513 test SMILES.
    2. Compute Morgan FP-2048 (ECFP4) for all (Tanimoto on these).
    3. Load X_117 substrate (pyramid cache) + slice K=20 RFE-surviving
       columns from nb2240 summary -> shared K=20 feature substrate
       for train (4139), unb (253), te (513).
    4. Cross-fit on 253 (5-fold scaffold CV, kf_seed=1001):
         per fold, pool = 4139 train + fold-train portion of 253.
         For each val row: find top-100 Tanimoto NN in pool;
           fit LGBM(max_depth=4, n_est=100, lr=0.05) on those 100
           rows' K=20 features -> predict val row.
    5. Deploy: pool = 4139 train + ALL 253 unblind (with their labels);
       for each of 513 test rows: find top-100 NN in pool;
       fit LGBM on those 100 -> predict the test row.
    6. Evaluate cross-fit RAE on 253; emit te(513) for deploy.

GATE (mean_rae = cross-fit RAE on 253):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2741_local_knn_lgbm.py
    data/processed/nb2741_summary.json
    data/processed/nb2741_pred_oof.npy   (253,) float32 cross-fit OOF
    data/processed/te_nb2741.npy         (513,) float32 deploy
    submissions/nb2741_local_knn_lgbm.csv
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
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2741"

X117_TRAIN_PATH = DATA_PROCESSED / "pyramid" / "X_117_train.npy"
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

K_NN = 100              # top-K nearest neighbors per test row
N_FOLDS = 5
KF_SEED = 1001

# Local LGBM hyperparams (small, data-cheap)
LGBM_MAX_DEPTH = 4
LGBM_N_EST = 100
LGBM_LR = 0.05

# Gate thresholds (cross-fit mean RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2103_K28_REF = 0.4737  # global LGBM K=28 baseline (sanity reference)


def _lgbm_params() -> dict:
    return dict(
        objective="regression",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=15,             # bounded by max_depth=4
        n_estimators=LGBM_N_EST,
        learning_rate=LGBM_LR,
        min_child_samples=2,       # local: only 100 rows, small leaves OK
        reg_lambda=1.0,
        random_state=KF_SEED,
        n_jobs=1,                  # 1 thread per local fit (avoid contention)
        verbosity=-1,
    )


def _tanimoto_pairwise(fp_q: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto similarity (n_q, n_pool) on uint8 Morgan bit-FPs."""
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    inter = a @ b.T
    denom = a_sum[:, None] + b_sum[None, :] - inter
    denom = np.maximum(denom, 1.0)
    return (inter / denom).astype(np.float32)


def _local_lgbm_predict(
    sim_row: np.ndarray,
    feats_pool: np.ndarray,
    y_pool: np.ndarray,
    feat_query: np.ndarray,
    k: int,
) -> tuple[float, int, float, float]:
    """For a single query row: pick top-k pool by Tanimoto, fit local LGBM,
    predict the query.  Returns (pred, n_used, top1_sim, mean_sim_top_k)."""
    n_pool = sim_row.shape[0]
    k_eff = min(k, n_pool)
    # top-k indices by similarity
    if k_eff >= n_pool:
        idx_top = np.arange(n_pool)
    else:
        part = np.argpartition(-sim_row, kth=k_eff - 1)[:k_eff]
        # sort selected by sim descending for diagnostics
        idx_top = part[np.argsort(-sim_row[part])]
    sim_top = sim_row[idx_top]
    X_local = feats_pool[idx_top]
    y_local = y_pool[idx_top]
    mdl = lgb.LGBMRegressor(**_lgbm_params())
    mdl.fit(X_local, y_local)
    pred = float(mdl.predict(feat_query[None, :])[0])
    return pred, int(k_eff), float(sim_top[0]), float(sim_top.mean())


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-test-row LOCAL LGBM on top-{K_NN} Tanimoto neighbors")
    print(f"          K=20 features sliced from X_117; FP-2048 Tanimoto NN")
    print(f"          local LGBM: max_depth={LGBM_MAX_DEPTH}  "
          f"n_est={LGBM_N_EST}  lr={LGBM_LR}")
    print(f"          {N_FOLDS}-fold scaffold-CV on 253, kf_seed={KF_SEED}")
    print(f"          GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- 1. Load truth + indices ----
    tr = load_train()
    te = load_test()
    tr = tr[tr["pec50"].notna() & tr["smiles"].notna()].reset_index(drop=True)
    n_train = len(tr)
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )

    unb_idx = np.load(UNBLIND_IDX).astype(int)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_train={n_train}  n_unb={n_unb}  n_test={n_test}")

    # ---- 2. Morgan FP-2048 for train / unb / test ----
    print("[fp] standardize + morgan_fp_batch ...")
    tr_smi_std = []
    tr_keep_mask = np.zeros(n_train, dtype=bool)
    for i, s in enumerate(tr["smiles"].tolist()):
        m = standardize(s)
        if m is not None:
            tr_smi_std.append(Chem.MolToSmiles(m))
            tr_keep_mask[i] = True
        else:
            tr_smi_std.append("")  # placeholder, will drop
    tr_keep_idx = np.where(tr_keep_mask)[0]
    tr_smi_std = [tr_smi_std[i] for i in tr_keep_idx]
    tr_y = tr.loc[tr_keep_mask, "pec50"].to_numpy(dtype=np.float32)
    n_train_kept = len(tr_smi_std)
    print(f"   train kept after standardize: {n_train_kept}/{n_train}")

    te_smi_std = []
    for s in test_smiles:
        m = standardize(s)
        te_smi_std.append(Chem.MolToSmiles(m) if m is not None else "")

    fp_tr = morgan_fp_batch(tr_smi_std).astype(np.uint8)
    fp_te = morgan_fp_batch(te_smi_std).astype(np.uint8)
    fp_unb = fp_te[unb_idx]
    print(f"   fp_tr={fp_tr.shape}  fp_te={fp_te.shape}  fp_unb={fp_unb.shape}")

    # ---- 3. Load X_117 + slice K=20 substrate ----
    if not (X117_TRAIN_PATH.exists() and X117_UNB_PATH.exists()
            and X117_TE_PATH.exists()):
        raise FileNotFoundError("X_117 cache missing in data/processed/pyramid/")
    X117_train = np.load(X117_TRAIN_PATH).astype(np.float32)
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_train.shape != (n_train, 117):
        raise ValueError(
            f"X117_train shape {X117_train.shape} expected ({n_train},117)"
        )
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape}")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape}")
    # Filter to kept-train rows (align with fp_tr / tr_y)
    X117_train = X117_train[tr_keep_idx]
    # Sanitize NaN/Inf
    X117_train = np.where(np.isfinite(X117_train), X117_train, 0.0).astype(np.float32)
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)
    print(f"[feat] X117_train={X117_train.shape}  "
          f"X117_unb={X117_unb.shape}  X117_te={X117_te.shape}")

    # K=20 indices from nb2240
    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"K=20 expected, got {len(k20_idx)}"
    X_tr = X117_train[:, k20_idx]
    X_unb = X117_unb[:, k20_idx]
    X_te_full = X117_te[:, k20_idx]
    print(f"[feat] K=20 sliced: X_tr={X_tr.shape}  "
          f"X_unb={X_unb.shape}  X_te={X_te_full.shape}")

    # ---- 4. Cross-fit on 253 ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    pred_oof = np.full(n_unb, np.nan, dtype=np.float64)
    top1_sim_oof = np.zeros(n_unb, dtype=np.float32)
    mean_sim_oof = np.zeros(n_unb, dtype=np.float32)

    print(f"\n[cf] scaffold-CV  folds={N_FOLDS}  seed={KF_SEED}  K={K_NN}")
    print("-" * 78)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # pool = 4139 train + fold-train portion of 253
        fp_pool = np.concatenate([fp_tr, fp_unb[tr_loc]], axis=0)
        X_pool = np.concatenate([X_tr, X_unb[tr_loc]], axis=0)
        y_pool = np.concatenate(
            [tr_y, y_unb[tr_loc].astype(np.float32)], axis=0
        )
        sim_va = _tanimoto_pairwise(fp_unb[va_loc], fp_pool)
        ts_fold = time.time()
        for j, row in enumerate(va_loc):
            pred, n_used, top1, mean_k = _local_lgbm_predict(
                sim_va[j], X_pool, y_pool, X_unb[row], K_NN,
            )
            pred_oof[row] = pred
            top1_sim_oof[row] = top1
            mean_sim_oof[row] = mean_k
        rae_fold = float(rae(y_unb[va_loc], pred_oof[va_loc]))
        print(f"   fold {fold_i}: n_va={len(va_loc):3d}  "
              f"pool={len(y_pool)}  RAE={rae_fold:.4f}  "
              f"top1_sim_med={np.median(top1_sim_oof[va_loc]):.3f}  "
              f"wall={time.time()-ts_fold:.1f}s")

    if np.isnan(pred_oof).any():
        raise RuntimeError("cross-fit OOF has NaN")

    rae_cf = float(rae(y_unb, pred_oof))
    print(f"\n[cf] cross-fit RAE on 253 = {rae_cf:.4f}")
    print(f"     vs chemprop_aux ref      = {CHEMPROP_AUX_REF:.4f}  "
          f"(d {rae_cf - CHEMPROP_AUX_REF:+.4f})")
    print(f"     vs nb2103 K=28 ref       = {NB2103_K28_REF:.4f}  "
          f"(d {rae_cf - NB2103_K28_REF:+.4f})")
    print(f"[cf] top1_sim: med={np.median(top1_sim_oof):.3f}  "
          f"mean={top1_sim_oof.mean():.3f}  "
          f"min={top1_sim_oof.min():.3f}")
    print(f"[cf] mean_sim_top{K_NN}: med={np.median(mean_sim_oof):.3f}  "
          f"mean={mean_sim_oof.mean():.3f}")

    # ---- 5. Deploy te(513): pool = 4139 train + 253 unblind ----
    print("\n[deploy] pool = 4139 train + 253 unblind; predict 513 te")
    fp_pool_full = np.concatenate([fp_tr, fp_unb], axis=0)
    X_pool_full = np.concatenate([X_tr, X_unb], axis=0)
    y_pool_full = np.concatenate(
        [tr_y, y_unb.astype(np.float32)], axis=0
    )
    sim_te = _tanimoto_pairwise(fp_te, fp_pool_full)
    te_deploy = np.zeros(n_test, dtype=np.float32)
    top1_sim_te = np.zeros(n_test, dtype=np.float32)
    mean_sim_te = np.zeros(n_test, dtype=np.float32)
    ts = time.time()
    for j in range(n_test):
        pred, _, top1, mean_k = _local_lgbm_predict(
            sim_te[j], X_pool_full, y_pool_full, X_te_full[j], K_NN,
        )
        te_deploy[j] = pred
        top1_sim_te[j] = top1
        mean_sim_te[j] = mean_k
        if (j + 1) % 100 == 0:
            print(f"   te {j+1}/{n_test}  wall={time.time()-ts:.1f}s")

    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}")
    print(f"[deploy] te top1_sim med={np.median(top1_sim_te):.3f}  "
          f"mean={top1_sim_te.mean():.3f}")

    # ---- 6. Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof.astype(np.float32))
    np.save(te_path, te_deploy.astype(np.float32))
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_local_knn_lgbm.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if rae_cf < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_cf < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   cross_fit_rae       = {rae_cf:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {rae_cf < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = {rae_cf < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "per_test_row_local_LGBM_topK_tanimoto_neighbors",
        "rationale": (
            "Per-test-row LGBM fit on top-K=100 Tanimoto NN; K=20 features; "
            "tests whether local trees beat global trees in novel-scaffold tail"
        ),
        "k_nn": K_NN,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_n_estimators": LGBM_N_EST,
        "lgbm_learning_rate": LGBM_LR,
        "lgbm_params": _lgbm_params(),
        "feat_dim_local": 20,
        "fp_dim": 2048,
        "n_train_kept": int(n_train_kept),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "k20_idx_source": str(NB2240_SUMMARY),
        "k20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "k20_surviving_names": k20_names,
        "cross_fit_rae": rae_cf,
        "mean_rae": rae_cf,  # alias for gate consumers
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_ref": NB2103_K28_REF,
        "delta_vs_chemprop_aux": rae_cf - CHEMPROP_AUX_REF,
        "delta_vs_nb2103_K28": rae_cf - NB2103_K28_REF,
        "oof_top1_sim_median": float(np.median(top1_sim_oof)),
        "oof_top1_sim_mean": float(top1_sim_oof.mean()),
        "oof_top1_sim_min": float(top1_sim_oof.min()),
        "oof_mean_sim_topK_median": float(np.median(mean_sim_oof)),
        "te_top1_sim_median": float(np.median(top1_sim_te)),
        "te_top1_sim_mean": float(top1_sim_te.mean()),
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "anchor": "standalone",
        "anchor_pre_unblind": True,  # train+test only, no POST anchor dependency
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_train_kept", "n_unb", "n_test",
        "cross_fit_rae", "delta_vs_chemprop_aux", "delta_vs_nb2103_K28",
        "oof_top1_sim_median", "oof_mean_sim_topK_median",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
