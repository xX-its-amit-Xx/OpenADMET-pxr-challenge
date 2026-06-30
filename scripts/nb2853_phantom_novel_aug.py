"""nb2853 -- Phantom novel-scaffold augmentation for K=20 LGBM base.

NEW PARADIGM:
    Cycle-134 paradigm exhaustion + cycle-169 substrate-change rule motivate
    a TRAIN-SIDE augmentation that biases the LGBM toward the F2
    novel-scaffold tail (90% of failure mass per pm06).  We generate phantom
    training rows by interpolating the most DISTANT train pairs (lowest
    pairwise Tanimoto on ECFP4), under the hypothesis that
        x_phantom = 0.5 * (x_i + x_j)   y_phantom = 0.5 * (y_i + y_j)
    populates the off-manifold region the 513 test set lives in (median
    top-1 test->train Tanimoto = 0.52; 90% of failures sit at scaf_train_freq=0
    in pm06).  This is the inverse of Mixup (nb2691): nb2691 used
    uniform pairs with Beta(0.2, 0.2) lam (mostly near-original rows);
    nb2853 hand-picks the 100 most distant pairs and averages with lam = 0.5
    (full midpoint), so each phantom is maximally novel relative to either
    parent.

PROTOCOL:
    1.  Compute Morgan(2048) FPs on the 4139 train compounds.
    2.  Build pairwise Tanimoto on the 4139 train; pick top-100 pairs with
        LOWEST Tanimoto (i, j with i<j, deterministic ordering).
    3.  Featurise train + unblind + test with `combined` (Morgan + RDKit,
        ~2265 cols) and impute via shared col-medians.
    4.  Phantom rows: x_phantom_k = 0.5*(X_tr[i_k] + X_tr[j_k]),
                      y_phantom_k = 0.5*(y_tr[i_k] + y_tr[j_k]).
    5.  5-fold scaffold CV on 253 unblind (kf_seed=1001).  Per fold:
            train_X = stack(X_tr [4139], X_phantom [100], X_unb[tr_idx])
            train_y = concat(y_tr, y_phantom, y_unb[tr_idx])
            LGBM(depth=4, n_est=300, lr=0.03)  -- K=20-style small-tree LGBM
            predict on held-out 1/5 of unblind + on full 513 test.
    6.  Pooled OOF RAE on 253 = mean_rae for gate.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else               -> "FAIL"

Inputs:
    data/raw/pxr-challenge_TRAIN.csv
    data/raw/pxr-challenge_TEST_BLINDED.csv
    data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv

Outputs:
    data/processed/nb2853_summary.json
    data/processed/nb2853_pred_oof.npy    (253,) float32
    data/processed/te_nb2853.npy          (513,) float32
    submissions/nb2853_phantom_novel_aug.csv
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

from pxr.chem import morgan_fp_batch, bemis_murcko
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

TAG = "nb2853"

# -- Phantom-aug params ---------------------------------------------------
N_PHANTOM = 100             # top-100 LOWEST-Tanimoto train pairs
LAM = 0.5                   # midpoint averaging (max-novel phantom)

# -- LGBM hyperparams (K=20-style: depth=4, small trees, MSE) -------------
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_EST = 300
LGBM_LR = 0.03
LGBM_MIN_CHILD = 5
LGBM_REG_LAMBDA = 2.0
LGBM_SEED = 0

# -- Cross-fit params -----------------------------------------------------
KF_SEED = 1001
N_FOLDS = 5

# -- Gates ---------------------------------------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# -- Reference numbers ---------------------------------------------------
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="regression",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_EST,
        learning_rate=LGBM_LR,
        min_child_samples=LGBM_MIN_CHILD,
        reg_lambda=LGBM_REG_LAMBDA,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _find_top_distant_pairs(fp: np.ndarray, n_pairs: int):
    """Find top-`n_pairs` (i, j) pairs with i<j minimising Tanimoto on `fp`.

    fp: (N, 2048) uint8 (Morgan FP).  Streams over rows in blocks to keep
    memory bounded.  We track the n_pairs lowest similarities globally
    with a sorted insertion buffer.
    """
    fp = fp.astype(np.float32)
    n = fp.shape[0]
    fp_sum = fp.sum(axis=1)
    BLOCK = 256
    # max-heap of (-sim, i, j) so we can pop the largest-sim entry once
    # the buffer is full and a smaller-sim pair appears.  Use np arrays for
    # speed (n_pairs is small -- O(100)).
    cur_sims = np.full(n_pairs, np.inf, dtype=np.float32)
    cur_ij = np.zeros((n_pairs, 2), dtype=np.int32)
    cur_max_pos = 0          # index in cur_sims holding the current MAX sim
    cur_max_val = np.inf     # current max sim value
    for s in range(0, n, BLOCK):
        e = min(n, s + BLOCK)
        # only upper triangle: i in [s..e), j in [i+1..n)
        for ii in range(s, e):
            row = fp[ii]
            inter = fp[ii + 1 :] @ row
            denom = fp_sum[ii + 1 :] + fp_sum[ii] - inter
            denom = np.maximum(denom, 1.0)
            sims = (inter / denom).astype(np.float32)
            # All sims < cur_max_val are candidates; quick mask
            cand_mask = sims < cur_max_val
            if not cand_mask.any():
                continue
            cand_local_j = np.where(cand_mask)[0]
            cand_sims = sims[cand_local_j]
            cand_global_j = cand_local_j + (ii + 1)
            # Greedy insertion: for each candidate, if it beats current max,
            # replace the current max slot and recompute the new max.
            for k in range(len(cand_sims)):
                csim = float(cand_sims[k])
                if csim >= cur_max_val:
                    continue
                cur_sims[cur_max_pos] = csim
                cur_ij[cur_max_pos, 0] = ii
                cur_ij[cur_max_pos, 1] = int(cand_global_j[k])
                # update new max
                cur_max_pos = int(np.argmax(cur_sims))
                cur_max_val = float(cur_sims[cur_max_pos])
        if (s // BLOCK) % 4 == 0:
            print(f"      pair-scan progress: {e}/{n} "
                  f"cur_max_sim_in_top{n_pairs}={cur_max_val:.4f}")
    # sort the buffer ascending by sim
    order = np.argsort(cur_sims)
    sims_sorted = cur_sims[order]
    ij_sorted = cur_ij[order]
    return ij_sorted, sims_sorted


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Phantom novel-scaffold augmentation (top-{N_PHANTOM} "
          f"distant train pairs, lam={LAM})")
    print(f"         LGBM(depth={LGBM_MAX_DEPTH}, leaves={LGBM_NUM_LEAVES}, "
          f"n_est={LGBM_N_EST}, lr={LGBM_LR})")
    print(f"         {N_FOLDS}-fold scaffold CV on 253, kf_seed={KF_SEED}")
    print(f"         gates  PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
    print("=" * 78)

    needed = {
        "TRAIN":        DATA_RAW / "pxr-challenge_TRAIN.csv",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for k, p in needed.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"missing raw csv {k}: {p}")

    # ---- Load raw data ----
    train_df = pd.read_csv(needed["TRAIN"])
    test_df = pd.read_csv(needed["TEST_BLINDED"])
    unb_df = pd.read_csv(needed["UNBLINDED"])

    # Train has duplicate SMILES (assay replicates).  Aggregate to median
    # pEC50 per SMILES so we have a clean 4139-ish row table.
    # Use raw rows directly to preserve assay structure but drop NaNs.
    train_df = train_df.dropna(subset=["SMILES", "pEC50"]).reset_index(drop=True)
    smiles_tr_full = train_df["SMILES"].astype(str).tolist()
    y_tr_full = train_df["pEC50"].astype(np.float64).values
    # Median-aggregate by SMILES so the 4139-row interpretation holds
    agg = (
        train_df.groupby("SMILES", as_index=False)
        .agg(pEC50=("pEC50", "median"))
    )
    smiles_tr = agg["SMILES"].astype(str).tolist()
    y_tr = agg["pEC50"].astype(np.float64).values
    n_tr = len(smiles_tr)

    te_names = test_df["Molecule Name"].tolist()
    te_smiles = test_df["SMILES"].astype(str).tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(
        drop=True
    )
    unb_idx_in_te = np.array(
        [name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int
    )
    y_unb = unb_df["pEC50"].astype(np.float64).values
    unb_smiles = unb_df["SMILES"].astype(str).tolist()
    n_unb = len(unb_df)
    n_te = len(test_df)

    print(f"[load] train (median-agg by SMILES) n={n_tr}  "
          f"test_blinded n={n_te}  unblinded n={n_unb}")

    # ---- Scaffolds for CV on the 253 ----
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in 253 = {n_unique_scaf}")

    # ---- Morgan FPs on train for pairwise Tanimoto ----
    print("\n" + "-" * 78)
    print(f"STEP 1: Morgan FP on n_tr={n_tr} train compounds for "
          f"pairwise Tanimoto")
    print("-" * 78)
    t_fp = time.time()
    fp_tr = morgan_fp_batch(smiles_tr).astype(np.uint8)
    print(f"   fp_tr shape={fp_tr.shape}  wall={time.time()-t_fp:.1f}s")

    # ---- Find top-N_PHANTOM lowest-Tanimoto train pairs ----
    print("\n" + "-" * 78)
    print(f"STEP 2: find top-{N_PHANTOM} LOWEST-Tanimoto train pairs "
          f"(most distant)")
    print("-" * 78)
    t_pair = time.time()
    ij_pairs, sim_pairs = _find_top_distant_pairs(fp_tr, N_PHANTOM)
    print(f"   pair-scan wall = {time.time()-t_pair:.1f}s")
    print(f"   min sim in top-{N_PHANTOM} = {float(sim_pairs.min()):.4f}")
    print(f"   max sim in top-{N_PHANTOM} = {float(sim_pairs.max()):.4f}")
    print(f"   mean sim in top-{N_PHANTOM} = {float(sim_pairs.mean()):.4f}")
    print(f"   first 5 pairs (i, j, sim):")
    for k in range(min(5, len(ij_pairs))):
        i, j = int(ij_pairs[k, 0]), int(ij_pairs[k, 1])
        print(f"      ({i:4d}, {j:4d})  sim={float(sim_pairs[k]):.4f}  "
              f"y_i={y_tr[i]:.2f}  y_j={y_tr[j]:.2f}  "
              f"y_phantom={LAM*y_tr[i]+(1-LAM)*y_tr[j]:.2f}")

    # ---- Featurise everything with `combined` (Morgan + RDKit) ----
    print("\n" + "-" * 78)
    print(f"STEP 3: featurise (combined: Morgan + RDKit) and impute")
    print("-" * 78)
    t_feat = time.time()
    X_tr = combined(smiles_tr)
    X_unb = combined(unb_smiles)
    X_te = combined(te_smiles)
    # Phantom features: midpoint of the (i, j) train rows in feature space
    X_phantom = np.zeros((N_PHANTOM, X_tr.shape[1]), dtype=np.float32)
    y_phantom = np.zeros(N_PHANTOM, dtype=np.float64)
    for k in range(N_PHANTOM):
        i, j = int(ij_pairs[k, 0]), int(ij_pairs[k, 1])
        X_phantom[k] = LAM * X_tr[i] + (1.0 - LAM) * X_tr[j]
        y_phantom[k] = LAM * y_tr[i] + (1.0 - LAM) * y_tr[j]
    # Impute with shared median across (train, phantom, unblind, test)
    X_all = np.vstack([X_tr, X_phantom, X_unb, X_te])
    X_all = impute(X_all)
    X_tr = X_all[:n_tr].astype(np.float32)
    X_phantom = X_all[n_tr : n_tr + N_PHANTOM].astype(np.float32)
    X_unb = X_all[n_tr + N_PHANTOM : n_tr + N_PHANTOM + n_unb].astype(
        np.float32
    )
    X_te = X_all[n_tr + N_PHANTOM + n_unb :].astype(np.float32)
    print(f"   X_tr={X_tr.shape}  X_phantom={X_phantom.shape}  "
          f"X_unb={X_unb.shape}  X_te={X_te.shape}  "
          f"wall={time.time()-t_feat:.1f}s")
    print(f"   y_phantom stats: mean={y_phantom.mean():.3f}  "
          f"std={y_phantom.std():.3f}  "
          f"min={y_phantom.min():.3f}  max={y_phantom.max():.3f}")
    print(f"   y_tr      stats: mean={y_tr.mean():.3f}  "
          f"std={y_tr.std():.3f}")

    # ---- 5-fold scaffold CV on 253 ----
    print("\n" + "-" * 78)
    print(f"STEP 4: 5-fold scaffold CV on n_unb={n_unb}  kf_seed={KF_SEED}")
    print(f"        per-fold train = (4139 train + {N_PHANTOM} phantom + "
          f"4/5 unblind)")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED
    )
    oof_unb = np.full(n_unb, np.nan, dtype=np.float64)
    te_per_fold = np.zeros((N_FOLDS, n_te), dtype=np.float64)
    fold_raes: list[float] = []
    fold_records: list[dict] = []
    for fold, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        # train block: 4139 train + 100 phantom + 4/5 unblind
        X_fold = np.vstack(
            [X_tr, X_phantom, X_unb[tr_loc]]
        ).astype(np.float32)
        y_fold = np.concatenate(
            [y_tr, y_phantom, y_unb[tr_loc]]
        ).astype(np.float64)

        mdl = lgb.LGBMRegressor(**_lgbm_params(LGBM_SEED))
        mdl.fit(X_fold, y_fold)

        oof_unb[va_loc] = mdl.predict(X_unb[va_loc])
        te_per_fold[fold] = mdl.predict(X_te)
        r_va = float(rae(y_unb[va_loc], oof_unb[va_loc]))
        fold_raes.append(r_va)
        fold_records.append({
            "fold": int(fold),
            "n_tr_block": int(X_fold.shape[0]),
            "n_va": int(len(va_loc)),
            "rae_fold": r_va,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   fold {fold}: n_tr_block={X_fold.shape[0]}  "
              f"n_va={len(va_loc):3d}  RAE={r_va:.4f}  "
              f"wall={time.time()-ts:.1f}s")
    if np.isnan(oof_unb).any():
        raise RuntimeError("scaffold split did not cover all rows")

    rae_oof = float(rae(y_unb, oof_unb))
    print(f"\n   pooled OOF RAE       = {rae_oof:.4f}  "
          f"(per-fold {min(fold_raes):.4f} .. {max(fold_raes):.4f})")
    print(f"   delta vs nb2171      = {rae_oof - NB2171_REF:+.4f}")
    print(f"   delta vs chemprop_aux= {rae_oof - CHEMPROP_AUX_REF:+.4f}")

    # ---- Deploy: train on (4139 + 100 phantom + 253 unblind) -> predict 513 ----
    print("\n" + "-" * 78)
    print("STEP 5: deploy refit -- train on (4139 + 100 phantom + 253 unb)")
    print("-" * 78)
    X_deploy = np.vstack([X_tr, X_phantom, X_unb]).astype(np.float32)
    y_deploy = np.concatenate([y_tr, y_phantom, y_unb]).astype(np.float64)
    mdl_deploy = lgb.LGBMRegressor(**_lgbm_params(LGBM_SEED))
    mdl_deploy.fit(X_deploy, y_deploy)
    pred_te_513 = mdl_deploy.predict(X_te).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx_in_te]))
    print(f"   te_513 mean/std        = "
          f"{float(pred_te_513.mean()):.3f}/{float(pred_te_513.std()):.3f}")
    print(f"   te[unb_idx] in-sample  = {te_unb_in_rae:.4f}")

    # Also report fold-bag mean of te
    te_cv_bag = te_per_fold.mean(axis=0).astype(np.float32)
    te_cv_bag_unb_rae = float(rae(y_unb, te_cv_bag[unb_idx_in_te]))
    print(f"   te fold-bag in-sample  = {te_cv_bag_unb_rae:.4f}")

    # ---- Gate ----
    mean_rae = rae_oof
    print("\n" + "-" * 78)
    print("STEP 6: GATE")
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

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_phantom_novel_aug.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513,
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "phantom_novel_scaffold_train_augmentation_lgbm",
        "paradigm": ("interpolate_max_distant_train_pairs_lam_0p5_midpoint"
                     "_then_full_combined_lgbm"),
        "rationale": ("Phantom rows from top-100 LOWEST-Tanimoto train "
                      "pairs populate the off-manifold region the 513 test "
                      "set (median top-1 sim 0.52) and 90%-novel-scaffold "
                      "failure tail (pm06) live in; midpoint lam=0.5 "
                      "maximises novelty per phantom vs Mixup's near-edge "
                      "Beta(0.2,0.2)."),
        "anchor_base": None,
        "anchor_pre_unblind": None,
        "feature_set": "combined_morgan_rdkit_2265",
        "feat_dim": int(X_tr.shape[1]),
        "n_phantom": N_PHANTOM,
        "lam_phantom": LAM,
        "phantom_pair_sim_min": float(sim_pairs.min()),
        "phantom_pair_sim_max": float(sim_pairs.max()),
        "phantom_pair_sim_mean": float(sim_pairs.mean()),
        "phantom_y_mean": float(y_phantom.mean()),
        "phantom_y_std": float(y_phantom.std()),
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_num_leaves": LGBM_NUM_LEAVES,
        "lgbm_n_estimators": LGBM_N_EST,
        "lgbm_learning_rate": LGBM_LR,
        "lgbm_min_child_samples": LGBM_MIN_CHILD,
        "lgbm_reg_lambda": LGBM_REG_LAMBDA,
        "lgbm_seed": LGBM_SEED,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_tr": int(n_tr),
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "n_unique_scaffolds_unb": int(n_unique_scaf),
        "fold_records": fold_records,
        "fold_raes": fold_raes,
        "pooled_oof_rae": rae_oof,
        "mean_rae": mean_rae,
        "deploy_te_unb_in_rae": te_unb_in_rae,
        "te_cv_bag_unb_rae": te_cv_bag_unb_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "pred_oof_mean": float(oof_unb.mean()),
        "pred_oof_std": float(oof_unb.std()),
        "truth_mean": float(y_unb.mean()),
        "truth_std": float(y_unb.std()),
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "delta_vs_chemprop_aux": mean_rae - CHEMPROP_AUX_REF,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled OOF RAE      = {rae_oof:.4f}")
    print(f"   per-fold RAEs       = "
          + ", ".join(f"{r:.4f}" for r in fold_raes))
    print(f"   delta vs nb2171     = {mean_rae - NB2171_REF:+.4f}")
    print(f"   deploy te[unb_idx]  = {te_unb_in_rae:.4f}")
    print(f"   te_cv_bag[unb_idx]  = {te_cv_bag_unb_rae:.4f}")
    print(f"   verdict             = {verdict}")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_oof_rae",
        "deploy_te_unb_in_rae",
        "delta_vs_nb2171",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
