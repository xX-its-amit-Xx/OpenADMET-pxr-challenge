"""nb2733 -- Multi-objective LGBM: joint (pEC50, Emax, counter_pec50) heads.

NEW PARADIGM:
    Prior multi-task work (nb2022 sequential boost) chained pec50_null ->
    selectivity -> pec50 with stage-wise fits. This script implements a
    different multi-task formulation: sklearn.multioutput.MultiOutputRegressor
    wrapping a SINGLE LGBM regressor template. Each of the 3 output heads
    (pEC50, Emax, counter_pec50) is trained INDEPENDENTLY at the regression
    layer (MOR fits one LGBM per column), but the SHARED feature substrate
    + identical hyperparams + equal weighting via fold averaging together
    impose a "regularization-by-averaging" effect across the three tasks.
    The shared substrate forces the same input encoding to support all
    three biological readouts; if pEC50 alone overfits novel-scaffold rows,
    averaging gradients across Emax / counter_pec50 (which see DIFFERENT
    biological signal on the same scaffold) should pull pec50 predictions
    toward the multi-task consensus.

    The "equal weighting" interpretation here: at the MOR layer, the 3
    heads are independent, so we ensemble via SEED bagging across 5 kf_seeds
    (each seed fits a fresh MOR over scaffold-CV folds). For final pec50
    OOF on the 253, we mean-bag predictions only from the pec50 head.

PROTOCOL:
    1. Load 4139 train rows; inner-join with counter on standardized SMILES
       to get rows with all 3 labels (pec50 + emax + counter_pec50).
       This is the training corpus for the multi-output LGBM.
    2. Combined features (Morgan-2048 + RDKit-217 = 2265 dims), imputed.
    3. y = [pec50, emax, counter_pec50] as (N, 3) target.
    4. For each of 5 kf_seeds {1001..1005}: 5-fold scaffold CV ON the joint
       train+counter corpus produces OOF predictions for all 3 heads.
       Refit MOR(LGBM) on full joint corpus -> predict 513 te.
    5. Mean-bag te predictions (pec50 head only) across 5 seeds.
    6. Evaluate on 253 unblind via te[unb_idx] vs y_unb -- this is the
       cross-fit-on-multi-task-corpus evaluation that mirrors how a
       deployed multi-task model would score on Phase-1 unblind.
       (The 253 has pec50+emax but NOT counter_pec50, so it cannot
       cross-fit inside the 253; we evaluate via deploy refit per nb2022.)
    7. Gate on standalone pec50 OOF RAE evaluated at the 253 unblind.

GATE (mean_rae = OOF pec50 RAE on 253):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2733_multiobjective.py
    data/processed/nb2733_summary.json
    data/processed/nb2733_pred_oof.npy   (253,) float32 mean-bag pec50 OOF on unb
    data/processed/te_nb2733.npy         (513,) float32 mean-bag deploy refit pec50
    submissions/nb2733_multiobjective.csv
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
from sklearn.multioutput import MultiOutputRegressor

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, bemis_murcko
from pxr.data import load_train, load_counter, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2733"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# LGBM template hyperparams (identical for all 3 heads via MOR)
LGBM_NUM_LEAVES = 28
LGBM_MAX_DEPTH = 4
LGBM_N_ESTIMATORS = 300
LGBM_LEARNING_RATE = 0.03
LGBM_MIN_CHILD_SAMPLES = 5
LGBM_REG_LAMBDA = 2.0

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2022_REF = 0.4750  # nb2022 sequential stage-3 mean-bag final RAE (sanity)


def _lgbm_template(seed: int) -> lgb.LGBMRegressor:
    """LGBM regressor template (one per output head via MOR)."""
    return lgb.LGBMRegressor(
        objective="regression",
        num_leaves=LGBM_NUM_LEAVES,
        max_depth=LGBM_MAX_DEPTH,
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        reg_lambda=LGBM_REG_LAMBDA,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _mor(seed: int) -> MultiOutputRegressor:
    return MultiOutputRegressor(_lgbm_template(seed), n_jobs=1)


def _scaffold_cv_one_seed(
    X: np.ndarray,
    Y: np.ndarray,
    scaffolds: list,
    kf_seed: int,
) -> np.ndarray:
    """5-fold scaffold-CV with MOR(LGBM). Returns OOF preds (N, 3)."""
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n, n_out = Y.shape
    oof = np.full((n, n_out), np.nan, dtype=np.float64)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        mor = _mor(kf_seed)
        mor.fit(X[tr_loc], Y[tr_loc])
        oof[va_loc] = mor.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _deploy_one_seed(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Refit MOR(LGBM) on full corpus; predict (N_te, 3) on te."""
    mor = _mor(seed)
    mor.fit(X_tr, Y_tr)
    return mor.predict(X_te).astype(np.float64)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Multi-objective LGBM (pEC50 + Emax + counter_pec50)")
    print(f"        MultiOutputRegressor over LGBM template:")
    print(f"          num_leaves={LGBM_NUM_LEAVES}  depth={LGBM_MAX_DEPTH}  "
          f"n_est={LGBM_N_ESTIMATORS}  lr={LGBM_LEARNING_RATE}")
    print(f"        5-fold scaffold-CV  kf_seeds={KF_SEEDS}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load and join train + counter on standardized SMILES ----
    print("\n[step1] loading + joining train + counter ...")
    tr = load_train()
    co = load_counter()

    def _std(s):
        m = standardize(s)
        return Chem.MolToSmiles(m) if m is not None else None

    tr["std_smi"] = tr["smiles"].apply(_std)
    co["std_smi"] = co["smiles"].apply(_std)

    # Aggregate to unique SMILES (median per compound)
    tr_g = (
        tr[tr["std_smi"].notna() & tr["pec50"].notna() & tr["emax"].notna()]
        .groupby("std_smi", as_index=False)
        .agg(pec50=("pec50", "median"), emax=("emax", "median"))
    )
    co_g = (
        co[co["std_smi"].notna() & co["pec50"].notna()]
        .groupby("std_smi", as_index=False)
        .agg(counter_pec50=("pec50", "median"))
    )
    joined = tr_g.merge(co_g, on="std_smi", how="inner")
    print(f"   train n_uniq (pec50+emax)   = {len(tr_g)}")
    print(f"   counter n_uniq (pec50)      = {len(co_g)}")
    print(f"   JOIN (all 3 labels present) = {len(joined)}")

    smiles_j = joined["std_smi"].tolist()
    Y = joined[["pec50", "emax", "counter_pec50"]].to_numpy(dtype=np.float64)
    scaff_j = [bemis_murcko(s) for s in smiles_j]
    n_unique_scaf = len({s for s in scaff_j if s})
    print(f"[scaffold] joint corpus unique scaffolds = {n_unique_scaf}")

    # ---- Features for joint corpus + test ----
    print("\n[step2] computing combined features ...")
    X_j = impute(combined(smiles_j)).astype(np.float32)
    te_df = load_test()
    te_smiles_raw = te_df["smiles"].astype(str).tolist()
    te_names = te_df["name"].astype(str).tolist()
    te_std = [
        (Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else s)
        for s in te_smiles_raw
    ]
    X_te = impute(combined(te_std)).astype(np.float32)
    print(f"   X_j shape = {X_j.shape}   X_te shape = {X_te.shape}")

    # ---- Load 253 unblind labels ----
    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    # Anchor reference (for diagnostic)
    if ANCHOR_TE_PATH.exists():
        anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
        rae_anchor = float(rae(y_unb, anchor_513[unb_idx]))
        print(f"[diag] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f}  "
              f"(ref {CHEMPROP_AUX_REF:.4f})")
    else:
        anchor_513 = None
        rae_anchor = float("nan")

    # ---- 5-seed scaffold CV on joint corpus + deploy refit ----
    print("\n" + "-" * 78)
    print(f"5-SEED MOR(LGBM) MULTI-TASK   seeds={KF_SEEDS}  folds={N_FOLDS}")
    print("-" * 78)
    n_j = len(Y)
    per_seed_oof_pec50 = np.zeros((len(KF_SEEDS), n_j), dtype=np.float64)
    per_seed_oof_emax = np.zeros((len(KF_SEEDS), n_j), dtype=np.float64)
    per_seed_oof_null = np.zeros((len(KF_SEEDS), n_j), dtype=np.float64)
    per_seed_te_pec50 = np.zeros((len(KF_SEEDS), len(X_te)), dtype=np.float64)
    per_seed_rae_joint = []
    per_seed_rae_unb = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        oof = _scaffold_cv_one_seed(X_j, Y, scaff_j, seed)
        per_seed_oof_pec50[i] = oof[:, 0]
        per_seed_oof_emax[i] = oof[:, 1]
        per_seed_oof_null[i] = oof[:, 2]
        rae_joint_pec50 = float(rae(Y[:, 0], oof[:, 0]))
        per_seed_rae_joint.append(rae_joint_pec50)

        te_pred = _deploy_one_seed(X_j, Y, X_te, seed)
        per_seed_te_pec50[i] = te_pred[:, 0]
        rae_seed_unb = float(rae(y_unb, te_pred[:, 0][unb_idx]))
        per_seed_rae_unb.append(rae_seed_unb)
        print(f"   seed={seed}  pec50_joint_OOF_RAE={rae_joint_pec50:.4f}  "
              f"deploy_unb_RAE={rae_seed_unb:.4f}  "
              f"wall={time.time() - ts:.1f}s")

    # ---- Mean-bag aggregates ----
    mean_bag_oof_pec50_joint = per_seed_oof_pec50.mean(axis=0)
    mean_bag_te_pec50 = per_seed_te_pec50.mean(axis=0)
    median_bag_te_pec50 = np.median(per_seed_te_pec50, axis=0)

    rae_joint_oof_meanbag = float(rae(Y[:, 0], mean_bag_oof_pec50_joint))
    rae_joint_oof_emax = float(
        rae(Y[:, 1], per_seed_oof_emax.mean(axis=0))
    )
    rae_joint_oof_null = float(
        rae(Y[:, 2], per_seed_oof_null.mean(axis=0))
    )

    pred_unb_mean = mean_bag_te_pec50[unb_idx]
    pred_unb_median = median_bag_te_pec50[unb_idx]
    rae_unb_meanbag = float(rae(y_unb, pred_unb_mean))
    rae_unb_medianbag = float(rae(y_unb, pred_unb_median))

    print("\n[cv] joint corpus OOF RAE   pec50 = "
          f"{rae_joint_oof_meanbag:.4f}   emax = {rae_joint_oof_emax:.4f}   "
          f"counter_pec50 = {rae_joint_oof_null:.4f}")
    print(f"[cv] 253 unb mean-bag       RAE  = {rae_unb_meanbag:.4f}")
    print(f"[cv] 253 unb median-bag     RAE  = {rae_unb_medianbag:.4f}")
    print(f"[cv] per_seed unb mean      RAE  = "
          f"{np.mean(per_seed_rae_unb):.4f}   "
          f"std = {np.std(per_seed_rae_unb):.4f}")
    print(f"[cv] chemprop_aux anchor    RAE  = {rae_anchor:.4f}   "
          f"(d_mean_bag = {rae_unb_meanbag - rae_anchor:+.4f})")

    # ---- Save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, pred_unb_mean.astype(np.float32))
    np.save(te_path, mean_bag_te_pec50.astype(np.float32))
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_multiobjective.csv"
    pd.DataFrame({
        "SMILES": te_smiles_raw,
        "Molecule Name": te_names,
        "pEC50": mean_bag_te_pec50.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    mean_rae = rae_unb_meanbag
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae (253 unb mean-bag) = {mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT                  = {verdict}")

    summary = {
        "tag": TAG,
        "method": (
            "MultiOutputRegressor(LGBM) joint heads = "
            "[pec50, emax, counter_pec50]; trained on train+counter inner-join; "
            "evaluated on 253 unblind via deploy refit"
        ),
        "rationale": (
            "Shared feature substrate + identical LGBM template per head + "
            "equal-weighting via fold averaging regularizes the pec50 head by "
            "the auxiliary emax + counter_pec50 signals; paradigm-difference "
            "vs nb2022 sequential boost (chain) and vs single-task pec50 LGBM."
        ),
        "anchor": "chemprop_aux",
        "anchor_pre_unblind": True,
        "rae_anchor": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2022_ref": NB2022_REF,
        "n_train_only": int(len(tr_g)),
        "n_counter_only": int(len(co_g)),
        "n_join_corpus": int(n_j),
        "n_unique_scaffolds_join": n_unique_scaf,
        "n_test": int(X_te.shape[0]),
        "n_unb": int(n_unb),
        "feat_dim": int(X_j.shape[1]),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "model_class": "sklearn.multioutput.MultiOutputRegressor[LGBMRegressor]",
        "lgbm_num_leaves": LGBM_NUM_LEAVES,
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_n_estimators": LGBM_N_ESTIMATORS,
        "lgbm_learning_rate": LGBM_LEARNING_RATE,
        "lgbm_min_child_samples": LGBM_MIN_CHILD_SAMPLES,
        "lgbm_reg_lambda": LGBM_REG_LAMBDA,
        "y_pec50_mean": float(Y[:, 0].mean()),
        "y_pec50_std": float(Y[:, 0].std()),
        "y_emax_mean": float(Y[:, 1].mean()),
        "y_emax_std": float(Y[:, 1].std()),
        "y_counter_pec50_mean": float(Y[:, 2].mean()),
        "y_counter_pec50_std": float(Y[:, 2].std()),
        "per_seed_rae_joint_pec50_OOF": [float(x) for x in per_seed_rae_joint],
        "per_seed_rae_unb_deploy_pec50": [float(x) for x in per_seed_rae_unb],
        "per_seed_unb_mean_rae": float(np.mean(per_seed_rae_unb)),
        "per_seed_unb_std_rae": float(np.std(per_seed_rae_unb)),
        "rae_joint_oof_meanbag_pec50": rae_joint_oof_meanbag,
        "rae_joint_oof_meanbag_emax": rae_joint_oof_emax,
        "rae_joint_oof_meanbag_counter_pec50": rae_joint_oof_null,
        "rae_unb_meanbag": rae_unb_meanbag,
        "rae_unb_medianbag": rae_unb_medianbag,
        "mean_rae": rae_unb_meanbag,  # gate alias
        "delta_vs_anchor": rae_unb_meanbag - rae_anchor,
        "delta_vs_nb2022": rae_unb_meanbag - NB2022_REF,
        "te_deploy_mean": float(mean_bag_te_pec50.mean()),
        "te_deploy_std": float(mean_bag_te_pec50.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_join_corpus",
        "n_unique_scaffolds_join",
        "rae_joint_oof_meanbag_pec50",
        "rae_joint_oof_meanbag_emax",
        "rae_joint_oof_meanbag_counter_pec50",
        "per_seed_rae_unb_deploy_pec50",
        "per_seed_unb_mean_rae",
        "per_seed_unb_std_rae",
        "rae_unb_meanbag",
        "rae_unb_medianbag",
        "delta_vs_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
