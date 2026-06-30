"""nb2804 -- Pure sklearn GroupKFold by scaffold on K=20 substrate.

NEW PARADIGM (CV-protocol axis):
    `pxr.eval.scaffold_kfold_indices` does a *largest-first greedy* assignment
    of scaffold groups to folds in order to minimise fold-size variance.
    That's the "scaffold-CV" protocol used by ~all PRIMARY candidates,
    including nb2103/nb2700/nb2171.  It is scaffold-disjoint, but the
    greedy fold-balancing trades a controlled amount of group-leakage
    information at fold boundaries (large families may be split into
    pieces of fold-uneven sizes that share other scaffolds, etc.).

    sklearn.model_selection.GroupKFold is STRICTER in one specific
    sense: it deterministically partitions the unique groups into
    n_splits buckets with no shuffling and no fold-size balancing
    knob -- the only guarantee is that no group spans two folds.
    Fold sizes are whatever they fall out to be (rounding effects
    on the unique-group count).

    The two protocols disagree on ~which~ scaffold lands in which
    fold, even at fixed n_splits=5.  If the K=20 LGBM residual model
    is sensitive to the train-vs-val scaffold partition (i.e. some
    scaffold families are easier to extrapolate to than others),
    the pure GroupKFold variant will report a noticeably different
    mean RAE than the greedy-balanced one.

    Hypothesis: pure GroupKFold by scaffold either (a) tightens the
    estimate because no fold-balance heuristic injects implicit
    selection on which scaffolds get reserved together, or (b)
    inflates it because some folds end up with extreme size imbalance
    that hurts mean aggregation.  Either way, the delta vs
    nb2700-style scaffold-CV tells us whether the cycle-167
    nb2171 0.4682 ceiling is robust to CV-protocol or sits on a
    pxr-specific greedy-balance optimism.

PROTOCOL:
    1. Load X_117 substrate, slice K=20 surviving columns
       (nb2240 k20_surviving_idx_in_117).
    2. residual = y_unb - chemprop_aux_te[unb_idx]  (PRE-clean anchor).
    3. Compute Bemis-Murcko scaffold per unblind row (string keys;
       empty/singleton scaffolds mapped to per-row unique placeholder
       to preserve their no-grouping behaviour, identical to
       scaffold_kfold_indices).
    4. To make the GroupKFold result kf_seed-dependent (sklearn
       GroupKFold itself is order-deterministic with no shuffle knob),
       deterministically permute the *group labels themselves* with
       np.random.default_rng(kf_seed) before passing to GroupKFold.
       This keeps the protocol pure sklearn-style "no scaffold spans
       two folds" while letting us pick a reproducible kf_seed=1001
       partition.
    5. LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03,
       min_child_samples=5, reg_lambda=2.0) on raw K=20 features
       (no scaler -- match nb2240 K=20 LGBM baseline exactly).
    6. 5-fold GroupKFold CV at kf_seed=1001 -> single per-fold RAE
       trace and mean.  Also report fold-size variance and number of
       unique scaffolds per fold as diagnostic.
    7. Deploy: refit on full 253 -> predict 513 te residual.
    8. Compare mean_rae to scaffold_kfold_indices baseline
       (nb2240 K=20 reference = 0.4630).

GATE (mean RAE on the 5-fold GroupKFold):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2804_groupkfold_scaffold.py
    data/processed/nb2804_summary.json
    data/processed/nb2804_pred_oof.npy   (253,) float32 CORRECTED pred
    data/processed/te_nb2804.npy         (513,) float32 deploy refit
    submissions/nb2804_groupkfold_scaffold.csv
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
from sklearn.model_selection import GroupKFold

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2804"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEED = 1001

# Gate thresholds (mean RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # scaffold_kfold_indices baseline at K=20


def _lgbm_params(seed: int) -> dict:
    """Same LGBM hp as task spec / nb2240 / nb2700."""
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


def _seeded_group_labels(raw_groups: list[str], kf_seed: int) -> np.ndarray:
    """Deterministically permute the *group label namespace* with kf_seed.

    sklearn GroupKFold has no shuffle/seed knob -- it walks unique groups in
    discovery order and round-robins them across folds.  To get a
    kf_seed-dependent partition while preserving the "no scaffold spans two
    folds" guarantee, we shuffle the unique-group identifiers themselves and
    re-map every row to its shuffled identifier.  The resulting integer-group
    array passed to GroupKFold gives a different but still scaffold-disjoint
    partition for each kf_seed.
    """
    unique_groups = sorted(set(raw_groups))
    rng = np.random.default_rng(kf_seed)
    perm = rng.permutation(len(unique_groups))
    remap = {g: int(perm[i]) for i, g in enumerate(unique_groups)}
    return np.array([remap[g] for g in raw_groups], dtype=np.int64)


def _scaffold_keys_for_unb(unb_smiles: list[str]) -> list[str]:
    """Bemis-Murcko per row; singletons get unique per-row placeholders
    so they never group together (mirrors scaffold_kfold_indices semantics).
    """
    keys: list[str] = []
    for i, s in enumerate(unb_smiles):
        sc = bemis_murcko(s)
        if sc and isinstance(sc, str) and len(sc) > 0:
            keys.append(sc)
        else:
            keys.append(f"__singleton_{i}__")
    return keys


def _groupkfold_cv(
    X: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One pure-sklearn GroupKFold pass.  Returns (oof_residual, per-fold dicts).
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(gkf.split(X, residual, groups=groups)):
        # Guard: every val group must be absent from train
        train_groups = set(int(g) for g in groups[tr_loc])
        val_groups = set(int(g) for g in groups[va_loc])
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(
                f"GroupKFold leak: fold {fold_i} has {len(overlap)} "
                f"overlapping group ids"
            )
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
        fold_diags.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "n_groups_tr": int(len(train_groups)),
            "n_groups_va": int(len(val_groups)),
        })
    if np.isnan(oof).any():
        raise RuntimeError("GroupKFold did not cover all rows (NaN remaining)")
    return oof, fold_diags


def _deploy_te(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit LGBM on full 253 unblind, predict 513 te residual."""
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- pure sklearn GroupKFold by scaffold on K=20 substrate")
    print(f"        anchor = {ANCHOR}  n_folds={N_FOLDS}  kf_seed={KF_SEED}")
    print(f"        ref nb2240 K=20 scaffold_kfold_indices = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Truth + anchor + scaffolds ----
    te = load_test()
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

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"anchor te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load X_117 + slice K=20 ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape} expected ({n_unb},117)")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape} expected ({n_test},117)")
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)

    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"

    X_unb = X117_unb[:, k20_idx].astype(np.float32)
    X_te = X117_te[:, k20_idx].astype(np.float32)
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # ---- Scaffold groups ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    raw_scaffold_keys = _scaffold_keys_for_unb(unb_smiles)
    n_unique_scaf = len(set(raw_scaffold_keys))
    n_singletons = sum(1 for k in raw_scaffold_keys if k.startswith("__singleton_"))
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}  "
          f"singletons = {n_singletons}")
    if n_unique_scaf < N_FOLDS:
        raise RuntimeError(
            f"Need >= {N_FOLDS} unique scaffolds to run {N_FOLDS}-fold "
            f"GroupKFold; have {n_unique_scaf}"
        )

    groups = _seeded_group_labels(raw_scaffold_keys, KF_SEED)
    print(f"[groups] mapped to {len(set(groups.tolist()))} integer ids "
          f"(kf_seed={KF_SEED})")

    # ---- GroupKFold CV ----
    print("\n" + "-" * 78)
    print(f"PURE GROUPKFOLD CV  n_folds={N_FOLDS}  kf_seed={KF_SEED}")
    print("-" * 78)
    ts = time.time()
    resid_oof, fold_diags = _groupkfold_cv(
        X_unb, residual, groups, KF_SEED,
    )
    pred_corr_oof = anchor + resid_oof
    mean_rae = float(rae(y_unb, pred_corr_oof))

    # Per-fold RAE breakdown (computed *inside* each fold's val rows
    # against the same anchor slice -- diagnostic only; mean_rae is the
    # global aggregate over all 253 OOF preds).
    gkf_for_diag = GroupKFold(n_splits=N_FOLDS)
    per_fold_rae: list[float] = []
    fold_sizes_va: list[int] = []
    fold_sizes_tr: list[int] = []
    for fold_i, (tr_loc, va_loc) in enumerate(
        gkf_for_diag.split(X_unb, residual, groups=groups)
    ):
        y_va = y_unb[va_loc]
        pred_va = pred_corr_oof[va_loc]
        per_fold_rae.append(float(rae(y_va, pred_va)))
        fold_sizes_va.append(int(len(va_loc)))
        fold_sizes_tr.append(int(len(tr_loc)))

    fold_size_std = float(np.std(fold_sizes_va))
    fold_size_min = int(min(fold_sizes_va))
    fold_size_max = int(max(fold_sizes_va))

    print(f"[cv] global mean_rae        = {mean_rae:.4f}")
    print(f"[cv] per_fold_rae (val)     = "
          f"[{', '.join(f'{r:.4f}' for r in per_fold_rae)}]")
    print(f"[cv] per_fold_rae mean/std  = "
          f"{np.mean(per_fold_rae):.4f} / {np.std(per_fold_rae):.4f}")
    print(f"[cv] fold sizes (val)       = {fold_sizes_va}  "
          f"std={fold_size_std:.2f}  min={fold_size_min}  max={fold_size_max}")
    print(f"[cv] anchor RAE             = {rae_anchor:.4f}  "
          f"(d_vs_anchor = {mean_rae - rae_anchor:+.4f})")
    print(f"[cv] vs nb2240 K=20 ref     = "
          f"{NB2240_K20_REF:.4f}  (d = {mean_rae - NB2240_K20_REF:+.4f})  "
          f"-- positive = pure GroupKFold harder than greedy scaffold-CV")
    print(f"[cv] wall = {time.time() - ts:.1f}s")

    # ---- Comparison vs scaffold_kfold_indices on the SAME data ----
    # Build a kf_seed=1001 scaffold_kfold_indices partition using the same
    # *raw* scaffold strings so the comparison is apples-to-apples.
    skf_splits = scaffold_kfold_indices(
        raw_scaffold_keys, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    skf_oof = np.full(n_unb, np.nan, dtype=np.float64)
    for fold_i, (tr_loc, va_loc) in enumerate(skf_splits):
        mdl = lgb.LGBMRegressor(**_lgbm_params(KF_SEED))
        mdl.fit(X_unb[tr_loc], residual[tr_loc])
        skf_oof[va_loc] = mdl.predict(X_unb[va_loc])
    skf_pred_corr = anchor + skf_oof
    skf_mean_rae = float(rae(y_unb, skf_pred_corr))
    delta_protocol = mean_rae - skf_mean_rae
    print(f"[cv] greedy scaffold_kfold_indices same seed = "
          f"{skf_mean_rae:.4f}  (protocol delta = {delta_protocol:+.4f})")

    # ---- Deploy te ----
    te_resid_513 = _deploy_te(X_unb, residual, X_te, KF_SEED)
    te_deploy = (te_anchor_513 + te_resid_513.astype(np.float64)).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(refit on full 253, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = pred_corr_oof.astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_groupkfold_scaffold.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae            = {mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "pure_sklearn_GroupKFold_by_scaffold_K20_LGBM_residual_chemprop_aux",
        "rationale": (
            "Pure sklearn GroupKFold(n_splits=5) with scaffold group ids "
            "(deterministic group-id permutation by kf_seed=1001 to enable "
            "reproducible cross-CV protocol comparison) on the K=20 "
            "substrate; compares to scaffold_kfold_indices greedy-balanced "
            "baseline on identical features + anchor"
        ),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "x117_unb_path": str(X117_UNB_PATH),
        "x117_te_path": str(X117_TE_PATH),
        "k20_idx_source": str(NB2240_SUMMARY),
        "k20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "k20_surviving_names": k20_names,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_singleton_scaffolds": int(n_singletons),
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "cv_protocol": "sklearn.model_selection.GroupKFold(n_splits=5)",
        "group_seed_strategy": (
            "np.random.default_rng(kf_seed).permutation over unique scaffold "
            "labels -> remap each row to permuted integer id before "
            "GroupKFold.split"
        ),
        "feat_dim": int(X_unb.shape[1]),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEED),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "mean_rae": mean_rae,
        "per_fold_rae": [float(r) for r in per_fold_rae],
        "per_fold_rae_mean": float(np.mean(per_fold_rae)),
        "per_fold_rae_std": float(np.std(per_fold_rae)),
        "fold_sizes_val": fold_sizes_va,
        "fold_sizes_train": fold_sizes_tr,
        "fold_size_val_std": fold_size_std,
        "fold_size_val_min": fold_size_min,
        "fold_size_val_max": fold_size_max,
        "fold_diagnostics": fold_diags,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "delta_vs_nb2240_K20_scaffold_kfold": mean_rae - NB2240_K20_REF,
        "nb2240_K20_scaffold_kfold_ref": NB2240_K20_REF,
        "scaffold_kfold_indices_same_seed_rae": skf_mean_rae,
        "cv_protocol_delta_groupkfold_minus_scaffold_kfold": delta_protocol,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
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
        "mean_rae",
        "per_fold_rae",
        "per_fold_rae_mean",
        "per_fold_rae_std",
        "fold_sizes_val",
        "fold_size_val_std",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_scaffold_kfold",
        "scaffold_kfold_indices_same_seed_rae",
        "cv_protocol_delta_groupkfold_minus_scaffold_kfold",
        "te_unb_in_sample_rae",
        "n_unique_scaffolds",
        "n_singleton_scaffolds",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
