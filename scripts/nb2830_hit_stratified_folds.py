"""nb2830 -- Hit-stratified scaffold folds on K=20 substrate.

NEW PARADIGM (CV-protocol / hit-stratification axis):
    Standard `pxr.eval.scaffold_kfold_indices` greedy-balances fold ROW
    counts but is hit-blind: scaffolds containing high-activity
    (avg pEC50 >= 5.5) compounds can end up concentrated in 1-2 folds
    by chance, especially given that only 1.6% of train are hits and
    the unblind 253 carries similar imbalance.  When a fold's TRAIN
    partition is hit-depleted, the K=20 LGBM on chemprop_aux residual
    learns to under-predict the activity tail; when a fold's VAL
    partition is hit-depleted, the per-fold RAE drops artificially
    (anchor alone is fine on inactives).  Both effects inflate mean-RAE
    variance.

    nb2830 reshapes the partition by:
      1) Computing per-scaffold "hit indicator" = (mean pEC50 across the
         scaffold's unblind rows) >= 5.5.
      2) StratifiedKFold(n_splits=5, shuffle=True, kf_seed=1001) over
         the UNIQUE scaffolds using the per-scaffold hit indicator as
         the stratification label.
      3) Mapping each row to its scaffold's assigned fold (so no
         scaffold spans two folds, matching scaffold_kfold_indices'
         disjointness guarantee).
      4) LGBM(K=20, standard nb2240 hp) on chemprop_aux residual ->
         5-fold OOF RAE.

    Compared head-to-head against scaffold_kfold_indices on the SAME
    (X_unb, residual, anchor) tuple.  If the cycle-167 nb2171 0.4682
    ceiling carried hit-mix optimism from greedy fold-balancing,
    hit-stratified folds will report a tighter (lower-variance, possibly
    lower-mean) per-fold RAE distribution.

PROTOCOL:
    1. Load X_117 substrate (nb2240) -> slice K=20 surviving cols.
    2. anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    3. Compute Bemis-Murcko scaffold per unblind row (singletons get
       per-row unique placeholders, matching scaffold_kfold_indices
       semantics).
    4. Per-scaffold hit indicator = (mean(y_unb over rows of scaffold)
       >= 5.5).  Singletons get the indicator of their single row.
    5. StratifiedKFold(n_splits=5, shuffle=True, kf_seed=1001) on unique
       scaffold indices labelled by the hit indicator -> assigns each
       scaffold to one of 5 folds.  Every row inherits its scaffold's
       fold id, so no scaffold spans two folds.
    6. 5-fold CV: LGBM (max_depth=4, num_leaves=15, n_est=300, lr=0.03,
       min_child_samples=5, reg_lambda=2.0) on raw K=20 features
       (no scaler -- match nb2240 K=20 LGBM baseline exactly).
    7. Deploy: refit on full 253 -> predict 513 te residual.
    8. Compare mean_rae to scaffold_kfold_indices baseline
       (nb2240 K=20 reference = 0.4630) using the SAME LGBM seed.

GATE (mean RAE on the 5-fold hit-stratified CV):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2830_hit_stratified_folds.py
    data/processed/nb2830_summary.json
    data/processed/nb2830_pred_oof.npy   (253,) float32 CORRECTED pred
    data/processed/te_nb2830.npy         (513,) float32 deploy refit
    submissions/nb2830_hit_stratified_folds.csv
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
from sklearn.model_selection import StratifiedKFold

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2830"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEED = 1001
HIT_THRESHOLD = 5.5  # avg pEC50 >= 5.5 -> hit indicator True

# Gate thresholds (mean RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # scaffold_kfold_indices baseline at K=20


def _lgbm_params(seed: int) -> dict:
    """Same LGBM hp as nb2240 / nb2700 / nb2804 / nb2813."""
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


def _scaffold_keys_for_unb(unb_smiles: list[str]) -> list[str]:
    """Bemis-Murcko per row; empty scaffolds get unique per-row placeholders
    so they never group (mirrors scaffold_kfold_indices semantics)."""
    keys: list[str] = []
    for i, s in enumerate(unb_smiles):
        sc = bemis_murcko(s)
        if sc and isinstance(sc, str) and len(sc) > 0:
            keys.append(sc)
        else:
            keys.append(f"__singleton_{i}__")
    return keys


def _hit_stratified_splits(
    raw_scaffold_keys: list[str],
    y_unb: np.ndarray,
    n_folds: int,
    kf_seed: int,
    hit_threshold: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """StratifiedKFold over unique scaffolds, labelled by per-scaffold
    hit indicator (mean(y) over scaffold's rows >= hit_threshold).

    Every row inherits the fold id of its scaffold, so the resulting
    splits are scaffold-disjoint just like scaffold_kfold_indices.

    Returns
    -------
    splits : list[(tr_loc, va_loc)]  -- 5 fold tuples of row-index arrays
    diag   : dict                    -- per-fold hit-count diagnostics
    """
    n_unb = len(y_unb)
    # Unique scaffolds in stable (sorted) order so the build is reproducible
    unique_scaf = sorted(set(raw_scaffold_keys))
    n_scaf = len(unique_scaf)
    if n_scaf < n_folds:
        raise RuntimeError(
            f"Need >= {n_folds} unique scaffolds for {n_folds}-fold "
            f"StratifiedKFold; have {n_scaf}"
        )

    # Per-scaffold hit indicator = mean(y over scaffold rows) >= threshold
    scaf_to_rows: dict[str, list[int]] = {s: [] for s in unique_scaf}
    for i, s in enumerate(raw_scaffold_keys):
        scaf_to_rows[s].append(i)
    scaf_mean_y = np.array(
        [float(np.mean(y_unb[scaf_to_rows[s]])) for s in unique_scaf],
        dtype=np.float64,
    )
    scaf_hit = (scaf_mean_y >= hit_threshold).astype(np.int32)
    n_hit_scaf = int(scaf_hit.sum())
    n_nonhit_scaf = int((1 - scaf_hit).sum())

    # If either stratum has < n_folds members, StratifiedKFold will refuse;
    # in that case fall back gracefully to a single-class label so the
    # split still runs (this preserves scaffold-disjointness but the
    # stratification effectively degrades to shuffled K-Fold over scaffolds).
    if min(n_hit_scaf, n_nonhit_scaf) < n_folds:
        warnings.warn(
            f"Hit stratum has only {n_hit_scaf} (vs {n_nonhit_scaf} "
            f"non-hit) scaffolds; StratifiedKFold requires >= {n_folds} "
            f"per class.  Falling back to unstratified shuffled scaffold "
            f"K-Fold."
        )
        skf_labels = np.zeros_like(scaf_hit)
    else:
        skf_labels = scaf_hit

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=kf_seed)
    # Build per-scaffold fold id, then expand to per-row
    scaf_fold = np.full(n_scaf, -1, dtype=np.int32)
    for fold_i, (_, va_scaf_idx) in enumerate(skf.split(unique_scaf, skf_labels)):
        for j in va_scaf_idx:
            scaf_fold[j] = fold_i
    if (scaf_fold < 0).any():
        raise RuntimeError("StratifiedKFold did not assign every scaffold a fold")

    scaf_to_fold = {s: int(scaf_fold[i]) for i, s in enumerate(unique_scaf)}
    per_row_fold = np.array(
        [scaf_to_fold[s] for s in raw_scaffold_keys], dtype=np.int32,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    per_fold_diag = []
    all_rows = np.arange(n_unb, dtype=np.int64)
    for fold_i in range(n_folds):
        va_loc = all_rows[per_row_fold == fold_i]
        tr_loc = all_rows[per_row_fold != fold_i]
        # Scaffold disjoint check
        tr_scafs = {raw_scaffold_keys[i] for i in tr_loc}
        va_scafs = {raw_scaffold_keys[i] for i in va_loc}
        overlap = tr_scafs & va_scafs
        if overlap:
            raise RuntimeError(
                f"Fold {fold_i} scaffold leak: {len(overlap)} shared scaffolds"
            )
        n_hit_rows_va = int((y_unb[va_loc] >= hit_threshold).sum())
        n_hit_rows_tr = int((y_unb[tr_loc] >= hit_threshold).sum())
        n_hit_scaf_va = int(
            sum(1 for s in va_scafs if scaf_to_fold[s] == fold_i
                and scaf_hit[unique_scaf.index(s)] == 1)
        )
        per_fold_diag.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "n_scaf_tr": int(len(tr_scafs)),
            "n_scaf_va": int(len(va_scafs)),
            "n_hit_rows_tr": n_hit_rows_tr,
            "n_hit_rows_va": n_hit_rows_va,
            "n_hit_scaf_va": n_hit_scaf_va,
            "hit_rate_va": (n_hit_rows_va / max(len(va_loc), 1)),
        })
        splits.append((tr_loc, va_loc))

    diag = {
        "n_unique_scaffolds": int(n_scaf),
        "hit_threshold": float(hit_threshold),
        "n_hit_scaffolds": int(n_hit_scaf),
        "n_nonhit_scaffolds": int(n_nonhit_scaf),
        "n_hit_rows_total": int((y_unb >= hit_threshold).sum()),
        "stratified_fallback_used": bool(
            min(n_hit_scaf, n_nonhit_scaf) < n_folds
        ),
        "per_fold": per_fold_diag,
    }
    return splits, diag


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
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HIT-STRATIFIED scaffold folds on K=20 substrate")
    print(f"        anchor = {ANCHOR}  hit_threshold = {HIT_THRESHOLD}")
    print(f"        n_folds={N_FOLDS}  kf_seed={KF_SEED}")
    print(f"        ref nb2240 K=20 scaffold_kfold = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Truth + anchor ----
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
        raise ValueError(f"X117_unb shape {X117_unb.shape}")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape}")
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

    # ---- Scaffold keys + hit-stratified fold assignment ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    raw_scaffold_keys = _scaffold_keys_for_unb(unb_smiles)
    n_unique_scaf = len(set(raw_scaffold_keys))
    n_singletons = sum(1 for k in raw_scaffold_keys if k.startswith("__singleton_"))
    n_hit_rows = int((y_unb >= HIT_THRESHOLD).sum())
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}  "
          f"singletons = {n_singletons}")
    print(f"[hit] threshold={HIT_THRESHOLD}  n_hit_rows={n_hit_rows}  "
          f"hit_rate={n_hit_rows / n_unb:.4f}")

    splits, strat_diag = _hit_stratified_splits(
        raw_scaffold_keys, y_unb, N_FOLDS, KF_SEED, HIT_THRESHOLD,
    )
    print(f"[strat] n_hit_scaffolds = {strat_diag['n_hit_scaffolds']}  "
          f"n_nonhit_scaffolds = {strat_diag['n_nonhit_scaffolds']}")
    print(f"[strat] fallback_used = {strat_diag['stratified_fallback_used']}")
    for d in strat_diag["per_fold"]:
        print(f"   fold {d['fold']}: n_tr={d['n_tr']:3d}  n_va={d['n_va']:3d}  "
              f"hit_rows_va={d['n_hit_rows_va']:2d}  "
              f"hit_rate_va={d['hit_rate_va']:.4f}  "
              f"hit_scaf_va={d['n_hit_scaf_va']}")

    # ---- Hit-stratified scaffold 5-fold CV ----
    print("\n" + "-" * 78)
    print(f"HIT-STRATIFIED SCAFFOLD CV  n_folds={N_FOLDS}  kf_seed={KF_SEED}")
    print("-" * 78)
    ts = time.time()
    resid_oof = np.full(n_unb, np.nan, dtype=np.float64)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**_lgbm_params(KF_SEED))
        mdl.fit(X_unb[tr_loc], residual[tr_loc])
        resid_oof[va_loc] = mdl.predict(X_unb[va_loc])
    if np.isnan(resid_oof).any():
        raise RuntimeError("Hit-stratified CV did not cover all rows (NaN OOF)")

    pred_corr_oof = anchor + resid_oof
    mean_rae = float(rae(y_unb, pred_corr_oof))
    per_fold_rae: list[float] = []
    fold_sizes_va: list[int] = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        y_va = y_unb[va_loc]
        pred_va = pred_corr_oof[va_loc]
        per_fold_rae.append(float(rae(y_va, pred_va)))
        fold_sizes_va.append(int(len(va_loc)))

    fold_size_std = float(np.std(fold_sizes_va))
    print(f"[cv] global mean_rae        = {mean_rae:.4f}")
    print(f"[cv] per_fold_rae           = "
          f"[{', '.join(f'{r:.4f}' for r in per_fold_rae)}]")
    print(f"[cv] per_fold mean/std      = "
          f"{np.mean(per_fold_rae):.4f} / {np.std(per_fold_rae):.4f}")
    print(f"[cv] fold sizes (val)       = {fold_sizes_va}  "
          f"std={fold_size_std:.2f}")
    print(f"[cv] vs anchor              = "
          f"{mean_rae - rae_anchor:+.4f}  (anchor {rae_anchor:.4f})")
    print(f"[cv] vs nb2240 K=20         = "
          f"{mean_rae - NB2240_K20_REF:+.4f}  (ref {NB2240_K20_REF:.4f})")
    print(f"[cv] wall = {time.time() - ts:.1f}s")

    # ---- Side-by-side comparison vs scaffold_kfold_indices same seed ----
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
    skf_per_fold_rae = []
    skf_per_fold_hit_rate = []
    for fold_i, (tr_loc, va_loc) in enumerate(skf_splits):
        skf_per_fold_rae.append(
            float(rae(y_unb[va_loc], skf_pred_corr[va_loc]))
        )
        skf_per_fold_hit_rate.append(
            float((y_unb[va_loc] >= HIT_THRESHOLD).mean())
        )
    delta_protocol = mean_rae - skf_mean_rae
    print(f"[cv] greedy scaffold_kfold_indices same seed = "
          f"{skf_mean_rae:.4f}  (protocol delta = {delta_protocol:+.4f})")
    print(f"[cv] greedy per-fold RAE = "
          f"[{', '.join(f'{r:.4f}' for r in skf_per_fold_rae)}]  "
          f"std={np.std(skf_per_fold_rae):.4f}")
    print(f"[cv] greedy per-fold hit-rate = "
          f"[{', '.join(f'{r:.4f}' for r in skf_per_fold_hit_rate)}]  "
          f"std={np.std(skf_per_fold_hit_rate):.4f}")

    # ---- Deploy ----
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

    sub_csv = SUBMISSIONS / f"{TAG}_hit_stratified_folds.csv"
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
        "method": "hit_stratified_scaffold_folds_K20_LGBM_chemprop_aux_residual",
        "rationale": (
            "StratifiedKFold(n_splits=5, shuffle=True, kf_seed=1001) over "
            "unique Bemis-Murcko scaffolds, stratified by per-scaffold hit "
            "indicator (mean(y_unb) >= 5.5).  Every row inherits its "
            "scaffold's fold -- scaffold-disjoint just like "
            "scaffold_kfold_indices.  Goal: preserve hit ratio across "
            "folds so per-fold RAE no longer carries hit-mix variance "
            "from greedy row-balancing."
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
        "hit_threshold": float(HIT_THRESHOLD),
        "n_hit_rows": n_hit_rows,
        "hit_rate_global": float(n_hit_rows / n_unb),
        "cv_protocol": (
            "sklearn.StratifiedKFold(n_splits=5, shuffle=True, "
            "random_state=kf_seed) over unique scaffolds labelled by "
            "per-scaffold mean(y_unb)>=5.5 indicator"
        ),
        "stratification_diag": strat_diag,
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
        "fold_size_val_std": fold_size_std,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "delta_vs_nb2240_K20_scaffold_kfold": mean_rae - NB2240_K20_REF,
        "nb2240_K20_scaffold_kfold_ref": NB2240_K20_REF,
        "scaffold_kfold_indices_same_seed_rae": skf_mean_rae,
        "scaffold_kfold_indices_per_fold_rae": [
            float(r) for r in skf_per_fold_rae
        ],
        "scaffold_kfold_indices_per_fold_rae_std": float(
            np.std(skf_per_fold_rae)
        ),
        "scaffold_kfold_indices_per_fold_hit_rate": [
            float(r) for r in skf_per_fold_hit_rate
        ],
        "scaffold_kfold_indices_per_fold_hit_rate_std": float(
            np.std(skf_per_fold_hit_rate)
        ),
        "cv_protocol_delta_hitstrat_minus_scaffold_kfold": delta_protocol,
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
        "scaffold_kfold_indices_per_fold_rae_std",
        "scaffold_kfold_indices_per_fold_hit_rate_std",
        "cv_protocol_delta_hitstrat_minus_scaffold_kfold",
        "te_unb_in_sample_rae",
        "n_unique_scaffolds",
        "n_singleton_scaffolds",
        "n_hit_rows",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
