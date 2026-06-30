"""nb2814 -- GroupShuffleSplit (random scaffold-aware splits) on K=20 substrate.

NEW PARADIGM (CV-protocol axis, continuation of nb2804/nb2805 series):

    Both `scaffold_kfold_indices` (greedy-balanced) and pure sklearn
    `GroupKFold` partition all 253 unblind rows into exactly N_FOLDS
    disjoint scaffold buckets, yielding a SINGLE per-row OOF prediction.
    The single OOF estimate has fold-partition variance baked into it
    (cf. nb2804 protocol delta vs greedy and nb2060/nb2095/nb2171
    seed dispersion findings).

    `sklearn.model_selection.GroupShuffleSplit(n_splits=20, train_size=0.8)`
    is structurally different:
      * Each split independently samples train/val from the unique-group
        namespace (no fold-balancing, no fold-disjointness across splits).
      * train_size=0.8 of UNIQUE GROUPS, not rows -- consistent with the
        scaffold-aware contract.
      * A given row can appear in val multiple times (across the 20
        splits) or zero times.  Per-row OOF prediction is the mean of
        predictions made on splits where that row was in val.
      * The aggregate mean RAE is therefore a *resampled-mean* estimate
        rather than a single 5-fold OOF estimate.  Variance is reduced
        because each row's prediction averages over multiple
        scaffold-disjoint train sets.

    Hypothesis: 20-way random scaffold-aware resampling produces a
    tighter, less seed-sensitive mean RAE than 5-fold scaffold CV.  If
    the cycle-167 nb2171 0.4682 ceiling at K=20 is robust to CV
    protocol, the GroupShuffleSplit mean should land in the same band.
    If the band is materially lower, the prior single-fold OOF was
    upward-biased by adverse fold partitions.  If materially higher,
    GroupShuffleSplit's smaller train-side scaffold support (20% of
    scaffolds reserved as val every split, vs 20% per fold in 5-fold)
    is hurting model capacity.

    NOTE: scaffold groups are seeded once via the same
    `seeded-permute-then-pass` trick as nb2804 so kf_seed=1001 controls
    the partition reproducibly.  GroupShuffleSplit itself uses
    `random_state=42` per the task spec.

PROTOCOL:
    1. Load X_117 substrate, slice K=20 surviving columns
       (nb2240 k20_surviving_idx_in_117).
    2. residual = y_unb - chemprop_aux_te[unb_idx]  (PRE-clean anchor).
    3. Compute Bemis-Murcko scaffold per unblind row (singletons mapped
       to per-row unique placeholder, identical to scaffold_kfold_indices
       and nb2804 semantics).
    4. sklearn.model_selection.GroupShuffleSplit(
           n_splits=20, train_size=0.8, random_state=42)
       over the scaffold groups.
    5. For each of 20 splits: fit LGBM K=20 (same HP as nb2240/nb2804)
       on the fold-train rows, predict val rows.  Accumulate per-row
       prediction sums and counts.
    6. Per-row OOF residual prediction = sum / count (averaged across
       splits where the row was val).  Rows with count=0 (never sampled
       as val across the 20 splits) get fallback: a single LGBM trained
       on ALL OTHER rows in the unblind set (still scaffold-aware:
       train scaffold set excludes the never-val row's scaffold).
    7. pred_corr_oof = anchor + residual_oof; mean_rae = rae(y_unb, ...).
       Also report distribution of per-row val-counts as diagnostic.
    8. Deploy: refit on full 253 -> predict 513 te residual.

GATE (mean RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2814_group_shuffle_split.py
    data/processed/nb2814_summary.json
    data/processed/nb2814_pred_oof.npy   (253,) float32 CORRECTED pred
    data/processed/te_nb2814.npy         (513,) float32 deploy refit
    submissions/nb2814_group_shuffle_split.csv
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
from sklearn.model_selection import GroupShuffleSplit

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2814"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_SPLITS = 20
TRAIN_SIZE = 0.8
GSS_SEED = 42        # GroupShuffleSplit random_state (per task spec)
KF_SEED = 1001       # group-id permutation seed + LGBM seed
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630
NB2171_CEILING = 0.4682  # cycle-167 deep-30 reference


def _lgbm_params(seed: int) -> dict:
    """Same LGBM hp as nb2240/nb2700/nb2804 K=20 baseline."""
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

    Same trick as nb2804: shuffles unique scaffold identifiers so kf_seed
    controls the integer label assignment passed to GroupShuffleSplit
    (which itself uses GSS_SEED=42 for the split randomization).
    """
    unique_groups = sorted(set(raw_groups))
    rng = np.random.default_rng(kf_seed)
    perm = rng.permutation(len(unique_groups))
    remap = {g: int(perm[i]) for i, g in enumerate(unique_groups)}
    return np.array([remap[g] for g in raw_groups], dtype=np.int64)


def _scaffold_keys_for_unb(unb_smiles: list[str]) -> list[str]:
    keys: list[str] = []
    for i, s in enumerate(unb_smiles):
        sc = bemis_murcko(s)
        if sc and isinstance(sc, str) and len(sc) > 0:
            keys.append(sc)
        else:
            keys.append(f"__singleton_{i}__")
    return keys


def _groupshuffle_oof(
    X: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    train_size: float,
    gss_seed: int,
    lgbm_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run n_splits GroupShuffleSplit passes; accumulate per-row residual
    predictions and val-counts.  Returns
        (resid_oof_sum, val_count, per_split_diagnostics).
    """
    n = len(residual)
    pred_sum = np.zeros(n, dtype=np.float64)
    pred_cnt = np.zeros(n, dtype=np.int64)
    gss = GroupShuffleSplit(
        n_splits=n_splits, train_size=train_size, random_state=gss_seed,
    )
    diags: list[dict] = []
    for split_i, (tr_loc, va_loc) in enumerate(
        gss.split(X, residual, groups=groups)
    ):
        train_groups = set(int(g) for g in groups[tr_loc])
        val_groups = set(int(g) for g in groups[va_loc])
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(
                f"GroupShuffleSplit leak: split {split_i} has "
                f"{len(overlap)} overlapping scaffold group ids"
            )
        mdl = lgb.LGBMRegressor(**_lgbm_params(lgbm_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        pred_va = mdl.predict(X[va_loc])
        pred_sum[va_loc] += pred_va
        pred_cnt[va_loc] += 1
        diags.append({
            "split": split_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "n_groups_tr": int(len(train_groups)),
            "n_groups_va": int(len(val_groups)),
        })
    return pred_sum, pred_cnt, diags


def _fallback_for_uncovered(
    X: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    uncovered_idx: np.ndarray,
    lgbm_seed: int,
) -> np.ndarray:
    """For rows never sampled as val across 20 splits, predict each one
    via a leave-its-scaffold-out LGBM trained on the remaining 252 rows
    (still scaffold-aware: no row from the uncovered row's scaffold
    appears in train).  Returns an array of length len(uncovered_idx).
    """
    out = np.zeros(len(uncovered_idx), dtype=np.float64)
    for j, i in enumerate(uncovered_idx.tolist()):
        g_i = int(groups[i])
        train_mask = groups != g_i
        if not train_mask.any():
            # degenerate; predict zero residual
            out[j] = 0.0
            continue
        mdl = lgb.LGBMRegressor(**_lgbm_params(lgbm_seed))
        mdl.fit(X[train_mask], residual[train_mask])
        out[j] = float(mdl.predict(X[i:i + 1])[0])
    return out


def _deploy_te(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def main() -> dict:
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- GroupShuffleSplit (random scaffold-aware) K=20 LGBM residual")
    print(f"        anchor = {ANCHOR}  n_splits={N_SPLITS}  "
          f"train_size={TRAIN_SIZE}")
    print(f"        gss_seed={GSS_SEED}  kf_seed={KF_SEED}")
    print(f"        ref nb2240 K=20 scaffold_kfold = {NB2240_K20_REF:.4f}  "
          f"nb2171 ceiling = {NB2171_CEILING:.4f}")
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
    if n_unique_scaf < 2:
        raise RuntimeError(
            f"Need >= 2 unique scaffolds to run GroupShuffleSplit; "
            f"have {n_unique_scaf}"
        )

    groups = _seeded_group_labels(raw_scaffold_keys, KF_SEED)
    print(f"[groups] mapped to {len(set(groups.tolist()))} integer ids "
          f"(kf_seed={KF_SEED})")

    # ---- GroupShuffleSplit CV ----
    print("\n" + "-" * 78)
    print(f"GROUPSHUFFLESPLIT  n_splits={N_SPLITS}  "
          f"train_size={TRAIN_SIZE}  gss_seed={GSS_SEED}")
    print("-" * 78)
    ts = time.time()
    pred_sum, pred_cnt, split_diags = _groupshuffle_oof(
        X_unb, residual, groups,
        n_splits=N_SPLITS, train_size=TRAIN_SIZE,
        gss_seed=GSS_SEED, lgbm_seed=KF_SEED,
    )

    # Per-row val-count distribution diagnostic
    cnt_min = int(pred_cnt.min())
    cnt_max = int(pred_cnt.max())
    cnt_mean = float(pred_cnt.mean())
    cnt_std = float(pred_cnt.std())
    n_uncovered = int((pred_cnt == 0).sum())
    print(f"[cv] per-row val_count min/max/mean/std = "
          f"{cnt_min}/{cnt_max}/{cnt_mean:.2f}/{cnt_std:.2f}  "
          f"n_uncovered={n_uncovered}")

    # Per-split fold-size summary
    val_sizes = [d["n_va"] for d in split_diags]
    print(f"[cv] per-split val sizes mean/std = "
          f"{np.mean(val_sizes):.1f}/{np.std(val_sizes):.1f}  "
          f"min={min(val_sizes)} max={max(val_sizes)}")

    # ---- Aggregate per-row OOF residual ----
    resid_oof = np.zeros(n_unb, dtype=np.float64)
    covered_mask = pred_cnt > 0
    resid_oof[covered_mask] = pred_sum[covered_mask] / pred_cnt[covered_mask]

    n_fallback = 0
    if not covered_mask.all():
        uncovered_idx = np.where(~covered_mask)[0]
        print(f"[fallback] {len(uncovered_idx)} rows never sampled as val; "
              f"running leave-scaffold-out fallback LGBM each")
        fb_preds = _fallback_for_uncovered(
            X_unb, residual, groups, uncovered_idx, lgbm_seed=KF_SEED,
        )
        resid_oof[uncovered_idx] = fb_preds
        n_fallback = int(len(uncovered_idx))

    pred_corr_oof = anchor + resid_oof
    mean_rae = float(rae(y_unb, pred_corr_oof))

    # Per-split RAE diagnostic: compute RAE on the val rows of each split
    # (these RAEs are NOT averaged into mean_rae; the per-row mean
    # aggregation is the canonical CV summary).  Useful sanity check.
    gss_for_diag = GroupShuffleSplit(
        n_splits=N_SPLITS, train_size=TRAIN_SIZE, random_state=GSS_SEED,
    )
    per_split_rae: list[float] = []
    for split_i, (tr_loc, va_loc) in enumerate(
        gss_for_diag.split(X_unb, residual, groups=groups)
    ):
        mdl = lgb.LGBMRegressor(**_lgbm_params(KF_SEED))
        mdl.fit(X_unb[tr_loc], residual[tr_loc])
        pred_va = anchor[va_loc] + mdl.predict(X_unb[va_loc])
        per_split_rae.append(float(rae(y_unb[va_loc], pred_va)))

    per_split_rae_mean = float(np.mean(per_split_rae))
    per_split_rae_std = float(np.std(per_split_rae))

    print(f"[cv] global mean_rae (row-avg OOF) = {mean_rae:.4f}")
    print(f"[cv] per-split RAE mean/std        = "
          f"{per_split_rae_mean:.4f} / {per_split_rae_std:.4f}")
    print(f"[cv] anchor RAE                    = {rae_anchor:.4f}  "
          f"(d_vs_anchor = {mean_rae - rae_anchor:+.4f})")
    print(f"[cv] vs nb2240 K=20 ref            = "
          f"{NB2240_K20_REF:.4f}  (d = {mean_rae - NB2240_K20_REF:+.4f})")
    print(f"[cv] vs nb2171 ceiling             = "
          f"{NB2171_CEILING:.4f}  (d = {mean_rae - NB2171_CEILING:+.4f})")
    print(f"[cv] wall = {time.time() - ts:.1f}s")

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

    sub_csv = SUBMISSIONS / f"{TAG}_group_shuffle_split.csv"
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
        "method": (
            "GroupShuffleSplit_n20_train08_scaffold_K20_LGBM_residual_"
            "chemprop_aux"
        ),
        "rationale": (
            "20 random scaffold-aware splits via "
            "sklearn.model_selection.GroupShuffleSplit(n_splits=20, "
            "train_size=0.8, random_state=42); per-row OOF prediction is "
            "the mean over splits where the row was in val; fallback "
            "leave-its-scaffold-out LGBM for any row never sampled as val; "
            "compares to nb2804 GroupKFold and nb2240 scaffold_kfold_indices "
            "single-fold OOF baselines on identical K=20 features + anchor"
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
        "n_splits": N_SPLITS,
        "train_size": TRAIN_SIZE,
        "gss_seed": GSS_SEED,
        "kf_seed": KF_SEED,
        "cv_protocol": (
            "sklearn.model_selection.GroupShuffleSplit(n_splits=20, "
            "train_size=0.8, random_state=42); per-row OOF averaged over "
            "val occurrences"
        ),
        "group_seed_strategy": (
            "np.random.default_rng(kf_seed).permutation over unique scaffold "
            "labels -> remap each row to permuted integer id before "
            "GroupShuffleSplit.split"
        ),
        "feat_dim": int(X_unb.shape[1]),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEED),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "mean_rae": mean_rae,
        "per_split_rae": [float(r) for r in per_split_rae],
        "per_split_rae_mean": per_split_rae_mean,
        "per_split_rae_std": per_split_rae_std,
        "per_row_val_count_min": cnt_min,
        "per_row_val_count_max": cnt_max,
        "per_row_val_count_mean": cnt_mean,
        "per_row_val_count_std": cnt_std,
        "n_uncovered_fallback": n_fallback,
        "per_split_diagnostics": split_diags,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "delta_vs_nb2240_K20_scaffold_kfold": mean_rae - NB2240_K20_REF,
        "delta_vs_nb2171_ceiling": mean_rae - NB2171_CEILING,
        "nb2240_K20_scaffold_kfold_ref": NB2240_K20_REF,
        "nb2171_ceiling_ref": NB2171_CEILING,
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
        "per_split_rae_mean",
        "per_split_rae_std",
        "per_row_val_count_min",
        "per_row_val_count_max",
        "per_row_val_count_mean",
        "n_uncovered_fallback",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_scaffold_kfold",
        "delta_vs_nb2171_ceiling",
        "te_unb_in_sample_rae",
        "n_unique_scaffolds",
        "n_singleton_scaffolds",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
