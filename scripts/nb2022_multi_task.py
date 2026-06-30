"""nb2022 -- Multi-task LGBM predicting (pec50, pec50_null, selectivity).

HYPOTHESIS:
    nb1202 (single counter-assay-axis LGBM used as a +1-dim aux feature) failed
    to beat nb2103 K=28 (this-alone 0.7444 vs nb2103 0.4737). Diagnosis: a
    single counter-assay scalar carries low signal AND the joint encoding is
    not forced. A multi-task / sequential-boost setup that explicitly chains
    three biologically distinct targets -- pec50_null (off-target promiscuity),
    selectivity (pec50 - pec50_null, intrinsic PXR specificity), pec50 (raw
    on-target potency) -- may force a shared representation: stage-3 sees both
    the predicted promiscuity AND the predicted selectivity, so it can learn
    the conditional pec50|null,sel structure rather than just pec50|null.

PROTOCOL:
    1. Load 4139 train rows (pec50). Inner-join with counter-assay (pec50_null)
       on standardized SMILES to get JOIN rows (both labels). Train-only rows
       (pec50 but no null) are used for the FINAL pec50 fit; stage-1 + stage-2
       are JOIN-only because they need a null target / selectivity target.
    2. Combined features (Morgan-2048 + RDKit-217 = 2265 dims).
    3. STAGE-1 (JOIN): LGBM(MSE) K=28 -> pec50_null. 5-fold scaffold cross-fit
       gives null_oof on JOIN; refit-full -> null_te (513) + null_tr_only.
    4. STAGE-2 (JOIN): LGBM(MSE) K=28 -> selectivity (= pec50 - pec50_null),
       feature = [combined, null_pred]. 5-fold scaffold cross-fit using
       null_oof (no leakage); refit-full -> sel_te + sel_tr_only.
    5. STAGE-3 (FULL TRAIN = JOIN + train-only): LGBM(MSE) K=28 -> pec50,
       feature = [combined, null_pred, sel_pred]. 5-fold scaffold cross-fit;
       5-seed bag (seeds 0/1/7/42/137). Refit-full -> final_te (513).
    6. Final pec50 OOF on 253 = final_te[unb_idx]. Compute mean-bag RAE.
    7. Apply post-hoc rank-stretch (per memory: feedback_rank_stretch_universal,
       s in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30}) honest 5-fold cross-fit
       on the 253 unblind. Take best-s.
    8. Compare standalone + best-stretch RAE vs nb2103 K=28 (mean-bag 0.4737;
       script ref 0.5057 per spec) at margin 0.003. If beats: write deploy CSV.

OUTPUTS:
    scripts/nb2022_multi_task.py
    data/processed/nb2022_summary.json
    data/processed/nb2022_mean_bag_oof.npy           (253,) float32 -- final stage pec50 OOF
    data/processed/nb2022_te_meanbag.npy             (513,) float32 -- refit-full deploy
    data/processed/nb2022_te_meanbag_stretched.npy   (513,) float32 -- best-s stretched
    submissions/nb2022_multitask.csv                 IFF beats nb2103 K=28
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
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.data import load_train, load_counter, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2022"
NB2103_K28_REF = 0.5057          # spec-supplied "compare vs nb2103 (0.5057)"
NB2103_K28_MEAN_BAG_REF = 0.4737  # actual mean-bag OOF on 253 (from nb2103 summary)
DECISION_MARGIN = 0.003

CHEMPROP_AUX_TE = DATA_PROCESSED / "te_chemprop_aux.npy"
UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5
LGBM_K = 28
STRETCH_GRID = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]


def _murcko(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) or ""
    except Exception:
        return ""


def _lgbm_params(seed: int, num_leaves: int = LGBM_K) -> dict:
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=num_leaves,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _scaffold_cross_fit(
    X: np.ndarray,
    y: np.ndarray,
    scaffolds: list[str],
    seed: int,
    num_leaves: int = LGBM_K,
) -> np.ndarray:
    """5-fold scaffold cross-fit; return OOF predictions (NaN-free)."""
    folds = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for tr_idx, va_idx in folds:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed, num_leaves=num_leaves))
        mdl.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = mdl.predict(X[va_idx])
    assert not np.any(np.isnan(oof)), "OOF has NaNs"
    return oof


def _refit_full_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    num_leaves: int = LGBM_K,
) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed, num_leaves=num_leaves))
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te).astype(np.float64)


def _cross_fit_stretch_best_s(p_pred: np.ndarray, y_true: np.ndarray,
                              s_grid: list[float], seed: int = 42) -> tuple[float, float, np.ndarray]:
    """Honest 5-fold cross-fit grid-search for best scalar stretch.

    Recipe (per feedback_rank_stretch_universal): stretched = mu + s * (p - mu)
    with per-fold mu from train fold. Returns (best_s, pooled_rae, stretched_oof).
    """
    n = len(p_pred)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    s_arr = np.asarray(s_grid, dtype=np.float64)

    # For each fold: pick the s that minimises train-fold RAE; apply on val fold.
    stretched = np.full(n, np.nan, dtype=np.float64)
    fold_best_s = []
    for tr_idx, va_idx in kf.split(np.arange(n)):
        p_tr, y_tr = p_pred[tr_idx], y_true[tr_idx]
        p_va = p_pred[va_idx]
        mu_tr = float(p_tr.mean())
        # Search s on train fold.
        best_s_loc, best_r_loc = 1.0, np.inf
        for s in s_arr:
            pred_tr_s = mu_tr + s * (p_tr - mu_tr)
            r = rae(y_tr, pred_tr_s)
            if r < best_r_loc:
                best_r_loc, best_s_loc = r, float(s)
        fold_best_s.append(best_s_loc)
        stretched[va_idx] = mu_tr + best_s_loc * (p_va - mu_tr)

    assert not np.any(np.isnan(stretched))
    pooled_rae = float(rae(y_true, stretched))
    mean_s = float(np.mean(fold_best_s))
    return mean_s, pooled_rae, stretched


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- multi-task LGBM (pec50_null -> selectivity -> pec50)")
    print(f"   spec_ref nb2103 = {NB2103_K28_REF:.4f}   actual_mean_bag = {NB2103_K28_MEAN_BAG_REF:.4f}")
    print(f"   decision margin = {DECISION_MARGIN}")
    print("=" * 78)

    summary: dict = {
        "tag": TAG,
        "purpose": ("Multi-task sequential-boost LGBM: stage1=pec50_null, "
                    "stage2=selectivity, stage3=pec50 conditioned on both."),
        "lgbm_num_leaves": LGBM_K,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "nb2103_K28_spec_ref": NB2103_K28_REF,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
    }

    # ---- 1. Build JOIN (pec50 + pec50_null) and TRAIN-ONLY (pec50 only) ----
    print("\n[step1] loading + joining train + counter ...")
    tr = load_train()
    co = load_counter()
    tr["std_smi"] = tr["smiles"].apply(
        lambda s: Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else None
    )
    co["std_smi"] = co["smiles"].apply(
        lambda s: Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else None
    )
    tr_g = (
        tr[tr["std_smi"].notna() & tr["pec50"].notna()]
        .groupby("std_smi", as_index=False)
        .agg(pec50=("pec50", "median"))
    )
    co_g = (
        co[co["std_smi"].notna() & co["pec50"].notna()]
        .groupby("std_smi", as_index=False)
        .agg(pec50_null=("pec50", "median"))
    )
    joined = tr_g.merge(co_g, on="std_smi", how="inner")
    train_only = tr_g[~tr_g["std_smi"].isin(joined["std_smi"])].reset_index(drop=True)
    print(f"   train n_uniq      = {len(tr_g)}")
    print(f"   counter n_uniq    = {len(co_g)}")
    print(f"   JOIN (both)       = {len(joined)}")
    print(f"   train-only (no null) = {len(train_only)}")

    smiles_join = joined["std_smi"].tolist()
    y_pec50_j = joined["pec50"].to_numpy(dtype=np.float64)
    y_null_j = joined["pec50_null"].to_numpy(dtype=np.float64)
    y_sel_j = y_pec50_j - y_null_j
    scaff_j = [_murcko(s) for s in smiles_join]

    smiles_to = train_only["std_smi"].tolist()
    y_pec50_to = train_only["pec50"].to_numpy(dtype=np.float64)
    scaff_to = [_murcko(s) for s in smiles_to]

    # ---- 2. Features ----
    print("\n[step2] computing combined features (Morgan + RDKit) ...")
    X_j = impute(combined(smiles_join)).astype(np.float32)
    X_to = impute(combined(smiles_to)).astype(np.float32) if len(smiles_to) else np.empty((0, X_j.shape[1]), dtype=np.float32)
    te_df = load_test()
    te_smiles_raw = te_df["smiles"].astype(str).tolist()
    te_std_smi = [
        Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else s
        for s in te_smiles_raw
    ]
    X_te = impute(combined(te_std_smi)).astype(np.float32)
    print(f"   X_j  = {X_j.shape}   X_to = {X_to.shape}   X_te = {X_te.shape}")

    # ---- 3. STAGE-1: pec50_null on JOIN ----
    print("\n[step3] STAGE-1 LGBM -> pec50_null (JOIN) ...")
    null_oof_seeds = []
    null_te_seeds = []
    null_to_seeds = []  # null pred for train-only rows (refit-full)
    for s in SEEDS:
        ts = time.time()
        oof = _scaffold_cross_fit(X_j, y_null_j, scaff_j, seed=s, num_leaves=LGBM_K)
        null_oof_seeds.append(oof)
        te_s = _refit_full_predict(X_j, y_null_j, X_te, seed=s, num_leaves=LGBM_K)
        null_te_seeds.append(te_s)
        if len(smiles_to):
            to_s = _refit_full_predict(X_j, y_null_j, X_to, seed=s, num_leaves=LGBM_K)
            null_to_seeds.append(to_s)
        print(f"   seed={s:3d}  stage1 OOF RAE={rae(y_null_j, oof):.4f}  wall={time.time()-ts:.1f}s")
    null_oof_mean = np.mean(np.stack(null_oof_seeds, axis=0), axis=0)
    null_te_mean = np.mean(np.stack(null_te_seeds, axis=0), axis=0)
    null_to_mean = (
        np.mean(np.stack(null_to_seeds, axis=0), axis=0)
        if len(smiles_to) else np.empty((0,), dtype=np.float64)
    )
    stage1_rae = float(rae(y_null_j, null_oof_mean))
    print(f"[step3] STAGE-1 mean-bag OOF RAE (pec50_null) = {stage1_rae:.4f}")

    # ---- 4. STAGE-2: selectivity on JOIN, feature = [combined, null_oof_pred] ----
    print("\n[step4] STAGE-2 LGBM -> selectivity (pec50 - pec50_null) on JOIN ...")
    X_j_s2 = np.hstack([X_j, null_oof_mean.reshape(-1, 1).astype(np.float32)])
    sel_oof_seeds = []
    sel_te_seeds = []
    sel_to_seeds = []
    X_te_s2 = np.hstack([X_te, null_te_mean.reshape(-1, 1).astype(np.float32)])
    X_to_s2 = (
        np.hstack([X_to, null_to_mean.reshape(-1, 1).astype(np.float32)])
        if len(smiles_to) else np.empty((0, X_j_s2.shape[1]), dtype=np.float32)
    )
    for s in SEEDS:
        ts = time.time()
        oof = _scaffold_cross_fit(X_j_s2, y_sel_j, scaff_j, seed=s, num_leaves=LGBM_K)
        sel_oof_seeds.append(oof)
        te_s = _refit_full_predict(X_j_s2, y_sel_j, X_te_s2, seed=s, num_leaves=LGBM_K)
        sel_te_seeds.append(te_s)
        if len(smiles_to):
            to_s = _refit_full_predict(X_j_s2, y_sel_j, X_to_s2, seed=s, num_leaves=LGBM_K)
            sel_to_seeds.append(to_s)
        print(f"   seed={s:3d}  stage2 OOF RAE={rae(y_sel_j, oof):.4f}  wall={time.time()-ts:.1f}s")
    sel_oof_mean = np.mean(np.stack(sel_oof_seeds, axis=0), axis=0)
    sel_te_mean = np.mean(np.stack(sel_te_seeds, axis=0), axis=0)
    sel_to_mean = (
        np.mean(np.stack(sel_to_seeds, axis=0), axis=0)
        if len(smiles_to) else np.empty((0,), dtype=np.float64)
    )
    stage2_rae = float(rae(y_sel_j, sel_oof_mean))
    print(f"[step4] STAGE-2 mean-bag OOF RAE (selectivity) = {stage2_rae:.4f}")

    # ---- 5. STAGE-3: pec50 on FULL TRAIN, feature = [combined, null, sel] ----
    print("\n[step5] STAGE-3 LGBM -> pec50 (FULL TRAIN) with [combined, null, sel] ...")
    # Build full-train matrix (JOIN + train-only). For JOIN rows use OOF
    # stage-1/2 preds (no leakage); for train-only rows use refit-full preds.
    X_join_s3 = np.hstack([
        X_j,
        null_oof_mean.reshape(-1, 1).astype(np.float32),
        sel_oof_mean.reshape(-1, 1).astype(np.float32),
    ])
    if len(smiles_to):
        X_to_s3 = np.hstack([
            X_to,
            null_to_mean.reshape(-1, 1).astype(np.float32),
            sel_to_mean.reshape(-1, 1).astype(np.float32),
        ])
    else:
        X_to_s3 = np.empty((0, X_join_s3.shape[1]), dtype=np.float32)
    X_full_s3 = np.vstack([X_join_s3, X_to_s3]) if len(smiles_to) else X_join_s3.copy()
    y_full = np.concatenate([y_pec50_j, y_pec50_to]) if len(smiles_to) else y_pec50_j.copy()
    scaff_full = scaff_j + scaff_to

    X_te_s3 = np.hstack([
        X_te,
        null_te_mean.reshape(-1, 1).astype(np.float32),
        sel_te_mean.reshape(-1, 1).astype(np.float32),
    ])
    print(f"   X_full_s3 = {X_full_s3.shape}   X_te_s3 = {X_te_s3.shape}")

    pec50_oof_seeds = []
    pec50_te_seeds = []
    for s in SEEDS:
        ts = time.time()
        oof = _scaffold_cross_fit(X_full_s3, y_full, scaff_full, seed=s, num_leaves=LGBM_K)
        pec50_oof_seeds.append(oof)
        te_s = _refit_full_predict(X_full_s3, y_full, X_te_s3, seed=s, num_leaves=LGBM_K)
        pec50_te_seeds.append(te_s)
        print(f"   seed={s:3d}  stage3 OOF RAE(full)={rae(y_full, oof):.4f}  wall={time.time()-ts:.1f}s")
    pec50_oof_mean = np.mean(np.stack(pec50_oof_seeds, axis=0), axis=0)
    pec50_te_mean = np.mean(np.stack(pec50_te_seeds, axis=0), axis=0)
    full_oof_rae = float(rae(y_full, pec50_oof_mean))
    print(f"[step5] STAGE-3 mean-bag OOF RAE (full train) = {full_oof_rae:.4f}")

    # ---- 6. Final pec50 OOF on 253 ----
    print("\n[step6] final pec50 OOF on 253 ...")
    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    final_unb = pec50_te_mean[unb_idx]
    rae_unb = float(rae(y_unb, final_unb))
    print(f"   nb2022 standalone te[unb_idx] RAE = {rae_unb:.4f}")

    # Diagnostic: anchor + per-seed-bag mean/std on 253
    anchor_unb = np.load(CHEMPROP_AUX_TE).astype(np.float64)[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"   chemprop_aux te[unb_idx] RAE      = {rae_anchor:.4f}")

    # ---- 7. Post-hoc rank-stretch ----
    print("\n[step7] honest 5-fold cross-fit rank-stretch sweep ...")
    best_s, rae_stretched_pooled, stretched_oof = _cross_fit_stretch_best_s(
        final_unb, y_unb, STRETCH_GRID, seed=42
    )
    print(f"   best mean-fold s = {best_s:.3f}   pooled stretched RAE = {rae_stretched_pooled:.4f}")

    # Apply best-fold-mean stretch to deploy (513) for the CSV output
    # (deploy stretch uses mean of fold-best s and global te mean as mu).
    mu_deploy = float(pec50_te_mean.mean())
    te_stretched_deploy = mu_deploy + best_s * (pec50_te_mean - mu_deploy)

    # ---- 8. Decision vs nb2103 K=28 (use actual mean-bag ref 0.4737) ----
    print("\n[step8] decision vs nb2103 K=28 ...")
    rae_best = min(rae_unb, rae_stretched_pooled)
    beats_actual = rae_best < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats_spec = rae_best < NB2103_K28_REF - DECISION_MARGIN
    flat_actual = abs(rae_best - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN
    if beats_actual:
        verdict = "BEATS_NB2103_K28_MEAN_BAG"
    elif flat_actual:
        verdict = "FLAT_VS_NB2103_K28_MEAN_BAG"
    elif beats_spec:
        verdict = "BEATS_NB2103_K28_SPEC_REF_ONLY"
    else:
        verdict = "WORSE_THAN_NB2103_K28"
    delta_vs_mean_bag = rae_best - NB2103_K28_MEAN_BAG_REF
    delta_vs_spec = rae_best - NB2103_K28_REF
    print(f"   rae_best(standalone, stretched) = {rae_best:.4f}")
    print(f"   delta vs nb2103 mean_bag ({NB2103_K28_MEAN_BAG_REF:.4f}) = {delta_vs_mean_bag:+.4f}")
    print(f"   delta vs spec_ref       ({NB2103_K28_REF:.4f}) = {delta_vs_spec:+.4f}")
    print(f"   verdict = {verdict}")

    # ---- Save artefacts ----
    out_oof = DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy"
    out_te = DATA_PROCESSED / f"{TAG}_te_meanbag.npy"
    out_te_str = DATA_PROCESSED / f"{TAG}_te_meanbag_stretched.npy"
    np.save(out_oof, final_unb.astype(np.float32))
    np.save(out_te, pec50_te_mean.astype(np.float32))
    np.save(out_te_str, te_stretched_deploy.astype(np.float32))
    print(f"\n[save] OOF (253)               -> {out_oof}")
    print(f"[save] te mean-bag (513)       -> {out_te}")
    print(f"[save] te stretched (513)      -> {out_te_str}")

    # Deploy CSV only if beats nb2103 K=28 mean-bag (actual, the harder gate).
    csv_path = None
    if beats_actual:
        sub = pd.DataFrame({
            "SMILES": te_smiles_raw,
            "Molecule Name": te_df["name"].tolist(),
            "pEC50": te_stretched_deploy.astype(np.float64),
        })
        csv_path = SUBMISSIONS / f"{TAG}_multitask.csv"
        sub.to_csv(csv_path, index=False)
        print(f"[save] DEPLOY CSV (beats gate) -> {csv_path}")
    else:
        print(f"[save] DEPLOY CSV skipped (verdict={verdict})")

    summary.update({
        "n_join": int(len(smiles_join)),
        "n_train_only": int(len(smiles_to)),
        "n_full_train": int(len(y_full)),
        "n_test": int(X_te.shape[0]),
        "n_unb": int(n_unb),
        "feat_dim_stage3": int(X_full_s3.shape[1]),
        "stage1_rae_oof_meanbag": stage1_rae,
        "stage2_rae_oof_meanbag": stage2_rae,
        "stage3_rae_oof_full_meanbag": full_oof_rae,
        "rae_chemprop_aux_unb": rae_anchor,
        "rae_this_alone_unb": rae_unb,
        "stretch_grid": STRETCH_GRID,
        "best_stretch_s": best_s,
        "rae_stretched_unb_crossfit": rae_stretched_pooled,
        "rae_best_min_standalone_stretched": rae_best,
        "delta_best_vs_nb2103_mean_bag": delta_vs_mean_bag,
        "delta_best_vs_nb2103_spec_ref": delta_vs_spec,
        "verdict": verdict,
        "beats_nb2103_mean_bag": bool(beats_actual),
        "flat_vs_nb2103_mean_bag": bool(flat_actual),
        "beats_nb2103_spec_ref": bool(beats_spec),
        "deploy_csv": str(csv_path) if csv_path else None,
        "out_oof_path": str(out_oof),
        "out_te_path": str(out_te),
        "out_te_stretched_path": str(out_te_str),
        "wall_sec": round(time.time() - t0, 2),
    })

    out_summary = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] summary -> {out_summary}")
    print(f"[done] wall = {time.time() - t0:.1f}s")

    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_join", "n_full_train", "feat_dim_stage3",
        "stage1_rae_oof_meanbag", "stage2_rae_oof_meanbag", "stage3_rae_oof_full_meanbag",
        "rae_chemprop_aux_unb", "rae_this_alone_unb",
        "best_stretch_s", "rae_stretched_unb_crossfit", "rae_best_min_standalone_stretched",
        "delta_best_vs_nb2103_mean_bag", "delta_best_vs_nb2103_spec_ref",
        "verdict", "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
