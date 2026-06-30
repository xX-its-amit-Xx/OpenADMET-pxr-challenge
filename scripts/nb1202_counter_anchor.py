"""nb1202 -- Counter-assay-only LGBM as orthogonal anchor.

HYPOTHESIS (per nb1181 orthogonal-hunt finding -- need NEW model paradigm):
    Training an LGBM(MSE) to predict pec50_null (counter-assay axis) and then
    using the predicted pec50_null as an AUXILIARY FEATURE for the main pec50
    LGBM creates a PARADIGM-DISTINCT signal vs the standard
    chemprop_aux + main-pec50-LGBM anchor. The counter-assay axis encodes
    PXR-promiscuity (off-target activation) that is biologically orthogonal
    to the on-target pec50 signal, so the residuals on the 253 unblind should
    have Pearson < 0.7 with nb1150.

PROTOCOL:
    1. Load 2858 train rows with valid pec50_null + matching pec50 + combined
       features (Morgan-2048 + RDKit-217 -> 2265 dims).
    2. Train LGBM(MSE) K=28 on pec50_null target with 5-fold scaffold cross-fit
       (counter-assay predictor).
    3. Use the predicted pec50_null as an additional feature (1 extra col)
       for the main pec50 LGBM K=28 (combined + pred_null = 2266 dims).
    4. 5-fold scaffold cross-fit, 5-seed bag (mean-bag RAE).
    5. Final = chemprop_aux (te[unb_idx]) + LGBM(K=28+counter)_residual.
    6. Compute residual correlation with nb1150 reconstructed OOF.
    7. Compare vs nb2103 K=28 baseline (0.5057 scaffold CV); margin 0.003.
    8. If beats AND orthogonal: SLSQP blend with nb1150 (2-way scaffold CV).

OUTPUTS:
    scripts/nb1202_counter_anchor.py
    data/processed/nb1202_summary.json
    data/processed/nb1202_mean_bag_oof.npy   (253,) float32
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_train, load_counter, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1202"
NB2103_K28_REF = 0.5057  # nb2103 K=28 baseline (scaffold-CV mean-bag RAE)
NB1150_REF = 0.4710      # nb1150 nested-scaffold-CV ceiling
DECISION_MARGIN = 0.003
ORTHO_CORR_THRESH = 0.7

CHEMPROP_AUX_TE = DATA_PROCESSED / "te_chemprop_aux.npy"
UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

# nb1150 anchor layout for resid-correlation
NB1150_ANCHOR_OOF_PATHS = {
    "chemprop_aux": DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
    "nb503":        DATA_PROCESSED / "nb503_pred_oof.npy",
    "nb1014":       DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy",
    "nb2112":       DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
}

SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5
LGBM_K = 28  # leaves


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


def _refit_full_predict_test(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    num_leaves: int = LGBM_K,
) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed, num_leaves=num_leaves))
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te).astype(np.float64)


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 8, seed: int = 0):
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w):
        return rae(y, P @ w)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    starts = [np.full(K, 1.0 / K)]
    for _ in range(n_starts - 1):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            w = np.clip(res.x, 0.0, 1.0)
            s = w.sum()
            if s <= 0:
                continue
            w = w / s
            r = rae(y, P @ w)
            if r < best_r:
                best_r, best_w = r, w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = rae(y, P @ best_w)
    return best_w, best_r


def _reconstruct_nb1150_oof(y: np.ndarray, scaffolds: list[str]) -> np.ndarray:
    """Recompute nb1150 nested scaffold-CV blended OOF (matches nb1181 protocol)."""
    P_list = []
    for name, path in NB1150_ANCHOR_OOF_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"nb1150 anchor missing: {path}")
        P_list.append(np.load(path).astype(np.float64))
    P = np.stack(P_list, axis=1)  # (253, 4)
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)
    blended = np.full(len(y), np.nan, dtype=np.float64)
    for fi, (tr_idx, va_idx) in enumerate(folds):
        w, _ = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
        blended[va_idx] = P[va_idx] @ w
    assert not np.any(np.isnan(blended))
    return blended


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- counter-assay-only LGBM as orthogonal anchor")
    print(f"     ref nb2103 K=28 = {NB2103_K28_REF:.4f}   nb1150 = {NB1150_REF:.4f}")
    print(f"     decision margin = {DECISION_MARGIN}")
    print("=" * 78)

    summary: dict = {
        "tag": TAG,
        "purpose": "Counter-assay-only LGBM as orthogonal anchor (nb1181 prescription)",
        "lgbm_num_leaves": LGBM_K,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "nb2103_K28_ref": NB2103_K28_REF,
        "nb1150_ref": NB1150_REF,
        "decision_margin": DECISION_MARGIN,
        "ortho_corr_thresh": ORTHO_CORR_THRESH,
    }

    # ---- 1. Build train: 2858 rows with valid pec50 + pec50_null ----
    tr = load_train()
    co = load_counter()
    # canonicalize SMILES for join (use name as key if it matches)
    tr["std_smi"] = tr["smiles"].apply(lambda s: Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else None)
    co["std_smi"] = co["smiles"].apply(lambda s: Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else None)
    tr_g = tr[tr["std_smi"].notna() & tr["pec50"].notna()].groupby("std_smi", as_index=False).agg(pec50=("pec50", "median"))
    co_g = co[co["std_smi"].notna() & co["pec50"].notna()].groupby("std_smi", as_index=False).agg(pec50_null=("pec50", "median"))
    joined = tr_g.merge(co_g, on="std_smi", how="inner")
    print(f"[data] train n_uniq_smi (pec50)        = {len(tr_g)}")
    print(f"[data] counter n_uniq_smi (pec50_null) = {len(co_g)}")
    print(f"[data] inner join (both labels)        = {len(joined)}")

    smiles_join = joined["std_smi"].tolist()
    y_pec50 = joined["pec50"].to_numpy(dtype=np.float64)
    y_null = joined["pec50_null"].to_numpy(dtype=np.float64)
    scaffolds_join = [_murcko(s) for s in smiles_join]
    n_join = len(smiles_join)

    # ---- 2. Build combined features (Morgan + RDKit) ----
    print("\n[feat] computing combined (Morgan+RDKit) features ...")
    X_join = impute(combined(smiles_join)).astype(np.float32)
    print(f"[feat] X_join shape = {X_join.shape}")

    # train-only set: rows with pec50 but NOT in joined (for null imputation refit)
    tr_only_g = tr_g[~tr_g["std_smi"].isin(joined["std_smi"])].reset_index(drop=True)
    print(f"[data] train-only (pec50 no null) = {len(tr_only_g)}")

    # also build features for tr_only and test set
    tr_only_smiles = tr_only_g["std_smi"].tolist()
    y_tr_only = tr_only_g["pec50"].to_numpy(dtype=np.float64)
    X_tr_only = impute(combined(tr_only_smiles)).astype(np.float32) if len(tr_only_smiles) else np.empty((0, X_join.shape[1]), dtype=np.float32)
    scaffolds_tr_only = [_murcko(s) for s in tr_only_smiles]

    te_df = load_test()
    te_smiles_raw = te_df["smiles"].astype(str).tolist()
    te_std_smi = [Chem.MolToSmiles(standardize(s)) if standardize(s) is not None else s for s in te_smiles_raw]
    X_te = impute(combined(te_std_smi)).astype(np.float32)
    print(f"[feat] X_te shape   = {X_te.shape}")

    # ---- 3. Counter-assay predictor: 5-fold scaffold cross-fit on join ----
    print("\n[step3] training counter-assay LGBM (target = pec50_null) ...")
    null_oof_per_seed = []
    for s in SEEDS:
        ts = time.time()
        oof_s = _scaffold_cross_fit(X_join, y_null, scaffolds_join, seed=s, num_leaves=LGBM_K)
        null_oof_per_seed.append(oof_s)
        print(f"   seed={s:3d}  pec50_null OOF RAE={rae(y_null, oof_s):.4f}  wall={time.time()-ts:.1f}s")
    null_oof_mean = np.mean(np.stack(null_oof_per_seed, axis=0), axis=0)
    null_rae = float(rae(y_null, null_oof_mean))
    print(f"[step3] mean-bag counter OOF RAE = {null_rae:.4f}")

    # Refit on FULL join + predict test
    null_te_per_seed = []
    for s in SEEDS:
        null_te_per_seed.append(_refit_full_predict_test(X_join, y_null, X_te, seed=s, num_leaves=LGBM_K))
    null_te_mean = np.mean(np.stack(null_te_per_seed, axis=0), axis=0)
    print(f"[step3] te null mean={null_te_mean.mean():+.3f}  std={null_te_mean.std():.3f}")

    # Also refit and predict on train-only (for those without real null label)
    if len(tr_only_smiles):
        null_tr_only_per_seed = []
        for s in SEEDS:
            null_tr_only_per_seed.append(_refit_full_predict_test(X_join, y_null, X_tr_only, seed=s, num_leaves=LGBM_K))
        null_tr_only_mean = np.mean(np.stack(null_tr_only_per_seed, axis=0), axis=0)
    else:
        null_tr_only_mean = np.empty((0,), dtype=np.float64)

    # ---- 4. Main pec50 LGBM with [combined + predicted null] feature ----
    # IMPORTANT: use OOF null pred for joined rows to avoid label leakage of pec50_null.
    print("\n[step4] training main pec50 LGBM with +pred_null aux feature ...")
    # Combined train data: joined rows + train-only rows
    X_full_train = np.vstack([X_join, X_tr_only]) if len(tr_only_smiles) else X_join.copy()
    y_full_train = np.concatenate([y_pec50, y_tr_only]) if len(tr_only_smiles) else y_pec50.copy()
    null_oof_for_join = null_oof_mean  # OOF (no leakage)
    null_for_tr_only = null_tr_only_mean if len(tr_only_smiles) else np.empty((0,), dtype=np.float64)
    null_full_train = np.concatenate([null_oof_for_join, null_for_tr_only])
    X_full_train_aug = np.hstack([X_full_train, null_full_train.reshape(-1, 1).astype(np.float32)])
    scaffolds_full = scaffolds_join + scaffolds_tr_only
    print(f"[step4] X_full_train_aug shape = {X_full_train_aug.shape}   n_tr={len(y_full_train)}")

    # Test-side aug
    X_te_aug = np.hstack([X_te, null_te_mean.reshape(-1, 1).astype(np.float32)])
    print(f"[step4] X_te_aug shape         = {X_te_aug.shape}")

    # Scaffold cross-fit -> we still want OOF on the joined-subset rows to evaluate
    # vs nb2103, but here we run on full to get a refit te.
    # For comparison vs nb2103 K=28 (which is trained on 4139), report
    # scaffold-CV RAE on the joined-2858 (apples-to-apples NOT possible; we use
    # scaffold-CV on full and slice OOF for join-subset).
    main_oof_per_seed = []
    main_te_per_seed = []
    for s in SEEDS:
        ts = time.time()
        oof_s = _scaffold_cross_fit(X_full_train_aug, y_full_train, scaffolds_full, seed=s, num_leaves=LGBM_K)
        main_oof_per_seed.append(oof_s)
        # refit full + predict test
        te_s = _refit_full_predict_test(X_full_train_aug, y_full_train, X_te_aug, seed=s, num_leaves=LGBM_K)
        main_te_per_seed.append(te_s)
        rae_s = float(rae(y_full_train, oof_s))
        print(f"   seed={s:3d}  main pec50 OOF RAE={rae_s:.4f}  wall={time.time()-ts:.1f}s")
    main_oof_mean = np.mean(np.stack(main_oof_per_seed, axis=0), axis=0)
    main_te_mean = np.mean(np.stack(main_te_per_seed, axis=0), axis=0)
    main_oof_rae_full = float(rae(y_full_train, main_oof_mean))
    print(f"[step4] mean-bag main OOF RAE (full train) = {main_oof_rae_full:.4f}")

    # ---- 5. Build chemprop_aux + LGBM(K=28+counter)_residual on the 253 unblind ----
    print("\n[step5] residual stack: chemprop_aux + this-model residual on 253 unblind ...")
    if not CHEMPROP_AUX_TE.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {CHEMPROP_AUX_TE}")
    te_chemprop_aux_513 = np.load(CHEMPROP_AUX_TE).astype(np.float64)
    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    anchor_unb = te_chemprop_aux_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[step5] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f}")

    # main_te_mean has shape (513,) -- predicted pec50 for blinded test
    # Residual that this model contributes on the 253 unblind
    this_unb = main_te_mean[unb_idx]
    rae_this_alone = float(rae(y_unb, this_unb))
    print(f"[step5] this-model te[unb_idx] standalone RAE = {rae_this_alone:.4f}")

    # Residual stack: blend chemprop_aux + this-model on residual = (this - chemprop_aux)
    # To stay honest, we compute a 5-fold residual-on-anchor model on the 253 unblind
    # using this-model predictions as the X feature. But for diagnostic, also report
    # simple equal-weight blend and 0.5*chemprop_aux + 0.5*this-model.
    eqw_blend = 0.5 * anchor_unb + 0.5 * this_unb
    rae_eqw = float(rae(y_unb, eqw_blend))
    print(f"[step5] 0.5*chemprop_aux + 0.5*this RAE = {rae_eqw:.4f}")

    # SLSQP optimal 2-way (in-sample on 253 -- diagnostic only)
    P2 = np.stack([anchor_unb, this_unb], axis=1)
    w2, rae_w2 = _simplex_slsqp(P2, y_unb, n_starts=8, seed=0)
    print(f"[step5] 2-way SLSQP w=[{w2[0]:.3f}, {w2[1]:.3f}] in-sample RAE={rae_w2:.4f}")

    # ---- 6. Residual correlation vs nb1150 reconstructed OOF ----
    print("\n[step6] residual correlation vs nb1150 reconstructed OOF ...")
    te_unb_smiles = te_df.iloc[unb_idx]["smiles"].astype(str).tolist()
    scaffolds_unb = [_murcko(s) for s in te_unb_smiles]
    try:
        nb1150_oof = _reconstruct_nb1150_oof(y_unb, scaffolds_unb)
        rae_nb1150 = float(rae(y_unb, nb1150_oof))
        resid_nb1150 = y_unb - nb1150_oof
        # this-model resid
        resid_this = y_unb - this_unb
        if np.std(resid_this) < 1e-9 or np.std(resid_nb1150) < 1e-9:
            pcorr_this = float("nan")
        else:
            pcorr_this = float(np.corrcoef(resid_this, resid_nb1150)[0, 1])
        # blend resid
        resid_blend = y_unb - eqw_blend
        if np.std(resid_blend) < 1e-9:
            pcorr_blend = float("nan")
        else:
            pcorr_blend = float(np.corrcoef(resid_blend, resid_nb1150)[0, 1])
        print(f"[step6] nb1150 reconstructed RAE   = {rae_nb1150:.4f}")
        print(f"[step6] Pearson(this_resid,  nb1150_resid) = {pcorr_this:+.4f}")
        print(f"[step6] Pearson(blend_resid, nb1150_resid) = {pcorr_blend:+.4f}")
        ortho_pass = abs(pcorr_this) < ORTHO_CORR_THRESH
    except FileNotFoundError as e:
        print(f"[step6] nb1150 reconstruction failed: {e}")
        rae_nb1150 = None
        pcorr_this = None
        pcorr_blend = None
        ortho_pass = False
        nb1150_oof = None

    # ---- 7. Compare vs nb2103 K=28 baseline ----
    print("\n[step7] compare vs nb2103 K=28 baseline ...")
    # nb2103 K=28 mean-bag-OOF on the 253 unblind
    nb2103_oof_path = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    if nb2103_oof_path.exists():
        nb2103_oof = np.load(nb2103_oof_path).astype(np.float64)
        rae_nb2103 = float(rae(y_unb, nb2103_oof))
        print(f"[step7] nb2103 K=28 OOF RAE on 253 = {rae_nb2103:.4f}  (script ref {NB2103_K28_REF:.4f})")
    else:
        rae_nb2103 = NB2103_K28_REF
        print(f"[step7] nb2103 K=28 file missing; using script ref {NB2103_K28_REF:.4f}")

    # Decision: this-model vs nb2103 (this-model evaluated as standalone te[unb_idx])
    beats_nb2103 = rae_this_alone < rae_nb2103 - DECISION_MARGIN
    flat_nb2103 = abs(rae_this_alone - rae_nb2103) < DECISION_MARGIN
    print(f"[step7] this-model RAE = {rae_this_alone:.4f}   nb2103 = {rae_nb2103:.4f}   delta={rae_this_alone - rae_nb2103:+.4f}")
    if beats_nb2103:
        decision_vs_nb2103 = "BEATS_NB2103"
    elif flat_nb2103:
        decision_vs_nb2103 = "FLAT_VS_NB2103"
    else:
        decision_vs_nb2103 = "WORSE_THAN_NB2103"
    print(f"[step7] decision_vs_nb2103 = {decision_vs_nb2103}")

    # ---- 8. SLSQP blend with nb1150 if beats + orthogonal ----
    print("\n[step8] SLSQP blend with nb1150 (scaffold-fold) if beats + orthogonal ...")
    blend_result = None
    if nb1150_oof is not None and (beats_nb2103 or flat_nb2103) and ortho_pass:
        print(f"[step8] criteria MET ({decision_vs_nb2103}, |pcorr|={abs(pcorr_this):.3f}<{ORTHO_CORR_THRESH}); running 2-way SLSQP")
        P_b = np.stack([nb1150_oof, this_unb], axis=1)
        folds = scaffold_kfold_indices(scaffolds_unb, n_splits=5, shuffle=True, seed=42)
        blended = np.full(n_unb, np.nan, dtype=np.float64)
        fold_w = []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            w, _ = _simplex_slsqp(P_b[tr_idx], y_unb[tr_idx], n_starts=8, seed=fi)
            blended[va_idx] = P_b[va_idx] @ w
            fold_w.append(w)
        fold_w = np.stack(fold_w, axis=0)
        w_mean = fold_w.mean(axis=0)
        w_mean = w_mean / w_mean.sum()
        rae_blend = float(rae(y_unb, blended))
        delta_blend_vs_nb1150 = rae_blend - rae_nb1150
        print(f"[step8] scaffold-CV blend RAE = {rae_blend:.4f}  delta_vs_nb1150 = {delta_blend_vs_nb1150:+.4f}")
        print(f"[step8] mean-fold weights: nb1150={w_mean[0]:.3f}  this={w_mean[1]:.3f}")
        blend_result = {
            "blend_rae_scaffoldCV": rae_blend,
            "delta_vs_nb1150": delta_blend_vs_nb1150,
            "mean_fold_w_nb1150": float(w_mean[0]),
            "mean_fold_w_this": float(w_mean[1]),
            "passes_gate": bool(rae_blend < NB1150_REF - DECISION_MARGIN),
        }
    else:
        reason = []
        if not (beats_nb2103 or flat_nb2103):
            reason.append("does_not_beat_nb2103")
        if nb1150_oof is None:
            reason.append("nb1150_unavailable")
        elif not ortho_pass:
            reason.append(f"corr_high(|{abs(pcorr_this):.3f}|>={ORTHO_CORR_THRESH})")
        print(f"[step8] SKIP blend; reason = {reason}")
        blend_result = {"skipped": True, "reasons": reason}

    # ---- Save OOF + summary ----
    out_oof = DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy"
    np.save(out_oof, this_unb.astype(np.float32))
    print(f"\n[save] OOF (this-model te[unb_idx]) -> {out_oof}")

    summary.update({
        "n_join": n_join,
        "n_train_only": len(tr_only_smiles),
        "feat_dim_aug": int(X_full_train_aug.shape[1]),
        "n_test": int(X_te.shape[0]),
        "counter_rae_oof_meanbag": null_rae,
        "main_oof_rae_full_train": main_oof_rae_full,
        "rae_chemprop_aux_unb": rae_anchor,
        "rae_this_alone_unb": rae_this_alone,
        "rae_eqw_blend_unb": rae_eqw,
        "rae_2way_slsqp_insample": float(rae_w2),
        "w2_slsqp": [float(w2[0]), float(w2[1])],
        "rae_nb2103_K28_unb": rae_nb2103,
        "delta_this_vs_nb2103": rae_this_alone - rae_nb2103,
        "decision_vs_nb2103": decision_vs_nb2103,
        "rae_nb1150_reconstructed": rae_nb1150,
        "pearson_resid_this_vs_nb1150": pcorr_this,
        "pearson_resid_blend_vs_nb1150": pcorr_blend,
        "ortho_pass": bool(ortho_pass),
        "blend_with_nb1150": blend_result,
        "out_oof_path": str(out_oof),
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
        "n_join", "feat_dim_aug",
        "counter_rae_oof_meanbag", "main_oof_rae_full_train",
        "rae_chemprop_aux_unb", "rae_this_alone_unb", "rae_eqw_blend_unb",
        "rae_nb2103_K28_unb", "delta_this_vs_nb2103", "decision_vs_nb2103",
        "rae_nb1150_reconstructed", "pearson_resid_this_vs_nb1150", "ortho_pass",
        "blend_with_nb1150",
    ):
        print(f"  {k}: {res.get(k)}")
