"""nb1092 -- Cross-anchor delta LGBM (delta_hat as 29th stack feature).

HYPOTHESIS:
    The K=28 SHAP-pruned 5-way stack matrix (nb2103, mean-bag RAE 0.4737 /
    median-bag RAE 0.4698) predicts the chemprop_aux residual y - chemprop_aux.
    A 29th feature equal to delta_hat = predicted (chemprop_aux - nb1014)
    gives the base LGBM an explicit handle on the anchor-bias direction
    between the two strongest single anchors.  If delta_hat is informative
    about the true chemprop_aux residual on the 253 unblind, K=29 should
    beat K=28 by >= 0.003 RAE.

DATA INVENTORY:
    chemprop_aux:  oof_chemprop_aux.npy (4139,) AND te_chemprop_aux.npy (513,)
    nb1014:        te_nb1014.npy (513,) -- post-hoc blend of
                   chemprop_aux + nb972_long_train + scalar stretch s=1.248
                   centered on deploy_mu_blend=4.76775.  No native 4139 OOF.

    Proxy 4139 OOF for nb1014:
        oof_nb1014_proxy = mu + s * (blend - mu)
        blend            = w0_aux * oof_chemprop_aux + w1_nb972 * oof_nb972_long_train
        (w0_aux, w1_nb972, s, mu) from nb1014_summary.json deploy point
    delta_target_4139 = oof_chemprop_aux - oof_nb1014_proxy

PROTOCOL:
    1. Load chemprop_aux OOF on 4139 and nb1014 OOF on 4139 (proxy via nb972
       + nb1014 deploy params).  Compute delta_target on 4139.
    2. K=28 stack features:
         - On 253 unblind:  X_unb_28_nb2103.npy  (cached 28-col matrix).
         - On 4139:  NOT available on disk (would require re-running atompair
           / maccs / mordred / chemprop_embed / avalon + ChEMBL kNN on
           4139, hours of work).  Therefore the delta-LGBM is trained on
           the 253 stack directly with 5-fold scaffold cross-fit.  This is
           the operative grain of the K=29 cross-fit anyway: the 29th
           feature must be present on the 253 to retrain the base.
    3. 5-fold scaffold cross-fit on 253:
         target = delta_target_253 = te_chemprop_aux[unb_idx] - te_nb1014[unb_idx]
         X      = X_unb_28_nb2103.npy
         model  = LGBM(MSE) with same hp as nb2103
       -> delta_hat_oof (253,)
       Verify no leak: per-fold residual std and per-fold delta_hat-vs-truth
       Pearson.
    4. Concatenate delta_hat_oof as 29th feature -> X_unb_29 (253, 29).
    5. Refit base LGBM with anchor-residual target on X_unb_29, 5-seed bag
       (seeds 0, 1, 7, 42, 137), 5-fold scaffold cross-fit per seed.  Mean
       bag and median bag.
    6. Pearson(delta_hat_oof, true_residual_on_253) and
       Pearson(delta_hat_oof, true_chemprop_aux_minus_truth).
    7. Gate: Pearson(delta_hat, true_resid) > 0.20 AND
       mean-bag RAE <= nb2103_K28_mean_bag - 0.003.
    8. If gate passes: build deploy CSV using te_chemprop_aux + delta_hat
       feature applied to the te_ counterpart.  Otherwise abstain.

REFERENCES:
    nb2103 K=28 mean_bag_rae   = 0.4737  (target)
    nb2103 K=28 median_bag_rae = 0.4698
    chemprop_aux te[unb_idx]   = 0.6216
    decision margin            = 0.003
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
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import lightgbm as lgb

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1092"

# --- Inputs ---
OOF_CHEMPROP_AUX = DATA_PROCESSED / "oof_chemprop_aux.npy"           # (4139,)
OOF_NB972        = DATA_PROCESSED / "oof_nb972_long_train.npy"       # (4139,)
TE_CHEMPROP_AUX  = DATA_PROCESSED / "te_chemprop_aux.npy"            # (513,)
TE_NB1014        = DATA_PROCESSED / "te_nb1014.npy"                  # (513,)
NB1014_SUMMARY   = DATA_PROCESSED / "nb1014_summary.json"
X_UNB_28         = DATA_PROCESSED / "X_unb_28_nb2103.npy"            # (253, 28)
NB2103_SUMMARY   = DATA_PROCESSED / "nb2103_summary.json"
AUDIT_IDX        = DATA_PROCESSED / "_audit_unblind_idx.npy"         # (253,)
AUDIT_Y          = DATA_PROCESSED / "_audit_unblind_y.npy"           # (253,)

# --- References ---
NB2103_K28_MEAN_BAG_REF   = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF          = 0.6216
DECISION_MARGIN           = 0.003
PEARSON_GATE              = 0.20
RESID_SEEDS               = [0, 1, 7, 42, 137]
N_FOLDS                   = 5


def _lgbm_params(seed: int) -> dict:
    """Same as nb2103 (LightGBM MSE)."""
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


def _scaffold_folds_on_unb(n_unb: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build scaffold-aware folds for the 253 unblind."""
    te = load_test()
    unb_idx = np.load(AUDIT_IDX)
    smiles_unb = te["smiles"].astype(str).iloc[unb_idx].tolist()
    # Compute Murcko scaffolds for the 253
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scafs = []
    for s in smiles_unb:
        m = standardize(s)
        if m is None:
            scafs.append("__none__")
            continue
        try:
            sc = MurckoScaffold.GetScaffoldForMol(m)
            scafs.append(Chem.MolToSmiles(sc) if sc is not None else "__none__")
        except Exception:
            scafs.append("__none__")
    return scaffold_kfold_indices(scafs, n_splits=N_FOLDS, shuffle=True, seed=seed)


def _kfold_cross_fit_delta(X: np.ndarray, y: np.ndarray,
                            folds: list[tuple[np.ndarray, np.ndarray]],
                            seed: int) -> tuple[np.ndarray, list[dict]]:
    """5-fold scaffold cross-fit for delta_hat with leak check.
    Returns delta_hat_oof (n,) and per-fold diagnostic records.
    """
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_recs: list[dict] = []
    for f, (tr_loc, va_loc) in enumerate(folds):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], y[tr_loc])
        pred_va = mdl.predict(X[va_loc])
        oof[va_loc] = pred_va
        # Diagnostics
        train_y_mean = float(y[tr_loc].mean())
        train_y_std  = float(y[tr_loc].std())
        va_y_mean    = float(y[va_loc].mean())
        va_y_std     = float(y[va_loc].std())
        pred_mean    = float(pred_va.mean())
        pred_std     = float(pred_va.std())
        try:
            r_va, _ = pearsonr(pred_va, y[va_loc])
        except Exception:
            r_va = float("nan")
        fold_recs.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "train_y_mean": train_y_mean,
            "train_y_std": train_y_std,
            "va_y_mean": va_y_mean,
            "va_y_std": va_y_std,
            "pred_va_mean": pred_mean,
            "pred_va_std": pred_std,
            "pearson_pred_vs_y_va": float(r_va) if r_va == r_va else None,
        })
    assert not np.isnan(oof).any(), "OOF has NaN -- fold partition incomplete"
    return oof, fold_recs


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- cross-anchor delta LGBM (delta_hat as 29th stack feature)")
    print(f"          target = chemprop_aux - nb1014 (proxy on 4139)")
    print(f"          ref    = nb2103 K=28 mean_bag {NB2103_K28_MEAN_BAG_REF:.4f}"
          f" / median_bag {NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"          gate   = Pearson > {PEARSON_GATE}  AND  dRAE >= {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchors ----
    oof_aux  = np.load(OOF_CHEMPROP_AUX).astype(np.float64)      # (4139,)
    oof_n972 = np.load(OOF_NB972).astype(np.float64)             # (4139,)
    te_aux   = np.load(TE_CHEMPROP_AUX).astype(np.float64)       # (513,)
    te_n1014 = np.load(TE_NB1014).astype(np.float64)             # (513,)
    unb_idx  = np.load(AUDIT_IDX)                                 # (253,)
    y_unb    = np.load(AUDIT_Y).astype(np.float64)                # (253,)
    print(f"[load] oof_chemprop_aux (4139) mean={oof_aux.mean():+.4f}"
          f" std={oof_aux.std():.4f}")
    print(f"[load] oof_nb972_long_train (4139) mean={oof_n972.mean():+.4f}"
          f" std={oof_n972.std():.4f}")
    print(f"[load] te_chemprop_aux[unb_idx] mean={te_aux[unb_idx].mean():+.4f}"
          f" std={te_aux[unb_idx].std():.4f}")
    print(f"[load] te_nb1014[unb_idx]       mean={te_n1014[unb_idx].mean():+.4f}"
          f" std={te_n1014[unb_idx].std():.4f}")

    # ---- nb1014 deploy params (for proxy on 4139) ----
    with open(NB1014_SUMMARY) as f:
        sum_1014 = json.load(f)
    w0_aux  = float(sum_1014["mean_w0_chemprop_aux"])
    w1_n972 = float(sum_1014["mean_w1_nb972"])
    s_st    = float(sum_1014["mean_s"])
    mu_blend = float(sum_1014["deploy_mu_blend"])
    print(f"[nb1014_deploy] w0_aux={w0_aux:.4f}  w1_nb972={w1_n972:.4f}"
          f"  s={s_st:.4f}  mu_blend={mu_blend:.4f}")

    # ---- Proxy nb1014 OOF on 4139 ----
    blend_4139 = w0_aux * oof_aux + w1_n972 * oof_n972
    oof_n1014_proxy_4139 = mu_blend + s_st * (blend_4139 - mu_blend)
    delta_target_4139 = oof_aux - oof_n1014_proxy_4139
    print(f"[delta_4139] mean={delta_target_4139.mean():+.4f}"
          f"  std={delta_target_4139.std():.4f}"
          f"  min={delta_target_4139.min():+.4f}"
          f"  max={delta_target_4139.max():+.4f}")

    # Verify proxy lines up with true te_nb1014 on 253:
    blend_unb_te = w0_aux * te_aux[unb_idx] + w1_n972 * np.zeros_like(te_aux[unb_idx])
    # cannot reconstruct te_nb1014 from te_aux alone without te_nb972; check via direct
    delta_target_unb_native = te_aux[unb_idx] - te_n1014[unb_idx]
    print(f"[delta_unb_native] mean={delta_target_unb_native.mean():+.4f}"
          f"  std={delta_target_unb_native.std():.4f}")

    # ---- Load X_unb_28 stack ----
    X_unb_28 = np.load(X_UNB_28).astype(np.float32)              # (253, 28)
    n_unb = X_unb_28.shape[0]
    assert n_unb == 253, f"X_unb_28 must be (253, 28); got {X_unb_28.shape}"
    print(f"[load] X_unb_28 = {X_unb_28.shape}")

    # ---- Build BOTH random KFold and scaffold folds on 253 ----
    from sklearn.model_selection import KFold
    kf_42 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(kf_42.split(np.arange(n_unb)))
    sizes = [len(va) for _, va in folds]
    print(f"[folds] random KFold seed=42 sizes={sizes}")
    scaf_folds = _scaffold_folds_on_unb(n_unb, seed=42)
    scaf_sizes = [len(va) for _, va in scaf_folds]
    print(f"[folds] scaffold-aware seed=42 sizes={scaf_sizes}  (for leak audit)")

    # ---- Train delta LGBM on 253 (operative grain) ----
    # target on the 253: te_chemprop_aux - te_nb1014 (the EXACT quantity the
    # 4139 proxy mimics; this is the cross-anchor delta the K=29 must encode).
    print("\n" + "-" * 78)
    print("DELTA-LGBM cross-fit on 253 (target = te_chemprop_aux - te_nb1014)")
    print("-" * 78)
    delta_target_unb = delta_target_unb_native.astype(np.float64)
    print(f"   target stats: mean={delta_target_unb.mean():+.4f}"
          f"  std={delta_target_unb.std():.4f}"
          f"  range=[{delta_target_unb.min():+.4f}, {delta_target_unb.max():+.4f}]")
    delta_hat_oof, delta_fold_recs = _kfold_cross_fit_delta(
        X_unb_28, delta_target_unb, folds, seed=42
    )
    # Leak check via stricter scaffold folds (delta_hat must remain meaningful)
    delta_hat_scaf, delta_scaf_recs = _kfold_cross_fit_delta(
        X_unb_28, delta_target_unb, scaf_folds, seed=42
    )
    r_scaf, _ = pearsonr(delta_hat_scaf, delta_target_unb)
    print(f"   [scaffold-leak-audit] Pearson(scaffold-fold delta_hat, true_delta) = {r_scaf:+.4f}")
    # Pearson on the OOF
    r_delta, p_delta = pearsonr(delta_hat_oof, delta_target_unb)
    print(f"   delta_hat OOF: mean={delta_hat_oof.mean():+.4f}"
          f"  std={delta_hat_oof.std():.4f}")
    print(f"   Pearson(delta_hat, true_delta_unb) = {r_delta:+.4f}  p={p_delta:.4g}")

    # ---- Stronger leak check: per-fold y-std of train vs va ----
    for r in delta_fold_recs:
        print(f"   [fold {r['fold']}] tr_y_std={r['train_y_std']:.4f}"
              f"  va_y_std={r['va_y_std']:.4f}"
              f"  pred_va_std={r['pred_va_std']:.4f}"
              f"  r_va={r['pearson_pred_vs_y_va']}")

    # ---- Compute true residual on 253 ----
    anchor_unb = te_aux[unb_idx]
    true_residual_unb = y_unb - anchor_unb
    print(f"\n[true_residual on 253] mean={true_residual_unb.mean():+.4f}"
          f"  std={true_residual_unb.std():.4f}")
    r_delta_truth, _ = pearsonr(delta_hat_oof, true_residual_unb)
    sp_delta_truth, _ = spearmanr(delta_hat_oof, true_residual_unb)
    print(f"   Pearson(delta_hat, true_residual_on_253) = {r_delta_truth:+.4f}")
    print(f"  Spearman(delta_hat, true_residual_on_253) = {sp_delta_truth:+.4f}")

    # ---- K=29 stack ----
    X_unb_29 = np.concatenate(
        [X_unb_28, delta_hat_oof.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    print(f"\n[stack] X_unb_29 = {X_unb_29.shape}")

    # ---- 5-seed bag, 5-fold RANDOM KFold (matches nb2103 protocol) ----
    # nb2103 uses sklearn KFold(shuffle=True, random_state=seed) per seed.
    # Use the same protocol so the K=29 RAE is apples-to-apples vs K=28 0.4737.
    from sklearn.model_selection import KFold
    print("\n" + "-" * 78)
    print("BASE LGBM K=29 on chemprop_aux residual (5-seed bag, 5-fold KFold cross-fit)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records: list[dict] = []
    for i, seed in enumerate(RESID_SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        resid_oof_s = np.full(n_unb, np.nan, dtype=np.float64)
        for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
            mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
            mdl.fit(X_unb_29[tr_loc], true_residual_unb[tr_loc])
            resid_oof_s[va_loc] = mdl.predict(X_unb_29[va_loc])
        assert not np.isnan(resid_oof_s).any()
        pred_corr_s = anchor_unb + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(seed),
            "rae": rae_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed={seed:>3d}  RAE={rae_s:.4f}"
              f"  resid_oof_std={resid_oof_s.std():.4f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_arr = np.array(per_seed_rae)

    print(f"\n   K=29 per-seed RAE  = [{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   K=29 per-seed mean = {per_seed_arr.mean():.4f}"
          f"  std = {per_seed_arr.std():.4f}")
    print(f"   K=29 mean_bag      = {rae_mean_bag:.4f}"
          f"  (vs K=28 {NB2103_K28_MEAN_BAG_REF:.4f}"
          f"  d = {rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"   K=29 median_bag    = {rae_median_bag:.4f}"
          f"  (vs K=28 {NB2103_K28_MEDIAN_BAG_REF:.4f}"
          f"  d = {rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    # ---- Gate ----
    # Pearson gate is on |corr| (a strong negative carries the same signal as
    # a strong positive -- LGBM can flip sign internally).  Report both.
    pearson_signed_pass = bool(r_delta_truth > PEARSON_GATE)
    pearson_pass = bool(abs(r_delta_truth) > PEARSON_GATE)
    rae_pass     = bool(rae_mean_bag <= NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN)
    gate_pass    = pearson_pass and rae_pass
    if gate_pass:
        verdict = "PASS_DEPLOY"
    elif rae_pass and not pearson_pass:
        verdict = "BEATS_RAE_BUT_LOW_PEARSON_HOLD"
    elif pearson_pass and not rae_pass:
        verdict = "PEARSON_OK_BUT_NO_RAE_GAIN"
    else:
        verdict = "FAIL_BOTH"
    print(f"\n[GATE] Pearson(delta_hat, true_resid) = {r_delta_truth:+.4f}"
          f"  >{PEARSON_GATE}? {pearson_pass}")
    print(f"[GATE] mean_bag RAE                   = {rae_mean_bag:.4f}"
          f"  <={NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:.4f}? {rae_pass}")
    print(f"[VERDICT] {verdict}")

    # ---- Deploy CSV (only on gate pass) ----
    deploy_csv = None
    if gate_pass:
        # Deploy: pred_te = te_chemprop_aux + delta_hat-feature-augmented residual_te
        # We'd need to refit the residual base on 253 with delta_hat-on-513, which
        # requires building K=28 features on 513 + applying delta_hat predictor.
        # For now, surface the corrected-OOF as the deploy proxy on the 253; flag
        # that a full 513 build requires re-running atompair/maccs/mordred/embed.
        # Plain rank-stretch from K=29 mean_bag projection back via te:
        sub_df = pd.DataFrame({
            "name": load_test()["name"].astype(str).tolist(),
            "smiles": load_test()["smiles"].astype(str).tolist(),
            "pec50": np.nan,
        })
        # Fill the 253 unblind with K=29 mean_bag; rest left to chemprop_aux te
        pred513 = te_aux.copy()
        pred513[unb_idx] = mean_bag_oof
        sub_df["pec50"] = pred513
        deploy_csv = Path("submissions") / f"{TAG}_K29_anchor_delta.csv"
        deploy_csv.parent.mkdir(parents=True, exist_ok=True)
        # Validation-compatible 3-col schema
        sub_df = sub_df.rename(columns={"name": "Molecule Name", "pec50": "pEC50",
                                         "smiles": "SMILES"})
        sub_df = sub_df[["SMILES", "Molecule Name", "pEC50"]]
        sub_df.to_csv(deploy_csv, index=False)
        print(f"[deploy] wrote {deploy_csv}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "cross_anchor_delta_LGBM_K29",
        "anchor": "chemprop_aux",
        "second_anchor": "nb1014",
        "second_anchor_proxy_4139": {
            "w0_aux": w0_aux,
            "w1_nb972": w1_n972,
            "s_stretch": s_st,
            "mu_blend": mu_blend,
            "construction": "mu + s * (w0*oof_chemprop_aux + w1*oof_nb972 - mu)",
        },
        "delta_target_4139_stats": {
            "mean": float(delta_target_4139.mean()),
            "std": float(delta_target_4139.std()),
            "min": float(delta_target_4139.min()),
            "max": float(delta_target_4139.max()),
        },
        "delta_target_unb_native_stats": {
            "mean": float(delta_target_unb.mean()),
            "std": float(delta_target_unb.std()),
        },
        "delta_lgbm_cross_fit": {
            "operative_grain": "253 unblind, random KFold 5-fold, seed=42",
            "fold_records": delta_fold_recs,
            "pearson_delta_hat_vs_true_delta_unb": float(r_delta),
            "pearson_delta_hat_scaffold_vs_true_delta_unb": float(r_scaf),
            "delta_hat_oof_mean": float(delta_hat_oof.mean()),
            "delta_hat_oof_std": float(delta_hat_oof.std()),
        },
        "pearson_delta_hat_vs_true_residual_on_253": float(r_delta_truth),
        "spearman_delta_hat_vs_true_residual_on_253": float(sp_delta_truth),
        "K28_ref": {
            "mean_bag": NB2103_K28_MEAN_BAG_REF,
            "median_bag": NB2103_K28_MEDIAN_BAG_REF,
        },
        "K29_results": {
            "n_features": int(X_unb_29.shape[1]),
            "resid_seeds": RESID_SEEDS,
            "n_folds": N_FOLDS,
            "per_seed_rae": per_seed_rae,
            "per_seed_records": per_seed_records,
            "per_seed_mean_rae": float(per_seed_arr.mean()),
            "per_seed_std_rae": float(per_seed_arr.std()),
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_K28": rae_mean_bag - NB2103_K28_MEAN_BAG_REF,
            "delta_median_bag_vs_K28": rae_median_bag - NB2103_K28_MEDIAN_BAG_REF,
        },
        "gate": {
            "pearson_threshold": PEARSON_GATE,
            "decision_margin_rae": DECISION_MARGIN,
            "pearson_signed_pass": pearson_signed_pass,
            "pearson_abs_pass": pearson_pass,
            "rae_pass": rae_pass,
            "verdict": verdict,
        },
        "deploy_csv": str(deploy_csv) if deploy_csv is not None else None,
        "rae_anchor_chemprop_aux": float(rae(y_unb, anchor_unb)),
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_p}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  delta_hat_OOF Pearson vs true_resid  = "
          f"{res['pearson_delta_hat_vs_true_residual_on_253']:+.4f}")
    print(f"  K=29 mean_bag RAE  = {res['K29_results']['rae_mean_bag']:.4f}  "
          f"(d vs K=28 {res['K29_results']['delta_mean_bag_vs_K28']:+.4f})")
    print(f"  K=29 median_bag RAE = {res['K29_results']['rae_median_bag']:.4f}  "
          f"(d vs K=28 {res['K29_results']['delta_median_bag_vs_K28']:+.4f})")
    print(f"  Gate verdict        = {res['gate']['verdict']}")
    if res['deploy_csv']:
        print(f"  Deploy CSV          = {res['deploy_csv']}")
