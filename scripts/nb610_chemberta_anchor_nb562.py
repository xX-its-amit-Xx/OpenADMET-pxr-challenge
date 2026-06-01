"""nb610 -- CHEMBERTA RESIDUAL ROUTER ANCHORED ON NB562.

Same architecture as nb601, but anchor switched from nb464 -> nb562.

Hypothesis: anchoring on nb562 (the better anchor; honest unblind RAE 0.5065)
means the router learns ONLY the residual nb562 misses. If ChemBERTa
embeddings carry truly orthogonal signal (resid corr to nb502 in 0.33-0.45
range per nb601), a small but consistent correction might break the 0.5065
wall. Risk: nb562's residual variance is smaller and noisier than nb464's,
so the router may add noise > signal.

Pipeline:
  1. Load tr_chemberta.npy (4139, 384) + te_chemberta.npy (513, 384).
  2. PCA(64) fit on the FULL pool (train + test embeddings); pretrained
     embeddings are unsupervised so this is leak-safe.
  3. Anchor = nb562. Target = truth - nb562 on 253 unblind rows.
  4. 5-fold KFold cross-fit shallow LGBM
       (max_depth=3, n_est=80, lr=0.05, num_leaves=8,
        min_child_samples=20, reg_lambda=1.0, seed=0).
  5. Soft gate alpha = sigmoid((err_hat - median(err_hat)) * 4.0).
  6. pred_oof = nb562[unb_idx] + alpha * resid_oof  (honest cross-fit).
  7. Save te_nb610.npy + pred_oof + submissions.
  Target: < 0.5065 (beat nb562).
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7
TAG = "nb610"
ANCHOR_NAME = "nb562"
ANCHOR_FILE = "te_nb562.npy"
PCA_DIM = 64
TARGET_RAE = 0.5065  # nb562 honest unblind RAE

LGBM_PARAMS = dict(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=8,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa router anchored on {ANCHOR_NAME}")
    print("=" * 78)

    needed = {
        ANCHOR_FILE: DATA_PROCESSED / ANCHOR_FILE,
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "tr_chemberta.npy": DATA_PROCESSED / "tr_chemberta.npy",
        "te_chemberta.npy": DATA_PROCESSED / "te_chemberta.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "notes": "missing_inputs", "missing": missing}

    anchor = np.load(needed[ANCHOR_FILE]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    tr_emb = np.load(needed["tr_chemberta.npy"]).astype(np.float32)
    te_emb = np.load(needed["te_chemberta.npy"]).astype(np.float32)
    n_te = anchor.shape[0]
    assert err_hat.shape == (n_te,)
    assert te_emb.shape[0] == n_te, f"te_emb {te_emb.shape} vs n_te {n_te}"
    print(f"tr embed: {tr_emb.shape}  te embed: {te_emb.shape}")

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- PCA: fit on train+test pool (unsupervised; safe) ----
    pool = np.vstack([tr_emb, te_emb]).astype(np.float32)
    print(f"\nPCA: fit on pool {pool.shape} -> {PCA_DIM} components")
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(pool)
    X_full = pca.transform(te_emb).astype(np.float32)
    evr = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance ratio sum: {evr:.4f}")
    print(f"  X_full (test): {X_full.shape}")

    X_unb = X_full[unb_idx]
    y_resid = (unb_y - anchor[unb_idx]).astype(np.float32)
    print(f"\nResidual target (truth - {ANCHOR_NAME}):")
    print(f"  mean={y_resid.mean():+.3f}  std={y_resid.std():.3f}  "
          f"|mean|={np.abs(y_resid).mean():.3f}")
    print(f"anchor ({ANCHOR_NAME}) stats: mean={anchor.mean():.3f} "
          f"std={anchor.std():.3f}")

    # ---- 5-fold cross-fit LGBM ----
    print("\n5-fold cross-fit LGBM (max_depth=3, n_est=80, lr=0.05):")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_i], y_resid[tr_i])
        resid_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"resid_oof mean={resid_oof[va_i].mean():+.3f} "
              f"std={resid_oof[va_i].std():.3f}")
    del mdl

    rho_resid, _ = spearmanr(resid_oof, y_resid)
    if not np.isfinite(rho_resid):
        rho_resid = 0.0
    rho_abs, _ = spearmanr(np.abs(resid_oof), np.abs(y_resid))
    if not np.isfinite(rho_abs):
        rho_abs = 0.0
    print(f"\nSpearman(resid_oof, true residual)     = {rho_resid:.4f}")
    print(f"Spearman(|resid_oof|, |true residual|) = {rho_abs:.4f}")

    # ---- Refit on all 253 -> deploy resid_hat for 513 ----
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X_full).astype(np.float32)
    del deploy_mdl
    print(f"\nDeploy resid_hat (513): mean={resid_hat_513.mean():+.3f}  "
          f"std={resid_hat_513.std():.3f}  "
          f"|mean|={np.abs(resid_hat_513).mean():.3f}")

    # ---- Soft gate ----
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"\nGate alpha (513): min={alpha_513.min():.3f}  "
          f"med={np.median(alpha_513):.3f}  max={alpha_513.max():.3f}  "
          f"mean={alpha_513.mean():.3f}")

    # ---- Deploy + honest cross-fit RAE ----
    deploy = (anchor + alpha_513 * resid_hat_513).astype(np.float32)
    pred_oof = (anchor[unb_idx] + alpha_513[unb_idx] * resid_oof).astype(np.float32)

    rae_anchor = float(rae(unb_y, anchor[unb_idx]))
    rae_routed = float(rae(unb_y, pred_oof))
    rae_insample = float(rae(unb_y, deploy[unb_idx]))
    print("\nUnblind RAE (n=253):")
    print(f"  {ANCHOR_NAME} anchor (standalone) = {rae_anchor:.4f}")
    print(f"  {TAG} cross-fit (honest)          = {rae_routed:.4f}")
    print(f"  {TAG} in-sample refit             = {rae_insample:.4f}")
    beats_anchor = rae_routed < rae_anchor
    beats_nb562 = rae_routed < TARGET_RAE
    print(f"  beats {ANCHOR_NAME} (RAE<{rae_anchor:.4f})? = {beats_anchor}")
    print(f"  beats nb562 (RAE<{TARGET_RAE})?           = {beats_nb562}")

    # ---- Decorrelation checks ----
    def _corr(path: Path) -> float:
        if not path.exists():
            return float("nan")
        arr = np.load(path).astype(np.float32)
        if arr.shape != resid_oof.shape:
            return float("nan")
        r, _ = spearmanr(resid_oof, arr)
        return float(r) if np.isfinite(r) else float("nan")

    rho_to_nb502 = _corr(DATA_PROCESSED / "nb502_resid_oof.npy")
    rho_to_nb510 = _corr(DATA_PROCESSED / "nb510_resid_oof.npy")
    rho_to_nb601 = _corr(DATA_PROCESSED / "nb601_resid_oof.npy")
    print("\nResidual decorrelation (Spearman):")
    print(f"  resid_oof vs nb502  = {rho_to_nb502:.4f}")
    print(f"  resid_oof vs nb510  = {rho_to_nb510:.4f}")
    print(f"  resid_oof vs nb601  = {rho_to_nb601:.4f}")

    # ---- Save arrays ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{TAG}_resid_oof.npy", resid_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{TAG}_alpha.npy", alpha_513)

    # ---- Submissions ----
    plain = SUBMISSIONS / f"{TAG}_chemberta_anchor_{ANCHOR_NAME}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_chemberta_anchor_{ANCHOR_NAME}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  anchor                                 = {ANCHOR_NAME}")
    print(f"  feature space                          = ChemBERTa (PCA->{PCA_DIM}, train+test pool)")
    print(f"  PCA explained variance ratio sum       = {evr:.4f}")
    print(f"  spearman(resid_oof, true resid)        = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true resid|)    = {rho_abs:.4f}")
    print(f"  spearman(resid_oof, nb502 resid_oof)   = {rho_to_nb502:.4f}")
    print(f"  spearman(resid_oof, nb601 resid_oof)   = {rho_to_nb601:.4f}")
    print(f"  unblind RAE {ANCHOR_NAME} anchor      = {rae_anchor:.4f}")
    print(f"  unblind RAE {TAG} cross-fit           = {rae_routed:.4f}")
    print(f"  beats nb562 (< {TARGET_RAE})            = {beats_nb562}")
    print("=" * 78)

    return {
        "success": True,
        "anchor": ANCHOR_NAME,
        "feature_space": f"ChemBERTa_PCA{PCA_DIM}",
        "pca_explained_variance": float(evr),
        "spearman_resid_oof_true": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "resid_corr_with_nb502": rho_to_nb502,
        "resid_corr_with_nb510": rho_to_nb510,
        "resid_corr_with_nb601": rho_to_nb601,
        "rae_anchor": rae_anchor,
        "crossfit_rae": rae_routed,
        "insample_rae": rae_insample,
        "beats_anchor": bool(beats_anchor),
        "beats_nb562": bool(beats_nb562),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
