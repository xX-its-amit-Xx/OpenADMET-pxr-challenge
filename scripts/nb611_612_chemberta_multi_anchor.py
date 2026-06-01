"""nb611 + nb612 -- CHEMBERTA RESIDUAL ROUTERS WITH ALTERNATE ANCHORS.

Same architecture as nb610 (ChemBERTa PCA-64 + shallow LGBM + sigmoid gate
+ 5-fold cross-fit), but anchor swapped:
  nb611 -> anchor = nb503
  nb612 -> anchor = chemprop_aux

Hypothesis: ChemBERTa carries orthogonal residual signal vs each anchor.
nb503 = Karpathy DANN; chemprop_aux = chemprop with aux multitask head.
Pairing the same router with different anchors stress-tests whether the
gain is anchor-specific.

Pipeline (per anchor):
  1. Load tr_chemberta.npy (4139, 384) + te_chemberta.npy (513, 384).
  2. PCA(64) fit on the FULL pool (train + test embeddings) -- unsupervised.
  3. Target = truth - anchor on 253 unblind rows.
  4. 5-fold KFold cross-fit shallow LGBM
       (max_depth=3, n_est=80, lr=0.05, num_leaves=8,
        min_child_samples=20, reg_lambda=1.0, seed=0).
  5. Soft gate alpha = sigmoid((err_hat - median(err_hat)) * 4.0).
  6. pred_oof = anchor[unb_idx] + alpha * resid_oof (honest cross-fit).
  7. Save te_nbXXX.npy + pred_oof + submissions.
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
PCA_DIM = 64

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

CONFIGS = [
    {"tag": "nb611", "anchor_name": "nb503",        "anchor_file": "te_nb503.npy"},
    {"tag": "nb612", "anchor_name": "chemprop_aux", "anchor_file": "te_chemprop_aux.npy"},
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_one(
    *,
    tag: str,
    anchor_name: str,
    anchor_path: Path,
    err_hat: np.ndarray,
    tr_emb: np.ndarray,
    te_emb: np.ndarray,
    te_df: pd.DataFrame,
    unb_idx: np.ndarray,
    unb_y: np.ndarray,
) -> dict:
    print("\n" + "=" * 78)
    print(f"{tag} -- ChemBERTa router anchored on {anchor_name}")
    print("=" * 78)

    anchor = np.load(anchor_path).astype(np.float32)
    n_te = anchor.shape[0]
    assert err_hat.shape == (n_te,)
    assert te_emb.shape[0] == n_te
    n_unb = len(unb_idx)

    # ---- PCA: fit on train+test pool (unsupervised; safe) ----
    pool = np.vstack([tr_emb, te_emb]).astype(np.float32)
    print(f"PCA: fit on pool {pool.shape} -> {PCA_DIM} components")
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(pool)
    X_full = pca.transform(te_emb).astype(np.float32)
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance ratio sum: {evr:.4f}")

    X_unb = X_full[unb_idx]
    y_resid = (unb_y - anchor[unb_idx]).astype(np.float32)
    print(f"Residual target (truth - {anchor_name}):")
    print(f"  mean={y_resid.mean():+.3f}  std={y_resid.std():.3f}  "
          f"|mean|={np.abs(y_resid).mean():.3f}")
    print(f"anchor ({anchor_name}) stats: mean={anchor.mean():.3f} "
          f"std={anchor.std():.3f}")

    # ---- 5-fold cross-fit LGBM ----
    print("5-fold cross-fit LGBM (max_depth=3, n_est=80, lr=0.05):")
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
    print(f"Spearman(resid_oof, true residual)     = {rho_resid:.4f}")
    print(f"Spearman(|resid_oof|, |true residual|) = {rho_abs:.4f}")

    # ---- Refit on all 253 -> deploy resid_hat for 513 ----
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X_full).astype(np.float32)
    del deploy_mdl
    print(f"Deploy resid_hat (513): mean={resid_hat_513.mean():+.3f}  "
          f"std={resid_hat_513.std():.3f}  "
          f"|mean|={np.abs(resid_hat_513).mean():.3f}")

    # ---- Soft gate ----
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"Gate alpha (513): min={alpha_513.min():.3f}  "
          f"med={np.median(alpha_513):.3f}  max={alpha_513.max():.3f}  "
          f"mean={alpha_513.mean():.3f}")

    # ---- Deploy + honest cross-fit RAE ----
    deploy = (anchor + alpha_513 * resid_hat_513).astype(np.float32)
    pred_oof = (anchor[unb_idx] + alpha_513[unb_idx] * resid_oof).astype(np.float32)

    rae_anchor = float(rae(unb_y, anchor[unb_idx]))
    rae_routed = float(rae(unb_y, pred_oof))
    rae_insample = float(rae(unb_y, deploy[unb_idx]))
    print(f"Unblind RAE (n=253):")
    print(f"  {anchor_name} anchor (standalone) = {rae_anchor:.4f}")
    print(f"  {tag} cross-fit (honest)         = {rae_routed:.4f}")
    print(f"  {tag} in-sample refit            = {rae_insample:.4f}")
    beats_anchor = rae_routed < rae_anchor
    print(f"  beats {anchor_name} (RAE<{rae_anchor:.4f})? = {beats_anchor}")

    # ---- Save arrays ----
    np.save(DATA_PROCESSED / f"te_{tag}.npy", deploy)
    np.save(DATA_PROCESSED / f"{tag}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{tag}_resid_oof.npy", resid_oof)
    np.save(DATA_PROCESSED / f"te_{tag}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{tag}_alpha.npy", alpha_513)

    # ---- Submissions ----
    plain = SUBMISSIONS / f"{tag}_chemberta_anchor_{anchor_name}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{tag}_chemberta_anchor_{anchor_name}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"Wrote {DATA_PROCESSED / f'te_{tag}.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    return {
        "tag": tag,
        "anchor": anchor_name,
        "feature_space": f"ChemBERTa_PCA{PCA_DIM}",
        "pca_explained_variance": evr,
        "spearman_resid_oof_true": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "rae_anchor": rae_anchor,
        "crossfit_rae": rae_routed,
        "insample_rae": rae_insample,
        "beats_anchor": bool(beats_anchor),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


def main() -> dict:
    print("=" * 78)
    print("nb611 + nb612 -- ChemBERTa multi-anchor routers")
    print("=" * 78)

    needed = {
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "tr_chemberta.npy": DATA_PROCESSED / "tr_chemberta.npy",
        "te_chemberta.npy": DATA_PROCESSED / "te_chemberta.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for cfg in CONFIGS:
        needed[cfg["anchor_file"]] = DATA_PROCESSED / cfg["anchor_file"]
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "notes": "missing_inputs", "missing": missing}

    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    tr_emb = np.load(needed["tr_chemberta.npy"]).astype(np.float32)
    te_emb = np.load(needed["te_chemberta.npy"]).astype(np.float32)
    print(f"tr embed: {tr_emb.shape}  te embed: {te_emb.shape}")

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    print(f"test n={len(te_names)}  unblind n={len(unb_idx)}")

    results = {}
    for cfg in CONFIGS:
        res = run_one(
            tag=cfg["tag"],
            anchor_name=cfg["anchor_name"],
            anchor_path=DATA_PROCESSED / cfg["anchor_file"],
            err_hat=err_hat,
            tr_emb=tr_emb,
            te_emb=te_emb,
            te_df=te_df,
            unb_idx=unb_idx,
            unb_y=unb_y,
        )
        results[cfg["tag"]] = res

    print("\n" + "=" * 78)
    print("=== nb611 + nb612 COMBINED SUMMARY ===")
    for tag, r in results.items():
        print(
            f"  {tag} anchor={r['anchor']:<13} rae_anchor={r['rae_anchor']:.4f}  "
            f"crossfit_rae={r['crossfit_rae']:.4f}  "
            f"beats_anchor={r['beats_anchor']}"
        )
    print("=" * 78)

    return {"success": True, "results": results}


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
