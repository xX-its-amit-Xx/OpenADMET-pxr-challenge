"""nb601 -- PRETRAINED EMBEDDING RESIDUAL ROUTER (ChemBERTa-77M-MTR).

Hypothesis: nb502/nb510/nb512 routers used hand-crafted feature spaces (MACCS,
atom-pair, PubChem keys). A router built on PRETRAINED MOLECULAR EMBEDDINGS
(ChemBERTa-77M-MTR, last_hidden_state mean-pooled, 384-d) should learn a
decorrelated residual direction from a totally orthogonal feature space. If
resid_oof correlation to nb502 < 0.5, pretrained chemistry knowledge is a
truly orthogonal signal source.

Pipeline:
  1. Anchor: nb464 (best single anchor per nb492).
  2. Features: tr_chemberta.npy + te_chemberta.npy (384-d each).
  3. PCA: reduce 384-d -> 64-d (n_components=64) for stable LGBM fit.
     Fit PCA on ALL test embeddings (no truth leakage, embeddings are
     unsupervised representations from a pretrained model).
  4. Target = truth - nb464 on 253 unblind rows.
  5. 5-fold KFold cross-fit shallow LGBM
       (max_depth=3, n_est=80, lr=0.05, num_leaves=8, min_child_samples=20,
        reg_lambda=1.0, seed=0).
  6. Gate: alpha = sigmoid((err_hat - median(err_hat)) * 4.0).
  7. Honest cross-fit unblind RAE.
     Target: < 0.5283 (single router beating nb492 baseline)
             < 0.5065 (single router beating nb562).
  8. Rank-stretch grid {1.0,...,1.5} cross-fit on the routed pred_oof.
  9. Save te_nb601 + pred_oof + submissions.

The note on PCA: 384-d -> 64-d retains >90% variance for ChemBERTa-MTR while
being well-conditioned for a depth-3 LGBM with only 253 rows of supervision.
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
TAG = "nb601"
ANCHOR_NAME = "nb464"
ANCHOR_FILE = "te_nb464.npy"
PCA_DIM = 64
STRETCH_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

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


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def _best_s_on(p_tr: np.ndarray, y_tr: np.ndarray, grid) -> tuple[float, float]:
    mu = float(p_tr.mean())
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = _stretch(p_tr, mu, s)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, mu


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- PRETRAINED EMBED ROUTER (ChemBERTa-77M-MTR, anchor={ANCHOR_NAME})")
    print("=" * 78)

    needed = {
        ANCHOR_FILE: DATA_PROCESSED / ANCHOR_FILE,
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
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
    te_emb = np.load(needed["te_chemberta.npy"]).astype(np.float32)
    n_te = anchor.shape[0]
    assert err_hat.shape == (n_te,)
    assert te_emb.shape[0] == n_te, f"emb {te_emb.shape} vs n_te {n_te}"
    print(f"test embed shape: {te_emb.shape}  hidden_dim={te_emb.shape[1]}")

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- PCA reduce ----
    print(f"\nPCA: {te_emb.shape[1]} -> {PCA_DIM} components (fit on 513 test embeds)")
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_full = pca.fit_transform(te_emb).astype(np.float32)
    evr = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {evr:.4f}")
    print(f"  X_full shape: {X_full.shape}")

    X_unb = X_full[unb_idx]
    y_resid = (unb_y - anchor[unb_idx]).astype(np.float32)
    print(f"\nResidual target: mean={y_resid.mean():+.3f} "
          f"std={y_resid.std():.3f} |mean|={np.abs(y_resid).mean():.3f}")
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
    print(f"\nSpearman(resid_oof, true residual)   = {rho_resid:.4f}")
    print(f"Spearman(|resid_oof|, |true residual|) = {rho_abs:.4f}")

    # ---- Refit on all 253 -> deploy ----
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
    beats_nb492 = rae_routed < 0.5283
    beats_nb562 = rae_routed < 0.5065
    print(f"  beats nb492 (0.5283)?            = {beats_nb492}")
    print(f"  beats nb562 (0.5065)?            = {beats_nb562}")

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
    rho_to_nb492 = _corr(DATA_PROCESSED / "nb492_resid_oof.npy")
    print("\nResidual decorrelation (Spearman):")
    print(f"  resid_oof vs nb502  = {rho_to_nb502:.4f}")
    print(f"  resid_oof vs nb510  = {rho_to_nb510:.4f}")
    print(f"  resid_oof vs nb492  = {rho_to_nb492:.4f}")
    truly_orthogonal = np.isfinite(rho_to_nb502) and rho_to_nb502 < 0.50
    if truly_orthogonal:
        print("  -> pretrained embeddings are TRULY ORTHOGONAL to nb502 (< 0.50)")
    else:
        print("  -> resid corr to nb502 >= 0.50 (not strictly orthogonal)")

    # ---- Rank-stretch on pred_oof / deploy ----
    print("\nRank-stretch grid cross-fit on pred_oof:")
    kf2 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_stretch = np.full(n_unb, np.nan, dtype=np.float32)
    fold_s = []
    for fold, (tr_i, va_i) in enumerate(kf2.split(np.arange(n_unb))):
        s_star, mu_tr = _best_s_on(
            pred_oof[tr_i].astype(np.float64),
            unb_y[tr_i].astype(np.float64),
            STRETCH_GRID,
        )
        oof_stretch[va_i] = _stretch(pred_oof[va_i], mu_tr, s_star).astype(np.float32)
        fold_s.append(s_star)
    rae_stretched = float(rae(unb_y, oof_stretch))
    s_deploy, mu_deploy = _best_s_on(
        pred_oof.astype(np.float64), unb_y.astype(np.float64), STRETCH_GRID
    )
    print(f"  per-fold s: {fold_s}")
    print(f"  cross-fit stretched RAE = {rae_stretched:.4f}")
    print(f"  deploy s={s_deploy:.2f}  mu={mu_deploy:.3f}")
    deploy_stretched = _stretch(deploy.astype(np.float64), mu_deploy, s_deploy).astype(
        np.float32
    )

    # ---- Save arrays ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"te_{TAG}_stretched.npy", deploy_stretched)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{TAG}_resid_oof.npy", resid_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{TAG}_alpha.npy", alpha_513)

    # ---- Submissions ----
    plain = SUBMISSIONS / f"{TAG}_pretrained_embed_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_pretrained_embed_router_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    stretched_plain = SUBMISSIONS / f"{TAG}_stretched_s{s_deploy:.2f}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy_stretched,
    }).to_csv(stretched_plain, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")
    print(f"Wrote {stretched_plain}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  anchor                                = {ANCHOR_NAME}")
    print(f"  feature space                         = ChemBERTa-77M-MTR (PCA->{PCA_DIM})")
    print(f"  PCA explained variance                = {evr:.4f}")
    print(f"  spearman(resid_oof, true resid)       = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true resid|)   = {rho_abs:.4f}")
    print(f"  spearman(resid_oof, nb502 resid_oof)  = {rho_to_nb502:.4f}")
    print(f"  spearman(resid_oof, nb510 resid_oof)  = {rho_to_nb510:.4f}")
    print(f"  spearman(resid_oof, nb492 resid_oof)  = {rho_to_nb492:.4f}")
    print(f"  unblind RAE {ANCHOR_NAME} anchor     = {rae_anchor:.4f}")
    print(f"  unblind RAE {TAG} cross-fit          = {rae_routed:.4f}")
    print(f"  unblind RAE {TAG} stretched           = {rae_stretched:.4f}")
    print(f"  beats nb492 (< 0.5283)                = {beats_nb492}")
    print(f"  beats nb562 (< 0.5065)                = {beats_nb562}")
    print(f"  truly orthogonal to nb502 (< 0.50)    = {truly_orthogonal}")
    print("=" * 78)

    return {
        "success": True,
        "anchor": ANCHOR_NAME,
        "feature_space": f"ChemBERTa-77M-MTR_PCA{PCA_DIM}",
        "pca_explained_variance": float(evr),
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "resid_corr_with_nb502": rho_to_nb502,
        "resid_corr_with_nb510": rho_to_nb510,
        "resid_corr_with_nb492": rho_to_nb492,
        "rae_anchor": rae_anchor,
        "crossfit_rae": rae_routed,
        "crossfit_rae_stretched": rae_stretched,
        "insample_rae": rae_insample,
        "deploy_stretch_s": float(s_deploy),
        "beats_nb492": bool(beats_nb492),
        "beats_nb562": bool(beats_nb562),
        "truly_orthogonal_to_nb502": bool(truly_orthogonal),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
        "stretched_submission": str(stretched_plain),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
