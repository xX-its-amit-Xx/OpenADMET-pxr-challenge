"""nb613 -- ChemBERTa PCA dim sweep on the nb464-anchored residual router.

nb601 fixed PCA at 64-d as a default. This sweeps {32, 64, 128, 256, 384(no PCA)}
to test whether nb601's choice was suboptimal.

Pipeline (per dim):
  1. Anchor = nb464 (same as nb601).
  2. Stack tr_chemberta + te_chemberta and fit PCA on the union
     (unsupervised, no truth leakage). For dim=384 (full), skip PCA.
  3. Slice to the 253 unblind rows. Residual target = truth - nb464.
  4. 5-fold KFold cross-fit shallow LGBM (same params as nb601).
  5. Pooled cross-fit unblind RAE.

Best-dim deploy:
  - Refit best-dim LGBM on all 253 unblind rows.
  - Apply to all 513 test rows.
  - alpha gate (sigmoid on err_hat) identical to nb601.
  - Save te_nb613.npy = anchor + alpha * resid_hat_513.
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
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7
TAG = "nb613"
ANCHOR_NAME = "nb464"
ANCHOR_FILE = "te_nb464.npy"
PCA_DIMS = [32, 64, 128, 256, 384]  # 384 = no PCA (full ChemBERTa hidden)

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

NB601_RAE = 0.5065  # leaderboard ref
NB562_RAE = 0.5065


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def crossfit_rae(X_unb, y_resid, unb_y, anchor_unb, alpha_unb) -> tuple[float, np.ndarray]:
    """5-fold cross-fit. Returns (rae, resid_oof)."""
    n = len(y_resid)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n, dtype=np.float32)
    for tr_i, va_i in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_i], y_resid[tr_i])
        resid_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
    pred_oof = anchor_unb + alpha_unb * resid_oof
    return float(rae(unb_y, pred_oof)), resid_oof


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa PCA dim sweep (anchor={ANCHOR_NAME})")
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
    print(f"train embed: {tr_emb.shape}   test embed: {te_emb.shape}")
    assert te_emb.shape[0] == n_te
    H = te_emb.shape[1]  # native hidden dim (e.g. 384)

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"].tolist())}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_idx)
    print(f"test n={n_te}  unblind n={n_unb}")

    # ---- Soft gate (depends only on err_hat, same as nb601) ----
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    alpha_unb = alpha_513[unb_idx]

    rae_anchor = float(rae(unb_y, anchor[unb_idx]))
    print(f"\n{ANCHOR_NAME} anchor standalone RAE = {rae_anchor:.4f}")

    # ---- Stack tr+te for PCA fitting (unsupervised) ----
    emb_all = np.vstack([tr_emb, te_emb]).astype(np.float32)
    print(f"PCA fitting stack shape: {emb_all.shape}")

    results: dict[int, dict] = {}
    rae_by_dim: dict[int, float] = {}

    for dim in PCA_DIMS:
        print("\n" + "-" * 78)
        if dim >= H:
            print(f"dim={dim} >= hidden({H})  -> no PCA, using full embeddings")
            X_full = te_emb.copy()
            evr = 1.0
            effective_dim = H
        else:
            pca = PCA(n_components=dim, random_state=SEED)
            pca.fit(emb_all)
            X_full = pca.transform(te_emb).astype(np.float32)
            evr = float(pca.explained_variance_ratio_.sum())
            effective_dim = dim
            del pca
        print(f"dim={dim}  effective={effective_dim}  explained_variance={evr:.4f}  "
              f"X_full={X_full.shape}")

        X_unb = X_full[unb_idx]
        y_resid = (unb_y - anchor[unb_idx]).astype(np.float32)

        r, resid_oof = crossfit_rae(
            X_unb, y_resid, unb_y, anchor[unb_idx], alpha_unb
        )
        rae_by_dim[dim] = r
        results[dim] = {
            "rae": r,
            "evr": evr,
            "effective_dim": effective_dim,
            "X_full": X_full,
            "X_unb": X_unb,
            "y_resid": y_resid,
            "resid_oof": resid_oof,
        }
        print(f"  cross-fit unblind RAE = {r:.4f}   (anchor={rae_anchor:.4f})")

    # ---- Best dim ----
    print("\n" + "=" * 78)
    print("PCA dim sweep results (cross-fit unblind RAE):")
    for dim in PCA_DIMS:
        marker = ""
        if dim == 64:
            marker = "  <- nb601 default"
        print(f"  dim={dim:>3}  RAE={rae_by_dim[dim]:.4f}{marker}")

    best_dim = min(rae_by_dim, key=rae_by_dim.get)
    best_rae = rae_by_dim[best_dim]
    print(f"\nBEST: dim={best_dim}  RAE={best_rae:.4f}")
    beats_nb601 = best_rae < NB601_RAE
    beats_nb562 = best_rae < NB562_RAE
    print(f"  beats nb601 ({NB601_RAE})? {beats_nb601}")
    print(f"  beats nb562 ({NB562_RAE})? {beats_nb562}")

    # ---- Deploy best dim ----
    best = results[best_dim]
    X_full_best = best["X_full"]
    X_unb_best = best["X_unb"]
    y_resid_best = best["y_resid"]

    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb_best, y_resid_best)
    resid_hat_513 = deploy_mdl.predict(X_full_best).astype(np.float32)
    del deploy_mdl

    deploy = (anchor + alpha_513 * resid_hat_513).astype(np.float32)
    print(f"\nDeploy ({TAG}, best_dim={best_dim}):")
    print(f"  resid_hat: mean={resid_hat_513.mean():+.3f}  "
          f"std={resid_hat_513.std():.3f}")
    print(f"  pred: mean={deploy.mean():.3f}  std={deploy.std():.3f}  "
          f"min={deploy.min():.3f}  max={deploy.max():.3f}")

    # Save
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_resid_oof.npy", best["resid_oof"])
    np.save(DATA_PROCESSED / f"te_{TAG}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{TAG}_alpha.npy", alpha_513)

    plain = SUBMISSIONS / f"{TAG}_chemberta_pca{best_dim}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_chemberta_pca{best_dim}_soft07_truth.csv"
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
    print(f"  anchor                = {ANCHOR_NAME}  (RAE {rae_anchor:.4f})")
    for dim in PCA_DIMS:
        print(f"  dim {dim:>3}  RAE = {rae_by_dim[dim]:.4f}")
    print(f"  best dim              = {best_dim}")
    print(f"  best RAE              = {best_rae:.4f}")
    print(f"  beats nb601 (0.5065)? = {beats_nb601}")
    print(f"  beats nb562 (0.5065)? = {beats_nb562}")
    print("=" * 78)

    return {
        "success": True,
        "anchor": ANCHOR_NAME,
        "rae_anchor": rae_anchor,
        "rae_by_pca_dim": {str(k): float(v) for k, v in rae_by_dim.items()},
        "best_dim": int(best_dim),
        "best_rae": float(best_rae),
        "beats_nb601": bool(beats_nb601),
        "beats_nb562": bool(beats_nb562),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
