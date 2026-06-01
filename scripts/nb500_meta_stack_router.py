"""nb500 -- META STACK ROUTER (stack-on-stack).

Stacks the 6 existing residual routers (nb472, nb481, nb482, nb490, nb491,
nb492) into a single meta-learner over the 253 unblind rows. The meta-LGBM
predicts truth directly (NOT a residual -- the base routers have already
extracted residual signal from their anchors).

Feature matrix on 253 unblind (45 cols):
  - 6 router pred_oof values             (nb472..nb492)
  - 33 multimodal features (build_feature_matrix from nb481)
  - 6 router resid_oof signed values

Target: unblind truth pEC50.

Cross-fit: 5-fold KFold shallow LGBM
  (max_depth=3, n_est=80, lr=0.05, num_leaves=8, min_child_samples=20,
   reg_lambda=1.0, seed=0)

Honest cross-fit RAE target: < 0.5283 (nb492 single).

Deploy on 513:
  - 6 router te_*.npy preds (CAVEAT: these are refit on all 253 unblind so
    they are technically in-sample for the meta-router. We cannot avoid this
    without re-running each base router cross-fit-on-cross-fit which is
    prohibitively expensive. Flagged in summary.)
  - 33 multimodal features (deploy version, same as cross-fit since it's
    built once for 513)
  - 6 router resid_hat te values:
        * nb472/481/482: te_nb*_resid_hat.npy directly
        * nb490/491/492: derived as (te_nb49x - anchor) / alpha
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7

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

# (tag, anchor_te_file)
ROUTERS = [
    ("nb472", "te_nb432.npy"),
    ("nb481", "te_nb432.npy"),
    ("nb482", "te_nb432.npy"),
    ("nb490", "te_chemprop_aux.npy"),
    ("nb491", "te_nb420_frontier.npy"),
    ("nb492", "te_nb464.npy"),
]


def _import_nb481():
    nb481_path = Path(__file__).parent / "nb481_residual_router_extended.py"
    spec = importlib.util.spec_from_file_location("nb481_mod", nb481_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_or_derive_resid_hat_te(
    tag: str, anchor_te: np.ndarray, alpha_te: np.ndarray
) -> np.ndarray:
    """Return (513,) signed residual deploy vector."""
    direct = DATA_PROCESSED / f"te_{tag}_resid_hat.npy"
    if direct.exists():
        arr = np.load(direct).astype(np.float32)
        if arr.shape == anchor_te.shape:
            return arr
    # Derive: deploy = anchor + alpha * resid_hat  =>  resid_hat = (deploy - anchor) / alpha
    deploy = np.load(DATA_PROCESSED / f"te_{tag}.npy").astype(np.float32)
    eps = 1e-6
    return ((deploy - anchor_te) / np.maximum(alpha_te, eps)).astype(np.float32)


def _load_or_derive_resid_oof(
    tag: str, anchor_te: np.ndarray, alpha_te: np.ndarray, unb_idx: np.ndarray
) -> np.ndarray:
    """Return (n_unb,) signed residual OOF."""
    direct = DATA_PROCESSED / f"{tag}_resid_oof.npy"
    if direct.exists():
        arr = np.load(direct).astype(np.float32)
        if arr.shape == (len(unb_idx),):
            return arr
    # Derive: pred_oof = anchor[unb_idx] + alpha[unb_idx] * resid_oof
    pred_oof = np.load(DATA_PROCESSED / f"{tag}_pred_oof.npy").astype(np.float32)
    eps = 1e-6
    return (
        (pred_oof - anchor_te[unb_idx])
        / np.maximum(alpha_te[unb_idx], eps)
    ).astype(np.float32)


def main() -> dict:
    print("=" * 78)
    print("nb500 -- META STACK ROUTER (stack-on-stack)")
    print("=" * 78)

    needed = {
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for tag, anchor_file in ROUTERS:
        needed[f"te_{tag}.npy"] = DATA_PROCESSED / f"te_{tag}.npy"
        needed[f"{tag}_pred_oof.npy"] = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        needed[anchor_file] = DATA_PROCESSED / anchor_file
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Anchors + alphas ----
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    med_err = float(np.median(err_hat))
    alpha_te = (1.0 / (1.0 + np.exp(-(err_hat - med_err) * 4.0))).astype(np.float32)

    # ---- Build 33-col multimodal feature matrix (once) ----
    print("\nBuilding multimodal feature matrix (from nb481) ...")
    nb481_mod = _import_nb481()
    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    X_mm, mm_names = nb481_mod.build_feature_matrix(te_df, nb432)
    X_mm = np.nan_to_num(X_mm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"  multimodal shape: {X_mm.shape}  cols={len(mm_names)}")

    # ---- Per-router columns ----
    router_pred_oof: dict[str, np.ndarray] = {}
    router_pred_te: dict[str, np.ndarray] = {}
    router_resid_oof: dict[str, np.ndarray] = {}
    router_resid_te: dict[str, np.ndarray] = {}
    print("\nLoading router OOFs + deploys:")
    for tag, anchor_file in ROUTERS:
        anchor_te = np.load(needed[anchor_file]).astype(np.float32)
        pred_oof = np.load(needed[f"{tag}_pred_oof.npy"]).astype(np.float32)
        pred_te = np.load(needed[f"te_{tag}.npy"]).astype(np.float32)
        resid_oof = _load_or_derive_resid_oof(
            tag, anchor_te, alpha_te, unb_idx
        )
        resid_te = _load_or_derive_resid_hat_te(tag, anchor_te, alpha_te)
        router_pred_oof[tag] = pred_oof
        router_pred_te[tag] = pred_te
        router_resid_oof[tag] = resid_oof
        router_resid_te[tag] = resid_te
        rae_router = float(rae(unb_y, pred_oof))
        print(f"  {tag} (anchor={anchor_file:28s}) pred_oof RAE={rae_router:.4f}  "
              f"resid_oof std={resid_oof.std():.3f}")

    # ---- Assemble META feature matrices ----
    pred_cols = [f"{tag}_pred" for tag, _ in ROUTERS]
    resid_cols = [f"{tag}_resid" for tag, _ in ROUTERS]
    feat_names = pred_cols + mm_names + resid_cols

    # Unblind (253, 45)
    pred_oof_block = np.stack(
        [router_pred_oof[tag] for tag, _ in ROUTERS], axis=1
    ).astype(np.float32)
    resid_oof_block = np.stack(
        [router_resid_oof[tag] for tag, _ in ROUTERS], axis=1
    ).astype(np.float32)
    X_unb = np.concatenate(
        [pred_oof_block, X_mm[unb_idx], resid_oof_block], axis=1
    ).astype(np.float32)
    X_unb = np.nan_to_num(X_unb, nan=0.0, posinf=0.0, neginf=0.0)

    # Deploy (513, 45)
    pred_te_block = np.stack(
        [router_pred_te[tag] for tag, _ in ROUTERS], axis=1
    ).astype(np.float32)
    resid_te_block = np.stack(
        [router_resid_te[tag] for tag, _ in ROUTERS], axis=1
    ).astype(np.float32)
    X_te = np.concatenate(
        [pred_te_block, X_mm, resid_te_block], axis=1
    ).astype(np.float32)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nMETA matrix unblind: {X_unb.shape}  deploy: {X_te.shape}  "
          f"n_feats={len(feat_names)}")
    assert X_unb.shape == (n_unb, 45), f"expected 45 cols got {X_unb.shape[1]}"
    assert X_te.shape == (n_te, 45)

    # ---- 5-fold cross-fit ----
    print("\n5-fold KFold cross-fit shallow LGBM (target=truth):")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_i], unb_y[tr_i])
        pred_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"pred_oof mean={pred_oof[va_i].mean():.3f} "
              f"std={pred_oof[va_i].std():.3f}")
    del mdl

    rae_meta_oof = float(rae(unb_y, pred_oof))
    print(f"\nHonest cross-fit RAE (n=253) nb500 = {rae_meta_oof:.4f}")
    NB492_TARGET = 0.5283
    beats_nb492 = rae_meta_oof < NB492_TARGET
    print(f"  beats nb492 (<{NB492_TARGET:.4f}) = {beats_nb492}")

    # ---- Deploy: refit on all 253 ----
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, unb_y)
    deploy = deploy_mdl.predict(X_te).astype(np.float32)
    print(f"\nDeploy nb500 (513): mean={deploy.mean():.3f} std={deploy.std():.3f}")

    # ---- Importances ----
    imp = deploy_mdl.feature_importances_
    order = np.argsort(-imp)
    print("\nTop 10 META feature importances:")
    top5_names: list[str] = []
    for rk, i in enumerate(order[:10]):
        print(f"  {feat_names[i]:30s}  {imp[i]}")
        if rk < 5:
            top5_names.append(feat_names[i])

    # ---- Save ----
    np.save(DATA_PROCESSED / "te_nb500.npy", deploy)
    np.save(DATA_PROCESSED / "nb500_pred_oof.npy", pred_oof)

    plain = SUBMISSIONS / "nb500_meta_stack_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / "nb500_meta_stack_router_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb500.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb500_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb500 SUMMARY ===")
    print(f"  n_features                                = {len(feat_names)}")
    print(f"  cross-fit RAE nb500 (honest, n=253)        = {rae_meta_oof:.4f}")
    print(f"  beats nb492 (<0.5283)                      = {beats_nb492}")
    print("  CAVEAT: deploy te_*.npy from base routers are refit on all 253")
    print("    -> meta deploy is partially in-sample wrt base-router refit.")
    print("    Cross-fit OOF score IS honest.")
    print("=" * 78)

    return {
        "success": True,
        "n_features": len(feat_names),
        "crossfit_rae": rae_meta_oof,
        "beats_nb492": bool(beats_nb492),
        "top5_features": top5_names,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
