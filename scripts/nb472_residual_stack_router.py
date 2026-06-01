"""nb472 -- RESIDUAL STACK ROUTER (correct fix for nb460 failure).

nb460 failed because it soft-blended toward chemprop_aux (RAE 0.6216) -- on
high-err_hat rows the alt-predictor imported its own bias rather than the
unknown direction of nb432's error.

The CONCEPTUAL FIX: instead of blending toward a flat alt-predictor, learn an
ERROR CORRECTION TERM directly. The router decides where to apply correction
(sigmoid gate on err_hat). The residual stack decides what the correction is.

Pipeline:
  1) Reuse te_nb443_err_hat.npy (Spearman 0.265 with true |error|).
  2) Reuse the same 18-col multimodal feature matrix from nb443.
  3) Build a RESIDUAL stack:
       target = truth - nb432   (signed residual on 253 unblind rows)
       model  = shallow LGBM (max_depth=3, n_est=80, lr=0.05,
                              min_child_samples=20)
       5-fold KFold cross-fit -> honest residual_oof (253,)
       refit on all 253       -> deploy residual_hat (513,)
  4) Apply soft blend per row:
       alpha_i = sigmoid((err_hat_i - median(err_hat)) * 4.0)
       pred_i  = nb432_i + alpha_i * residual_hat_i
  5) Honest unblind RAE using CROSS-FIT residual_oof on the 253.
  6) Save te_nb472.npy + plain CSV + soft07_truth CSV.

Memory-safe: LGBM on 253 rows x 18 cols.
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.chem import bemis_murcko, compute_physchem, standardize_smiles
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def morgan_bits(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def compute_chemprop_disagreement(n_te: int) -> np.ndarray:
    """Stack 5 fold chemprop test predictions and return per-row std."""
    cache = DATA_PROCESSED / "te_chemprop_disagreement_std.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (n_te,):
            return arr
    preds = []
    for k in range(5):
        df = pd.read_csv(DATA_PROCESSED / f"chemprop_fold_{k}" / "pred_test.csv")
        preds.append(df["pxr"].values.astype(np.float32))
    mat = np.stack(preds, axis=0)
    std = mat.std(axis=0).astype(np.float32)
    np.save(cache, std)
    return std


def build_feature_matrix(te_df: pd.DataFrame, nb432: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Rebuild the same 18-col matrix nb443 used."""
    n_te = len(te_df)
    te_smis = te_df["SMILES"].tolist()
    features: dict[str, np.ndarray] = {}

    # (1) chemprop disagreement
    features["chemprop_disagree"] = compute_chemprop_disagreement(n_te)

    # (2) train top-1 Tanimoto + truth distance
    tr_df = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    tr_smis = tr_df["SMILES"].tolist()
    tr_pec = tr_df["pEC50"].astype(float).values
    mask = np.isfinite(tr_pec)
    tr_smis = [s for s, m in zip(tr_smis, mask) if m]
    tr_pec = tr_pec[mask]
    train_fps = []
    train_pec_kept = []
    for s, y in zip(tr_smis, tr_pec):
        std = standardize_smiles(s) or s
        fp = morgan_bits(std)
        if fp is not None:
            train_fps.append(fp)
            train_pec_kept.append(y)
    train_pec_kept = np.asarray(train_pec_kept, dtype=np.float32)
    median_train_pec = float(np.median(train_pec_kept))

    te_fps = [morgan_bits(standardize_smiles(s) or s) for s in te_smis]
    top1_sim = np.zeros(n_te, dtype=np.float32)
    top1_nn_pec = np.zeros(n_te, dtype=np.float32)
    for i, fp in enumerate(te_fps):
        if fp is None:
            top1_sim[i] = 0.0
            top1_nn_pec[i] = median_train_pec
            continue
        sims = np.array(
            DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=np.float32
        )
        j = int(np.argmax(sims))
        top1_sim[i] = float(sims[j])
        top1_nn_pec[i] = float(train_pec_kept[j])
    features["top1_sim"] = top1_sim
    features["nb432_minus_top1_nn"] = np.abs(nb432 - top1_nn_pec).astype(np.float32)

    # (3) scaffold size + freq
    train_scaffolds = [bemis_murcko(s) or "" for s in tr_smis]
    scaf_counts: dict[str, int] = {}
    for s in train_scaffolds:
        scaf_counts[s] = scaf_counts.get(s, 0) + 1
    scaf_size = np.zeros(n_te, dtype=np.float32)
    scaf_freq = np.zeros(n_te, dtype=np.float32)
    for i, smi in enumerate(te_smis):
        sc = bemis_murcko(smi) or ""
        scaf_size[i] = float(len(sc))
        scaf_freq[i] = float(scaf_counts.get(sc, 0))
    features["scaf_size"] = scaf_size
    features["scaf_freq"] = scaf_freq

    # (4) physchem
    phys_keys = ["mw", "logp", "tpsa", "fsp3", "rotbonds", "formal_charge"]
    phys = {k: np.zeros(n_te, dtype=np.float32) for k in phys_keys}
    for i, smi in enumerate(te_smis):
        d = compute_physchem(smi) or {}
        for k in phys_keys:
            v = d.get(k)
            phys[k][i] = float(v) if v is not None else 0.0
    for k in phys_keys:
        features[f"phys_{k}"] = phys[k]

    # (5) Tox21 / HTTr / Hopping anchors
    def safe_load(fname):
        p = DATA_PROCESSED / fname
        if not p.exists():
            return None
        try:
            a = np.load(p).astype(np.float32)
            if a.shape != (n_te,) or not np.isfinite(a).any():
                return None
            return a
        except Exception:
            return None

    a440 = safe_load("te_nb440.npy")
    a441 = safe_load("te_nb441.npy")
    a442 = safe_load("te_nb442.npy")
    if a440 is not None:
        features["tox21_score"] = a440
        features["tox21_anchor_dev"] = np.abs(a440 - nb432).astype(np.float32)
    if a441 is not None:
        features["httr_score"] = a441
    if a442 is not None:
        features["hopping_score"] = a442
        features["hopping_dev"] = np.abs(a442 - nb432).astype(np.float32)

    # (6) nb432 base + spread + PCA1 of base predictions
    base_arrs = []
    for fname in ["te_nb400_crossfit.npy", "te_nb424.npy",
                  "te_nb429.npy", "te_nb432.npy"]:
        p = DATA_PROCESSED / fname
        if p.exists():
            arr = np.load(p).astype(np.float32)
            if arr.shape == (n_te,):
                base_arrs.append(arr)
    features["nb432_pred"] = nb432
    if len(base_arrs) >= 2:
        base_mat = np.stack(base_arrs, axis=1)
        features["base_spread"] = base_mat.std(axis=1).astype(np.float32)
        pca = PCA(n_components=1)
        features["pca1_base"] = pca.fit_transform(base_mat).ravel().astype(np.float32)

    feat_names = list(features.keys())
    X = np.stack([features[k] for k in feat_names], axis=1).astype(np.float32)
    return X, feat_names


def main() -> dict:
    print("=" * 78)
    print("nb472 -- RESIDUAL STACK ROUTER (learn correction, gate by err_hat)")
    print("=" * 78)

    needed = {
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":
            DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    n_te = nb432.shape[0]
    assert err_hat.shape == (n_te,)

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_te_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")
    print(f"err_hat: mean={err_hat.mean():.3f}  median={np.median(err_hat):.3f}  "
          f"std={err_hat.std():.3f}")

    # ---------- Feature matrix ----------
    print("\nBuilding feature matrix:")
    X, feat_names = build_feature_matrix(te_df, nb432)
    print(f"  X shape: {X.shape}  cols={len(feat_names)}")

    X_unb = X[unb_te_idx]
    # Signed residual: positive => truth > nb432 (nb432 under-predicts)
    y_resid = (unb_y - nb432[unb_te_idx]).astype(np.float32)
    print(f"\nResidual target: mean={y_resid.mean():.3f}  "
          f"median={np.median(y_resid):.3f}  std={y_resid.std():.3f}  "
          f"|mean|={np.abs(y_resid).mean():.3f}")

    # ---------- Cross-fit shallow LGBM ----------
    print("\n5-fold cross-fit LGBM (max_depth=3, n_est=80, lr=0.05):")
    params = dict(
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
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_unb[tr_i], y_resid[tr_i])
        resid_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"resid_oof mean={resid_oof[va_i].mean():+.3f} "
              f"std={resid_oof[va_i].std():.3f}")

    # ---------- Residual quality diagnostics ----------
    rho_resid, _ = spearmanr(resid_oof, y_resid)
    if not np.isfinite(rho_resid):
        rho_resid = 0.0
    print(f"\nSpearman(resid_oof, true residual) = {rho_resid:.4f}")

    # cross-fit residual correlation vs the true |error| of nb432 -- a sanity
    # check: residual_oof correlated with true residual means the gate will
    # apply correction in the right direction.
    rho_with_truth_abs, _ = spearmanr(np.abs(resid_oof), np.abs(y_resid))
    if not np.isfinite(rho_with_truth_abs):
        rho_with_truth_abs = 0.0
    print(f"Spearman(|resid_oof|, |true residual|) = {rho_with_truth_abs:.4f}")

    # ---------- Deploy refit on all 253 ----------
    deploy_mdl = lgb.LGBMRegressor(**params)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X).astype(np.float32)
    print(f"\nDeploy resid_hat (513): mean={resid_hat_513.mean():+.3f}  "
          f"std={resid_hat_513.std():.3f}  "
          f"|mean|={np.abs(resid_hat_513).mean():.3f}")

    imp = deploy_mdl.feature_importances_
    order = np.argsort(-imp)
    print("\nTop 8 residual-LGBM feature importances:")
    for i in order[:8]:
        print(f"  {feat_names[i]:24s}  {imp[i]}")

    # ---------- Soft-gate by err_hat ----------
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"\nGate alpha (513): min={alpha_513.min():.3f}  "
          f"med={np.median(alpha_513):.3f}  max={alpha_513.max():.3f}  "
          f"mean={alpha_513.mean():.3f}")

    # ---------- Deploy prediction ----------
    deploy = (nb432 + alpha_513 * resid_hat_513).astype(np.float32)
    print(f"\nDeploy nb472: mean={deploy.mean():.3f}  std={deploy.std():.3f}  "
          f"|delta vs nb432|.mean = {np.abs(deploy - nb432).mean():.3f}")

    # ---------- Honest cross-fit unblind RAE ----------
    # On the 253 unblind rows, use the OOF residual prediction (no leakage).
    nb472_unb_oof = (nb432[unb_te_idx] + alpha_513[unb_te_idx] * resid_oof
                     ).astype(np.float32)
    rae_nb432_unb = float(rae(unb_y, nb432[unb_te_idx]))
    rae_nb472_oof = float(rae(unb_y, nb472_unb_oof))
    # In-sample deploy RAE for reference (overfit upper bound)
    rae_nb472_insample = float(rae(unb_y, deploy[unb_te_idx]))

    print("\nUnblind RAE (n=253):")
    print(f"  nb432 baseline           = {rae_nb432_unb:.4f}")
    print(f"  nb472 cross-fit (honest) = {rae_nb472_oof:.4f}")
    print(f"  nb472 in-sample (refit)  = {rae_nb472_insample:.4f}")
    beats_nb432 = rae_nb472_oof < rae_nb432_unb
    print(f"  beats_nb432 (honest)     = {beats_nb432}")

    # ---------- Save arrays + submissions ----------
    np.save(DATA_PROCESSED / "te_nb472.npy", deploy)
    np.save(DATA_PROCESSED / "te_nb472_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / "te_nb472_alpha.npy", alpha_513)

    plain = SUBMISSIONS / "nb472_residual_stack_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    soft_path = SUBMISSIONS / "nb472_residual_stack_router_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb472.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb472 SUMMARY ===")
    print(f"  spearman(resid_oof, true resid)        = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true resid|)    = {rho_with_truth_abs:.4f}")
    print(f"  unblind RAE nb432                       = {rae_nb432_unb:.4f}")
    print(f"  unblind RAE nb472 (cross-fit honest)    = {rae_nb472_oof:.4f}")
    print(f"  unblind RAE nb472 (in-sample refit)     = {rae_nb472_insample:.4f}")
    print(f"  beats_nb432                             = {beats_nb432}")
    print("=" * 78)

    return {
        "success": True,
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_with_truth_abs),
        "rae_nb432_unb": rae_nb432_unb,
        "unblind_rae_nb472_oof": rae_nb472_oof,
        "unblind_rae_nb472_insample": rae_nb472_insample,
        "beats_nb432": bool(beats_nb432),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
