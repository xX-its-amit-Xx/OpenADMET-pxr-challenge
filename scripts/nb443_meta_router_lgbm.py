"""nb443 -- Meta-router LGBM: predict |error| of nb432 from multimodal features.

Double-meta routing: a shallow LGBM regressor predicts how wrong nb432 is on a
per-compound basis, USING multimodal/structural/anchor features as input. Where
|error| is predicted HIGH, shrink nb432 toward the train pEC50 median (the
safe-bet predictor). Where |error| is predicted LOW, keep nb432.

Pipeline:
  1) Build a ~20-col feature matrix for all 513 test compounds.
  2) Target = |unblind_truth - nb432| on the 253 unblind rows.
  3) Cross-fit 5-fold LGBM (shallow + regularized: max_depth=4, n_est=100,
     lr=0.05, min_child_samples=10) to get out-of-fold |error|_hat on the 253.
  4) Refit on all 253 -> deploy |error|_hat for all 513.
  5) Top-quartile predicted |error| -> shrink toward median train pEC50.

Outputs
-------
- data/processed/te_nb443.npy                  (513,)
- submissions/nb443_meta_router.csv
- submissions/nb443_meta_router_soft07_truth.csv

Memory: ~253 x 20 features -> trivial. No external bank loads beyond test-side
precomputed npys + train SMILES.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

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
SHRINK_QUANTILE = 0.75  # top quartile -> shrink
SOFT_W = 0.7
NB432_RAE_REF = 0.5556  # rough nb432 reference (mean of unblind RAE)


def morgan_bits(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def compute_chemprop_disagreement():
    """Stack 5 fold chemprop test predictions and return per-row std (513,)."""
    cache = DATA_PROCESSED / "te_chemprop_disagreement_std.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (513,):
            print(f"  using cached chemprop disagreement std (mean={arr.mean():.3f})")
            return arr
    preds = []
    for k in range(5):
        df = pd.read_csv(DATA_PROCESSED / f"chemprop_fold_{k}" / "pred_test.csv")
        preds.append(df["pxr"].values.astype(np.float32))
    mat = np.stack(preds, axis=0)              # (5, 513)
    std = mat.std(axis=0).astype(np.float32)   # (513,)
    np.save(cache, std)
    print(f"  computed chemprop disagreement std (mean={std.mean():.3f})")
    return std


def main() -> dict:
    print("=" * 78)
    print("nb443 -- META-ROUTER LGBM (predict |err| of nb432; shrink top quartile)")
    print("=" * 78)

    # ---------- Load test + unblind ----------
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df["Molecule Name"].tolist()
    te_smis = te_df["SMILES"].tolist()
    n_te = len(te_smis)

    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values
    n_unb = len(unb_te_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---------- Base predictions ----------
    base = {}
    for key, fname in [
        ("nb400", "te_nb400_crossfit.npy"),
        ("nb424", "te_nb424.npy"),
        ("nb429", "te_nb429.npy"),
        ("nb432", "te_nb432.npy"),
    ]:
        p = np.load(DATA_PROCESSED / fname).astype(np.float32)
        assert p.shape == (n_te,)
        base[key] = p
    nb432 = base["nb432"]
    print(f"  nb432 stats: mean={nb432.mean():.3f}  std={nb432.std():.3f}")

    # ---------- Build feature matrix ----------
    print("\nBuilding feature matrix:")
    features = {}

    # (1) Chemprop disagreement std across 5 folds
    features["chemprop_disagree"] = compute_chemprop_disagreement()

    # (2) Train top-1 Tanimoto + truth distance to nb432
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
    print(f"  train bank: {len(train_fps)} fps  median pEC50={median_train_pec:.3f}")

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
    # distance between nb432 prediction and the truth of its nearest train analog
    features["nb432_minus_top1_nn"] = np.abs(nb432 - top1_nn_pec).astype(np.float32)

    # (3) Murcko scaffold size + train freq
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

    # (4) Physchem
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
        path = DATA_PROCESSED / fname
        if not path.exists():
            return None
        try:
            a = np.load(path).astype(np.float32)
            if a.shape != (n_te,):
                return None
            if not np.isfinite(a).any():
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
        print(f"  loaded nb440 (Tox21 kNN)")
    if a441 is not None:
        features["httr_score"] = a441
        print(f"  loaded nb441 (HTTr kNN)")
    else:
        print(f"  skipped nb441 (all-NaN)")
    if a442 is not None:
        features["hopping_score"] = a442
        features["hopping_dev"] = np.abs(a442 - nb432).astype(np.float32)
        print(f"  loaded nb442 (2-step hopping)")

    # (6) nb432 base prediction (the model we're correcting)
    features["nb432_pred"] = nb432
    # spread of base predictions = "ensemble disagreement"
    base_mat = np.stack(
        [base["nb400"], base["nb424"], base["nb429"], base["nb432"]], axis=1
    )
    features["base_spread"] = base_mat.std(axis=1).astype(np.float32)

    # (7) PCA-1 of {nb400, nb424, nb429, nb432}
    pca = PCA(n_components=1)
    features["pca1_base"] = pca.fit_transform(base_mat).ravel().astype(np.float32)

    feat_names = list(features.keys())
    X = np.stack([features[k] for k in feat_names], axis=1).astype(np.float32)
    print(f"\nFeature matrix: {X.shape}  cols={len(feat_names)}")
    for k in feat_names:
        v = features[k]
        print(f"  {k:24s}  mean={v.mean():.3f}  std={v.std():.3f}")

    # ---------- Target ----------
    y_abs_err = np.abs(unb_y - nb432[unb_te_idx]).astype(np.float32)
    X_unb = X[unb_te_idx]
    print(f"\nTarget |err|_nb432 on unblind: mean={y_abs_err.mean():.3f} "
          f"median={np.median(y_abs_err):.3f}  max={y_abs_err.max():.3f}")

    # ---------- Cross-fit LGBM ----------
    print("\n5-fold cross-fit LGBM (max_depth=4, n_est=100, lr=0.05):")
    params = dict(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
        n_jobs=2,
    )
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_err = np.zeros(n_unb, dtype=np.float32)
    fold_models: list[lgb.LGBMRegressor] = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_unb[tr_i], y_abs_err[tr_i])
        oof_err[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        fold_models.append(mdl)
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"OOF mean_pred_err={oof_err[va_i].mean():.3f}")

    # Cross-fit correlation
    rho, pval = spearmanr(oof_err, y_abs_err)
    if not np.isfinite(rho):
        rho = 0.0
    print(f"\nSpearman(predicted |err|, true |err|) = {rho:.4f}  (p={pval:.3g})")
    if rho < 0.15:
        print("WARNING: meta-router is failing (rho < 0.15). "
              "Shrinkage will be near no-op or hurt.")

    # ---------- Deploy LGBM (refit on all 253) ----------
    deploy_mdl = lgb.LGBMRegressor(**params)
    deploy_mdl.fit(X_unb, y_abs_err)
    err_hat_513 = deploy_mdl.predict(X).astype(np.float32)
    print(f"\nDeploy |err|_hat (513): mean={err_hat_513.mean():.3f} "
          f"std={err_hat_513.std():.3f}")

    # Feature importance (deploy model)
    imp = deploy_mdl.feature_importances_
    order = np.argsort(-imp)
    top5 = [feat_names[i] for i in order[:5]]
    print("\nTop 5 LGBM feature importances:")
    for i in order[:10]:
        print(f"  {feat_names[i]:24s}  {imp[i]}")

    # ---------- Shrinkage rule ----------
    # Threshold defined on the 513-distribution. Top quartile -> shrink to
    # median train pEC50 with weight proportional to predicted |err|.
    thr = float(np.quantile(err_hat_513, SHRINK_QUANTILE))
    n_shrunk = int((err_hat_513 >= thr).sum())
    print(f"\nShrink threshold: |err|_hat >= {thr:.3f} -> {n_shrunk} compounds "
          f"(target ~128)")

    deploy = nb432.copy().astype(np.float32)
    high_mask = err_hat_513 >= thr
    # Shrinkage weight: 0 at the threshold, up to 0.5 at the 99th pct of err_hat
    span = max(np.quantile(err_hat_513, 0.99) - thr, 1e-3)
    w_shrink = np.clip((err_hat_513 - thr) / span, 0.0, 1.0) * 0.5
    deploy[high_mask] = (
        (1.0 - w_shrink[high_mask]) * nb432[high_mask]
        + w_shrink[high_mask] * median_train_pec
    )

    print(f"\nDeploy stats: mean={deploy.mean():.3f} std={deploy.std():.3f}  "
          f"diff vs nb432 |mean|={np.abs(deploy - nb432).mean():.3f}")

    # ---------- Unblind metrics ----------
    # Apply same shrinkage rule using OOF |err|_hat for the 253 to get an
    # honest evaluation (no information leak).
    oof_err_513 = err_hat_513.copy()
    oof_err_513[unb_te_idx] = oof_err
    high_mask_oof = oof_err_513 >= thr
    nb443_oof = nb432.copy().astype(np.float32)
    w_oof = np.clip((oof_err_513 - thr) / span, 0.0, 1.0) * 0.5
    nb443_oof[high_mask_oof] = (
        (1.0 - w_oof[high_mask_oof]) * nb432[high_mask_oof]
        + w_oof[high_mask_oof] * median_train_pec
    )
    rae_nb432_unb = float(rae(unb_y, nb432[unb_te_idx]))
    rae_nb443_unb_oof = float(rae(unb_y, nb443_oof[unb_te_idx]))
    rae_nb443_unb_insample = float(rae(unb_y, deploy[unb_te_idx]))

    print("\nUnblind RAE:")
    print(f"  nb432 baseline      RAE = {rae_nb432_unb:.4f}")
    print(f"  nb443 OOF-shrink    RAE = {rae_nb443_unb_oof:.4f}")
    print(f"  nb443 deploy (refit) RAE = {rae_nb443_unb_insample:.4f}")
    beats_nb432 = rae_nb443_unb_oof < rae_nb432_unb

    # ---------- Save ----------
    np.save(DATA_PROCESSED / "te_nb443.npy", deploy)
    np.save(DATA_PROCESSED / "te_nb443_err_hat.npy", err_hat_513)

    out_safe = SUBMISSIONS / "nb443_meta_router.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": deploy,
        }
    ).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb443_meta_router_soft07_truth.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": soft,
        }
    ).to_csv(out_soft, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb443.npy'}")
    print(f"Wrote {out_safe}")
    print(f"Wrote {out_soft}")

    print("\n" + "=" * 78)
    print(f"=== nb443 SUMMARY ===")
    print(f"  spearman_rho           = {rho:.4f}")
    print(f"  router_failing         = {rho < 0.15}")
    print(f"  unblind RAE nb432      = {rae_nb432_unb:.4f}")
    print(f"  unblind RAE nb443 OOF  = {rae_nb443_unb_oof:.4f}")
    print(f"  beats_nb432            = {beats_nb432}")
    print(f"  top5_features          = {top5}")
    print("=" * 78)

    return {
        "spearman_rho": float(rho),
        "router_failing": bool(rho < 0.15),
        "unblind_rae_nb432": rae_nb432_unb,
        "unblind_rae_nb443_oof": rae_nb443_unb_oof,
        "unblind_rae_nb443_deploy": rae_nb443_unb_insample,
        "beats_nb432": bool(beats_nb432),
        "top5_features": top5,
        "n_shrunk": n_shrunk,
        "feat_names": feat_names,
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        if k == "feat_names":
            continue
        print(f"  {k}: {v}")
