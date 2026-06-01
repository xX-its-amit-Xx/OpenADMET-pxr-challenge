"""nb481 -- RESIDUAL STACK ROUTER (EXTENDED FEATURES).

Same architecture as nb472 (signed-residual LGBM cross-fit + sigmoid err_hat
gate) but with an EXTENDED feature matrix: nb472's 18 base columns plus 12
new cliff / disagreement / extremity signals.

NEW features (12):
  (A) train-NN truth-spread   : std of top-10 NN train pec50 (cliff signal)
  (B) train-NN truth-mean     : sim-weighted mean of top-10 NN train pec50
  (C) scaffold-cluster id     : hash(murcko_scaffold) -> int32
  (D) scaffold-cluster size   : count of this scaffold in train
  (E) counter null pred       : te_lgbm_counter (oof / nb411)
  (F) counter delta vs nb432  : nb432 - counter_pred
  (G) BindingDB anchor        : te_lgbm_bindingdb
  (H) Tox21 anchor            : te_nb417_tox21
  (I) nb432 - chemprop_aux    : disagreement vs aux GNN
  (J) nb432 - nb411 delta     : counter-assay disagreement
  (K) |nb432 - median train|  : extremeness vs train central value
  (L) physchem extremity z    : sum |z| across 6 physchem (vs train stats)

Missing sources -> zeros, logged.

Hyperparams identical to nb472:
  LGBM(max_depth=3, n_est=80, lr=0.05, num_leaves=8, min_child_samples=20,
       reg_lambda=1.0)
  5-fold KFold on the 253 unblind rows -> honest residual_oof
  sigmoid((err_hat - median)*4) gate

Target: cross-fit unblind RAE < 0.5410 (nb472).
Outputs: te_nb481.npy, nb481_pred_oof.npy, nb481_resid_oof.npy.
"""
from __future__ import annotations

import os
import sys
import warnings
import hashlib

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

MISSING_LOG: list[str] = []


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def morgan_bits(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def hash_to_int(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def compute_chemprop_disagreement(n_te: int) -> np.ndarray:
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


def safe_load_te(fname: str, n_te: int) -> np.ndarray | None:
    p = DATA_PROCESSED / fname
    if not p.exists():
        MISSING_LOG.append(fname)
        return None
    try:
        a = np.load(p).astype(np.float32)
        if a.shape != (n_te,) or not np.isfinite(a).any():
            MISSING_LOG.append(f"{fname} (bad shape/all-nan)")
            return None
        return a
    except Exception:
        MISSING_LOG.append(f"{fname} (load err)")
        return None


def build_feature_matrix(
    te_df: pd.DataFrame, nb432: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    n_te = len(te_df)
    te_smis = te_df["SMILES"].tolist()
    features: dict[str, np.ndarray] = {}

    # ---------- (1) chemprop disagreement ----------
    features["chemprop_disagree"] = compute_chemprop_disagreement(n_te)

    # ---------- (2) train side: fps, pec, scaffolds ----------
    tr_df = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    tr_smis_raw = tr_df["SMILES"].tolist()
    tr_pec_raw = tr_df["pEC50"].astype(float).values
    mask = np.isfinite(tr_pec_raw)
    tr_smis = [s for s, m in zip(tr_smis_raw, mask) if m]
    tr_pec = tr_pec_raw[mask]
    train_fps: list = []
    train_pec_kept: list = []
    train_scaffolds: list = []
    for s, y in zip(tr_smis, tr_pec):
        std = standardize_smiles(s) or s
        fp = morgan_bits(std)
        if fp is None:
            continue
        train_fps.append(fp)
        train_pec_kept.append(y)
        train_scaffolds.append(bemis_murcko(s) or "")
    train_pec_kept = np.asarray(train_pec_kept, dtype=np.float32)
    median_train_pec = float(np.median(train_pec_kept))

    # scaffold counts
    scaf_counts: dict[str, int] = {}
    for s in train_scaffolds:
        scaf_counts[s] = scaf_counts.get(s, 0) + 1

    # ---------- (3) test side: fps, sims, NN aggregates ----------
    te_fps = [morgan_bits(standardize_smiles(s) or s) for s in te_smis]
    top1_sim = np.zeros(n_te, dtype=np.float32)
    top1_nn_pec = np.zeros(n_te, dtype=np.float32)
    nn10_spread = np.zeros(n_te, dtype=np.float32)   # NEW (A)
    nn10_mean = np.zeros(n_te, dtype=np.float32)     # NEW (B)
    K_NN = 10
    for i, fp in enumerate(te_fps):
        if fp is None:
            top1_sim[i] = 0.0
            top1_nn_pec[i] = median_train_pec
            nn10_spread[i] = 0.0
            nn10_mean[i] = median_train_pec
            continue
        sims = np.array(
            DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=np.float32
        )
        j = int(np.argmax(sims))
        top1_sim[i] = float(sims[j])
        top1_nn_pec[i] = float(train_pec_kept[j])
        if len(sims) >= K_NN:
            top_idx = np.argpartition(-sims, K_NN - 1)[:K_NN]
        else:
            top_idx = np.argsort(-sims)
        nn_pec = train_pec_kept[top_idx]
        nn_sims = sims[top_idx]
        nn10_spread[i] = float(np.std(nn_pec))
        wsum = float(nn_sims.sum())
        if wsum > 1e-6:
            nn10_mean[i] = float((nn_pec * nn_sims).sum() / wsum)
        else:
            nn10_mean[i] = float(nn_pec.mean())
    features["top1_sim"] = top1_sim
    features["nb432_minus_top1_nn"] = np.abs(nb432 - top1_nn_pec).astype(np.float32)
    features["nn10_spread"] = nn10_spread          # NEW (A)
    features["nn10_wmean"] = nn10_mean             # NEW (B)
    features["nn10_dev_vs_nb432"] = np.abs(nb432 - nn10_mean).astype(np.float32)

    # ---------- (4) scaffold features ----------
    scaf_size = np.zeros(n_te, dtype=np.float32)
    scaf_freq = np.zeros(n_te, dtype=np.float32)
    scaf_id = np.zeros(n_te, dtype=np.float32)        # NEW (C)
    scaf_train_size = np.zeros(n_te, dtype=np.float32)  # NEW (D)
    for i, smi in enumerate(te_smis):
        sc = bemis_murcko(smi) or ""
        scaf_size[i] = float(len(sc))
        scaf_freq[i] = float(scaf_counts.get(sc, 0))
        scaf_id[i] = float(hash_to_int(sc) % 100000) if sc else 0.0
        scaf_train_size[i] = float(scaf_counts.get(sc, 0))
    features["scaf_size"] = scaf_size
    features["scaf_freq"] = scaf_freq
    features["scaf_cluster_id"] = scaf_id              # NEW (C)
    features["scaf_train_count"] = scaf_train_size     # NEW (D)

    # ---------- (5) physchem + extremity z-score ----------
    phys_keys = ["mw", "logp", "tpsa", "fsp3", "rotbonds", "formal_charge"]
    phys = {k: np.zeros(n_te, dtype=np.float32) for k in phys_keys}
    for i, smi in enumerate(te_smis):
        d = compute_physchem(smi) or {}
        for k in phys_keys:
            v = d.get(k)
            phys[k][i] = float(v) if v is not None else 0.0
    for k in phys_keys:
        features[f"phys_{k}"] = phys[k]

    # extremity z: compute train physchem stats then sum |z| across 6
    tr_phys = {k: np.zeros(len(tr_smis), dtype=np.float32) for k in phys_keys}
    for i, s in enumerate(tr_smis):
        d = compute_physchem(s) or {}
        for k in phys_keys:
            v = d.get(k)
            tr_phys[k][i] = float(v) if v is not None else 0.0
    extremity = np.zeros(n_te, dtype=np.float32)
    for k in phys_keys:
        mu = float(np.nanmean(tr_phys[k]))
        sd = float(np.nanstd(tr_phys[k])) + 1e-6
        extremity += np.abs((phys[k] - mu) / sd).astype(np.float32)
    features["physchem_extremity_z"] = extremity       # NEW (L)

    # ---------- (6) Tox21 / HTTr / Hopping ----------
    a440 = safe_load_te("te_nb440.npy", n_te)
    a441 = safe_load_te("te_nb441.npy", n_te)
    a442 = safe_load_te("te_nb442.npy", n_te)
    if a440 is not None:
        features["tox21_score"] = a440
        features["tox21_anchor_dev"] = np.abs(a440 - nb432).astype(np.float32)
    if a441 is not None:
        features["httr_score"] = a441
    if a442 is not None:
        features["hopping_score"] = a442
        features["hopping_dev"] = np.abs(a442 - nb432).astype(np.float32)

    # ---------- (7) nb432 base + spread + PCA1 ----------
    base_arrs = []
    for fname in [
        "te_nb400_crossfit.npy", "te_nb424.npy",
        "te_nb429.npy", "te_nb432.npy",
    ]:
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

    # ---------- (8) NEW counter / aux / bindingdb / tox21 anchors ----------
    counter = safe_load_te("te_nb411.npy", n_te)
    if counter is None:
        counter = safe_load_te("te_oof_lgbm_counter_soft.npy", n_te)
    if counter is not None:
        features["counter_pred"] = counter                    # NEW (E)
        features["counter_delta_vs_nb432"] = (nb432 - counter).astype(np.float32)  # NEW (F)
        features["nb432_minus_nb411_abs"] = np.abs(nb432 - counter).astype(np.float32)  # NEW (J)
    else:
        features["counter_pred"] = np.zeros(n_te, dtype=np.float32)
        features["counter_delta_vs_nb432"] = np.zeros(n_te, dtype=np.float32)
        features["nb432_minus_nb411_abs"] = np.zeros(n_te, dtype=np.float32)

    bdb = safe_load_te("te_lgbm_bindingdb.npy", n_te)
    if bdb is not None:
        features["bindingdb_anchor"] = bdb                    # NEW (G)
        features["bindingdb_dev_vs_nb432"] = np.abs(bdb - nb432).astype(np.float32)
    else:
        features["bindingdb_anchor"] = np.zeros(n_te, dtype=np.float32)
        features["bindingdb_dev_vs_nb432"] = np.zeros(n_te, dtype=np.float32)

    tox21 = safe_load_te("te_nb417_tox21.npy", n_te)
    if tox21 is None:
        tox21 = safe_load_te("tox21_bio_te.npy", n_te)
    if tox21 is not None:
        features["tox21_anchor_nb417"] = tox21                # NEW (H)
    else:
        features["tox21_anchor_nb417"] = np.zeros(n_te, dtype=np.float32)

    aux = safe_load_te("te_chemprop_aux.npy", n_te)
    if aux is not None:
        features["nb432_minus_aux_signed"] = (nb432 - aux).astype(np.float32)  # NEW (I)
        features["nb432_minus_aux_abs"] = np.abs(nb432 - aux).astype(np.float32)
    else:
        features["nb432_minus_aux_signed"] = np.zeros(n_te, dtype=np.float32)
        features["nb432_minus_aux_abs"] = np.zeros(n_te, dtype=np.float32)

    # ---------- (9) NEW extremeness vs median train pec ----------
    features["nb432_extremeness"] = np.abs(
        nb432 - median_train_pec
    ).astype(np.float32)                                   # NEW (K)

    feat_names = list(features.keys())
    X = np.stack([features[k] for k in feat_names], axis=1).astype(np.float32)
    return X, feat_names


def main() -> dict:
    print("=" * 78)
    print("nb481 -- RESIDUAL STACK ROUTER EXTENDED (30+ cols)")
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
        print("MISSING (required):", missing)
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
    print("\nBuilding extended feature matrix:")
    X, feat_names = build_feature_matrix(te_df, nb432)
    print(f"  X shape: {X.shape}  cols={len(feat_names)}")
    if MISSING_LOG:
        print("  Missing feature sources (filled with 0):")
        for f in MISSING_LOG:
            print(f"    - {f}")
    else:
        print("  All optional feature sources loaded successfully.")

    # NaN safety
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    X_unb = X[unb_te_idx]
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

    # ---------- Diagnostics ----------
    rho_resid, _ = spearmanr(resid_oof, y_resid)
    if not np.isfinite(rho_resid):
        rho_resid = 0.0
    print(f"\nSpearman(resid_oof, true residual) = {rho_resid:.4f}")
    rho_abs, _ = spearmanr(np.abs(resid_oof), np.abs(y_resid))
    if not np.isfinite(rho_abs):
        rho_abs = 0.0
    print(f"Spearman(|resid_oof|, |true residual|) = {rho_abs:.4f}")

    # ---------- Deploy ----------
    deploy_mdl = lgb.LGBMRegressor(**params)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X).astype(np.float32)
    print(f"\nDeploy resid_hat (513): mean={resid_hat_513.mean():+.3f}  "
          f"std={resid_hat_513.std():.3f}  "
          f"|mean|={np.abs(resid_hat_513).mean():.3f}")

    imp = deploy_mdl.feature_importances_
    order = np.argsort(-imp)
    print("\nTop 10 residual-LGBM feature importances:")
    top5_names: list[str] = []
    for rk, i in enumerate(order[:10]):
        print(f"  {feat_names[i]:28s}  {imp[i]}")
        if rk < 5:
            top5_names.append(feat_names[i])

    # ---------- Soft-gate ----------
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"\nGate alpha (513): min={alpha_513.min():.3f}  "
          f"med={np.median(alpha_513):.3f}  max={alpha_513.max():.3f}  "
          f"mean={alpha_513.mean():.3f}")

    deploy = (nb432 + alpha_513 * resid_hat_513).astype(np.float32)
    print(f"\nDeploy nb481: mean={deploy.mean():.3f}  std={deploy.std():.3f}  "
          f"|delta vs nb432|.mean = {np.abs(deploy - nb432).mean():.3f}")

    # ---------- Honest cross-fit unblind RAE ----------
    nb481_unb_oof = (nb432[unb_te_idx] + alpha_513[unb_te_idx] * resid_oof
                     ).astype(np.float32)
    rae_nb432_unb = float(rae(unb_y, nb432[unb_te_idx]))
    rae_nb481_oof = float(rae(unb_y, nb481_unb_oof))
    rae_nb481_insample = float(rae(unb_y, deploy[unb_te_idx]))

    print("\nUnblind RAE (n=253):")
    print(f"  nb432 baseline           = {rae_nb432_unb:.4f}")
    print(f"  nb481 cross-fit (honest) = {rae_nb481_oof:.4f}")
    print(f"  nb481 in-sample (refit)  = {rae_nb481_insample:.4f}")
    NB472_TARGET = 0.5410
    beats_nb472 = rae_nb481_oof < NB472_TARGET
    beats_nb432 = rae_nb481_oof < rae_nb432_unb
    print(f"  beats nb472 (<0.5410)    = {beats_nb472}")
    print(f"  beats nb432              = {beats_nb432}")

    # ---------- Save arrays + submissions ----------
    np.save(DATA_PROCESSED / "te_nb481.npy", deploy)
    np.save(DATA_PROCESSED / "te_nb481_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / "te_nb481_alpha.npy", alpha_513)
    np.save(DATA_PROCESSED / "nb481_pred_oof.npy", nb481_unb_oof)
    np.save(DATA_PROCESSED / "nb481_resid_oof.npy", resid_oof)

    plain = SUBMISSIONS / "nb481_residual_router_extended.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    soft_path = SUBMISSIONS / "nb481_residual_router_extended_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb481.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb481_pred_oof.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb481_resid_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb481 SUMMARY ===")
    print(f"  n_features                              = {len(feat_names)}")
    print(f"  missing feature sources                 = {len(MISSING_LOG)}")
    print(f"  spearman(resid_oof, true resid)         = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true resid|)     = {rho_abs:.4f}")
    print(f"  unblind RAE nb432                        = {rae_nb432_unb:.4f}")
    print(f"  unblind RAE nb481 (cross-fit honest)     = {rae_nb481_oof:.4f}")
    print(f"  unblind RAE nb481 (in-sample refit)      = {rae_nb481_insample:.4f}")
    print(f"  beats nb472 target (0.5410)              = {beats_nb472}")
    print("=" * 78)

    return {
        "success": True,
        "n_features": len(feat_names),
        "missing_sources": MISSING_LOG,
        "top5_features": top5_names,
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "rae_nb432_unb": rae_nb432_unb,
        "unblind_rae_nb481_oof": rae_nb481_oof,
        "unblind_rae_nb481_insample": rae_nb481_insample,
        "beats_nb472": bool(beats_nb472),
        "beats_nb432": bool(beats_nb432),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
