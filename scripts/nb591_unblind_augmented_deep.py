"""nb591 -- UNBLIND-AUGMENTED DEEP LGBM BASE (orthogonal feature space).

Variant of nb590 with:
  - Different features: MACCS(167) + AtomPair(2048) + Avalon(512) = 2727 cols
    (concatenated; deliberately disjoint from nb590's Morgan+RDKit-desc).
  - Deeper LGBM: max_depth=8, num_leaves=128, n_est=400, lr=0.03,
    min_child_samples=10.
  - Same 5-fold KFold cross-fit on 253 unblind + 4139 train augmentation.

Hypothesis: a different feature space + different hyperparams gives a genuinely
orthogonal base model. Combine with the unblind-augmentation and we get a fresh
signal that blends well with nb562 (Morgan/RDKit-desc family).

Cached arrays used (skip if missing for unblind, computed live there):
  data/processed/tr_atompair.npy   (4139, 2048)  uint8
  data/processed/te_atompair.npy   (513,  2048)  uint8
  data/processed/tr_avalon512.npy  (4139, 512)   uint8
  data/processed/te_avalon512.npy  (513,  512)   uint8
  data/processed/te_maccs.npy      (513,  167)   uint8    [reused if present]

MACCS for train and unblind is computed live; AtomPair + Avalon for unblind
also computed live (no cached unblind arrays).

Target: cross-fit < 0.5065 standalone, OR < 0.50 in blend with nb562.
Memory budget: ~50 MB for (4392+513) x 2727 float32.
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
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb591"
NB562_RAE = 0.5065  # standalone target
NB562_BLEND_TARGET = 0.50  # blended target

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=128,
    min_child_samples=10,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)


# ----- Fingerprint helpers -----
def fp_maccs(smiles_list) -> np.ndarray:
    out = np.zeros((len(smiles_list), 167), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(m)
        for j in fp.GetOnBits():
            out[i, j] = 1
    return out


def fp_atompair(smiles_list, n_bits: int = 2048) -> np.ndarray:
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            m, nBits=n_bits
        )
        for j in fp.GetOnBits():
            out[i, j] = 1
    return out


def fp_avalon(smiles_list, n_bits: int = 512) -> np.ndarray:
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        fp = pyAvalonTools.GetAvalonFP(m, nBits=n_bits)
        for j in fp.GetOnBits():
            out[i, j] = 1
    return out


def _load_or_compute(path: Path, fn, smiles_list, label: str) -> np.ndarray:
    if path.exists():
        arr = np.load(path)
        print(f"  loaded {label} from cache: {arr.shape}")
        return arr
    print(f"  computing {label} ({len(smiles_list)} smiles)...")
    arr = fn(smiles_list)
    np.save(path, arr)
    print(f"  saved {label} -> {path}  {arr.shape}")
    return arr


def main() -> dict:
    print("=" * 78)
    print("nb591 -- UNBLIND-AUGMENTED DEEP LGBM (MACCS+AtomPair+Avalon)")
    print("=" * 78)

    needed = {
        "TRAIN":        DATA_RAW / "pxr-challenge_TRAIN.csv",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    train_df = pd.read_csv(needed["TRAIN"])
    test_df = pd.read_csv(needed["TEST_BLINDED"])
    unb_df = pd.read_csv(needed["UNBLINDED"])

    te_names = test_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(
        drop=True
    )
    unb_idx = np.array(
        [name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int
    )

    n_tr = len(train_df)
    n_te = len(test_df)
    n_unb = len(unb_df)
    print(f"\ntrain n={n_tr}  test n={n_te}  unblind n={n_unb}")

    # ---- Featurise: MACCS + AtomPair + Avalon ----
    print("\nFeaturising train...")
    tr_maccs = _load_or_compute(
        DATA_PROCESSED / "tr_maccs.npy", fp_maccs,
        train_df["SMILES"].tolist(), "tr_maccs"
    )
    tr_ap = _load_or_compute(
        DATA_PROCESSED / "tr_atompair.npy", fp_atompair,
        train_df["SMILES"].tolist(), "tr_atompair"
    )
    tr_av = _load_or_compute(
        DATA_PROCESSED / "tr_avalon512.npy", fp_avalon,
        train_df["SMILES"].tolist(), "tr_avalon512"
    )

    print("\nFeaturising test...")
    te_maccs_path = DATA_PROCESSED / "te_maccs.npy"
    if te_maccs_path.exists():
        cached = np.load(te_maccs_path)
        if cached.shape == (n_te, 167):
            te_maccs = cached
            print(f"  loaded te_maccs from cache: {te_maccs.shape}")
        else:
            te_maccs = _load_or_compute(
                DATA_PROCESSED / "te_maccs167.npy", fp_maccs,
                test_df["SMILES"].tolist(), "te_maccs167"
            )
    else:
        te_maccs = _load_or_compute(
            te_maccs_path, fp_maccs,
            test_df["SMILES"].tolist(), "te_maccs"
        )
    te_ap = _load_or_compute(
        DATA_PROCESSED / "te_atompair.npy", fp_atompair,
        test_df["SMILES"].tolist(), "te_atompair"
    )
    te_av = _load_or_compute(
        DATA_PROCESSED / "te_avalon512.npy", fp_avalon,
        test_df["SMILES"].tolist(), "te_avalon512"
    )

    print("\nFeaturising unblind (live, not cached)...")
    unb_smi = unb_df["SMILES"].tolist()
    unb_maccs = fp_maccs(unb_smi)
    unb_ap = fp_atompair(unb_smi)
    unb_av = fp_avalon(unb_smi)
    print(f"  unblind: maccs={unb_maccs.shape}  ap={unb_ap.shape}  "
          f"av={unb_av.shape}")

    # ---- Stack to (N, 2727) float32 ----
    X_tr = np.hstack([tr_maccs, tr_ap, tr_av]).astype(np.float32)
    X_unb = np.hstack([unb_maccs, unb_ap, unb_av]).astype(np.float32)
    X_te = np.hstack([te_maccs, te_ap, te_av]).astype(np.float32)
    print(f"\n  X_tr={X_tr.shape}  X_unb={X_unb.shape}  X_te={X_te.shape}  "
          f"(~{(X_tr.nbytes + X_unb.nbytes + X_te.nbytes) / 1e6:.1f} MB)")

    y_tr = train_df["pEC50"].astype(np.float64).values
    y_unb = unb_df["pEC50"].astype(np.float64).values

    # ---- 5-fold cross-fit on unblind ----
    print("\n" + "-" * 78)
    print("5-FOLD CROSS-FIT (train + (k-1)/k unblind  -> held-out unblind)")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_unb = np.full(n_unb, np.nan, dtype=np.float64)
    te_per_fold = np.zeros((N_FOLDS, n_te), dtype=np.float64)
    fold_raes: list[float] = []
    for fold, (tr_idx_unb, va_idx_unb) in enumerate(
        kf.split(np.arange(n_unb))
    ):
        X_fold = np.vstack([X_tr, X_unb[tr_idx_unb]])
        y_fold = np.concatenate([y_tr, y_unb[tr_idx_unb]])
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_fold, y_fold)
        oof_unb[va_idx_unb] = mdl.predict(X_unb[va_idx_unb])
        te_per_fold[fold] = mdl.predict(X_te)
        r_va = float(rae(y_unb[va_idx_unb], oof_unb[va_idx_unb]))
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_idx_unb):3d} -> "
              f"n_total={X_fold.shape[0]}  n_va={len(va_idx_unb):3d}  "
              f"RAE={r_va:.4f}")

    rae_oof = float(rae(y_unb, oof_unb))
    print(f"\nPooled cross-fit RAE = {rae_oof:.4f}  "
          f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})")
    print(f"nb562 standalone target: {NB562_RAE:.4f}  -> beats: "
          f"{rae_oof < NB562_RAE}")

    te_cv_bag = te_per_fold.mean(axis=0)

    # ---- Deploy on full (4139 + 253) ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit on 4139 train + 253 unblind = 4392 rows)")
    print("-" * 78)
    X_deploy = np.vstack([X_tr, X_unb])
    y_deploy = np.concatenate([y_tr, y_unb])
    mdl_deploy = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_deploy.fit(X_deploy, y_deploy)
    te_deploy = mdl_deploy.predict(X_te).astype(np.float32)
    print(f"  te_deploy mean/std = {te_deploy.mean():.3f} / "
          f"{te_deploy.std():.3f}")
    print(f"  te_cv_bag  mean/std = {te_cv_bag.mean():.3f} / "
          f"{te_cv_bag.std():.3f}")

    # ---- Blend with nb562 (load if present) ----
    blend_rae: float | None = None
    resid_corr: float | None = None
    nb562_oof_path = DATA_PROCESSED / "nb562_pred_oof.npy"
    if nb562_oof_path.exists():
        oof_562 = np.load(nb562_oof_path)
        if oof_562.shape == oof_unb.shape:
            # Residual correlation (orthogonality check)
            r1 = y_unb - oof_unb
            r2 = y_unb - oof_562.astype(np.float64)
            if r1.std() > 0 and r2.std() > 0:
                resid_corr = float(np.corrcoef(r1, r2)[0, 1])
            # Simple 50/50 blend
            blend_oof = 0.5 * oof_unb + 0.5 * oof_562.astype(np.float64)
            blend_rae = float(rae(y_unb, blend_oof))
            print(f"\n  blend 50/50 with nb562:  RAE={blend_rae:.4f}  "
                  f"(target<{NB562_BLEND_TARGET})")
            print(f"  residual correlation with nb562: {resid_corr:.3f}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy)
    np.save(
        DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof_unb.astype(np.float32)
    )
    np.save(
        DATA_PROCESSED / f"te_{TAG}_cvbag.npy", te_cv_bag.astype(np.float32)
    )

    plain = SUBMISSIONS / f"{TAG}_unblind_aug_deep.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         te_deploy,
    }).to_csv(plain, index=False)

    soft = te_deploy.copy()
    soft[unb_idx] = (
        SOFT_W * y_unb.astype(np.float32)
        + (1.0 - SOFT_W) * te_deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_unblind_aug_deep_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'te_{TAG}_cvbag.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("=== nb591 SUMMARY ===")
    print(f"  cross-fit RAE (253)         = {rae_oof:.4f}")
    print(f"  per-fold RAE                = "
          f"{[f'{r:.4f}' for r in fold_raes]}")
    print(f"  nb562 reference RAE         = {NB562_RAE:.4f}")
    print(f"  beats nb562 standalone      = {rae_oof < NB562_RAE}")
    if blend_rae is not None:
        print(f"  50/50 blend w/nb562 RAE     = {blend_rae:.4f}")
        print(f"  beats {NB562_BLEND_TARGET} blend target = "
              f"{blend_rae < NB562_BLEND_TARGET}")
        print(f"  resid corr w/nb562          = {resid_corr:.3f}")
    print(f"  te_deploy mean/std (513)    = "
          f"{te_deploy.mean():.3f} / {te_deploy.std():.3f}")
    print("=" * 78)

    return {
        "success": True,
        "rae_crossfit": rae_oof,
        "rae_per_fold": [float(r) for r in fold_raes],
        "rae_nb562_target": NB562_RAE,
        "beats_nb562": bool(rae_oof < NB562_RAE),
        "blend_rae": blend_rae,
        "resid_corr_with_nb562": resid_corr,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
