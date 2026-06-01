"""nb502 -- ALT-FEATURE RESIDUAL ROUTER (MACCS keys).

Hypothesis: nb492's residual router used the engineered 33-col multimodal
feature matrix. A router built on a COMPLETELY DIFFERENT feature space --
substructure keys (MACCS, 167 bits) -- should learn a decorrelated residual
direction. If resid_oof correlation to nb492 < 0.5, blending nb502 + nb492
will give the orthogonality the parent ensemble craves.

Pipeline mirrors nb492 exactly except for the feature matrix:
  1. Anchor: nb464 (te_nb464.npy).
  2. Features: MACCS keys, 167 bits (rdkit.Chem.MACCSkeys).
  3. Target = truth - nb464 on 253 unblind rows.
  4. 5-fold KFold cross-fit shallow LGBM (same hyperparams as nb472/nb481/nb492):
       max_depth=3, n_est=80, lr=0.05, num_leaves=8, min_child_samples=20,
       reg_lambda=1.0.
  5. Gate: alpha = sigmoid((err_hat - median(err_hat)) * 4.0).
  6. Honest unblind RAE via cross-fit residual_oof (target: < 0.5283).
  7. Refit on all 253 -> deploy resid_hat (513,).
  8. Save te_nb502.npy + pred_oof + resid_oof; emit plain + soft07_truth CSVs.

Memory: MACCS = 167 bits/molecule. 4139 train + 513 test ~= 0.8 MB total.
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
from rdkit.Chem import MACCSkeys
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.chem import standardize_smiles
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7
TAG = "nb502"
ANCHOR_NAME = "nb464"
ANCHOR_FILE = "te_nb464.npy"

# MACCSkeys.GenMACCSKeys returns 167 bits indexed 0..166 (bit 0 unused but kept).
MACCS_DIM = 167

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


def maccs_bits(smi: str) -> np.ndarray | None:
    if not smi:
        return None
    std = standardize_smiles(smi) or smi
    m = Chem.MolFromSmiles(std)
    if m is None:
        return None
    bv = MACCSkeys.GenMACCSKeys(m)
    arr = np.zeros(MACCS_DIM, dtype=np.uint8)
    for i in range(MACCS_DIM):
        if bv.GetBit(i):
            arr[i] = 1
    return arr


def build_maccs_matrix(smiles: list[str], tag: str) -> np.ndarray:
    cache = DATA_PROCESSED / f"{tag}_maccs.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (len(smiles), MACCS_DIM):
            print(f"  loaded cached MACCS: {cache.name} shape={arr.shape}")
            return arr
    print(f"  computing MACCS for {len(smiles)} {tag} molecules ...")
    out = np.zeros((len(smiles), MACCS_DIM), dtype=np.uint8)
    n_fail = 0
    for i, smi in enumerate(smiles):
        bits = maccs_bits(smi)
        if bits is None:
            n_fail += 1
            continue
        out[i] = bits
    print(f"    failed to parse: {n_fail}")
    np.save(cache, out)
    return out


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- ALT-FEATURE RESIDUAL ROUTER (MACCS keys, anchor={ANCHOR_NAME})")
    print("=" * 78)

    needed = {
        ANCHOR_FILE: DATA_PROCESSED / ANCHOR_FILE,
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    anchor = np.load(needed[ANCHOR_FILE]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    n_te = anchor.shape[0]
    assert err_hat.shape == (n_te,)

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    te_smis = te_df["SMILES"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")
    print(f"err_hat: mean={err_hat.mean():.3f} median={np.median(err_hat):.3f} "
          f"std={err_hat.std():.3f}")
    print(f"anchor ({ANCHOR_NAME}) stats: mean={anchor.mean():.3f} "
          f"std={anchor.std():.3f}")

    # ---- MACCS feature matrix ----
    print("\nBuilding MACCS feature matrix (TEST only -- router is trained on")
    print("the 253 unblind subset, deployed to all 513):")
    X = build_maccs_matrix(te_smis, "te").astype(np.float32)
    bit_var = X.var(axis=0)
    n_const = int((bit_var == 0).sum())
    print(f"  X shape: {X.shape}  constant bits: {n_const}/{MACCS_DIM}")

    X_unb = X[unb_idx]
    y_resid = (unb_y - anchor[unb_idx]).astype(np.float32)
    print(f"\nResidual target: mean={y_resid.mean():+.3f} "
          f"std={y_resid.std():.3f} |mean|={np.abs(y_resid).mean():.3f}")

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
    resid_hat_513 = deploy_mdl.predict(X).astype(np.float32)
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
    print(f"  beats nb492 (0.5283)?            = {beats_nb492}")

    # ---- Correlation against nb492 residual_oof (the orthogonality test) ----
    nb492_oof_path = DATA_PROCESSED / "nb492_resid_oof.npy"
    rho_to_nb492 = float("nan")
    if nb492_oof_path.exists():
        nb492_oof = np.load(nb492_oof_path).astype(np.float32)
        if nb492_oof.shape == resid_oof.shape:
            r, _ = spearmanr(resid_oof, nb492_oof)
            if np.isfinite(r):
                rho_to_nb492 = float(r)
            print(f"  Spearman(resid_oof_{TAG}, resid_oof_nb492) = {rho_to_nb492:.4f}")
            if rho_to_nb492 < 0.5:
                print(f"  -> DECORRELATED (< 0.5) : good blend candidate")
            else:
                print(f"  -> correlated (>= 0.5)  : may not blend well")
    else:
        print(f"  (nb492_resid_oof.npy not found; skipping decorrelation check)")

    # ---- Save arrays + submissions ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{TAG}_resid_oof.npy", resid_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{TAG}_alpha.npy", alpha_513)

    plain = SUBMISSIONS / f"{TAG}_altfeat_router_maccs.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_altfeat_router_maccs_soft07_truth.csv"
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
    print(f"  anchor                                = {ANCHOR_NAME}")
    print(f"  feature space                         = MACCS (167 bits)")
    print(f"  spearman(resid_oof, true resid)       = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true resid|)   = {rho_abs:.4f}")
    print(f"  spearman(resid_oof, nb492 resid_oof)  = {rho_to_nb492:.4f}")
    print(f"  unblind RAE {ANCHOR_NAME} anchor     = {rae_anchor:.4f}")
    print(f"  unblind RAE {TAG} cross-fit          = {rae_routed:.4f}")
    print(f"  unblind RAE {TAG} in-sample          = {rae_insample:.4f}")
    print(f"  beats nb492 (< 0.5283)                = {beats_nb492}")
    print("=" * 78)

    return {
        "success": True,
        "anchor": ANCHOR_NAME,
        "feature_space": "MACCS_167",
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "resid_corr_with_nb492": rho_to_nb492,
        "rae_anchor": rae_anchor,
        "crossfit_rae": rae_routed,
        "insample_rae": rae_insample,
        "beats_nb492": bool(beats_nb492),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
