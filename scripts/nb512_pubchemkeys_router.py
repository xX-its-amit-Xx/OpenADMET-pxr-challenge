"""nb512 -- PUBCHEM-STYLE SUBSTRUCTURE KEYS RESIDUAL ROUTER.

Hypothesis: PubChem 881-bit CACTVS substructure keys encode a 5th distinct
substructure space (relative to MACCS keys nb502, AtomPair distances nb510,
TopologicalTorsion 4-atom paths nb511, and the multimodal 33-col matrix
nb492). The shallow LGBM router on top of nb464 should learn a residual
signal that is decorrelated from the rest.

FINGERPRINT USED:
  RDKit does NOT ship the official PubChem 881-bit CACTVS substructure
  key SMARTS list, and no precomputed CACTVS file is present in this repo
  (see search of data/processed/). Per the task spec, we fall back to
  Avalon fingerprints (rdkit.Avalon.pyAvalonTools.GetAvalonFP, 512 bits)
  as the orthogonal-substructure proxy. Avalon is a substructure-style
  fingerprint developed by Novartis using a curated path/feature
  enumeration that is distinct from both MACCS and PubChem CACTVS.

  --> fingerprint_used = "Avalon_512" (FALLBACK; not PubChem CACTVS).

Pipeline mirrors nb510/nb511 exactly except for the feature matrix:
  1. Anchor: nb464 (te_nb464.npy).
  2. Features: Avalon fingerprints, 512 bits.
  3. Target = truth - nb464 on 253 unblind rows.
  4. 5-fold KFold cross-fit shallow LGBM (max_depth=3, n_est=80, lr=0.05,
     num_leaves=8, min_child_samples=20, reg_lambda=1.0).
  5. Gate: alpha = sigmoid((err_hat - median(err_hat)) * 4.0).
  6. Honest unblind RAE via cross-fit resid_oof (target: < 0.5283 nb492,
     ideally near or below 0.5126 nb502).
  7. Refit on all 253 -> deploy resid_hat (513,).
  8. Save te_nb512.npy + pred_oof + resid_oof; emit plain + soft07 CSVs.
  9. Cache fingerprints at data/processed/{tr,te}_avalon512.npy for reuse.

Memory: 4139 train + 513 test * 512 bits uint8 = ~2.4 MB. Trivial.
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
from rdkit.Avalon import pyAvalonTools
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
TAG = "nb512"
ANCHOR_NAME = "nb464"
ANCHOR_FILE = "te_nb464.npy"

FP_DIM = 512  # Avalon native size
FP_NAME = "Avalon_512"  # fingerprint_used flag
FP_CACHE_KEY = "avalon512"

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


def avalon_bits(smi: str) -> np.ndarray | None:
    if not smi:
        return None
    std = standardize_smiles(smi) or smi
    m = Chem.MolFromSmiles(std)
    if m is None:
        return None
    bv = pyAvalonTools.GetAvalonFP(m, FP_DIM)
    arr = np.zeros(FP_DIM, dtype=np.uint8)
    for i in bv.GetOnBits():
        arr[i] = 1
    return arr


def build_avalon_matrix(smiles: list[str], tag: str) -> np.ndarray:
    cache = DATA_PROCESSED / f"{tag}_{FP_CACHE_KEY}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (len(smiles), FP_DIM):
            print(f"  loaded cached Avalon: {cache.name} shape={arr.shape}")
            return arr
    print(f"  computing Avalon-{FP_DIM} for {len(smiles)} {tag} molecules ...")
    out = np.zeros((len(smiles), FP_DIM), dtype=np.uint8)
    n_fail = 0
    for i, smi in enumerate(smiles):
        bits = avalon_bits(smi)
        if bits is None:
            n_fail += 1
            continue
        out[i] = bits
    print(f"    failed to parse: {n_fail}")
    np.save(cache, out)
    print(f"    cached -> {cache}")
    return out


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- PUBCHEMKEYS/AVALON RESIDUAL ROUTER (anchor={ANCHOR_NAME})")
    print(f"fingerprint_used = {FP_NAME} (fallback for PubChem CACTVS)")
    print("=" * 78)

    needed = {
        ANCHOR_FILE: DATA_PROCESSED / ANCHOR_FILE,
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "TRAIN": DATA_RAW / "pxr-challenge_TRAIN.csv",
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

    tr_df = pd.read_csv(needed["TRAIN"])
    tr_unique = tr_df.drop_duplicates(subset=["Molecule Name"]).reset_index(drop=True)
    tr_smis = tr_unique["SMILES"].tolist()
    print(f"\ntrain n={len(tr_smis)}  test n={n_te}")

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_idx)
    print(f"unblind n={n_unb}")
    print(f"err_hat: mean={err_hat.mean():.3f} median={np.median(err_hat):.3f} "
          f"std={err_hat.std():.3f}")
    print(f"anchor ({ANCHOR_NAME}) stats: mean={anchor.mean():.3f} "
          f"std={anchor.std():.3f}")

    # ---- Avalon feature matrices ----
    print(f"\nBuilding {FP_NAME} feature matrices:")
    X_tr = build_avalon_matrix(tr_smis, "tr").astype(np.float32)
    X = build_avalon_matrix(te_smis, "te").astype(np.float32)
    bit_var = X.var(axis=0)
    n_const = int((bit_var == 0).sum())
    print(f"  X_train shape: {X_tr.shape}")
    print(f"  X_test  shape: {X.shape}  constant bits: {n_const}/{FP_DIM}")

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
    print(f"\nSpearman(resid_oof, true residual)     = {rho_resid:.4f}")
    print(f"Spearman(|resid_oof|, |true residual|) = {rho_abs:.4f}")

    # ---- Refit on all 253 -> deploy ----
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X).astype(np.float32)

    # ---- Top-5 importances ----
    imp = deploy_mdl.feature_importances_
    top_idx = np.argsort(imp)[::-1][:5]
    top_imps = [(int(i), int(imp[i])) for i in top_idx]
    print(f"\nTop-5 LGBM feature importances (bit_idx, gain):")
    for bi, g in top_imps:
        print(f"  bit {bi:4d}  importance={g}")

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
    beats_nb502 = rae_routed < 0.5126
    print(f"  beats nb492 (0.5283)?            = {beats_nb492}")
    print(f"  beats nb502 (0.5126)?            = {beats_nb502}")

    # ---- Correlations against nb492/nb502/nb510/nb511 resid_oof ----
    rhos = {}
    for peer in ("nb492", "nb502", "nb510", "nb511"):
        path = DATA_PROCESSED / f"{peer}_resid_oof.npy"
        rhos[peer] = float("nan")
        if path.exists():
            peer_oof = np.load(path).astype(np.float32)
            if peer_oof.shape == resid_oof.shape:
                r, _ = spearmanr(resid_oof, peer_oof)
                if np.isfinite(r):
                    rhos[peer] = float(r)
                print(f"  Spearman(resid_oof_{TAG}, resid_oof_{peer}) = "
                      f"{rhos[peer]:.4f}")

    # ---- Save arrays + submissions ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{TAG}_resid_oof.npy", resid_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{TAG}_alpha.npy", alpha_513)

    plain = SUBMISSIONS / f"{TAG}_pubchemkeys_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_pubchemkeys_router_soft07_truth.csv"
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
    print(f"  fingerprint_used                      = {FP_NAME} (fallback)")
    print(f"  spearman(resid_oof, true resid)       = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true resid|)   = {rho_abs:.4f}")
    print(f"  spearman(resid_oof, nb492 resid_oof)  = {rhos.get('nb492', float('nan')):.4f}")
    print(f"  spearman(resid_oof, nb502 resid_oof)  = {rhos.get('nb502', float('nan')):.4f}")
    print(f"  spearman(resid_oof, nb510 resid_oof)  = {rhos.get('nb510', float('nan')):.4f}")
    print(f"  spearman(resid_oof, nb511 resid_oof)  = {rhos.get('nb511', float('nan')):.4f}")
    print(f"  unblind RAE {ANCHOR_NAME} anchor     = {rae_anchor:.4f}")
    print(f"  unblind RAE {TAG} cross-fit          = {rae_routed:.4f}")
    print(f"  unblind RAE {TAG} in-sample          = {rae_insample:.4f}")
    print(f"  beats nb492 (< 0.5283)                = {beats_nb492}")
    print(f"  beats nb502 (< 0.5126)                = {beats_nb502}")
    print(f"  top-5 importances (bit, gain)         = {top_imps}")
    print("=" * 78)

    return {
        "success": True,
        "anchor": ANCHOR_NAME,
        "fingerprint_used": FP_NAME,
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "resid_corr_with_nb492": rhos.get("nb492", float("nan")),
        "resid_corr_with_nb502": rhos.get("nb502", float("nan")),
        "resid_corr_with_nb510": rhos.get("nb510", float("nan")),
        "resid_corr_with_nb511": rhos.get("nb511", float("nan")),
        "rae_anchor": rae_anchor,
        "crossfit_rae": rae_routed,
        "insample_rae": rae_insample,
        "beats_nb492": bool(beats_nb492),
        "beats_nb502": bool(beats_nb502),
        "top5_importances": top_imps,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
