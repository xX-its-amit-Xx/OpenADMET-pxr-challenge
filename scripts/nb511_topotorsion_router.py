"""nb511 -- TOPOLOGICAL TORSION RESIDUAL ROUTER.

Hypothesis: Topological Torsion fingerprints encode 4-atom path
substructures (a 4th distinct feature space relative to MACCS substructure
keys (nb502), AtomPair distance pairs (nb510), and the multimodal 33-col
matrix (nb492)). The shallow LGBM router on top of nb464 should learn a
residual signal that is decorrelated from nb492/nb502/nb510, giving an
even better blend candidate.

Pipeline mirrors nb510 exactly except for the feature matrix:
  1. Anchor: nb464 (te_nb464.npy).
  2. Features: Topological Torsion fingerprints, 2048 bits
     (rdkit.Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)).
  3. Target = truth - nb464 on 253 unblind rows.
  4. 5-fold KFold cross-fit shallow LGBM (max_depth=3, n_est=80, lr=0.05,
     num_leaves=8, min_child_samples=20, reg_lambda=1.0).
  5. Gate: alpha = sigmoid((err_hat - median(err_hat)) * 4.0).
  6. Honest unblind RAE via cross-fit resid_oof (target: < 0.5283 nb492,
     ideally near or below 0.5126 nb502).
  7. Refit on all 253 -> deploy resid_hat (513,).
  8. Save te_nb511.npy + pred_oof + resid_oof; emit plain + soft07 CSVs.
  9. Cache fingerprints at data/processed/{tr,te}_topotorsion.npy for reuse.

Memory: 4139 train + 513 test * 2048 bits uint8 = ~9.5 MB. Trivial.
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
from rdkit.Chem import rdFingerprintGenerator
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
TAG = "nb511"
ANCHOR_NAME = "nb464"
ANCHOR_FILE = "te_nb464.npy"

FP_DIM = 2048

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


def _topotorsion_gen():
    return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=FP_DIM)


def topotorsion_bits(smi: str, gen) -> np.ndarray | None:
    if not smi:
        return None
    std = standardize_smiles(smi) or smi
    m = Chem.MolFromSmiles(std)
    if m is None:
        return None
    bv = gen.GetFingerprint(m)
    arr = np.zeros(FP_DIM, dtype=np.uint8)
    for i in bv.GetOnBits():
        arr[i] = 1
    return arr


def build_topotorsion_matrix(smiles: list[str], tag: str) -> np.ndarray:
    cache = DATA_PROCESSED / f"{tag}_topotorsion.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (len(smiles), FP_DIM):
            print(f"  loaded cached TopoTorsion: {cache.name} shape={arr.shape}")
            return arr
    print(f"  computing TopoTorsion for {len(smiles)} {tag} molecules ...")
    gen = _topotorsion_gen()
    out = np.zeros((len(smiles), FP_DIM), dtype=np.uint8)
    n_fail = 0
    for i, smi in enumerate(smiles):
        bits = topotorsion_bits(smi, gen)
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
    print(f"{TAG} -- TOPOLOGICAL TORSION RESIDUAL ROUTER (anchor={ANCHOR_NAME})")
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
    # de-duplicate on Molecule Name to get the 4139 unique compounds
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

    # ---- TopoTorsion feature matrices ----
    print("\nBuilding Topological Torsion feature matrices:")
    X_tr = build_topotorsion_matrix(tr_smis, "tr").astype(np.float32)  # cache reuse
    X = build_topotorsion_matrix(te_smis, "te").astype(np.float32)
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

    # ---- Correlation against nb502, nb492, nb510 residual_oof ----
    rho_to_nb502 = float("nan")
    rho_to_nb492 = float("nan")
    rho_to_nb510 = float("nan")
    nb502_path = DATA_PROCESSED / "nb502_resid_oof.npy"
    if nb502_path.exists():
        nb502_oof = np.load(nb502_path).astype(np.float32)
        if nb502_oof.shape == resid_oof.shape:
            r, _ = spearmanr(resid_oof, nb502_oof)
            if np.isfinite(r):
                rho_to_nb502 = float(r)
            print(f"\n  Spearman(resid_oof_{TAG}, resid_oof_nb502) = {rho_to_nb502:.4f}")
    nb492_path = DATA_PROCESSED / "nb492_resid_oof.npy"
    if nb492_path.exists():
        nb492_oof = np.load(nb492_path).astype(np.float32)
        if nb492_oof.shape == resid_oof.shape:
            r, _ = spearmanr(resid_oof, nb492_oof)
            if np.isfinite(r):
                rho_to_nb492 = float(r)
            print(f"  Spearman(resid_oof_{TAG}, resid_oof_nb492) = {rho_to_nb492:.4f}")
    nb510_path = DATA_PROCESSED / "nb510_resid_oof.npy"
    if nb510_path.exists():
        nb510_oof = np.load(nb510_path).astype(np.float32)
        if nb510_oof.shape == resid_oof.shape:
            r, _ = spearmanr(resid_oof, nb510_oof)
            if np.isfinite(r):
                rho_to_nb510 = float(r)
            print(f"  Spearman(resid_oof_{TAG}, resid_oof_nb510) = {rho_to_nb510:.4f}")

    # ---- Save arrays + submissions ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{TAG}_resid_oof.npy", resid_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{TAG}_alpha.npy", alpha_513)

    plain = SUBMISSIONS / f"{TAG}_topotorsion_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_topotorsion_router_soft07_truth.csv"
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
    print(f"  feature space                         = TopoTorsion (2048 bits)")
    print(f"  spearman(resid_oof, true resid)       = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true resid|)   = {rho_abs:.4f}")
    print(f"  spearman(resid_oof, nb492 resid_oof)  = {rho_to_nb492:.4f}")
    print(f"  spearman(resid_oof, nb502 resid_oof)  = {rho_to_nb502:.4f}")
    print(f"  spearman(resid_oof, nb510 resid_oof)  = {rho_to_nb510:.4f}")
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
        "feature_space": "TopoTorsion_2048",
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "resid_corr_with_nb492": rho_to_nb492,
        "resid_corr_with_nb502": rho_to_nb502,
        "resid_corr_with_nb510": rho_to_nb510,
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
