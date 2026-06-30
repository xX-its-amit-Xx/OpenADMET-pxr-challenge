"""nb2540 -- LGBM with monotone constraints on biological-prior features (X_117 + bio).

NEW PARADIGM:
    PXR is a hydrophobic ligand-binding domain (~1300 A^3 with broad
    hydrophobic surface and few polar contacts).  Biological prior is that
    lipophilicity (LogP) and molecular size (MW) are monotonically
    positively related to PXR activation potency over the chemical space we
    score.  We enforce this prior structurally on LightGBM via
    `monotone_constraints` rather than letting purely statistical splits
    fight it on n=253.

PROTOCOL:
    1. Load X_117 substrate (test 513 -> slice to unb 253) from
       data/processed/pyramid/X_117_unb.npy + X_117_te.npy.
    2. Append a small "bio block" of 4 RDKit-computed physchem features at
       known column positions appended to X_117:
            117: MolLogP        (Crippen logP)       -- monotone +1
            118: MolWt          (molecular weight)   -- monotone +1
            119: NumAtomFsp3    (fsp3 count)         -- monotone  0
                 (fsp3 is a 3D-shape proxy; PXR pocket is broad/flat so
                  fsp3 should not be forced monotone -- treat as
                  unconstrained "carrier" feature so the model can learn
                  the small fsp3 effect freely while LogP+MW carry the
                  hydrophobic prior.)
            120: TPSA           (topological polar surface area) monotone -1
                 (PXR's hydrophobic LBD is anti-correlated with PSA; we
                  enforce monotone-decreasing prior to align with biology.)
       All 117 substrate features (AtomPair, MACCS, Mordred top-K,
       ChempropEmbed top-K, Avalon top-K + ChEMBL kNN + mean_sim) get 0
       (unconstrained).  Final feature dim = 121.
    3. Build residual = y_unb - chemprop_aux[unb_idx] (only verified-clean
       PRE-unblind anchor; FROZEN per Kaggle chemprop dead-end memo).
    4. LightGBM(MSE, max_depth=4, num_leaves=15, n_est=300, lr=0.03,
       min_child_samples=5, reg_lambda=2.0, monotone_constraints=[...],
       monotone_constraints_method='basic').
       NOTE: task said "max_depth=4, n_est=300, lr=0.03" -- num_leaves=15
       inherited from nb2013 family (matches max_depth=4 cap, 2^4-1=15).
    5. 5-fold scaffold CV on 253 unblind, 5 kf_seeds {1001..1005}.
       (scaffold_kfold_indices from pxr.eval -- random KFold is forbidden
       by the cv-protocol-audit memo.)
    6. Aggregate per-seed corrected RAE; report mean-bag RAE.
    7. Deploy: refit on full 253 per seed, predict 513 residual, mean-bag.

GATE (on mean-bag RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4601 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2540_monotone_constraint_lgbm.py
    data/processed/nb2540_summary.json
    data/processed/nb2540_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2540.npy         (513,) float32 deploy refit
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2540"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# LGBM hyperparameters (per task spec; num_leaves bound by 2^max_depth-1)
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_EST = 300
LGBM_LR = 0.03
LGBM_MIN_CHILD = 5
LGBM_REG_LAMBDA = 2.0

# Bio block layout (appended to X_117 at fixed offsets)
BIO_FEAT_NAMES = ["MolLogP", "MolWt", "NumAtomFsp3", "TPSA"]
BIO_FEAT_CONSTRAINTS = [+1, +1, 0, -1]  # see header rationale

CHEMPROP_AUX_REF = 0.6216


# --------------------------- Bio feature computation -------------------------
def _compute_bio_features(smiles_list: list[str]) -> np.ndarray:
    """Compute RDKit MolLogP, MolWt, NumAtomFsp3, TPSA per SMILES."""
    n = len(smiles_list)
    out = np.zeros((n, len(BIO_FEAT_NAMES)), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = standardize(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            out[i, :] = np.nan
            continue
        try:
            logp = float(Crippen.MolLogP(mol))
        except Exception:
            logp = np.nan
        try:
            mw = float(Descriptors.MolWt(mol))
        except Exception:
            mw = np.nan
        try:
            # NumAtomFsp3 = fraction sp3 * heavy_atoms; we use the count
            # variant Lipinski.NumAtomStereoCenters... no, use fraction *
            # heavy atom count which matches "NumAtomFsp3" intent.
            fsp3_frac = float(Lipinski.FractionCSP3(mol))
            n_heavy = mol.GetNumHeavyAtoms()
            num_sp3 = fsp3_frac * n_heavy
        except Exception:
            num_sp3 = np.nan
        try:
            tpsa = float(Descriptors.TPSA(mol))
        except Exception:
            tpsa = np.nan
        out[i, 0] = logp
        out[i, 1] = mw
        out[i, 2] = num_sp3
        out[i, 3] = tpsa
    # Median impute any NaNs column-wise
    for c in range(out.shape[1]):
        col = out[:, c]
        bad = ~np.isfinite(col)
        if bad.any():
            med = float(np.nanmedian(col)) if (~bad).any() else 0.0
            col[bad] = med
            out[:, c] = col
    return out


def _lgbm_params(seed: int, monotone_constraints: list[int]) -> dict:
    return dict(
        objective="regression",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_EST,
        learning_rate=LGBM_LR,
        min_child_samples=LGBM_MIN_CHILD,
        reg_lambda=LGBM_REG_LAMBDA,
        monotone_constraints=monotone_constraints,
        monotone_constraints_method="basic",
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
    mono: list[int],
) -> np.ndarray:
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed, mono))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    mono: list[int],
) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed, mono))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


# ============================================================================
# MAIN
# ============================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM(MSE) + monotone constraints on biological-prior block")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold")
    print(f"        kf_seeds = {KF_SEEDS}")
    print(f"        LGBM(depth={LGBM_MAX_DEPTH}, leaves={LGBM_NUM_LEAVES}, "
          f"n_est={LGBM_N_EST}, lr={LGBM_LR})")
    print(f"        bio block: {list(zip(BIO_FEAT_NAMES, BIO_FEAT_CONSTRAINTS))}")
    print(f"        GATE: mean_rae < {GATE_PROMOTE} PROMOTE; "
          f"< {GATE_MARGINAL} MARGINAL_BEAT; else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist()
    test_names = te["name"].astype(str).tolist()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"anchor te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load X_117 substrate ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape}")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape}")
    # Sanitize NaN/Inf carried in cache
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)
    print(f"[feat] X117_unb = {X117_unb.shape}  X117_te = {X117_te.shape}")

    # ---- Compute bio block (4 cols) for 513, slice unb ----
    print("\n[bio] computing MolLogP, MolWt, NumAtomFsp3, TPSA for 513...")
    X_bio_te = _compute_bio_features(test_smiles)
    X_bio_unb = X_bio_te[unb_idx]
    print(f"[bio] X_bio_te = {X_bio_te.shape}")
    for j, name in enumerate(BIO_FEAT_NAMES):
        col = X_bio_te[:, j]
        print(f"        {name:13s} mean={col.mean():7.2f}  std={col.std():6.2f}  "
              f"min={col.min():6.2f}  max={col.max():6.2f}")

    # ---- Concatenate substrate + bio block ----
    X_unb = np.concatenate([X117_unb, X_bio_unb], axis=1).astype(np.float32)
    X_te = np.concatenate([X117_te, X_bio_te], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    assert feat_dim == 121, f"feat_dim {feat_dim} != 121"
    print(f"[feat] CONCAT X_unb = {X_unb.shape}  X_te = {X_te.shape}")

    # ---- Build monotone constraints vector ----
    # X_117 substrate is column 0..116 (all 0); bio block 117..120
    monotone_constraints = [0] * 117 + list(BIO_FEAT_CONSTRAINTS)
    assert len(monotone_constraints) == feat_dim
    n_pos = sum(1 for c in monotone_constraints if c > 0)
    n_neg = sum(1 for c in monotone_constraints if c < 0)
    n_zero = sum(1 for c in monotone_constraints if c == 0)
    print(f"[mono] vec length={len(monotone_constraints)}  "
          f"+1={n_pos}  -1={n_neg}  0={n_zero}")
    print(f"[mono] bio positions:")
    for j, (name, c) in enumerate(zip(BIO_FEAT_NAMES, BIO_FEAT_CONSTRAINTS)):
        sign = "+1" if c > 0 else ("-1" if c < 0 else " 0")
        print(f"        col {117 + j:3d}  {name:13s}  monotone={sign}")

    # ---- Scaffolds ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  dim={feat_dim}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof = _scaffold_cv_one_seed(
            X_unb, residual, unb_scaffolds, seed, monotone_constraints,
        )
        per_seed_oof_resid[i] = resid_oof
        te_resid = _deploy_te_one_seed(
            X_unb, residual, X_te, seed, monotone_constraints,
        )
        per_seed_te_resid[i] = te_resid
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"wall={time.time() - ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_resid = per_seed_oof_resid.mean(axis=0)
    median_bag_resid = np.median(per_seed_oof_resid, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag   RAE = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE = {rae_median_bag:.4f}")
    print(f"[cv] anchor     RAE = {rae_anchor:.4f}  "
          f"(d_mean_bag = {rae_mean_bag - rae_anchor:+.4f})")

    # ---- Deploy te (mean-bag corrected) ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_resid).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_bag_rae        = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": ("lgbm_mse_monotone_bio_prior_block_on_X117_residual_"
                   "on_chemprop_aux"),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "x117_unb_path": str(X117_UNB_PATH),
        "x117_te_path": str(X117_TE_PATH),
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "lgbm_params": {
            "objective": "regression",
            "max_depth": LGBM_MAX_DEPTH,
            "num_leaves": LGBM_NUM_LEAVES,
            "n_estimators": LGBM_N_EST,
            "learning_rate": LGBM_LR,
            "min_child_samples": LGBM_MIN_CHILD,
            "reg_lambda": LGBM_REG_LAMBDA,
            "monotone_constraints_method": "basic",
        },
        "bio_block": [
            {"name": n, "col": 117 + j, "monotone": c}
            for j, (n, c) in enumerate(zip(BIO_FEAT_NAMES, BIO_FEAT_CONSTRAINTS))
        ],
        "feat_dim": int(feat_dim),
        "n_monotone_pos": n_pos,
        "n_monotone_neg": n_neg,
        "n_monotone_zero": n_zero,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "delta_vs_anchor": rae_mean_bag - rae_anchor,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "pre_unblind_clean_anchor": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
