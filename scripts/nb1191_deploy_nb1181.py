"""nb1191 -- DEPLOY triple naive 1/3 mean blend (from nb1181) on 513 rows.

Blend recipe (nb1181 verdict; honest cross-fit RAE 0.5566 on 253 unblind):
    te_nb1191 = (te_nb1140 + te_nb1162 + te_nb1172) / 3

Components (513-row deploy artifacts, each = nb1070 anchor + 5-seed mean-bag
residual on a distinct feature family):
  te_nb1140   nb1130 mean-bag deploy   Morgan(2048) + RDKit-desc(217)
  te_nb1162   nb1153 mean-bag deploy   Mordred(1533)
  te_nb1172   nb1172 mean-bag deploy   AtomPair(2048)

If te_nb1172.npy is missing we BUILD it inline here (5 shallow LGBM Huber
seeds 0/1/7/42/137, fit on ALL 253 unblind rows with target
y_unb - nb1070_pred_oof, predict on 513 cached AtomPair test features,
mean-bag the 5 residuals, add to te_nb1070).

Caveat (per feedback_lb_two_regime_calibration): the three deploys are
POST-unblind refits, so in_RAE on te_nb1191[unb_idx] is in-sample and
optimistic. The LB-faithful anchor is the nb1181 honest cross-fit RAE
(0.5566 for the naive 1/3 mean variant).

Outputs:
  submissions/nb1191_deploy_nb1181.csv   SMILES, Molecule Name, pEC50  (513)
  data/processed/te_nb1191.npy           float32  (513,)
  data/processed/te_nb1172.npy           float32  (513,)  -- built if missing
  data/processed/nb1191_summary.json
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
import pandas as pd

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1191"
ANCHOR = "nb1070"
RESID_SEEDS = [0, 1, 7, 42, 137]
NB1181_CROSSFIT_LB_ANCHOR = 0.5566

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"  # (513, 2048) uint8

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1172 cross-fit bag."""
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _build_te_atompair_513(n_test_expected: int, te_smiles: list[str]) -> np.ndarray:
    """Build the 513x2048 hashed AtomPair fingerprint matrix from raw SMILES.

    Used only as a fallback if te_atompair.npy is missing. Median-imputes
    failed parses (returns zeros for None mols since FP is binary).
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    X = np.zeros((n_test_expected, 2048), dtype=np.uint8)
    fails = 0
    for i, smi in enumerate(te_smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fails += 1
                continue
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=2048
            )
            arr = np.zeros((2048,), dtype=np.uint8)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            X[i] = arr
        except Exception:
            fails += 1
            continue
    print(f"[build_atompair] n_test={n_test_expected}  failed_parses={fails}")
    return X


def _load_or_build_atompair_te(n_test_expected: int,
                               te_smiles: list[str]) -> np.ndarray:
    if ATOMPAIR_TE_PATH.exists():
        X_te = np.load(ATOMPAIR_TE_PATH)
        if X_te.shape[0] != n_test_expected or X_te.shape[1] != 2048:
            raise ValueError(
                f"AtomPair test cache shape mismatch: {X_te.shape} "
                f"vs (n_test={n_test_expected}, 2048)"
            )
        print(f"[feat] loaded cached AtomPair {X_te.shape} from {ATOMPAIR_TE_PATH}")
        return X_te.astype(np.float32)
    print(f"[feat] {ATOMPAIR_TE_PATH} missing -- building from raw SMILES")
    X_te = _build_te_atompair_513(n_test_expected, te_smiles)
    np.save(ATOMPAIR_TE_PATH, X_te)
    print(f"[save] {ATOMPAIR_TE_PATH}  shape={X_te.shape}")
    return X_te.astype(np.float32)


def _build_te_nb1172(te_nb1070: np.ndarray,
                     nb1070_oof: np.ndarray,
                     y_unb: np.ndarray,
                     unb_idx: np.ndarray,
                     te_smiles: list[str]) -> tuple[np.ndarray, dict]:
    """Build te_nb1172.npy in-process if missing.

    Mirrors nb1140/nb1162 deploy convention but with AtomPair-2048 features.
    """
    from lightgbm import LGBMRegressor

    n_test = len(te_smiles)
    n_unb = len(y_unb)
    print("-" * 78)
    print("BUILDING te_nb1172.npy (deploy companion of nb1172, AtomPair-2048)")
    print("-" * 78)

    X_test = _load_or_build_atompair_te(n_test, te_smiles)
    X_unb = X_test[unb_idx]
    print(f"[feat] X_test shape = {X_test.shape}  X_unb shape = {X_unb.shape}")

    residual_target = y_unb - nb1070_oof
    print(f"[resid] mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}  "
          f"min={residual_target.min():+.4f}  "
          f"max={residual_target.max():+.4f}")

    per_seed_residual_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_records = []
    rae_anchor_te_in = float(rae(y_unb, te_nb1070[unb_idx]))
    for i, s in enumerate(RESID_SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb, residual_target)
        residual_in_s = mdl.predict(X_unb)
        residual_pred_513_s = mdl.predict(X_test).astype(np.float64)
        per_seed_residual_513[i] = residual_pred_513_s
        in_pred_corr_253_s = nb1070_oof + residual_in_s
        in_rae_corr_oof_s = float(rae(y_unb, in_pred_corr_253_s))
        te_seed_s = te_nb1070 + residual_pred_513_s
        in_rae_te_s = float(rae(y_unb, te_seed_s[unb_idx]))
        per_seed_records.append({
            "seed": int(s),
            "residual_pred_513_mean": float(residual_pred_513_s.mean()),
            "residual_pred_513_std": float(residual_pred_513_s.std()),
            "in_rae_253_corr_on_oof_anchor": in_rae_corr_oof_s,
            "in_rae_253_te_anchor": in_rae_te_s,
        })
        print(f"   seed {s:3d}:  resid_513 mean={residual_pred_513_s.mean():+.4f} "
              f" std={residual_pred_513_s.std():.4f}  "
              f"in_RAE_corr(OOF)={in_rae_corr_oof_s:.4f}  "
              f"in_RAE(te)={in_rae_te_s:.4f}")

    mean_bagged_residual_513 = per_seed_residual_513.mean(axis=0)
    te_nb1172 = te_nb1070 + mean_bagged_residual_513
    in_rae_253 = float(rae(y_unb, te_nb1172[unb_idx]))
    print(f"[bag] mean_bagged_residual_513 mean={mean_bagged_residual_513.mean():+.4f}  "
          f"std={mean_bagged_residual_513.std():.4f}")
    print(f"[deploy] te_nb1172 shape={te_nb1172.shape}  "
          f"mean={te_nb1172.mean():.3f}  std={te_nb1172.std():.3f}  "
          f"min={te_nb1172.min():.3f}  max={te_nb1172.max():.3f}")
    print(f"[diag] in_RAE(te_nb1172[unb_idx]) = {in_rae_253:.4f}  "
          f"in_RAE(te_{ANCHOR}[unb_idx]) = {rae_anchor_te_in:.4f}")

    out_path = DATA_PROCESSED / "te_nb1172.npy"
    np.save(out_path, te_nb1172.astype(np.float32))
    print(f"[save] {out_path}")

    build_info = {
        "built_inline": True,
        "n_seeds": len(RESID_SEEDS),
        "seeds": RESID_SEEDS,
        "feature_dim": int(X_unb.shape[1]),
        "feature_source": "atompair_cached_2048",
        "residual_target_mean": float(residual_target.mean()),
        "residual_target_std": float(residual_target.std()),
        "mean_bagged_residual_513_mean": float(mean_bagged_residual_513.mean()),
        "mean_bagged_residual_513_std": float(mean_bagged_residual_513.std()),
        "te_nb1172_mean": float(te_nb1172.mean()),
        "te_nb1172_std": float(te_nb1172.std()),
        "in_rae_te_nb1172_unb": in_rae_253,
        "in_rae_anchor_te_unb": rae_anchor_te_in,
        "per_seed_records": per_seed_records,
    }
    return te_nb1172.astype(np.float64), build_info


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY triple naive 1/3 mean blend (nb1181 verdict)")
    print(f"          components: te_nb1140 + te_nb1162 + te_nb1172")
    print(f"          anchor LB target = nb1181 honest cross-fit "
          f"{NB1181_CROSSFIT_LB_ANCHOR:.4f}")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth, nb1070 anchor ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)

    te_nb1070 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    nb1070_oof = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    print(f"[load] te_{ANCHOR}.npy shape={te_nb1070.shape}  "
          f"{ANCHOR}_pred_oof shape={nb1070_oof.shape}")

    assert te_nb1070.shape[0] == n_test, (
        f"te_{ANCHOR} shape {te_nb1070.shape} mismatch n_test={n_test}"
    )
    assert nb1070_oof.shape[0] == n_unb, (
        f"{ANCHOR}_pred_oof shape {nb1070_oof.shape} mismatch n_unb={n_unb}"
    )

    # ---- Load te_nb1140 ----
    te_nb1140_path = DATA_PROCESSED / "te_nb1140.npy"
    if not te_nb1140_path.exists():
        raise FileNotFoundError(
            f"{te_nb1140_path} missing -- run scripts/nb1140_deploy_nb1130.py first."
        )
    te_nb1140 = np.load(te_nb1140_path).astype(np.float64)
    print(f"[load] te_nb1140.npy shape={te_nb1140.shape}  "
          f"mean={te_nb1140.mean():.3f}  std={te_nb1140.std():.3f}")
    assert te_nb1140.shape[0] == n_test

    # ---- Load te_nb1162 ----
    te_nb1162_path = DATA_PROCESSED / "te_nb1162.npy"
    if not te_nb1162_path.exists():
        raise FileNotFoundError(
            f"{te_nb1162_path} missing -- run scripts/nb1162_deploy_nb1153.py first."
        )
    te_nb1162 = np.load(te_nb1162_path).astype(np.float64)
    print(f"[load] te_nb1162.npy shape={te_nb1162.shape}  "
          f"mean={te_nb1162.mean():.3f}  std={te_nb1162.std():.3f}")
    assert te_nb1162.shape[0] == n_test

    # ---- Load or build te_nb1172 ----
    te_nb1172_path = DATA_PROCESSED / "te_nb1172.npy"
    if te_nb1172_path.exists():
        te_nb1172 = np.load(te_nb1172_path).astype(np.float64)
        print(f"[load] te_nb1172.npy shape={te_nb1172.shape}  "
              f"mean={te_nb1172.mean():.3f}  std={te_nb1172.std():.3f}")
        build_info = {"built_inline": False, "loaded_from": str(te_nb1172_path)}
    else:
        print(f"[load] te_nb1172.npy MISSING -- building inline ...")
        te_nb1172, build_info = _build_te_nb1172(
            te_nb1070, nb1070_oof, y_unb, unb_idx, te_smiles.tolist()
        )
    assert te_nb1172.shape[0] == n_test, (
        f"te_nb1172 shape {te_nb1172.shape} mismatch n_test={n_test}"
    )

    # ---- Per-component in-sample diagnostics ----
    print("\n" + "-" * 78)
    print("PER-COMPONENT IN-SAMPLE DIAGNOSTIC (te_X[unb_idx] -- optimistic)")
    print("-" * 78)
    in_rae_nb1140 = float(rae(y_unb, te_nb1140[unb_idx]))
    in_rae_nb1162 = float(rae(y_unb, te_nb1162[unb_idx]))
    in_rae_nb1172 = float(rae(y_unb, te_nb1172[unb_idx]))
    in_rae_anchor = float(rae(y_unb, te_nb1070[unb_idx]))
    rae_anchor_oof = float(rae(y_unb, nb1070_oof))
    print(f"   te_{ANCHOR}[unb_idx]  in_RAE = {in_rae_anchor:.4f}  "
          f"(anchor OOF pooled RAE = {rae_anchor_oof:.4f})")
    print(f"   te_nb1140[unb_idx]    in_RAE = {in_rae_nb1140:.4f}  "
          f"(nb1130 Morgan+RDKit  residual bag)")
    print(f"   te_nb1162[unb_idx]    in_RAE = {in_rae_nb1162:.4f}  "
          f"(nb1153 Mordred       residual bag)")
    print(f"   te_nb1172[unb_idx]    in_RAE = {in_rae_nb1172:.4f}  "
          f"(nb1172 AtomPair      residual bag)")

    # ---- Triple naive 1/3 mean blend ----
    te_nb1191 = (te_nb1140 + te_nb1162 + te_nb1172) / 3.0

    # ---- Validate row count and finiteness ----
    assert te_nb1191.shape == (n_test,), (
        f"te_nb1191 shape {te_nb1191.shape} != ({n_test},)"
    )
    n_nan = int(np.isnan(te_nb1191).sum())
    n_inf = int((~np.isfinite(te_nb1191) & ~np.isnan(te_nb1191)).sum())
    assert n_nan == 0, f"te_nb1191 has {n_nan} NaN values"
    assert n_inf == 0, f"te_nb1191 has {n_inf} non-finite values"
    print(f"\n[validate] row count = {len(te_nb1191)} (expected 513)  "
          f"NaN = {n_nan}  non-finite = {n_inf}")

    # ---- 513 distribution summary ----
    te_mean = float(te_nb1191.mean())
    te_std = float(te_nb1191.std())
    te_min = float(te_nb1191.min())
    te_max = float(te_nb1191.max())
    print(f"[deploy] te_nb1191  mean={te_mean:.4f}  std={te_std:.4f}  "
          f"min={te_min:.4f}  max={te_max:.4f}")

    # ---- In-sample RAE on 253 unblind subset ----
    in_rae_253 = float(rae(y_unb, te_nb1191[unb_idx]))
    delta_vs_anchor_te = in_rae_253 - in_rae_anchor
    print("\n" + "-" * 78)
    print("IN-SAMPLE DIAGNOSTIC (te_nb1191[unb_idx] -- optimistic)")
    print("-" * 78)
    print(f"   in_RAE(te_nb1191[unb_idx]) = {in_rae_253:.4f}")
    print(f"   in_RAE(te_{ANCHOR}[unb_idx]) = {in_rae_anchor:.4f}")
    print(f"   delta vs anchor-in-sample   = {delta_vs_anchor_te:+.4f}")
    print(f"   nb1181 honest cross-fit RAE = {NB1181_CROSSFIT_LB_ANCHOR:.4f}  "
          f"(LB-faithful anchor)")

    # ---- Save te artefact ----
    te_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_out, te_nb1191.astype(np.float32))
    print(f"\n[save] {te_out}")

    # ---- Save submission CSV (3 cols: SMILES, Molecule Name, pEC50) ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_nb1191.astype(np.float64),
    })
    sub_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_deploy_nb1181.csv")
    sub.to_csv(sub_path, index=False)
    print(f"[save] {sub_path}  rows={len(sub)}")
    assert len(sub) == 513, f"submission CSV has {len(sub)} rows, expected 513"

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "deploy_companion_of": "nb1181",
        "blend_recipe": "naive 1/3 mean of (te_nb1140, te_nb1162, te_nb1172)",
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "components": {
            "te_nb1140": {
                "path": str(te_nb1140_path),
                "in_rae_unb": in_rae_nb1140,
                "mean": float(te_nb1140.mean()),
                "std": float(te_nb1140.std()),
            },
            "te_nb1162": {
                "path": str(te_nb1162_path),
                "in_rae_unb": in_rae_nb1162,
                "mean": float(te_nb1162.mean()),
                "std": float(te_nb1162.std()),
            },
            "te_nb1172": {
                "path": str(te_nb1172_path),
                "in_rae_unb": in_rae_nb1172,
                "mean": float(te_nb1172.mean()),
                "std": float(te_nb1172.std()),
                "build_info": build_info,
            },
        },
        "rae_anchor_te_in_sample_253": in_rae_anchor,
        "rae_anchor_oof_253": rae_anchor_oof,
        "te_nb1191_mean": te_mean,
        "te_nb1191_std": te_std,
        "te_nb1191_min": te_min,
        "te_nb1191_max": te_max,
        "in_rae_253": in_rae_253,
        "delta_in_sample_vs_anchor_te": delta_vs_anchor_te,
        "nb1181_crossfit_lb_anchor": NB1181_CROSSFIT_LB_ANCHOR,
        "row_count": int(len(te_nb1191)),
        "n_nan": n_nan,
        "n_non_finite": n_inf,
        "te_path": str(te_out),
        "submission_path": sub_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "DEPLOY artifact: naive 1/3 mean of three POST-unblind deploys. "
            "in_RAE is in-sample and optimistic; LB-faithful number is the "
            "nb1181 cross-fit RAE 0.5566 for the naive 1/3 mean variant."
        ),
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
        "te_nb1191_mean",
        "te_nb1191_std",
        "te_nb1191_min",
        "te_nb1191_max",
        "in_rae_253",
        "rae_anchor_te_in_sample_253",
        "delta_in_sample_vs_anchor_te",
        "nb1181_crossfit_lb_anchor",
        "row_count",
        "n_nan",
        "te_path",
        "submission_path",
    ):
        print(f"  {k}: {res.get(k)}")
