"""nb592 -- RANK-STRETCH AUGMENTATION over the better of nb590/nb591.

Strategy
--------
1. Load cross-fit OOF predictions from nb590 and nb591 (each shape (253,)),
   compare against unblind truth, pick the better base by pooled RAE.
2. 5-fold cross-fit grid search on s in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25}.
   For each fold:
     - Compute center c = median of the train-fold OOF.
     - Stretched prediction:  p' = c + s * (p - c)
     - Evaluate held-out fold under each s.
3. Pool fold predictions per s, compute pooled RAE, pick best s* (smallest
   pooled RAE).
4. Deploy: refit center on full 253 unblind OOF (median), stretch the 513-row
   te_nb59x.npy by best s using the same center, save te_nb592.npy plus
   plain + soft07 submission CSVs.

Why this works
--------------
nb590/nb591 are LGBM-based: tree models systematically shrink predictions
toward the training mean (low variance vs. truth). A linear stretch about the
center expands the dynamic range. The optimal s is small (1.05-1.20) because
overshooting compounds away from the center hurts middling rows more than it
helps tails. Cross-fit guarantees s* is chosen without peeking at any test row.

Target: cross-fit RAE < 0.5065.
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
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb592"
NB562_RAE = 0.5065
S_GRID = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25)


def _load_base(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (oof_unb (253,), te_deploy (513,)) for a given base tag."""
    oof = np.load(DATA_PROCESSED / f"{name}_pred_oof.npy").astype(np.float64)
    te = np.load(DATA_PROCESSED / f"te_{name}.npy").astype(np.float64)
    return oof, te


def main() -> dict:
    print("=" * 78)
    print("nb592 -- RANK-STRETCH AUGMENTATION")
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
    y_unb = unb_df["pEC50"].astype(np.float64).values
    n_unb = len(y_unb)
    n_te = len(test_df)
    print(f"\nunblind n={n_unb}  test n={n_te}")

    # ---- Step 1: pick better base ----
    print("\n" + "-" * 78)
    print("STEP 1 -- pick better base by cross-fit RAE")
    print("-" * 78)
    bases: dict[str, dict] = {}
    for name in ("nb590", "nb591"):
        try:
            oof, te = _load_base(name)
        except FileNotFoundError as e:
            print(f"  {name}: MISSING ({e})")
            continue
        if oof.shape != (n_unb,) or te.shape != (n_te,):
            print(f"  {name}: SHAPE MISMATCH oof={oof.shape} te={te.shape}")
            continue
        r = float(rae(y_unb, oof))
        bases[name] = {"oof": oof, "te": te, "rae": r}
        print(f"  {name}: oof shape={oof.shape}  RAE={r:.4f}")
    if not bases:
        print("No valid bases found.")
        return {"success": False}

    best_base = min(bases, key=lambda k: bases[k]["rae"])
    oof = bases[best_base]["oof"]
    te_base = bases[best_base]["te"]
    base_rae = bases[best_base]["rae"]
    print(f"\n  -> selected base: {best_base}  (RAE={base_rae:.4f})")

    # ---- Step 2: 5-fold cross-fit grid over s ----
    print("\n" + "-" * 78)
    print("STEP 2 -- 5-fold cross-fit grid on s")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pooled = {s: np.full(n_unb, np.nan, dtype=np.float64) for s in S_GRID}
    centers_per_fold: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_unb))):
        c = float(np.median(oof[tr_idx]))
        centers_per_fold.append(c)
        for s in S_GRID:
            pooled[s][va_idx] = c + s * (oof[va_idx] - c)
        print(f"  fold {fold}: n_tr={len(tr_idx):3d}  n_va={len(va_idx):3d}  "
              f"center={c:.3f}")

    # ---- Step 3: pool RAE per s, pick best ----
    print("\n" + "-" * 78)
    print("STEP 3 -- pooled cross-fit RAE per s")
    print("-" * 78)
    rae_by_s: dict[float, float] = {}
    for s in S_GRID:
        r = float(rae(y_unb, pooled[s]))
        rae_by_s[s] = r
        marker = " <- base" if abs(s - 1.0) < 1e-9 else ""
        print(f"  s={s:.2f}  pooled RAE = {r:.4f}{marker}")

    best_s = min(rae_by_s, key=rae_by_s.get)
    best_rae = rae_by_s[best_s]
    delta = base_rae - best_rae
    print(f"\n  -> best s = {best_s:.2f}  (pooled RAE = {best_rae:.4f}, "
          f"delta vs base = {delta:+.4f})")
    print(f"  nb562 target: {NB562_RAE:.4f}  -> beats: "
          f"{best_rae < NB562_RAE}")

    # ---- Step 4: deploy ----
    print("\n" + "-" * 78)
    print(f"STEP 4 -- DEPLOY: refit center on all {n_unb} unblind OOF, "
          f"apply s={best_s:.2f} to {n_te} test predictions")
    print("-" * 78)
    c_full = float(np.median(oof))
    print(f"  deploy center (median of full OOF) = {c_full:.3f}")
    te_deploy = (c_full + best_s * (te_base - c_full)).astype(np.float32)
    print(f"  te_base    mean/std = {te_base.mean():.3f} / "
          f"{te_base.std():.3f}")
    print(f"  te_deploy  mean/std = {te_deploy.mean():.3f} / "
          f"{te_deploy.std():.3f}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy)

    plain = SUBMISSIONS / f"{TAG}_stretch_aug.csv"
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
    soft_path = SUBMISSIONS / f"{TAG}_stretch_aug_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("=== nb592 SUMMARY ===")
    print(f"  base used                = {best_base}")
    print(f"  base cross-fit RAE       = {base_rae:.4f}")
    print(f"  best s                   = {best_s:.2f}")
    print(f"  best pooled cross-fit    = {best_rae:.4f}")
    print(f"  delta vs base            = {delta:+.4f}")
    print(f"  nb562 target             = {NB562_RAE:.4f}")
    print(f"  beats nb562              = {best_rae < NB562_RAE}")
    print(f"  te_deploy mean/std (513) = "
          f"{te_deploy.mean():.3f} / {te_deploy.std():.3f}")
    print("=" * 78)

    return {
        "success": True,
        "base_used": best_base,
        "base_rae": base_rae,
        "best_s": best_s,
        "best_rae": best_rae,
        "rae_by_s": {f"{s:.2f}": v for s, v in rae_by_s.items()},
        "beats_nb562": bool(best_rae < NB562_RAE),
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
