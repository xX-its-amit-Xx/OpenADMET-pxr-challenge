"""nb902 -- SMILES-augmentation test-time training.

For each of the 513 test compounds we generate 50 random SMILES variants
(non-canonical, randomized atom ordering via ``Chem.MolToSmiles(mol,
doRandom=True, canonical=False)``). Each variant lands on a slightly
different Morgan/RDKit feature vector because the descriptor calculator and
the bit-hash are sensitive to atom ordering / canonical form (Morgan
proper is order-invariant, RDKit 2D descriptors and a handful of
fingerprint corner cases are not). Averaging predictions across all 50
variants is a cheap, single-model variance-reduction trick.

Pipeline:
  1.  Train nb120-style Huber LGBM (alpha=1.0) on the full 4139-row CRC
      set using the canonical SMILES.  We reuse the te_chemprop_aux/te_nb120
      assay-decomp augmentation pattern from nb120, but to keep the
      script self-contained we just fit a Huber LGBM on the bare combined
      features -- the goal is to measure the variance-reduction effect of
      SMILES augmentation, not to chase the absolute best single model.
  2.  Generate 50 random SMILES per test compound -> (513, 50) string array.
  3.  Featurize each variant with ``combined`` (Morgan + RDKit).
      Cache the (50, 513, 2265) float32 cube at
      ``C:/pxr_artifacts/nb902_X_aug.npy`` (~232 MB).
  4.  Predict pEC50 for each variant -> (50, 513) cube,
      mean per compound -> ``te_nb902.npy``.
  5.  Evaluate in-sample RAE on the 253-compound Phase-2 unblind set,
      write submissions/nb902_smiles_aug_ttt.csv.

Hypothesis: averaging predictions across canonical-variant feature
vectors produces an ensemble-like noise floor reduction without having
to train multiple models.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pxr.data import load_train, load_test  # noqa: E402
from pxr.featurize import combined, impute  # noqa: E402
from pxr.eval import rae  # noqa: E402
from pxr.paths import DATA_PROCESSED, SUBMISSIONS  # noqa: E402

ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

N_AUG = 50
SEED = 42

HUBER_PARAMS = dict(
    objective="huber",
    alpha=1.0,
    n_estimators=1500,
    num_leaves=64,
    learning_rate=0.03,
    min_child_samples=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.1,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)


def random_smiles(smi: str, n: int, rng: np.random.Generator) -> list[str]:
    """Return up to ``n`` random non-canonical SMILES variants for ``smi``.

    Uses ``Chem.MolToSmiles(mol, doRandom=True, canonical=False)`` with
    fresh RDKit calls; the first variant is the canonical SMILES so a
    baseline featurization is always included.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi] * n
    out: list[str] = [Chem.MolToSmiles(mol)]  # canonical first
    seen = {out[0]}
    # cap attempts to avoid infinite loops on tiny molecules
    for _ in range(n * 4):
        if len(out) >= n:
            break
        try:
            v = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        except Exception:
            v = out[0]
        if v not in seen:
            seen.add(v)
            out.append(v)
    while len(out) < n:  # pad with canonical if not enough uniques
        out.append(out[0])
    return out[:n]


def build_aug_smiles(smiles: list[str]) -> np.ndarray:
    """Return (N, N_AUG) object array of random SMILES variants."""
    rng = np.random.default_rng(SEED)
    out = np.empty((len(smiles), N_AUG), dtype=object)
    for i, smi in enumerate(smiles):
        out[i] = random_smiles(smi, N_AUG, rng)
        if (i + 1) % 50 == 0:
            print(f"  random_smiles: {i + 1}/{len(smiles)}", flush=True)
    return out


def featurize_aug(aug_smi: np.ndarray) -> np.ndarray:
    """Return (N_AUG, N_test, D) float32 feature cube.

    We featurize column-by-column (all variants 0, then all variants 1, ...)
    so the impute step gets sensible column medians at each replicate.
    """
    n_test, n_aug = aug_smi.shape
    feats = []
    for k in range(n_aug):
        t0 = time.time()
        Xk = impute(combined(aug_smi[:, k].tolist())).astype(np.float32)
        feats.append(Xk)
        if (k + 1) % 5 == 0 or k == 0:
            print(f"  featurize_aug variant {k + 1}/{n_aug}  "
                  f"shape={Xk.shape}  {time.time() - t0:.1f}s", flush=True)
    return np.stack(feats, axis=0)


def main() -> float:
    print("=== nb902: SMILES augmentation test-time training ===\n")

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)

    print(f"[data] train={len(tr)}  test={len(te)}")

    # ---- Train Huber LGBM on canonical features ----
    print("[feat] computing canonical train/test combined features...")
    X_tr = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    X_te = impute(combined(te["smiles"].tolist())).astype(np.float32)
    print(f"[feat] X_tr={X_tr.shape}  X_te={X_te.shape}")

    print("[lgbm] fitting Huber(alpha=1.0)...")
    t0 = time.time()
    model = lgb.LGBMRegressor(**HUBER_PARAMS)
    model.fit(X_tr, y_tr)
    print(f"[lgbm] fit in {time.time() - t0:.1f}s, "
          f"best_iter={model.best_iteration_}")

    # canonical baseline prediction (k=0 in our cube)
    te_canon = model.predict(X_te)
    print(f"[canon] mean={te_canon.mean():.3f}  std={te_canon.std():.3f}")

    # ---- Build N_AUG SMILES variants per test compound ----
    print(f"\n[aug] generating {N_AUG} random SMILES per test compound...")
    aug_smi_path = ART / "nb902_aug_smiles.npy"
    if aug_smi_path.exists():
        aug_smi = np.load(aug_smi_path, allow_pickle=True)
        print(f"[aug] loaded cached SMILES from {aug_smi_path}  "
              f"shape={aug_smi.shape}")
    else:
        aug_smi = build_aug_smiles(te["smiles"].tolist())
        np.save(aug_smi_path, aug_smi, allow_pickle=True)
        print(f"[aug] saved {aug_smi_path}  shape={aug_smi.shape}")

    # ---- Featurize all variants ----
    print(f"\n[feat-aug] featurizing {N_AUG} variants ...")
    cube_path = ART / "nb902_X_aug.npy"
    if cube_path.exists():
        X_cube = np.load(cube_path, mmap_mode="r")
        print(f"[feat-aug] loaded cube from {cube_path}  shape={X_cube.shape}")
    else:
        X_cube = featurize_aug(aug_smi)  # (N_AUG, 513, 2265)
        print(f"[feat-aug] cube={X_cube.shape}  "
              f"{X_cube.nbytes / 1e6:.1f} MB")
        np.save(cube_path, X_cube)
        print(f"[feat-aug] saved {cube_path}")

    # ---- Predict every variant ----
    print(f"\n[predict] running model on {N_AUG} variants...")
    pred_cube = np.zeros((N_AUG, len(te)), dtype=np.float32)
    for k in range(N_AUG):
        pred_cube[k] = model.predict(np.asarray(X_cube[k]))
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  variant {k + 1:2d}/{N_AUG}  "
                  f"mean={pred_cube[k].mean():.3f}  "
                  f"std={pred_cube[k].std():.3f}", flush=True)

    te_mean = pred_cube.mean(axis=0)
    te_var = pred_cube.var(axis=0).mean()
    print(f"\n[agg] te_mean.shape={te_mean.shape}  "
          f"mean across-variant variance = {te_var:.5f}")

    # Save predictions cube + mean
    np.save(ART / "nb902_pred_cube.npy", pred_cube)
    np.save(DATA_PROCESSED / "te_nb902.npy", te_mean.astype(np.float32))

    # ---- In-sample RAE on 253 Phase-2 unblind ----
    idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_un = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    in_rae_aug = float(rae(y_un, te_mean[idx]))
    in_rae_canon = float(rae(y_un, te_canon[idx]))
    print(f"\n[eval] in-sample RAE on {len(idx)} unblind:")
    print(f"        canonical only : {in_rae_canon:.4f}")
    print(f"        50x aug average: {in_rae_aug:.4f}  "
          f"(delta={in_rae_aug - in_rae_canon:+.4f})")

    # ---- Submission ----
    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_mean,
    })
    out = SUBMISSIONS / "nb902_smiles_aug_ttt.csv"
    sub.to_csv(out, index=False)
    print(f"[sub] wrote {out}  ({len(sub)} rows)")

    return in_rae_aug


if __name__ == "__main__":
    in_rae = main()
    print(f"\nDONE  in_RAE={in_rae:.4f}  n_aug={N_AUG}")
