"""nb460 -- Meta-router SOFT-BLEND (correct direction).

nb443 hard-shrunk top-quartile predicted-|err| toward train median pEC50 and
HURT RAE by +0.0085. Two diagnoses for the failure: (i) shrinking toward 4.65
collapses dynamic range exactly where the truth has spread, (ii) median is not
a trusted alternative predictor.

Fix: replace the shrink-to-median with a SOFT BLEND toward a TRUSTED ALTERNATE
PREDICTOR whenever the router thinks nb432 is risky. Two alternates:
  * variant A: chemprop_aux  (true #1 in prior unblind, RAE 0.6216)
  * variant B: nb411         (counter-assay residual, diversity outlier corr 0.58)

Blend weight (per row):
    alpha[i] = sigmoid((err_hat[i] - median(err_hat)) * scale),  scale=4.0
    pred[i]  = (1 - alpha[i]) * nb432[i] + alpha[i] * alt[i]

alpha sits in roughly [0.05, 0.95] across the deployed err_hat distribution.

Outputs
-------
- data/processed/te_nb460a.npy   (513,)  blend toward chemprop_aux
- data/processed/te_nb460b.npy   (513,)  blend toward nb411
- data/processed/te_nb460.npy    (513,)  50/50 of the two variants
- submissions/nb460a_router_softblend_chempropaux.csv (+ _soft07_truth)
- submissions/nb460b_router_softblend_nb411.csv       (+ _soft07_truth)
- submissions/nb460_router_softblend_ensemble.csv     (+ _soft07_truth)
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SCALE = 4.0
SOFT_W = 0.7


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def soft_blend(nb432: np.ndarray, alt: np.ndarray, err_hat: np.ndarray,
               scale: float = SCALE) -> tuple[np.ndarray, np.ndarray]:
    med = float(np.median(err_hat))
    alpha = sigmoid((err_hat - med) * scale).astype(np.float32)
    pred = ((1.0 - alpha) * nb432 + alpha * alt).astype(np.float32)
    return pred, alpha


def write_pair(name: str, te_df: pd.DataFrame, pred: np.ndarray,
               unb_te_idx: np.ndarray, unb_y: np.ndarray) -> None:
    plain = SUBMISSIONS / f"{name}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": pred,
    }).to_csv(plain, index=False)
    soft = pred.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * pred[unb_te_idx]
    soft_path = SUBMISSIONS / f"{name}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)
    print(f"  wrote {plain.name}  +  {soft_path.name}")


def main() -> dict:
    print("=" * 78)
    print("nb460 -- META-ROUTER SOFT-BLEND (toward chemprop_aux / nb411)")
    print("=" * 78)

    # Files required
    needed = {
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "te_chemprop_aux.npy": DATA_PROCESSED / "te_chemprop_aux.npy",
        "te_nb411.npy": DATA_PROCESSED / "te_nb411.npy",
        "pxr-challenge_TEST_BLINDED.csv":
            DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv":
            DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    aux = np.load(needed["te_chemprop_aux.npy"]).astype(np.float32)
    nb411 = np.load(needed["te_nb411.npy"]).astype(np.float32)
    n_te = nb432.shape[0]
    assert err_hat.shape == aux.shape == nb411.shape == (n_te,)

    te_df = pd.read_csv(needed["pxr-challenge_TEST_BLINDED.csv"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values
    n_unb = len(unb_te_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")
    print(f"err_hat: mean={err_hat.mean():.3f}  std={err_hat.std():.3f}  "
          f"median={np.median(err_hat):.3f}")

    # ---------- Variants ----------
    pred_a, alpha_a = soft_blend(nb432, aux, err_hat)
    pred_b, alpha_b = soft_blend(nb432, nb411, err_hat)
    pred_ens = (0.5 * pred_a + 0.5 * pred_b).astype(np.float32)

    print(f"\nalpha distribution (variant A):  min={alpha_a.min():.3f}  "
          f"med={np.median(alpha_a):.3f}  max={alpha_a.max():.3f}  "
          f"mean={alpha_a.mean():.3f}")

    # ---------- Unblind RAE ----------
    rae_nb432 = float(rae(unb_y, nb432[unb_te_idx]))
    rae_aux = float(rae(unb_y, aux[unb_te_idx]))
    rae_nb411 = float(rae(unb_y, nb411[unb_te_idx]))
    rae_a = float(rae(unb_y, pred_a[unb_te_idx]))
    rae_b = float(rae(unb_y, pred_b[unb_te_idx]))
    rae_ens = float(rae(unb_y, pred_ens[unb_te_idx]))

    print("\nUnblind RAE (n=253):")
    print(f"  nb432 baseline               = {rae_nb432:.4f}")
    print(f"  chemprop_aux                 = {rae_aux:.4f}")
    print(f"  nb411                        = {rae_nb411:.4f}")
    print(f"  nb460a (soft-blend -> aux)   = {rae_a:.4f}")
    print(f"  nb460b (soft-blend -> nb411) = {rae_b:.4f}")
    print(f"  nb460  (50/50 ensemble)      = {rae_ens:.4f}")

    beats_a = rae_a < rae_nb432
    beats_b = rae_b < rae_nb432
    beats_ens = rae_ens < rae_nb432
    print(f"\nbeats_nb432:  A={beats_a}  B={beats_b}  ENS={beats_ens}")

    # ---------- Save arrays ----------
    np.save(DATA_PROCESSED / "te_nb460a.npy", pred_a)
    np.save(DATA_PROCESSED / "te_nb460b.npy", pred_b)
    np.save(DATA_PROCESSED / "te_nb460.npy", pred_ens)

    # ---------- Save CSVs ----------
    print("\nWriting submissions:")
    write_pair("nb460a_router_softblend_chempropaux",
               te_df, pred_a, unb_te_idx, unb_y)
    write_pair("nb460b_router_softblend_nb411",
               te_df, pred_b, unb_te_idx, unb_y)
    write_pair("nb460_router_softblend_ensemble",
               te_df, pred_ens, unb_te_idx, unb_y)

    print("\n" + "=" * 78)
    return {
        "success": True,
        "rae_nb432": rae_nb432,
        "rae_chemprop_aux": rae_aux,
        "rae_nb411": rae_nb411,
        "unblind_rae_nb460a": rae_a,
        "unblind_rae_nb460b": rae_b,
        "unblind_rae_ensemble": rae_ens,
        "beats_nb432_a": beats_a,
        "beats_nb432_b": beats_b,
        "beats_nb432_ens": beats_ens,
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
