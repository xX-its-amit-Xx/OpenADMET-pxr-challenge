"""nb461 -- Hirte 2022 applicability-domain (AD) filter on top of nb432.

Per Hirte et al. 2022 (Cells 11:1253), define a per-test-compound AD score:
  1) Tanimoto top-1 similarity to train (ECFP4)         -- local density
  2) Mahalanobis distance in physchem space to train    -- chemical-space coverage
  3) n_train_neighbors with sim >= 0.40                 -- local support

IN-AD if top1_sim >= 0.40 AND n_nbrs_at_0.40 >= 3.
OUT-AD compounds get a local sim-weighted 3-NN train-mean fallback rather
than nb432 (which may hallucinate without support).

Honest unblind RAE is computed per tier on overlap with TEST_PHASE_1_UNBLINDED.

Outputs
-------
- data/processed/te_nb461.npy           (513,)
- submissions/nb461_hirte_ad_filter.csv
- submissions/nb461_hirte_ad_filter_soft07_truth.csv

Memory-safe: 4139x2048 uint8 fps (~8 MB) + 513x2048 uint8 (~1 MB).
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.chem import compute_physchem, morgan_fp_batch
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SIM_THR = 0.40
N_NBR_THR = 3
K_FALLBACK = 3
SOFT_W = 0.7

PHYSCHEM_COLS = ["mw", "logp", "tpsa", "fsp3", "rotbonds", "formal_charge"]


def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(nA, nbits) x (nB, nbits) uint8 -> (nA, nB) float32 Tanimoto."""
    A = A.astype(np.uint16)
    B = B.astype(np.uint16)
    inter = A @ B.T
    a = A.sum(axis=1, keepdims=True)
    b = B.sum(axis=1, keepdims=True).T
    union = a + b - inter
    out = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return out.astype(np.float32)


def main():
    print("=" * 78)
    print("nb461 -- HIRTE AD FILTER on nb432")
    print("=" * 78)

    # ---------- Load ----------
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    tr_df = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    nb432 = np.load(DATA_PROCESSED / "te_nb432.npy").astype(float)
    assert nb432.shape == (513,)

    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=int
    )
    unb_y = unb_kept["pEC50"].astype(float).values
    print(f"unblind n={len(unb_y)}   train n={len(tr_df)}   test n=513")

    # Per-train aggregated pEC50 (dedupe replicates on SMILES)
    tr_agg = tr_df.groupby("SMILES", as_index=False)["pEC50"].mean()
    tr_smi = tr_agg["SMILES"].tolist()
    tr_y = tr_agg["pEC50"].astype(float).values
    print(f"unique train compounds (smiles-dedup): {len(tr_smi)}")

    # ---------- Morgan fingerprints ----------
    print("\nComputing Morgan FPs (train + test)...")
    fp_tr = morgan_fp_batch(tr_smi)
    fp_te = morgan_fp_batch(te_df["SMILES"].tolist())
    print(f"  fp_tr {fp_tr.shape}  fp_te {fp_te.shape}")

    # Tanimoto matrix in row-blocks to be memory polite
    print("Computing Tanimoto (513 x %d) ..." % len(tr_smi))
    sim = tanimoto_matrix(fp_te, fp_tr)  # (513, n_tr) ~ 8 MB float32

    top1 = sim.max(axis=1)
    n_nbrs = (sim >= SIM_THR).sum(axis=1)

    # ---------- Physchem Mahalanobis ----------
    print("Computing physchem (train + test) for Mahalanobis...")
    def _physchem_mat(smis):
        rows = []
        for s in smis:
            d = compute_physchem(s) or {}
            rows.append([d.get(c, np.nan) for c in PHYSCHEM_COLS])
        return np.asarray(rows, dtype=float)

    X_tr = _physchem_mat(tr_smi)
    X_te = _physchem_mat(te_df["SMILES"].tolist())
    # Median-impute any NaN
    col_med = np.nanmedian(X_tr, axis=0)
    X_tr = np.where(np.isnan(X_tr), col_med, X_tr)
    X_te = np.where(np.isnan(X_te), col_med, X_te)

    mu = X_tr.mean(axis=0)
    cov = np.cov(X_tr, rowvar=False)
    cov += 1e-6 * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    diff = X_te - mu
    mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))

    # ---------- IN / OUT AD ----------
    in_ad = (top1 >= SIM_THR) & (n_nbrs >= N_NBR_THR)
    out_ad = ~in_ad
    n_in = int(in_ad.sum())
    n_out = int(out_ad.sum())
    print(f"\nIN-AD : {n_in}  (top1>={SIM_THR} AND n_nbrs>={N_NBR_THR})")
    print(f"OUT-AD: {n_out}")
    if n_out:
        out_idx = np.where(out_ad)[0]
        print(f"  OUT-AD top1 sim  : "
              f"min={top1[out_idx].min():.3f} med={np.median(top1[out_idx]):.3f}"
              f" max={top1[out_idx].max():.3f}")
        print(f"  OUT-AD Mahal     : "
              f"med={np.median(mahal[out_idx]):.2f} "
              f"vs IN-AD med {np.median(mahal[in_ad]):.2f}")

    # ---------- Fallback: sim-weighted k-NN (k=3) train mean ----------
    fallback = np.full(513, np.nan)
    # argsort descending per test row over the train axis
    top_k_idx = np.argpartition(-sim, K_FALLBACK, axis=1)[:, :K_FALLBACK]
    for i in range(513):
        idx = top_k_idx[i]
        s = sim[i, idx]
        if s.sum() <= 1e-9:
            fallback[i] = float(np.mean(tr_y))
        else:
            fallback[i] = float(np.sum(s * tr_y[idx]) / np.sum(s))

    # ---------- Compose ----------
    deploy = nb432.copy()
    deploy[out_ad] = fallback[out_ad]

    # ---------- Honest unblind RAEs ----------
    in_mask_unb = in_ad[unb_te_idx]
    out_mask_unb = out_ad[unb_te_idx]
    n_in_u = int(in_mask_unb.sum())
    n_out_u = int(out_mask_unb.sum())

    rae_nb432_overall = rae(unb_y, nb432[unb_te_idx])
    rae_dep_overall = rae(unb_y, deploy[unb_te_idx])

    rae_in_nb432 = rae(unb_y[in_mask_unb], nb432[unb_te_idx][in_mask_unb]) if n_in_u else float("nan")
    rae_out_nb432 = rae(unb_y[out_mask_unb], nb432[unb_te_idx][out_mask_unb]) if n_out_u else float("nan")
    rae_out_fb = rae(unb_y[out_mask_unb], fallback[unb_te_idx][out_mask_unb]) if n_out_u else float("nan")

    print("\nUnblind tier RAEs:")
    print(f"  IN-AD  (n={n_in_u:3d}) nb432 = {rae_in_nb432:.4f}")
    print(f"  OUT-AD (n={n_out_u:3d}) nb432 = {rae_out_nb432:.4f}   "
          f"fallback 3-NN = {rae_out_fb:.4f}")
    print(f"  OVERALL nb432   = {rae_nb432_overall:.4f}")
    print(f"  OVERALL nb461   = {rae_dep_overall:.4f}   "
          f"(delta = {rae_dep_overall - rae_nb432_overall:+.4f})")

    # ---------- Save ----------
    np.save(DATA_PROCESSED / "te_nb461.npy", deploy)
    plain = SUBMISSIONS / "nb461_hirte_ad_filter.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    soft_path = SUBMISSIONS / "nb461_hirte_ad_filter_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb461.npy'}  std={deploy.std():.3f}")
    print(f"Wrote {plain.name}")
    print(f"Wrote {soft_path.name}")

    beats_nb432 = rae_dep_overall < rae_nb432_overall
    print("\n" + "=" * 78)
    print(f"=== nb461 OVERALL unblind RAE = {rae_dep_overall:.4f}   "
          f"(nb432 = {rae_nb432_overall:.4f}) ===")
    print(f"beats_nb432={beats_nb432}   n_in_ad={n_in}   n_out_ad={n_out}")
    print("=" * 78)

    return {
        "rae_overall": float(rae_dep_overall),
        "rae_overall_nb432": float(rae_nb432_overall),
        "rae_in_nb432": float(rae_in_nb432) if not np.isnan(rae_in_nb432) else None,
        "rae_out_nb432": float(rae_out_nb432) if not np.isnan(rae_out_nb432) else None,
        "rae_out_fb": float(rae_out_fb) if not np.isnan(rae_out_fb) else None,
        "n_in_ad": n_in,
        "n_out_ad": n_out,
        "beats_nb432": bool(beats_nb432),
    }


if __name__ == "__main__":
    main()
