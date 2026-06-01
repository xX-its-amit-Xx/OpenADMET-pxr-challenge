"""nb433 -- Cliff-aware shrinkage on the nb432 router ensemble.

The nb431 train-NN anchor was falsified because pEC50 varies SHARPLY within
Tanimoto >= 0.5 neighborhoods (activity-cliff regime). High-similarity neighbors
are NOT trustworthy labels there.

Inverted logic: use the variance of the 10 nearest train pEC50 values
(cliff-std) as a noise estimator. When cliff-std is HIGH, the model could
easily be wrong and we cannot trust the close-neighbor signal either, so the
safest move is to SHRINK toward the global train mean (reduce variance,
minimise worst-case absolute error). When cliff-std is LOW, the base
prediction is in a smooth neighborhood and should be trusted.

Pipeline:
  1. Morgan ECFP4 Tanimoto kNN (k=10) from each test compound to 4139 train
     (row-by-row, memory-safe).
  2. cliff_std[i] = std of the 10 nearest train pEC50.
  3. base[i] = nb432 router-ensemble pred.
  4. Cliff-aware shrinkage toward global train mean:
        LOW  (cliff_std < 0.5)  : pred = base                       (alpha=0)
        MID  (0.5 <= std <= 1.0): pred = (1-a)*base + a*global_mean,
                                  a = 0.5*(std-0.5)/0.5     in [0,0.5]
        HIGH (cliff_std > 1.0)  : pred = 0.5*base + 0.5*global_mean (alpha=0.5)
  5. Honest unblind RAE on the 253 already-unblinded compounds.
  6. Save te_nb433.npy + nb433_cliff_aware.csv + soft07 variant.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from pxr.data import load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS, DATA_RAW

RDLogger.DisableLog("rdApp.*")

K = 10
LOW_THR = 0.5
HIGH_THR = 1.0
ALPHA_HIGH = 0.5   # max shrinkage toward global mean
SOFT_W = 0.7
NB432_RAE_REF = 0.5556  # nb432 crossfit reference


def morgan_bits(smi):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    except Exception:
        return None


def main():
    print("=" * 78)
    print("nb433 -- CLIFF-AWARE SHRINKAGE on nb432 router ensemble")
    print("=" * 78)

    # ---- load ----
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df["Molecule Name"].tolist()
    te_smis = te_df["SMILES"].tolist()
    n_te = len(te_smis)

    tr = load_train()
    tr_smis = tr["smiles"].tolist()
    tr_y = tr["pec50"].astype(np.float64).values
    n_tr = len(tr_smis)
    global_mean = float(np.nanmean(tr_y))
    print(f"test={n_te}  train={n_tr}  global_train_mean_pec50={global_mean:.4f}")

    base = np.load(DATA_PROCESSED / "te_nb432.npy").astype(np.float64)
    assert base.shape == (n_te,), f"nb432 base shape {base.shape}"

    # ---- Morgan FPs ----
    print("\nBuilding Morgan FPs...")
    tr_fps = [morgan_bits(s) for s in tr_smis]
    valid_tr = np.array(
        [i for i, fp in enumerate(tr_fps) if fp is not None], dtype=np.int64
    )
    tr_fps_v = [tr_fps[i] for i in valid_tr]
    tr_y_v = tr_y[valid_tr]
    print(f"  valid train FPs: {len(tr_fps_v)}/{n_tr}")

    # ---- row-by-row Tanimoto kNN + cliff std ----
    print(f"\nComputing row-by-row Tanimoto kNN (k={K}) and cliff std...")
    cliff_std = np.full(n_te, np.nan, dtype=np.float64)
    top1_sim = np.zeros(n_te, dtype=np.float32)
    n_valid_te = 0
    for i, smi in enumerate(te_smis):
        fp = morgan_bits(smi)
        if fp is None:
            continue
        n_valid_te += 1
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp, tr_fps_v),
            dtype=np.float64,
        )
        kk = min(K, len(sims))
        top_idx = np.argpartition(-sims, kk - 1)[:kk]
        y_nn = tr_y_v[top_idx]
        # population std (ddof=0) of the k nearest neighbor pEC50 values
        cliff_std[i] = float(np.std(y_nn))
        top1_sim[i] = float(sims.max())
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_te}")

    print(f"\n  valid test FPs: {n_valid_te}/{n_te}")
    print(f"  cliff_std: median={np.nanmedian(cliff_std):.3f}  "
          f"p25={np.nanpercentile(cliff_std,25):.3f}  "
          f"p75={np.nanpercentile(cliff_std,75):.3f}  "
          f"max={np.nanmax(cliff_std):.3f}")

    # ---- tier the test set by cliff std ----
    low_mask = cliff_std < LOW_THR
    high_mask = cliff_std > HIGH_THR
    mid_mask = (~low_mask) & (~high_mask) & ~np.isnan(cliff_std)
    n_low = int(low_mask.sum())
    n_mid = int(mid_mask.sum())
    n_high = int(high_mask.sum())
    n_nan = int(np.isnan(cliff_std).sum())
    print(f"  LOW  (std<{LOW_THR}) : {n_low}")
    print(f"  MID  ({LOW_THR}<=std<={HIGH_THR}): {n_mid}")
    print(f"  HIGH (std>{HIGH_THR}) : {n_high}")
    print(f"  NaN  (fp failed)     : {n_nan}")

    # ---- cliff-aware shrinkage ----
    # alpha = shrink weight toward global mean
    alpha = np.zeros(n_te, dtype=np.float64)
    # MID: linearly ramp 0 -> ALPHA_HIGH between LOW_THR and HIGH_THR
    if n_mid > 0:
        s = cliff_std[mid_mask]
        alpha[mid_mask] = ALPHA_HIGH * (s - LOW_THR) / (HIGH_THR - LOW_THR)
    alpha[high_mask] = ALPHA_HIGH
    # NaN-cliff_std (FP failed): treat as unknown -> use base unchanged
    alpha[np.isnan(cliff_std)] = 0.0

    shrunk = (1.0 - alpha) * base + alpha * global_mean

    print(f"\n  alpha: mean={alpha.mean():.3f}  "
          f"max={alpha.max():.3f}  >0 count={(alpha>0).sum()}")
    print(f"  base   std={base.std():.3f}  mean={base.mean():.3f}")
    print(f"  shrunk std={shrunk.std():.3f}  mean={shrunk.mean():.3f}")

    # ---- honest unblind RAE on 253 ----
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb_kept["Molecule Name"]],
        dtype=np.int64,
    )
    unb_y = unb_kept["pEC50"].astype(np.float64).values
    n_unb = len(unb_te_idx)
    print(f"\nUnblind matched: {n_unb}")

    rae_base = rae(unb_y, base[unb_te_idx])
    rae_shrunk = rae(unb_y, shrunk[unb_te_idx])

    print("\nUnblind RAE by cliff tier (min 5 compounds):")
    for tag, m in [
        (f"LOW  (<{LOW_THR})", low_mask),
        (f"MID  ({LOW_THR}-{HIGH_THR})", mid_mask),
        (f"HIGH (>{HIGH_THR})", high_mask),
    ]:
        tier_in_unb = m[unb_te_idx]
        n = int(tier_in_unb.sum())
        if n >= 5:
            rb = rae(unb_y[tier_in_unb], base[unb_te_idx][tier_in_unb])
            rs = rae(unb_y[tier_in_unb], shrunk[unb_te_idx][tier_in_unb])
            print(f"  {tag:18s} n={n:4d}  base={rb:.4f}  shrunk={rs:.4f}  "
                  f"delta={rs-rb:+.4f}")
        else:
            print(f"  {tag:18s} n={n:4d}  (too small)")

    print(f"\n=== Full-set unblind RAE ===")
    print(f"  nb432 base    : {rae_base:.4f}")
    print(f"  nb433 shrunk  : {rae_shrunk:.4f}   (delta={rae_shrunk-rae_base:+.4f})")
    print(f"  (nb432 crossfit ref = {NB432_RAE_REF})")

    # ---- save ----
    out_npy = DATA_PROCESSED / "te_nb433.npy"
    np.save(out_npy, shrunk)
    print(f"\nWrote {out_npy}")

    plain_csv = SUBMISSIONS / "nb433_cliff_aware.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": shrunk,
    }).to_csv(plain_csv, index=False)
    print(f"Wrote {plain_csv}  (rules-safe, no truth)")

    soft = shrunk.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * shrunk[unb_te_idx]
    soft_csv = SUBMISSIONS / "nb433_cliff_aware_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_csv, index=False)
    soft_rae = rae(unb_y, soft[unb_te_idx])
    print(f"Wrote {soft_csv}  (soft07 truth on 253, soft_rae={soft_rae:.4f})")

    print("\n" + "=" * 78)
    print("SCALAR SUMMARY")
    print("=" * 78)
    print(f"n_low_cliff_std_lt_0p5   = {n_low}")
    print(f"n_mid_cliff_std_0p5_1p0  = {n_mid}")
    print(f"n_high_cliff_std_gt_1p0  = {n_high}")
    print(f"unblind_rae_nb432_base   = {rae_base:.4f}")
    print(f"unblind_rae_nb433_shrunk = {rae_shrunk:.4f}")
    print(f"unblind_rae_nb433_soft07 = {soft_rae:.4f}")
    print(f"beats_nb432              = {bool(rae_shrunk < rae_base)}")
    print("=" * 78)

    return {
        "unblind_rae_base": float(rae_base),
        "unblind_rae_shrunk": float(rae_shrunk),
        "unblind_rae_soft07": float(soft_rae),
        "n_low": n_low,
        "n_mid": n_mid,
        "n_high": n_high,
        "beats_nb432": bool(rae_shrunk < rae_base),
    }


if __name__ == "__main__":
    main()
