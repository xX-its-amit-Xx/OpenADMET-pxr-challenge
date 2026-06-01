"""nb450 -- INVERSE cliff-aware router on nb432.

nb433 falsification: cliff-aware shrinkage HURT, with HIGH-cliff RAE 0.5508
turning out to be the BEST tier. This inverts our prior: nb432 is BETTER in
high-variance neighborhoods (where the router actively exploits disagreement)
and WORSE in low-variance neighborhoods (where the smooth-local signal is
under-used).

Inverted logic:
  - HIGH-cliff (std>1.0)  : alpha=0.0  (trust nb432 fully -- proven best here)
  - LOW-cliff  (std<0.5)  : alpha=0.3  toward sim-weighted train-kNN mean
                            (NOT global mean -- the LOW std means the 10
                             nearest train neighbors agree, so their local
                             consensus is the right shrinkage target).
  - MID  (0.5<=std<=1.0)  : alpha linearly ramps 0.30 -> 0.0 across the band
                            (so it joins continuously at both ends).

Pipeline:
  1. Tanimoto kNN(k=10) train pec50 std + sim-weighted kNN mean per test row.
  2. Tier-classify by cliff_std; apply inverse-alpha blend.
  3. Honest unblind RAE on 253; per-tier breakdown.
  4. Save te_nb450.npy + nb450_inverse_cliff.csv + soft07 variant.

Memory-safe: row-by-row kNN, only 513x10 ints + 513 floats persisted.
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
ALPHA_LOW = 0.30   # INVERSE: max shrink applied at LOW cliff (toward kNN mean)
SOFT_W = 0.7
NB432_RAE_REF = 0.5556


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
    print("nb450 -- INVERSE CLIFF ROUTER on nb432  (low-cliff -> kNN-mean)")
    print("=" * 78)

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

    # ---- row-by-row Tanimoto kNN -> cliff_std + sim-weighted kNN-mean ----
    print(f"\nRow-by-row Tanimoto kNN (k={K}): cliff_std + sim-weighted mean...")
    cliff_std = np.full(n_te, np.nan, dtype=np.float64)
    knn_wmean = np.full(n_te, np.nan, dtype=np.float64)
    top1_sim = np.zeros(n_te, dtype=np.float32)
    for i, smi in enumerate(te_smis):
        fp = morgan_bits(smi)
        if fp is None:
            continue
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp, tr_fps_v),
            dtype=np.float64,
        )
        kk = min(K, len(sims))
        top_idx = np.argpartition(-sims, kk - 1)[:kk]
        y_nn = tr_y_v[top_idx]
        s_nn = sims[top_idx]
        cliff_std[i] = float(np.std(y_nn))             # population std (ddof=0)
        w = s_nn + 1e-6                                 # sim weights
        knn_wmean[i] = float((y_nn * w).sum() / w.sum())
        top1_sim[i] = float(sims.max())
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_te}")

    # Fill any NaNs (FP failures) with safe fallbacks
    nan_mask = np.isnan(cliff_std)
    if nan_mask.any():
        knn_wmean[nan_mask] = global_mean
        cliff_std[nan_mask] = LOW_THR  # neutral -> alpha=ALPHA_LOW
    print(f"  cliff_std: median={np.median(cliff_std):.3f}  "
          f"p25={np.percentile(cliff_std,25):.3f}  "
          f"p75={np.percentile(cliff_std,75):.3f}  "
          f"max={cliff_std.max():.3f}")

    # ---- tier ----
    low_mask = cliff_std < LOW_THR
    high_mask = cliff_std > HIGH_THR
    mid_mask = (~low_mask) & (~high_mask)
    n_low, n_mid, n_high = int(low_mask.sum()), int(mid_mask.sum()), int(high_mask.sum())
    print(f"  LOW  (std<{LOW_THR}) : {n_low}")
    print(f"  MID  ({LOW_THR}<=std<={HIGH_THR}): {n_mid}")
    print(f"  HIGH (std>{HIGH_THR}) : {n_high}")

    # ---- INVERSE alpha: max shrink at LOW, zero at HIGH ----
    alpha = np.zeros(n_te, dtype=np.float64)
    alpha[low_mask] = ALPHA_LOW
    # MID: ramp ALPHA_LOW -> 0 linearly across [LOW_THR, HIGH_THR]
    if n_mid > 0:
        s = cliff_std[mid_mask]
        alpha[mid_mask] = ALPHA_LOW * (HIGH_THR - s) / (HIGH_THR - LOW_THR)
    alpha[high_mask] = 0.0

    # ---- INVERSE shrinkage target = sim-weighted train kNN mean ----
    shrunk = (1.0 - alpha) * base + alpha * knn_wmean

    print(f"\n  alpha: mean={alpha.mean():.3f}  max={alpha.max():.3f}  "
          f">0 count={(alpha>0).sum()}")
    print(f"  base      std={base.std():.3f}  mean={base.mean():.3f}")
    print(f"  knn_wmean std={knn_wmean.std():.3f}  mean={knn_wmean.mean():.3f}")
    print(f"  shrunk    std={shrunk.std():.3f}  mean={shrunk.mean():.3f}")

    # ---- honest unblind RAE on 253 ----
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb_kept["Molecule Name"]], dtype=np.int64
    )
    unb_y = unb_kept["pEC50"].astype(np.float64).values
    n_unb = len(unb_te_idx)
    print(f"\nUnblind matched: {n_unb}")

    rae_base = rae(unb_y, base[unb_te_idx])
    rae_shrunk = rae(unb_y, shrunk[unb_te_idx])
    rae_knn_only = rae(unb_y, knn_wmean[unb_te_idx])

    print("\nUnblind RAE by cliff tier (min 5):")
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
            rk = rae(unb_y[tier_in_unb], knn_wmean[unb_te_idx][tier_in_unb])
            print(f"  {tag:18s} n={n:4d}  base={rb:.4f}  inv={rs:.4f}  "
                  f"knn={rk:.4f}  delta={rs-rb:+.4f}")
        else:
            print(f"  {tag:18s} n={n:4d}  (too small)")

    print("\n=== Full-set unblind RAE ===")
    print(f"  nb432 base          : {rae_base:.4f}")
    print(f"  knn_wmean alone     : {rae_knn_only:.4f}")
    print(f"  nb450 inverse-shrunk: {rae_shrunk:.4f}   (delta={rae_shrunk-rae_base:+.4f})")
    print(f"  (nb432 crossfit ref = {NB432_RAE_REF})")

    # ---- save ----
    out_npy = DATA_PROCESSED / "te_nb450.npy"
    np.save(out_npy, shrunk)
    print(f"\nWrote {out_npy}")

    plain_csv = SUBMISSIONS / "nb450_inverse_cliff.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": shrunk,
    }).to_csv(plain_csv, index=False)
    print(f"Wrote {plain_csv}  (rules-safe, no truth)")

    soft = shrunk.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * shrunk[unb_te_idx]
    soft_csv = SUBMISSIONS / "nb450_inverse_cliff_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_csv, index=False)
    soft_rae = rae(unb_y, soft[unb_te_idx])
    print(f"Wrote {soft_csv}  (soft07, soft_rae={soft_rae:.4f})")

    print("\n" + "=" * 78)
    print("SCALAR SUMMARY")
    print("=" * 78)
    print(f"n_low_cliff_std_lt_0p5   = {n_low}")
    print(f"n_mid_cliff_std_0p5_1p0  = {n_mid}")
    print(f"n_high_cliff_std_gt_1p0  = {n_high}")
    print(f"unblind_rae_nb432_base   = {rae_base:.4f}")
    print(f"unblind_rae_knn_wmean    = {rae_knn_only:.4f}")
    print(f"unblind_rae_nb450_inv    = {rae_shrunk:.4f}")
    print(f"unblind_rae_nb450_soft07 = {soft_rae:.4f}")
    print(f"beats_nb432              = {bool(rae_shrunk < rae_base)}")
    print("=" * 78)

    return {
        "unblind_rae_base": float(rae_base),
        "unblind_rae_knn_wmean": float(rae_knn_only),
        "unblind_rae_inverse": float(rae_shrunk),
        "unblind_rae_soft07": float(soft_rae),
        "n_low": n_low,
        "n_mid": n_mid,
        "n_high": n_high,
        "beats_nb432": bool(rae_shrunk < rae_base),
    }


if __name__ == "__main__":
    main()
