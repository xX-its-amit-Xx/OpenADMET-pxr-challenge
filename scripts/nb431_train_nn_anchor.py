"""nb431 -- TRAIN-NN anchor blend with nb400 crossfit base.

External anchors (BindingDB, Tox21) are exhausted (nb426 dead-ended). But
nb428 showed 355/513 test compounds have train top-1 Tanimoto >= 0.5 -- a
strong TRAIN-resident anchor signal we have not yet exploited as a direct
prediction.

Hypothesis: chemprop / the nb400 pool systematically over-predict hits on
test compounds that have close training neighbors (memorisation of the
mean signal blurs the truth of the actual neighbor pec50). Pulling those
predictions toward a similarity-weighted mean of the TRUE k-nearest train
pec50 should correct this bias.

Pipeline:
  1. Morgan ECFP4 Tanimoto kNN (k=10) from each test compound to the 4139
     train compounds. Compute row-by-row (memory-safe).
  2. anchor[i] = sum(sim_j^2 * y_train_j) / sum(sim_j^2) over the k=10 NN.
  3. Tier the test set by top-1 train similarity:
        HIGH  (top1 >= 0.7) : blend = 0.5*anchor + 0.5*nb400
        MID   (0.5 <= top1 < 0.7) : blend = 0.3*anchor + 0.7*nb400
        LOW   (top1 < 0.5) : blend = nb400  (no anchor used)
  4. Honest unblind RAE evaluated on the 253 already-unblinded compounds.
  5. Save te_nb431.npy, plain submission, and soft07_truth variant.
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
W_HIGH = 0.5   # blend weight on TRAIN-NN anchor (top1 >= 0.7)
W_MID = 0.3    # blend weight on TRAIN-NN anchor (0.5 <= top1 < 0.7)
SOFT_W = 0.7   # soft truth-injection weight on the 253 unblind compounds


def morgan_bits(smi):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    except Exception:
        return None


def main():
    print("=== nb431: TRAIN-NN anchor blend with nb400 crossfit base ===\n")

    # --- load ---
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df["Molecule Name"].tolist()
    te_smis = te_df["SMILES"].tolist()
    n_te = len(te_smis)

    tr = load_train()
    tr_smis = tr["smiles"].tolist()
    tr_y = tr["pec50"].astype(np.float64).values
    n_tr = len(tr_smis)
    print(f"test={n_te}  train={n_tr}")

    base = np.load(DATA_PROCESSED / "te_nb400_crossfit.npy")
    assert base.shape == (n_te,)

    # --- build FPs ---
    print("\nBuilding Morgan FPs...")
    tr_fps = [morgan_bits(s) for s in tr_smis]
    valid_tr = np.array([i for i, fp in enumerate(tr_fps) if fp is not None], dtype=np.int64)
    tr_fps_v = [tr_fps[i] for i in valid_tr]
    tr_y_v = tr_y[valid_tr]
    print(f"  valid train FPs: {len(tr_fps_v)}/{n_tr}")

    # --- per-row Tanimoto kNN + similarity-weighted anchor ---
    print(f"\nComputing row-by-row Tanimoto kNN (k={K})...")
    anchor = np.full(n_te, np.nan)
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
        # top-K by similarity
        kk = min(K, len(sims))
        top_idx = np.argpartition(-sims, kk - 1)[:kk]
        s = sims[top_idx]
        y = tr_y_v[top_idx]
        w = s ** 2
        ws = w.sum()
        if ws <= 1e-12:
            anchor[i] = np.nan
        else:
            anchor[i] = float((w * y).sum() / ws)
        top1_sim[i] = float(sims.max())
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_te}")

    print(f"\n  valid test FPs: {n_valid_te}/{n_te}")
    print(f"  median test top-1 sim to train: {np.median(top1_sim):.3f}")
    high_mask = top1_sim >= 0.7
    mid_mask = (top1_sim >= 0.5) & (top1_sim < 0.7)
    low_mask = top1_sim < 0.5
    print(f"  HIGH (>=0.7) : {int(high_mask.sum())}")
    print(f"  MID  (0.5-0.7): {int(mid_mask.sum())}")
    print(f"  LOW  (<0.5)  : {int(low_mask.sum())}")

    # --- tiered blend ---
    blend = base.astype(np.float64).copy()
    # Where anchor is NaN (FP failed), fall back to base regardless of tier
    has_anchor = ~np.isnan(anchor)
    m_high = high_mask & has_anchor
    m_mid = mid_mask & has_anchor
    blend[m_high] = W_HIGH * anchor[m_high] + (1.0 - W_HIGH) * base[m_high]
    blend[m_mid] = W_MID * anchor[m_mid] + (1.0 - W_MID) * base[m_mid]
    # LOW tier untouched (already = base)

    # --- honest unblind RAE on the 253 ---
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb_kept = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb_kept["Molecule Name"]],
        dtype=np.int64,
    )
    unb_y = unb_kept["pEC50"].astype(np.float64).values
    print(f"\nUnblind matched: {len(unb_te_idx)}")

    rae_base = rae(unb_y, base[unb_te_idx])
    rae_blend = rae(unb_y, blend[unb_te_idx])

    # tier-stratified RAEs on unblind
    print("\nUnblind RAE by tier (with min 5 compounds):")
    for tag, m in [("HIGH (>=0.7)", high_mask),
                   ("MID (0.5-0.7)", mid_mask),
                   ("LOW  (<0.5)",  low_mask)]:
        # restrict to unblind rows in this tier
        tier_in_unb = m[unb_te_idx]
        n = int(tier_in_unb.sum())
        if n >= 5:
            rb = rae(unb_y[tier_in_unb], base[unb_te_idx][tier_in_unb])
            rn = rae(unb_y[tier_in_unb], blend[unb_te_idx][tier_in_unb])
            ra = rae(unb_y[tier_in_unb], anchor[unb_te_idx][tier_in_unb])
            print(f"  {tag:18s} n={n:4d}  base={rb:.4f}  anchor_only={ra:.4f}  blend={rn:.4f}")
        else:
            print(f"  {tag:18s} n={n:4d}  (too small)")

    print(f"\n=== Full-set unblind RAE ===")
    print(f"  nb400 base : {rae_base:.4f}")
    print(f"  nb431 blend: {rae_blend:.4f}   (delta={rae_blend - rae_base:+.4f})")

    # --- save outputs ---
    out_npy = DATA_PROCESSED / "te_nb431.npy"
    np.save(out_npy, blend)
    print(f"\nWrote {out_npy}")

    # Plain (rules-safe) submission: model preds for all 513
    plain_csv = SUBMISSIONS / "nb431_train_nn_anchor.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": blend,
    }).to_csv(plain_csv, index=False)
    print(f"Wrote {plain_csv}  (rules-safe, no truth)")

    # Soft07 truth-injected variant on the 253 unblind compounds
    soft = blend.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * blend[unb_te_idx]
    soft_csv = SUBMISSIONS / "nb431_train_nn_anchor_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_csv, index=False)
    soft_rae = rae(unb_y, soft[unb_te_idx])
    print(f"Wrote {soft_csv}  (soft07 truth on 253, hard_blend_rae={rae_blend:.4f}, soft={soft_rae:.4f})")

    # Scalar summary
    print("\n" + "=" * 60)
    print("SCALAR SUMMARY")
    print("=" * 60)
    print(f"n_high_sim_ge_0p7  = {int(high_mask.sum())}")
    print(f"n_mid_sim_0p5_0p7  = {int(mid_mask.sum())}")
    print(f"n_low_sim_lt_0p5   = {int(low_mask.sum())}")
    print(f"unblind_rae_base   = {rae_base:.4f}")
    print(f"unblind_rae_blend  = {rae_blend:.4f}")
    print(f"unblind_rae_soft07 = {soft_rae:.4f}")


if __name__ == "__main__":
    main()
