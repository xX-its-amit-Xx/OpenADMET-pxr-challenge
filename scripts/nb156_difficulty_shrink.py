"""nb156 -- Difficulty-aware shrinkage blend.

For each compound, compute a 'difficulty' score (low neighbor similarity +
high neighbor disagreement). Shrink predictions toward neighbor mean when
difficulty is high. Validates via leave-one-out kNN on train (OOF).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def morgan_bits(smis, r=2, nb=2048):
    fps = []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(None); continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=nb))
    return fps


def main():
    print("=== nb156: Difficulty-aware shrinkage ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    oof_base = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    te_base  = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")
    print(f"Base OOF RAE = {rae(y_tr, oof_base):.4f}")

    print("Computing Morgan FPs...")
    fps_tr = morgan_bits(smiles_tr)
    fps_te = morgan_bits(smiles_te)
    print("  fps ready")

    # For each train compound, find K=5 nearest train neighbors (leave-self-out)
    K = 5
    nb_mean_tr = np.full(len(y_tr), np.nan)
    nb_spread_tr = np.full(len(y_tr), np.nan)
    top1_sim_tr = np.full(len(y_tr), np.nan)
    print("Leave-one-out kNN on train (slow but precise)...")
    for i in range(len(y_tr)):
        if fps_tr[i] is None: continue
        sims = []
        for j in range(len(y_tr)):
            if i == j or fps_tr[j] is None: continue
            sims.append((DataStructs.TanimotoSimilarity(fps_tr[i], fps_tr[j]), j))
        sims.sort(reverse=True)
        topk = sims[:K]
        top1_sim_tr[i] = topk[0][0]
        labels = [y_tr[j] for _, j in topk]
        nb_mean_tr[i]  = np.mean(labels)
        nb_spread_tr[i]= np.max(labels) - np.min(labels)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(y_tr)}")

    # Difficulty proxy on train: inverse similarity weighted spread
    difficulty_tr = (1 - top1_sim_tr) * (1 + nb_spread_tr)
    print(f"Train difficulty: median={np.nanmedian(difficulty_tr):.3f}")

    # Test difficulty (from existing parquet)
    d_te = pd.read_parquet(DATA_PROCESSED / "test_difficulty.parquet")
    # Reorder to match smiles_te order via name lookup
    name_to_diff = dict(zip(d_te["name"], d_te["difficulty"]))
    name_to_nbm  = dict(zip(d_te["name"], d_te["nb_pec50_mean"]))
    te_names = te_df["name"].tolist() if "name" in te_df.columns else te_df["Molecule Name"].tolist()
    difficulty_te = np.array([name_to_diff.get(n, np.nan) for n in te_names])
    nbmean_te = np.array([name_to_nbm.get(n, np.nan) for n in te_names])

    # Sweep shrinkage lambdas; pick best OOF
    print("\nSweeping shrinkage lambdas...")
    print(f"{'lambda':>10} {'OOF RAE':>10}")
    best_lam, best_rae = 0.0, rae(y_tr, oof_base)
    print(f"{0.0:>10.3f} {best_rae:>10.4f}  (baseline)")
    for lam in [0.02, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30]:
        # Shrink toward nb_mean_tr proportional to difficulty (normalized 0-1)
        w = lam * (difficulty_tr - np.nanmin(difficulty_tr)) / (np.nanmax(difficulty_tr) - np.nanmin(difficulty_tr))
        w = np.nan_to_num(w, nan=0.0)
        pred = (1 - w) * oof_base + w * np.nan_to_num(nb_mean_tr, nan=np.nanmedian(y_tr))
        r = rae(y_tr, pred)
        flag = "  <-- BEST" if r < best_rae else ""
        if r < best_rae:
            best_rae, best_lam = r, lam
        print(f"{lam:>10.3f} {r:>10.4f}{flag}")

    print(f"\nBest lambda: {best_lam}  OOF RAE: {best_rae:.4f}")
    print(f"Delta from baseline: {best_rae - rae(y_tr, oof_base):+.4f}")

    if best_lam > 0:
        # Apply to test
        w_te = best_lam * (difficulty_te - np.nanmin(difficulty_te)) / (np.nanmax(difficulty_te) - np.nanmin(difficulty_te))
        w_te = np.nan_to_num(w_te, nan=0.0)
        te_pred = (1 - w_te) * te_base + w_te * np.nan_to_num(nbmean_te, nan=np.nanmedian(y_tr))
        np.save(DATA_PROCESSED / "te_nb156_diffshrink.npy", te_pred)
        np.save(DATA_PROCESSED / "oof_nb156_diffshrink.npy", oof_base)  # placeholder
        # Save submission
        sub = pd.DataFrame({"Molecule Name": te_names, "pEC50": te_pred})
        sub.to_csv(SUBMISSIONS / "224_diffshrink.csv", index=False)
        print(f"\nSaved 224_diffshrink.csv  te_std={te_pred.std():.4f}")
    else:
        print("\nNo improvement from shrinkage. Not writing submission.")


if __name__ == "__main__":
    main()
