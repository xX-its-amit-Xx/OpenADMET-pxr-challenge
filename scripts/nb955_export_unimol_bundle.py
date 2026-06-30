"""nb955 — export the data bundle the Uni-Mol Kaggle notebook needs.

Three CSVs aligned to the SAME scaffold folds (seed=42) as nb952/nb953/nb954,
so Uni-Mol's scaffold-CV OOF on the 4139 plugs straight into the degradation-curve
comparison and its 253/513 predictions are directly gradeable.

  unimol_train.csv   (4139): name, smiles, pec50, scaffold_fold   [fine-tune + OOF]
  unimol_eval253.csv  (253): pos513, smiles, y                    [honest eval]
  unimol_test513.csv  (513): name, smiles                         [deploy]

These get synced to Kaggle via `python scripts/kaggle_push.py --data` so the
Uni-Mol kernel mounts them (no in-kernel HF download of labels needed).
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

D = "data/processed"


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    smiles = tr["smiles"].tolist()
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None
                 for s in smiles]
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)

    fold_of = np.full(len(tr), -1, int)
    for k, (_, va) in enumerate(folds):
        fold_of[va] = k
    assert (fold_of >= 0).all(), "some train row unassigned to a fold"

    train_df = pd.DataFrame({
        "name": tr["name"], "smiles": tr["smiles"],
        "pec50": tr["pec50"].astype(float), "scaffold_fold": fold_of,
    })
    # carry nb952's per-compound max-sim-to-fold-train so the Kaggle notebook can
    # build the degradation curve in-mount (same seed-42 folds -> aligns 1:1)
    msim_path = f"{D}/nb952_max_sim_4139.npy"
    if os.path.exists(msim_path):
        msim = np.load(msim_path)
        if len(msim) == len(train_df):
            train_df["max_sim"] = msim
    train_df.to_csv(f"{D}/unimol_train.csv", index=False)

    te = load_test()
    unb_idx = np.load(f"{D}/_audit_unblind_idx.npy")
    y = np.load(f"{D}/_audit_unblind_y.npy")
    eval_df = pd.DataFrame({
        "pos513": unb_idx, "smiles": te["smiles"].to_numpy()[unb_idx], "y": y,
    })
    eval_df.to_csv(f"{D}/unimol_eval253.csv", index=False)

    test_df = pd.DataFrame({"name": te["name"] if "name" in te else te.index,
                            "smiles": te["smiles"]})
    test_df.to_csv(f"{D}/unimol_test513.csv", index=False)

    # ALSO write parquet — push_data() syncs data/processed/*.parquet to Kaggle
    # (it does NOT sync *.csv from processed), so the Uni-Mol kernel mounts these.
    train_df.to_parquet(f"{D}/unimol_train.parquet", index=False)
    eval_df.to_parquet(f"{D}/unimol_eval253.parquet", index=False)
    test_df.to_parquet(f"{D}/unimol_test513.parquet", index=False)

    print("exported Uni-Mol bundle (csv + parquet for Kaggle sync):")
    print(f"  unimol_train.csv   {train_df.shape}  fold sizes={np.bincount(fold_of).tolist()}")
    print(f"  unimol_eval253.csv {eval_df.shape}   y mean={y.mean():.3f} std={y.std():.3f}")
    print(f"  unimol_test513.csv {test_df.shape}")
    # sanity: eval253 SMILES must be a subset of test513 at the right positions
    same = (te['smiles'].to_numpy()[unb_idx] == eval_df['smiles'].to_numpy()).all()
    print(f"  eval253 SMILES align with test513[pos513]: {same}")


if __name__ == "__main__":
    main()
