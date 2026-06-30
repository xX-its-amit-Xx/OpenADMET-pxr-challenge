"""
Merge MACE-OFF23 strain feature chunks and copy to C:/pxr_work.
Usage: python scripts/merge_mace_strain.py
"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

MACE_DIR = Path("C:/pxr_work/mace_strain")
OUT_FILE = MACE_DIR / "mace_strain_features.csv"

# Try Explorer path if local doesn't exist
if not MACE_DIR.exists():
    MACE_DIR = Path("/scratch/shenoy.am/pxr_work/mace_strain")

def main():
    chunks = sorted(MACE_DIR.glob("chunk_*.csv"))
    if not chunks:
        print(f"No chunk files found in {MACE_DIR}")
        sys.exit(1)

    print(f"Found {len(chunks)} chunk files")
    dfs = []
    for c in chunks:
        df = pd.read_csv(c)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset="smiles")

    # Stats
    n_total = len(merged)
    n_valid = merged["mace_strain"].notna().sum()
    print(f"Total: {n_total} rows, {n_valid} valid ({n_total-n_valid} errors)")
    print(merged[["mace_strain", "mace_e_pose", "mace_e_globalmin"]].describe())

    merged.to_csv(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")

    # Also check alignment with our train+test
    try:
        from src.pxr.data import load_train, load_test
        from src.pxr.chem import standardize
        train = load_train()
        test = load_test()
        all_smiles = pd.concat([train[["smiles"]], test[["smiles"]]]).drop_duplicates()
        merged_aligned = all_smiles.merge(merged, on="smiles", how="left")
        frac = merged_aligned["mace_strain"].notna().mean()
        print(f"Coverage of train+test SMILES: {frac:.1%} ({merged_aligned['mace_strain'].notna().sum()}/{len(merged_aligned)})")
    except Exception as e:
        print(f"Could not check alignment: {e}")


if __name__ == "__main__":
    main()
