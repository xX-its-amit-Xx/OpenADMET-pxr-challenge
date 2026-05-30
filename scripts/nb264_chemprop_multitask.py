"""nb264 -- Chemprop multi-task on PXR + counter + 6 Papyrus NR targets.

Trains a single MPNN GNN with 8 task heads jointly. Shared message passing
learns cross-target structural patterns. Slower than LGBM but a real
biology-informed model.

Uses chemprop 2.x Python API. Runs ~30-60 min on CPU for full dataset.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED

TARGETS = ["pxr", "null", "pxr_pap", "fxr", "pparg", "lxra", "rxra", "vdr"]


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def build_multi_csv(out_path="data/processed/chemprop_multitask.csv"):
    """Build chemprop-ready CSV with SMILES + 8 target columns."""
    tr = load_train(); tr = add_standard_columns(tr)
    tr_main = tr[["std_smiles", "pec50"]].rename(columns={"pec50": "pxr"})

    co = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    co_clean = co[["SMILES", "pEC50"]].rename(columns={"SMILES": "std_smiles", "pEC50": "null"})
    co_clean["std_smiles"] = co_clean["std_smiles"].apply(std_smi)
    co_clean = co_clean.dropna(subset=["std_smiles", "null"])

    papyrus = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    papyrus_pivot = papyrus[papyrus["target_name"].isin(["PXR", "FXR", "PPARg", "LXRa", "RXRa", "VDR"])].copy()
    papyrus_pivot["std_smiles"] = papyrus_pivot["std_smiles"].apply(std_smi)
    papyrus_pivot = papyrus_pivot.dropna(subset=["std_smiles", "pec50"])
    rename_map = {"PXR": "pxr_pap", "FXR": "fxr", "PPARg": "pparg", "LXRa": "lxra", "RXRa": "rxra", "VDR": "vdr"}
    papyrus_pivot["target_name"] = papyrus_pivot["target_name"].map(rename_map)
    papyrus_wide = papyrus_pivot.groupby(["std_smiles", "target_name"])["pec50"].median().unstack()

    full = tr_main.set_index("std_smiles")
    full = full.join(co_clean.set_index("std_smiles"), how="outer")
    full = full.join(papyrus_wide, how="outer")
    full = full.reset_index()

    # Ensure all target cols exist
    for t in TARGETS:
        if t not in full.columns:
            full[t] = np.nan
    label_mask = full[TARGETS].notna().any(axis=1)
    full = full[label_mask].reset_index(drop=True)
    full = full.rename(columns={"std_smiles": "smiles"})
    cols_order = ["smiles"] + TARGETS
    full = full[cols_order]
    full.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(full)} rows, {full[TARGETS].notna().sum().sum()} non-NaN labels")
    return full


def main():
    print("=== nb264: Chemprop multi-task ===\n")
    df = build_multi_csv()

    # Identify PXR-labeled rows (for OOF evaluation)
    has_pxr = df["pxr"].notna()
    pxr_idx = np.where(has_pxr.values)[0]
    print(f"PXR-labeled rows: {len(pxr_idx)}")

    # Build scaffolds for CV
    from rdkit.Chem.Scaffolds import MurckoScaffold
    def scaff(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) if mol else ""
        except: return ""
    df["scaffold"] = df["smiles"].apply(scaff)
    pxr_scaffs = df.iloc[pxr_idx]["scaffold"].tolist()
    folds = scaffold_kfold_indices(pxr_scaffs, n_splits=5)

    # Test data
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    te_df["smiles"] = te_df["SMILES"].apply(std_smi)
    te_df = te_df.dropna(subset=["smiles"])
    test_smiles = te_df["smiles"].tolist()
    te_csv_path = "data/processed/chemprop_test.csv"
    pd.DataFrame({"smiles": test_smiles}).to_csv(te_csv_path, index=False)

    # Use chemprop CLI for training - more battle-tested than python API for multi-task
    # Save each fold's train+val to separate CSV, run chemprop train, then predict
    import subprocess
    n_folds = 5
    oof_pxr = np.full(len(df), np.nan)
    test_preds = []

    for fold_i, (ti_rel, vi_rel) in enumerate(folds):
        print(f"\n--- Fold {fold_i+1}/{n_folds} ---")
        ti = pxr_idx[ti_rel]
        vi = pxr_idx[vi_rel]
        # Add ALL non-PXR rows to training
        non_pxr_idx = np.where(~has_pxr.values)[0]
        ti_full = np.concatenate([ti, non_pxr_idx])

        train_df = df.iloc[ti_full][["smiles"] + TARGETS]
        val_df = df.iloc[vi][["smiles"] + TARGETS]
        fold_dir = Path(f"data/processed/chemprop_fold_{fold_i}")
        fold_dir.mkdir(exist_ok=True)
        train_df.to_csv(fold_dir / "train.csv", index=False)
        val_df.to_csv(fold_dir / "val.csv", index=False)

        # Train
        cmd = [
            "chemprop", "train",
            "--data-path", str(fold_dir / "train.csv"),
            "--task-type", "regression",
            "--save-dir", str(fold_dir / "model"),
            "--smiles-columns", "smiles",
            "--target-columns"] + TARGETS + [
            "--epochs", "20",  # quick for testing
            "--batch-size", "64",
            "--num-workers", "0",
            "--accelerator", "cpu",
        ]
        print(f"  Train cmd: chemprop train ...")
        import time; t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(f"  rc={r.returncode}, elapsed={time.time()-t0:.0f}s")
        if r.returncode != 0:
            print(f"  STDERR: {r.stderr[-1500:]}")
            continue

        # Predict on val
        pred_val_path = fold_dir / "pred_val.csv"
        cmd_pred = [
            "chemprop", "predict",
            "--test-path", str(fold_dir / "val.csv"),
            "--model-path", str(fold_dir / "model"),
            "--preds-path", str(pred_val_path),
            "--smiles-columns", "smiles",
            "--num-workers", "0",
            "--accelerator", "cpu",
        ]
        r = subprocess.run(cmd_pred, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and pred_val_path.exists():
            pv = pd.read_csv(pred_val_path)
            print(f"  val pred cols: {pv.columns.tolist()}")
            if "pxr" in pv.columns:
                oof_pxr[vi] = pv["pxr"].values

        # Predict on test
        pred_te_path = fold_dir / "pred_test.csv"
        cmd_pred_te = [
            "chemprop", "predict",
            "--test-path", te_csv_path,
            "--model-path", str(fold_dir / "model"),
            "--preds-path", str(pred_te_path),
            "--smiles-columns", "smiles",
            "--num-workers", "0",
            "--accelerator", "cpu",
        ]
        r = subprocess.run(cmd_pred_te, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and pred_te_path.exists():
            pt = pd.read_csv(pred_te_path)
            if "pxr" in pt.columns:
                test_preds.append(pt["pxr"].values)

    # Compute OOF RAE
    valid = ~np.isnan(oof_pxr) & ~np.isnan(df["pxr"].values)
    if valid.sum() > 100:
        y_pxr = df["pxr"].values
        r_oof = rae(y_pxr[valid], oof_pxr[valid])
        print(f"\n=== Multi-task chemprop OOF RAE: {r_oof:.4f} (vs nb224 0.2891) ===")
    else:
        print("\nWARNING: insufficient OOF predictions")

    # Average test preds
    if test_preds:
        te_pred = np.mean(test_preds, axis=0)
        print(f"Test predictions: mean={te_pred.mean():.3f}, std={te_pred.std():.3f}")
        np.save(DATA_PROCESSED / "te_nb264_chemprop_mt.npy", te_pred)

    # Save OOF aligned to original train order
    tr = load_train(); tr_std = add_standard_columns(tr)
    smi_to_oof = dict(zip(df["smiles"], oof_pxr))
    oof_aligned = np.array([smi_to_oof.get(s, np.nan) for s in tr_std["std_smiles"]])
    np.save(DATA_PROCESSED / "oof_nb264_chemprop_mt.npy", oof_aligned)
    print(f"Saved oof_nb264_chemprop_mt.npy ({(~np.isnan(oof_aligned)).sum()}/{len(oof_aligned)} valid)")


if __name__ == "__main__":
    main()
