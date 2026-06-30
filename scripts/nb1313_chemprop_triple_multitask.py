"""nb1313 -- Fresh chemprop training with TRIPLE multitask.

Targets: PXR pEC50 + counter pEC50 (pec50_null) + single_conc median_log2_fc
(aggregated to per-compound median over the 2,740 train compounds with SP).

Architecture: chemprop 2.x MPNN
  - BondMessagePassing, depth=3, message-hidden-dim=300
  - FFN: 2 layers, dropout 0.1, 3-head regression
  - NaN-masked loss on absent labels

Procedure:
  1. Build SMILES -> {pxr, null, sp_med} CSV; outer-join the three sources on
     standardized SMILES.
  2. Scaffold-aware 5-fold split on the PXR-labeled rows (the 4,139 train).
     Each fold's training data includes all rows with ANY non-NaN label and
     non-validation PXR rows.
  3. Per fold: train chemprop CLI (CPU). Predict on val + 513 test.
  4. OOF PXR pred (4,139). Average per-fold test preds -> te_nb1313.
  5. Extract nb1313_pred_oof.npy = OOF[unb_idx] (the 253 unblind slice).
  6. Compute RAE on 253 unblind, and in_RAE on 513 unblind slice via te[unb_idx]
     vs unblind truth. Verdict: anchor candidate if 0.55 < OOF_RAE < 0.5771.

CAVEAT: 5 folds CPU. Use --epochs 12 with --patience 4 early stopping so each
fold should finish in ~3-5 min. Hard timeout 12 min per fold.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from pxr.chem import add_standard_columns
from pxr.data import load_counter, load_single_conc, load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1313"
TARGETS = ["pxr", "null", "sp_med"]
N_FOLDS = 5
EPOCHS = 12
PATIENCE = 4
PER_FOLD_TIMEOUT_S = 12 * 60  # 12 minutes hard cap


def murcko(smi: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return ""


def build_triple_csv(out_path: Path):
    """Build chemprop-ready CSV with smiles + 3 target columns."""
    tr = load_train()
    tr = add_standard_columns(tr)
    tr_main = tr[["std_smiles", "pec50"]].rename(columns={"pec50": "pxr"})

    co = load_counter()
    co = add_standard_columns(co)
    co_main = co[["std_smiles", "pec50"]].rename(columns={"pec50": "null"})

    sp = load_single_conc()
    sp = add_standard_columns(sp)
    sp_agg = (
        sp.groupby("std_smiles")["median_log2_fc"].median().reset_index().rename(
            columns={"median_log2_fc": "sp_med"}
        )
    )

    full = tr_main.set_index("std_smiles")
    full = full.join(co_main.set_index("std_smiles"), how="outer")
    full = full.join(sp_agg.set_index("std_smiles"), how="outer")
    full = full.reset_index()

    for t in TARGETS:
        if t not in full.columns:
            full[t] = np.nan
    label_mask = full[TARGETS].notna().any(axis=1)
    full = full[label_mask].reset_index(drop=True)
    full = full.rename(columns={"std_smiles": "smiles"})
    full = full[["smiles"] + TARGETS]
    full.to_csv(out_path, index=False)
    nn_per = full[TARGETS].notna().sum().to_dict()
    print(
        f"  triple csv: {len(full)} rows | per-target non-NaN: {nn_per}",
        flush=True,
    )
    return full


def run_chemprop(cmd: list[str], timeout: int) -> tuple[int, str]:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stderr[-2000:] if r.returncode != 0 else "")
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {time.time()-t0:.0f}s"


def main():
    t_start = time.time()
    print(f"=== {TAG}: chemprop triple multitask ===", flush=True)

    work_dir = DATA_PROCESSED / f"{TAG}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    full_csv = work_dir / "triple_full.csv"
    df = build_triple_csv(full_csv)

    has_pxr = df["pxr"].notna()
    pxr_idx = np.where(has_pxr.values)[0]
    print(f"  PXR-labeled rows: {len(pxr_idx)}", flush=True)

    df["scaffold"] = df["smiles"].apply(murcko)
    pxr_scaffs = df.iloc[pxr_idx]["scaffold"].tolist()
    folds = scaffold_kfold_indices(pxr_scaffs, n_splits=N_FOLDS)

    te = load_test()
    te = add_standard_columns(te)
    test_smiles = te["std_smiles"].tolist()
    test_csv = work_dir / "test.csv"
    pd.DataFrame({"smiles": test_smiles}).to_csv(test_csv, index=False)

    chemprop_bin = (
        Path(sys.executable).parent / ("chemprop.exe" if os.name == "nt" else "chemprop")
    )

    oof_pxr = np.full(len(df), np.nan)
    test_preds: list[np.ndarray] = []
    fold_times: list[float] = []
    fold_status: list[str] = []

    for fold_i, (ti_rel, vi_rel) in enumerate(folds):
        t_fold = time.time()
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---", flush=True)
        ti = pxr_idx[ti_rel]
        vi = pxr_idx[vi_rel]
        # Add non-PXR rows (counter-only or SP-only) to training
        non_pxr_idx = np.where(~has_pxr.values)[0]
        ti_full = np.concatenate([ti, non_pxr_idx])

        fold_dir = work_dir / f"fold_{fold_i}"
        fold_dir.mkdir(exist_ok=True)
        df.iloc[ti_full][["smiles"] + TARGETS].to_csv(
            fold_dir / "train.csv", index=False
        )
        df.iloc[vi][["smiles"] + TARGETS].to_csv(fold_dir / "val.csv", index=False)

        model_dir = fold_dir / "model"

        train_cmd = [
            str(chemprop_bin), "train",
            "--data-path", str(fold_dir / "train.csv"),
            "--task-type", "regression",
            "--save-dir", str(model_dir),
            "--smiles-columns", "smiles",
            "--target-columns", *TARGETS,
            "--epochs", str(EPOCHS),
            "--patience", str(PATIENCE),
            "--batch-size", "64",
            "--num-workers", "0",
            "--accelerator", "cpu",
            "--depth", "3",
            "--message-hidden-dim", "300",
            "--ffn-num-layers", "2",
            "--ffn-hidden-dim", "300",
            "--dropout", "0.1",
        ]
        rc, err = run_chemprop(train_cmd, PER_FOLD_TIMEOUT_S)
        train_t = time.time() - t_fold
        print(f"  train rc={rc} elapsed={train_t:.0f}s", flush=True)
        if rc != 0:
            print(f"  TRAIN FAIL: {err[-600:]}", flush=True)
            fold_status.append(f"train_fail_rc{rc}")
            fold_times.append(train_t)
            continue

        # Predict on val
        pred_val_path = fold_dir / "pred_val.csv"
        cmd_pred_val = [
            str(chemprop_bin), "predict",
            "--test-path", str(fold_dir / "val.csv"),
            "--model-path", str(model_dir),
            "--preds-path", str(pred_val_path),
            "--smiles-columns", "smiles",
            "--num-workers", "0",
            "--accelerator", "cpu",
        ]
        rc, err = run_chemprop(cmd_pred_val, 600)
        if rc == 0 and pred_val_path.exists():
            pv = pd.read_csv(pred_val_path)
            if "pxr" in pv.columns:
                oof_pxr[vi] = pv["pxr"].values
                print(f"  val pred OK n={len(pv)} cols={pv.columns.tolist()}", flush=True)
            else:
                print(f"  val pred missing pxr col: {pv.columns.tolist()}", flush=True)
        else:
            print(f"  val predict failed rc={rc}: {err[-500:]}", flush=True)

        # Predict on test
        pred_te_path = fold_dir / "pred_test.csv"
        cmd_pred_te = [
            str(chemprop_bin), "predict",
            "--test-path", str(test_csv),
            "--model-path", str(model_dir),
            "--preds-path", str(pred_te_path),
            "--smiles-columns", "smiles",
            "--num-workers", "0",
            "--accelerator", "cpu",
        ]
        rc, err = run_chemprop(cmd_pred_te, 600)
        if rc == 0 and pred_te_path.exists():
            pt = pd.read_csv(pred_te_path)
            if "pxr" in pt.columns:
                test_preds.append(pt["pxr"].values)
                print(
                    f"  test pred OK mean={pt['pxr'].mean():.3f} std={pt['pxr'].std():.3f}",
                    flush=True,
                )
        else:
            print(f"  test predict failed rc={rc}: {err[-500:]}", flush=True)

        fold_time = time.time() - t_fold
        fold_times.append(fold_time)
        fold_status.append("ok")
        print(f"  fold {fold_i} total wall: {fold_time:.0f}s", flush=True)

    # Evaluate
    valid = ~np.isnan(oof_pxr) & ~np.isnan(df["pxr"].values)
    if valid.sum() > 100:
        y_pxr = df["pxr"].values
        oof_rae_full = float(rae(y_pxr[valid], oof_pxr[valid]))
        print(
            f"\nOOF RAE on {int(valid.sum())} PXR train rows: {oof_rae_full:.4f}",
            flush=True,
        )
    else:
        oof_rae_full = float("nan")
        print(f"\nWARNING: insufficient OOF predictions ({int(valid.sum())})", flush=True)

    # Map OOF on full chemprop df back to canonical train order
    tr_canon = load_train()
    tr_canon = add_standard_columns(tr_canon)
    smi_to_oof = dict(zip(df["smiles"], oof_pxr))
    oof_train = np.array(
        [smi_to_oof.get(s, np.nan) for s in tr_canon["std_smiles"]]
    )
    np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof_train)
    print(
        f"  oof_train aligned: {(~np.isnan(oof_train)).sum()}/{len(oof_train)} valid",
        flush=True,
    )

    # 253-unblind slice
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    unb_y = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")

    if test_preds:
        # Mean across fold models
        te_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    else:
        te_pred = np.full(len(test_smiles), np.nan)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred)
    print(
        f"  te_{TAG}.npy: mean={te_pred.mean():.3f} std={te_pred.std():.3f}",
        flush=True,
    )

    # OOF -> 253 unblind pred_oof. We rely on the train compounds; the 253
    # unblind are TEST compounds (not in train). So nb1313_pred_oof.npy is
    # te_pred[unb_idx] -- this is "deploy" prediction on 253 (in-sample slice
    # of the 513 test prediction). NOT a true cross-fit unblind OOF.
    pred_oof_253 = te_pred[unb_idx]
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof_253)

    # RAE on 253 unblind from deploy te (= in_RAE for 253)
    if np.isfinite(pred_oof_253).all():
        in_rae_253 = float(rae(unb_y, pred_oof_253))
    else:
        in_rae_253 = float("nan")
    print(f"  in_RAE on 253 unblind (te[unb_idx]): {in_rae_253:.4f}", flush=True)

    # Compare with nb1070 anchor
    nb1070_anchor_rae = 0.5771  # bag_median_rae from nb1070 summary
    delta_vs_nb1070 = (
        in_rae_253 - nb1070_anchor_rae if np.isfinite(in_rae_253) else float("nan")
    )
    is_new_anchor_candidate = (
        np.isfinite(in_rae_253)
        and (in_rae_253 < nb1070_anchor_rae)
        and (in_rae_253 > 0.30)  # sanity: leak-suspicion floor
    )

    wall_total = time.time() - t_start
    summary = {
        "tag": TAG,
        "targets": TARGETS,
        "n_folds": N_FOLDS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "per_fold_timeout_s": PER_FOLD_TIMEOUT_S,
        "fold_status": fold_status,
        "fold_wall_sec": fold_times,
        "fold_wall_sec_mean": float(np.mean(fold_times)) if fold_times else None,
        "n_oof_valid_train": int((~np.isnan(oof_train)).sum()),
        "oof_rae_full_pxr": oof_rae_full,
        "in_rae_253_unblind": in_rae_253,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "nb1070_anchor_rae": nb1070_anchor_rae,
        "delta_vs_nb1070": delta_vs_nb1070,
        "is_new_anchor_candidate": bool(is_new_anchor_candidate),
        "wall_sec_total": wall_total,
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}  wall={wall_total:.0f}s", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
