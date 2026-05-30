"""nb259 -- Multi-task neural network: PXR + counter-assay + 7 NR targets jointly.

Channeling Kaggle MoA 2020 winning pattern: use auxiliary related targets to
force the model to learn richer shared representations.

Architecture:
- Input: combined features (~2265 dim Morgan + RDKit)
- Shared trunk: 2048 -> 1024 -> 512 (with batchnorm + dropout)
- 9 task-specific heads:
  - pec50_pxr (PXR — our main target)
  - pec50_null (counter-assay)
  - pec50_pxr_papyrus
  - pec50_fxr
  - pec50_pparg
  - pec50_lxra
  - pec50_rxra
  - pec50_vdr
  - pec50_car (CHEMBL5071)

Loss: per-head MAE, weighted by inverse-frequency (PXR gets weight 5, others 1).
For each compound: gradient flows ONLY through heads with labels.

This forces the shared trunk to learn a representation that supports ALL these
related-but-different binding tasks simultaneously. The PXR head benefits from
patterns learned across the NR superfamily.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


TARGETS = ["pxr", "null", "pxr_pap", "fxr", "pparg", "lxra", "rxra", "vdr"]


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None


def build_multi_dataset():
    """Build a unified DataFrame: std_smiles + 8 targets (pec50_X)."""
    tr = load_train(); tr = add_standard_columns(tr)
    tr_main = tr[["std_smiles", "pec50"]].rename(columns={"pec50": "pxr"})
    print(f"Train PXR: {len(tr_main)}")

    # Counter-assay
    co = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    co_clean = co[["SMILES", "pEC50"]].rename(columns={"SMILES": "std_smiles", "pEC50": "null"})
    co_clean["std_smiles"] = co_clean["std_smiles"].apply(std_smi)
    co_clean = co_clean.dropna(subset=["std_smiles", "null"])
    print(f"Counter assay: {len(co_clean)}")

    # Papyrus NR
    papyrus = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    papyrus_pivot = papyrus[papyrus["target_name"].isin(["PXR", "FXR", "PPARg", "LXRa", "RXRa", "VDR"])].copy()
    papyrus_pivot["std_smiles"] = papyrus_pivot["std_smiles"].apply(std_smi)
    papyrus_pivot = papyrus_pivot.dropna(subset=["std_smiles", "pec50"])
    # Pivot: one row per smiles, columns per target
    rename_map = {"PXR": "pxr_pap", "FXR": "fxr", "PPARg": "pparg", "LXRa": "lxra", "RXRa": "rxra", "VDR": "vdr"}
    papyrus_pivot["target_name"] = papyrus_pivot["target_name"].map(rename_map)
    papyrus_wide = papyrus_pivot.groupby(["std_smiles", "target_name"])["pec50"].median().unstack()
    print(f"Papyrus pivot: {len(papyrus_wide)} compounds, columns: {papyrus_wide.columns.tolist()}")

    # Merge
    full = tr_main.set_index("std_smiles")
    full = full.join(co_clean.set_index("std_smiles"), how="outer")
    full = full.join(papyrus_wide, how="outer")
    full = full.reset_index()
    print(f"Merged: {len(full)} unique compounds, {full.notna().sum(axis=1).mean():.2f} labels per compound")

    # Filter: must have at least 1 label
    label_cols = [c for c in TARGETS if c in full.columns]
    label_mask = full[label_cols].notna().any(axis=1)
    full = full[label_mask].reset_index(drop=True)
    print(f"After label filter: {len(full)}")

    return full, label_cols


class MultiTaskNN(nn.Module):
    def __init__(self, input_dim, n_tasks, hidden=(2048, 1024, 512), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(n_tasks)])

    def forward(self, x):
        z = self.trunk(x)
        outs = torch.cat([h(z) for h in self.heads], dim=1)  # (B, n_tasks)
        return outs


def masked_mae_loss(pred, y, mask, task_weights=None):
    """Mean absolute error with NaN masking. task_weights: per-task weight."""
    err = torch.abs(pred - y) * mask  # zero where no label
    if task_weights is not None:
        err = err * task_weights.unsqueeze(0)
    n_obs = mask.sum().clamp(min=1)
    return err.sum() / n_obs


def train_one_fold(X_tr, Y_tr, mask_tr, X_va, Y_va, mask_va, n_tasks, task_weights, device, epochs=40):
    model = MultiTaskNN(X_tr.shape[1], n_tasks).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32).to(device)
    M_tr_t = torch.tensor(mask_tr, dtype=torch.float32).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    Y_va_t = torch.tensor(Y_va, dtype=torch.float32).to(device)
    M_va_t = torch.tensor(mask_va, dtype=torch.float32).to(device)
    w_t = torch.tensor(task_weights, dtype=torch.float32).to(device)
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    batch_size = 256
    n = X_tr.shape[0]
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        train_loss = 0
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            opt.zero_grad()
            pred = model(X_tr_t[b])
            loss = masked_mae_loss(pred, Y_tr_t[b], M_tr_t[b], w_t)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(b)
        train_loss /= n
        model.eval()
        with torch.no_grad():
            pred_va = model(X_va_t)
            val_loss = masked_mae_loss(pred_va, Y_va_t, M_va_t, w_t).item()
        sched.step()
        if not np.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    model.load_state_dict(best_state)
    return model


def main():
    print("=== nb259: Multi-task NN ===\n")
    full, label_cols = build_multi_dataset()
    print(f"\nLabel columns: {label_cols}")
    print(f"Per-task label counts:")
    for c in label_cols:
        print(f"  {c}: {full[c].notna().sum()}")

    # Featurize
    print("\nFeaturizing...")
    smiles_list = full["std_smiles"].tolist()
    X = combined(smiles_list); X = impute(X).astype(np.float32)
    print(f"X: {X.shape}")

    # Labels matrix + mask
    Y = full[label_cols].values
    mask = ~np.isnan(Y)
    Y_filled = np.where(mask, Y, 0.0).astype(np.float32)
    mask_f = mask.astype(np.float32)
    print(f"Y: {Y.shape}, label density: {mask.mean():.3f}")

    # Task weights: PXR most important
    task_weights = np.ones(len(label_cols))
    task_weights[0] = 5.0  # PXR (label_cols[0] = 'pxr')
    print(f"Task weights: {dict(zip(label_cols, task_weights))}")

    # 5-fold scaffold CV on PXR-having compounds (others used for training only)
    tr_csv = load_train()
    tr_csv_std = add_standard_columns(tr_csv)
    pxr_smiles_set = set(tr_csv_std["std_smiles"])
    has_pxr = full["std_smiles"].isin(pxr_smiles_set)
    print(f"PXR-labeled compounds in merged set: {has_pxr.sum()}")

    # Standardize: each train sample is one row; we 5-fold based on scaffolds of PXR-labeled rows
    # Build scaffold for each row
    from rdkit.Chem.Scaffolds import MurckoScaffold
    def murcko(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaff)
        except: return ""
    full["scaffold"] = full["std_smiles"].apply(murcko)
    pxr_idx = np.where(has_pxr.values)[0]
    pxr_scaffs = full.iloc[pxr_idx]["scaffold"].tolist()
    folds = scaffold_kfold_indices(pxr_scaffs, n_splits=5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # OOF predictions for PXR only (head 0)
    oof_pxr = np.full(len(full), np.nan)
    test_preds = []  # for test compounds — we'll predict after CV

    # Featurize test
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    te_smiles = te_df["SMILES"].tolist()
    X_te = combined(te_smiles); X_te = impute(X_te).astype(np.float32)
    print(f"Test X: {X_te.shape}")

    print("\nTraining 5 folds...")
    t0 = time.time()
    for fold_i, (ti_rel, vi_rel) in enumerate(folds):
        # ti_rel and vi_rel are indices into pxr_idx
        ti = pxr_idx[ti_rel]; vi = pxr_idx[vi_rel]
        # Add ALL non-PXR-labeled compounds to training (their PXR mask is 0, gradient flows from other heads)
        non_pxr_idx = np.where(~has_pxr.values)[0]
        ti_full = np.concatenate([ti, non_pxr_idx])
        # Shuffle
        np.random.shuffle(ti_full)

        X_train = X[ti_full]
        Y_train = Y_filled[ti_full]
        M_train = mask_f[ti_full]
        X_val = X[vi]
        Y_val = Y_filled[vi]
        M_val = mask_f[vi]
        model = train_one_fold(X_train, Y_train, M_train, X_val, Y_val, M_val,
                                len(label_cols), task_weights, device, epochs=30)
        model.eval()
        with torch.no_grad():
            pred_vi = model(torch.tensor(X[vi], dtype=torch.float32).to(device)).cpu().numpy()
            pred_te = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
        oof_pxr[vi] = pred_vi[:, 0]  # PXR head
        test_preds.append(pred_te[:, 0])
        elapsed = time.time() - t0
        # Eval this fold
        y_va_pxr = full.iloc[vi]["pxr"].values
        valid_mask = ~np.isnan(y_va_pxr)
        if valid_mask.sum() > 0:
            r = rae(y_va_pxr[valid_mask], pred_vi[valid_mask, 0])
            print(f"  fold {fold_i+1}: val pxr RAE = {r:.4f}  elapsed={elapsed:.0f}s")

    # Final OOF (PXR-labeled only)
    pxr_y_full = full["pxr"].values
    valid = ~np.isnan(pxr_y_full)
    r_oof = rae(pxr_y_full[valid], oof_pxr[valid])
    print(f"\nFinal multi-task NN OOF (PXR): {r_oof:.4f}  (vs nb224 0.2891)")

    # Average test predictions
    te_pred = np.mean(test_preds, axis=0)
    print(f"Test predictions: mean={te_pred.mean():.3f}, std={te_pred.std():.3f}")

    # Save: align OOF to original PXR train order
    tr = load_train(); tr_std = add_standard_columns(tr)
    smi_to_oof = dict(zip(full["std_smiles"], oof_pxr))
    oof_aligned = np.array([smi_to_oof.get(s, np.nan) for s in tr_std["std_smiles"]])
    print(f"OOF aligned: {(~np.isnan(oof_aligned)).sum()}/{len(oof_aligned)}")

    np.save(DATA_PROCESSED / "oof_nb259_multitask_nn.npy", oof_aligned)
    np.save(DATA_PROCESSED / "te_nb259_multitask_nn.npy", te_pred)
    print("Saved oof/te_nb259_multitask_nn.npy")

    # Stack with 239
    print("\n=== 5-way SLSQP w/ nb259 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")

    y_full = tr["pec50"].values
    valid = ~np.isnan(oof_aligned) & np.isfinite(oof_aligned)
    if valid.sum() < len(y_full):
        print(f"WARNING: {(~valid).sum()} OOF missing; filling with mean")
        oof_aligned = np.where(valid, oof_aligned, np.nanmean(oof_aligned))

    M = np.column_stack([nb224, nb179s, mtd, loso, oof_aligned])
    def loss(w): return rae(y_full, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way SLSQP OOF: {best.fun:.4f}")
    for n, w in zip(['nb224', 'nb179s', 'mtd', 'loso', 'nb259'], best.x):
        print(f"  {n}: {w:.4f}")


if __name__ == "__main__":
    main()
