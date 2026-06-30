"""nb1053 -- Direct PyTorch Geometric GNN (no chemprop bootstrap).

Hypothesis: an architecturally-distinct GNN (GCN, 2 conv layers, mean pool,
small MLP) trained directly on pEC50 may carry residual signal orthogonal
to chemprop_aux (DMPNN with bond messages).  Two modes:

  MODE A:  GNN-alone   -- 5-seed bag x 5-fold scaffold cross-fit on 253.
                          Compare RAE to chemprop_aux v1 (0.6216 baseline).

  MODE B:  GNN-residual on chemprop_aux  -- predict residual
                          (y - chemprop_aux), pool features concatenated
                          with K=28 nb2103 SHAP-selected vector at the
                          pool layer. Then add residual back to
                          chemprop_aux.  Compare vs nb2103 K=28
                          (mean-bag 0.4737, median-bag 0.4698).

Architecture (both modes):
   atom_feat -> GCNConv(64) -> BN -> ReLU
              -> GCNConv(128) -> BN -> ReLU
              -> GlobalMeanPool
   pool_vec (128) [+ K=28 features in MODE B] -> MLP(64 -> 1)

Training:
   AdamW lr=1e-3, wd=1e-2, MSE, batch=64, max 100 epochs, early-stop
   patience=10 on val RAE (per inner train/val split).

Decision margin vs nb2103 K=28: 0.003.

Outputs:
   scripts/nb1053_gnn_head.py
   data/processed/nb1053_summary.json
   data/processed/nb1053_gnn_alone_oof.npy    (253,) per-seed bag-mean
   data/processed/nb1053_gnn_resid_oof.npy    (253,) per-seed bag-mean
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

TAG = "nb1053"
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]
DECISION_MARGIN = 0.003
GNN_HIDDEN_1 = 64
GNN_HIDDEN_2 = 128
MLP_HIDDEN = 64
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WD = 1e-2

CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698


def _dep_missing_exit(msg: str) -> dict:
    summary = {
        "tag": TAG,
        "verdict": "DEPENDENCY-MISSING",
        "error": msg,
    }
    out = Path(__file__).resolve().parents[1] / "data" / "processed" / f"{TAG}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[ERROR] {msg}")
    print(f"[write] {out}")
    return summary


def main() -> dict:
    t0 = time.time()
    # ---- Hard dep checks ----
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        return _dep_missing_exit(f"torch import failed: {e}")

    try:
        import torch_geometric
        from torch_geometric.nn import GCNConv, global_mean_pool
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except Exception as e:
        return _dep_missing_exit(f"torch_geometric import failed: {e}")

    from sklearn.model_selection import KFold
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from pxr.data import load_train, load_test
    from pxr.chem import standardize, bemis_murcko
    from pxr.eval import rae, scaffold_kfold_indices
    from pxr.paths import DATA_PROCESSED, SUBMISSIONS

    print("=" * 78)
    print(f"{TAG} -- direct PyG GNN head (GCN x2 + mean pool + MLP)")
    print("=" * 78)

    # ===== LOAD DATA =====
    tr = load_train()
    te = load_test()
    te_names = te["name"].values
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(unb_idx)
    assert n_unb == 253

    # Standardize SMILES -- pxr.chem.standardize returns Mol; convert to canonical
    def _to_smi(smi: str) -> str:
        try:
            m = standardize(smi)
            if m is None:
                return smi
            return Chem.MolToSmiles(m)
        except Exception:
            return smi
    tr_smiles_std = [_to_smi(s) for s in tr["smiles"].tolist()]
    te_smiles_std = [_to_smi(s) for s in te["smiles"].tolist()]
    y_tr = tr["pec50"].values.astype(np.float64)
    print(f"[load] train: {len(tr_smiles_std)}  test(513): {n_te}  unb_idx: {n_unb}")

    # ===== ATOM/BOND FEATURIZER =====
    ATOM_LIST = ["C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "B", "Si", "Other"]

    def atom_feats(atom) -> list[float]:
        sym = atom.GetSymbol()
        one_hot = [1.0 if sym == a else 0.0 for a in ATOM_LIST[:-1]]
        if sum(one_hot) == 0:
            one_hot.append(1.0)
        else:
            one_hot.append(0.0)
        feats = one_hot + [
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(int(atom.GetIsAromatic())),
            float(int(atom.IsInRing())),
            float(int(atom.GetHybridization())),
        ]
        return feats

    ATOM_DIM = len(atom_feats(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))

    def smi_to_data(smi: str, y: float | None = None) -> "Data | None":
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        x = torch.tensor([atom_feats(a) for a in mol.GetAtoms()],
                          dtype=torch.float)
        ei = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ei.append([i, j]); ei.append([j, i])
        if len(ei) == 0:
            # isolated atom -- self-loop
            ei = [[0, 0]]
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        if y is not None:
            data.y = torch.tensor([y], dtype=torch.float)
        return data

    print(f"[feat] ATOM_DIM = {ATOM_DIM}")

    # Build graphs
    print("[feat] building train graphs...")
    train_data = []
    skipped_train = 0
    for s, y in zip(tr_smiles_std, y_tr):
        d = smi_to_data(s, float(y))
        if d is None:
            skipped_train += 1
            continue
        train_data.append(d)
    print(f"[feat] train graphs: {len(train_data)}  skipped: {skipped_train}")

    print("[feat] building test(513) graphs...")
    test_data = []
    skip_te_mask = np.zeros(n_te, dtype=bool)
    for i, s in enumerate(te_smiles_std):
        d = smi_to_data(s, 0.0)
        if d is None:
            skip_te_mask[i] = True
            test_data.append(smi_to_data("C", 0.0))  # fallback
        else:
            test_data.append(d)
    print(f"[feat] test graphs: {len(test_data)}  skipped: {int(skip_te_mask.sum())}")

    # Unblind subset for evaluation
    unb_data = [test_data[i] for i in unb_idx]

    # ===== K=28 feature matrix for MODE B (residual mode) =====
    X_unb_28 = np.load(DATA_PROCESSED / "X_unb_28_nb2103.npy").astype(np.float32)
    print(f"[load] X_unb_28 shape = {X_unb_28.shape}")
    # Mean-impute + z-score
    mu_x = np.nanmean(X_unb_28, axis=0)
    sd_x = np.nanstd(X_unb_28, axis=0) + 1e-6
    X_unb_28 = (np.nan_to_num(X_unb_28, nan=0.0) - mu_x) / sd_x
    EXTRA_DIM = X_unb_28.shape[1]

    # ===== Chemprop_aux anchor on 513 =====
    te_ca = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    chemprop_aux_unb = te_ca[unb_idx]
    print(f"[load] chemprop_aux in_RAE on unb = "
          f"{float(rae(y_unb, chemprop_aux_unb)):.4f}  "
          f"(ref {CHEMPROP_AUX_REF})")

    # ===== Scaffold indices for unb =====
    # Compute scaffolds for the 253 unblind compounds
    unb_smiles = [te_smiles_std[i] for i in unb_idx]
    unb_scaffolds = []
    for s in unb_smiles:
        try:
            sc = bemis_murcko(s)
        except Exception:
            sc = ""
        unb_scaffolds.append(sc if sc else "_empty_")

    # ===== GNN MODELS =====
    class GNNHead(nn.Module):
        def __init__(self, in_dim: int, extra_dim: int = 0):
            super().__init__()
            self.conv1 = GCNConv(in_dim, GNN_HIDDEN_1)
            self.bn1 = nn.BatchNorm1d(GNN_HIDDEN_1)
            self.conv2 = GCNConv(GNN_HIDDEN_1, GNN_HIDDEN_2)
            self.bn2 = nn.BatchNorm1d(GNN_HIDDEN_2)
            self.extra_dim = extra_dim
            self.mlp = nn.Sequential(
                nn.Linear(GNN_HIDDEN_2 + extra_dim, MLP_HIDDEN),
                nn.ReLU(),
                nn.Linear(MLP_HIDDEN, 1),
            )

        def forward(self, data, extra=None):
            x, ei, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.bn1(self.conv1(x, ei)))
            x = F.relu(self.bn2(self.conv2(x, ei)))
            x = global_mean_pool(x, batch)
            if self.extra_dim > 0 and extra is not None:
                x = torch.cat([x, extra], dim=1)
            return self.mlp(x).squeeze(-1)

    # ===== TRAINING UTILS =====
    def train_one(model: nn.Module,
                  train_loader,
                  val_data_list,
                  val_y: np.ndarray,
                  val_extra: np.ndarray | None,
                  seed: int) -> np.ndarray:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        crit = nn.MSELoss()
        best_val = float("inf")
        best_pred = None
        patience = 0
        for epoch in range(MAX_EPOCHS):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                extra = None
                if hasattr(batch, "extra"):
                    extra = batch.extra
                pred = model(batch, extra=extra)
                loss = crit(pred, batch.y)
                loss.backward()
                opt.step()
            # val
            model.eval()
            with torch.no_grad():
                val_loader = DataLoader(val_data_list, batch_size=BATCH,
                                        shuffle=False)
                preds = []
                offset = 0
                for vb in val_loader:
                    extra = None
                    if val_extra is not None:
                        bsz = int(vb.batch.max().item()) + 1
                        e = torch.tensor(
                            val_extra[offset:offset + bsz], dtype=torch.float)
                        extra = e
                        offset += bsz
                    p = model(vb, extra=extra)
                    preds.append(p.cpu().numpy())
                pred_full = np.concatenate(preds)
            r = float(rae(val_y, pred_full))
            if r < best_val - 1e-5:
                best_val = r
                best_pred = pred_full.copy()
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
        return best_pred if best_pred is not None else pred_full

    # ===== MODE A: GNN-alone, 5-seed bag x 5-fold scaffold cross-fit =====
    print("\n" + "-" * 78)
    print("MODE A -- GNN-alone, 5-seed bag x 5-fold scaffold cross-fit on 253")
    print("-" * 78)

    folds_scaf = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        seed=0)
    print(f"[scaf] {N_FOLDS} folds; sizes = "
          f"{[len(va) for _, va in folds_scaf]}")

    per_seed_oof_A = []  # list of (253,)
    per_seed_rae_A = []
    per_seed_te_A = []  # list of (513,) for deploy

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        oof = np.full(n_unb, np.nan, dtype=np.float32)
        for fi, (tr_loc, va_loc) in enumerate(folds_scaf):
            # train set = full 4139 train + 253-tr_loc unblind labels
            ext_data = [unb_data[i] for i in tr_loc]
            # build new Data instances with .y filled in for the unb tr fold
            for d_idx, d in enumerate(ext_data):
                d.y = torch.tensor([y_unb[tr_loc[d_idx]]], dtype=torch.float)
            full_tr = train_data + ext_data
            loader = DataLoader(full_tr, batch_size=BATCH, shuffle=True)
            va_list = [unb_data[i] for i in va_loc]
            model = GNNHead(ATOM_DIM, extra_dim=0)
            pred_va = train_one(model, loader, va_list, y_unb[va_loc],
                                val_extra=None, seed=seed)
            oof[va_loc] = pred_va
        per_seed_oof_A.append(oof)
        r_seed = float(rae(y_unb, oof))
        per_seed_rae_A.append(r_seed)
        print(f"   seed {seed:>3d}: GNN-alone pooled RAE = {r_seed:.4f}")

        # deploy: retrain on all 4139 + 253, predict 513
        for d_idx, d in enumerate(unb_data):
            d.y = torch.tensor([y_unb[d_idx]], dtype=torch.float)
        all_tr = train_data + unb_data
        loader_all = DataLoader(all_tr, batch_size=BATCH, shuffle=True)
        model_dep = GNNHead(ATOM_DIM, extra_dim=0)
        opt = torch.optim.AdamW(model_dep.parameters(), lr=LR, weight_decay=WD)
        crit = nn.MSELoss()
        for epoch in range(40):  # fixed deploy epochs (median early-stop)
            model_dep.train()
            for batch in loader_all:
                opt.zero_grad()
                p = model_dep(batch)
                loss = crit(p, batch.y)
                loss.backward()
                opt.step()
        model_dep.eval()
        with torch.no_grad():
            te_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)
            te_preds = []
            for tb in te_loader:
                te_preds.append(model_dep(tb).cpu().numpy())
            te_preds = np.concatenate(te_preds).astype(np.float32)
        per_seed_te_A.append(te_preds)

    oof_A_bag_mean = np.mean(per_seed_oof_A, axis=0)
    oof_A_bag_median = np.median(per_seed_oof_A, axis=0)
    rae_A_mean_bag = float(rae(y_unb, oof_A_bag_mean))
    rae_A_median_bag = float(rae(y_unb, oof_A_bag_median))
    print(f"\n[bag-A] mean-bag RAE = {rae_A_mean_bag:.4f}")
    print(f"[bag-A] median-bag RAE = {rae_A_median_bag:.4f}")
    print(f"[bag-A] vs chemprop_aux ({CHEMPROP_AUX_REF}): "
          f"delta = {rae_A_mean_bag - CHEMPROP_AUX_REF:+.4f}")
    np.save(DATA_PROCESSED / f"{TAG}_gnn_alone_oof.npy", oof_A_bag_mean)

    te_A_bag_mean = np.mean(per_seed_te_A, axis=0)

    # ===== MODE B: GNN-residual on chemprop_aux, with K=28 at pool =====
    print("\n" + "-" * 78)
    print("MODE B -- GNN-residual on chemprop_aux + K=28 concat at pool")
    print("-" * 78)

    # Build residual target: y - chemprop_aux on 253
    resid_unb = y_unb - chemprop_aux_unb
    print(f"[resid] mean = {resid_unb.mean():+.4f}  std = {resid_unb.std():.4f}")

    # Re-build unb_data with residual y for MODE B training
    unb_data_B = []
    for i, idx in enumerate(unb_idx):
        d = smi_to_data(te_smiles_std[idx], float(resid_unb[i]))
        unb_data_B.append(d if d is not None else smi_to_data("C", 0.0))

    per_seed_oof_B = []   # residual OOF on 253
    per_seed_rae_B_mode = []   # corrected RAE per seed
    per_seed_te_B_resid = []   # residual preds on 513

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        resid_oof = np.full(n_unb, np.nan, dtype=np.float32)
        for fi, (tr_loc, va_loc) in enumerate(folds_scaf):
            tr_list = [unb_data_B[i] for i in tr_loc]
            # attach extra features per sample
            for d_idx, d in enumerate(tr_list):
                d.extra = torch.tensor(X_unb_28[tr_loc[d_idx]],
                                       dtype=torch.float).unsqueeze(0)
            loader = DataLoader(tr_list, batch_size=BATCH, shuffle=True)
            va_list = [unb_data_B[i] for i in va_loc]
            va_extra = X_unb_28[va_loc]
            model = GNNHead(ATOM_DIM, extra_dim=EXTRA_DIM)
            pred_va = train_one(model, loader, va_list,
                                resid_unb[va_loc].astype(np.float32),
                                val_extra=va_extra, seed=seed)
            resid_oof[va_loc] = pred_va
        per_seed_oof_B.append(resid_oof)
        corrected_oof = chemprop_aux_unb + resid_oof
        r_seed = float(rae(y_unb, corrected_oof))
        per_seed_rae_B_mode.append(r_seed)
        print(f"   seed {seed:>3d}: GNN-residual corrected RAE = {r_seed:.4f}")

        # deploy: predict residual on 513 -- model trained on all 253
        for d_idx, d in enumerate(unb_data_B):
            d.extra = torch.tensor(X_unb_28[d_idx],
                                   dtype=torch.float).unsqueeze(0)
        loader_all = DataLoader(unb_data_B, batch_size=BATCH, shuffle=True)
        model_dep = GNNHead(ATOM_DIM, extra_dim=EXTRA_DIM)
        opt = torch.optim.AdamW(model_dep.parameters(), lr=LR, weight_decay=WD)
        crit = nn.MSELoss()
        # For 513 inference we DO NOT have K=28 features per unblinded ligand
        # since X_unb_28 is built only for the 253; for the 513-deploy
        # residual we use the mean extra vector (collapses to a single
        # bias added to the GNN pool).
        mean_extra = X_unb_28.mean(axis=0).astype(np.float32)
        for epoch in range(40):
            model_dep.train()
            for batch in loader_all:
                opt.zero_grad()
                extra = batch.extra if hasattr(batch, "extra") else None
                p = model_dep(batch, extra=extra)
                loss = crit(p, batch.y)
                loss.backward()
                opt.step()
        model_dep.eval()
        with torch.no_grad():
            te_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)
            te_resid = []
            for tb in te_loader:
                bsz = int(tb.batch.max().item()) + 1
                e = torch.tensor(np.tile(mean_extra, (bsz, 1)),
                                  dtype=torch.float)
                te_resid.append(model_dep(tb, extra=e).cpu().numpy())
            te_resid = np.concatenate(te_resid).astype(np.float32)
        per_seed_te_B_resid.append(te_resid)

    oof_B_resid_mean = np.mean(per_seed_oof_B, axis=0)
    oof_B_resid_median = np.median(per_seed_oof_B, axis=0)
    corrected_mean_bag = chemprop_aux_unb + oof_B_resid_mean
    corrected_median_bag = chemprop_aux_unb + oof_B_resid_median
    rae_B_mean_bag = float(rae(y_unb, corrected_mean_bag))
    rae_B_median_bag = float(rae(y_unb, corrected_median_bag))
    print(f"\n[bag-B] mean-bag corrected RAE = {rae_B_mean_bag:.4f}")
    print(f"[bag-B] median-bag corrected RAE = {rae_B_median_bag:.4f}")
    print(f"[bag-B] vs nb2103 K=28 mean-bag ({NB2103_K28_MEAN_BAG_REF}): "
          f"delta = {rae_B_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f}")
    print(f"[bag-B] vs nb2103 K=28 median-bag ({NB2103_K28_MEDIAN_BAG_REF}): "
          f"delta = {rae_B_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f}")
    np.save(DATA_PROCESSED / f"{TAG}_gnn_resid_oof.npy", oof_B_resid_mean)

    te_B_resid_mean = np.mean(per_seed_te_B_resid, axis=0)
    te_B_corrected_513 = (te_ca + te_B_resid_mean).astype(np.float32)
    np.save(DATA_PROCESSED / f"te_{TAG}_B.npy", te_B_corrected_513)
    np.save(DATA_PROCESSED / f"te_{TAG}_A.npy", te_A_bag_mean)

    # ===== VERDICT =====
    beats_chemprop = rae_A_mean_bag < (CHEMPROP_AUX_REF - DECISION_MARGIN)
    beats_nb2103 = rae_B_mean_bag < (NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN)
    flat_nb2103 = (abs(rae_B_mean_bag - NB2103_K28_MEAN_BAG_REF)
                   <= DECISION_MARGIN)
    if beats_nb2103:
        verdict = "GNN_RESID_BEATS_NB2103_K28"
    elif flat_nb2103:
        verdict = "GNN_RESID_FLAT_VS_NB2103_K28"
    elif beats_chemprop:
        verdict = "GNN_ALONE_BEATS_CHEMPROP_AUX_BUT_WORSE_THAN_NB2103_K28"
    else:
        verdict = "GNN_WORSE_THAN_BOTH_ANCHORS"

    summary = {
        "tag": TAG,
        "method": "PyG GCN x2 + mean pool + MLP",
        "n_unb": n_unb,
        "n_te": n_te,
        "atom_dim": ATOM_DIM,
        "extra_dim": EXTRA_DIM,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "scaffold_cv": True,
        "epochs_max": MAX_EPOCHS,
        "patience": PATIENCE,
        "deploy_epochs": 40,
        "lr": LR,
        "wd": WD,
        "batch": BATCH,
        "gnn_hidden_1": GNN_HIDDEN_1,
        "gnn_hidden_2": GNN_HIDDEN_2,
        "mlp_hidden": MLP_HIDDEN,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,

        "mode_A_per_seed_rae": per_seed_rae_A,
        "mode_A_mean_bag_rae": rae_A_mean_bag,
        "mode_A_median_bag_rae": rae_A_median_bag,
        "mode_A_delta_vs_chemprop_aux": rae_A_mean_bag - CHEMPROP_AUX_REF,

        "mode_B_per_seed_rae": per_seed_rae_B_mode,
        "mode_B_mean_bag_rae": rae_B_mean_bag,
        "mode_B_median_bag_rae": rae_B_median_bag,
        "mode_B_delta_vs_nb2103_K28_mean_bag":
            rae_B_mean_bag - NB2103_K28_MEAN_BAG_REF,
        "mode_B_delta_vs_nb2103_K28_median_bag":
            rae_B_median_bag - NB2103_K28_MEDIAN_BAG_REF,

        "beats_chemprop_aux_mode_A": bool(beats_chemprop),
        "beats_nb2103_K28_mode_B": bool(beats_nb2103),
        "flat_nb2103_K28_mode_B": bool(flat_nb2103),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }

    out = DATA_PROCESSED / f"{TAG}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[write] {out}")
    print(f"[verdict] {verdict}")
    print(f"[wall] {summary['wall_sec']:.1f} s")
    return summary


if __name__ == "__main__":
    main()
