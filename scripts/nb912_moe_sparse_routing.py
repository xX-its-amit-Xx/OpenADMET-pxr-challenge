"""nb912 -- Mixture-of-Experts with sparse top-2 routing.

Five small MLP experts, each consuming a different feature space:
    M  - Morgan 2048-d (ECFP4 bits as float)
    R  - RDKit ~217 2D descriptors (median-imputed, z-scored)
    MA - MACCS 167-d
    AP - AtomPair 2048-d (cached)
    CB - ChemBERTa 384-d (cached, te_chemberta.npy / tr_chemberta.npy)

A gating MLP on (Morgan + RDKit) ~2265 -> 32 -> 5 softmax produces per-compound
expert weights. We keep only the TOP-2 gates per compound, renormalize, and mix
those two experts' predictions. Trained end-to-end with Smooth-L1 (Huber) loss,
AdamW lr=1e-3, 30 epochs, batch=256, CPU torch.

Hypothesis: different feature spaces capture different chemistry (fingerprint
topology vs physchem vs pharmacophore vs learned LM embedding). Sparse gating
lets the model learn per-compound which 2 experts to trust.

Outputs:
    C:/pxr_artifacts/nb912_moe.pt              full checkpoint (state_dict)
    data/processed/te_nb912.npy                (513,) float32 test predictions
    submissions/nb912_moe_sparse_routing.csv   513-row submission
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
RDLogger.DisableLog("rdApp.*")

from pxr.data import load_train
from pxr.chem import add_standard_columns
from pxr.eval import rae
from pxr.featurize import rdkit_desc, morgan, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

TAG = "nb912"
SEED = 0
EPOCHS = 30
BATCH = 256
LR = 1e-3
WD = 1e-4
TOPK = 2
N_EXPERTS = 5

ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Feature helpers (load caches or compute fresh)
# ---------------------------------------------------------------------------
def _load_or_compute_maccs(smiles: list[str], cache: Path) -> np.ndarray:
    if cache.exists():
        x = np.load(cache)
        if x.shape == (len(smiles), 167):
            return x.astype(np.float32)
    out = np.zeros((len(smiles), 167), dtype=np.float32)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(str(smi)) if smi else None
        if m is None:
            continue
        bv = MACCSkeys.GenMACCSKeys(m)
        arr = np.zeros(167, dtype=np.int8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(bv, arr)
        out[i] = arr.astype(np.float32)
    np.save(cache, out.astype(np.uint8))
    return out


def _load_or_compute_atompair(smiles: list[str], cache: Path) -> np.ndarray:
    if cache.exists():
        x = np.load(cache)
        if x.shape == (len(smiles), 2048):
            return x.astype(np.float32)
    out = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(str(smi)) if smi else None
        if m is None:
            continue
        bv = AllChem.GetHashedAtomPairFingerprintAsBitVect(m, nBits=2048)
        arr = np.zeros(2048, dtype=np.int8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(bv, arr)
        out[i] = arr.astype(np.float32)
    np.save(cache, out.astype(np.uint8))
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ExpertMLP(nn.Module):
    """3-layer MLP: in -> 128 -> 64 -> 1."""
    def __init__(self, d_in: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Gate(nn.Module):
    """Small MLP on Morgan+RDKit -> 32 -> N_EXPERTS softmax."""
    def __init__(self, d_in: int, n_experts: int = N_EXPERTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(),
            nn.Linear(32, n_experts),
        )

    def forward(self, x):
        return self.net(x)  # logits


class MoE(nn.Module):
    def __init__(self, dims: dict[str, int], gate_dim: int, topk: int = TOPK):
        super().__init__()
        self.keys = ["M", "R", "MA", "AP", "CB"]
        self.experts = nn.ModuleDict({k: ExpertMLP(dims[k]) for k in self.keys})
        self.gate = Gate(gate_dim, n_experts=len(self.keys))
        self.topk = topk

    def forward(self, feats: dict[str, torch.Tensor], gate_x: torch.Tensor,
                return_gates: bool = False):
        # expert predictions: (B, N_experts)
        preds = torch.stack([self.experts[k](feats[k]) for k in self.keys], dim=1)

        # gate logits -> softmax over all experts
        logits = self.gate(gate_x)                            # (B, N)
        full_soft = F.softmax(logits, dim=-1)

        # sparse top-2: zero everyone else, renormalize over the kept 2
        top_vals, top_idx = torch.topk(full_soft, k=self.topk, dim=-1)   # (B, k)
        sparse = torch.zeros_like(full_soft)
        sparse.scatter_(1, top_idx, top_vals)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        y = (preds * sparse).sum(dim=-1)                      # (B,)
        if return_gates:
            return y, sparse, full_soft
        return y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(f"{TAG} -- MoE sparse-top{TOPK} routing across {N_EXPERTS} feature spaces")
    print("=" * 78)

    # Sanity-check required raw files
    blind = DATA_RAW / "pxr-challenge_TEST_BLINDED.csv"
    if not blind.exists():
        print("MISSING test file:", blind); return {"success": False}

    # ---------- Data ----------
    tr = add_standard_columns(load_train())
    smi_tr = tr["std_smiles"].tolist()
    y_tr = tr["pec50"].astype(np.float32).values
    te_df = pd.read_csv(blind)
    smi_te = te_df["SMILES"].tolist()
    n_tr, n_te = len(smi_tr), len(smi_te)
    print(f"Train n={n_tr}  Test n={n_te}")

    # ---------- Featurize ----------
    print("Featurizing (Morgan 2048, RDKit ~217, MACCS 167, AtomPair 2048, ChemBERTa 384)...")

    # Morgan + RDKit (computed fresh; small and fast)
    M_tr = morgan(smi_tr).astype(np.float32)               # (n,2048)
    M_te = morgan(smi_te).astype(np.float32)
    R_tr_raw = rdkit_desc(smi_tr).astype(np.float32)
    R_te_raw = rdkit_desc(smi_te).astype(np.float32)
    # Impute + standardize jointly (fit on train)
    all_R = impute(np.vstack([R_tr_raw, R_te_raw]))
    mu = all_R[:n_tr].mean(axis=0)
    sd = all_R[:n_tr].std(axis=0) + 1e-6
    R_tr = ((all_R[:n_tr] - mu) / sd).astype(np.float32)
    R_te = ((all_R[n_tr:] - mu) / sd).astype(np.float32)
    # Clip extreme outliers in descriptor space (helps MLP stability)
    R_tr = np.clip(R_tr, -5, 5); R_te = np.clip(R_te, -5, 5)

    # MACCS
    MA_tr = _load_or_compute_maccs(smi_tr, DATA_PROCESSED / "tr_maccs.npy")
    MA_te = _load_or_compute_maccs(smi_te, DATA_PROCESSED / "te_maccs.npy")

    # AtomPair
    AP_tr = _load_or_compute_atompair(smi_tr, DATA_PROCESSED / "tr_atompair.npy")
    AP_te = _load_or_compute_atompair(smi_te, DATA_PROCESSED / "te_atompair.npy")

    # ChemBERTa (cached - REQUIRED)
    cb_tr_p = DATA_PROCESSED / "tr_chemberta.npy"
    cb_te_p = DATA_PROCESSED / "te_chemberta.npy"
    CB_tr = np.load(cb_tr_p).astype(np.float32)
    CB_te = np.load(cb_te_p).astype(np.float32)
    # z-score ChemBERTa per dim using train stats (already roughly centered)
    mu_cb = CB_tr.mean(axis=0); sd_cb = CB_tr.std(axis=0) + 1e-6
    CB_tr = (CB_tr - mu_cb) / sd_cb
    CB_te = (CB_te - mu_cb) / sd_cb

    dims = {"M": M_tr.shape[1], "R": R_tr.shape[1], "MA": MA_tr.shape[1],
            "AP": AP_tr.shape[1], "CB": CB_tr.shape[1]}
    print("  dims:", dims)

    # Gate input: Morgan + RDKit-zscored ~ 2265
    G_tr = np.concatenate([M_tr, R_tr], axis=1).astype(np.float32)
    G_te = np.concatenate([M_te, R_te], axis=1).astype(np.float32)
    gate_dim = G_tr.shape[1]
    print(f"  gate input dim: {gate_dim}")

    # ---------- Tensors ----------
    def t(x): return torch.from_numpy(np.ascontiguousarray(x))
    feats_tr = {"M": t(M_tr), "R": t(R_tr), "MA": t(MA_tr), "AP": t(AP_tr), "CB": t(CB_tr)}
    feats_te = {"M": t(M_te), "R": t(R_te), "MA": t(MA_te), "AP": t(AP_te), "CB": t(CB_te)}
    G_tr_t = t(G_tr); G_te_t = t(G_te); y_tr_t = t(y_tr)

    ds = TensorDataset(
        feats_tr["M"], feats_tr["R"], feats_tr["MA"], feats_tr["AP"], feats_tr["CB"],
        G_tr_t, y_tr_t,
    )
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    # ---------- Model / Optim ----------
    model = MoE(dims, gate_dim=gate_dim, topk=TOPK)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber

    # ---------- Train ----------
    print(f"Training {EPOCHS} epochs (CPU, batch={BATCH}, Huber + AdamW lr={LR})...")
    model.train()
    for ep in range(1, EPOCHS + 1):
        tot, n_seen = 0.0, 0
        for bM, bR, bMA, bAP, bCB, bG, by in dl:
            feats = {"M": bM, "R": bR, "MA": bMA, "AP": bAP, "CB": bCB}
            pred = model(feats, bG)
            loss = loss_fn(pred, by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * by.shape[0]; n_seen += by.shape[0]
        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            print(f"  epoch {ep:02d}/{EPOCHS}  huber={tot/n_seen:.4f}")

    # ---------- Predict ----------
    model.eval()
    with torch.no_grad():
        te_pred, sparse_te, full_te = model(feats_te, G_te_t, return_gates=True)
        te_pred = te_pred.cpu().numpy().astype(np.float32)
        sparse_te = sparse_te.cpu().numpy()
        full_te = full_te.cpu().numpy()
    keys = ["M", "R", "MA", "AP", "CB"]
    chosen = (sparse_te > 0).sum(axis=0)
    print(f"  test pred mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print("  expert selection counts (sparse top-2 over 513 test compounds):")
    for k, c in zip(keys, chosen):
        print(f"    {k:>3}: {int(c):4d}  (mean full-soft weight={full_te[:, keys.index(k)].mean():.3f})")

    # ---------- Unblind in_RAE ----------
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    unb_y = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float32)
    in_rae = float(rae(unb_y, te_pred[unb_idx]))
    print("=" * 78)
    print(f"UNBLIND (n={len(unb_idx)}): in_RAE = {in_rae:.4f}")
    print(f"  truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std = {te_pred[unb_idx].mean():.3f} / {te_pred[unb_idx].std():.3f}")
    print("=" * 78)

    # ---------- Save artifacts ----------
    ckpt = ART / f"{TAG}_moe.pt"
    torch.save({
        "model_state": model.state_dict(),
        "dims": dims, "gate_dim": gate_dim, "topk": TOPK,
        "epochs": EPOCHS, "lr": LR, "wd": WD, "batch": BATCH,
        "seed": SEED, "in_rae": in_rae,
        "rdkit_mu": mu, "rdkit_sd": sd, "cb_mu": mu_cb, "cb_sd": sd_cb,
    }, ckpt)
    print(f"Wrote checkpoint -> {ckpt}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred)
    np.save(DATA_PROCESSED / f"te_{TAG}_gate_sparse.npy", sparse_te.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}_gate_full.npy", full_te.astype(np.float32))

    sub = SUBMISSIONS / f"{TAG}_moe_sparse_routing.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         te_pred,
    }).to_csv(sub, index=False)
    print(f"Wrote submission -> {sub}")

    return {
        "success": True,
        "in_rae": in_rae,
        "submission": str(sub),
        "checkpoint": str(ckpt),
        "n_epochs": EPOCHS,
    }


if __name__ == "__main__":
    out = main()
    print("\nRESULT:", out)
