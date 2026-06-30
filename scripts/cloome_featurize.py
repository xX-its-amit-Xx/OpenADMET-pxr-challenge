"""cloome_featurize.py — extract CLOOME molecule-tower embeddings (PHENOTYPIC axis: trained contrastively
vs cell-painting microscopy). Reconstruct the 'transformer' MLP from the checkpoint, load weights, feed
1024-bit Morgan r3 (chirality on) -> 512-d phenotypic embedding for 4139 train + 513 test.
Local CPU (tiny MLP). Saves to C:/pxr_struct/cloome/.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

CKPT = "C:/pxr_struct/cloome/cloome-bioactivity.pt"
OUT = "C:/pxr_struct/cloome"; os.makedirs(OUT, exist_ok=True)
NBITS, RADIUS = 1024, 3


def mol_tower():
    """4x [Linear+BN1d+ReLU] then Linear(1024->512); matches transformer.layers.* in the ckpt."""
    layers = []
    for _ in range(4):
        layers.append(nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU()))
    layers.append(nn.Sequential(nn.Linear(1024, 512)))
    return nn.ModuleList(layers)


class MolEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.layers = mol_tower()
    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x


def morgan1024(smis):
    X = np.zeros((len(smis), NBITS), dtype=np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=NBITS, useChirality=True)
        on = list(fp.GetOnBits())
        X[i, on] = 1.0
    return X


def main():
    sd_full = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    # extract module.transformer.* -> layers.*
    tsd = {}
    for k, v in sd_full.items():
        kk = k[7:] if k.startswith("module.") else k
        if kk.startswith("transformer."):
            tsd[kk[len("transformer."):]] = v
    enc = MolEncoder()
    missing, unexpected = enc.load_state_dict(tsd, strict=False)
    print(f"loaded mol tower; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  MISSING (first 6):", missing[:6])
    enc.eval()

    for tag, csv in [("4139", "data/processed/unimol_train.csv"), ("513", "data/processed/unimol_test513.csv")]:
        df = pd.read_csv(csv); smis = df["smiles"].astype(str).tolist()
        X = morgan1024(smis)
        with torch.no_grad():
            emb = enc(torch.from_numpy(X)).numpy().astype(np.float32)
        np.save(f"{OUT}/cloome_emb_{tag}.npy", emb)
        print(f"  {tag}: morgan{X.shape} -> emb{emb.shape}  emb_std={emb.std():.3f} (non-degenerate if >0.1)")
    print(f"saved -> {OUT}/cloome_emb_*.npy   NEXT: nb1008 integration test")


if __name__ == "__main__":
    main()
