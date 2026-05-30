"""nb287 -- AlphaFold-2 / Evoformer-inspired multi-track architecture for PXR pEC50.

Core AF2 idea: iteratively co-refine 1D (per-residue) and 2D (per-pair) tracks
with cross-updates. Adapted to molecules:
  - 1D track: per-atom embeddings (atom type, degree, charge, hybridization, aromatic)
  - 2D track: per-atom-pair embeddings (initial = graph distance + bond type on path)
  - Evoformer block (x4):
      * "Triangle" update on 2D: f(pair_ij + pair_ik + pair_jk) hop-mixing
      * Atom self-attention (1D) biased by 2D pair representation
      * Cross updates: 1D->2D via outer_product_mean, 2D->1D via attention pooling
  - Head: mean-pool 1D track -> MLP -> pec50

Spec:
  d_1D=64, d_2D=32, n_heads=4, n_blocks=4, dropout=0.1, MAX_ATOMS=80
  60 epochs, Adam lr=3e-4, MAE loss, cosine LR schedule, scaffold 5-fold CV
"""
import os, sys, time, warnings, math
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device("cpu")

# ---------------- Constants ----------------
MAX_ATOMS = 80
ATOM_NUMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H,B,C,N,O,F,Si,P,S,Cl,Br,I
HYBRIDS = [Chem.rdchem.HybridizationType.SP,
           Chem.rdchem.HybridizationType.SP2,
           Chem.rdchem.HybridizationType.SP3,
           Chem.rdchem.HybridizationType.SP3D,
           Chem.rdchem.HybridizationType.SP3D2]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]
# atom_num_onehot(13) + degree(7: 0..5+other) + charge(6) + hybrid(6: 5+other)
# + aromatic(1) + in_ring(1) + numH(6) = 40
N_ATOM_FEAT = (len(ATOM_NUMS) + 1) + 7 + 6 + (len(HYBRIDS) + 1) + 1 + 1 + 6  # 40
N_PAIR_FEAT = 11 + (len(BOND_TYPES) + 1)  # 11 distance bins (0..10+) + bond-type one-hot (5)

# Architecture spec
D_1D = 64
D_2D = 32
N_HEADS = 4
N_BLOCKS = 4
DROPOUT = 0.1


# ---------------- Featurization ----------------
def one_hot(val, choices):
    v = [0] * (len(choices) + 1)
    if val in choices:
        v[choices.index(val)] = 1
    else:
        v[-1] = 1
    return v


def atom_feats(atom):
    feats = []
    feats += one_hot(atom.GetAtomicNum(), ATOM_NUMS)         # 13
    feats += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])    # 7
    feats += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])  # 6
    feats += one_hot(atom.GetHybridization(), HYBRIDS)        # 6
    feats += [int(atom.GetIsAromatic())]                       # 1
    feats += [int(atom.IsInRing())]                            # 1
    feats += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])    # 6
    return feats


def featurize_mol(smi):
    """Return (x_1D[N,F_atom], pair[N,N,F_pair], mask[N]) or None if invalid / >MAX_ATOMS."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n == 0 or n > MAX_ATOMS:
        return None
    # 1D atom features
    x = np.array([atom_feats(a) for a in mol.GetAtoms()], dtype=np.float32)  # [n, F_atom]
    # 2D pair features
    dist = Chem.GetDistanceMatrix(mol)  # [n,n] shortest path lengths
    dist = np.clip(dist, 0, 10).astype(np.int64)
    pair_dist = np.eye(11, dtype=np.float32)[dist]  # [n,n,11]
    # Bond-type matrix: one-hot if directly bonded, else "no bond" slot
    bond_idx = np.full((n, n), len(BOND_TYPES), dtype=np.int64)  # default = "no bond"
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = b.GetBondType()
        if bt in BOND_TYPES:
            k = BOND_TYPES.index(bt)
        else:
            k = len(BOND_TYPES)  # unknown -> "no bond" bucket
        bond_idx[i, j] = k
        bond_idx[j, i] = k
    pair_bond = np.eye(len(BOND_TYPES) + 1, dtype=np.float32)[bond_idx]  # [n,n,5]
    pair = np.concatenate([pair_dist, pair_bond], axis=-1)  # [n,n,16]
    mask = np.ones(n, dtype=np.float32)
    return x, pair, mask


def pad_to_max(x, pair, mask):
    n = x.shape[0]
    F_a = x.shape[1]
    F_p = pair.shape[-1]
    x_p = np.zeros((MAX_ATOMS, F_a), dtype=np.float32); x_p[:n] = x
    p_p = np.zeros((MAX_ATOMS, MAX_ATOMS, F_p), dtype=np.float32); p_p[:n, :n] = pair
    m_p = np.zeros(MAX_ATOMS, dtype=np.float32); m_p[:n] = mask
    return x_p, p_p, m_p


# ---------------- Model ----------------
class TriangleUpdate(nn.Module):
    """Light 'triangle' update on 2D track.

    Approximation of AF2 triangle multiplicative: pair_ij <- MLP(pair_ij +
    mean_k(pair_ik + pair_jk)).
    """
    def __init__(self, d_pair):
        super().__init__()
        self.ln = nn.LayerNorm(d_pair)
        self.mlp = nn.Sequential(
            nn.Linear(d_pair, 2 * d_pair),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2 * d_pair, d_pair),
        )

    def forward(self, z, mask):
        # z: [B, N, N, d_pair]; mask: [B, N]
        z_n = self.ln(z)
        # Mean over hop k of (pair_ik + pair_jk)
        # outgoing: mean_k z[:, i, k]; incoming: mean_k z[:, k, j]
        m = mask.unsqueeze(-1)  # [B, N, 1]
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B,1]
        # z_out: average over j-axis of z[:,i,j]
        z_out = (z_n * m.unsqueeze(1)).sum(dim=2) / denom.unsqueeze(-1)  # [B, N, d_pair]
        z_in = (z_n * m.unsqueeze(2)).sum(dim=1) / denom.unsqueeze(-1)   # [B, N, d_pair]
        # one extra hop: combine z_out[i] + z_in[j]
        hop = z_out.unsqueeze(2) + z_in.unsqueeze(1)  # [B, N, N, d_pair]
        return z + self.mlp(z_n + hop)


class AtomAttention(nn.Module):
    """Multi-head self-attention over 1D track biased by 2D pair representation."""
    def __init__(self, d_1d, d_2d, n_heads):
        super().__init__()
        assert d_1d % n_heads == 0
        self.h = n_heads
        self.d_head = d_1d // n_heads
        self.ln_x = nn.LayerNorm(d_1d)
        self.ln_z = nn.LayerNorm(d_2d)
        self.qkv = nn.Linear(d_1d, 3 * d_1d, bias=False)
        self.bias_proj = nn.Linear(d_2d, n_heads, bias=False)
        self.out = nn.Linear(d_1d, d_1d)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, z, mask):
        # x: [B,N,d_1d], z: [B,N,N,d_2d], mask: [B,N]
        B, N, _ = x.shape
        x_n = self.ln_x(x)
        z_n = self.ln_z(z)
        qkv = self.qkv(x_n).reshape(B, N, 3, self.h, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each [B,N,h,d_head]
        # attention scores: [B, h, N, N]
        attn = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(self.d_head)
        bias = self.bias_proj(z_n).permute(0, 3, 1, 2)  # [B,h,N,N]
        attn = attn + bias
        # mask: -inf where key invalid
        key_mask = (mask < 0.5).unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        attn = attn.masked_fill(key_mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v).reshape(B, N, -1)
        return x + self.drop(self.out(out))


class OuterProductMean(nn.Module):
    """1D -> 2D update: outer product of left/right projections of x.

    Note: AF2 uses a low-dim outer product to save memory. We keep d_proj small
    (8) since intermediate tensor is [B, N, N, d_proj*d_proj].
    """
    def __init__(self, d_1d, d_2d, d_proj=8):
        super().__init__()
        self.ln = nn.LayerNorm(d_1d)
        self.left = nn.Linear(d_1d, d_proj)
        self.right = nn.Linear(d_1d, d_proj)
        self.out = nn.Linear(d_proj * d_proj, d_2d)

    def forward(self, x, z, mask):
        x_n = self.ln(x)
        a = self.left(x_n)   # [B,N,d_proj]
        b = self.right(x_n)  # [B,N,d_proj]
        # outer product: [B, N, N, d_proj, d_proj]
        op = a.unsqueeze(2).unsqueeze(-1) * b.unsqueeze(1).unsqueeze(-2)
        op = op.reshape(*op.shape[:3], -1)
        return z + self.out(op)


class PairToAtom(nn.Module):
    """2D -> 1D readout: attention-pool z along j-axis into x."""
    def __init__(self, d_1d, d_2d, n_heads):
        super().__init__()
        self.h = n_heads
        self.ln = nn.LayerNorm(d_2d)
        self.score = nn.Linear(d_2d, n_heads)
        self.val = nn.Linear(d_2d, d_1d)
        self.out = nn.Linear(d_1d, d_1d)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, z, mask):
        z_n = self.ln(z)
        # scores: [B,N,N,h]
        s = self.score(z_n)
        key_mask = (mask < 0.5).unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]
        s = s.masked_fill(key_mask, -1e9)
        attn = F.softmax(s, dim=2)  # softmax over j
        v = self.val(z_n)  # [B,N,N,d_1d]
        # split v by heads to weight by per-head attn
        B, N, _, D = v.shape
        v = v.reshape(B, N, N, self.h, D // self.h)
        out = (attn.unsqueeze(-1) * v).sum(dim=2)  # [B,N,h,d_head]
        out = out.reshape(B, N, D)
        return x + self.drop(self.out(out))


class EvoformerBlock(nn.Module):
    def __init__(self, d_1d, d_2d, n_heads):
        super().__init__()
        self.tri = TriangleUpdate(d_2d)
        self.attn = AtomAttention(d_1d, d_2d, n_heads)
        self.opm = OuterProductMean(d_1d, d_2d)
        self.p2a = PairToAtom(d_1d, d_2d, n_heads)
        # FFN on 1D
        self.ln = nn.LayerNorm(d_1d)
        self.ffn = nn.Sequential(
            nn.Linear(d_1d, 2 * d_1d),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2 * d_1d, d_1d),
        )

    def forward(self, x, z, mask):
        z = self.tri(z, mask)
        x = self.attn(x, z, mask)
        z = self.opm(x, z, mask)
        x = self.p2a(x, z, mask)
        x = x + self.ffn(self.ln(x))
        return x, z


class EvoformerPxR(nn.Module):
    def __init__(self, n_atom_feat=N_ATOM_FEAT, n_pair_feat=N_PAIR_FEAT,
                 d_1d=D_1D, d_2d=D_2D, n_heads=N_HEADS, n_blocks=N_BLOCKS):
        super().__init__()
        self.embed_x = nn.Linear(n_atom_feat, d_1d)
        self.embed_z = nn.Linear(n_pair_feat, d_2d)
        self.blocks = nn.ModuleList([EvoformerBlock(d_1d, d_2d, n_heads)
                                     for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(d_1d)
        self.head = nn.Sequential(
            nn.Linear(d_1d, d_1d),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_1d, 1),
        )

    def forward(self, x_in, z_in, mask):
        # x_in: [B,N,F_a]; z_in: [B,N,N,F_p]; mask: [B,N]
        x = self.embed_x(x_in)
        z = self.embed_z(z_in)
        for blk in self.blocks:
            x, z = blk(x, z, mask)
        x = self.ln_out(x)
        # mean-pool over valid atoms
        m = mask.unsqueeze(-1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


# ---------------- Dataset ----------------
class MolDS(torch.utils.data.Dataset):
    def __init__(self, feats, ys=None):
        self.feats = feats
        self.ys = ys

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        x, z, m = self.feats[idx]
        out = (torch.from_numpy(x), torch.from_numpy(z), torch.from_numpy(m))
        if self.ys is not None:
            out = out + (torch.tensor(self.ys[idx], dtype=torch.float32),)
        return out


def collate(batch):
    if len(batch[0]) == 4:
        xs, zs, ms, ys = zip(*batch)
        return torch.stack(xs), torch.stack(zs), torch.stack(ms), torch.stack(ys)
    xs, zs, ms = zip(*batch)
    return torch.stack(xs), torch.stack(zs), torch.stack(ms)


# ---------------- Train / Eval ----------------
def train_fold(train_ds, val_ds, te_ds, epochs=60, batch_size=16, lr=3e-4):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=False, collate_fn=collate)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=batch_size,
                                            shuffle=False, collate_fn=collate)
    model = EvoformerPxR().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        for x, z, m, y in train_loader:
            x, z, m, y = x.to(DEVICE), z.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            pred = model(x, z, m)
            loss = F.l1_loss(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vps, vys = [], []
            for x, z, m, y in val_loader:
                x, z, m = x.to(DEVICE), z.to(DEVICE), m.to(DEVICE)
                vps.append(model(x, z, m).cpu().numpy()); vys.append(y.numpy())
        vp = np.concatenate(vps); vy = np.concatenate(vys)
        vmae = np.abs(vp - vy).mean()
        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vps = []
        for x, z, m, y in val_loader:
            x, z, m = x.to(DEVICE), z.to(DEVICE), m.to(DEVICE)
            vps.append(model(x, z, m).cpu().numpy())
        val_pred = np.concatenate(vps)
        tps = []
        for batch in te_loader:
            x, z, m = (b.to(DEVICE) for b in batch[:3])
            tps.append(model(x, z, m).cpu().numpy())
        te_pred = np.concatenate(tps)
    return val_pred, te_pred, best_val, n_params


def main():
    print("=== nb287: Evoformer-inspired PXR model ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_full = tr["pec50"].values.astype(np.float32)
    smi_tr = tr["std_smiles"].tolist()
    scaffolds_full = tr["scaffold"].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")

    def std(s):
        try:
            m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else str(s)
        except Exception:
            return str(s)
    smi_te = [std(s) for s in te_df["SMILES"].tolist()]
    print(f"Train: {len(smi_tr)}, Test: {len(smi_te)}, MAX_ATOMS={MAX_ATOMS}")

    print("Featurizing train molecules...")
    t0 = time.time()
    tr_feats_raw = [featurize_mol(s) for s in smi_tr]
    keep_idx = [i for i, f in enumerate(tr_feats_raw) if f is not None]
    skip_train = len(tr_feats_raw) - len(keep_idx)
    print(f"  built in {time.time()-t0:.1f}s; kept {len(keep_idx)}/{len(tr_feats_raw)} "
          f"(skipped {skip_train} with >{MAX_ATOMS} atoms or invalid)")

    tr_feats = [pad_to_max(*tr_feats_raw[i]) for i in keep_idx]
    y_keep = y_full[keep_idx]
    scaffolds_keep = [scaffolds_full[i] for i in keep_idx]

    print("Featurizing test molecules...")
    te_feats_raw = [featurize_mol(s) for s in smi_te]
    te_kept_mask = np.array([f is not None for f in te_feats_raw])
    n_te_skip = (~te_kept_mask).sum()
    print(f"  test kept {te_kept_mask.sum()}/{len(smi_te)} (skipped {n_te_skip})")
    # For padded test, build for kept; for skipped use median later
    te_kept_idx = np.where(te_kept_mask)[0]
    te_feats = [pad_to_max(*te_feats_raw[i]) for i in te_kept_idx]

    folds = scaffold_kfold_indices(scaffolds_keep, n_splits=5)
    EPOCHS = 60

    oof = np.full(len(y_keep), np.nan, dtype=np.float32)
    te_preds = []
    total_params = None
    for fi, (ti, vi) in enumerate(folds):
        t0 = time.time()
        train_ds = MolDS([tr_feats[i] for i in ti], y_keep[list(ti)])
        val_ds = MolDS([tr_feats[i] for i in vi], y_keep[list(vi)])
        te_ds = MolDS(te_feats)
        vp, tp, best_val, n_params = train_fold(train_ds, val_ds, te_ds,
                                                 epochs=EPOCHS)
        if total_params is None:
            total_params = n_params
            print(f"  model params: {n_params:,}")
        oof[vi] = vp
        te_preds.append(tp)
        fold_rae = rae(y_keep[vi], vp)
        print(f"  fold {fi}: best_val_mae={best_val:.4f} fold_rae={fold_rae:.4f} "
              f"time={time.time()-t0:.1f}s", flush=True)
        # checkpoint
        np.save(DATA_PROCESSED / "oof_nb287_evoformer_partial.npy", oof)

    te_pred_kept = np.mean(te_preds, axis=0)  # [n_te_kept]

    # Build full-length OOF aligned to original train order
    # Compounds skipped during training: fill with median pec50 (won't be evaluated
    # as such, but keep shape == 4139 for stacking)
    oof_full = np.full(len(y_full), float(np.median(y_full)), dtype=np.float32)
    for ki, oi in zip(keep_idx, range(len(keep_idx))):
        oof_full[ki] = oof[oi]

    # Build full-length test predictions: skipped test compounds get median
    median_y = float(np.median(y_keep))
    te_pred_full = np.full(len(smi_te), median_y, dtype=np.float32)
    for k_pos, t_idx in enumerate(te_kept_idx):
        te_pred_full[t_idx] = te_pred_kept[k_pos]

    # Metrics on KEPT subset for fairness
    oof_rae = rae(y_keep, oof)
    sp, _ = spearmanr(y_keep, oof)
    mae = float(np.abs(y_keep - oof).mean())
    print(f"\nOOF (kept) RAE={oof_rae:.4f}  Spearman={sp:.4f}  MAE={mae:.4f}")
    print(f"te_pred (full): mean={te_pred_full.mean():.3f} std={te_pred_full.std():.3f} "
          f"min={te_pred_full.min():.3f} max={te_pred_full.max():.3f}")

    np.save(DATA_PROCESSED / "oof_nb287_evoformer.npy", oof_full)
    np.save(DATA_PROCESSED / "te_nb287_evoformer.npy", te_pred_full)

    # ---------------- SLSQP blend ----------------
    print("\n=== 5-way SLSQP blend: nb224 + nb179s + mtd + loso + nb287 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof_full])
    y64 = y_full.astype(np.float64)

    def loss(w):
        return rae(y64, M @ w)

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res
    print(f"5-way SLSQP OOF RAE: {best.fun:.4f}")
    labels = ["nb224", "nb179s", "mtd", "loso", "nb287"]
    for lab, w in zip(labels, best.x):
        print(f"  {lab}: {w:.4f}")
    print(f"\nnb287 weight: {best.x[4]:.4f}")


if __name__ == "__main__":
    main()
