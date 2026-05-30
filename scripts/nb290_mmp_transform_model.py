"""nb290 -- Explicit MMP transform NN.

Predicts ΔpEC50 from (fragment_removed_FP, fragment_added_FP, context_FP).
At inference, applies trained transform to K=10 most-similar train neighbors
and averages predictions weighted by similarity.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA
from rdkit.DataStructs import BulkTanimotoSimilarity, ConvertToNumpyArray
import torch, torch.nn as nn

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED

NBITS = 512


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception: return None


def morgan_arr(smi, n_bits=NBITS, radius=2):
    if not smi: return np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8); ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def morgan_fp(smi, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None


def mine_mmp_pairs(smiles_list, y, max_compounds=None):
    """Mine single-cut MMP pairs. Returns list of (i, j, frag_removed_smi, frag_added_smi, ctx_smi, dy)."""
    n = len(smiles_list) if max_compounds is None else min(max_compounds, len(smiles_list))
    # Per-compound: cut once -> dict[ctx_smi] = list of (i, frag_smi)
    ctx_dict = {}
    t0 = time.time()
    for i in range(n):
        smi = smiles_list[i]
        if not smi: continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        try:
            frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False, maxCutBonds=20)
        except Exception:
            continue
        for core, chains in frags:
            # single-cut: core is empty; chains = "A.B" where one piece becomes context
            pieces = chains.split('.')
            if len(pieces) != 2: continue
            a, b = pieces
            # Use 'a' as context, 'b' as variable (and vice versa)
            for ctx, frag in [(a, b), (b, a)]:
                # Heuristic ratio: frag heavy atoms / total < 0.5
                try:
                    fm = Chem.MolFromSmiles(frag)
                    if fm is None or fm.GetNumHeavyAtoms() == 0: continue
                except Exception: continue
                ctx_dict.setdefault(ctx, []).append((i, frag))
        if (i + 1) % 500 == 0:
            print(f"    fragment scan {i+1}/{n} ({time.time()-t0:.1f}s)")
    # Build pairs from same context
    pairs = []
    for ctx, lst in ctx_dict.items():
        if len(lst) < 2: continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                i, fi = lst[a]; j, fj = lst[b]
                if i == j: continue
                dy = y[j] - y[i]
                if abs(dy) <= 0.5: continue
                pairs.append((i, j, fi, fj, ctx, dy))
    return pairs


class TransformNet(nn.Module):
    def __init__(self, in_dim=3 * NBITS, h1=512, h2=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def main():
    print("=== nb290: MMP transform NN ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv('data/raw/pxr-challenge_TEST_BLINDED.csv')
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()

    print("Mining MMP pairs (single-cut)...")
    pairs = mine_mmp_pairs(smiles_tr, y_tr, max_compounds=len(smiles_tr))
    print(f"Found {len(pairs)} MMP pairs with |delta_y|>0.5")
    if len(pairs) < 100:
        print("Too few pairs; saving placeholders and skipping training.")
        np.save(DATA_PROCESSED / "oof_nb290_mmp_transform.npy", np.full(len(y_tr), y_tr.mean()))
        np.save(DATA_PROCESSED / "te_nb290_mmp_transform.npy", np.full(len(smiles_te), y_tr.mean()))
        return

    # Subsample pairs to keep memory tractable (cap 200k pairs)
    MAX_PAIRS = 200_000
    if len(pairs) > MAX_PAIRS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), MAX_PAIRS, replace=False)
        pairs = [pairs[i] for i in idx]
        print(f"  Subsampled to {len(pairs)} pairs")

    # Build training tensors
    print("Featurizing pair inputs...")
    X_pairs = np.zeros((len(pairs), 3 * NBITS), dtype=np.float32)
    y_pairs = np.zeros(len(pairs), dtype=np.float32)
    for k, (i, j, frem, fadd, ctx, dy) in enumerate(pairs):
        X_pairs[k, :NBITS] = morgan_arr(frem)
        X_pairs[k, NBITS:2*NBITS] = morgan_arr(fadd)
        X_pairs[k, 2*NBITS:] = morgan_arr(ctx)
        y_pairs[k] = dy
    print(f"X_pairs shape: {X_pairs.shape}, y mean={y_pairs.mean():.3f} std={y_pairs.std():.3f}")

    device = 'cpu'
    model = TransformNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()
    Xt = torch.from_numpy(X_pairs); yt = torch.from_numpy(y_pairs)
    bs = 256
    print("Training MLP 50 epochs...")
    for ep in range(50):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs]
            opt.zero_grad()
            p = model(Xt[idx]); l = loss_fn(p, yt[idx])
            l.backward(); opt.step(); tot += l.item() * len(idx)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1} loss={tot/len(Xt):.4f}")
    model.eval()

    # Inference helper: for compound q, find K=10 train neighbors, build pseudo
    # (removed, added, ctx) via symmetric-difference fingerprints on Morgan r=2 nbits=NBITS.
    print("Computing FPs for inference...")
    arr_tr = np.stack([morgan_arr(s) for s in smiles_tr])
    arr_te = np.stack([morgan_arr(s) for s in smiles_te])
    fp_tr_big = [morgan_fp(s, n_bits=2048) for s in smiles_tr]
    fp_te_big = [morgan_fp(s, n_bits=2048) for s in smiles_te]
    fp_tr_arr_big = np.zeros((len(smiles_tr), 2048), dtype=np.uint8)
    for i, fp in enumerate(fp_tr_big):
        if fp is not None: ConvertToNumpyArray(fp, fp_tr_arr_big[i])

    def predict_set(query_fps_big, query_arr_small, exclude_self=None):
        out = np.full(len(query_fps_big), y_tr.mean())
        K = 10
        for i, fpq in enumerate(query_fps_big):
            if fpq is None: continue
            sims = np.array(BulkTanimotoSimilarity(fpq, fp_tr_big))
            if exclude_self is not None and exclude_self[i] is not None:
                for j in exclude_self[i]: sims[j] = -1
            top_idx = np.argsort(sims)[::-1][:K]
            top_sims = sims[top_idx]
            if top_sims[0] <= 0: continue
            qbits = query_arr_small[i]
            nbits = arr_tr[top_idx]
            removed = np.clip(nbits - qbits[None, :], 0, 1)
            added = np.clip(qbits[None, :] - nbits, 0, 1)
            ctx = np.minimum(nbits, qbits[None, :])
            X_inf = np.concatenate([removed, added, ctx], axis=1).astype(np.float32)
            with torch.no_grad():
                deltas = model(torch.from_numpy(X_inf)).numpy()
            base = y_tr[top_idx]
            pred = base + deltas
            w = np.maximum(top_sims, 0)
            if w.sum() <= 0: continue
            out[i] = float((pred * w).sum() / w.sum())
        return out

    print("Predicting test...")
    te_pred = predict_set(fp_te_big, arr_te)
    print(f"te_pred mean={te_pred.mean():.3f} std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "te_nb290_mmp_transform.npy", te_pred)

    print("Predicting OOF (LOO via scaffold mask)...")
    scaffolds = tr['scaffold'].tolist()
    scaf_to_idx = {}
    for i, s in enumerate(scaffolds): scaf_to_idx.setdefault(s, []).append(i)
    excl = [scaf_to_idx[scaffolds[i]] for i in range(len(smiles_tr))]
    oof = predict_set(fp_tr_big, arr_tr, exclude_self=excl)
    r = rae(y_tr, oof); sp, _ = spearmanr(y_tr, oof)
    print(f"OOF RAE={r:.4f}  Spearman={sp:.4f}")
    np.save(DATA_PROCESSED / "oof_nb290_mmp_transform.npy", oof)

    # SLSQP 5-way
    print("\n=== 5-way SLSQP with nb290 ===")
    try:
        nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
        nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
        mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
        loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
        M = np.column_stack([nb224, nb179s, mtd, loso, oof])
        def loss(w): return rae(y_tr, M @ w)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0, 1.0)] * 5
        best = None
        for seed in range(80):
            w0 = np.random.default_rng(seed).dirichlet(np.ones(5))
            r2 = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'ftol': 1e-9, 'maxiter': 200})
            if best is None or r2.fun < best.fun: best = r2
        print(f"SLSQP OOF: {best.fun:.4f}  weights={np.round(best.x, 4)}")
    except Exception as e:
        print(f"SLSQP skipped: {e}")


if __name__ == "__main__":
    main()
