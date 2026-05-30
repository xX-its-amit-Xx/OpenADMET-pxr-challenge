"""nb299 -- NR-CLIP: contrastive dual-encoder for compound x NR-target pairs.

Two towers:
  - Compound tower: combined(Morgan+RDKit) -> MLP -> 128-d embedding
  - Target tower: target-specific learnable 128-d embedding (per UniProt)

Positives: known (compound, NR_target) binding pairs (any pec50 >= 4.5)
Negatives: random NR_target for the same compound
Loss: InfoNCE over batch.

The COMPOUND embedding is then used as a feature for LGBM on PXR.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def main():
    print("=== nb299: NR-CLIP contrastive dual-encoder ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    # NR binding pairs from papyrus_full_wide
    try:
        pap = pd.read_parquet("data/external/papyrus_full_wide.parquet")
        target_cols = [c for c in pap.columns if c not in ('SMILES', 'std_smiles')]
        print(f"NR targets: {len(target_cols)}")
    except FileNotFoundError:
        print("papyrus_full_wide not found; skipping (return zeros)")
        emb_tr = np.zeros((len(smiles_tr), 128))
        emb_te = np.zeros((len(smiles_te), 128))
        np.save(DATA_PROCESSED / "oof_nb299_nrclip.npy", np.full(len(y), y.mean()))
        np.save(DATA_PROCESSED / "te_nb299_nrclip.npy", np.full(len(smiles_te), y.mean()))
        return

    # Build positives: (smiles, target_idx) for entries with pec50 >= 4.5
    pos_pairs = []
    for ti, t in enumerate(target_cols):
        sub = pap.dropna(subset=[t])
        for s, p in zip(sub['SMILES'], sub[t]):
            if p >= 4.5:
                pos_pairs.append((str(s), ti))
    print(f"Positive pairs (pec50>=4.5): {len(pos_pairs)}")

    # Unique compounds
    all_smiles = list(set([s for s, _ in pos_pairs] + smiles_tr + smiles_te))
    sm_idx = {s: i for i, s in enumerate(all_smiles)}
    print(f"Unique compounds: {len(all_smiles)}")

    print("Featurising compounds...")
    BATCH = 2000
    feats = []
    for i in range(0, len(all_smiles), BATCH):
        f = impute(combined(all_smiles[i:i+BATCH])).astype(np.float32)
        feats.append(f)
        if (i // BATCH) % 5 == 0:
            print(f"  {min(i+BATCH, len(all_smiles))}/{len(all_smiles)}")
    feats = np.vstack(feats)
    mu = feats.mean(0); sd = feats.std(0) + 1e-6
    feats = ((feats - mu) / sd).clip(-5, 5).astype(np.float32)
    print(f"  feats: {feats.shape}")

    class ChemTower(nn.Module):
        def __init__(self, d_in, d_out=128):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, d_out),
            )
        def forward(self, x): return F.normalize(self.body(x), dim=-1)

    class TargetTower(nn.Module):
        def __init__(self, n_targets, d_out=128):
            super().__init__()
            self.emb = nn.Embedding(n_targets, d_out)
        def forward(self, ti): return F.normalize(self.emb(ti), dim=-1)

    n_t = len(target_cols)
    chem = ChemTower(feats.shape[1], 128)
    tgt = TargetTower(n_t, 128)
    opt = torch.optim.Adam(list(chem.parameters()) + list(tgt.parameters()), lr=1e-3)
    Xt = torch.tensor(feats)

    pos_idx = [(sm_idx[s], ti) for s, ti in pos_pairs if s in sm_idx]
    si = torch.tensor([p[0] for p in pos_idx], dtype=torch.long)
    ti = torch.tensor([p[1] for p in pos_idx], dtype=torch.long)
    print(f"\nTraining NR-CLIP for 8 epochs on {len(si)} pairs...")
    B = 1024
    n = len(si)
    for epoch in range(8):
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, B):
            ii = perm[i:i+B]
            cs = chem(Xt[si[ii]])  # (B, 128)
            tg = tgt(ti[ii])        # (B, 128)
            sim = cs @ tg.t() / 0.07
            labels = torch.arange(len(ii))
            loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"  ep{epoch+1}: loss={np.mean(losses):.4f}")

    chem.eval()
    with torch.no_grad():
        emb_all = chem(Xt).cpu().numpy()
    emb_tr = np.array([emb_all[sm_idx[s]] if s in sm_idx else np.zeros(128) for s in smiles_tr])
    emb_te = np.array([emb_all[sm_idx[s]] if s in sm_idx else np.zeros(128) for s in smiles_te])
    print(f"CLIP embeddings: train{emb_tr.shape}, test{emb_te.shape}")

    # LGBM with combined+CLIP
    X_tr_base = impute(combined(smiles_tr)).astype(np.float32)
    X_te_base = impute(combined(smiles_te)).astype(np.float32)
    X_tr = np.column_stack([X_tr_base, emb_tr]); X_te = np.column_stack([X_te_base, emb_te])

    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=10, objective='mae',
                n_jobs=4, random_state=42, verbose=-1)
    oof = np.zeros(len(y))
    te_preds = []
    for tii, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr[tii], y[tii], eval_set=[(X_tr[vi], y[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        te_preds.append(md.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y, oof)
    sp, _ = spearmanr(y, oof)
    print(f"\nNR-CLIP+base LGBM OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb299_nrclip.npy", oof)
    np.save(DATA_PROCESSED / "te_nb299_nrclip.npy", te_pred)

    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}, weight(nb299)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
