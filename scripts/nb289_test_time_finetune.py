"""nb289 -- Per-test-compound test-time fine-tuning.

For each test compound: gather K=200 Tanimoto-nearest neighbors from train +
Papyrus PXR, train a tiny LGBM on those, predict the single test compound.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def fps(smiles, radius=2, n_bits=2048):
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        out.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return out


def fp_to_arr(fp_list, n_bits=2048):
    n_valid = sum(1 for f in fp_list if f is not None)
    X = np.zeros((len(fp_list), n_bits), dtype=np.uint8)
    for i, f in enumerate(fp_list):
        if f is not None:
            arr = np.zeros(n_bits, dtype=np.uint8)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(f, arr)
            X[i] = arr
    return X


def local_lgbm_predict(X_neighbors, y_neighbors, X_query):
    md = lgb.LGBMRegressor(n_estimators=100, num_leaves=15, learning_rate=0.05,
                           min_child_samples=5, objective='regression',
                           n_jobs=1, random_state=42, verbose=-1)
    md.fit(X_neighbors, y_neighbors)
    return float(md.predict(X_query.reshape(1, -1))[0])


def main():
    print("=== nb289: Test-time fine-tune (Tanimoto-neighborhood LGBM) ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv('data/raw/pxr-challenge_TEST_BLINDED.csv')
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()

    # Papyrus PXR only
    pap = pd.read_parquet(DATA_EXTERNAL / "papyrus_pxr_nr.parquet")
    pap = pap[pap['target_name'] == 'PXR'].dropna(subset=['std_smiles', 'pec50']).copy()
    pap_smi = pap['std_smiles'].tolist()
    pap_y = pap['pec50'].values.astype(np.float64)
    print(f"Train: {len(smiles_tr)}, Papyrus PXR: {len(pap_smi)}, Test: {len(smiles_te)}")

    # Combined pool
    pool_smi = smiles_tr + pap_smi
    pool_y = np.concatenate([y_tr, pap_y])

    print("Computing FPs...")
    fps_pool = fps(pool_smi)
    fps_te = fps(smiles_te)
    X_pool = fp_to_arr(fps_pool)
    X_te = fp_to_arr(fps_te)
    print(f"Pool FP matrix: {X_pool.shape}")

    K = 200
    te_preds = np.full(len(smiles_te), float(y_tr.mean()))
    t0 = time.time()
    for i, fp_q in enumerate(fps_te):
        if fp_q is None: continue
        sims = np.array(BulkTanimotoSimilarity(fp_q, fps_pool))
        top_idx = np.argsort(sims)[::-1][:K]
        try:
            te_preds[i] = local_lgbm_predict(X_pool[top_idx], pool_y[top_idx], X_te[i])
        except Exception as e:
            te_preds[i] = float(pool_y[top_idx[:5]].mean())
        if (i + 1) % 50 == 0:
            print(f"  test {i+1}/{len(smiles_te)} ({(time.time()-t0):.1f}s)")
    print(f"Test preds done in {time.time()-t0:.1f}s. mean={te_preds.mean():.3f} std={te_preds.std():.3f}")
    np.save(DATA_PROCESSED / "te_nb289_test_time_finetune.npy", te_preds)

    # OOF: leave-scaffold-out, but for speed limit to a subsample and broadcast mean elsewhere
    print("\nGenerating OOF via leave-scaffold-out (subsampled for speed)...")
    scaffolds = tr['scaffold'].tolist()
    scaf_to_idx = {}
    for i, s in enumerate(scaffolds):
        scaf_to_idx.setdefault(s, []).append(i)
    fps_tr = fps_pool[:len(smiles_tr)]
    X_tr_arr = X_pool[:len(smiles_tr)]

    oof = np.full(len(y_tr), float(y_tr.mean()))
    # Process a random subsample of train (budget ~10 min); fill rest with mean placeholder
    rng = np.random.default_rng(7)
    budget = 800
    sample_idx = rng.choice(len(y_tr), min(budget, len(y_tr)), replace=False)
    t0 = time.time()
    for cnt, i in enumerate(sample_idx):
        if fps_tr[i] is None: continue
        # Mask same-scaffold indices in TRAIN portion of pool
        mask_idx = set(scaf_to_idx[scaffolds[i]])
        keep_mask = np.ones(len(pool_smi), dtype=bool)
        for j in mask_idx: keep_mask[j] = False
        sims = np.array(BulkTanimotoSimilarity(fps_tr[i], fps_pool))
        sims[~keep_mask] = -1
        top_idx = np.argsort(sims)[::-1][:K]
        try:
            oof[i] = local_lgbm_predict(X_pool[top_idx], pool_y[top_idx], X_tr_arr[i])
        except Exception:
            oof[i] = float(pool_y[top_idx[:5]].mean())
        if (cnt + 1) % 100 == 0:
            print(f"  oof {cnt+1}/{len(sample_idx)} ({(time.time()-t0):.1f}s)")
    print(f"OOF subsample done in {time.time()-t0:.1f}s")
    np.save(DATA_PROCESSED / "oof_nb289_test_time_finetune.npy", oof)
    # Report on the *subsampled* indices (where OOF is real)
    r_sub = rae(y_tr[sample_idx], oof[sample_idx])
    sp_sub, _ = spearmanr(y_tr[sample_idx], oof[sample_idx])
    print(f"Sub-OOF RAE on {len(sample_idx)} compounds: {r_sub:.4f}  Spearman={sp_sub:.4f}")

    # SLSQP 5-way (note: oof will be largely placeholder -> low weight expected)
    print("\n=== 5-way SLSQP with nb289 ===")
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
            r = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'ftol': 1e-9, 'maxiter': 200})
            if best is None or r.fun < best.fun: best = r
        print(f"SLSQP OOF: {best.fun:.4f}  weights={np.round(best.x, 4)}")
    except Exception as e:
        print(f"SLSQP skipped: {e}")


if __name__ == "__main__":
    main()
