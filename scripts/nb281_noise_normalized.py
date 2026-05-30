"""nb281 -- Noise-normalized + Spearman-optimized models.

User insight: 0.2-0.3 of LB RAE is just assay noise. Optimizing for raw RAE
overfits noise. Switch objective:
1. Precision-weighted LGBM: sample_weight = 1/SE^2 (low-SE compounds get
   higher weight)
2. Relabel with Bayesian shrinkage: smooth noisy y toward neighborhood mean
3. LambdaRank-style Spearman optimization (LGBM ranker objective)

Compare OOF + LB submission candidates.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def morgan(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def main():
    print("=== nb281: Noise-normalized + Spearman models ===\n")
    tr_csv = pd.read_csv('data/raw/pxr-challenge_TRAIN.csv')
    se_col = 'pEC50_std.error (-log10(molarity))'

    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv('data/raw/pxr-challenge_TEST_BLINDED.csv')
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    te_names = te_df['Molecule Name'].tolist()
    sm = dict(zip(te_df['Molecule Name'], te_df['SMILES']))

    # Get SE aligned to train order
    name_to_se = dict(zip(tr_csv['Molecule Name'], tr_csv[se_col]))
    tr_names = tr['name'].tolist() if 'name' in tr.columns else tr['Molecule Name'].tolist()
    se = np.array([name_to_se.get(n, 0.2) for n in tr_names])
    se = np.clip(se, 0.05, 1.0)  # avoid div by ~0
    print(f"SE distribution: min={se.min():.3f} max={se.max():.3f} median={np.median(se):.3f}")

    # ============================
    # Approach 1: Precision-weighted LGBM
    # ============================
    print("\n=== Approach 1: Precision-weighted LGBM (w = 1/SE^2) ===")
    weights = 1.0 / (se ** 2)
    weights = weights / weights.mean()  # normalize
    print(f"Weights: min={weights.min():.3f} max={weights.max():.3f} median={np.median(weights):.3f}")

    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, objective='mae', n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)

    for tag, w in [('unweighted', np.ones(len(y_tr))), ('precision', weights)]:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X_tr[ti], y_tr[ti], sample_weight=w[ti],
                   eval_set=[(X_tr[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X_tr[vi])
            te_preds.append(md.predict(X_te))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        # Noise-normalized error: |y-pred|/SE
        nn_err = (np.abs(y_tr - oof) / se).mean()
        # Spearman
        sp, _ = spearmanr(y_tr, oof)
        print(f"  {tag}: OOF RAE={r:.4f}  noise-norm-MAE={nn_err:.3f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
        np.save(DATA_PROCESSED / f"oof_nb281_{tag}.npy", oof)
        np.save(DATA_PROCESSED / f"te_nb281_{tag}.npy", te_pred)

    # ============================
    # Approach 2: Bayesian shrinkage relabeling
    # ============================
    print("\n=== Approach 2: Bayesian shrinkage relabel ===")
    # For each compound, shrink y toward neighborhood mean by uncertainty
    print("  Computing FPs and neighborhoods...")
    fps = morgan(smiles_tr)
    # For each compound, find 5 nearest neighbors and compute mean
    y_relabeled = y_tr.copy()
    n_relabeled = 0
    K = 5
    for i in range(len(y_tr)):
        if fps[i] is None: continue
        sims = np.array(BulkTanimotoSimilarity(fps[i], fps))
        sims[i] = -1  # exclude self
        top_idx = np.argsort(sims)[::-1][:K]
        top_y = y_tr[top_idx]
        top_sims = sims[top_idx]
        if top_sims[0] < 0.4: continue  # require some neighborhood
        nb_mean = (top_sims * top_y).sum() / top_sims.sum()
        # Shrinkage factor = SE / (SE + neighborhood_std)
        # Higher SE → shrink more toward nb_mean
        nb_std = top_y.std()
        shrink = se[i] / (se[i] + max(nb_std, 0.1))
        y_relabeled[i] = (1 - shrink) * y_tr[i] + shrink * nb_mean
        if abs(y_relabeled[i] - y_tr[i]) > 0.01:
            n_relabeled += 1
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(y_tr)}")
    print(f"  Relabeled (|shift|>0.01): {n_relabeled}/{len(y_tr)}")
    print(f"  Mean shrinkage: {(y_relabeled - y_tr).mean():.3f}, std: {(y_relabeled - y_tr).std():.3f}")

    # Train on relabeled
    print("  Training LGBM on relabeled y...")
    oof = np.zeros(len(y_tr))
    te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr[ti], y_relabeled[ti],
               eval_set=[(X_tr[vi], y_tr[vi])],  # eval against original
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        te_preds.append(md.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    r_orig = rae(y_tr, oof)
    sp, _ = spearmanr(y_tr, oof)
    print(f"  Relabeled OOF RAE (vs original): {r_orig:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb281_relabeled.npy", oof)
    np.save(DATA_PROCESSED / "te_nb281_relabeled.npy", te_pred)

    # ============================
    # Approach 3: LambdaRank (Spearman-optimizing)
    # ============================
    print("\n=== Approach 3: LambdaRank ===")
    # LightGBM LambdaRank requires query groups; use scaffold as group
    LGBM_RANK = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                     min_child_samples=10, objective='lambdarank', n_jobs=4, random_state=42, verbose=-1,
                     metric='ndcg')
    # Need integer relevance labels for ranker — discretize y into 10 bins
    y_bins = np.digitize(y_tr, np.quantile(y_tr, np.arange(0.1, 1.0, 0.1)))  # 0..9
    oof_rank = np.zeros(len(y_tr))
    te_rank_preds = []
    for ti, vi in folds:
        # LGBMRanker needs group sizes
        ti_arr = np.array(ti)
        n_train = len(ti_arr)
        group_train = [n_train]  # one group of all train samples
        try:
            md = lgb.LGBMRanker(**LGBM_RANK)
            md.fit(X_tr[ti_arr], y_bins[ti_arr], group=group_train)
            oof_rank[vi] = md.predict(X_tr[vi])
            te_rank_preds.append(md.predict(X_te))
        except Exception as e:
            print(f"  LambdaRank fold err: {e}")
            oof_rank[vi] = y_tr.mean()
            te_rank_preds.append(np.full(len(smiles_te), y_tr.mean()))
    te_rank = np.mean(te_rank_preds, axis=0)
    # LambdaRank produces scores (not pec50); compute Spearman + map ranks back
    sp, _ = spearmanr(y_tr, oof_rank)
    # Convert ranks to pec50: map sorted scores to sorted train pec50
    order = np.argsort(oof_rank)
    pec_sorted = np.sort(y_tr)
    oof_mapped = np.zeros(len(y_tr))
    for rank_i, idx in enumerate(order):
        oof_mapped[idx] = pec_sorted[rank_i]
    te_order = np.argsort(te_rank)
    te_mapped = np.zeros(len(smiles_te))
    pec_q = np.quantile(y_tr, np.arange(len(smiles_te)) / len(smiles_te))
    for rank_i, idx in enumerate(te_order):
        te_mapped[idx] = pec_q[rank_i]
    r_mapped = rae(y_tr, oof_mapped)
    print(f"  LambdaRank: Spearman={sp:.4f}, OOF RAE after rank-mapping={r_mapped:.4f}")
    np.save(DATA_PROCESSED / "oof_nb281_lambdarank.npy", oof_mapped)
    np.save(DATA_PROCESSED / "te_nb281_lambdarank.npy", te_mapped)

    # 5-way SLSQP with best of these
    print("\n=== 5-way SLSQP w/ nb281_precision ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    nb281_p = np.load(DATA_PROCESSED / "oof_nb281_precision.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, nb281_p])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way SLSQP OOF: {best.fun:.4f}, nb281_precision weight={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
