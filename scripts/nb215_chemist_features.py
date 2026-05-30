"""nb215 -- Medicinal chemist evidence features + LGBM.

Per-compound features encoding "analog evidence":
- Top-1/5/20 PXR neighbor pEC50 stats (mean/std/min/max/weighted-mean)
- Top-1/5/20 counter-assay neighbor pEC50 stats (proxy for assay-artifact risk)
- Counter-assay direct pEC50 (real for ~70% of train, predicted for rest + test)
- Scaffold density features (count + mean pEC50 of same-scaffold training compounds)
- Local cliff risk (top-5 neighbor pEC50 std × inverse top-1 similarity)

Combined with morgan + rdkit features, 5-fold scaffold CV LGBM.
Goal: produce a model with predictions diverse from existing pool.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import lightgbm as lgb

from pxr.data import load_train, load_test, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.featurize import combined as feat_combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
COLLAPSE_THRESH = 0.58
PREV_BEST = 0.296172

t0 = time.time()


def _stats_per_query(query_fps, neighbor_fps, neighbor_pec, k_list=(1, 5, 20),
                     exclude_self=False):
    """Returns (n_query, n_features) ndarray. 7 stats per k."""
    n_q = len(query_fps)
    cols_per_k = 7
    out = np.zeros((n_q, len(k_list) * cols_per_k), dtype=np.float64)
    for i, qfp in enumerate(query_fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, neighbor_fps),
                        dtype=np.float64)
        if exclude_self:
            sims[i] = -1.0
        order = np.argsort(sims)[::-1]
        for j, k in enumerate(k_list):
            top_idx = order[:k]
            top_sims = sims[top_idx]
            top_pec = neighbor_pec[top_idx]
            base = j * cols_per_k
            out[i, base + 0] = top_sims[0] if k >= 1 else 0.0  # top-1 sim
            out[i, base + 1] = top_sims.mean()
            out[i, base + 2] = top_pec.mean()
            out[i, base + 3] = top_pec.std() if k > 1 else 0.0
            out[i, base + 4] = top_pec.min()
            out[i, base + 5] = top_pec.max()
            sw = top_sims.sum()
            out[i, base + 6] = (np.average(top_pec, weights=top_sims)
                                if sw > 1e-9 else top_pec.mean())
    return out


def main():
    print("=== nb215: Medicinal chemist evidence features + LGBM ===\n", flush=True)

    tr_df = load_train()
    te_df = load_test()
    cn_df = load_counter()
    print(f"Train: {len(tr_df)}, Test: {len(te_df)}, Counter: {len(cn_df)}\n", flush=True)

    y_tr = tr_df["pec50"].values.astype(np.float64)
    n_tr = len(tr_df)

    scaffolds = tr_df["smiles"].map(bemis_murcko).tolist()
    test_scaffolds = te_df["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ECFP4 fingerprints (RDKit objects for similarity)
    print("Computing Morgan FPs...", flush=True)
    tr_mols = [Chem.MolFromSmiles(s) for s in tr_df["smiles"]]
    te_mols = [Chem.MolFromSmiles(s) for s in te_df["smiles"]]
    cn_mols = [Chem.MolFromSmiles(s) for s in cn_df["smiles"]]
    tr_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in tr_mols]
    te_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in te_mols]
    cn_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in cn_mols]
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    cn_pec = cn_df["pec50"].values.astype(np.float64)

    # ----- PXR neighbor evidence (fold-aware for train) -----
    print("\nPXR neighbor evidence...", flush=True)
    nne_tr = np.zeros((n_tr, 21))
    for fi, (tr_idx, va_idx) in enumerate(splits):
        sub_fps = [tr_fps[i] for i in tr_idx]
        sub_pec = y_tr[tr_idx]
        va_fps = [tr_fps[i] for i in va_idx]
        nne_tr[va_idx] = _stats_per_query(va_fps, sub_fps, sub_pec)
        print(f"  fold {fi+1}/{N_FOLDS} ({time.time()-t0:.0f}s)", flush=True)
    nne_te = _stats_per_query(te_fps, tr_fps, y_tr)
    print(f"  PXR-NNE done ({time.time()-t0:.0f}s)", flush=True)

    # ----- Counter-assay neighbor evidence (use ALL counter for both tr and te, no fold-leak: counter is a different label) -----
    print("\nCounter-assay neighbor evidence...", flush=True)
    cn_nne_tr = _stats_per_query(tr_fps, cn_fps, cn_pec)
    cn_nne_te = _stats_per_query(te_fps, cn_fps, cn_pec)
    print(f"  counter-NNE done ({time.time()-t0:.0f}s)", flush=True)

    # ----- Scaffold density features -----
    print("\nScaffold density features...", flush=True)
    scaffold_to_pec = {}
    for s, p in zip(scaffolds, y_tr):
        scaffold_to_pec.setdefault(s, []).append(p)

    def scaffold_feats(scaff_list, fold_exclude_idx=None):
        # Optional fold-aware: exclude pec50 of compounds in fold_exclude_idx
        feats = np.zeros((len(scaff_list), 4))
        if fold_exclude_idx is not None:
            excl_set = set(fold_exclude_idx)
            sc_to_p = {}
            for i, (s, p) in enumerate(zip(scaffolds, y_tr)):
                if i in excl_set: continue
                sc_to_p.setdefault(s, []).append(p)
        else:
            sc_to_p = scaffold_to_pec
        for i, s in enumerate(scaff_list):
            ps = sc_to_p.get(s, [])
            feats[i, 0] = len(ps)
            feats[i, 1] = float(np.mean(ps)) if ps else 0.0
            feats[i, 2] = float(np.std(ps)) if len(ps) > 1 else 0.0
            feats[i, 3] = float(np.median(ps)) if ps else 0.0
        return feats

    sc_tr = np.zeros((n_tr, 4))
    for fi, (tr_idx, va_idx) in enumerate(splits):
        sc_va = scaffold_feats([scaffolds[i] for i in va_idx], fold_exclude_idx=set(va_idx))
        sc_tr[va_idx] = sc_va
    sc_te = scaffold_feats(test_scaffolds)
    print(f"  scaffold feats done ({time.time()-t0:.0f}s)", flush=True)

    # ----- Counter-assay direct pEC50 lookup (real for overlap) -----
    print("\nCounter-assay direct pEC50 lookup...", flush=True)
    # Match by SMILES (exact)
    cn_lookup = dict(zip(cn_df["smiles"], cn_df["pec50"]))
    cn_direct_tr = np.array([cn_lookup.get(s, np.nan) for s in tr_df["smiles"]])
    cn_direct_te = np.array([cn_lookup.get(s, np.nan) for s in te_df["smiles"]])
    print(f"  train cn-overlap: {(~np.isnan(cn_direct_tr)).sum()}/{n_tr}", flush=True)
    print(f"  test cn-overlap: {(~np.isnan(cn_direct_te)).sum()}/{len(te_df)}", flush=True)

    # ----- Cliff risk feature: top-5 std × (1 - top-1 sim) -----
    cliff_tr = nne_tr[:, 1*7+3] * (1.0 - nne_tr[:, 0*7+0])
    cliff_te = nne_te[:, 1*7+3] * (1.0 - nne_te[:, 0*7+0])

    # ----- Combine all -----
    print("\nBuilding combined feature matrix...", flush=True)
    X_tr_base = impute(feat_combined(tr_df["smiles"].tolist()))
    X_te_base = impute(feat_combined(te_df["smiles"].tolist()))

    # Stack everything
    X_tr_extra = np.hstack([
        nne_tr, cn_nne_tr, sc_tr,
        cn_direct_tr.reshape(-1, 1),
        cliff_tr.reshape(-1, 1),
    ])
    X_te_extra = np.hstack([
        nne_te, cn_nne_te, sc_te,
        cn_direct_te.reshape(-1, 1),
        cliff_te.reshape(-1, 1),
    ])
    # Impute the extras (cn_direct has NaN for non-overlap)
    X_tr_extra = impute(X_tr_extra)
    X_te_extra = impute(X_te_extra)

    X_tr = np.hstack([X_tr_base, X_tr_extra]).astype(np.float32)
    X_te = np.hstack([X_te_base, X_te_extra]).astype(np.float32)

    print(f"\nFeature matrix shape: train {X_tr.shape}, test {X_te.shape}", flush=True)
    print(f"Extra feature count: {X_tr_extra.shape[1]} ({time.time()-t0:.0f}s)\n", flush=True)

    # ----- LGBM 5-fold CV -----
    print("Training LGBM (5-fold scaffold CV)...", flush=True)
    oof = np.full(n_tr, np.nan)
    test_preds = np.zeros(len(te_df))

    for fi, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=64, learning_rate=0.03,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, objective="regression_l1",
            random_state=SEED, verbose=-1,
        )
        m.fit(
            X_tr[tr_idx], y_tr[tr_idx],
            eval_set=[(X_tr[va_idx], y_tr[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        test_preds += m.predict(X_te) / N_FOLDS
        print(f"  fold {fi+1}/{N_FOLDS}: best_iter={m.best_iteration_}  ({time.time()-t0:.0f}s)", flush=True)

    r = rae(y_tr, oof)
    ratio = test_preds.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***BEATS PREV BEST***" if (ratio >= COLLAPSE_THRESH and r < PREV_BEST) else ""
    print(f"\n=== nb215 OOF: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat} ===", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    out_stem = "nb215_chemist_features"
    np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{out_stem}.npy", test_preds)
    sub = pd.DataFrame({
        "SMILES": te_df["smiles"].values,
        "Molecule Name": te_df["name"].values,
        "pEC50": test_preds,
    })
    sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
    print(f"Saved: {out_stem}", flush=True)


if __name__ == "__main__":
    main()
