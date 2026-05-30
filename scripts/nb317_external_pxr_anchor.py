"""nb317 -- External PXR direct-anchor + multi-task features.

Builds on nb316's hidden data fetch (5 sources, 441 train/test InChIKey
overlaps). For test compounds that have a measured value in PubChem
AID 1346985 (Tox21 hPXR activation), AID 720659 (NCATS PXR), AID 1346984
(CYP3A4 induction-via-PXR), OR Octant CYP3A4 — anchor predictions toward
those measurements.

Additional: use each external dataset as a MULTI-TASK feature column
(LGBM-predicted pec50 from each external source as input to final PXR LGBM).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import inchi
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def ikey(smi):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        return inchi.MolToInchiKey(m) if m else None
    except: return None


SOURCES = {
    'tox21_pxr_aid1346985': 'data/external/pubchem_aid_1346985_tox21_pxr/aid_1346985.parquet',
    'tox21_cyp3a4_aid1346984': 'data/external/pubchem_aid_1346984_tox21_cyp3a4/aid_1346984.parquet',
    'ncats_pxr_aid720659': 'data/external/pubchem_aid_720659_ncats_pxr/aid_720659.parquet',
    'octant_cyp3a4': 'data/external/openadmet_octant_cyp/cyp3a4_inhibition.parquet',
    'expansionrx': 'data/external/openadmet_expansionrx/admet_panel.parquet',
}


def load_source(path):
    p = Path(path)
    if not p.exists():
        print(f"  MISSING: {path}")
        return None
    df = pd.read_parquet(p)
    if 'inchikey' not in df.columns:
        for sc in ('std_smiles', 'smiles', 'SMILES'):
            if sc in df.columns:
                df['inchikey'] = df[sc].apply(ikey)
                break
    return df


def main():
    print("=== nb317: external PXR direct anchors + multi-task features ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()
    te_names = te_df['Molecule Name'].tolist()

    print("Computing InChIKeys for train + test...")
    ik_tr = [ikey(s) for s in smiles_tr]
    ik_te = [ikey(s) for s in smiles_te]

    # ============================
    # Per-source overlap analysis
    # ============================
    anchor_te = np.full(len(smiles_te), np.nan)
    n_te_hits = 0
    src_test_anchors = {}
    src_train_features = {}  # source -> (N_tr, ) of predicted/measured value where available

    for src_name, path in SOURCES.items():
        df = load_source(path)
        if df is None: continue
        df = df.dropna(subset=['inchikey'])
        if 'pec50' not in df.columns:
            print(f"  {src_name}: no pec50, skip"); continue
        src_y_by_ik = df.dropna(subset=['pec50']).groupby('inchikey')['pec50'].median().to_dict()
        # First-block (stereo-tolerant) index too
        first_block = {}
        for k, v in src_y_by_ik.items():
            fb = k.split("-")[0]
            first_block.setdefault(fb, []).append(v)
        first_block_med = {fb: float(np.median(vs)) for fb, vs in first_block.items()}

        # Test overlap
        te_lookup = np.full(len(smiles_te), np.nan)
        for i, k in enumerate(ik_te):
            if k is None: continue
            if k in src_y_by_ik:
                te_lookup[i] = src_y_by_ik[k]
            elif k.split("-")[0] in first_block_med:
                te_lookup[i] = first_block_med[k.split("-")[0]]
        # Train overlap (for multitask featurization)
        tr_lookup = np.full(len(y), np.nan)
        for i, k in enumerate(ik_tr):
            if k is None: continue
            if k in src_y_by_ik:
                tr_lookup[i] = src_y_by_ik[k]
            elif k.split("-")[0] in first_block_med:
                tr_lookup[i] = first_block_med[k.split("-")[0]]

        n_tr_hit = int(np.isfinite(tr_lookup).sum())
        n_te_hit = int(np.isfinite(te_lookup).sum())
        print(f"  {src_name}: test_hits={n_te_hit}, train_hits={n_tr_hit}, src_size={len(df)}")
        src_test_anchors[src_name] = te_lookup
        src_train_features[src_name] = tr_lookup

    # ============================
    # Direct test anchor: average per-compound across source hits
    # ============================
    print("\nBuilding direct test anchors (average across source hits)...")
    anchor_count = np.zeros(len(smiles_te), dtype=int)
    anchor_sum = np.zeros(len(smiles_te))
    for src_name, vals in src_test_anchors.items():
        # Only use sources that ARE pec50 (PXR-direct: skip CYP3A4 induction signal — different mechanism)
        if 'tox21_pxr' in src_name or 'ncats_pxr' in src_name:
            mask = np.isfinite(vals)
            anchor_count += mask.astype(int)
            anchor_sum[mask] += vals[mask]
    anchor_te = np.where(anchor_count > 0, anchor_sum / np.maximum(anchor_count, 1), np.nan)
    n_anchored = int(np.isfinite(anchor_te).sum())
    print(f"Test compounds with PXR-direct external anchor: {n_anchored}/{len(smiles_te)}")

    # Blend anchor with nb302 (heavy anchor weight when external value present)
    nb302_te = np.load(DATA_PROCESSED / "te_nb302_full_pool.npy")
    nb302_oof = np.load(DATA_PROCESSED / "oof_nb302_full_pool.npy")
    final_te = nb302_te.copy()
    for i in range(len(smiles_te)):
        if np.isfinite(anchor_te[i]):
            final_te[i] = 0.6 * anchor_te[i] + 0.4 * nb302_te[i]

    print(f"\nFinal te: mean={final_te.mean():.3f}  std={final_te.std():.3f}")
    print(f"Anchored: {n_anchored}, non-anchored: {len(smiles_te) - n_anchored}")

    # ============================
    # Multi-task LGBM: use src_train_features as input columns
    # ============================
    print("\nMulti-task LGBM (combined + external pec50 columns)...")
    X_tr_base = impute(combined(smiles_tr)).astype(np.float32)
    X_te_base = impute(combined(smiles_te)).astype(np.float32)

    extra_tr = []; extra_te = []
    for src_name, vals_tr in src_train_features.items():
        vals_te = src_test_anchors[src_name]
        # Median-impute missing values per source
        med = np.nanmedian(vals_tr) if np.isfinite(vals_tr).any() else 5.0
        extra_tr.append(np.where(np.isfinite(vals_tr), vals_tr, med))
        extra_te.append(np.where(np.isfinite(vals_te), vals_te, med))
    if extra_tr:
        X_tr = np.column_stack([X_tr_base] + extra_tr)
        X_te = np.column_stack([X_te_base] + extra_te)
    else:
        X_tr = X_tr_base; X_te = X_te_base
    print(f"Feature dim: {X_tr.shape[1]}")

    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=10, objective='mae',
                n_jobs=4, random_state=42, verbose=-1)
    oof = np.zeros(len(y))
    te_preds_mt = []
    for ti, vi in folds:
        m = lgb.LGBMRegressor(**LGBM)
        m.fit(X_tr[ti], y[ti], eval_set=[(X_tr[vi], y[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = m.predict(X_tr[vi])
        te_preds_mt.append(m.predict(X_te))
    te_pred_mt = np.mean(te_preds_mt, axis=0)
    r = rae(y, oof)
    sp, _ = spearmanr(y, oof)
    print(f"\nMultitask LGBM OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred_mt.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb317_external_mt.npy", oof)
    np.save(DATA_PROCESSED / "te_nb317_external_mt.npy", te_pred_mt)
    np.save(DATA_PROCESSED / "te_nb317_anchored.npy", final_te)

    # Save the anchored submission
    sub = pd.DataFrame({
        'Molecule Name': te_names,
        'SMILES': te_df['SMILES'],
        'pEC50': final_te,
    })
    out = SUBMISSIONS / "nb317_external_anchored.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out}")

    # 5-way SLSQP with multitask
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}, weight(nb317_mt)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
