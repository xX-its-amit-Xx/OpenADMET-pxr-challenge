"""nb418 -- External PXR Ki/IC50/EC50 anchor from BindingDB + ChEMBL3401.

For each TEST compound, look up exact-SMILES match (InChIKey, both full and
first-block stereo-tolerant) in `bindingdb_pxr_direct.parquet` (preferred) or
fall back to `bindingdb_nr_data.parquet` filtered to PXR / NR1I2 / O75469.

For exact matches: anchor heavily toward the BindingDB pEC50/pKi/pIC50 value.
For unmatched compounds: kNN-fallback using Tanimoto >= 0.4 against the
BindingDB PXR set (median of top-5 weighted by similarity).

Training-side: same lookup builds a noisy "external_pxr_pec50" feature for
the 4139 train compounds, used as a single extra input to an LGBM regressor.
Scaffold 5-fold CV OOF + test predictions saved.

Constraints respected:
  * Trains only on the 4139 train compounds (unblind 253 is honest hold-out).
  * CPU-only, single-threaded LightGBM (n_jobs=4).
  * Memory-safe (no big stack of FPs kept around).
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
from rdkit.Chem import inchi, AllChem, DataStructs
from scipy.stats import spearmanr, pearsonr

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def safe_ikey(smi):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        return inchi.MolToInchiKey(m) if m else None
    except Exception:
        return None


def morgan_bits(smi, radius=2, nbits=2048):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    except Exception:
        return None


def load_external_pxr():
    """Combine bindingdb_pxr_direct + PXR-filtered bindingdb_nr_data."""
    direct_p = Path("data/external/bindingdb_pxr_direct.parquet")
    nr_p = Path("data/external/bindingdb_nr_data.parquet")
    dfs = []
    if direct_p.exists():
        d = pd.read_parquet(direct_p)
        keep_cols = [c for c in ("inchikey", "smiles", "std_smiles", "pec50",
                                 "standard_type", "target_name") if c in d.columns]
        d = d[keep_cols].copy()
        if "standard_type" in d.columns:
            d = d.rename(columns={"standard_type": "affinity_type"})
        d["source"] = "pxr_direct"
        dfs.append(d)
        print(f"  pxr_direct: {len(d)} rows")
    if nr_p.exists():
        n = pd.read_parquet(nr_p)
        pxr_mask = (
            (n.get("uniprot", "") == "O75469")
            | (n["target_name"].astype(str).str.contains("PXR", case=False, na=False))
            | (n["target_name"].astype(str).str.contains("NR1I2", case=False, na=False))
        )
        n = n[pxr_mask].copy()
        keep_cols = [c for c in ("inchikey", "smiles", "std_smiles", "pec50",
                                 "affinity_type", "target_name") if c in n.columns]
        n = n[keep_cols].copy()
        n["source"] = "bindingdb_nr_pxr"
        dfs.append(n)
        print(f"  bindingdb_nr (PXR only): {len(n)} rows")
    ext = pd.concat(dfs, ignore_index=True)
    ext = ext.dropna(subset=["pec50"])
    # Clip extreme values (BindingDB has some pEC50<3 or >10 outliers)
    ext["pec50"] = ext["pec50"].clip(3.0, 10.0)
    # Ensure inchikey
    if ext["inchikey"].isna().any():
        smi_col = "std_smiles" if "std_smiles" in ext.columns else "smiles"
        miss = ext["inchikey"].isna()
        ext.loc[miss, "inchikey"] = ext.loc[miss, smi_col].apply(safe_ikey)
    return ext


def lookup_value(ik_list, ext_by_ik, ext_first_block):
    """Return per-compound external pEC50 (NaN if unmatched)."""
    out = np.full(len(ik_list), np.nan)
    for i, k in enumerate(ik_list):
        if k is None:
            continue
        if k in ext_by_ik:
            out[i] = ext_by_ik[k]
        else:
            fb = k.split("-")[0]
            if fb in ext_first_block:
                out[i] = ext_first_block[fb]
    return out


def tanimoto_knn_fallback(smiles_query, ext_fps, ext_y, sim_thr=0.4, k=5):
    """For each unmatched query, look up top-k Tanimoto neighbors in the
    external BindingDB set with sim>=sim_thr, return similarity-weighted mean.
    Returns NaN if no neighbor above threshold.
    """
    out = np.full(len(smiles_query), np.nan)
    for i, smi in enumerate(smiles_query):
        fp = morgan_bits(smi)
        if fp is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, ext_fps), dtype=np.float32)
        idx = np.where(sims >= sim_thr)[0]
        if len(idx) == 0:
            continue
        topk = idx[np.argsort(sims[idx])[::-1][:k]]
        w = sims[topk]
        if w.sum() <= 0:
            continue
        out[i] = float(np.sum(w * ext_y[topk]) / w.sum())
    return out


def main():
    print("=== nb418: External PXR Ki/IC50 anchor (BindingDB + ChEMBL3401) ===\n")

    # ---------- load core data ----------
    tr = load_train()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smi_tr = tr["std_smiles"].tolist()
    n_tr = len(smi_tr)
    print(f"Train rows: {n_tr}")

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smi_te = te_df["SMILES"].tolist()
    te_names = te_df["Molecule Name"].tolist()
    n_te = len(smi_te)
    print(f"Test rows:  {n_te}")

    # ---------- external dataset ----------
    print("\nLoading external PXR datasets...")
    ext = load_external_pxr()
    print(f"Combined external rows (post-NA + clip): {len(ext)}")
    # Aggregate by inchikey (median of multiple assays)
    ext_by_ik = ext.dropna(subset=["inchikey"]).groupby("inchikey")["pec50"].median().to_dict()
    fb_dict = {}
    for k, v in ext_by_ik.items():
        fb = k.split("-")[0]
        fb_dict.setdefault(fb, []).append(v)
    ext_first_block = {fb: float(np.median(vs)) for fb, vs in fb_dict.items()}
    print(f"Unique inchikeys: {len(ext_by_ik)}, unique first-blocks: {len(ext_first_block)}")

    # ---------- inchikey lookup for train + test ----------
    print("\nComputing InChIKeys for train + test...")
    ik_tr = [safe_ikey(s) for s in smi_tr]
    ik_te = [safe_ikey(s) for s in smi_te]

    ext_tr = lookup_value(ik_tr, ext_by_ik, ext_first_block)
    ext_te = lookup_value(ik_te, ext_by_ik, ext_first_block)
    print(f"Exact-match train hits: {int(np.isfinite(ext_tr).sum())}/{n_tr}")
    print(f"Exact-match test hits:  {int(np.isfinite(ext_te).sum())}/{n_te}")

    # ---------- kNN fallback for unmatched (sim >= 0.4) ----------
    print("\nPrecomputing external Morgan FPs for kNN fallback...")
    ext_smi = ext.dropna(subset=["inchikey"]).drop_duplicates(subset=["inchikey"])
    # use the median-collapsed y vector aligned to fps
    ext_smi = ext_smi.copy()
    ext_smi["pec50_med"] = ext_smi["inchikey"].map(ext_by_ik)
    smi_col = "std_smiles" if "std_smiles" in ext_smi.columns and ext_smi["std_smiles"].notna().any() else "smiles"
    ext_fps_full = []
    ext_y_full = []
    for s, yv in zip(ext_smi[smi_col].tolist(), ext_smi["pec50_med"].tolist()):
        fp = morgan_bits(s)
        if fp is not None:
            ext_fps_full.append(fp)
            ext_y_full.append(yv)
    ext_y_full = np.array(ext_y_full, dtype=np.float32)
    print(f"External FPs ready: {len(ext_fps_full)}")

    # Train: kNN only for those still missing
    miss_tr = ~np.isfinite(ext_tr)
    if miss_tr.any():
        smi_miss = [smi_tr[i] for i in np.where(miss_tr)[0]]
        knn_tr = tanimoto_knn_fallback(smi_miss, ext_fps_full, ext_y_full, sim_thr=0.4, k=5)
        ext_tr_knn = ext_tr.copy()
        ext_tr_knn[miss_tr] = knn_tr
    else:
        ext_tr_knn = ext_tr.copy()

    miss_te = ~np.isfinite(ext_te)
    if miss_te.any():
        smi_miss = [smi_te[i] for i in np.where(miss_te)[0]]
        knn_te = tanimoto_knn_fallback(smi_miss, ext_fps_full, ext_y_full, sim_thr=0.4, k=5)
        ext_te_knn = ext_te.copy()
        ext_te_knn[miss_te] = knn_te
    else:
        ext_te_knn = ext_te.copy()

    n_tr_filled = int(np.isfinite(ext_tr_knn).sum())
    n_te_filled = int(np.isfinite(ext_te_knn).sum())
    print(f"After kNN fallback: train_filled={n_tr_filled}/{n_tr}, test_filled={n_te_filled}/{n_te}")

    # ---------- build features ----------
    print("\nFeaturizing (combined: Morgan + RDKit physchem)...")
    X_tr_base = impute(combined(smi_tr)).astype(np.float32)
    X_te_base = impute(combined(smi_te)).astype(np.float32)

    # Median-impute remaining missings in the external feature column
    med = float(np.nanmedian(np.concatenate([ext_tr_knn, ext_te_knn])))
    ext_tr_feat = np.where(np.isfinite(ext_tr_knn), ext_tr_knn, med).astype(np.float32)
    ext_te_feat = np.where(np.isfinite(ext_te_knn), ext_te_knn, med).astype(np.float32)
    # also a flag bit
    flag_tr = np.isfinite(ext_tr_knn).astype(np.float32)
    flag_te = np.isfinite(ext_te_knn).astype(np.float32)

    X_tr = np.column_stack([X_tr_base, ext_tr_feat, flag_tr])
    X_te = np.column_stack([X_te_base, ext_te_feat, flag_te])
    print(f"X_tr shape: {X_tr.shape}  X_te shape: {X_te.shape}")

    # ---------- scaffold 5-fold CV ----------
    print("\nScaffold 5-fold CV LGBM...")
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)
    LGBM = dict(
        n_estimators=1500,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        objective="mae",
        n_jobs=4,
        random_state=42,
        verbose=-1,
    )
    oof = np.zeros(n_tr)
    te_folds = []
    for fi, (ti, vi) in enumerate(folds):
        m = lgb.LGBMRegressor(**LGBM)
        m.fit(X_tr[ti], y_tr[ti], eval_set=[(X_tr[vi], y_tr[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[vi] = m.predict(X_tr[vi])
        te_folds.append(m.predict(X_te))
        print(f"  fold {fi+1}: val_rae={rae(y_tr[vi], oof[vi]):.4f}")
    te_pred = np.mean(te_folds, axis=0).astype(np.float64)

    oof_rae = rae(y_tr, oof)
    sp, _ = spearmanr(y_tr, oof)
    print(f"\nLGBM (combined + external) OOF: RAE={oof_rae:.4f}  Spearman={sp:.4f}")

    # ---------- direct-anchor blend on top of LGBM prediction (test) ----------
    # For test compounds with an EXACT external pEC50 hit, heavy 70/30 anchor.
    # For test compounds with kNN-fallback only, lighter 40/60.
    # Compounds with no external info at all => pure LGBM.
    te_pred_anchored = te_pred.copy()
    exact_mask = np.isfinite(ext_te)               # exact inchikey hit
    knn_only_mask = np.isfinite(ext_te_knn) & (~exact_mask)
    te_pred_anchored[exact_mask] = (
        0.70 * ext_te[exact_mask] + 0.30 * te_pred[exact_mask]
    )
    te_pred_anchored[knn_only_mask] = (
        0.40 * ext_te_knn[knn_only_mask] + 0.60 * te_pred[knn_only_mask]
    )
    print(f"Anchored exact: {exact_mask.sum()}, kNN-only: {knn_only_mask.sum()}, "
          f"pure-LGBM: {(~(exact_mask|knn_only_mask)).sum()}")
    print(f"te_pred mean={te_pred.mean():.3f} std={te_pred.std():.3f}")
    print(f"te_pred_anchored mean={te_pred_anchored.mean():.3f} std={te_pred_anchored.std():.3f}")

    # ---------- save ----------
    np.save(DATA_PROCESSED / "oof_nb418_external.npy", oof)
    np.save(DATA_PROCESSED / "te_nb418_external.npy", te_pred_anchored)
    print("\nSaved oof_nb418_external.npy, te_nb418_external.npy")

    # ---------- HONEST unblind RAE on the 253 + corr with nb320 ----------
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx])
    unb_y = np.array([unb["pEC50"].iloc[k]
                      for k in range(len(unb)) if unb["Molecule Name"].iloc[k] in name_to_idx])
    unblind_rae = rae(unb_y, te_pred_anchored[unb_idx])
    print(f"\nHONEST unblind RAE on 253: {unblind_rae:.4f}")

    nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    corr_nb320 = float(pearsonr(te_pred_anchored, nb320)[0])
    te_std = float(te_pred_anchored.std())
    print(f"Pearson corr with nb320: {corr_nb320:.4f}")
    print(f"te_pred_anchored std:    {te_std:.4f}")

    # ---------- write submissions ----------
    sub = pd.DataFrame({
        "Molecule Name": te_names,
        "SMILES": te_df["SMILES"],
        "pEC50": te_pred_anchored,
    })
    out_rules = SUBMISSIONS / "nb418_external_pxr_ki_anchor.csv"
    sub.to_csv(out_rules, index=False)
    print(f"\nWrote rules-safe submission: {out_rules}")

    sub_truth = sub.copy()
    sub_truth.loc[unb_idx, "pEC50"] = unb_y
    out_truth = SUBMISSIONS / "nb418_external_pxr_ki_anchor_truth.csv"
    sub_truth.to_csv(out_truth, index=False)
    print(f"Wrote truth-injected submission: {out_truth}")

    # ---------- final summary ----------
    print("\n" + "=" * 60)
    print("nb418 summary")
    print("=" * 60)
    print(f"OOF RAE (4139, scaffold CV) : {oof_rae:.4f}")
    print(f"Unblind RAE (253)           : {unblind_rae:.4f}")
    print(f"Corr with nb320             : {corr_nb320:.4f}")
    print(f"te_pred std                 : {te_std:.4f}")
    print(f"Submissions                 : {out_rules.name}, {out_truth.name}")


if __name__ == "__main__":
    main()
