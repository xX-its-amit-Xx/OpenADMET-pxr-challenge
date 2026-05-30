"""nb148 -- PXR structural & regulatory context features.

PXR isn't just a ligand binder — it's a transcription factor. Its activation
ripples through a network of regulated genes (CYP3A4, MDR1, etc.) and
cofactor interactions. We pull this context:

  1. PXR-regulated gene set (CYP3A4, CYP2B6, MDR1/ABCB1, BCRP/ABCG2, etc.)
     - For each gene, query ChEMBL bulk for compound activity against it
     - PXR agonists should also modulate these downstream genes' activity

  2. PXR response element (DR-3, ER-6) sequence motifs
     - Use these to find OTHER transcription factors that bind similar motifs
     - Cross-reference their compound modulators in ChEMBL bulk

  3. PXR cofactor interactions (SRC-1, PGC-1α, RIP140)
     - Compounds that affect cofactor proteins may indirectly affect PXR signal

  4. DBD vs LBD conformational coupling — proxy via co-crystal ligand database
     - PXR PDB entries' ligands -> known agonist set
     - Tanimoto similarity to known PXR agonists is already nb220 territory but
       we can extend to include AhR/CAR (other xenosensors with similar DNA elements)

Feasibility check: we have ChEMBL bulk with 6,710 targets. Many of these
are PXR-regulated genes or related TFs. Extract that subset as a 'PXR
regulatory network' feature stack — sim-weighted activity profile against
the network instead of all 6,710 targets.

This is a curated subset of nb146 with biological interpretability.
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
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)

# Curated PXR regulatory network (PXR target genes + related transcription factors)
# Sources: CTD, GeneCards, KEGG xenobiotic metabolism pathway, ChEMBL target IDs
PXR_NETWORK = {
    # PXR target genes (genes whose expression PXR regulates)
    "CHEMBL340":    "CYP3A4",      # PXR's main downstream
    "CHEMBL3577":   "CYP2C9",
    "CHEMBL2392":   "CYP2C19",
    "CHEMBL1832":   "CYP1A2",
    "CHEMBL4302530":"MDR1/ABCB1",  # PXR-induced transporter
    "CHEMBL5393":   "BCRP/ABCG2",
    "CHEMBL2789":   "MRP2/ABCC2",
    "CHEMBL3622":   "OATP1B1",
    "CHEMBL2095203":"UGT1A1",       # phase 2 metabolism
    "CHEMBL2424":   "SULT",          # sulfotransferase
    # Related nuclear receptor xenosensors (similar DNA-binding motifs)
    "CHEMBL3401485":"PXR (NR1I2)",
    "CHEMBL1075322":"CAR (NR1I3)",   # PXR's closest paralog, binds same DR-4 motifs
    "CHEMBL1075167":"AhR",           # different family but same xenosensor role
    "CHEMBL1741186":"FXR (NR1H4)",   # bile acid receptor
    "CHEMBL2828":   "LXRa",
    "CHEMBL1075167":"VDR",
    "CHEMBL2034":   "RXRa",          # PXR's heterodimer partner!
    # Cofactor proteins
    "CHEMBL1075168":"SRC-1",
    # Phase 1/2 metabolism partners
    "CHEMBL2424":   "GSTP1",
}


def morgan_fp(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def main():
    print("=== nb148: PXR structural/regulatory context ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    print("Loading ChEMBL bulk...")
    bulk = pd.read_parquet(DATA_EXTERNAL / "chembl_bulk_activities.parquet",
                            columns=["canonical_smiles","target_chembl_id","pchembl_value"])
    bulk = bulk.dropna(subset=["canonical_smiles","target_chembl_id","pchembl_value"])
    print(f"  {len(bulk):,} rows")

    # Subset to PXR regulatory network targets
    pxr_net_ids = list(PXR_NETWORK.keys())
    bulk_net = bulk[bulk["target_chembl_id"].isin(pxr_net_ids)].copy()
    print(f"  PXR network records: {len(bulk_net):,}  "
          f"({bulk_net['target_chembl_id'].nunique()} of {len(pxr_net_ids)} targets covered)")

    # Per (compound, target) median
    bulk_net = bulk_net.groupby(["canonical_smiles","target_chembl_id"])["pchembl_value"].median().reset_index()
    ref_smiles = sorted(bulk_net["canonical_smiles"].unique())
    target_list = sorted(bulk_net["target_chembl_id"].unique())
    smi_idx = {s:i for i,s in enumerate(ref_smiles)}
    tgt_idx = {t:i for i,t in enumerate(target_list)}
    print(f"  Ref compounds: {len(ref_smiles):,}  targets: {len(target_list)}")

    # Sparse matrix
    rows = bulk_net["canonical_smiles"].map(smi_idx).values
    cols = bulk_net["target_chembl_id"].map(tgt_idx).values
    vals = bulk_net["pchembl_value"].values
    A = csr_matrix((vals, (rows, cols)), shape=(len(ref_smiles), len(target_list)))
    M = csr_matrix(([1.0]*len(vals), (rows, cols)), shape=(len(ref_smiles), len(target_list)))

    # Reference FPs
    print("Building reference FPs...")
    t0 = time.time()
    ref_fps = [morgan_fp(s) for s in ref_smiles]
    valid = [f is not None for f in ref_fps]
    ref_fps_v = [f for f, v in zip(ref_fps, valid) if v]
    keep = np.array(valid)
    A_v = A[keep]; M_v = M[keep]
    print(f"  Valid: {len(ref_fps_v):,} in {time.time()-t0:.0f}s")

    K = 15; MIN_SIM = 0.30
    n_t = A_v.shape[1]

    def expand(qsmiles, label):
        n = len(qsmiles)
        feat_avg = np.full((n, n_t), np.nan, dtype=np.float32)
        feat_eng = np.zeros((n, n_t), dtype=np.float32)
        feat_max_sim = np.zeros(n); feat_n = np.zeros(n)
        t0 = time.time()
        for i, qs in enumerate(qsmiles):
            qfp = morgan_fp(qs)
            if qfp is None: continue
            sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, ref_fps_v), dtype=np.float32)
            feat_max_sim[i] = sims.max()
            top = np.argsort(sims)[::-1][:K]
            top = top[sims[top] >= MIN_SIM]
            feat_n[i] = len(top)
            if len(top) == 0: continue
            top_sims = sims[top]
            sub_A = A_v[top].toarray(); sub_M = M_v[top].toarray()
            mask = sub_M > 0
            for t_idx in range(n_t):
                cm = mask[:, t_idx]
                if cm.sum() > 0:
                    w = top_sims[cm]
                    feat_avg[i, t_idx] = np.dot(w, sub_A[cm, t_idx]) / w.sum()
                    feat_eng[i, t_idx] = cm.sum() / len(top)
            if (i+1) % 500 == 0:
                print(f"  {label}: {i+1}/{n}  ({time.time()-t0:.0f}s)")
        return feat_avg, feat_eng, feat_max_sim, feat_n

    tr_avg, tr_eng, tr_msim, tr_nn = expand(smiles_tr, "train")
    te_avg, te_eng, te_msim, te_nn = expand(smiles_te, "test")

    # Per-target correlations with PXR pEC50
    print("\nPXR network target correlations with PXR pEC50:")
    for t_idx, t in enumerate(target_list):
        col = tr_avg[:, t_idx]
        mask = np.isfinite(col)
        if mask.sum() < 50: continue
        if col[mask].std() < 0.01: continue
        rho, p = spearmanr(col[mask], y_tr[mask])
        name = PXR_NETWORK.get(t, t)
        print(f"  {t:18s} ({name:20s}): ρ={rho:+.3f}  p={p:.2e}  n={mask.sum()}")

    # Fill NaN
    for t_idx in range(n_t):
        col = tr_avg[:, t_idx]
        med = np.nanmedian(col)
        if not np.isfinite(med): med = 5.0
        tr_avg[:, t_idx] = np.where(np.isfinite(col), col, med)
        te_avg[:, t_idx] = np.where(np.isfinite(te_avg[:, t_idx]), te_avg[:, t_idx], med)

    # Augmented LGBM
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    X_tr_aug = np.hstack([X_tr_base, tr_avg, tr_eng, tr_msim.reshape(-1,1), tr_nn.reshape(-1,1)])
    X_te_aug = np.hstack([X_te_base, te_avg, te_eng, te_msim.reshape(-1,1), te_nn.reshape(-1,1)])
    print(f"\nAugmented features: {X_tr_aug.shape[1]}")

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)
    for name, Xt, Xe in [("base_only", X_tr_base, X_te_base),
                          ("pxr_network", X_tr_aug, X_te_aug)]:
        oof = np.zeros(len(y_tr)); te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(Xt[tr_idx], y_tr[tr_idx],
                  eval_set=[(Xt[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(Xt[va_idx])
            te_preds.append(m.predict(Xe))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
        print(f"  {name:14s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
        if name == "pxr_network":
            np.save(DATA_PROCESSED / "oof_nb148_pxr_network.npy", oof)
            np.save(DATA_PROCESSED / "te_nb148_pxr_network.npy", te_pred)
            print(f"  Saved")


if __name__ == "__main__":
    main()
