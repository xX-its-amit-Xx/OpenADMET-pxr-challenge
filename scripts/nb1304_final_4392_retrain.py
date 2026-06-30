"""nb1304 — Final retrain on 4392 compounds (4139 train + 253 unblinded truth).

Run AFTER nb1301/1302/1303 gates confirm we have our best model config.
Strategy:
  - Combine 4139 training compounds + 253 Phase-1-unblinded compounds
  - Retrain 4-GBM (LGBM+XGB+HistGB+CatBoost) with full physics features
  - Re-use cached CheMeleon embeddings for the 253 from chemeleon_te.npy
  - Re-use TabPFN bagged on 4392 context
  - GNN sisterNR OOF: use existing gnn_te.npy slice for the 253 (no retrain needed)
  - Produce two submissions:
      nb1304_260_blind.csv    — predictions for 260 still-blinded compounds
      nb1304_513_hybrid.csv  — true labels for 253 + model preds for 260

Physics features (AIMNet2/strain/D4/DBSTEP/OrbMol) cover all 4652 compounds already.
Validation: cross-val on 4392 with scaffold splits. Do NOT treat this as a gate —
adding 253 ground-truth rows always helps (oracle upper bound). The real signal is
final LB score on the 260 blind compounds.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SD   = "C:/pxr_work/search"
UBD  = "C:/pxr_work/phase1_unblind"
OUT  = "C:/pxr_work/phase1_unblind"
SUBS = "D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/submissions"
os.makedirs(OUT, exist_ok=True)

TABPFN_CKPT = "C:/pxr_work/tabpfn_v2/tabpfn-v2-regressor.ckpt"
N_SEEDS = 3

def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""

def rae_(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d else float("nan")

# ────────────────────────────────────────────────────────────────────────────
# 1. Load the 4139 training compounds
# ────────────────────────────────────────────────────────────────────────────
print("="*60); print("Step 1: Load training data")
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
y_tr = tr["pec50"].to_numpy()
tr_smi = tr["smiles"].tolist()
tr_names = tr["name"].tolist() if "name" in tr.columns else [f"TR_{i}" for i in range(len(tr))]
print(f"  Train: {len(tr)} compounds")

# ────────────────────────────────────────────────────────────────────────────
# 2. Load 253 unblinded truth labels from the RAW HF CSV (avoid Mol-object bug)
# ────────────────────────────────────────────────────────────────────────────
print("Step 2: Load 253 unblinded labels")
raw_path = f"{UBD}/phase1_unblinded_raw.csv"
raw = pd.read_csv(raw_path)
name_col = next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pec_col  = next(c for c in raw.columns if "pec50" in c.lower() or "activity" in c.lower())
raw = raw[[name_col, pec_col]].dropna()
raw.columns = ["name", "pec50_true"]
print(f"  Unblinded: {len(raw)} rows, pEC50 [{raw.pec50_true.min():.2f}, {raw.pec50_true.max():.2f}]")

# Match to test set to get SMILES
te = load_test().reset_index(drop=True)
te_smi = te["smiles"].tolist()
unblind_mask = te["name"].isin(set(raw["name"]))
blind_mask   = ~unblind_mask
unblind_idx  = te.index[unblind_mask].tolist()
blind_idx    = te.index[blind_mask].tolist()
print(f"  Matched {unblind_mask.sum()} unblinded, {blind_mask.sum()} still blind in test set")

# Get SMILES + true labels for the 253 in test-set order
te_ub = te[unblind_mask].merge(raw, on="name", how="left").reset_index(drop=True)
y_ub  = te_ub["pec50_true"].to_numpy()
ub_smi   = te_ub["smiles"].tolist()
ub_names = te_ub["name"].tolist()
blind_smi   = [te_smi[i] for i in blind_idx]
blind_names = te["name"].iloc[blind_idx].tolist()

# ────────────────────────────────────────────────────────────────────────────
# 3. Build 4392-compound feature matrices
# ────────────────────────────────────────────────────────────────────────────
print("Step 3: Build features (4392 = 4139 + 253)")

# Combined (Morgan+RDKit) for all 4392
all_smi   = tr_smi + ub_smi
all_names = tr_names + ub_names
all_y     = np.concatenate([y_tr, y_ub])
scaf_all  = np.array([murcko(s) for s in all_smi])

print(f"  Combined features (2265)...")
Xcomb = impute(combined(all_smi)).astype(np.float32)
print(f"  Xcomb: {Xcomb.shape}")

# Physics features for all 4392 (cached files cover 4652 = 4139 train + 513 test)
def aligned_phys(csv_path, cols, all_names_arg):
    if not os.path.exists(csv_path):
        print(f"    MISSING: {csv_path}"); return np.zeros((len(all_names_arg), len(cols)), np.float32)
    df = pd.read_csv(csv_path).drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(all_names_arg)[cols]
    X = sub.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X)); X[idx] = np.take(med, idx[1])
    nan_rows = int(np.isnan(X).any(1).sum())
    print(f"    {os.path.basename(csv_path)}: {X.shape[1]} cols, {nan_rows} NaN rows post-impute")
    return X.astype(np.float32)

phys_blocks = []
for csv_path, cols in [
    ("C:/pxr_work/aimnet2/aimnet_features.csv",
     ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
      "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]),
    ("C:/pxr_work/strain/strain_features.csv",
     ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
      "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]),
    ("C:/pxr_work/d4/d4_features.csv",
     ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max",
      "d4_c6diag_mean","d4_c6diag_std","d4_c6_total","d4_edisp",
      "d4_edisp_per_atom","d4_cn_mean","d4_cn_max","d4_qeeq_min",
      "d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]),
    ("C:/pxr_work/dbstep/dbstep_features.csv",
     ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65",
      "ster_L","ster_Bmin","ster_Bmax","ster_aniso",
      "npr1","npr2","asphericity","spherocity","eccentricity",
      "radgyr","inertial_sf"]),
    ("C:/pxr_work/orbmol/orbmol_features.csv",
     ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd",
      "orb_conf_mean","orb_conf_std","orb_conf_node_mean","orb_conf_node_std",
      "orb_conf_node_min","orb_node_emb_mean","orb_node_emb_std",
      "orb_node_emb_norm"]),
]:
    phys_blocks.append(aligned_phys(csv_path, cols, all_names))

Xphys = np.hstack(phys_blocks)
X4392 = np.hstack([Xcomb, Xphys])
print(f"  X4392 full: {X4392.shape}")

# ────────────────────────────────────────────────────────────────────────────
# 4. CheMeleon LGBM on 4392 — use cached embeddings for train+unblind
# ────────────────────────────────────────────────────────────────────────────
print("Step 4: CheMeleon OOF on 4392")
# chemeleon_tr.npy = 4139×d, chemeleon_te.npy = 513×d (covers all 513 test compounds)
ecm_tr = np.load(f"{SD}/chemeleon_tr.npy", allow_pickle=True)
ecm_te = np.load(f"{SD}/chemeleon_te.npy", allow_pickle=True)
if ecm_tr.ndim > 2: ecm_tr = ecm_tr.reshape(ecm_tr.shape[0], -1)
if ecm_te.ndim > 2: ecm_te = ecm_te.reshape(ecm_te.shape[0], -1)
print(f"  CheMeleon tr={ecm_tr.shape}, te={ecm_te.shape}")

# 4392 embeddings: stack train (4139) + unblinded slice of te
ecm_ub = ecm_te[unblind_idx]   # 253 × d
ecm_bl = ecm_te[blind_idx]     # 260 × d
ecm_4392 = np.vstack([ecm_tr, ecm_ub])
print(f"  CheMeleon 4392: {ecm_4392.shape}")

# Scaffold CV on 4392
from lightgbm import LGBMRegressor as LGB
chemeleon_oof_4392 = np.zeros(len(all_y))
chemeleon_te_260   = np.zeros(len(blind_idx))
for trn, val in scaffold_kfold_indices(scaf_all.tolist(), n_splits=5, seed=42):
    m = LGB(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)
    m.fit(ecm_4392[trn], all_y[trn])
    chemeleon_oof_4392[val] = m.predict(ecm_4392[val])

# Production preds for 260 blind: train on all 4392
m_cm = LGB(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)
m_cm.fit(ecm_4392, all_y); chemeleon_te_260 = m_cm.predict(ecm_bl)
print(f"  CheMeleon OOF RAE (4392): {rae_(all_y, chemeleon_oof_4392):.4f}")

# ────────────────────────────────────────────────────────────────────────────
# 5. TabPFN on 4392 (bagged, capped context)
# ────────────────────────────────────────────────────────────────────────────
print("Step 5: TabPFN bagged on 4392")
tabpfn_oof_4392 = np.zeros(len(all_y))
tabpfn_te_260   = np.zeros(len(blind_idx))
try:
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    from tabpfn import TabPFNRegressor
    CTX, BAGS = 1200, 3

    def tabpfn_bagged(Xfit, yfit, Xpred, ctx=CTX, bags=BAGS, seed=0):
        preds = []
        rng = np.random.RandomState(seed)
        for b in range(bags):
            idx = rng.choice(len(Xfit), min(ctx, len(Xfit)), replace=False)
            m = TabPFNRegressor(model_path=TABPFN_CKPT, n_estimators=4, random_state=b)
            m.fit(Xfit[idx], yfit[idx]); preds.append(m.predict(Xpred))
        return np.mean(preds, axis=0)

    # Use scaffold folds
    Xcm_tr_arr = ecm_4392.astype(np.float32)
    for trn, val in scaffold_kfold_indices(scaf_all.tolist(), n_splits=5, seed=42):
        tabpfn_oof_4392[val] = tabpfn_bagged(Xcm_tr_arr[trn], all_y[trn], Xcm_tr_arr[val])
    tabpfn_te_260 = tabpfn_bagged(Xcm_tr_arr, all_y, ecm_bl.astype(np.float32))
    print(f"  TabPFN OOF RAE (4392): {rae_(all_y, tabpfn_oof_4392):.4f}")
except Exception as e:
    print(f"  TabPFN failed: {e}. Using cached tabpfn_oof as fallback.")
    # Fallback: use original 4139 tabpfn_oof; pad 253 with mean
    tfn_oof_4139 = np.load(f"{SD}/tabpfn_oof.npy")
    tfn_te_513   = np.load(f"{SD}/tabpfn_te.npy")
    tabpfn_oof_4392[:len(y_tr)] = tfn_oof_4139
    tabpfn_oof_4392[len(y_tr):] = tfn_te_513[0]  # placeholder — ub preds
    tabpfn_te_260 = tfn_te_513[blind_idx]
    print(f"  Fallback TabPFN OOF (not 4392-trained).")

# ────────────────────────────────────────────────────────────────────────────
# 6. Sister-NR GNN — use existing GNN te preds for 253 (no retrain)
# ────────────────────────────────────────────────────────────────────────────
print("Step 6: Sister-NR GNN (no retrain — use cached te preds)")
gnn_oof_4139 = np.load(f"{SD}/gnn_oof.npy")
gnn_te_513   = np.load(f"{SD}/gnn_te.npy")
# Extend: OOF for 4139 train, use gnn_te for the 253 unblinded (they were "test" to the GNN)
gnn_oof_4392 = np.concatenate([gnn_oof_4139, gnn_te_513[unblind_idx]])
gnn_te_260   = gnn_te_513[blind_idx]

# ────────────────────────────────────────────────────────────────────────────
# 7. 4-GBM stack on X4392 + physics features
# ────────────────────────────────────────────────────────────────────────────
print("Step 7: 4-GBM ensemble on 4392")

# GBM for each seed, blended
def gbm_4392(X4392, all_y, scaf_all, seed):
    folds = scaffold_kfold_indices(scaf_all.tolist(), n_splits=5, seed=seed)
    oof = np.zeros(len(all_y))
    for trn, val in folds:
        preds = []
        for mdl in [
            LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1),
            XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=4, verbosity=0,
                         tree_method="hist"),
            HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=64, learning_rate=0.05),
            CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0),
        ]:
            mdl.fit(X4392[trn], all_y[trn]); preds.append(mdl.predict(X4392[val]))
        oof[val] = np.mean(preds, axis=0)
    return oof

# Production GBM — train on all 4392
def gbm_4392_prod(X4392, all_y, Xblind):
    preds = []
    for mdl in [
        LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1),
        XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=4, verbosity=0,
                     tree_method="hist"),
        HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=64, learning_rate=0.05),
        CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0),
    ]:
        mdl.fit(X4392, all_y); preds.append(mdl.predict(Xblind))
    return np.mean(preds, axis=0)

gbm_oofs = []
for seed in range(N_SEEDS):
    print(f"  GBM seed {seed}...", end=" ", flush=True)
    oof = gbm_4392(X4392, all_y, scaf_all, seed)
    print(f"RAE={rae_(all_y, oof):.4f}")
    gbm_oofs.append(oof)

gbm_oof_4392 = np.mean(gbm_oofs, axis=0)
print(f"  GBM ensemble OOF RAE (4392): {rae_(all_y, gbm_oof_4392):.4f}")

# 260-blind GBM predictions
X4392_blind = np.hstack([
    impute(combined(blind_smi)).astype(np.float32),
    np.hstack([aligned_phys(p, c, blind_names) for p, c in [
        ("C:/pxr_work/aimnet2/aimnet_features.csv",
         ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
          "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]),
        ("C:/pxr_work/strain/strain_features.csv",
         ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
          "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]),
        ("C:/pxr_work/d4/d4_features.csv",
         ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max",
          "d4_c6diag_mean","d4_c6diag_std","d4_c6_total","d4_edisp",
          "d4_edisp_per_atom","d4_cn_mean","d4_cn_max","d4_qeeq_min",
          "d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]),
        ("C:/pxr_work/dbstep/dbstep_features.csv",
         ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65",
          "ster_L","ster_Bmin","ster_Bmax","ster_aniso",
          "npr1","npr2","asphericity","spherocity","eccentricity",
          "radgyr","inertial_sf"]),
        ("C:/pxr_work/orbmol/orbmol_features.csv",
         ["orb_energy","orb_energy_per_ha","orb_fmax","orb_frms","orb_fstd",
          "orb_conf_mean","orb_conf_std","orb_conf_node_mean","orb_conf_node_std",
          "orb_conf_node_min","orb_node_emb_mean","orb_node_emb_std",
          "orb_node_emb_norm"]),
    ]])
])
gbm_te_260 = gbm_4392_prod(X4392, all_y, X4392_blind)
print(f"  GBM 260-blind prod preds: mean={gbm_te_260.mean():.3f}")

# ────────────────────────────────────────────────────────────────────────────
# 8. Ensemble: blend GBM + CheMeleon + TabPFN + GNN (same weights as deployed)
# ────────────────────────────────────────────────────────────────────────────
print("Step 8: Blend ensemble")
W = {"gbm": 0.4, "chemeleon": 0.2, "tabpfn": 0.2, "gnn": 0.2}

ens_oof_4392 = (W["gbm"]      * gbm_oof_4392 +
                W["chemeleon"] * chemeleon_oof_4392 +
                W["tabpfn"]    * tabpfn_oof_4392 +
                W["gnn"]       * gnn_oof_4392)

ens_te_260   = (W["gbm"]      * gbm_te_260 +
                W["chemeleon"] * chemeleon_te_260 +
                W["tabpfn"]    * tabpfn_te_260 +
                W["gnn"]       * gnn_te_260)

cv_rae = rae_(all_y, ens_oof_4392)
print(f"  Ensemble 4392 CV RAE: {cv_rae:.4f}")

# Clip to training range
lo = float(np.quantile(all_y, 0.02)); hi = float(np.quantile(all_y, 0.98))
ens_te_260_clipped = np.clip(ens_te_260, lo, hi)

# ────────────────────────────────────────────────────────────────────────────
# 9. Build submission CSVs
# ────────────────────────────────────────────────────────────────────────────
print("Step 9: Build submissions")

# a) 260-blind-only submission
sub_260 = pd.DataFrame({"Molecule Name": blind_names, "pEC50": ens_te_260_clipped})
p260 = f"{SUBS}/nb1304_260_blind_ensemble.csv"
sub_260.to_csv(p260, index=False)
print(f"  Saved {p260}")

# b) Hybrid 513: true labels for 253 + model preds for 260
# The challenge expects predictions for all 513; submit true labels for 253
sub_513 = te[["name"]].copy()
sub_513.columns = ["Molecule Name"]
pred_513 = np.zeros(513)
# Fill in true labels for 253 unblinded
for i, idx in enumerate(unblind_idx):
    pred_513[idx] = y_ub[i]
# Fill in model predictions for 260 blind
for i, idx in enumerate(blind_idx):
    pred_513[idx] = ens_te_260_clipped[i]
sub_513["pEC50"] = pred_513
p513 = f"{SUBS}/nb1304_513_hybrid_ensemble.csv"
sub_513.to_csv(p513, index=False)
print(f"  Saved {p513}")

# ────────────────────────────────────────────────────────────────────────────
# 10. Save metadata
# ────────────────────────────────────────────────────────────────────────────
meta = {
    "n_train": 4392, "n_4139": len(y_tr), "n_unblinded_added": len(y_ub),
    "n_blind_predicted": len(blind_idx),
    "cv_rae_4392": round(cv_rae, 4),
    "blend_weights": W,
    "sub_260": p260, "sub_513_hybrid": p513,
    "note": "submit sub_513_hybrid — true labels for 253 + model for 260",
}
json.dump(meta, open(f"{OUT}/retrain_4392_meta.json", "w"), indent=2)
print(f"\n{'='*60}")
print(f"DONE — 4392 retrain complete")
print(f"  CV RAE (4392 scaffold-CV): {cv_rae:.4f}")
print(f"  Submit: {p513}")
print(json.dumps(meta, indent=2))
