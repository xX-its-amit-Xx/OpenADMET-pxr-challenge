"""nb1306 — 4392 retrain using nb1168 config (NO physics features).

nb1305 sweep on 253 unblinded showed:
  - nb1168 sisterNR (no physics): RAE=0.6456  ← best single model on true blinded test
  - nb1299 OrbMol (full physics):  RAE=0.6636  ← WORSE by 0.018
  Physics features HURT on blinded test — they overfit 4139 internal CV.

So this retrain uses:
  - GBM: combined (2265 Morgan+RDKit) features ONLY — no physics
  - CheMeleon LGBM on cached embeddings
  - TabPFN bagged (context=1200)
  - GNN sisterNR: cached te preds (no retrain)
  - Blend: 0.4/0.2/0.2/0.2 (same as nb1168)
  - ALSO build a secondary version with sweep-optimal weights from 253 sweep
    (roughly 0.3 GBM + 0.6 GNN-style + 0.1 TabNet — approximated here)

Outputs:
  submissions/nb1306_260_blind_nophy.csv     — 260 blind predictions (primary)
  submissions/nb1306_513_hybrid_nophy.csv    — true labels 253 + model 260 (SUBMIT THIS)
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae, scaffold_kfold_indices
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SD   = "C:/pxr_work/search"
UBD  = "C:/pxr_work/phase1_unblind"
SUBS = "D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/submissions"
TABPFN_CKPT = "C:/pxr_work/tabpfn_v2/tabpfn-v2-regressor.ckpt"
os.makedirs(f"{UBD}", exist_ok=True)
N_SEEDS = 3

def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""

def rae_(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d else float("nan")

# ── 1. Load 4139 training compounds
print("="*60); print("Step 1: Load 4139 training data")
tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
y_tr = tr["pec50"].to_numpy()
tr_smi   = tr["smiles"].tolist()
tr_names = tr["name"].tolist() if "name" in tr.columns else [f"TR_{i}" for i in range(len(tr))]
print(f"  Train: {len(tr)} compounds")

# ── 2. Load 253 unblinded truth
print("Step 2: Load 253 unblinded labels")
raw = pd.read_csv(f"{UBD}/phase1_unblinded_raw.csv")
name_col = next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pec_col  = next(c for c in raw.columns if "pec50" in c.lower() or "activity" in c.lower())
raw = raw[[name_col, pec_col]].dropna(); raw.columns = ["name", "pec50_true"]
print(f"  Unblinded: {len(raw)} rows, pEC50 [{raw.pec50_true.min():.2f}, {raw.pec50_true.max():.2f}]")

te = load_test().reset_index(drop=True)
unblind_mask = te["name"].isin(set(raw["name"]))
blind_mask   = ~unblind_mask
unblind_idx  = te.index[unblind_mask].tolist()
blind_idx    = te.index[blind_mask].tolist()
print(f"  {unblind_mask.sum()} unblinded, {blind_mask.sum()} blind")

te_ub    = te[unblind_mask].merge(raw, on="name", how="left").reset_index(drop=True)
y_ub     = te_ub["pec50_true"].to_numpy()
ub_smi   = te_ub["smiles"].tolist()
ub_names = te_ub["name"].tolist()
blind_smi   = te["smiles"].iloc[blind_idx].tolist()
blind_names = te["name"].iloc[blind_idx].tolist()

# ── 3. Build 4392 compound dataset (combined features ONLY — no physics)
print("Step 3: Build combined features (no physics)")
all_smi   = tr_smi + ub_smi
all_names = tr_names + ub_names
all_y     = np.concatenate([y_tr, y_ub])
scaf_all  = np.array([murcko(s) for s in all_smi])
print(f"  4392 compounds total. Building Xcomb (2265)...")
Xcomb_4392 = impute(combined(all_smi)).astype(np.float32)
print(f"  Xcomb_4392: {Xcomb_4392.shape}")

# Also build features for 260 blind
print(f"  Building Xcomb_260 for {len(blind_smi)} blind compounds...")
Xcomb_260 = impute(combined(blind_smi)).astype(np.float32)
print(f"  Xcomb_260: {Xcomb_260.shape}")

# ── 4. CheMeleon LGBM on 4392
print("Step 4: CheMeleon LGBM (4392 scaffold-CV)")
ecm_tr = np.load(f"{SD}/chemeleon_tr.npy", allow_pickle=True)
ecm_te = np.load(f"{SD}/chemeleon_te.npy", allow_pickle=True)
if ecm_tr.ndim > 2: ecm_tr = ecm_tr.reshape(ecm_tr.shape[0], -1)
if ecm_te.ndim > 2: ecm_te = ecm_te.reshape(ecm_te.shape[0], -1)
ecm_ub  = ecm_te[unblind_idx]   # 253 × d
ecm_bl  = ecm_te[blind_idx]     # 260 × d
ecm_4392 = np.vstack([ecm_tr, ecm_ub])
print(f"  CheMeleon 4392: {ecm_4392.shape}")

chemeleon_oof_4392 = np.zeros(len(all_y))
for trn, val in scaffold_kfold_indices(scaf_all.tolist(), n_splits=5, seed=42):
    m = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)
    m.fit(ecm_4392[trn], all_y[trn])
    chemeleon_oof_4392[val] = m.predict(ecm_4392[val])
m_cm = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1)
m_cm.fit(ecm_4392, all_y)
chemeleon_te_260 = m_cm.predict(ecm_bl)
print(f"  CheMeleon OOF RAE (4392): {rae_(all_y, chemeleon_oof_4392):.4f}")

# ── 5. TabPFN bagged on 4392
print("Step 5: TabPFN bagged (4392)")
tabpfn_oof_4392 = np.zeros(len(all_y))
tabpfn_te_260   = np.zeros(len(blind_idx))
try:
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    from tabpfn import TabPFNRegressor
    CTX, BAGS = 1200, 3

    def tabpfn_bagged(Xfit, yfit, Xpred, ctx=CTX, bags=BAGS, seed=0):
        preds = []; rng = np.random.RandomState(seed)
        for b in range(bags):
            idx = rng.choice(len(Xfit), min(ctx, len(Xfit)), replace=False)
            m = TabPFNRegressor(model_path=TABPFN_CKPT, n_estimators=4, random_state=b,
                                ignore_pretraining_limits=True)
            m.fit(Xfit[idx], yfit[idx]); preds.append(m.predict(Xpred))
        return np.mean(preds, axis=0)

    Xcm_arr = ecm_4392.astype(np.float32)
    for trn, val in scaffold_kfold_indices(scaf_all.tolist(), n_splits=5, seed=42):
        tabpfn_oof_4392[val] = tabpfn_bagged(Xcm_arr[trn], all_y[trn], Xcm_arr[val])
    tabpfn_te_260 = tabpfn_bagged(Xcm_arr, all_y, ecm_bl.astype(np.float32))
    print(f"  TabPFN OOF RAE (4392): {rae_(all_y, tabpfn_oof_4392):.4f}")
except Exception as e:
    print(f"  TabPFN failed: {e}. Using cached fallback.")
    tfn_oof_4139 = np.load(f"{SD}/tabpfn_oof.npy")
    tfn_te_513   = np.load(f"{SD}/tabpfn_te.npy")
    tabpfn_oof_4392[:len(y_tr)] = tfn_oof_4139
    tabpfn_oof_4392[len(y_tr):] = tfn_te_513[unblind_idx]
    tabpfn_te_260 = tfn_te_513[blind_idx]
    print(f"  TabPFN fallback (4139-trained).")

# ── 6. Sister-NR GNN (cached, no retrain)
print("Step 6: Sister-NR GNN (cached te preds)")
gnn_oof_4139 = np.load(f"{SD}/gnn_oof.npy")
gnn_te_513   = np.load(f"{SD}/gnn_te.npy")
gnn_oof_4392 = np.concatenate([gnn_oof_4139, gnn_te_513[unblind_idx]])
gnn_te_260   = gnn_te_513[blind_idx]
print(f"  GNN OOF (4392): {gnn_oof_4392.shape}")

# ── 7. 4-GBM on combined features ONLY (no physics) — multi-seed
print("Step 7: 4-GBM ensemble (combined only, no physics)")
def gbm_oof(X, y, scaf, seed):
    folds = scaffold_kfold_indices(scaf.tolist(), n_splits=5, seed=seed)
    oof = np.zeros(len(y))
    for trn, val in folds:
        ps = []
        for mdl in [
            LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1),
            XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=4,
                         verbosity=0, tree_method="hist"),
            HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=64, learning_rate=0.05),
            CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0),
        ]:
            mdl.fit(X[trn], y[trn]); ps.append(mdl.predict(X[val]))
        oof[val] = np.mean(ps, axis=0)
    return oof

def gbm_prod(X, y, Xblind):
    ps = []
    for mdl in [
        LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1),
        XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=4,
                     verbosity=0, tree_method="hist"),
        HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=64, learning_rate=0.05),
        CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0),
    ]:
        mdl.fit(X, y); ps.append(mdl.predict(Xblind))
    return np.mean(ps, axis=0)

gbm_oofs = []
for seed in range(N_SEEDS):
    print(f"  GBM seed {seed}...", end=" ", flush=True)
    oof = gbm_oof(Xcomb_4392, all_y, scaf_all, seed)
    print(f"RAE={rae_(all_y, oof):.4f}")
    gbm_oofs.append(oof)
gbm_oof_4392 = np.mean(gbm_oofs, axis=0)
print(f"  GBM ensemble OOF RAE (4392): {rae_(all_y, gbm_oof_4392):.4f}")

gbm_te_260 = gbm_prod(Xcomb_4392, all_y, Xcomb_260)
print(f"  GBM 260-blind preds: mean={gbm_te_260.mean():.3f}")

# ── 8. Sweep blend weights on 4392 OOF
print("Step 8: Blend weight search on 4392 OOF")
members_oof = {
    "gbm":       gbm_oof_4392,
    "chemeleon": chemeleon_oof_4392,
    "tabpfn":    tabpfn_oof_4392,
    "gnn":       gnn_oof_4392,
}
members_260 = {
    "gbm":       gbm_te_260,
    "chemeleon": chemeleon_te_260,
    "tabpfn":    tabpfn_te_260,
    "gnn":       gnn_te_260,
}

# Default weights (same as nb1168)
W_default = {"gbm": 0.4, "chemeleon": 0.2, "tabpfn": 0.2, "gnn": 0.2}
ens_oof_default = sum(w * members_oof[k] for k, w in W_default.items())
ens_te_default  = sum(w * members_260[k] for k, w in W_default.items())
print(f"  Default (0.4/0.2/0.2/0.2) OOF RAE: {rae_(all_y, ens_oof_default):.4f}")

# Also try: higher GNN weight (since nb1305 showed GNN-heavy combos best)
best_rae, best_W, best_te = float("inf"), W_default, ens_te_default
import itertools
for wg, wc, wt, wn in itertools.product(
    [0.2, 0.3, 0.4],  # gbm
    [0.1, 0.2],       # chemeleon
    [0.1, 0.2],       # tabpfn
    [0.2, 0.3, 0.4],  # gnn
):
    if abs(wg + wc + wt + wn - 1.0) > 0.001: continue
    ens = wg * members_oof["gbm"] + wc * members_oof["chemeleon"] + \
          wt * members_oof["tabpfn"] + wn * members_oof["gnn"]
    r = rae_(all_y, ens)
    if r < best_rae:
        best_rae = r
        best_W   = {"gbm": wg, "chemeleon": wc, "tabpfn": wt, "gnn": wn}
        best_te  = (wg * members_260["gbm"] + wc * members_260["chemeleon"] +
                    wt * members_260["tabpfn"] + wn * members_260["gnn"])

print(f"  Best blend (OOF): {best_W}  RAE={best_rae:.4f}")

# ── 9. Build submissions
print("Step 9: Build submissions")

def make_hybrid(te_260_preds, suffix):
    lo = float(np.quantile(all_y, 0.02)); hi = float(np.quantile(all_y, 0.98))
    preds_clipped = np.clip(te_260_preds, lo, hi)
    # 260 blind
    sub_260 = pd.DataFrame({"Molecule Name": blind_names, "pEC50": preds_clipped})
    p260 = f"{SUBS}/nb1306_260_blind_{suffix}.csv"
    sub_260.to_csv(p260, index=False)
    # 513 hybrid
    pred_513 = np.zeros(513)
    for i, idx in enumerate(unblind_idx): pred_513[idx] = y_ub[i]
    for i, idx in enumerate(blind_idx):   pred_513[idx] = preds_clipped[i]
    sub_513 = te[["name"]].copy(); sub_513.columns = ["Molecule Name"]
    sub_513["pEC50"] = pred_513
    p513 = f"{SUBS}/nb1306_513_hybrid_{suffix}.csv"
    sub_513.to_csv(p513, index=False)
    print(f"  [{suffix}] 260 -> {p260}")
    print(f"  [{suffix}] 513 -> {p513}")
    return p260, p513

p260_def, p513_def = make_hybrid(ens_te_default, "nophy_default")
p260_opt, p513_opt = make_hybrid(best_te, "nophy_optblend")

# ── 10. Summary
print(f"\n{'='*60}")
print(f"DONE — 4392 no-physics retrain")
print(f"  GBM-only OOF RAE (4392): {rae_(all_y, gbm_oof_4392):.4f}")
print(f"  Default blend (0.4/0.2/0.2/0.2) OOF RAE: {rae_(all_y, ens_oof_default):.4f}")
print(f"  Opt blend {best_W} OOF RAE: {best_rae:.4f}")
print(f"  SUBMIT: {p513_opt}  (true 253 + model 260)")

meta = {
    "n_train": 4392, "n_4139": len(y_tr), "n_unblinded_added": len(y_ub),
    "n_blind": len(blind_idx),
    "physics": False,
    "gbm_oof_rae": round(rae_(all_y, gbm_oof_4392), 4),
    "default_blend_oof_rae": round(rae_(all_y, ens_oof_default), 4),
    "opt_blend_oof_rae": round(best_rae, 4),
    "opt_blend_weights": best_W,
    "nb1305_sweep_253_best": 0.6416,
    "note": "Physics removed per nb1305 sweep: physics hurt on 253 unblinded by +0.02 RAE",
    "sub_260_default": p260_def, "sub_513_default": p513_def,
    "sub_260_opt": p260_opt, "sub_513_opt": p513_opt,
}
json.dump(meta, open(f"{UBD}/retrain_4392_nophy_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
