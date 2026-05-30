"""nb213 — ChemBERTa-77M-MTR frozen embeddings + Ridge/LGBM regression.

Uses DeepChem/ChemBERTa-77M-MTR as a frozen feature extractor: extract [CLS]
embeddings once (CPU-feasible), then train Ridge with scaffold 5-fold CV.
Goal: get a model with lower error-correlation to nb211 div15+Chemprop blend.

PREV_BEST = 0.296886 (nb211 div15 + Chemprop double-inflation blend)
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
PREV_BEST = 0.296886
COLLAPSE_THRESH = 0.58
BATCH = 64

t0 = time.time()
print("=== nb213: ChemBERTa-77M-MTR frozen embeddings + Ridge ===\n", flush=True)

# ── Load data ──────────────────────────────────────────────────────────────
tr = load_train()
te_df = load_test()
y_tr = tr["pec50"].values.astype(np.float64)
n_tr = len(y_tr)
scaffolds = tr["smiles"].map(bemis_murcko).tolist()
splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)
print(f"Train {n_tr:,}  Test {len(te_df):,}", flush=True)

# ── Load ChemBERTa ──────────────────────────────────────────────────────────
import torch
device = torch.device("cpu")  # CPU is fine for inference-only
BERT_AVAIL = False
MODEL_NAME = "chemberta_mtr"
D_MODEL = 384

try:
    from transformers import AutoTokenizer, AutoModel
    MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"
    print(f"Loading {MODEL_ID}...", flush=True)
    tok  = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModel.from_pretrained(MODEL_ID)
    base.eval()
    n_params = sum(p.numel() for p in base.parameters())
    print(f"Loaded: {n_params:,} params  D_MODEL={D_MODEL}", flush=True)
    BERT_AVAIL = True
except Exception as e:
    print(f"ChemBERTa-77M-MTR failed: {e}", flush=True)
    try:
        MODEL_ID = "seyonec/ChemBERTa-zinc-base-v1"
        print(f"Trying fallback {MODEL_ID}...", flush=True)
        tok  = AutoTokenizer.from_pretrained(MODEL_ID)
        base = AutoModel.from_pretrained(MODEL_ID)
        base.eval()
        n_params = sum(p.numel() for p in base.parameters())
        print(f"Fallback loaded: {n_params:,} params", flush=True)
        BERT_AVAIL = True
        MODEL_NAME = "chemberta_zinc"
    except Exception as e2:
        print(f"All BERT models failed: {e2}", flush=True)
        sys.exit(1)

# ── Extract embeddings ──────────────────────────────────────────────────────
def get_embeddings(smiles_list, batch_size=BATCH, max_len=128):
    """Extract [CLS] embeddings in batches, no gradient needed."""
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            out = base(**enc)
            cls = out.last_hidden_state[:, 0, :].numpy()  # (B, D_MODEL)
            all_emb.append(cls)
            if (i // batch_size) % 10 == 0:
                print(f"  Batch {i//batch_size+1}/{(len(smiles_list)-1)//batch_size+1}", end="\r", flush=True)
    print()
    return np.vstack(all_emb).astype(np.float32)


print(f"\nExtracting train embeddings ({n_tr} compounds)...", flush=True)
t_emb = time.time()
tr_smiles = tr["smiles"].tolist()
X_tr_emb = get_embeddings(tr_smiles)
print(f"Train embeddings: {X_tr_emb.shape}  ({time.time()-t_emb:.0f}s)", flush=True)

print(f"Extracting test embeddings ({len(te_df)} compounds)...", flush=True)
t_emb = time.time()
te_smiles = te_df["smiles"].tolist()
X_te_emb = get_embeddings(te_smiles)
print(f"Test embeddings: {X_te_emb.shape}  ({time.time()-t_emb:.0f}s)", flush=True)

# Save embeddings for reuse
np.save(str(DATA_PROCESSED / f"tr_emb_{MODEL_NAME}.npy"), X_tr_emb)
np.save(str(DATA_PROCESSED / f"te_emb_{MODEL_NAME}.npy"), X_te_emb)
print(f"Embeddings saved to data/processed/", flush=True)

# ── Ridge regression with scaffold CV ──────────────────────────────────────
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

print(f"\nRunning Ridge scaffold CV (alpha search)...", flush=True)

# Normalize embeddings
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_emb)
X_te_sc  = scaler.transform(X_te_emb)

alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
best_rae_ridge = 1e9
best_alpha     = None
best_oof_ridge = None
best_te_ridge  = None

for alpha in alphas:
    t_a = time.time()
    oof_r = np.full(n_tr, np.nan)
    for tr_idx, va_idx in splits:
        m = Ridge(alpha=alpha)
        m.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
        oof_r[va_idx] = m.predict(X_tr_sc[va_idx])
    r = rae(y_tr, oof_r)
    ratio = oof_r.std()
    # Final model for test
    m_final = Ridge(alpha=alpha)
    m_final.fit(X_tr_sc, y_tr)
    te_r = m_final.predict(X_te_sc)
    ratio_val = te_r.std() / oof_r.std() if oof_r.std() > 1e-9 else 0.0
    print(f"  Ridge a={alpha:6.1f}  OOF RAE={r:.4f}  ratio={ratio_val:.4f}  ({time.time()-t_a:.0f}s)", flush=True)
    if r < best_rae_ridge:
        best_rae_ridge = r
        best_alpha = alpha
        best_oof_ridge = oof_r.copy()
        best_te_ridge  = te_r.copy()

print(f"\nBest Ridge: alpha={best_alpha}  OOF RAE={best_rae_ridge:.4f}", flush=True)

# ── Also try LGBM on embeddings ────────────────────────────────────────────
try:
    import lightgbm as lgb
    print(f"\nRunning LGBM on ChemBERTa embeddings...", flush=True)

    lgb_params = dict(
        n_estimators=500, num_leaves=64, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        objective="regression_l1", random_state=SEED, n_jobs=2,
        verbose=-1,
    )
    oof_lgb = np.full(n_tr, np.nan)
    t_lgb = time.time()
    for tr_idx, va_idx in splits:
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(X_tr_emb[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr_emb[va_idx], y_tr[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[va_idx] = m.predict(X_tr_emb[va_idx])
    r_lgb = rae(y_tr, oof_lgb)
    m_final_lgb = lgb.LGBMRegressor(**lgb_params)
    m_final_lgb.fit(X_tr_emb, y_tr)
    te_lgb = m_final_lgb.predict(X_te_emb)
    ratio_lgb = te_lgb.std() / oof_lgb.std() if oof_lgb.std() > 1e-9 else 0.0
    print(f"LGBM on ChemBERTa emb: OOF RAE={r_lgb:.4f}  ratio={ratio_lgb:.4f}  ({time.time()-t_lgb:.0f}s)", flush=True)
    LGB_AVAIL = True
except Exception as e:
    print(f"LGBM failed: {e}", flush=True)
    LGB_AVAIL = False
    r_lgb = 1e9

# Pick best single model
if LGB_AVAIL and r_lgb < best_rae_ridge:
    best_single = "lgbm"
    oof_best = oof_lgb.copy()
    te_best  = te_lgb.copy()
    best_single_rae = r_lgb
else:
    best_single = "ridge"
    oof_best = best_oof_ridge.copy()
    te_best  = best_te_ridge.copy()
    best_single_rae = best_rae_ridge

print(f"\nBest single ChemBERTa model: {best_single}  OOF RAE={best_single_rae:.4f}", flush=True)
print(f"OOF std={oof_best.std():.4f}  test std={te_best.std():.4f}  ratio={te_best.std()/oof_best.std():.4f}", flush=True)

# ── Error correlation with nb211 blend ──────────────────────────────────────
oof211_p = DATA_PROCESSED / "oof_nb211_div15_chemprop_blend.npy"
if oof211_p.exists():
    oof211 = np.load(str(oof211_p)).flatten()
    te211  = np.load(str(DATA_PROCESSED / "te_nb211_div15_chemprop_blend.npy")).flatten()
    corr = np.corrcoef(y_tr - oof211, y_tr - oof_best)[0, 1]
    print(f"\nError correlation with nb211 blend: {corr:.4f}", flush=True)
    print(f"(nb93 Chemprop had corr=0.5315 with blend)", flush=True)
    print(f"(tree models have corr>0.83 with blend)", flush=True)

    # Save for potential blend
    np.save(str(DATA_PROCESSED / f"oof_nb213_{MODEL_NAME}.npy"), oof_best)
    np.save(str(DATA_PROCESSED / f"te_nb213_{MODEL_NAME}.npy"),  te_best)
    print(f"\nSaved oof/te_nb213_{MODEL_NAME}.npy", flush=True)

    if corr > 0.6:
        print("Correlation too high to improve blend significantly", flush=True)
    else:
        print("\nLow correlation — searching for optimal blend with nb211...", flush=True)
        THRESH = COLLAPSE_THRESH
        best_r, best_w = 1e9, 1.0
        te211_raw = np.load(str(DATA_PROCESSED / "te_nb211_div15_chemprop_blend.npy")).flatten()
        # nb211 te is already inflated to ratio=0.580; need raw nb209 for re-blend
        oof209_p = DATA_PROCESSED / "oof_nb209_direct_best_inflate.npy"
        te209_p  = DATA_PROCESSED / "te_nb209_direct_best_inflate.npy"
        if oof209_p.exists():
            oof209 = np.load(str(oof209_p)).flatten()
            te209  = np.load(str(te209_p)).flatten()
            # te209 is the inflated test (ratio=0.580); oof209 is the raw OOF
            # Build blend: w*oof209 + (1-w)*oof_best  →  then inflate test blend
            for w in np.arange(0.80, 1.001, 0.001):
                oof_b = w * oof209 + (1 - w) * oof_best
                te_b  = w * te209  + (1 - w) * te_best
                oof_s = oof_b.std()
                if te_b.std() < 1e-9:
                    continue
                te_b_inf = te_b.mean() + (te_b - te_b.mean()) * (THRESH * oof_s / te_b.std())
                ratio = te_b_inf.std() / oof_s
                r = rae(y_tr, oof_b)
                if ratio >= THRESH - 1e-9 and r < best_r:
                    best_r, best_w = r, w
            print(f"Best blend nb209+ChemBERTa: w={best_w:.3f}  OOF RAE={best_r:.6f}  (vs PREV {PREV_BEST:.6f})", flush=True)
            improvement = PREV_BEST - best_r
            print(f"Improvement: {improvement:.6f}", flush=True)

            if best_r < PREV_BEST:
                w = best_w
                oof_b = w * oof209 + (1 - w) * oof_best
                te_b  = w * te209  + (1 - w) * te_best
                oof_s = oof_b.std()
                te_b_inf = te_b.mean() + (te_b - te_b.mean()) * (THRESH * oof_s / te_b.std())
                sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": te_b_inf})
                assert len(sub) == 513 and sub["pEC50"].notna().all()
                out_stem = f"nb213_{MODEL_NAME}_blend"
                np.save(str(DATA_PROCESSED / f"oof_{out_stem}.npy"), oof_b)
                np.save(str(DATA_PROCESSED / f"te_{out_stem}.npy"),  te_b_inf)
                sub.to_csv(str(SUBMISSIONS / f"{out_stem}.csv"), index=False)
                print(f"\n*** NEW BEST! Saved {out_stem}.csv  OOF RAE={best_r:.6f} ***", flush=True)
            else:
                print("No improvement over nb211 (0.296886)", flush=True)
        else:
            print("nb209 OOF/test files not found — skipping blend", flush=True)
else:
    print("nb211 blend file not found — run nb211 first", flush=True)

print(f"\n=== Summary ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  ChemBERTa {MODEL_NAME}: best single OOF RAE={best_single_rae:.4f}", flush=True)
if oof211_p.exists():
    print(f"  Error corr with nb211: {corr:.4f}", flush=True)
