"""nb1125 — ARCHITECTURE bake-off + data exclusion (user request), validated CROSS-SERIES (the honest gate).

Same features (combined + chempropembed) as nb3200. Compare base learners: LGBM / XGBoost / CatBoost /
HistGradientBoosting / MLP(torch) / Ridge. Report standalone 253 RAE + MEAN PER-SERIES RAE (cross-series robustness,
the blinded-transfer proxy). Then: diverse GBM ensemble (does architecture diversity help robustness?), and data
exclusion (exclude noisy high-SE compounds; keep only test-relevant train). Does any beat LGBM / add to nb3200?
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
import torch, torch.nn as nn

P = "data/processed"


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def mlp_fit(Xtr, ytr, Xte, ep=120):
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xe = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    m = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
                      nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.SmoothL1Loss()
    n = len(Xt)
    for e in range(ep):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]; opt.zero_grad()
            loss = lossf(m(Xt[idx]).squeeze(1), yt[idx]); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(Xe).squeeze(1).numpy()


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    ytr = tr["pec50"].to_numpy(); se = tr["pec50_se"].to_numpy()
    print("featurizing...", flush=True)
    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")]).astype(np.float32)
    Xte = np.hstack([impute(combined(te["smiles"].to_numpy()[unb].tolist())),
                     np.load(f"{P}/te_chemprop_embed_300.npy")[unb]]).astype(np.float32)

    series = KMeans(6, n_init=5, random_state=0).fit_predict(PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(Xte)))
    def perseries(p):
        rs = [rae(y[series == k], p[series == k]) for k in range(6) if (series == k).sum() >= 5]
        return float(np.mean(rs))

    archs = {
        "LGBM": lambda Xa, ya, Xe: np.mean([lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s).fit(Xa, ya).predict(Xe) for s in range(3)], 0),
        "XGBoost": lambda Xa, ya, Xe: np.mean([xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=s).fit(Xa, ya).predict(Xe) for s in range(3)], 0),
        "CatBoost": lambda Xa, ya, Xe: np.mean([CatBoostRegressor(iterations=600, depth=6, learning_rate=0.04,
            verbose=0, random_seed=s).fit(Xa, ya).predict(Xe) for s in range(2)], 0),
        "HistGB": lambda Xa, ya, Xe: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05).fit(Xa, ya).predict(Xe),
        "MLP": lambda Xa, ya, Xe: np.mean([mlp_fit(Xa, ya, Xe) for _ in range(2)], 0),
        "Ridge": lambda Xa, ya, Xe: Ridge(alpha=10.0).fit(StandardScaler().fit_transform(Xa), ya).predict(
            StandardScaler().fit(Xa).transform(Xe)),
    }
    print(f"\nnb3200 anchor RAE {rae(y, anchor):.4f} (LGBM base + residual + clip)\n")
    print(f"{'architecture':14s} {'253_RAE':>8s} {'per_series':>11s} {'corr_err':>9s}")
    preds = {}
    for name, fn in archs.items():
        p = fn(Xtr, ytr, Xte); preds[name] = p
        print(f"{name:14s} {rae(y,p):>8.4f} {perseries(p):>11.4f} {np.corrcoef(p,err)[0,1]:>+9.3f}", flush=True)

    # diverse GBM ensemble
    ens = np.mean([preds["LGBM"], preds["XGBoost"], preds["CatBoost"]], 0)
    print(f"\n{'GBM ensemble':14s} {rae(y,ens):>8.4f} {perseries(ens):>11.4f} {np.corrcoef(ens,err)[0,1]:>+9.3f}")

    # data exclusion on LGBM
    print("\n=== data exclusion (LGBM base) ===")
    keep_lowse = se <= np.quantile(se, 0.8)               # drop noisiest 20%
    Ftr = fpf(tr["smiles"].tolist()); Fte = fpf(te["smiles"].to_numpy()[unb].tolist())
    inter = Ftr @ Fte.T; s = Ftr.sum(1); simmax = (inter / np.clip(s[:, None] + Fte.sum(1)[None, :] - inter, 1, None)).max(1)
    keep_rel = simmax >= 0.3                               # keep only train w/ Tanimoto>=0.3 to some test cpd
    for label, mask in [("all (baseline)", np.ones(len(ytr), bool)), ("excl noisiest 20%", keep_lowse),
                        (f"test-relevant only (n={keep_rel.sum()})", keep_rel)]:
        p = archs["LGBM"](Xtr[mask], ytr[mask], Xte)
        print(f"  {label:30s} 253_RAE {rae(y,p):.4f} | per_series {perseries(p):.4f}")

    json.dump({name: {"rae": float(rae(y, p)), "per_series": perseries(p)} for name, p in preds.items()},
              open(f"{P}/nb1125_archbakeoff.json", "w"), indent=2)
    print("\nNOTE: nb3200 0.4416 includes residual+clip; standalone bases ~0.5-0.6. Question: does any arch/ensemble")
    print("beat LGBM as the base OR improve per-series robustness? (features are likely the bottleneck, not the model.)")


if __name__ == "__main__":
    main()
