"""nb963 — does the ADMET-AI gain survive on (a) the 253-unblind eval and (b) STACKED on the
chemprop_aux anchor (the nb3200 substrate)? The degradation-curve win (nb962, 8/8 seeds) was on
raw LGBM-combined; both nb3200 and ADMET-AI are GNN-derived, so the gain could be redundant once
anchored. This is the test that decides whether ADMET reaches the ladder.

253-unblind, anchor = te_chemprop_aux (PRE-clean), residual LGBM cross-fit within the 253
(scaffold folds) — the nb3200 residual-stage protocol. Pooled RAE (LB-faithful).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from sklearn.impute import SimpleImputer

D = "data/processed"


def murcko(s):
    try:
        m = Chem.MolFromSmiles(s); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception: return None


def cross_fit_resid(X, resid, anchor, y, folds):
    """Anchor + LGBM(X) residual, scaffold-CV. Returns pooled RAE on y."""
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                              n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        pred[vai] = anchor[vai] + m.predict(X[vai])
    return float(rae(y, pred)), pred


def main():
    te = load_test()
    unb_idx = np.load(f"{D}/_audit_unblind_idx.npy")
    y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb_idx].tolist()
    scaf = [murcko(s) for s in smiles]
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)

    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb_idx]
    print(f"chemprop_aux anchor RAE on 253 = {rae(y, anchor):.4f}")

    Xc = impute(combined(smiles)).astype(np.float32)
    adf = pd.read_csv("C:/admet_out/admet_test.csv")          # 513 rows
    props = [c for c in adf.columns if c != "smiles" and pd.api.types.is_numeric_dtype(adf[c])]
    A_all = adf[props].to_numpy(float)
    A = A_all[unb_idx]
    A = SimpleImputer(strategy="median").fit_transform(A)
    A = np.clip(np.nan_to_num(A, posinf=1e6, neginf=-1e6), -1e6, 1e6).astype(np.float32)
    Xca = np.hstack([Xc, A])
    print(f"253: combined={Xc.shape} ADMET={A.shape}")

    resid = y - anchor
    print("\n=== (b) ANCHOR-STACKED residual cross-fit on 253 (pooled RAE; nb3200 substrate) ===")
    r_c, _ = cross_fit_resid(Xc, resid, anchor, y, folds)
    r_ca, pred_ca = cross_fit_resid(Xca, resid, anchor, y, folds)
    print(f"  chemprop_aux + LGBM(combined)        = {r_c:.4f}")
    print(f"  chemprop_aux + LGBM(combined+ADMET)  = {r_ca:.4f}   delta = {r_ca-r_c:+.4f}")
    print(f"  (nb3200 full stack ceiling = 0.4416; this simpler residual ~ {r_c:.3f})")

    print("\n=== (a) RAW deploy-style 253 eval (no anchor; direct LGBM cross-fit) ===")
    def direct(X):
        pred = np.full(len(y), np.nan)
        for tri, vai in folds:
            m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                                  n_jobs=4, verbose=-1).fit(X[tri], y[tri])
            pred[vai] = m.predict(X[vai])
        return float(rae(y, pred))
    d_c, d_ca = direct(Xc), direct(Xca)
    print(f"  LGBM(combined)       = {d_c:.4f}")
    print(f"  LGBM(combined+ADMET) = {d_ca:.4f}   delta = {d_ca-d_c:+.4f}")

    verdict = ("ADMET ADDS to anchor stack -> build ladder candidate" if r_ca < r_c - 0.002
               else "ADMET REDUNDANT once anchored (gain absorbed by chemprop_aux GNN)")
    print("\n" + "=" * 62 + f"\nVERDICT: {verdict}\n" + "=" * 62)
    json.dump({"anchor_rae": round(float(rae(y, anchor)), 4),
               "stacked_combined": round(r_c, 4), "stacked_combined_admet": round(r_ca, 4),
               "stacked_delta": round(r_ca - r_c, 4),
               "direct_combined": round(d_c, 4), "direct_combined_admet": round(d_ca, 4),
               "direct_delta": round(d_ca - d_c, 4)},
              open(f"{D}/nb963_admet_anchor_integration.json", "w"), indent=2)
    if r_ca < r_c - 0.002:
        np.save(f"{D}/nb963_pred_admet_stacked_253.npy", pred_ca)
    print(f"saved -> {D}/nb963_admet_anchor_integration.json")


if __name__ == "__main__":
    main()
