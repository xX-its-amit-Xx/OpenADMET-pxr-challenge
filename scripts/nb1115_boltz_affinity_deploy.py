"""nb1115 — Boltz-2 AFFINITY -> pEC50 calibration deploy test (user's idea; gating passed at Spearman -0.48).

Calibrate Boltz affinity (pred_value + prob_binary) -> pEC50 on the 300 cofolded TRAIN compounds, apply to the 253
test (cofolded). Honest deploy gates: standalone RAE, corr-with-nb3200-ERROR, blend pooled + LEAVE-SERIES-OUT
(cycle-305), and MARGINAL-OVER-RICH-Z (is the affinity scalar redundant with the z-embedding from the same cofold?).
Inputs: C:/tb/tmp/1/aff_test_all.csv (513, downloaded), C:/pxr_work/affinity/{calib300.csv, aff_calib_all.csv}.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"; MO = "C:/pxr_struct/boltz/modal"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    te = load_test()

    # calibration set: 300 train affinities + pEC50
    cal = pd.read_csv("C:/pxr_work/affinity/calib300.csv").reset_index().rename(columns={"index": "row"})
    acal = pd.read_csv("C:/tb/tmp/1/aff_calib_all.csv"); acal["row"] = acal["idx"].astype(int)
    cal = cal.merge(acal, on="row")
    Xc = cal[["affinity_pred_value", "affinity_prob_binary"]].apply(pd.to_numeric, errors="coerce").values
    yc = cal["pec50"].values
    ok = np.isfinite(Xc).all(1) & np.isfinite(yc); Xc, yc = Xc[ok], yc[ok]

    # test affinities (513), aligned by idx = load_test row order
    at = pd.read_csv("C:/tb/tmp/1/aff_test_all.csv"); at["row"] = at["idx"].astype(int)
    at = at.set_index("row").reindex(range(len(te)))
    Xt = at[["affinity_pred_value", "affinity_prob_binary"]].apply(pd.to_numeric, errors="coerce").values
    Xt = np.where(np.isfinite(Xt), Xt, np.nanmedian(Xc, 0))
    Xt_unb = Xt[unb]

    # calibrate affinity -> pEC50 (GBM); predict 253
    m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xc, yc)
    pred = m.predict(Xt_unb)
    print(f"nb3200 anchor RAE {rae(y, anchor):.4f}")
    print(f"Boltz-affinity calibrated standalone RAE {rae(y, pred):.4f}")
    print(f"corr(calibrated affinity pred, nb3200 error) {np.corrcoef(pred, err)[0,1]:+.3f}")

    scaf = [murcko(s) for s in te["smiles"].to_numpy()[unb]]
    richz = np.load(f"{MO}/test_richz.npy")[unb]
    richz = np.where(np.isfinite(richz), richz, 0.0)
    series = KMeans(6, n_init=5, random_state=0).fit_predict(
        PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(richz)))

    def blend_xseries(p):
        oof = anchor.copy()
        for kk in range(6):
            trn = series != kk; val = series == kk
            if val.sum() < 5: continue
            bw, bb = 0, rae(y[trn], anchor[trn])
            for w in np.linspace(0, 1, 41):
                r = rae(y[trn], (1 - w) * anchor[trn] + w * p[trn])
                if r < bb: bb, bw = r, w
            oof[val] = (1 - bw) * anchor[val] + bw * p[val]
        return rae(y, oof)

    bp = rae(y, anchor)
    for w in np.linspace(0, 1, 41): bp = min(bp, rae(y, (1 - w) * anchor + w * pred))
    print(f"blend pooled {bp:.4f} ({bp-rae(y,anchor):+.4f}) | blend X-SERIES {blend_xseries(pred):.4f} "
          f"({blend_xseries(pred)-rae(y,anchor):+.4f})")

    # marginal over rich-z: residual(y-nb3200) from [richz] vs [richz + affinity], 30-seed cross-fit
    A = np.hstack([richz, Xt_unb])
    def resid_blend(B):
        ds = []
        for seed in range(1200, 1215):
            oof = np.zeros(len(y))
            for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=seed):
                rm = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, n_jobs=4, verbose=-1, random_state=seed)
                rm.fit(StandardScaler().fit_transform(B)[trn], err[trn]); oof[val] = rm.predict(StandardScaler().fit_transform(B)[val])
            b = rae(y, anchor)
            for w in np.linspace(0, 1.5, 31): b = min(b, rae(y, anchor + w * oof))
            ds.append(b - rae(y, anchor))
        return float(np.mean(ds))
    print(f"\nMARGINAL: residual-blend rich-z only {resid_blend(richz):+.4f} | rich-z + affinity {resid_blend(A):+.4f} "
          f"(more negative w/ affinity = adds beyond rich-z)")
    json.dump({"standalone": float(rae(y, pred)), "corr_err": float(np.corrcoef(pred, err)[0, 1]),
               "blend_pooled": float(bp), "blend_xseries": float(blend_xseries(pred))},
              open(f"{P}/nb1115_boltz_affinity.json", "w"), indent=2)
    print("\nGATE: real lever if blend_xseries < 0.4416 AND it adds beyond rich-z.")


if __name__ == "__main__":
    main()
