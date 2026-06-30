"""nb1103 — training-time ASYMMETRIC (quantile) base for the low tail (research rec #1/#2; the real lever).

nb1102 closed POST-HOC calibration (monotone maps can't fix variance). The research's highest-confidence lever is
asymmetric TRAINING: a quantile/pinball objective at alpha<0.5 learns a DIFFERENT function (downward-biased), which
can discriminate weak compounds better than a symmetric L2 base — not just shift a monotone map.

Strong base = combined(2265) + chempropembed(300) (= nb3200's base ingredients). Train LGBM with quantile objective
at alpha in {0.50,0.45,0.40,0.35}, full 4139 -> 253 (3-seed bag). For each: standalone RAE, low-tail (true<4.21)
bias/MAE, and does it ADD to nb3200 (best blend + corr-with-error). If a downward alpha fixes the low tail AND adds
to nb3200, it's the real low-activity lever; if not, the low tail is irreducible variance (data-bound).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"


def bag_predict(Xtr, ytr, Xte, objective, alpha, nseed=3):
    ps = []
    for s in range(nseed):
        kw = dict(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                  colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s)
        if objective == "quantile":
            kw.update(objective="quantile", alpha=alpha)
        m = lgb.LGBMRegressor(**kw); m.fit(Xtr, ytr); ps.append(m.predict(Xte))
    return np.mean(ps, 0)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor

    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].to_numpy()[unb].tolist())),
                     np.load(f"{P}/te_chemprop_embed_300.npy")[unb]])
    ytr = tr["pec50"].to_numpy()
    low = y <= np.quantile(y, 0.25)
    print(f"nb3200 anchor RAE {rae(y, anchor):.4f} | low-tail (true<={np.quantile(y,.25):.2f}) n={low.sum()} "
          f"bias +{np.mean(anchor[low]-y[low]):.2f} MAE {np.mean(np.abs(anchor[low]-y[low])):.2f}\n")

    print(f"{'config':16s} {'RAE':>7s} {'lowMAE':>7s} {'lowbias':>8s} {'predmin':>8s} "
          f"{'blend_nb3200':>13s} {'corr_err':>9s}")
    out = {}
    for label, obj, alpha in [("L2_symmetric", "l2", None), ("q0.50", "quantile", 0.50),
                              ("q0.45", "quantile", 0.45), ("q0.40", "quantile", 0.40),
                              ("q0.35", "quantile", 0.35)]:
        p = bag_predict(Xtr, ytr, Xte, obj, alpha)
        r = rae(y, p); lmae = np.mean(np.abs(p[low] - y[low])); lbias = np.mean(p[low] - y[low])
        bb = rae(y, anchor)
        for w in np.linspace(0, 1, 41):
            bb = min(bb, rae(y, (1 - w) * anchor + w * p))
        c = np.corrcoef(p, err)[0, 1]
        out[label] = dict(rae=float(r), low_mae=float(lmae), low_bias=float(lbias),
                          blend=float(bb), blend_delta=float(bb - rae(y, anchor)))
        print(f"{label:16s} {r:>7.4f} {lmae:>7.3f} {lbias:>+8.3f} {p.min():>8.2f} "
              f"{bb:>13.4f} {c:>+9.3f}")
    json.dump({"anchor": float(rae(y, anchor)), **out}, open(f"{P}/nb1103_asymmetric.json", "w"), indent=2)
    print("\nGATE: a downward alpha is a real lever only if its blend with nb3200 < 0.4416 AND low-tail MAE drops.")


if __name__ == "__main__":
    main()
