"""nb1122 — does REFIT-on-253 help the BLINDED 260? (submission decision: plain nb3200 vs refit-on-4139+253)

The 253 (released) and 260 (blinded) are DIFFERENT series (adversarial AUC 0.984). Refitting on the 253 fits THEM
perfectly, but does adding a different-series set HELP or HURT generalization to a held-out series (the 260's situation)?
Proxy test: split the 253 into 6 chemical series; for each held-out series k, compare predicting it from
  A = 4139-only   vs   B = 4139 + (253 minus series k)
If B<A on held-out series, refit-on-253 helps the blinded 260 -> submit the refit. If B>=A, refit doesn't transfer
(or hurts) -> submit plain nb3200 (safer). This is the cleanest cross-series evidence for the final-submission choice.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"


def bag(Xa, ya, Xte, nseed=2):
    ps = []
    for s in range(nseed):
        m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s)
        m.fit(Xa, ya); ps.append(m.predict(Xte))
    return np.mean(ps, 0)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    ytr = tr["pec50"].to_numpy()
    # chempropembed-only (fast; skips slow combined featurization). Relative refit effect is what matters.
    Xtr = np.load(f"{P}/tr_chemprop_embed_300.npy")
    Xu = np.load(f"{P}/te_chemprop_embed_300.npy")[unb]

    series = KMeans(6, n_init=5, random_state=0).fit_predict(
        PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(Xu)))
    print("does adding cross-series 253 data help a HELD-OUT 253 series? (proxy for refit helping the blinded 260)\n")
    print(f"{'held-out series':16s} {'n':>4s} {'A=4139only':>11s} {'B=4139+rest253':>15s} {'delta':>8s}")
    ra, rb = [], []
    for k in range(6):
        val = series == k; other = (~val)
        if val.sum() < 5: continue
        pA = bag(Xtr, ytr, Xu[val])
        pB = bag(np.vstack([Xtr, Xu[other]]), np.concatenate([ytr, y[other]]), Xu[val])
        a, b = rae(y[val], pA), rae(y[val], pB)
        ra.append(a); rb.append(b)
        print(f"  series {k:<9d} {val.sum():>4d} {a:>11.4f} {b:>15.4f} {b-a:>+8.4f}")
    print(f"\n  MEAN: A(4139-only) {np.mean(ra):.4f} | B(4139+other-253-series) {np.mean(rb):.4f} | "
          f"delta {np.mean(rb)-np.mean(ra):+.4f}")
    verdict = ("REFIT HELPS blinded -> submit nb1118 refit" if np.mean(rb) < np.mean(ra) - 0.002
               else "REFIT neutral/hurts on a DIFFERENT series -> plain nb3200 is the safer final submission")
    print(f"  VERDICT: {verdict}")
    json.dump({"A_4139only": float(np.mean(ra)), "B_4139plus253": float(np.mean(rb)),
               "delta": float(np.mean(rb) - np.mean(ra)), "verdict": verdict},
              open(f"{P}/nb1122_refit_blinded.json", "w"), indent=2)


if __name__ == "__main__":
    main()
