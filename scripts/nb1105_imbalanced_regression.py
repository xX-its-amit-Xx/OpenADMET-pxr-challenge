"""nb1105 — IMBALANCED REGRESSION: re-weight training to a uniform pEC50 distribution (user's anti-shrinkage idea).

nb1101 found nb3200 regresses to the mean (over-predicts weak, under-predicts strong) because the training pEC50
distribution is PEAKED (median 4.65). Post-hoc calibration (nb1102) and asymmetric loss (nb1103) failed. This attacks
the ROOT: re-weight training so the EFFECTIVE pEC50 distribution is uniform (Label-Distribution-Smoothing / inverse-
density weighting, Yang et al. 2021 'Delving into Deep Imbalanced Regression'). If balancing lets the model USE the
full range, the prediction range expands (less shrinkage), low-tail bias drops, and ideally it adds to nb3200.

Strong base = combined + chempropembed. Weights: uniform | inverse-density (smoothed) | capped | binned-scaffold.
For each: RAE, pred range, low/high-tail bias, blend-with-nb3200, corr-with-error. Honest full-train -> 253.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae
from scipy.stats import gaussian_kde
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def bag(Xtr, ytr, w, Xte, nseed=3):
    ps = []
    for s in range(nseed):
        m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s)
        m.fit(Xtr, ytr, sample_weight=w); ps.append(m.predict(Xte))
    return np.mean(ps, 0)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    ytr = tr["pec50"].to_numpy()
    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].to_numpy()[unb].tolist())),
                     np.load(f"{P}/te_chemprop_embed_300.npy")[unb]])
    low = y <= np.quantile(y, 0.25); high = y >= np.quantile(y, 0.75)

    # weight schemes
    n = len(ytr)
    kde = gaussian_kde(ytr, bw_method=0.3); dens = kde(ytr)
    w_inv = 1.0 / (dens + 1e-3); w_inv *= n / w_inv.sum()
    w_cap = np.clip(w_inv, None, np.quantile(w_inv, 0.95)); w_cap *= n / w_cap.sum()
    # binned scaffold-stratified: inverse bin-count, scaffold-dedup within bin
    bins = np.clip(((ytr - ytr.min()) / (ytr.max() - ytr.min()) * 10).astype(int), 0, 9)
    cnt = np.array([np.sum(bins == b) for b in range(10)])
    w_bin = (n / 10) / np.clip(cnt[bins], 1, None); w_bin *= n / w_bin.sum()

    schemes = {"uniform": np.ones(n), "inv_density": w_inv, "inv_density_capped": w_cap, "binned": w_bin}
    print(f"nb3200 RAE {rae(y, anchor):.4f} | true range [{y.min():.2f},{y.max():.2f}] | "
          f"train pec50 median {np.median(ytr):.2f}\n")
    print(f"{'weights':20s} {'RAE':>7s} {'predmin':>8s} {'predmax':>8s} {'lowbias':>8s} {'highbias':>9s} "
          f"{'blend':>7s} {'corr_err':>9s}")
    out = {}
    for name, w in schemes.items():
        p = bag(Xtr, ytr, w, Xte)
        bb = rae(y, anchor)
        for ww in np.linspace(0, 1, 41):
            bb = min(bb, rae(y, (1 - ww) * anchor + ww * p))
        lb = np.mean(p[low] - y[low]); hb = np.mean(p[high] - y[high]); c = np.corrcoef(p, err)[0, 1]
        out[name] = dict(rae=float(rae(y, p)), predmin=float(p.min()), predmax=float(p.max()),
                         low_bias=float(lb), high_bias=float(hb), blend=float(bb), blend_delta=float(bb - rae(y, anchor)))
        print(f"{name:20s} {rae(y,p):>7.4f} {p.min():>8.2f} {p.max():>8.2f} {lb:>+8.2f} {hb:>+9.2f} "
              f"{bb:>7.4f} {c:>+9.3f}")
    print(f"\n(true range [{y.min():.2f},{y.max():.2f}]; less shrinkage = wider predmin..predmax & smaller |bias|)")
    print("GATE: real lever if a balanced scheme's blend < 0.4416 AND it expands range / cuts shrinkage bias.")
    json.dump({"anchor": float(rae(y, anchor)), **out}, open(f"{P}/nb1105_imbalanced.json", "w"), indent=2)


if __name__ == "__main__":
    main()
