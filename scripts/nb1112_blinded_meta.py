"""nb1112 — BLINDED vs UNBLINDED meta-analysis (are we overfitting to the 253? does it transfer to the blinded 260?).

User concern: all our honest validation is on the 253 UNBLINDED; the LB scores the full 513 incl ~260 still-blinded.
Recon: blinded & unblinded are equally close to train (sim 0.53 vs 0.515) BUT cross-sim only 0.26 = DIFFERENT series.
This quantifies the transfer risk and builds a permanent check:
  1. ADVERSARIAL VALIDATION: classify blinded-vs-unblinded in nb3200 feature space. AUC~0.5 => 253 representative;
     high AUC => distinct distributions => 253-tuned gates are biased.
  2. HARDEST BLINDED: rank the 260 by hardness (low train sim + neighbor-disagreement cliff proxy + doubly-novel
     vs train AND unblinded). Flag a watchlist + figure.
  3. CROSS-SERIES TRANSFER PROTOCOL: cluster the 253 into chemical series; leave-one-series-out, check whether a
     candidate residual signal tuned on K-1 series transfers to the held-out series. This is the check to ADOPT
     going forward (mimics unblind->blind transfer), replacing all-253 in-sample tuning.
Figures -> C:/pxr_work/figures.
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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = "data/processed"; FIG = "C:/pxr_work/figures"; os.makedirs(FIG, exist_ok=True)


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def main():
    te = load_test().reset_index(drop=True); tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    n = len(te); unb = np.load(f"{P}/_audit_unblind_idx.npy")
    blind = np.array([i for i in range(n) if i not in set(unb.tolist())])
    ytr = tr["pec50"].to_numpy()

    Xte = np.hstack([impute(combined(te["smiles"].tolist())), np.load(f"{P}/te_chemprop_embed_300.npy")])
    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Z = PCA(50, random_state=0).fit_transform(StandardScaler().fit(np.vstack([Xtr, Xte])).transform(Xte))

    # 1. adversarial validation: blinded(1) vs unblinded(0)
    lab = np.zeros(n); lab[blind] = 1
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=4, verbose=-1)
    oof = cross_val_predict(clf, Z, lab, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(lab, oof)
    print(f"=== ADVERSARIAL VALIDATION (blinded vs unblinded) ===")
    print(f"  AUC = {auc:.3f}  ({'~0.5 = indistinguishable (253 representative)' if auc<0.6 else 'DISTINGUISHABLE -> 253 is a biased validation sample'})")

    # 2. hardest blinded
    Fte = fpf(te["smiles"].to_numpy().tolist()); Ftr = fpf(tr["smiles"].tolist())
    inter = Fte @ Ftr.T; sim_tr = inter / np.clip(Fte.sum(1)[:, None] + Ftr.sum(1)[None, :] - inter, 1, None)
    top1 = sim_tr.max(1)
    # neighbor disagreement = std of train pEC50 among top-8 train neighbors (cliff proxy)
    nbr = np.argsort(sim_tr, 1)[:, ::-1][:, :8]
    disagree = np.array([np.std(ytr[nbr[i]]) for i in range(n)])
    # doubly-novel: low sim to BOTH train and unblinded
    ub = Fte[unb]; i2 = Fte @ ub.T; s2 = i2 / np.clip(Fte.sum(1)[:, None] + ub.sum(1)[None, :] - i2, 1, None)
    np.fill_diagonal(s2[:0], 0)
    sim_unb = s2.max(1)
    hard = (1 - top1) + 0.5 * disagree + 0.5 * (1 - sim_unb)        # hardness score
    bl_order = blind[np.argsort(hard[blind])[::-1]]
    print(f"\n=== HARDEST BLINDED (top 15 of {len(blind)}) ===")
    print(f"  {'name':18s} {'train_sim':>9s} {'nbr_disagree':>12s} {'unblind_sim':>11s}")
    names = te["name"].to_numpy() if "name" in te.columns else np.array([f"idx{i}" for i in range(n)])
    watch = []
    for i in bl_order[:15]:
        print(f"  {str(names[i])[:18]:18s} {top1[i]:>9.2f} {disagree[i]:>12.2f} {sim_unb[i]:>11.2f}")
        watch.append({"name": str(names[i]), "train_sim": float(top1[i]), "nbr_disagree": float(disagree[i])})
    print(f"  blinded hardness median {np.median(hard[blind]):.2f} vs unblinded {np.median(hard[unb]):.2f}")

    # 3. cross-series transfer protocol: cluster 253 into series, leave-one-series-out residual-transfer
    y = np.load(f"{P}/_audit_unblind_y.npy"); anchor = np.load(f"{P}/nb3200_pred_oof.npy"); resid = y - anchor
    Zub = Z[unb]; K = 6
    series = KMeans(K, n_init=5, random_state=0).fit_predict(StandardScaler().fit_transform(Zub))
    # demo candidate signal: physchem-residual model. Does its gain transfer across series?
    feat = impute(combined(te["smiles"].to_numpy()[unb].tolist()))[:, -217:]
    feat = StandardScaler().fit_transform(feat)
    in_series, cross_series = [], []
    for k in range(K):
        tr_m = series != k; te_m = series == k
        if te_m.sum() < 8: continue
        m = lgb.LGBMRegressor(n_estimators=150, num_leaves=15, learning_rate=0.05, n_jobs=4, verbose=-1)
        m.fit(feat[tr_m], resid[tr_m])
        pr = m.predict(feat[te_m])
        # in-series (fit+eval same series, optimistic) vs cross-series (held-out)
        cross_series.append(np.corrcoef(pr, resid[te_m])[0, 1] if pr.std() > 1e-9 else 0)
    print(f"\n=== CROSS-SERIES TRANSFER (leave-one-series-out, K={K}) ===")
    print(f"  mean cross-series corr(resid_pred, resid) = {np.nanmean(cross_series):+.3f}  "
          f"(near 0 => 253-tuned residual signals do NOT transfer across series = overfitting risk REAL)")

    # figure: UMAP-free 2D PCA scatter, train/unblind/blind + hardest flagged
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    Ztr2 = PCA(2, random_state=0).fit(StandardScaler().fit_transform(np.vstack([Xtr, Xte]))).transform(
        StandardScaler().fit(np.vstack([Xtr, Xte])).transform(np.vstack([Xtr, Xte])))
    a = len(Xtr)
    ax[0].scatter(Ztr2[:a, 0], Ztr2[:a, 1], s=4, c="lightgray", label="train")
    ax[0].scatter(Ztr2[a + unb, 0], Ztr2[a + unb, 1], s=16, c="#2471a3", label="unblinded 253")
    ax[0].scatter(Ztr2[a + blind, 0], Ztr2[a + blind, 1], s=16, c="#c0392b", label="blinded 260", alpha=0.7)
    ax[0].legend(); ax[0].set_title(f"Train / Unblinded / Blinded (adv-AUC {auc:.2f})"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].scatter(top1[unb], disagree[unb], s=14, c="#2471a3", alpha=0.6, label="unblinded")
    ax[1].scatter(top1[blind], disagree[blind], s=14, c="#c0392b", alpha=0.6, label="blinded")
    ax[1].set_xlabel("top-1 train similarity"); ax[1].set_ylabel("neighbor disagreement (cliff proxy)")
    ax[1].set_title("Hardness map (low sim + high disagreement = hard)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/nb1112_blinded_meta.png", dpi=115); plt.close()
    json.dump({"adv_auc": float(auc), "cross_series_corr": float(np.nanmean(cross_series)),
               "blind_hardness_median": float(np.median(hard[blind])), "unb_hardness_median": float(np.median(hard[unb])),
               "hardest_blinded": watch}, open(f"{P}/nb1112_blinded_meta.json", "w"), indent=2)
    print(f"\nwrote {FIG}/nb1112_blinded_meta.png + watchlist json")


if __name__ == "__main__":
    main()
