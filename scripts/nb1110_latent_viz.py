"""nb1110 — LATENT-SPACE visualization: train vs test distribution shift (+ where nb3200 fails).

User wants to SEE the train/test latent gap and attack it. UMAP of train(4139)+test(513) on the nb3200 feature space
(combined + chempropembed), fit jointly. Panels:
  1. train (gray) vs test (red)        -> where does test diverge from train?
  2. colored by pEC50 (train activity) -> the activity landscape
  3. the 253 unblind colored by |nb3200 error| -> are high-error test cpds in train-sparse regions?
Quantify the shift: test->train nearest-neighbor distance vs train->train; fraction of test in low-train-density.
Figures -> C:/pxr_work/figures (off the full D: drive).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import umap
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = "data/processed"; FIG = "C:/pxr_work/figures"; os.makedirs(FIG, exist_ok=True)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = np.abs(y - anchor)
    ytr = tr["pec50"].to_numpy()

    Xtr = np.hstack([impute(combined(tr["smiles"].tolist())), np.load(f"{P}/tr_chemprop_embed_300.npy")])
    Xte = np.hstack([impute(combined(te["smiles"].tolist())), np.load(f"{P}/te_chemprop_embed_300.npy")])
    X = np.vstack([Xtr, Xte]); sc = StandardScaler().fit(X)
    Z = PCA(50, random_state=0).fit_transform(sc.transform(X))
    print("UMAP embedding (4652 x 50 -> 2)...", flush=True)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(Z)
    e_tr, e_te = emb[:len(Xtr)], emb[len(Xtr):]
    e_te_unb = e_te[unb]

    # quantify shift in the 50-d PCA space
    Ztr, Zte = Z[:len(Xtr)], Z[len(Xtr):]
    d_te = cdist(Zte, Ztr).min(1)                         # each test -> nearest train
    d_tr = np.sort(cdist(Ztr[:500], Ztr))[:, 1]           # train -> nearest train (sample)
    frac_far = float(np.mean(d_te > np.quantile(d_tr, 0.95)))
    print(f"test->train NN dist median {np.median(d_te):.2f} vs train->train {np.median(d_tr):.2f}")
    print(f"fraction of test beyond the 95th pct of train-train NN dist: {frac_far:.1%}")
    print(f"corr(test->train NN dist, nb3200 |error|) on 253: {np.corrcoef(d_te[unb], err)[0,1]:+.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    ax[0].scatter(e_tr[:, 0], e_tr[:, 1], s=5, c="lightgray", label="train (4139)")
    ax[0].scatter(e_te[:, 0], e_te[:, 1], s=14, c="#c0392b", alpha=0.7, label="test (513)")
    ax[0].set_title("Train vs Test latent space (UMAP)"); ax[0].legend(); ax[0].set_xticks([]); ax[0].set_yticks([])
    s1 = ax[1].scatter(e_tr[:, 0], e_tr[:, 1], s=6, c=ytr, cmap="viridis")
    ax[1].set_title("Train activity landscape (pEC50)"); ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.colorbar(s1, ax=ax[1])
    s2 = ax[2].scatter(e_tr[:, 0], e_tr[:, 1], s=4, c="lightgray")
    s2 = ax[2].scatter(e_te_unb[:, 0], e_te_unb[:, 1], s=22, c=err, cmap="Reds", edgecolor="k", lw=0.3)
    ax[2].set_title("Test (253) colored by nb3200 |error|"); ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.colorbar(s2, ax=ax[2])
    plt.tight_layout(); plt.savefig(f"{FIG}/nb1110_latent_train_test.png", dpi=115); plt.close()
    np.save("C:/pxr_work/artifacts/umap_emb.npy", emb)
    json.dump({"test_nn_dist_median": float(np.median(d_te)), "train_nn_dist_median": float(np.median(d_tr)),
               "frac_test_far": frac_far, "corr_nndist_error": float(np.corrcoef(d_te[unb], err)[0, 1])},
              open(f"{P}/nb1110_latent.json", "w"), indent=2)
    print(f"wrote {FIG}/nb1110_latent_train_test.png")


if __name__ == "__main__":
    main()
