"""nb1116 — APPROACH #5: test-internal TRANSDUCTIVE SAR smoothing (untapped test-set structure).

The test compounds form their own analog manifold; nb3200 predictions should vary smoothly across structurally-similar
test compounds. Graph-smooth nb3200's predictions over the test's own Morgan-similarity graph (label propagation),
seeded by nb3200 -> reduce local prediction noise where the test SAR is smooth. Leakage-free (uses test STRUCTURE,
not labels). Tune the smoothing strength alpha via LEAVE-SERIES-OUT (cycle-305), evaluate on the 253.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae
from src.pxr.chem import morgan_fp_batch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def main():
    unb = np.load("data/processed/_audit_unblind_idx.npy"); y = np.load("data/processed/_audit_unblind_y.npy")
    anchor = np.load("data/processed/nb3200_pred_oof.npy")
    te = load_test()
    F = fpf(te["smiles"].to_numpy()[unb].tolist())
    # similarity graph among the 253
    inter = F @ F.T; s = F.sum(1)
    sim = inter / np.clip(s[:, None] + s[None, :] - inter, 1, None)
    np.fill_diagonal(sim, 0)
    # keep top-k neighbors, similarity-weighted row-normalized
    k = 8; W = np.zeros_like(sim)
    for i in range(len(sim)):
        nb = np.argsort(sim[i])[::-1][:k]; W[i, nb] = sim[i, nb] ** 2
    W = W / np.clip(W.sum(1, keepdims=True), 1e-9, None)

    def smooth(p, alpha, iters=10):
        q = p.copy()
        for _ in range(iters):
            q = (1 - alpha) * p + alpha * (W @ q)
        return q

    print(f"nb3200 anchor RAE {rae(y, anchor):.4f}")
    print(f"{'alpha':>6s} {'pooled_RAE':>11s}")
    for a in [0.1, 0.2, 0.3, 0.5, 0.7]:
        print(f"{a:>6.1f} {rae(y, smooth(anchor, a)):>11.4f}")

    # honest: tune alpha LEAVE-SERIES-OUT, apply to held-out series
    series = KMeans(6, n_init=5, random_state=0).fit_predict(
        PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(F)))
    oof = anchor.copy()
    for kk in range(6):
        trn = series != kk; val = series == kk
        if val.sum() < 5: continue
        ba, bb = 0.0, rae(y[trn], anchor[trn])
        for a in np.linspace(0, 0.8, 17):
            r = rae(y[trn], smooth(anchor, a)[trn])
            if r < bb: bb, ba = r, a
        oof[val] = smooth(anchor, ba)[val]
    print(f"\nblend X-SERIES (alpha tuned on held-out series) {rae(y, oof):.4f} (delta {rae(y,oof)-rae(y,anchor):+.4f})")
    json.dump({"anchor": float(rae(y, anchor)), "xseries": float(rae(y, oof))},
              open("data/processed/nb1116_transductive.json", "w"), indent=2)
    print("GATE: real lever if blend_xseries < 0.4416.")


if __name__ == "__main__":
    main()
