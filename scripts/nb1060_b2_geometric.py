"""nb1060 — [B2 v1] geometry-invariant CONTACT-PROFILE model from Boltz-API cofold coords.

Per compound, per conformational sample: for each PXR residue (1..293) compute its min-distance to any ligand atom
-> a 293-d contact profile. Aggregate mean + std across the 5 samples -> (293 mean, 293 std) = which residues the
ligand engages and how STABLY (the binding-mode geometry + fluctuation, residue-resolved). E(3)-invariant (distances
only). Richer than the 18-d geom. Tests: standalone SAR (scaffold CV on train) + as a FEATURE on nb3200 (253) +
marginal over rich-z+geom. If it adds -> full EGNN is justified (v2).

Run after the eval + train cofolds land (C:/pxr_struct/boltz_api/{eval,train}/feats/).
"""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; M = "C:/pxr_struct/boltz/modal"; AP = "C:/pxr_struct/boltz_api"; QL, QH = 0.05, 0.98
NRES = 293


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def profile(npz_path):
    """293-d (mean, std) per-residue min-distance-to-ligand contact profile, aligned to resid 1..293."""
    d = np.load(npz_path)
    if "ca" not in d.files:
        return None
    ca = d["ca"]; resids = d["resids"]; ligpad = d["ligpad"]; nlig = d["nlig"]   # ca (S,nres,3)
    S = ca.shape[0]
    prof = np.full((S, NRES), np.nan, np.float32)
    for s in range(S):
        lig = ligpad[s][:int(nlig[s])]
        if len(lig) == 0:
            continue
        dmat = np.sqrt(((ca[s][:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(1)  # (nres,) min dist to ligand
        for k, r in enumerate(resids):
            if 1 <= int(r) <= NRES:
                prof[s, int(r) - 1] = dmat[k]
    mean = np.nanmean(prof, 0); std = np.nanstd(prof, 0)
    return np.concatenate([np.nan_to_num(mean, nan=30.0), np.nan_to_num(std, nan=0.0)]).astype(np.float32)


def load_set(tag, id_list):
    out = {}
    for i in id_list:
        p = f"{AP}/{tag}/feats/{i}.npz"
        if os.path.exists(p):
            pr = profile(p)
            if pr is not None:
                out[i] = pr
    return out


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def main():
    # ---- eval 253 profiles ----
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    evp = load_set("eval", [int(i) for i in unb])
    cov = np.array([int(i) in evp for i in unb])
    print(f"eval contact profiles: {cov.sum()}/253")
    if cov.sum() < 200:
        print("  (waiting for more eval cofolds; run again when complete)"); return
    P = np.full((len(unb), 2 * NRES), np.nan, np.float32)
    for k, i in enumerate(unb):
        if int(i) in evp:
            P[k] = evp[int(i)]
    colmed = np.nanmedian(P, 0); idx = np.where(np.isnan(P)); P[idx] = np.take(colmed, idx[1])

    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    # structural block (rich-z + geom) for the marginal test
    geom = np.load(f"{M}/test_geom.npy"); richz = np.load(f"{M}/test_richz.npy")
    def proc(a, k=None):
        a = a[unb].copy(); c = np.nanmedian(a, 0); ix = np.where(np.isnan(a)); a[ix] = np.take(c, ix[1]); a = StandardScaler().fit_transform(a)
        return PCA(k, random_state=0).fit_transform(a).astype(np.float32) if k else a.astype(np.float32)
    struct = np.hstack([proc(richz, 15), proc(geom)])
    prof_pca = PCA(n_components=20, random_state=0).fit_transform(StandardScaler().fit_transform(P)).astype(np.float32)

    # standalone SAR: corr of profile-PCA with truth
    print(f"corr(profile PC1, true pEC50) = {np.corrcoef(prof_pca[:,0], y)[0,1]:+.3f}")

    SEEDS = list(range(1400, 1430))
    def test(extra, baseextra, label):
        ds = []
        for s in SEEDS:
            f = scaffold_kfold_indices(scaf, 5, seed=s)
            X1 = np.hstack([base, baseextra, extra]) if baseextra is not None else np.hstack([base, extra])
            X0 = np.hstack([base, baseextra]) if baseextra is not None else base
            ds.append(clipped(X1, resid, anchor, y, f) - clipped(X0, resid, anchor, y, f))
        ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
        print(f"  {label:38s}: {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}")
    print("contact-profile on nb3200 (30 seeds):")
    test(prof_pca, None, "contact-profile(PCA20)")
    test(prof_pca, struct, "contact-profile MARGINAL over rich-z+geom")
    json.dump({"eval_cov": int(cov.sum())}, open(f"{D}/nb1060_b2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
