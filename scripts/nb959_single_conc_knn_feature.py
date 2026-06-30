"""nb959 — AXIS-A probe 1: does a single-conc-neighbor activation feature flatten the
novel-scaffold degradation curve? (nb952 deep-extrap MAE ref = 0.5924)

The single-conc screen (10,870 cpds) is chemically LOCAL to the novel test (nb958: 83%
have a >=0.4 neighbor). Each SC compound has a single-point log2_fc activation + fdr —
a SEPARATE assay from the CRC pEC50 label, so using it as a feature is legitimate (no
pEC50 leakage). KEY check: are the 513 TEST compounds in the SC screen? If so their OWN
single-point activation is an available-at-test-time feature (potentially the strongest
signal we have for novel scaffolds).

Builds SC features (neighbors-only AND self+neighbors) for the 4139 train, runs scaffold-CV
LGBM combined vs combined+SC, reports the degradation curve. CPU.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import inchi
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
K = 5


def murcko(s):
    try:
        m = Chem.MolFromSmiles(s); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception: return None


def ikey14(s):
    try:
        m = Chem.MolFromSmiles(s); return inchi.MolToInchiKey(m)[:14] if m else None
    except Exception: return None


def load_sc():
    try:
        from src.pxr.data import load_single_conc; return load_single_conc()
    except Exception:
        import pandas as pd
        return pd.read_csv("data/raw/pxr-challenge_single_concentration_TRAIN.csv")


def sc_knn_features(fp_q, fp_sc, sc_act, sc_fdr, exclude_self=True, bs=150):
    """For each query, top-K SC neighbor activation features. Returns (n, 6)."""
    B = fp_sc.astype(np.float32); bsum = B.sum(1)[None, :]
    feats = np.zeros((len(fp_q), 6), np.float32)
    for i in range(0, len(fp_q), bs):
        Q = fp_q[i:i+bs].astype(np.float32)
        inter = Q @ B.T
        u = Q.sum(1)[:, None] + bsum - inter; u[u == 0] = 1.0
        sim = inter / u                                  # (chunk, n_sc)
        if exclude_self:
            sim = np.where(sim > 0.999, -1.0, sim)       # drop exact dupes
        for j in range(sim.shape[0]):
            idx = np.argpartition(sim[j], -K)[-K:]
            s = sim[j, idx]; ok = s > 0
            if not ok.any():
                continue
            s_ok = s[ok]; a_ok = sc_act[idx][ok]; f_ok = sc_fdr[idx][ok]
            w = s_ok / s_ok.sum()
            feats[i+j] = [np.sum(w * a_ok), a_ok.max(), np.sum(w * f_ok),
                          s_ok.max(), float((s_ok >= 0.4).sum()), np.mean(a_ok)]
    return feats   # [wmean_act, max_act, wmean_fdr, top_sim, n_nbr>=0.4, mean_act]


def curve(y, p, sv):
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sv >= lo) & (sv < hi); n = int(m.sum())
        if n == 0: rows.append((f"[{lo:.1f},{hi:.1f})", 0, None)); continue
        rows.append((f"[{lo:.1f},{hi:.1f})", n, round(float(np.mean(np.abs(y[m]-p[m]))), 4)))
    return rows


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test()
    sc = load_sc()
    y = tr["pec50"].to_numpy(float)
    smiles = tr["smiles"].tolist()
    scaf = [murcko(s) for s in smiles]
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)
    max_sim = np.load(f"{D}/nb952_max_sim_4139.npy")

    # dedup SC by inchikey14 -> activation
    sc_col = "log2_fc_estimate"; fdr_col = "neg_log10_fdr"
    sc = sc.dropna(subset=["smiles", sc_col]).copy()
    sc["ik"] = [ikey14(s) for s in sc["smiles"]]
    sc = sc.dropna(subset=["ik"]).groupby("ik").agg(
        smiles=("smiles", "first"), act=(sc_col, "median"), fdr=(fdr_col, "median")).reset_index()
    print(f"single-conc unique-by-inchikey: {len(sc)}")

    # KEY: test-compound coverage in SC
    te_ik = set(ikey14(s) for s in te["smiles"]) - {None}
    tr_ik = [ikey14(s) for s in smiles]
    sc_ik = set(sc["ik"])
    print(f"=== TEST coverage in single-conc: {len(te_ik & sc_ik)}/{len(te_ik)} test compounds "
          f"have their OWN single-point activation ===")
    print(f"    train coverage in single-conc: {sum(1 for k in tr_ik if k in sc_ik)}/{len(tr_ik)}")

    fp_sc = morgan_fp_batch(sc["smiles"].tolist())
    fp_tr = morgan_fp_batch(smiles)
    sc_act = np.nan_to_num(sc["act"].to_numpy(float), posinf=10.0, neginf=-10.0)
    sc_fdr = np.clip(np.nan_to_num(sc["fdr"].to_numpy(float), posinf=20.0, neginf=0.0), 0, 20)

    print("building SC-kNN features (neighbors-only) ...", flush=True)
    F_nbr = sc_knn_features(fp_tr, fp_sc, sc_act, sc_fdr, exclude_self=True)
    # self activation feature (NaN if train compound not in SC) -> realistic only if test covered
    ik_to_act = dict(zip(sc["ik"], sc_act))
    self_act = np.array([ik_to_act.get(k, np.nan) for k in tr_ik], np.float32)
    F_self = np.column_stack([F_nbr, self_act])

    Xc = impute(combined(smiles)).astype(np.float32)
    from sklearn.impute import SimpleImputer
    def imp(M):
        M = np.nan_to_num(np.asarray(M, np.float32), posinf=20.0, neginf=-20.0)
        M = np.clip(M, -1e6, 1e6)
        return SimpleImputer(strategy="median").fit_transform(M).astype(np.float32)

    # test cov=0 -> SC_self is a TRAIN-ONLY feature (collapses on test); SC_nbr is the honest one
    variants = {"combined (ref)": Xc,
                "combined+SC_nbr (HONEST)": np.hstack([Xc, imp(F_nbr)]),
                "combined+SC_self+nbr (TRAIN-ONLY-INVALID)": np.hstack([Xc, imp(F_self)])}
    ref_deep = json.load(open(f"{D}/nb952_stress_curve.json"))["lgbm_curve"]
    ref_deep = next(r for r in ref_deep if r["bin"] == "[0.0,0.3)")["mae"]

    res = {}
    for tag, X in variants.items():
        oof = np.full(len(y), np.nan)
        for tri, vai in folds:
            m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                                  n_jobs=4, verbose=-1).fit(X[tri], y[tri])
            oof[vai] = m.predict(X[vai])
        rows = curve(y, oof, max_sim); deep = next(mae for b, n, mae in rows if b == "[0.0,0.3)")
        res[tag] = {"overall_rae": round(float(rae(y, oof)), 4), "deep_extrap_mae": deep, "curve": rows}
        print(f"\n{tag}: overall RAE={res[tag]['overall_rae']}  deep@sim<0.3={deep}")
        for b, n, mae in rows: print(f"   {b:14s} n={n:5d} MAE={mae}")

    print("\n" + "=" * 60)
    print(f"deep-extrap MAE @ sim<0.3 (ref LGBM 0.5924):")
    for tag in variants: print(f"  {tag:24s} {res[tag]['deep_extrap_mae']}")
    print("=" * 60)
    json.dump({"test_cov": len(te_ik & sc_ik), "test_n": len(te_ik),
               "ref_deep": ref_deep, "results": res},
              open(f"{D}/nb959_single_conc_knn.json", "w"), indent=2)
    print(f"saved -> {D}/nb959_single_conc_knn.json")


if __name__ == "__main__":
    main()
