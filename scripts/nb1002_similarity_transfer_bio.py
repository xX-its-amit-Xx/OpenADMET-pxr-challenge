"""nb1002 — similarity-transfer biological fingerprint (user idea): for each compound, kNN into the
external multi-target bioactivity matrix (Papyrus) and borrow the neighbors' activity profile across
orthogonal targets -> a 'predicted biology' feature vector derived from REAL measured neighbor data
(NOT contaminating the primary PXR task). Test combined vs combined+bioFP on the degradation curve,
with STRICT (min-sim 0.5) vs LOOSE (min-sim 0.0) wrangling to see the impact. Multi-seed verify.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; OUT = "C:/pxr_struct/dash"
SEEDS = [42, 101, 202, 303, 404]


def build_bio_fp(fp_q, fp_ref, ref_targets, min_sim, k=10, bs=200):
    """kNN-transfer: for each query, sim-weighted mean of k neighbors' target-activity rows."""
    B = fp_ref.astype(np.float32); bsum = B.sum(1)[None, :]
    nt = ref_targets.shape[1]
    out = np.full((len(fp_q), nt + 1), np.nan)   # +1 = top-neighbor similarity
    for i in range(0, len(fp_q), bs):
        Q = fp_q[i:i+bs].astype(np.float32)
        inter = Q @ B.T; u = Q.sum(1)[:, None] + bsum - inter; u[u == 0] = 1.0
        sim = inter / u
        for j in range(sim.shape[0]):
            s = sim[j]; idx = np.argpartition(s, -k)[-k:]; sv = s[idx]
            keep = sv >= min_sim
            out[i+j, -1] = sv.max()
            if keep.sum() >= 1:
                w = sv[keep] / sv[keep].sum()
                rows = ref_targets[idx][keep]                # (kept, nt), may have NaN
                m = np.nansum(w[:, None] * np.nan_to_num(rows), 0)
                cnt = np.nansum(w[:, None] * (~np.isnan(rows)), 0)
                out[i+j, :nt] = np.where(cnt > 0, m / np.maximum(cnt, 1e-9), np.nan)
    return out


def curve(y, p, sv):
    BINS = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]; out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sv >= lo) & (sv < hi); n = int(m.sum())
        out.append(round(float(np.mean(np.abs(y[m]-p[m]))), 4) if n else None)
    return out


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    y = tr["pec50"].to_numpy(float); smiles = tr["smiles"].tolist()
    # external multi-target matrix (Papyrus wide): pick well-populated numeric target columns
    pap = pd.read_parquet("data/external/papyrus_wide_compound_x_target.parquet")
    smic = [c for c in pap.columns if "smile" in c.lower()][0]
    tcols = [c for c in pap.columns if c != smic and pd.api.types.is_numeric_dtype(pap[c])]
    pop = [c for c in tcols if pap[c].notna().sum() >= 200]      # >=200 measured
    pop = pop[:40]                                                # cap features
    print(f"Papyrus: {len(pap)} compounds, {len(pop)} populated target columns used")
    ref_t = pap[pop].to_numpy(float)
    fp_ref = morgan_fp_batch(pap[smic].astype(str).tolist())
    fp_tr = morgan_fp_batch(smiles)

    Xc = impute(combined(smiles)).astype(np.float32)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in smiles]
    max_sim = np.load(f"{D}/nb952_max_sim_4139.npy")
    ref_deep = next(r for r in json.load(open(f"{D}/nb952_stress_curve.json"))["lgbm_curve"] if r["bin"] == "[0.0,0.3)")["mae"]

    from sklearn.impute import SimpleImputer
    results = {}
    for tag, msim in [("loose(min0.0)", 0.0), ("strict(min0.5)", 0.5)]:
        bio = build_bio_fp(fp_tr, fp_ref, ref_t, min_sim=msim)
        cov = np.mean(np.isfinite(bio[:, 0]))
        Xb = SimpleImputer(strategy="median").fit_transform(np.nan_to_num(bio, nan=np.nan))
        Xb = np.clip(np.nan_to_num(Xb), -1e6, 1e6).astype(np.float32)
        Xcb = np.hstack([Xc, Xb])
        deltas = []
        for seed in SEEDS:
            folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
            oc = np.full(len(y), np.nan); ob = np.full(len(y), np.nan)
            for tri, vai in folds:
                oc[vai] = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xc[tri], y[tri]).predict(Xc[vai])
                ob[vai] = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xcb[tri], y[tri]).predict(Xcb[vai])
            deltas.append(rae(y, ob) - rae(y, oc))
        d = np.array(deltas); stable = d.mean() < 0 and abs(d.mean()) > d.std()
        results[tag] = {"coverage": round(float(cov), 3), "delta_mean": round(float(d.mean()), 5),
                        "delta_std": round(float(d.std()), 5), "wins": f"{int((d<0).sum())}/{len(SEEDS)}", "stable": bool(stable)}
        print(f"  {tag:16s} bioFP coverage={cov:.2f}  RAE delta={d.mean():+.5f} +/- {d.std():.5f}  wins={int((d<0).sum())}/{len(SEEDS)} stable={stable}")

    print("\n" + "=" * 60)
    any_help = any(v["stable"] for v in results.values())
    print(">>> similarity-transfer biology HELPS" if any_help
          else ">>> similarity-transfer biology absorbed/no help (external off-manifold to novel test + structure-derived)")
    print("=" * 60)
    json.dump({"ref_deep": ref_deep, "results": results}, open(f"{OUT}/nb1002_simtransfer.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb1002_simtransfer.json")


if __name__ == "__main__":
    main()
